"""
Hybrid JAX vs normix_numpy EM comparison.

The hybrid approach:
  - E-step: pure-JAX (vmap over pure-JAX log_kv)
  - M-step GIG:  'cpu_legacy' solver (normix_numpy scipy.kve + L-BFGS-B on CPU)
  - M-step other: unchanged (jaxopt LBFGS / closed-form)

Questions:
  1. E-step: does JAX vmap match or beat numpy at high dimension (468 stocks)?
  2. GIG M-step: does cpu_legacy close the gap to numpy?
  3. Other M-step: how does jaxopt LBFGS compare to numpy?

Data: S&P 500, 2552 obs x 468 stocks.
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

# ── Load data ──────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
X_full = df.values.astype(np.float64)
n_obs, n_stocks = X_full.shape
X_full_10 = X_full[:, :10]
print(f"Data: {n_obs} obs × {n_stocks} stocks", flush=True)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

gpu = [d for d in jax.devices() if d.platform == "gpu"]
gpu = gpu[0] if gpu else None
cpu = jax.devices("cpu")[0]

# ─── JAX model helpers ─────────────────────────────────────────────────
from normix import GeneralizedHyperbolic, VarianceGamma, NormalInverseGamma, NormalInverseGaussian

def init_gh(X, key):   return GeneralizedHyperbolic._initialize(X, key)
def init_vg(X, key):
    n, d = X.shape; mu = jnp.mean(X,axis=0); X_c = X-mu
    sigma=(X_c.T@X_c)/n+1e-4*jnp.eye(d); gamma=0.01*jax.random.normal(key,(d,),dtype=jnp.float64)
    return VarianceGamma.from_classical(mu=mu,gamma=gamma,sigma=sigma,alpha=1.,beta=1.)
def init_nig(X, key):
    n, d = X.shape; mu = jnp.mean(X,axis=0); X_c = X-mu
    sigma=(X_c.T@X_c)/n+1e-4*jnp.eye(d); gamma=0.01*jax.random.normal(key,(d,),dtype=jnp.float64)
    return NormalInverseGaussian.from_classical(mu=mu,gamma=gamma,sigma=sigma,mu_ig=1.,lam=1.)
def init_niga(X, key):
    n, d = X.shape; mu = jnp.mean(X,axis=0); X_c = X-mu
    sigma=(X_c.T@X_c)/n+1e-4*jnp.eye(d); gamma=0.01*jax.random.normal(key,(d,),dtype=jnp.float64)
    return NormalInverseGamma.from_classical(mu=mu,gamma=gamma,sigma=sigma,alpha=2.,beta=1.)

DISTS_JAX = [
    ("NIG",   init_nig,  "InvGaussian"),
    ("VG",    init_vg,   "Gamma"),
    ("NIG-α", init_niga, "InvGamma"),
    ("GH",    init_gh,   "GIG"),
]

# ─── normix_numpy helpers ──────────────────────────────────────────────
from normix_numpy.distributions.mixtures.generalized_hyperbolic import (
    GeneralizedHyperbolic as GH_np, REGULARIZATION_METHODS,
)
from normix_numpy.distributions.mixtures.variance_gamma import VarianceGamma as VG_np
from normix_numpy.distributions.mixtures.normal_inverse_gaussian import NormalInverseGaussian as NIG_np
from normix_numpy.distributions.mixtures.normal_inverse_gamma import NormalInverseGamma as NIGa_np

regularize_fn = REGULARIZATION_METHODS['det_sigma_one']

def profile_numpy_em(dist_cls, X, n_iter, label="", fix_tail=False):
    model = dist_cls()
    model._joint = model._create_joint_distribution()
    d = X.shape[1]; model._joint._d = d
    X_norm, center, scale = model._normalize_data(X)
    try:
        model._initialize_params(X_norm, random_state=42)
    except TypeError:
        model._initialize_params(X_norm)
    e_times, m_times, iter_times = [], [], []
    for i in range(n_iter):
        t_start = time.perf_counter()
        # Apply GH regularization only for GH distributions
        jt = model._joint
        if hasattr(jt, '_p') and hasattr(jt, '_a'):
            mu=jt._mu; gamma=jt._gamma; sigma=jt._L_Sigma@jt._L_Sigma.T
            p_val,a_val,b_val=jt._p,jt._a,jt._b; log_det_sigma=jt.log_det_Sigma
            reg = regularize_fn(mu,gamma,sigma,p_val,a_val,b_val,d,log_det_sigma=log_det_sigma)
            reg['a']=np.clip(reg['a'],1e-6,1e6); reg['b']=np.clip(reg['b'],1e-6,1e6)
            model._joint.set_classical_params(**reg)
        t0=time.perf_counter(); cond_exp=model._conditional_expectation_y_given_x(X_norm)
        t_e=time.perf_counter()-t0; e_times.append(t_e)
        try:
            t0=time.perf_counter(); model._m_step(X_norm, cond_exp, fix_tail=fix_tail)
        except TypeError:
            t0=time.perf_counter(); model._m_step(X_norm, cond_exp)
        t_m=time.perf_counter()-t0; m_times.append(t_m)
        iter_times.append(time.perf_counter()-t_start)
        ll = np.mean(model.logpdf(X_norm))
        print(f"  [{label}] iter {i}: E={t_e:.3f}s  M={t_m:.3f}s  ll={ll:.2f}", flush=True)
    return {'e': e_times, 'm': m_times, 'total': iter_times}

def profile_jax_em(model, X, n_iter, label="", solver='newton'):
    X = jnp.asarray(X, dtype=jnp.float64)
    e_times, m_times, iter_times = [], [], []
    for i in range(n_iter):
        t_start=time.perf_counter()
        t0=time.perf_counter(); exp=model.e_step(X); _=float(exp['E_Y'][0])
        t_e=time.perf_counter()-t0; e_times.append(t_e)
        t0=time.perf_counter()
        try:
            model=model.m_step(X,exp,solver=solver)
        except TypeError:
            model=model.m_step(X,exp)
        _=float(model._joint.mu[0])
        t_m=time.perf_counter()-t0; m_times.append(t_m)
        if hasattr(model,'regularize_det_sigma_one'): model=model.regularize_det_sigma_one(); _=float(model._joint.mu[0])
        ll=model.marginal_log_likelihood(X); _=float(ll)
        iter_times.append(time.perf_counter()-t_start)
        print(f"  [{label}] iter {i}: E={t_e:.3f}s  M={t_m:.3f}s  ll={float(ll):.2f}", flush=True)
    return {'e': e_times, 'm': m_times, 'total': iter_times, 'model': model}


# ═══════════════════════════════════════════════════════════════════════
# Run experiments
# ═══════════════════════════════════════════════════════════════════════

N_ITER = 5
SKIP = 1  # skip first iter (may have residual compilation)

configs = [("10 stocks", X_full_10), (f"{n_stocks} stocks", X_full)]

results = {}  # results[config][backend][dist] = {e, m, total}

for config_label, X_data in configs:
    results[config_label] = {}
    d = X_data.shape[1]
    X_dev = jax.device_put(jnp.asarray(X_data, dtype=np.float64), gpu or cpu)
    key = jax.device_put(jax.random.PRNGKey(42), gpu or cpu)

    print(f"\n{'='*80}", flush=True)
    print(f"  {config_label}  ({n_obs} obs × {d} stocks)", flush=True)
    print(f"{'='*80}", flush=True)

    # ── normix_numpy ──────────────────────────────────────────────────
    np_res = {}
    for short, cls, fix_tail in [("NIG",NIG_np,True),("VG",VG_np,True),("NIG-α",NIGa_np,True),("GH",GH_np,False)]:
        print(f"\n  normix_numpy / {short}:", flush=True)
        r = profile_numpy_em(cls, X_data, N_ITER+1, label=f"np-{short}", fix_tail=fix_tail)
        np_res[short] = {'e': np.mean(r['e'][SKIP:]), 'm': np.mean(r['m'][SKIP:]), 'total': np.mean(r['total'][SKIP:])}
        print(f"  => E={np_res[short]['e']:.3f}s  M={np_res[short]['m']:.3f}s  iter={np_res[short]['total']:.3f}s", flush=True)
    results[config_label]['numpy'] = np_res

    # ── JAX hybrid (cpu_legacy GIG) ───────────────────────────────────
    jax_hybrid_res = {}
    for short, init_fn, _ in DISTS_JAX:
        gig_solver = 'cpu_legacy' if short == 'GH' else 'newton'
        print(f"\n  JAX-hybrid / {short} (GIG solver={gig_solver}):", flush=True)
        model = init_fn(X_dev, key)
        print("  Warmup...", flush=True)
        profile_jax_em(model, X_dev, 1, label="warm", solver=gig_solver)
        model = init_fn(X_dev, key)
        r = profile_jax_em(model, X_dev, N_ITER, label=f"jh-{short}", solver=gig_solver)
        jax_hybrid_res[short] = {'e': np.mean(r['e'][SKIP:]), 'm': np.mean(r['m'][SKIP:]), 'total': np.mean(r['total'][SKIP:])}
        print(f"  => E={jax_hybrid_res[short]['e']:.3f}s  M={jax_hybrid_res[short]['m']:.3f}s  iter={jax_hybrid_res[short]['total']:.3f}s", flush=True)
    results[config_label]['jax_hybrid'] = jax_hybrid_res

    # ── JAX default (newton solver, for comparison) ───────────────────
    jax_default_res = {}
    for short, init_fn, _ in DISTS_JAX:
        print(f"\n  JAX-default / {short} (solver=newton):", flush=True)
        model = init_fn(X_dev, key)
        print("  Warmup...", flush=True)
        profile_jax_em(model, X_dev, 1, label="warm", solver='newton')
        model = init_fn(X_dev, key)
        r = profile_jax_em(model, X_dev, N_ITER, label=f"jd-{short}", solver='newton')
        jax_default_res[short] = {'e': np.mean(r['e'][SKIP:]), 'm': np.mean(r['m'][SKIP:]), 'total': np.mean(r['total'][SKIP:])}
        print(f"  => E={jax_default_res[short]['e']:.3f}s  M={jax_default_res[short]['m']:.3f}s  iter={jax_default_res[short]['total']:.3f}s", flush=True)
    results[config_label]['jax_default'] = jax_default_res


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}", flush=True)
print("FINAL COMPARISON TABLE (seconds, avg over timed iters)", flush=True)
print(f"{'='*100}", flush=True)
print(f"Device: {gpu or cpu}", flush=True)

for config_label, _ in configs:
    print(f"\n─── {config_label} ───", flush=True)
    print(f"{'Dist':<8} {'Backend':<16} {'E-step':>8} {'M-step':>8} {'iter':>8}  Notes", flush=True)
    print(f"{'-'*70}", flush=True)
    for short in ["NIG", "VG", "NIG-α", "GH"]:
        for bk, label, note in [
            ('numpy',       'normix_numpy  ', ''),
            ('jax_hybrid',  'JAX+cpu_legacy', 'GIG→cpu' if short=='GH' else ''),
            ('jax_default', 'JAX-default   ', 'GIG→JAX Newton'),
        ]:
            r = results[config_label][bk].get(short, {})
            if not r: continue
            e, m, t = r['e'], r['m'], r['total']
            print(f"{short:<8} {label:<16} {e:>8.3f} {m:>8.3f} {t:>8.3f}  {note}", flush=True)
    print(flush=True)

print("""
Key questions answered:
  1. E-step: JAX vmap vs numpy E-step at 468 stocks
  2. GIG M-step: cpu_legacy (normix_numpy scipy.kve) vs JAX Newton vs numpy
  3. Other M-steps (VG, NIG-α): jaxopt LBFGS vs numpy
""", flush=True)
