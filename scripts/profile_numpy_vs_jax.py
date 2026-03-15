"""
Compare normix_numpy (legacy NumPy/SciPy) vs normix (JAX) EM performance.

Runs identical EM fits on S&P 500 returns data for GH, VG, NIG, NIG-alpha,
timing E-step + M-step separately for both backends.

Key difference: numpy version uses vectorized scipy.special.kve calls (one call
per observation batch) while JAX version uses jax.vmap over pure_callback
(one callback per observation).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# ── Load data ────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
X_full = df.values.astype(np.float64)
n_obs, n_stocks = X_full.shape
print(f"Data: {n_obs} obs × {n_stocks} stocks", flush=True)

# Use subsets for comparison
CONFIGS = [
    ("10 stocks", X_full[:, :10]),
    (f"{n_stocks} stocks", X_full),
]

N_ITER = 5

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: normix_numpy benchmarks
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print("PART 1: normix_numpy (NumPy/SciPy backend)", flush=True)
print("=" * 80, flush=True)

from normix_numpy.distributions.mixtures.generalized_hyperbolic import GeneralizedHyperbolic as GH_np
from normix_numpy.distributions.mixtures.variance_gamma import VarianceGamma as VG_np
from normix_numpy.distributions.mixtures.normal_inverse_gaussian import NormalInverseGaussian as NIG_np
from normix_numpy.distributions.mixtures.normal_inverse_gamma import NormalInverseGamma as NIGa_np


def profile_numpy_em(dist_cls, X, n_iter, label="", fix_tail=False):
    """
    Profile numpy EM by running fit() with verbose timing.

    Since the numpy version's fit() doesn't expose step timings directly,
    we replicate the EM loop with manual timing.
    """
    model = dist_cls()

    # Initialize
    model._joint = model._create_joint_distribution()
    d = X.shape[1]
    model._joint._d = d

    # Normalize
    X_norm, center, scale = model._normalize_data(X)

    # Initialize params
    model._initialize_params(X_norm, random_state=42)

    # Get regularization
    from normix_numpy.distributions.mixtures.generalized_hyperbolic import REGULARIZATION_METHODS
    regularize_fn = REGULARIZATION_METHODS['det_sigma_one']

    e_times = []
    m_times = []
    reg_times = []
    ll_times = []
    iter_times = []

    for i in range(n_iter):
        t_iter_start = time.perf_counter()

        # Regularize (pre)
        t0 = time.perf_counter()
        jt = model._joint
        mu = jt._mu
        gamma = jt._gamma
        sigma = jt._L_Sigma @ jt._L_Sigma.T
        p_val, a_val, b_val = jt._p, jt._a, jt._b
        log_det_sigma = jt.log_det_Sigma
        regularized = regularize_fn(mu, gamma, sigma, p_val, a_val, b_val, d,
                                    log_det_sigma=log_det_sigma)
        regularized['a'] = np.clip(regularized['a'], 1e-6, 1e6)
        regularized['b'] = np.clip(regularized['b'], 1e-6, 1e6)
        model._joint.set_classical_params(**regularized)

        # E-step
        t0 = time.perf_counter()
        cond_exp = model._conditional_expectation_y_given_x(X_norm)
        t_e = time.perf_counter() - t0
        e_times.append(t_e)

        # M-step
        t0 = time.perf_counter()
        model._m_step(X_norm, cond_exp, fix_tail=fix_tail)
        t_m = time.perf_counter() - t0
        m_times.append(t_m)

        # Regularize (post)
        t0 = time.perf_counter()
        jt = model._joint
        mu = jt._mu
        gamma = jt._gamma
        sigma = jt._L_Sigma @ jt._L_Sigma.T
        p_val, a_val, b_val = jt._p, jt._a, jt._b
        log_det_sigma = jt.log_det_Sigma
        regularized = regularize_fn(mu, gamma, sigma, p_val, a_val, b_val, d,
                                    log_det_sigma=log_det_sigma)
        regularized['a'] = np.clip(regularized['a'], 1e-6, 1e6)
        regularized['b'] = np.clip(regularized['b'], 1e-6, 1e6)
        model._joint.set_classical_params(**regularized)
        t_reg = time.perf_counter() - t0
        reg_times.append(t_reg)

        # Log-likelihood
        t0 = time.perf_counter()
        ll = np.mean(model.logpdf(X_norm))
        t_ll = time.perf_counter() - t0
        ll_times.append(t_ll)

        t_iter = time.perf_counter() - t_iter_start
        iter_times.append(t_iter)

        print(f"  [{label}] iter {i:2d}: "
              f"E={t_e:.3f}s  M={t_m:.3f}s  reg={t_reg:.3f}s  "
              f"LL={t_ll:.3f}s  total={t_iter:.3f}s  ll={ll:.4f}", flush=True)

    return {
        'e_step': e_times,
        'm_step': m_times,
        'regularize': reg_times,
        'log_likelihood': ll_times,
        'iteration': iter_times,
    }


numpy_results = {}

for config_name, X_data in CONFIGS:
    print(f"\n--- {config_name} ---", flush=True)
    numpy_results[config_name] = {}

    dists = [
        ('NIG', NIG_np, True),
        ('VG', VG_np, True),
        ('NIG-alpha', NIGa_np, True),
        ('GH', GH_np, False),
    ]

    for short_name, cls, fix_tail in dists:
        print(f"\n  {short_name} (numpy):", flush=True)
        try:
            timings = profile_numpy_em(cls, X_data, N_ITER,
                                       label=f"np-{short_name}",
                                       fix_tail=fix_tail)
            skip = min(1, N_ITER - 1)
            e_avg = np.mean(timings['e_step'][skip:])
            m_avg = np.mean(timings['m_step'][skip:])
            iter_avg = np.mean(timings['iteration'][skip:])
            print(f"  SUMMARY: E={e_avg:.4f}s  M={m_avg:.4f}s  iter={iter_avg:.4f}s", flush=True)
            numpy_results[config_name][short_name] = {
                'avg_e_step': e_avg,
                'avg_m_step': m_avg,
                'avg_iteration': iter_avg,
            }
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            numpy_results[config_name][short_name] = {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: normix (JAX) benchmarks
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print("PART 2: normix (JAX backend)", flush=True)
print("=" * 80, flush=True)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}", flush=True)
print(f"JAX default device: {jax.devices()[0]}", flush=True)

cpu = jax.devices("cpu")[0]
gpu_devices = jax.devices("gpu") if any(d.platform == "gpu" for d in jax.devices()) else []
gpu = gpu_devices[0] if gpu_devices else None
print(f"CPU: {cpu}", flush=True)
if gpu:
    print(f"GPU: {gpu} ({gpu.device_kind})", flush=True)

from normix import (
    GeneralizedHyperbolic as GH_jax,
    VarianceGamma as VG_jax,
    NormalInverseGamma as NIGa_jax,
    NormalInverseGaussian as NIG_jax,
)


def init_jax_model(cls_name, X, key):
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    X_c = X - mu
    sigma = (X_c.T @ X_c) / n + 1e-4 * jnp.eye(d)
    gamma = 0.01 * jax.random.normal(key, (d,), dtype=jnp.float64)
    if cls_name == 'GH':
        return GH_jax._initialize(X, key)
    elif cls_name == 'VG':
        return VG_jax.from_classical(mu=mu, gamma=gamma, sigma=sigma, alpha=1.0, beta=1.0)
    elif cls_name == 'NIG':
        return NIG_jax.from_classical(mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.0)
    elif cls_name == 'NIG-alpha':
        return NIGa_jax.from_classical(mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0)


def profile_jax_em(model, X, n_iter, label=""):
    X = jnp.asarray(X, dtype=jnp.float64)
    e_times = []
    m_times = []
    reg_times = []
    ll_times = []
    iter_times = []

    for i in range(n_iter):
        t_iter_start = time.perf_counter()

        t0 = time.perf_counter()
        expectations = model.e_step(X)
        _ = float(expectations['E_Y'][0])
        t_e = time.perf_counter() - t0
        e_times.append(t_e)

        t0 = time.perf_counter()
        model = model.m_step(X, expectations)
        _ = float(model._joint.mu[0])
        t_m = time.perf_counter() - t0
        m_times.append(t_m)

        t0 = time.perf_counter()
        if hasattr(model, 'regularize_det_sigma_one'):
            model = model.regularize_det_sigma_one()
            _ = float(model._joint.mu[0])
        t_reg = time.perf_counter() - t0
        reg_times.append(t_reg)

        t0 = time.perf_counter()
        new_ll = model.marginal_log_likelihood(X)
        _ = float(new_ll)
        t_ll = time.perf_counter() - t0
        ll_times.append(t_ll)

        t_iter = time.perf_counter() - t_iter_start
        iter_times.append(t_iter)

        print(f"  [{label}] iter {i:2d}: "
              f"E={t_e:.3f}s  M={t_m:.3f}s  reg={t_reg:.3f}s  "
              f"LL={t_ll:.3f}s  total={t_iter:.3f}s  ll={float(new_ll):.4f}", flush=True)

    return {
        'e_step': e_times,
        'm_step': m_times,
        'regularize': reg_times,
        'log_likelihood': ll_times,
        'iteration': iter_times,
    }


jax_results = {}

for device_name, device in [("CPU", cpu)] + ([("GPU", gpu)] if gpu else []):
    jax_results[device_name] = {}

    for config_name, X_data in CONFIGS:
        jax_results[device_name][config_name] = {}
        print(f"\n--- {config_name} on JAX {device_name} ---", flush=True)

        X_jnp = jax.device_put(jnp.asarray(X_data, dtype=jnp.float64), device)
        key = jax.device_put(jax.random.PRNGKey(42), device)

        for short_name in ['NIG', 'VG', 'NIG-alpha', 'GH']:
            print(f"\n  {short_name} (JAX {device_name}):", flush=True)
            try:
                model = init_jax_model(short_name, X_jnp, key)

                # Warmup
                print(f"    Warmup...", flush=True)
                warmup = profile_jax_em(model, X_jnp, n_iter=1,
                                        label=f"jax-{short_name}-warm")

                model = init_jax_model(short_name, X_jnp, key)
                print(f"    Timed run...", flush=True)
                timings = profile_jax_em(model, X_jnp, n_iter=N_ITER,
                                         label=f"jax-{short_name}")

                skip = min(1, N_ITER - 1)
                e_avg = np.mean(timings['e_step'][skip:])
                m_avg = np.mean(timings['m_step'][skip:])
                iter_avg = np.mean(timings['iteration'][skip:])
                print(f"    SUMMARY: E={e_avg:.4f}s  M={m_avg:.4f}s  iter={iter_avg:.4f}s",
                      flush=True)
                jax_results[device_name][config_name][short_name] = {
                    'avg_e_step': e_avg,
                    'avg_m_step': m_avg,
                    'avg_iteration': iter_avg,
                }
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                jax_results[device_name][config_name][short_name] = {'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print("COMPARISON: NumPy vs JAX CPU vs JAX GPU", flush=True)
print("=" * 80, flush=True)

for config_name, _ in CONFIGS:
    print(f"\n--- {config_name} ---", flush=True)

    header = (f"{'Dist':<12} "
              f"{'NumPy E(s)':<12} {'NumPy M(s)':<12} {'NumPy tot(s)':<14} "
              f"{'JAX CPU E':<12} {'JAX CPU M':<12} {'JAX CPU tot':<14} "
              f"{'NP/JAX-CPU':<12}")
    if gpu:
        header += (f" {'JAX GPU E':<12} {'JAX GPU M':<12} {'JAX GPU tot':<14} "
                   f"{'NP/JAX-GPU':<12}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for sn in ['NIG', 'VG', 'NIG-alpha', 'GH']:
        np_r = numpy_results.get(config_name, {}).get(sn, {})
        jc_r = jax_results.get("CPU", {}).get(config_name, {}).get(sn, {})

        if 'error' in np_r or 'error' in jc_r:
            print(f"{sn:<12} ERROR", flush=True)
            continue

        np_e = np_r.get('avg_e_step', 0)
        np_m = np_r.get('avg_m_step', 0)
        np_t = np_r.get('avg_iteration', 0)
        jc_e = jc_r.get('avg_e_step', 0)
        jc_m = jc_r.get('avg_m_step', 0)
        jc_t = jc_r.get('avg_iteration', 0)
        ratio_cpu = np_t / jc_t if jc_t > 0 else 0

        line = (f"{sn:<12} "
                f"{np_e:<12.4f} {np_m:<12.4f} {np_t:<14.4f} "
                f"{jc_e:<12.4f} {jc_m:<12.4f} {jc_t:<14.4f} "
                f"{ratio_cpu:<12.2f}x")

        if gpu:
            jg_r = jax_results.get("GPU", {}).get(config_name, {}).get(sn, {})
            if 'error' not in jg_r:
                jg_e = jg_r.get('avg_e_step', 0)
                jg_m = jg_r.get('avg_m_step', 0)
                jg_t = jg_r.get('avg_iteration', 0)
                ratio_gpu = np_t / jg_t if jg_t > 0 else 0
                line += (f" {jg_e:<12.4f} {jg_m:<12.4f} {jg_t:<14.4f} "
                         f"{ratio_gpu:<12.2f}x")
            else:
                line += " ERROR"

        print(line, flush=True)

print("\n" + "=" * 80, flush=True)
print("ANALYSIS", flush=True)
print("=" * 80, flush=True)
print("""
NP/JAX-CPU < 1 means numpy is FASTER than JAX CPU.
NP/JAX-GPU < 1 means numpy is FASTER than JAX GPU.

Key differences in implementation:
  NumPy E-step: vectorized scipy.special.kve on entire array (one call)
  JAX E-step:   jax.vmap(GIG.expectation_params) → jax.grad → pure_callback per obs

  NumPy M-step (GH): scipy L-BFGS-B with analytical gradients, warm-started
  JAX M-step (GH):   scipy L-BFGS-B with numerical gradients via pure_callback,
                      multi-start (15+ starting points), no warm-start
""", flush=True)
