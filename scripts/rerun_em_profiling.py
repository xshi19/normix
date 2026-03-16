"""
Rerun EM profiling with current code (pure-JAX Bessel + Newton solver).

Reproduces the table structure of docs/tech_notes/em_gpu_profiling.md.
Runs 5 EM iterations (1 warmup + 4 timed) per distribution per device.
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("=" * 80, flush=True)
print("EM Algorithm Re-Profiling (current: pure-JAX Bessel + Newton solver)")
print("=" * 80, flush=True)
print(f"JAX version: {jax.__version__}", flush=True)
print(f"Devices: {[str(d) for d in jax.devices()]}", flush=True)

gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
cpu_devices = jax.devices("cpu")
gpu = gpu_devices[0] if gpu_devices else None
cpu = cpu_devices[0]
print(f"GPU: {gpu}", flush=True)

# ── Load data ──
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
X_full = df.values.astype(np.float64)
n_obs, n_stocks = X_full.shape
print(f"Data: {n_obs} obs × {n_stocks} stocks\n", flush=True)

from normix import (
    GeneralizedHyperbolic,
    VarianceGamma,
    NormalInverseGamma,
    NormalInverseGaussian,
)

def init_gh(X, key):   return GeneralizedHyperbolic._initialize(X, key)
def init_vg(X, key):
    n, d = X.shape; mu = jnp.mean(X,axis=0); X_c = X-mu
    sigma = (X_c.T@X_c)/n + 1e-4*jnp.eye(d)
    gamma = 0.01*jax.random.normal(key,(d,),dtype=jnp.float64)
    return VarianceGamma.from_classical(mu=mu,gamma=gamma,sigma=sigma,alpha=1.,beta=1.)
def init_nig(X, key):
    n, d = X.shape; mu = jnp.mean(X,axis=0); X_c = X-mu
    sigma = (X_c.T@X_c)/n + 1e-4*jnp.eye(d)
    gamma = 0.01*jax.random.normal(key,(d,),dtype=jnp.float64)
    return NormalInverseGaussian.from_classical(mu=mu,gamma=gamma,sigma=sigma,mu_ig=1.,lam=1.)
def init_niga(X, key):
    n, d = X.shape; mu = jnp.mean(X,axis=0); X_c = X-mu
    sigma = (X_c.T@X_c)/n + 1e-4*jnp.eye(d)
    gamma = 0.01*jax.random.normal(key,(d,),dtype=jnp.float64)
    return NormalInverseGamma.from_classical(mu=mu,gamma=gamma,sigma=sigma,alpha=2.,beta=1.)

DISTS = [
    ("NIG",   init_nig,  "InvGaussian", "closed-form"),
    ("VG",    init_vg,   "Gamma",       "jaxopt LBFGS"),
    ("NIG-α", init_niga, "InvGamma",    "jaxopt LBFGS"),
    ("GH",    init_gh,   "GIG",         "JAX Newton (pure-JAX Bessel)"),
]

def profile_em(model, X, n_iter=5, label=""):
    X = jnp.asarray(X, dtype=jnp.float64)
    e_times, m_times, iter_times = [], [], []
    for i in range(n_iter):
        t_start = time.perf_counter()
        t0 = time.perf_counter()
        exp = model.e_step(X); _ = float(exp['E_Y'][0])
        t_e = time.perf_counter() - t0
        t0 = time.perf_counter()
        model = model.m_step(X, exp); _ = float(model._joint.mu[0])
        t_m = time.perf_counter() - t0
        t0 = time.perf_counter()
        if hasattr(model, 'regularize_det_sigma_one'):
            model = model.regularize_det_sigma_one(); _ = float(model._joint.mu[0])
        ll = model.marginal_log_likelihood(X); _ = float(ll)
        t_total = time.perf_counter() - t_start
        e_times.append(t_e); m_times.append(t_m); iter_times.append(t_total)
        print(f"  [{label}] iter {i}: E={t_e:.3f}s  M={t_m:.3f}s  total={t_total:.3f}s  ll={float(ll):.2f}", flush=True)
    return {'e': e_times, 'm': m_times, 'total': iter_times, 'model': model}

def run_device(device, X_np, device_label, n_iter=5, skip=1):
    results = {}
    X = jax.device_put(jnp.asarray(X_np, dtype=jnp.float64), device)
    key = jax.device_put(jax.random.PRNGKey(42), device)
    for name, init_fn, sub, mtype in DISTS:
        print(f"\n  ── {name} ({device_label}, {X_np.shape[1]} stocks) ──", flush=True)
        try:
            model = init_fn(X, key)
            print("  Warmup (1 iter)...", flush=True)
            profile_em(model, X, n_iter=1, label="warm")
            model = init_fn(X, key)
            print(f"  Timed ({n_iter} iters)...", flush=True)
            r = profile_em(model, X, n_iter=n_iter, label=name)
            avg_e = np.mean(r['e'][skip:])
            avg_m = np.mean(r['m'][skip:])
            avg_t = np.mean(r['total'][skip:])
            print(f"  SUMMARY: E={avg_e:.3f}s  M={avg_m:.3f}s  iter={avg_t:.3f}s", flush=True)
            results[name] = {'e': avg_e, 'm': avg_m, 'total': avg_t}
        except Exception as ex:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(ex)}
    return results

# ═══════════════════════════════════════════════════════════════════════
# 10-stock experiment
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 1: 10 stocks, 5 timed EM iterations", flush=True)
print("=" * 80, flush=True)

X_small = X_full[:, :10]
cpu_10 = run_device(cpu, X_small, "CPU-10", n_iter=5)
if gpu: gpu_10 = run_device(gpu, X_small, "GPU-10", n_iter=5)

# ═══════════════════════════════════════════════════════════════════════
# 468-stock experiment
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80, flush=True)
print(f"EXPERIMENT 2: {n_stocks} stocks, 5 timed EM iterations", flush=True)
print("=" * 80, flush=True)

cpu_full = run_device(cpu, X_full, "CPU-full", n_iter=5)
if gpu: gpu_full = run_device(gpu, X_full, "GPU-full", n_iter=5)

# ═══════════════════════════════════════════════════════════════════════
# Summary tables
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80, flush=True)
print("SUMMARY TABLES", flush=True)
print("=" * 80, flush=True)

def fmt(d, k):
    if 'error' in d: return "ERROR"
    return f"{d.get(k, 0):.3f}"

print(f"\n10-stock iterations (seconds, avg over iters 2-5):", flush=True)
print(f"{'Dist':<8} {'Sub':<12} {'CPU E':>7} {'CPU M':>7} {'CPU tot':>8}", end="", flush=True)
if gpu: print(f"  {'GPU E':>7} {'GPU M':>7} {'GPU tot':>8}", end="", flush=True)
print(flush=True)
for name, _, sub, mtype in DISTS:
    r_cpu = cpu_10.get(name, {})
    line = f"{name:<8} {sub:<12} {fmt(r_cpu,'e'):>7} {fmt(r_cpu,'m'):>7} {fmt(r_cpu,'total'):>8}"
    if gpu:
        r_gpu = gpu_10.get(name, {})
        line += f"  {fmt(r_gpu,'e'):>7} {fmt(r_gpu,'m'):>7} {fmt(r_gpu,'total'):>8}"
    print(line, flush=True)

print(f"\n{n_stocks}-stock iterations (seconds, avg over iters 2-5):", flush=True)
print(f"{'Dist':<8} {'Sub':<12} {'CPU E':>7} {'CPU M':>7} {'CPU tot':>8}", end="", flush=True)
if gpu: print(f"  {'GPU E':>7} {'GPU M':>7} {'GPU tot':>8}", end="", flush=True)
print(flush=True)
for name, _, sub, mtype in DISTS:
    r_cpu = cpu_full.get(name, {})
    line = f"{name:<8} {sub:<12} {fmt(r_cpu,'e'):>7} {fmt(r_cpu,'m'):>7} {fmt(r_cpu,'total'):>8}"
    if gpu:
        r_gpu = gpu_full.get(name, {})
        line += f"  {fmt(r_gpu,'e'):>7} {fmt(r_gpu,'m'):>7} {fmt(r_gpu,'total'):>8}"
    print(line, flush=True)

print("\nCurrent architecture:", flush=True)
print("  - log_kv: pure JAX, zero scipy callbacks (lax.cond regime selection)", flush=True)
print("  - E-step: vmap over pure-JAX GIG expectation_params", flush=True)
print("  - M-step GIG: JAX Newton, autodiff Hessian, lax.scan(length=20), warm-start", flush=True)
print("  - M-step other: unchanged (jaxopt LBFGS / closed-form)", flush=True)
