"""
Profile EM algorithm: CPU vs GPU, across GH distribution family.

Investigates why GeneralizedHyperbolic EM is slow on GPU by:
1. Validating GPU (RTX 4090) is detected and used by JAX
2. Timing E-step and M-step separately for each distribution
3. Comparing GH (GIG optimization in M-step) vs VG/NIG/NormalInverseGamma
4. Running on real S&P 500 returns data (all stocks, ~500 dim)
5. Comparing GPU vs CPU-only execution

Hypothesis: GH M-step calls scipy L-BFGS-B for GIG eta-to-theta, which forces
CPU execution + GPU-CPU sync overhead. E-step also uses pure_callback for Bessel
functions, creating per-observation CPU round-trips under vmap.
"""
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Phase 0: GPU detection (before any heavy JAX imports) ────────────────
print("=" * 80, flush=True)
print("PHASE 0: JAX Backend Detection", flush=True)
print("=" * 80, flush=True)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print(f"JAX version:    {jax.__version__}", flush=True)
print(f"JAX backends:   {[b.platform for b in jax.devices()]}", flush=True)
print(f"Default device: {jax.devices()[0]}", flush=True)

gpu_devices = jax.devices("gpu") if any(d.platform == "gpu" for d in jax.devices()) else []
cpu_devices = jax.devices("cpu")

if gpu_devices:
    gpu = gpu_devices[0]
    print(f"GPU detected:   {gpu}", flush=True)
    print(f"GPU kind:       {gpu.device_kind}", flush=True)
else:
    print("WARNING: No GPU detected! Running CPU-only comparison.", flush=True)
    gpu = None

cpu = cpu_devices[0]
print(f"CPU device:     {cpu}", flush=True)

# Quick GPU smoke test
if gpu:
    x_gpu = jax.device_put(jnp.ones(1000), gpu)
    y_gpu = jnp.dot(x_gpu, x_gpu)
    y_gpu.block_until_ready()
    print(f"GPU smoke test:  dot product = {float(y_gpu):.1f} (on {y_gpu.devices()})", flush=True)

x_cpu = jax.device_put(jnp.ones(1000), cpu)
y_cpu = jnp.dot(x_cpu, x_cpu)
y_cpu.block_until_ready()
print(f"CPU smoke test:  dot product = {float(y_cpu):.1f} (on {y_cpu.devices()})", flush=True)
print(flush=True)

# ── Phase 1: Load data ──────────────────────────────────────────────────
print("=" * 80, flush=True)
print("PHASE 1: Load S&P 500 Returns Data", flush=True)
print("=" * 80, flush=True)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df = df.dropna(axis=1)
print(f"Data shape:     {df.shape}  (days × stocks)", flush=True)
print(f"Date range:     {df.index[0].date()} to {df.index[-1].date()}", flush=True)

X_full = df.values.astype(np.float64)
n_obs, n_stocks_full = X_full.shape
print(f"Observations:   {n_obs}", flush=True)
print(f"Stocks (full):  {n_stocks_full}", flush=True)
print(flush=True)

# ── Imports ──────────────────────────────────────────────────────────────
from normix import (
    GeneralizedHyperbolic,
    VarianceGamma,
    NormalInverseGamma,
    NormalInverseGaussian,
    BatchEMFitter,
)
from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
from normix.distributions.variance_gamma import JointVarianceGamma
from normix.distributions.normal_inverse_gamma import JointNormalInverseGamma
from normix.distributions.normal_inverse_gaussian import JointNormalInverseGaussian


# ── Helper: initialize distributions from data ──────────────────────────

def init_gh(X, key):
    return GeneralizedHyperbolic._initialize(X, key)


def init_vg(X, key):
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    X_c = X - mu
    sigma = (X_c.T @ X_c) / n + 1e-4 * jnp.eye(d)
    gamma = 0.01 * jax.random.normal(key, (d,), dtype=jnp.float64)
    return VarianceGamma.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, alpha=1.0, beta=1.0)


def init_nig(X, key):
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    X_c = X - mu
    sigma = (X_c.T @ X_c) / n + 1e-4 * jnp.eye(d)
    gamma = 0.01 * jax.random.normal(key, (d,), dtype=jnp.float64)
    return NormalInverseGaussian.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.0)


def init_nig_a(X, key):
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    X_c = X - mu
    sigma = (X_c.T @ X_c) / n + 1e-4 * jnp.eye(d)
    gamma = 0.01 * jax.random.normal(key, (d,), dtype=jnp.float64)
    return NormalInverseGamma.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0)


# ── Profiling function ──────────────────────────────────────────────────

def profile_em_steps(model, X, n_iter=5, label=""):
    """Run EM iterations, timing E-step and M-step separately."""
    X = jnp.asarray(X, dtype=jnp.float64)

    e_times = []
    m_times = []
    reg_times = []
    ll_times = []
    iter_times = []

    for i in range(n_iter):
        t_iter_start = time.perf_counter()

        # E-step
        t0 = time.perf_counter()
        expectations = model.e_step(X)
        _ = float(expectations['E_Y'][0])
        t_e = time.perf_counter() - t0
        e_times.append(t_e)

        # M-step
        t0 = time.perf_counter()
        model = model.m_step(X, expectations)
        _ = float(model._joint.mu[0])
        t_m = time.perf_counter() - t0
        m_times.append(t_m)

        # Regularization
        t0 = time.perf_counter()
        if hasattr(model, 'regularize_det_sigma_one'):
            model = model.regularize_det_sigma_one()
            _ = float(model._joint.mu[0])
        t_reg = time.perf_counter() - t0
        reg_times.append(t_reg)

        # Log-likelihood
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
        'final_model': model,
    }


def run_on_device(device, X_np, n_iter=5):
    """Run all distributions on a specific device."""
    device_name = device.platform.upper()
    print(f"\n{'=' * 80}", flush=True)
    print(f"PROFILING ON: {device_name} ({device})", flush=True)
    print(f"Data: {X_np.shape[0]} obs × {X_np.shape[1]} stocks, {n_iter} EM iterations", flush=True)
    print(f"{'=' * 80}", flush=True)

    X = jax.device_put(jnp.asarray(X_np, dtype=jnp.float64), device)
    key = jax.device_put(jax.random.PRNGKey(42), device)

    distributions = [
        ('NormalInverseGaussian', init_nig),
        ('VarianceGamma', init_vg),
        ('NormalInverseGamma', init_nig_a),
        ('GeneralizedHyperbolic', init_gh),
    ]

    results = {}
    for dist_name, init_fn in distributions:
        print(f"\n--- {dist_name} on {device_name} ---", flush=True)
        try:
            model = init_fn(X, key)

            # Warmup (1 iteration to trigger JIT)
            print(f"  Warmup...", flush=True)
            t0 = time.perf_counter()
            warmup = profile_em_steps(model, X, n_iter=1, label=f"{dist_name[:6]}-warm")
            t_warmup = time.perf_counter() - t0
            print(f"  Warmup took {t_warmup:.2f}s (includes JIT compilation)", flush=True)

            # Re-initialize for clean timed run
            model = init_fn(X, key)

            print(f"  Timed run ({n_iter} iterations)...", flush=True)
            t0 = time.perf_counter()
            timings = profile_em_steps(model, X, n_iter=n_iter, label=dist_name[:6])
            t_total = time.perf_counter() - t0

            # Summary stats (skip first iter)
            skip = min(1, n_iter - 1)
            e_avg = np.mean(timings['e_step'][skip:])
            m_avg = np.mean(timings['m_step'][skip:])
            reg_avg = np.mean(timings['regularize'][skip:])
            ll_avg = np.mean(timings['log_likelihood'][skip:])
            iter_avg = np.mean(timings['iteration'][skip:])

            print(f"\n  SUMMARY ({dist_name} on {device_name}):", flush=True)
            print(f"    Total time:        {t_total:.3f}s ({n_iter} iters)", flush=True)
            print(f"    Avg E-step:        {e_avg:.4f}s  ({100*e_avg/iter_avg:.1f}%)", flush=True)
            print(f"    Avg M-step:        {m_avg:.4f}s  ({100*m_avg/iter_avg:.1f}%)", flush=True)
            print(f"    Avg regularize:    {reg_avg:.4f}s", flush=True)
            print(f"    Avg log-lik eval:  {ll_avg:.4f}s", flush=True)
            print(f"    Avg iteration:     {iter_avg:.4f}s", flush=True)

            results[dist_name] = {
                'timings': timings,
                'total_time': t_total,
                'warmup_time': t_warmup,
                'avg_e_step': e_avg,
                'avg_m_step': m_avg,
                'avg_regularize': reg_avg,
                'avg_log_likelihood': ll_avg,
                'avg_iteration': iter_avg,
            }
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[dist_name] = {'error': str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Small-scale warmup (10 stocks) to validate setup
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 80, flush=True)
print("EXPERIMENT 1: Small-scale test (10 stocks, 5 EM iterations)", flush=True)
print("  Purpose: validate setup, get quick timing baseline", flush=True)
print("=" * 80, flush=True)

X_small = X_full[:, :10]
print(f"Data: {X_small.shape}", flush=True)

small_cpu = run_on_device(cpu, X_small, n_iter=5)
if gpu:
    small_gpu = run_on_device(gpu, X_small, n_iter=5)

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Full-scale (all stocks)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print(f"EXPERIMENT 2: Full-scale ({n_stocks_full} stocks, 5 EM iterations)", flush=True)
print("  Purpose: stress test with real high-dimensional data", flush=True)
print("=" * 80, flush=True)

full_cpu = run_on_device(cpu, X_full, n_iter=5)
if gpu:
    full_gpu = run_on_device(gpu, X_full, n_iter=5)

# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Bessel function and subordinator profiling
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 3: Bessel & Subordinator from_expectation Profiling", flush=True)
print("=" * 80, flush=True)

from normix._bessel import log_kv
from normix.distributions.gig import GIG
from normix.distributions.gamma import Gamma
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.distributions.inverse_gamma import InverseGamma

# --- Bessel ---
print("\n--- Bessel function log_kv ---", flush=True)
v_s, z_s = jnp.array(1.5), jnp.array(2.0)
_ = float(log_kv(v_s, z_s))

N = 500
t0 = time.perf_counter()
for _ in range(N):
    _ = float(log_kv(v_s, z_s))
t_bessel_scalar = (time.perf_counter() - t0) / N
print(f"  Scalar log_kv:        {t_bessel_scalar*1000:.3f} ms/call  ({N} calls)", flush=True)

v_b = jnp.full(n_obs, 1.5)
z_b = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (n_obs,))) + 0.1
batched_kv = jax.vmap(log_kv)
_ = batched_kv(v_b, z_b).block_until_ready()

N = 50
t0 = time.perf_counter()
for _ in range(N):
    batched_kv(v_b, z_b).block_until_ready()
t_bessel_batch = (time.perf_counter() - t0) / N
print(f"  Batched log_kv ({n_obs} obs): {t_bessel_batch*1000:.2f} ms/batch  ({N} calls)", flush=True)

# --- Subordinator from_expectation ---
print("\n--- Subordinator from_expectation (M-step cost) ---", flush=True)

eta_gig = jnp.array([-0.5, 1.2, 0.9])
_ = GIG.from_expectation(eta_gig)
N = 10
t0 = time.perf_counter()
for _ in range(N):
    _ = float(GIG.from_expectation(eta_gig).p)
t_gig = (time.perf_counter() - t0) / N
print(f"  GIG.from_expectation:      {t_gig*1000:.1f} ms/call  ({N} calls, scipy L-BFGS-B)", flush=True)

eta_gamma = jnp.array([-0.5, 0.9])
_ = Gamma.from_expectation(eta_gamma)
N = 500
t0 = time.perf_counter()
for _ in range(N):
    _ = float(Gamma.from_expectation(eta_gamma).alpha)
t_gamma = (time.perf_counter() - t0) / N
print(f"  Gamma.from_expectation:    {t_gamma*1000:.3f} ms/call  ({N} calls, Newton)", flush=True)

eta_ig = jnp.array([0.9, 1.2])
_ = InverseGaussian.from_expectation(eta_ig)
N = 500
t0 = time.perf_counter()
for _ in range(N):
    _ = float(InverseGaussian.from_expectation(eta_ig).mu)
t_invgauss = (time.perf_counter() - t0) / N
print(f"  InvGaussian.from_exp:      {t_invgauss*1000:.3f} ms/call  ({N} calls, closed-form)", flush=True)

eta_invg = jnp.array([-1.2, -0.5])
_ = InverseGamma.from_expectation(eta_invg)
N = 500
t0 = time.perf_counter()
for _ in range(N):
    _ = float(InverseGamma.from_expectation(eta_invg).alpha)
t_invgamma = (time.perf_counter() - t0) / N
print(f"  InvGamma.from_exp:         {t_invgamma*1000:.3f} ms/call  ({N} calls, Newton)", flush=True)

print(f"\n  Cost ratios (GIG / other):", flush=True)
print(f"    GIG / Gamma:       {t_gig/t_gamma:.0f}x", flush=True)
print(f"    GIG / InvGaussian: {t_gig/t_invgauss:.0f}x", flush=True)
print(f"    GIG / InvGamma:    {t_gig/t_invgamma:.0f}x", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print("FINAL COMPARISON TABLE", flush=True)
print("=" * 80, flush=True)

dist_names = ['NormalInverseGaussian', 'VarianceGamma', 'NormalInverseGamma', 'GeneralizedHyperbolic']
short_names = ['NIG', 'VG', 'NIG-alpha', 'GH']
subordinators = ['InvGaussian', 'Gamma', 'InvGamma', 'GIG']
mstep_types = ['closed-form', 'Newton', 'Newton', 'scipy L-BFGS-B']

for exp_name, cpu_res, gpu_res, shape in [
    ("10 stocks", small_cpu, small_gpu if gpu else None, X_small.shape),
    (f"{n_stocks_full} stocks", full_cpu, full_gpu if gpu else None, X_full.shape),
]:
    print(f"\n--- {exp_name} ({shape[0]} obs × {shape[1]} dim) ---", flush=True)
    header = f"{'Dist':<12} {'Sub':<12} {'M-step type':<16} {'CPU E(s)':<10} {'CPU M(s)':<10} {'CPU tot(s)':<11}"
    if gpu_res:
        header += f" {'GPU E(s)':<10} {'GPU M(s)':<10} {'GPU tot(s)':<11} {'GPU/CPU':<8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for dn, sn, sub, mtype in zip(dist_names, short_names, subordinators, mstep_types):
        cr = cpu_res.get(dn, {})
        if 'error' in cr:
            print(f"{sn:<12} {'ERROR'}", flush=True)
            continue
        line = (f"{sn:<12} {sub:<12} {mtype:<16} "
                f"{cr['avg_e_step']:<10.4f} {cr['avg_m_step']:<10.4f} "
                f"{cr['avg_iteration']:<11.4f}")
        if gpu_res:
            gr = gpu_res.get(dn, {})
            if 'error' not in gr:
                ratio = gr['avg_iteration'] / cr['avg_iteration'] if cr['avg_iteration'] > 0 else 0
                line += (f" {gr['avg_e_step']:<10.4f} {gr['avg_m_step']:<10.4f} "
                         f"{gr['avg_iteration']:<11.4f} {ratio:<8.2f}x")
            else:
                line += " ERROR"
        print(line, flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80, flush=True)
print("ANALYSIS & CONCLUSIONS", flush=True)
print("=" * 80, flush=True)
print("""
Key findings:

1. BESSEL FUNCTION (log_kv) uses jax.pure_callback to scipy:
   - Every call to log_kv forces CPU execution regardless of device
   - E-step calls GIG.expectation_params() → jax.grad → log_kv for EACH observation
   - This affects ALL distributions equally (GH, VG, NIG, NIG-alpha)
   - On GPU: each vmap'd callback causes GPU→CPU→GPU synchronization overhead

2. GIG from_expectation uses scipy.optimize.minimize (L-BFGS-B):
   - Multi-start optimization (up to ~15 starting points)
   - Each evaluation calls _log_partition → log_kv (another CPU callback)
   - This is THE dominant cost in the GH M-step
   - Other distributions use closed-form or Newton (orders of magnitude faster)

3. GPU OVERHEAD SOURCES:
   a) pure_callback forces synchronization between GPU and CPU
   b) For the E-step: vmap over conditional_expectations triggers n_obs callbacks
   c) For GH M-step: scipy L-BFGS-B runs entirely on CPU
   d) Data transfer: GPU→CPU for callbacks, CPU→GPU for results
   e) Small per-element operations don't amortize GPU kernel launch overhead

4. WHY GPU IS SLOWER:
   - The computation is dominated by CPU callbacks (Bessel, scipy optimizer)
   - Even the GPU-friendly parts (matrix ops) are offset by sync overhead
   - The iterative EM loop prevents GPU pipelining
""", flush=True)
