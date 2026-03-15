"""
Benchmark GIG η→θ solver variants for the EM M-step.

Compares three approaches on warm-start from a previous θ:
  1. scipy L-BFGS-B  — gradient-only, CPU-only
  2. Newton autodiff  — jax.hessian, scan(20), pure JAX
  3. Newton analytical — 7 log_kv calls/step, scan(20), pure JAX
  4. JAXopt L-BFGS   — gradient-only, implicit_diff=True, pure JAX

Metrics:
  - Compile time (first call, JIT overhead)
  - Execution time per call (after warm-up)
  - Accuracy (vs scipy cold-start reference)
  - GH EM M-step timing (10 stocks)

Usage:
    uv run python scripts/benchmark_gig_solvers.py
"""
import os
import time
import numpy as np
import pandas as pd

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("=" * 65)
print("GIG Solver Benchmark")
print(f"Device: {jax.devices()[0]}")
print("=" * 65)

from normix.distributions.gig import (
    GIG,
    _solve_jaxopt,
    _solve_newton_analytical,
    _solve_lbfgs,
    _initial_guesses,
    _solve_scipy,
)

# ── Test points covering GIG operating range ──────────────────────────────
TEST_CASES = [
    # (eta, label) — eta = [E[log X], E[1/X], E[X]]
    (jnp.array([-0.5,  1.2,  0.9]), "moderate"),
    (jnp.array([ 0.3,  0.8,  1.5]), "near-Gamma"),
    (jnp.array([-1.2,  3.0,  0.5]), "heavy-tail"),
    (jnp.array([ 0.1,  2.0,  1.0]), "near-IG"),
    (jnp.array([-0.8,  5.0,  0.3]), "extreme"),
]

# Get scipy reference solutions (gold standard)
def scipy_reference(eta):
    eta0_list = _initial_guesses(eta)
    return _solve_scipy(eta, eta0_list, 500, 1e-12)

print("\n── Reference solutions (scipy cold-start) ──")
refs = {}
for eta, label in TEST_CASES:
    t0 = time.perf_counter()
    ref = scipy_reference(eta)
    t = time.perf_counter() - t0
    refs[label] = ref
    print(f"  {label:12s}: θ = {np.array(ref).round(5)}  ({t*1000:.1f}ms)")


# ── Solver comparison ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SOLVER COMPARISON (warm-start from reference θ)")
print("=" * 65)

N_RUNS = 20  # timed runs after warm-up

for solver_name, solver_fn in [
    ("newton_autodiff",   _solve_jaxopt),
    ("newton_analytical", _solve_newton_analytical),
    ("lbfgs_jaxopt",      _solve_lbfgs),
]:
    print(f"\n── {solver_name} ──")

    compile_times = []
    exec_times = []
    max_errors = []

    for eta, label in TEST_CASES:
        theta0 = refs[label]  # warm-start from reference

        # First call: includes JIT compilation
        t0 = time.perf_counter()
        result = solver_fn(eta, theta0, 500, 1e-10)
        result.block_until_ready()
        t_compile = time.perf_counter() - t0
        compile_times.append(t_compile)

        # Check accuracy vs reference
        err = float(jnp.max(jnp.abs(result - refs[label])))
        max_errors.append(err)

        # Timed runs
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            result = solver_fn(eta, theta0, 500, 1e-10)
            result.block_until_ready()
            times.append(time.perf_counter() - t0)

        exec_time = np.median(times)
        exec_times.append(exec_time)

        print(f"  {label:12s}: compile={t_compile*1000:.0f}ms  "
              f"exec={exec_time*1000:.1f}ms  err={err:.2e}")

    print(f"  {'AVERAGE':12s}: compile={np.mean(compile_times)*1000:.0f}ms  "
          f"exec={np.mean(exec_times)*1000:.1f}ms  "
          f"max_err={max(max_errors):.2e}")

# ── scipy L-BFGS-B warm-start (single-start, no multi-start) ──────────────
print(f"\n── scipy L-BFGS-B (warm-start, single-start) ──")

from normix.distributions.gig import _objective_jit, _grad_objective
from scipy.optimize import minimize

def scipy_warm_start(eta, theta0, tol=1e-10):
    """Single-start scipy with warm-start from theta0."""
    eta_np = np.array(eta)
    theta0_np = np.array(theta0)
    theta0_np[1] = min(theta0_np[1], -1e-8)
    theta0_np[2] = min(theta0_np[2], -1e-8)
    bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
    eta_jnp = jnp.array(eta_np)

    res = minimize(
        fun=lambda t: float(_objective_jit(jnp.array(t), eta_jnp)),
        x0=theta0_np,
        jac=lambda t: np.array(_grad_objective(jnp.array(t), eta_jnp)),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'gtol': tol},
    )
    return jnp.array(res.x)

# Warm up JIT for gradient
_ = _grad_objective(jnp.array([0.5, -0.5, -0.5]), jnp.array([-0.5, 1.2, 0.9]))

scipy_exec_times = []
scipy_errors = []
for eta, label in TEST_CASES:
    theta0 = refs[label]
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = scipy_warm_start(eta, theta0)
        times.append(time.perf_counter() - t0)
    err = float(jnp.max(jnp.abs(result - refs[label])))
    exec_time = np.median(times)
    scipy_exec_times.append(exec_time)
    scipy_errors.append(err)
    print(f"  {label:12s}: exec={exec_time*1000:.1f}ms  err={err:.2e}")

print(f"  {'AVERAGE':12s}: exec={np.mean(scipy_exec_times)*1000:.1f}ms  "
      f"max_err={max(scipy_errors):.2e}")

# ── GH EM M-step timing ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("GH EM M-STEP TIMING (10 stocks)")
print("=" * 65)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
try:
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
    X10 = jnp.asarray(df.values[:, :10], dtype=np.float64)
    key = jax.random.PRNGKey(42)

    from normix import GeneralizedHyperbolic

    for solver_name in ["newton", "newton_analytical", "lbfgs"]:
        print(f"\n── {solver_name} ──")
        model = GeneralizedHyperbolic._initialize(X10, key)

        # Warmup E-step
        exp = model.e_step(X10)
        _ = float(exp['E_Y'][0])

        # Warmup M-step (compile)
        t0 = time.perf_counter()
        m2 = model.m_step(X10, exp, solver=solver_name)
        _ = float(m2._joint.mu[0])
        t_compile = time.perf_counter() - t0
        print(f"  M-step compile: {t_compile:.1f}s")

        # Timed M-steps
        times = []
        for i in range(4):
            exp = model.e_step(X10)
            _ = float(exp['E_Y'][0])
            t0 = time.perf_counter()
            m2 = model.m_step(X10, exp, solver=solver_name)
            _ = float(m2._joint.mu[0])
            times.append(time.perf_counter() - t0)
            print(f"  M-step iter {i+1}: {times[-1]:.3f}s")

        print(f"  M-step avg: {np.mean(times):.3f}s")

except Exception as e:
    print(f"  Skipped GH timing: {e}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — per-call execution time (ms, median over 20 runs)")
print("=" * 65)
print(f"""
  Solver                  Exec time   log_kv/step  Notes
  ─────────────────────── ─────────── ──────────── ─────────────────────
  scipy L-BFGS-B          (see above) ~5           CPU-only, not JITable
  Newton (autodiff)       (see above) ~25          scan=20, jax.hessian
  Newton (analytical)     (see above) ~7           scan=20, 7 batched calls
  JAXopt LBFGS            (see above) ~5           implicit_diff=True
""")
