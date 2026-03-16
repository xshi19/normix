"""
Benchmark GIG eta->theta solver variants for the EM M-step.

Compares:
  1. scipy L-BFGS-B wrapped around JAX objective/grad
  2. legacy pure NumPy/SciPy expectation->natural solver
  3. JAX Newton (autodiff Hessian), scan_length=20
  4. JAX Newton (autodiff Hessian), scan_length=5
  5. JAX Newton (analytical Hessian), scan_length=20
  6. JAX Newton (analytical Hessian), scan_length=5
  7. JAXopt LBFGS (implicit_diff=True)

For each solver we report:
  - compile time (for JAX solvers)
  - execution time after warmup
  - objective value f(theta)=psi(theta)-theta·eta
  - objective gap to the best solver
  - parameter error vs a high-accuracy scipy cold-start reference
"""
import os
import time
import numpy as np
import pandas as pd

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("=" * 88)
print("GIG Solver Benchmark")
print(f"Device: {jax.devices()[0]}")
print("=" * 88)

from normix.distributions.gig import (
    GIG,
    _solve_jaxopt,
    _solve_newton_analytical,
    _solve_lbfgs,
    _initial_guesses,
    _solve_scipy,
    _objective_jit,
    _grad_objective,
)
from normix_numpy.distributions.univariate.generalized_inverse_gaussian import (
    GeneralizedInverseGaussian as LegacyGIG,
)
from scipy.optimize import minimize

# ── Realistic test points generated from actual fitted GIGs ───────────────
SOURCE_GIGS = [
    GIG(p=0.5, a=1.0, b=1.0),
    GIG(p=1.5, a=2.0, b=0.5),
    GIG(p=-0.5, a=0.5, b=2.0),
    GIG(p=2.0, a=1.0, b=0.1),
]
TEST_CASES = []
for gig in SOURCE_GIGS:
    label = f"GIG(p={float(gig.p):.1f},a={float(gig.a):.1f},b={float(gig.b):.1f})"
    TEST_CASES.append((gig.expectation_params(), gig.natural_params(), label))

def scipy_reference(eta):
    eta0_list = _initial_guesses(eta)
    return _solve_scipy(eta, eta0_list, 500, 1e-12)

def objective_value(theta, eta):
    return float(_objective_jit(jnp.asarray(theta, dtype=jnp.float64),
                                jnp.asarray(eta, dtype=jnp.float64)))

def scipy_warm_start(eta, theta0, tol=1e-10):
    theta0_np = np.array(theta0, dtype=np.float64)
    theta0_np[1] = min(theta0_np[1], -1e-8)
    theta0_np[2] = min(theta0_np[2], -1e-8)
    bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
    eta_jnp = jnp.asarray(np.array(eta), dtype=jnp.float64)
    res = minimize(
        fun=lambda t: float(_objective_jit(jnp.asarray(t, dtype=jnp.float64), eta_jnp)),
        x0=theta0_np,
        jac=lambda t: np.array(_grad_objective(jnp.asarray(t, dtype=jnp.float64), eta_jnp)),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": tol**2, "gtol": tol},
    )
    return np.asarray(res.x), int(res.nit), float(res.fun)

def legacy_numpy_warm_start(eta, theta0):
    legacy = LegacyGIG()
    theta = legacy._expectation_to_natural(np.asarray(eta, dtype=float),
                                           theta0=np.asarray(theta0, dtype=float))
    return np.asarray(theta)

print("\n── Reference solutions (scipy cold-start, multi-start) ──")
refs = {}
for eta, _, label in TEST_CASES:
    t0 = time.perf_counter()
    ref = scipy_reference(eta)
    t = time.perf_counter() - t0
    refs[label] = ref
    print(f"  {label:28s}: θ = {np.array(ref).round(5)}  ({t*1000:.1f}ms)")

print("\n" + "=" * 88)
print("SOLVER COMPARISON (warm-start from known theta0)")
print("=" * 88)

N_RUNS_JAX = 10
N_RUNS_SCIPY = 50

solver_specs = [
    ("scipy_wrapped", None, None),
    ("legacy_numpy_scipy", None, None),
    ("newton_autodiff_s20", _solve_jaxopt, {"scan_length": 20}),
    ("newton_autodiff_s5", _solve_jaxopt, {"scan_length": 5}),
    ("newton_analytical_s20", _solve_newton_analytical, {"scan_length": 20}),
    ("newton_analytical_s5", _solve_newton_analytical, {"scan_length": 5}),
    ("lbfgs_jaxopt", _solve_lbfgs, {}),
]

rows = []

# Warm up shared JIT'd objective/grad for scipy-wrapped path
_ = _grad_objective(jnp.array([0.5, -0.5, -0.5]), TEST_CASES[0][0])

for eta, theta0, label in TEST_CASES:
    ref_theta = refs[label]
    best_obj = objective_value(ref_theta, eta)

    for solver_name, solver_fn, kwargs in solver_specs:
        compile_ms = None
        nit = None

        if solver_name == "scipy_wrapped":
            result, nit, obj_val = scipy_warm_start(eta, theta0)
            times = []
            for _ in range(N_RUNS_SCIPY):
                t0 = time.perf_counter()
                result, nit, obj_val = scipy_warm_start(eta, theta0)
                times.append(time.perf_counter() - t0)
            exec_ms = 1000.0 * float(np.median(times))
        elif solver_name == "legacy_numpy_scipy":
            result = legacy_numpy_warm_start(eta, theta0)
            times = []
            for _ in range(N_RUNS_SCIPY):
                t0 = time.perf_counter()
                result = legacy_numpy_warm_start(eta, theta0)
                times.append(time.perf_counter() - t0)
            exec_ms = 1000.0 * float(np.median(times))
            obj_val = objective_value(result, eta)
        else:
            t0 = time.perf_counter()
            result = solver_fn(eta, theta0, 500, 1e-10, **kwargs)
            result.block_until_ready()
            compile_ms = 1000.0 * (time.perf_counter() - t0)
            times = []
            for _ in range(N_RUNS_JAX):
                t0 = time.perf_counter()
                result = solver_fn(eta, theta0, 500, 1e-10, **kwargs)
                result.block_until_ready()
                times.append(time.perf_counter() - t0)
            exec_ms = 1000.0 * float(np.median(times))
            result = np.asarray(result)
            obj_val = objective_value(result, eta)

        theta_err = float(np.max(np.abs(np.asarray(result) - np.asarray(ref_theta))))
        obj_gap = obj_val - best_obj
        rows.append({
            "case": label,
            "solver": solver_name,
            "compile_ms": compile_ms,
            "exec_ms": exec_ms,
            "objective": obj_val,
            "obj_gap": obj_gap,
            "theta_err": theta_err,
            "nit": nit,
        })

print(f"{'solver':<24} {'case':<28} {'compile':>10} {'exec':>10} {'obj':>14} {'gap':>11} {'|Δθ|∞':>11} {'nit':>5}")
print("-" * 120)
for row in rows:
    compile_str = f"{row['compile_ms']:.0f}ms" if row["compile_ms"] is not None else "n/a"
    nit_str = str(row["nit"]) if row["nit"] is not None else "-"
    print(
        f"{row['solver']:<24} {row['case']:<28} "
        f"{compile_str:>10} {row['exec_ms']:>9.1f}ms "
        f"{row['objective']:>14.6e} {row['obj_gap']:>11.2e} "
        f"{row['theta_err']:>11.2e} {nit_str:>5}"
    )

# ── GH EM M-step timing ────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("GH EM M-STEP TIMING (10 stocks)")
print("=" * 88)

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
try:
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
    X10 = jnp.asarray(df.values[:, :10], dtype=np.float64)
    key = jax.random.PRNGKey(42)

    from normix import GeneralizedHyperbolic

    for solver_name, scan_length in [
        ("newton", 20),
        ("newton", 5),
        ("newton_analytical", 20),
        ("newton_analytical", 5),
        ("lbfgs", 20),
    ]:
        label = f"{solver_name}(scan={scan_length})" if solver_name != "lbfgs" else solver_name
        print(f"\n── {label} ──")
        model = GeneralizedHyperbolic._initialize(X10, key)

        # Warmup E-step
        exp = model.e_step(X10)
        _ = float(exp['E_Y'][0])

        # Warmup M-step (compile)
        t0 = time.perf_counter()
        m2 = model.m_step(X10, exp, solver=solver_name, scan_length=scan_length)
        _ = float(m2._joint.mu[0])
        t_compile = time.perf_counter() - t0
        print(f"  M-step compile: {t_compile:.1f}s")

        # Timed M-steps
        times = []
        for i in range(4):
            exp = model.e_step(X10)
            _ = float(exp['E_Y'][0])
            t0 = time.perf_counter()
            m2 = model.m_step(X10, exp, solver=solver_name, scan_length=scan_length)
            _ = float(m2._joint.mu[0])
            times.append(time.perf_counter() - t0)
            print(f"  M-step iter {i+1}: {times[-1]:.3f}s")

        print(f"  M-step avg: {np.mean(times):.3f}s")

except Exception as e:
    print(f"  Skipped GH timing: {e}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("SUMMARY")
print("=" * 88)
print(
    """
Notes:
  - 'scipy_wrapped' means NumPy/scipy optimizer calling JIT-compiled JAX objective/gradient:
        numpy theta -> jnp.asarray -> XLA-compiled JAX code -> float/ndarray -> scipy
    So it is NOT a pure callback path; only the optimizer loop is in scipy/Python.
  - 'legacy_numpy_scipy' is the pure NumPy/SciPy expectation-to-natural map from normix_numpy.
  - The Newton solvers are warm-started from the current theta0 supplied by the caller.
  - scan=5 vs scan=20 isolates the cost of the fixed unrolled Newton loop.
  - Objective values are compared on the same current JAX objective f(theta)=psi(theta)-theta·eta.
"""
)
