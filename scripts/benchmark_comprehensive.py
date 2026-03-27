"""
Comprehensive normix benchmark.

Measures compute time for:
  1. Bessel function: log_kv (JAX vs CPU, scalar vs batch)
  2. GIG η→θ solvers — all methods compared (new solve_bregman API)
  3. GIG expectation_params batch (E-step Bessel core)
  4-5. Full EM E-step and M-step: JAX vs CPU backend
  6. Full EM iterations: JAX vs CPU across distributions
  7. Device information
  8. GIG + EM end-to-end: all solver × backend combinations on real S&P 500 data

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/benchmark_comprehensive.py [--n-stocks N]

Results are printed as formatted tables. For GPU vs CPU comparison, ensure the
JAX CUDA runtime is available.

Performance notes:
- JAX Newton (method='newton') is slow on GPU for the scalar 3D GIG problem
  because lax.cond/lax.while_loop dispatch many small GPU kernels (~1ms each).
  backend='cpu' avoids this via scipy L-BFGS-B + scipy.kve.
- See docs/tech_notes/em_gpu_profiling.md for reference numbers.
"""

import os
import sys
import time
import argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

W = 100

def hdr(title: str) -> None:
    print(f"\n{'='*W}\n{title}\n{'='*W}", flush=True)

def row(label: str, *vals) -> None:
    label_w = 42
    print(f"  {label:<{label_w}}" + "".join(f"{v:>12}" for v in vals), flush=True)

def sep() -> None:
    print(f"  {'-'*(W-2)}", flush=True)

def timeit(fn, n_runs: int = 100, warmup: int = 5) -> float:
    """Return median elapsed time (seconds) over n_runs calls."""
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def jax_timeit(fn, n_runs: int = 100, warmup: int = 5) -> float:
    """Timeit for JAX functions (blocks until ready)."""
    def blocked():
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, dict):
            for v in result.values():
                if hasattr(v, "block_until_ready"):
                    v.block_until_ready()
    return timeit(blocked, n_runs=n_runs, warmup=warmup)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Bessel function benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_bessel() -> None:
    hdr("SECTION 1 — Bessel: log_kv(v, z)")

    from normix.utils.bessel import log_kv

    cases = [
        ("small-z  (v=0.5, z=0.001)",   0.5,  0.001),
        ("quadrature (v=1.0, z=2.0)",   1.0,  2.0),
        ("large-z  (v=1.0, z=30.0)",    1.0,  30.0),
        ("large-v  (v=50.0, z=5.0)",    50.0, 5.0),
        ("extreme  (v=100.0, z=200.0)", 100.0, 200.0),
    ]

    N_BATCH = 2552
    v_batch = jnp.full(N_BATCH, 0.5, dtype=jnp.float64)
    z_batch = jnp.linspace(0.5, 5.0, N_BATCH, dtype=jnp.float64)
    v_np = np.full(N_BATCH, 0.5)
    z_np = np.linspace(0.5, 5.0, N_BATCH)

    row("", "  JAX μs", "  CPU μs", "   ratio")
    sep()

    for label, v, z in cases:
        v_j = jnp.array(v); z_j = jnp.array(z)
        t_jax = jax_timeit(lambda: log_kv(v_j, z_j), n_runs=1000, warmup=100)
        t_cpu = timeit(lambda: log_kv(float(v), float(z), backend='cpu'), n_runs=1000, warmup=100)
        row(label, f"{t_jax*1e6:.1f}", f"{t_cpu*1e6:.1f}", f"{t_jax/t_cpu:.1f}×")

    sep()
    log_kv_jit = jax.jit(lambda v, z: jax.vmap(log_kv)(v, z))
    _ = log_kv_jit(v_batch, z_batch).block_until_ready()
    t_jax_batch = jax_timeit(lambda: log_kv_jit(v_batch, z_batch), n_runs=100, warmup=10)
    t_cpu_batch = timeit(lambda: log_kv(v_np, z_np, backend='cpu'), n_runs=100, warmup=10)
    row(f"batch N={N_BATCH} (ms)", f"{t_jax_batch*1e3:.1f}", f"{t_cpu_batch*1e3:.1f}",
        f"{t_jax_batch/t_cpu_batch:.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: GIG η→θ solvers — all methods
# ─────────────────────────────────────────────────────────────────────────────

def bench_gig_solvers() -> None:
    hdr("SECTION 2 — GIG η→θ solvers (new solve_bregman API)")

    from normix.distributions.generalized_inverse_gaussian import GeneralizedInverseGaussian as GIG

    test_cases = [
        ("symmetric  (p=0.5, a=b=1)",  GIG(p=jnp.array(0.5),  a=jnp.array(1.0),  b=jnp.array(1.0))),
        ("asymmetric (a>>b, p=0.5)",   GIG(p=jnp.array(0.5),  a=jnp.array(10.0), b=jnp.array(0.1))),
        ("asymmetric (a<<b, p=-1.0)",  GIG(p=jnp.array(-1.0), a=jnp.array(0.1),  b=jnp.array(10.0))),
        ("InvGauss limit (p=-½)",      GIG(p=jnp.array(-0.5), a=jnp.array(2.0),  b=jnp.array(1.0))),
    ]

    # ── Warm-start comparison ─────────────────────────────────────────────────
    # Solvers benchmarked: newton_analytical and cpu.
    # Excluded from timing (long XLA compile via log_kv lax.cond branches):
    #   newton     — jax.hessian through 4-branch lax.cond → minutes to compile
    #   lbfgs/bfgs — jaxopt gradient through log_kv lax.cond → minutes to compile
    # All solvers are functionally tested in tests/test_solvers.py.
    print(f"\n  Warm-start solvers (theta0 provided) — EM hot path", flush=True)
    print(f"  (newton/lbfgs/bfgs with jax backend omitted: long XLA compile via log_kv)", flush=True)
    solvers_warm = [
        ("newton_analytical", dict(backend="jax", method="newton", analytical_hessian=True, maxiter=20)),
        ("cpu   (scipy+kve)", dict(backend="cpu", method="lbfgs")),
    ]
    row("Case / solver", *[s[0] for s in solvers_warm], "ratio")
    sep()

    for label, gig in test_cases:
        eta   = gig.expectation_params()
        theta0 = gig.natural_params()
        times = []
        for sname, skw in solvers_warm:
            try:
                for _ in range(2):  # JIT warm-up
                    GIG.from_expectation(eta, theta0=theta0, **skw)
                t = timeit(lambda: GIG.from_expectation(eta, theta0=theta0, **skw),
                           n_runs=5, warmup=0)
                times.append(t)
            except Exception as e:
                times.append(None)
        t_vals = [f"{t*1e3:.1f}ms" if t is not None else "ERR" for t in times]
        if times[0] is not None and times[1] is not None:
            ratio = f"{times[0]/times[1]:.1f}×"
        else:
            ratio = "N/A"
        row(label, *t_vals, ratio)

    # ── Cold-start ───────────────────────────────────────────────────────────
    sep()
    print(f"\n  Cold-start (no theta0) — initial fit only", flush=True)
    row("Case", "cold-start ms", "", "")
    sep()
    for label, gig in test_cases[:2]:
        eta = gig.expectation_params()
        try:
            GIG.from_expectation(eta)           # compile / warm-up
            t = timeit(lambda: GIG.from_expectation(eta), n_runs=3, warmup=1)
            row(label, f"{t*1e3:.0f}", "(CPU multi-start)", "")
        except Exception as e:
            row(label, "ERR", str(e)[:40], "")

    # ── Cold-start (CPU multi-start for-loop) ────────────────────────────────
    # Note: vmap Newton multistart over GIG's Bessel (4-regime lax.cond) produces
    # very large XLA graphs — tested in tests/test_solvers.py, not benchmarked here.
    from normix.fitting.solvers import solve_bregman_multistart
    from normix.distributions.generalized_inverse_gaussian import (
        _log_partition_gig_cpu, _gig_cpu_grad, _initial_guesses,
    )

    gig0 = test_cases[0][1]
    eta0  = gig0.expectation_params()
    geom = jnp.sqrt(eta0[1] * eta0[2])
    eta_scaled = jnp.array([eta0[0] + 0.5*jnp.log(eta0[1]/eta0[2]), geom, geom])
    GIG_BOUNDS = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]

    theta0_list = _initial_guesses(eta_scaled)
    processed = [jnp.asarray(t, dtype=jnp.float64) for t in theta0_list]
    K = len(processed)

    # warm-up
    for _ in range(2):
        solve_bregman_multistart(
            _log_partition_gig_cpu, eta_scaled, processed,
            backend="cpu", method="lbfgs", bounds=GIG_BOUNDS, max_steps=300, tol=1e-9,
            grad_fn=_gig_cpu_grad,
        )

    t_loop = timeit(lambda: solve_bregman_multistart(
        _log_partition_gig_cpu, eta_scaled, processed,
        backend="cpu", method="lbfgs", bounds=GIG_BOUNDS, max_steps=300, tol=1e-9,
        grad_fn=_gig_cpu_grad,
    ), n_runs=5, warmup=1)

    sep()
    row(f"cold-start K={K} (CPU for-loop)", f"{t_loop*1e3:.0f}ms", "", "")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: GIG expectation_params batch
# ─────────────────────────────────────────────────────────────────────────────

def bench_gig_estep(N: int = 2552) -> None:
    hdr(f"SECTION 3 — GIG expectation_params (N={N})")

    from normix.distributions.generalized_inverse_gaussian import GIG

    rng = np.random.default_rng(42)
    p_vals = jnp.full(N, -0.5 - 2.0, dtype=jnp.float64)
    a_vals = jnp.array(rng.uniform(1.0, 5.0, N), dtype=jnp.float64)
    b_vals = jnp.array(rng.uniform(0.1, 3.0, N), dtype=jnp.float64)

    def jax_batch(p, a, b):
        def single(pi, ai, bi):
            return GIG(p=pi, a=ai, b=bi).expectation_params()
        return jax.vmap(single)(p, a, b)

    jax_batch_jit = jax.jit(jax_batch)
    _ = jax_batch_jit(p_vals, a_vals, b_vals)

    t_jax = jax_timeit(lambda: jax_batch_jit(p_vals, a_vals, b_vals), n_runs=50, warmup=5)

    try:
        t_cpu = timeit(
            lambda: GIG.expectation_params_batch(p_vals, a_vals, b_vals, backend='cpu'),
            n_runs=50, warmup=5)
        row(f"N={N} batch expectation_params", f"{t_jax*1e3:.1f} ms",
            f"{t_cpu*1e3:.1f} ms", f"{t_jax/t_cpu:.1f}×")
    except AttributeError:
        row(f"N={N} batch expectation_params", f"{t_jax*1e3:.1f} ms", "N/A", "")

    row("  per-sample", f"{t_jax/N*1e6:.1f} μs", "", "")


# ─────────────────────────────────────────────────────────────────────────────
# Sections 4 & 5: Full EM E-step and M-step
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n_stocks: int) -> np.ndarray:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
    return df.values.astype(np.float64)[:, :n_stocks]


def bench_em(n_stocks: int = 20) -> None:
    hdr(f"SECTION 4-5 — Full EM E-step and M-step (d={n_stocks})")

    from normix import (
        GeneralizedHyperbolic, VarianceGamma,
        NormalInverseGamma, NormalInverseGaussian,
    )

    try:
        X_raw = load_data(n_stocks)
    except FileNotFoundError:
        print(f"  [SKIP] data/sp500_returns.csv not found.")
        return

    X = jnp.asarray(X_raw, dtype=jnp.float64)
    n, d = X.shape
    print(f"  Data: {n} obs × {d} stocks", flush=True)

    sigma_emp = jnp.eye(d)

    def make_dist(cls, extra):
        return cls.from_classical(mu=jnp.zeros(d), gamma=jnp.zeros(d),
                                   sigma=sigma_emp, **extra)

    dists = [
        ("VG",    VarianceGamma,        {"alpha": 2.0, "beta": 1.0}),
        ("NIG",   NormalInverseGaussian,{"mu_ig": 1.0, "lam": 1.0}),
        ("NInvG", NormalInverseGamma,   {"alpha": 2.0, "beta": 1.0}),
        ("GH",    GeneralizedHyperbolic,{"p": -0.5, "a": 2.0, "b": 1.0}),
    ]

    row("Distribution", "E jax ms", "E cpu ms", "E speedup",
        "M jax ms", "M cpu ms", "M speedup")
    sep()

    for name, cls, extra in dists:
        try:
            model = make_dist(cls, extra)
            t_e_jax = jax_timeit(lambda: model.e_step(X, backend='jax'), n_runs=10, warmup=2)
            try:
                t_e_cpu = timeit(lambda: model.e_step(X, backend='cpu'), n_runs=10, warmup=2)
                e_ratio = f"{t_e_jax/t_e_cpu:.1f}×"
            except Exception:
                t_e_cpu = float('nan'); e_ratio = "N/A"

            expectations = model.e_step(X, backend='jax')
            t_m_jax = timeit(lambda: model.m_step(X, expectations), n_runs=5, warmup=2)
            try:
                if name == "GH":
                    t_m_cpu = timeit(
                        lambda: model.m_step(X, expectations, solver='cpu'),
                        n_runs=10, warmup=3)
                else:
                    t_m_cpu = t_m_jax
                m_ratio = f"{t_m_jax/t_m_cpu:.1f}×" if t_m_cpu > 0 else "N/A"
            except Exception:
                t_m_cpu = float('nan'); m_ratio = "N/A"

            row(name,
                f"{t_e_jax*1e3:.0f}ms",
                f"{t_e_cpu*1e3:.0f}ms" if not np.isnan(t_e_cpu) else "N/A",
                e_ratio,
                f"{t_m_jax*1e3:.0f}ms",
                f"{t_m_cpu*1e3:.0f}ms" if not np.isnan(t_m_cpu) else "N/A",
                m_ratio)
        except Exception as ex:
            row(name, "ERR", "", "", "", "", f"({ex})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Full EM iterations across distributions and backends
# ─────────────────────────────────────────────────────────────────────────────

def bench_em_full(n_stocks: int = 20, n_iter: int = 5) -> None:
    hdr(f"SECTION 6 — Full EM iterations (d={n_stocks}, {n_iter} iterations)")

    from normix import (
        GeneralizedHyperbolic, VarianceGamma,
        NormalInverseGamma, NormalInverseGaussian,
    )

    try:
        X_raw = load_data(n_stocks)
    except FileNotFoundError:
        print(f"  [SKIP] data/sp500_returns.csv not found.")
        return

    X = jnp.asarray(X_raw, dtype=jnp.float64)
    n, d = X.shape
    sigma_emp = jnp.eye(d)

    dists = [
        ("VG",    VarianceGamma,        {"alpha": 2.0, "beta": 1.0}),
        ("NIG",   NormalInverseGaussian,{"mu_ig": 1.0, "lam": 1.0}),
        ("NInvG", NormalInverseGamma,   {"alpha": 2.0, "beta": 1.0}),
        ("GH",    GeneralizedHyperbolic,{"p": -0.5, "a": 2.0, "b": 1.0}),
    ]

    def run_em(model, X, n_iter, e_backend, m_solver='newton'):
        exp = model.e_step(X, backend=e_backend)
        try:
            model.m_step(X, exp, solver=m_solver)
        except TypeError:
            model.m_step(X, exp)
        e_total = 0.0; m_total = 0.0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            exp = model.e_step(X, backend=e_backend)
            if hasattr(exp.get('E_Y', None), 'block_until_ready'):
                exp['E_Y'].block_until_ready()
            e_total += time.perf_counter() - t0
            t0 = time.perf_counter()
            try:
                model = model.m_step(X, exp, solver=m_solver)
            except TypeError:
                model = model.m_step(X, exp)
            m_total += time.perf_counter() - t0
        return e_total, m_total

    row("Dist", "E jax", "M jax", "tot jax", "E cpu", "M cpu", "tot cpu", "speedup")
    sep()

    for name, cls, extra in dists:
        try:
            model = cls.from_classical(
                mu=jnp.zeros(d), gamma=jnp.zeros(d), sigma=sigma_emp, **extra)
            e_jax, m_jax = run_em(model, X, n_iter, 'jax', 'newton')
            e_cpu, m_cpu = run_em(model, X, n_iter, 'cpu', 'cpu')
            t_jax = e_jax + m_jax; t_cpu = e_cpu + m_cpu
            row(name,
                f"{e_jax/n_iter:.2f}s", f"{m_jax/n_iter:.2f}s", f"{t_jax/n_iter:.2f}s",
                f"{e_cpu/n_iter:.2f}s", f"{m_cpu/n_iter:.2f}s", f"{t_cpu/n_iter:.2f}s",
                f"{t_jax/t_cpu:.1f}×")
        except Exception as ex:
            row(name, "ERR", "", "", "", "", "", f"({ex})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Device information
# ─────────────────────────────────────────────────────────────────────────────

def bench_device_info() -> None:
    hdr("SECTION 7 — Device Information")
    devices = jax.devices()
    print(f"  JAX version: {jax.__version__}", flush=True)
    print(f"  Available devices:", flush=True)
    for d in devices:
        print(f"    {d}", flush=True)
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        print("  [No GPU detected — CPU-only benchmarks]", flush=True)
    else:
        print(f"  GPU: {gpu_devices[0]}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: GIG + EM end-to-end — all solver × backend combinations
# ─────────────────────────────────────────────────────────────────────────────

def bench_em_solver_comparison(n_stocks: int = 20, n_iter: int = 10) -> None:
    hdr(f"SECTION 8 — GH/GIG EM solver × backend comparison  (d={n_stocks}, {n_iter} iter)")

    from normix import GeneralizedHyperbolic
    from normix.fitting.em import BatchEMFitter

    try:
        X_raw = load_data(n_stocks)
    except FileNotFoundError:
        print(f"  [SKIP] data/sp500_returns.csv not found.")
        return

    X = jnp.asarray(X_raw, dtype=jnp.float64)
    n, d = X.shape
    sigma_emp = jnp.cov(X_raw.T) + 1e-4 * np.eye(d)
    sigma_emp = jnp.asarray(sigma_emp, dtype=jnp.float64)
    print(f"  Data: {n} obs × {d} stocks", flush=True)

    def make_gh():
        return GeneralizedHyperbolic.from_classical(
            mu=jnp.zeros(d), gamma=jnp.zeros(d),
            sigma=sigma_emp, p=-0.5, a=2.0, b=1.0,
        )

    # On GPU, JAX Newton/lbfgs/bfgs M-steps for the scalar 3D GIG problem are
    # very slow (~seconds per call) due to many small GPU kernel launches from
    # lax.scan/lax.cond dispatch.  The cpu solver (scipy + scipy.kve) avoids
    # this completely.  We still include newton_analytical as a reference.
    # See docs/tech_notes/em_gpu_profiling.md for full analysis.
    configs = [
        # (label,                    e_backend, m_solver)
        ("jax E + cpu M  [FASTEST]", "jax",     "cpu"),
        ("cpu E + cpu M  [REF]",     "cpu",     "cpu"),
        ("jax E + newton_ana M",     "jax",     "newton_analytical"),
        # lbfgs/bfgs/newton jax M excluded: very slow on GPU for scalar GIG
        # (lax.while_loop + many small kernel dispatches per step)
    ]

    def run_timed(model, e_backend, m_solver, n_iter):
        """Run n_iter EM steps; return (e_time, m_time, final_ll)."""
        # warm-up (1 step)
        exp = model.e_step(X, backend=e_backend)
        try:
            model.m_step(X, exp, solver=m_solver)
        except TypeError:
            model.m_step(X, exp)

        e_total = 0.0; m_total = 0.0
        for _ in range(n_iter):
            t0 = time.perf_counter()
            exp = model.e_step(X, backend=e_backend)
            e_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            try:
                model = model.m_step(X, exp, solver=m_solver)
            except TypeError:
                model = model.m_step(X, exp)
            m_total += time.perf_counter() - t0

        # final log-likelihood
        ll = float(jnp.mean(jax.vmap(model.log_prob)(X)))
        return e_total, m_total, ll

    print(f"\n  Per-iteration times and final log-likelihood after {n_iter} EM steps:", flush=True)
    row("Config", "E/iter", "M/iter", "Total/iter", "Final LL")
    sep()

    reference_ll = None
    for label, e_bk, m_sv in configs:
        try:
            model = make_gh()
            e_t, m_t, ll = run_timed(model, e_bk, m_sv, n_iter)
            if reference_ll is None:
                reference_ll = ll
            delta_ll = f"Δ{ll - reference_ll:+.4f}" if reference_ll is not None else ""
            row(label,
                f"{e_t/n_iter*1e3:.0f}ms",
                f"{m_t/n_iter*1e3:.0f}ms",
                f"{(e_t+m_t)/n_iter*1e3:.0f}ms",
                f"{ll:.4f} {delta_ll}")
        except Exception as ex:
            row(label, "ERR", "", "", f"({type(ex).__name__}: {str(ex)[:40]})")

    # ── GPU vs CPU total if GPU available ────────────────────────────────────
    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if gpu_devices:
        sep()
        print(f"\n  GPU vs CPU device comparison:", flush=True)
        row("Backend", "Device", "Total/iter", "vs CPU")
        sep()
        cpu_time = None
        for device_label, e_bk, m_sv in [("CPU", "cpu", "cpu"), ("GPU jax", "jax", "cpu")]:
            try:
                model = make_gh()
                e_t, m_t, ll = run_timed(model, e_bk, m_sv, n_iter)
                total = (e_t + m_t) / n_iter
                if device_label == "CPU":
                    cpu_time = total
                ratio = f"{total/cpu_time:.2f}×" if cpu_time else "—"
                row(device_label, str(jax.devices()[0]), f"{total*1e3:.0f}ms", ratio)
            except Exception as ex:
                row(device_label, "N/A", "ERR", str(ex)[:30])

    # ── M-step solver breakdown for GIG specifically ─────────────────────────
    sep()
    print(f"\n  GIG M-step solver timing only (isolating η→θ solve, {n_iter} iters):", flush=True)
    row("Solver", "M/iter", "grad_norm check", "")
    sep()
    from normix import GIG
    gig = GIG(p=jnp.array(-0.5), a=jnp.array(2.0), b=jnp.array(1.0))
    eta = gig.expectation_params()
    theta0 = gig.natural_params()

    # newton/lbfgs/bfgs (JAX) excluded: long XLA compile through log_kv lax.cond.
    # All are functionally tested in tests/test_solvers.py.
    mstep_solvers = [
        ("newton_analytical (JAX 7-K)", dict(backend="jax", method="newton", analytical_hessian=True, maxiter=20)),
        ("cpu    (scipy+scipy.kve)",     dict(backend="cpu", method="lbfgs")),
    ]
    # warm-up
    for _, kw in mstep_solvers:
        for _ in range(3):
            try:
                GIG.from_expectation(eta, theta0=theta0, **kw)
            except Exception:
                pass

    for sname, kw in mstep_solvers:
        try:
            t = timeit(lambda: GIG.from_expectation(eta, theta0=theta0, **kw),
                       n_runs=20, warmup=2)
            result = GIG.from_expectation(eta, theta0=theta0, **kw)
            # Verify round-trip
            eta_check = result.expectation_params()
            err = float(jnp.max(jnp.abs(eta_check - eta)))
            row(sname, f"{t*1e3:.2f}ms", f"err={err:.2e}", "")
        except Exception as ex:
            row(sname, "ERR", str(ex)[:40], "")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="normix comprehensive benchmark")
    parser.add_argument("--n-stocks", type=int, default=20,
                        help="Number of stocks for EM benchmarks (default: 20)")
    parser.add_argument("--n-iter", type=int, default=5,
                        help="EM iterations for full pipeline benchmark (default: 5)")
    parser.add_argument("--sections", type=str, default="all",
                        help="Comma-separated sections to run: 1,2,...,8 or 'all'")
    args = parser.parse_args()

    sections = set(range(1, 9)) if args.sections == "all" else {int(s) for s in args.sections.split(",")}

    print(f"\nnormix Comprehensive Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)

    if 7 in sections:
        bench_device_info()
    if 1 in sections:
        bench_bessel()
    if 2 in sections:
        bench_gig_solvers()
    if 3 in sections:
        bench_gig_estep()
    if 4 in sections or 5 in sections:
        bench_em(n_stocks=args.n_stocks)
    if 6 in sections:
        bench_em_full(n_stocks=args.n_stocks, n_iter=args.n_iter)
    if 8 in sections:
        bench_em_solver_comparison(n_stocks=args.n_stocks, n_iter=args.n_iter)

    print(f"\n{'='*W}", flush=True)
    print("Benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
