"""
Comprehensive normix benchmark.

Measures compute time for:
  1. Bessel function: log_kv (JAX vs CPU, scalar vs batch)
  2. GIG optimization (eta->theta): JAX LBFGS-B, scipy LBFGS-B, JAX Newton variants
  3. GIG E-step: expectation_params (JAX vs CPU backend)
  4. Full EM E-step (quad forms + Bessel): JAX vs CPU backend
  5. Full EM M-step: JAX solver vs CPU solver
  6. Full EM iteration (E+M): CPU vs JAX, across distributions
  7. GPU vs CPU device for JAX backend

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/benchmark_comprehensive.py [--n-stocks N]

Results are printed as formatted tables. For profiling GPU vs CPU, make sure the
JAX CUDA runtime is available.
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

W = 88

def hdr(title: str) -> None:
    print(f"\n{'='*W}\n{title}\n{'='*W}", flush=True)

def row(label: str, *vals) -> None:
    label_w = 38
    print(f"  {label:<{label_w}}" + "".join(f"{v:>12}" for v in vals), flush=True)

def sep() -> None:
    print(f"  {'-'*(W-2)}", flush=True)

def timeit(fn, n_runs: int = 100, warmup: int = 5) -> float:
    """Return median elapsed time (seconds) over n_runs calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
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

    # Test scalar values across regimes
    cases = [
        ("small-z  (v=0.5, z=0.001)",  0.5,  0.001),
        ("quadrature (v=1.0, z=2.0)",  1.0,  2.0),
        ("large-z  (v=1.0, z=30.0)",   1.0,  30.0),
        ("large-v  (v=50.0, z=5.0)",   50.0, 5.0),
        ("extreme  (v=100.0, z=200.0)",100.0, 200.0),
    ]

    # Batch: N log_kv calls
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

    # Batch benchmark
    log_kv_jit = jax.jit(lambda v, z: jax.vmap(log_kv)(v, z))
    _ = log_kv_jit(v_batch, z_batch).block_until_ready()  # warm up

    t_jax_batch = jax_timeit(lambda: log_kv_jit(v_batch, z_batch), n_runs=100, warmup=10)
    t_cpu_batch = timeit(lambda: log_kv(v_np, z_np, backend='cpu'), n_runs=100, warmup=10)
    row(f"batch N={N_BATCH} (ms)", f"{t_jax_batch*1e3:.1f}", f"{t_cpu_batch*1e3:.1f}",
        f"{t_jax_batch/t_cpu_batch:.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: GIG eta->theta optimization
# ─────────────────────────────────────────────────────────────────────────────

def bench_gig_solvers() -> None:
    hdr("SECTION 2 — GIG eta→theta solvers")

    from normix.distributions.generalized_inverse_gaussian import GIG

    test_cases = [
        ("symmetric GIG",        GIG(p=jnp.array(0.5),  a=jnp.array(1.0), b=jnp.array(1.0))),
        ("asymmetric (a>>b)",    GIG(p=jnp.array(0.5),  a=jnp.array(10.0), b=jnp.array(0.1))),
        ("asymmetric (a<<b)",    GIG(p=jnp.array(-1.0), a=jnp.array(0.1),  b=jnp.array(10.0))),
        ("GIG ~ InvGauss (p=-½)",GIG(p=jnp.array(-0.5), a=jnp.array(2.0),  b=jnp.array(1.0))),
    ]

    row("Case / solver", " η→θ JAX ms", " η→θ CPU ms", "     ratio")
    sep()

    for label, gig in test_cases:
        eta = jig_eta = gig.expectation_params()

        # JAX solver (jaxopt LBFGS-B)
        try:
            t_jax = jax_timeit(lambda: GIG.from_expectation(eta), n_runs=50, warmup=5)
            t_jax_ms = f"{t_jax*1e3:.1f}"
        except Exception as e:
            t_jax_ms = "ERR"
            t_jax = float('nan')

        # CPU solver (scipy)
        try:
            t_cpu = timeit(lambda: GIG.from_expectation(eta, solver='cpu'), n_runs=50, warmup=5)
            t_cpu_ms = f"{t_cpu*1e3:.1f}"
        except Exception:
            try:
                t_cpu = timeit(lambda: GIG.from_expectation(eta), n_runs=50, warmup=5)
                t_cpu_ms = f"{t_cpu*1e3:.1f} (fallback)"
            except Exception:
                t_cpu_ms = "N/A"
                t_cpu = float('nan')

        ratio = f"{t_jax/t_cpu:.1f}×" if not (np.isnan(t_jax) or np.isnan(t_cpu)) else "N/A"
        row(label, t_jax_ms, t_cpu_ms, ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: GIG expectation_params (E-step Bessel core)
# ─────────────────────────────────────────────────────────────────────────────

def bench_gig_estep(N: int = 2552) -> None:
    hdr(f"SECTION 3 — GIG expectation_params (N={N})")

    from normix.distributions.generalized_inverse_gaussian import GIG

    # Batch of GIG params typical in GH E-step
    rng = np.random.default_rng(42)
    p_vals = jnp.full(N, -0.5 - 2.0, dtype=jnp.float64)  # p_post for NIG, d=4
    a_vals = jnp.array(rng.uniform(1.0, 5.0, N), dtype=jnp.float64)
    b_vals = jnp.array(rng.uniform(0.1, 3.0, N), dtype=jnp.float64)

    # JAX: vmap over individual GIG instances
    def jax_batch(p, a, b):
        def single(pi, ai, bi):
            return GIG(p=pi, a=ai, b=bi).expectation_params()
        return jax.vmap(single)(p, a, b)

    jax_batch_jit = jax.jit(jax_batch)
    _ = jax_batch_jit(p_vals, a_vals, b_vals)  # compile

    t_jax = jax_timeit(lambda: jax_batch_jit(p_vals, a_vals, b_vals), n_runs=50, warmup=5)

    # CPU: vectorized scipy
    try:
        t_cpu = timeit(
            lambda: GIG.expectation_params_batch(p_vals, a_vals, b_vals, backend='cpu'),
            n_runs=50, warmup=5)
        row(f"N={N} batch expectation_params", f"{t_jax*1e3:.1f} ms",
            f"{t_cpu*1e3:.1f} ms", f"{t_jax/t_cpu:.1f}×")
    except AttributeError:
        row(f"N={N} batch expectation_params", f"{t_jax*1e3:.1f} ms",
            "N/A", "")

    row("  per-sample", f"{t_jax/N*1e6:.1f} μs", "", "")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 & 5: Full EM E-step and M-step
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n_stocks: int) -> tuple:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
    X = df.values.astype(np.float64)[:, :n_stocks]
    return X

def bench_em(n_stocks: int = 20) -> None:
    hdr(f"SECTION 4-5 — Full EM E-step and M-step (d={n_stocks})")

    from normix import (
        GeneralizedHyperbolic,
        VarianceGamma,
        NormalInverseGamma,
        NormalInverseGaussian,
    )

    try:
        X_raw = load_data(n_stocks)
    except FileNotFoundError:
        print(f"  [SKIP] data/sp500_returns.csv not found. Run scripts/download_sp500_data.py first.")
        return

    X = jnp.asarray(X_raw, dtype=jnp.float64)
    n, d = X.shape
    print(f"  Data: {n} obs × {d} stocks", flush=True)

    key = jax.random.PRNGKey(0)
    sigma_emp = jnp.eye(d)

    def make_dist(cls, extra):
        return cls.from_classical(
            mu=jnp.zeros(d), gamma=jnp.zeros(d), sigma=sigma_emp, **extra
        )

    dists = [
        ("VG",      VarianceGamma,        {"alpha": 2.0, "beta": 1.0}),
        ("NIG",     NormalInverseGaussian,{"mu_ig": 1.0, "lam": 1.0}),
        ("NInvG",   NormalInverseGamma,   {"alpha": 2.0, "beta": 1.0}),
        ("GH",      GeneralizedHyperbolic,{"p": -0.5, "a": 2.0, "b": 1.0}),
    ]

    row("Distribution", "E-step JAX", "E-step CPU", " speedup", "M-step JAX", " speedup")
    sep()

    for name, cls, extra in dists:
        try:
            model = make_dist(cls, extra)

            # E-step JAX
            t_e_jax = jax_timeit(lambda: model.e_step(X, backend='jax'), n_runs=10, warmup=2)

            # E-step CPU
            try:
                t_e_cpu = timeit(lambda: model.e_step(X, backend='cpu'), n_runs=10, warmup=2)
                e_ratio = f"{t_e_jax/t_e_cpu:.1f}×"
            except Exception:
                t_e_cpu = float('nan'); e_ratio = "N/A"

            # M-step
            expectations = model.e_step(X, backend='jax')
            t_m = jax_timeit(lambda: model.m_step(X, expectations), n_runs=10, warmup=2)

            row(name,
                f"{t_e_jax*1e3:.0f}ms",
                f"{t_e_cpu*1e3:.0f}ms" if not np.isnan(t_e_cpu) else "N/A",
                e_ratio,
                f"{t_m*1e3:.0f}ms",
                "",
            )
        except Exception as ex:
            row(name, "ERR", "", "", "", f"({ex})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Full EM iterations across distributions and backends
# ─────────────────────────────────────────────────────────────────────────────

def bench_em_full(n_stocks: int = 20, n_iter: int = 5) -> None:
    hdr(f"SECTION 6 — Full EM iterations (d={n_stocks}, {n_iter} iterations)")

    from normix import (
        GeneralizedHyperbolic,
        VarianceGamma,
        NormalInverseGamma,
        NormalInverseGaussian,
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

    def run_em(model, X, n_iter, e_backend, m_solver=None):
        """Run n_iter EM steps, return total wall-clock time (excluding first warmup)."""
        # warmup iteration
        exp = model.e_step(X, backend=e_backend)
        model2 = model.m_step(X, exp)
        # timed iterations
        t0 = time.perf_counter()
        for _ in range(n_iter):
            exp = model.e_step(X, backend=e_backend)
            if hasattr(exp['E_Y'], 'block_until_ready'):
                exp['E_Y'].block_until_ready()
            model = model.m_step(X, exp)
        return time.perf_counter() - t0

    row("Distribution", "  JAX total", "  CPU total", "  speedup", "  JAX/iter", "  CPU/iter")
    sep()

    for name, cls, extra in dists:
        try:
            model = cls.from_classical(
                mu=jnp.zeros(d), gamma=jnp.zeros(d), sigma=sigma_emp, **extra
            )

            t_jax = run_em(model, X, n_iter, e_backend='jax')
            t_cpu = run_em(model, X, n_iter, e_backend='cpu')

            row(name,
                f"{t_jax:.2f}s",
                f"{t_cpu:.2f}s",
                f"{t_jax/t_cpu:.1f}×",
                f"{t_jax/n_iter*1e3:.0f}ms",
                f"{t_cpu/n_iter*1e3:.0f}ms",
            )
        except Exception as ex:
            row(name, "ERR", "", "", "", f"({ex})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: GPU vs CPU device detection
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
        print("  [No GPU detected — only CPU benchmarks available]", flush=True)
    else:
        print(f"  GPU detected: {gpu_devices[0]}", flush=True)
        print("  Note: Run with CUDA visible to benchmark GPU vs CPU.", flush=True)


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
                        help="Comma-separated sections to run: 1,2,3,4,5,6,7 or 'all'")
    args = parser.parse_args()

    sections = set(range(1, 8)) if args.sections == "all" else {int(s) for s in args.sections.split(",")}

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

    print(f"\n{'='*W}", flush=True)
    print("Benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
