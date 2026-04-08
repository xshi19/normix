"""
GIG η→θ solver benchmark — warm-start and cold-start.

Compares solver backends and methods for the GIG from_expectation call
that dominates the M-step of GeneralizedHyperbolic EM.

Usage:
    uv run python benchmarks/bench_gig_solvers.py
    uv run python benchmarks/bench_gig_solvers.py --save
"""

import os
import sys
import argparse

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.utils import timeit, save_result, fmt_time, hdr, sep

from normix.distributions.generalized_inverse_gaussian import GeneralizedInverseGaussian as GIG


# ---------------------------------------------------------------------------
# Test cases (GIG instances with known parameters)
# ---------------------------------------------------------------------------

TEST_CASES = [
    ("symmetric  (p=0.5, a=b=1)",
     GIG(p=jnp.array(0.5), a=jnp.array(1.0), b=jnp.array(1.0))),
    ("asymmetric (a>>b, p=0.5)",
     GIG(p=jnp.array(0.5), a=jnp.array(10.0), b=jnp.array(0.1))),
    ("asymmetric (a<<b, p=-1.0)",
     GIG(p=jnp.array(-1.0), a=jnp.array(0.1), b=jnp.array(10.0))),
    ("InvGauss limit (p=-½)",
     GIG(p=jnp.array(-0.5), a=jnp.array(2.0), b=jnp.array(1.0))),
]


# ---------------------------------------------------------------------------
# Warm-start solvers (EM hot path: theta0 provided)
# ---------------------------------------------------------------------------

WARM_SOLVERS = [
    ("jax/newton",   dict(backend="jax", method="newton", maxiter=20)),
    ("cpu/lbfgs",    dict(backend="cpu", method="lbfgs")),
    ("cpu/newton",   dict(backend="cpu", method="newton")),
]


def bench_warm_start() -> list[dict]:
    """Benchmark warm-start GIG.from_expectation (theta0 provided)."""
    results = []
    for label, gig in TEST_CASES:
        eta = gig.expectation_params()
        theta0 = gig.natural_params()
        row = {"case": label, "solvers": {}}

        for sname, skw in WARM_SOLVERS:
            try:
                for _ in range(3):
                    GIG.from_expectation(eta, theta0=theta0, **skw)
                t = timeit(
                    lambda: GIG.from_expectation(eta, theta0=theta0, **skw),
                    n_runs=10, warmup=2)
                result = GIG.from_expectation(eta, theta0=theta0, **skw)
                eta_check = result.expectation_params()
                err = float(jnp.max(jnp.abs(eta_check - eta)))
                row["solvers"][sname] = {"time_ms": t * 1e3, "roundtrip_err": err}
            except Exception as e:
                row["solvers"][sname] = {"time_ms": None, "error": str(e)[:50]}

        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Cold-start (no theta0 — multi-start CPU solver)
# ---------------------------------------------------------------------------

def bench_cold_start() -> list[dict]:
    """Benchmark cold-start GIG.from_expectation (no theta0)."""
    results = []
    for label, gig in TEST_CASES[:2]:
        eta = gig.expectation_params()
        try:
            GIG.from_expectation(eta)  # warm-up / compile
            t = timeit(lambda: GIG.from_expectation(eta), n_runs=3, warmup=1)
            results.append({"case": label, "time_ms": t * 1e3})
        except Exception as e:
            results.append({"case": label, "time_ms": None, "error": str(e)[:50]})
    return results


# ---------------------------------------------------------------------------
# GIG expectation_params batch (E-step core)
# ---------------------------------------------------------------------------

def bench_expectation_batch(n_batch: int = 2552) -> dict:
    """Benchmark batched GIG expectation_params: JAX vmap vs CPU."""
    rng = np.random.default_rng(42)
    p_vals = jnp.full(n_batch, -0.5 - 2.0, dtype=jnp.float64)
    a_vals = jnp.array(rng.uniform(1.0, 5.0, n_batch), dtype=jnp.float64)
    b_vals = jnp.array(rng.uniform(0.1, 3.0, n_batch), dtype=jnp.float64)

    from benchmarks.utils import jax_timeit

    jax_batch = jax.jit(lambda p, a, b: jax.vmap(
        lambda pi, ai, bi: GIG(p=pi, a=ai, b=bi).expectation_params()
    )(p, a, b))
    _ = jax_batch(p_vals, a_vals, b_vals)

    t_jax = jax_timeit(lambda: jax_batch(p_vals, a_vals, b_vals),
                       n_runs=50, warmup=5)
    try:
        t_cpu = timeit(
            lambda: GIG.expectation_params_batch(p_vals, a_vals, b_vals, backend='cpu'),
            n_runs=50, warmup=5)
        return {"n": n_batch, "jax_ms": t_jax * 1e3, "cpu_ms": t_cpu * 1e3,
                "ratio": t_jax / t_cpu if t_cpu > 0 else float('inf')}
    except (AttributeError, Exception):
        return {"n": n_batch, "jax_ms": t_jax * 1e3, "cpu_ms": None, "ratio": None}


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_results(warm_results, cold_results, batch_result):
    W = 100
    hdr("GIG η→θ Solvers — Warm Start (theta0 provided)", W)

    solver_names = [s[0] for s in WARM_SOLVERS]
    header_parts = f"  {'Case':<35}"
    for sn in solver_names:
        header_parts += f" {sn:>14}"
    print(header_parts)
    sep(W)

    for row in warm_results:
        parts = f"  {row['case']:<35}"
        for sn in solver_names:
            info = row["solvers"].get(sn, {})
            t = info.get("time_ms")
            if t is not None:
                parts += f" {fmt_time(t / 1e3):>14}"
            else:
                parts += f" {'ERR':>14}"
        print(parts)
    print(f"{'=' * W}")

    hdr("GIG η→θ Solvers — Cold Start (no theta0)", W)
    for r in cold_results:
        t = r.get("time_ms")
        t_str = fmt_time(t / 1e3) if t is not None else "ERR"
        print(f"  {r['case']:<35} {t_str:>14}")
    print(f"{'=' * W}")

    hdr("GIG expectation_params Batch", W)
    b = batch_result
    jax_str = f"{b['jax_ms']:.1f} ms"
    cpu_str = f"{b['cpu_ms']:.1f} ms" if b.get('cpu_ms') is not None else "N/A"
    ratio_str = f"{b['ratio']:.1f}×" if b.get('ratio') is not None else "N/A"
    print(f"  N={b['n']}: JAX={jax_str}, CPU={cpu_str}, ratio={ratio_str}")
    print(f"{'=' * W}")


def main():
    parser = argparse.ArgumentParser(description="GIG solver benchmark")
    parser.add_argument("--save", action="store_true",
                        help="Save results to benchmarks/results/")
    args = parser.parse_args()

    print(f"\nnormix GIG Solver Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    print("\nRunning warm-start benchmarks...", flush=True)
    warm_results = bench_warm_start()
    print("Running cold-start benchmarks...", flush=True)
    cold_results = bench_cold_start()
    print("Running expectation batch benchmark...", flush=True)
    batch_result = bench_expectation_batch()

    print_results(warm_results, cold_results, batch_result)

    if args.save:
        data = {
            "benchmark": "gig_solvers",
            "warm_start": warm_results,
            "cold_start": cold_results,
            "expectation_batch": batch_result,
        }
        path = save_result("gig_solvers", data)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
