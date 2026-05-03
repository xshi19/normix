"""
JIT-cache benchmark for the in-loop Newton solvers.

Phase-4 measurement (see ``docs/investigations/test_suite_performance_2026-04-28.md``).
Compares first-call (compile + run) vs cached-call (run only) latency for the
Newton solvers that drive the M-step. We focus on the JAX/JAX hot path because
that is where a fresh Python closure on every call forced JAX to re-trace the
same kernel and dominated GH M-step time in the previous run.

Cases
-----
GIG η → θ
    1. ``GIG.from_expectation(backend='jax', method='newton')``
       (now routes through a stable jitted kernel — see
       ``_gig_jax_newton_jit``).
    2. The same warm-started CPU/LBFGS path for reference.

Gamma η → θ
    3. ``Gamma.from_expectation`` driven by ``_newton_digamma`` (now decorated
       with ``@jax.jit``).
    4. CPU variant (``_newton_digamma_cpu``).

Usage
-----
    uv run python benchmarks/bench_jit_solvers.py
    uv run python benchmarks/bench_jit_solvers.py --save
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.utils import save_result, fmt_time, hdr, sep

from normix.distributions.generalized_inverse_gaussian import GeneralizedInverseGaussian as GIG
from normix.distributions.gamma import (
    Gamma, _newton_digamma, _newton_digamma_cpu,
)


# ---------------------------------------------------------------------------
# Timing helpers — separate first-call from cached-call latency
# ---------------------------------------------------------------------------

def _block(result):
    """Block JAX async dispatch so wall-clock includes device transfer."""
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, (tuple, list)):
        for v in result:
            _block(v)


def measure_first_then_cached(make_call, n_cached: int = 30):
    """Return ``(first_call_s, cached_call_median_s)``.

    ``make_call`` is a zero-arg factory returning the call to time.
    The first invocation pays compilation cost; subsequent invocations
    should hit the JAX cache.
    """
    t0 = time.perf_counter()
    res = make_call()
    _block(res)
    t_first = time.perf_counter() - t0

    times = []
    for _ in range(n_cached):
        t0 = time.perf_counter()
        res = make_call()
        _block(res)
        times.append(time.perf_counter() - t0)
    return t_first, float(np.median(times))


# ---------------------------------------------------------------------------
# GIG cases
# ---------------------------------------------------------------------------

GIG_CASES = [
    ("symmetric  (p=0.5, a=b=1)",
     GIG(p=jnp.array(0.5), a=jnp.array(1.0), b=jnp.array(1.0))),
    ("asymmetric (a>>b, p=0.5)",
     GIG(p=jnp.array(0.5), a=jnp.array(10.0), b=jnp.array(0.1))),
    ("asymmetric (a<<b, p=-1.0)",
     GIG(p=jnp.array(-1.0), a=jnp.array(0.1), b=jnp.array(10.0))),
    ("InvGauss limit (p=-½)",
     GIG(p=jnp.array(-0.5), a=jnp.array(2.0), b=jnp.array(1.0))),
]


def bench_gig() -> list[dict]:
    rows = []
    for label, gig in GIG_CASES:
        eta = gig.expectation_params()
        theta0 = gig.natural_params()

        def call_jax_newton():
            return GIG.from_expectation(
                eta, theta0=theta0, backend="jax", method="newton", maxiter=20)

        def call_cpu_lbfgs():
            return GIG.from_expectation(
                eta, theta0=theta0, backend="cpu", method="lbfgs", maxiter=200)

        first_jn, cached_jn = measure_first_then_cached(call_jax_newton)
        first_cl, cached_cl = measure_first_then_cached(call_cpu_lbfgs)

        # Roundtrip check after caching is warm
        out = call_jax_newton()
        eta_check = out.expectation_params()
        err = float(jnp.max(jnp.abs(eta_check - eta)))

        rows.append({
            "case": label,
            "jax_newton_first_ms": first_jn * 1e3,
            "jax_newton_cached_ms": cached_jn * 1e3,
            "cpu_lbfgs_first_ms":  first_cl * 1e3,
            "cpu_lbfgs_cached_ms": cached_cl * 1e3,
            "roundtrip_err": err,
        })
    return rows


# ---------------------------------------------------------------------------
# Gamma η → θ via _newton_digamma (used by Gamma + InverseGamma + VG/NInvG)
# ---------------------------------------------------------------------------

GAMMA_TARGETS = [
    ("alpha=1",   0.0  - jnp.log(jnp.array(1.0))),
    ("alpha=2",   jax.scipy.special.digamma(jnp.array(2.0)) - jnp.log(jnp.array(2.0))),
    ("alpha=10",  jax.scipy.special.digamma(jnp.array(10.0)) - jnp.log(jnp.array(10.0))),
    ("alpha=100", jax.scipy.special.digamma(jnp.array(100.0)) - jnp.log(jnp.array(100.0))),
]


def bench_gamma() -> list[dict]:
    rows = []
    for label, target in GAMMA_TARGETS:
        target_j = jnp.asarray(target, dtype=jnp.float64)
        target_f = float(target)

        def call_jax():
            return _newton_digamma(target_j)

        def call_cpu():
            return _newton_digamma_cpu(target_f)

        first_j, cached_j = measure_first_then_cached(call_jax)
        first_c, cached_c = measure_first_then_cached(call_cpu)

        rows.append({
            "case": label,
            "target": float(target),
            "jit_first_ms":  first_j * 1e3,
            "jit_cached_ms": cached_j * 1e3,
            "cpu_first_ms":  first_c * 1e3,
            "cpu_cached_ms": cached_c * 1e3,
        })
    return rows


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_gig(rows):
    W = 110
    hdr("GIG η → θ — first-call vs cached-call (warm-started)", W)
    print(
        f"  {'Case':<32} "
        f"{'jax/newton 1st':>16} {'jax/newton cached':>18} "
        f"{'cpu/lbfgs 1st':>15} {'cpu/lbfgs cached':>18}"
    )
    sep(W)
    for r in rows:
        print(
            f"  {r['case']:<32} "
            f"{fmt_time(r['jax_newton_first_ms'] / 1e3):>16} "
            f"{fmt_time(r['jax_newton_cached_ms'] / 1e3):>18} "
            f"{fmt_time(r['cpu_lbfgs_first_ms'] / 1e3):>15} "
            f"{fmt_time(r['cpu_lbfgs_cached_ms'] / 1e3):>18}"
        )
    print(f"{'=' * W}")


def print_gamma(rows):
    W = 100
    hdr("Gamma η → θ via _newton_digamma — first-call vs cached-call", W)
    print(
        f"  {'Case':<14} "
        f"{'jit 1st':>12} {'jit cached':>14} "
        f"{'cpu 1st':>12} {'cpu cached':>14}"
    )
    sep(W)
    for r in rows:
        print(
            f"  {r['case']:<14} "
            f"{fmt_time(r['jit_first_ms'] / 1e3):>12} "
            f"{fmt_time(r['jit_cached_ms'] / 1e3):>14} "
            f"{fmt_time(r['cpu_first_ms'] / 1e3):>12} "
            f"{fmt_time(r['cpu_cached_ms'] / 1e3):>14}"
        )
    print(f"{'=' * W}")


def print_summary(gig_rows, gamma_rows):
    """Headline cached-call ratios — what the EM loop actually sees."""
    W = 90
    hdr("Cached-call summary (representative case)", W)
    if gig_rows:
        r = gig_rows[0]
        ratio_jn_first = r["jax_newton_first_ms"] / r["jax_newton_cached_ms"]
        ratio_vs_cpu = r["jax_newton_cached_ms"] / r["cpu_lbfgs_cached_ms"]
        print(
            f"  GIG warm-start ({r['case']}):\n"
            f"    jax/newton first → cached: {fmt_time(r['jax_newton_first_ms']/1e3)}"
            f" → {fmt_time(r['jax_newton_cached_ms']/1e3)}  ({ratio_jn_first:.1f}× speedup)\n"
            f"    jax/newton cached vs cpu/lbfgs cached: {ratio_vs_cpu:.2f}×"
        )
    if gamma_rows:
        g = gamma_rows[0]
        ratio = g["jit_first_ms"] / g["jit_cached_ms"]
        print(
            f"  Gamma _newton_digamma ({g['case']}):\n"
            f"    jit first → cached: {fmt_time(g['jit_first_ms']/1e3)}"
            f" → {fmt_time(g['jit_cached_ms']/1e3)}  ({ratio:.1f}× speedup)"
        )
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JIT-cache solver benchmark")
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to benchmarks/results/")
    args = parser.parse_args()

    print("\nnormix JIT-cache Solver Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    print("\nRunning GIG benchmark...", flush=True)
    gig_rows = bench_gig()
    print("Running Gamma _newton_digamma benchmark...", flush=True)
    gamma_rows = bench_gamma()

    print_gig(gig_rows)
    print_gamma(gamma_rows)
    print_summary(gig_rows, gamma_rows)

    if args.save:
        data = {
            "benchmark": "jit_solvers",
            "gig": gig_rows,
            "gamma": gamma_rows,
        }
        path = save_result("jit_solvers", data)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
