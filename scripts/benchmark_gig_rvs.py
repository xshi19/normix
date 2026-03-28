"""
GIG random variate generation benchmark — Devroye vs PINV vs SciPy.

Compares three methods across multiple parameter sets and sample sizes:
  1. Devroye (2014) — TDR rejection on log(x), pure JAX
  2. PINV — numerical inverse CDF (CPU table + JAX sampling)
  3. SciPy — scipy.stats.geninvgauss (CPU baseline)

Reports timing table and validates correctness via KS test + moment comparison.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/benchmark_gig_rvs.py
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy import stats

from normix import GIG
from normix.distributions._gig_rvs import (
    gig_rvs_devroye, gig_build_pinv_table, gig_rvs_pinv,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAM_SETS = [
    ("p=1, a=1, b=1",       1.0, 1.0, 1.0),
    ("p=0.5, a=2, b=0.5",   0.5, 2.0, 0.5),
    ("p=-0.5, a=1, b=1",   -0.5, 1.0, 1.0),
    ("p=3, a=0.5, b=2",     3.0, 0.5, 2.0),
    ("p=-2, a=0.5, b=3",   -2.0, 0.5, 3.0),
    ("p=0.1, a=10, b=0.1",  0.1, 10.0, 0.1),
]

SAMPLE_SIZES = [100, 1_000, 10_000, 100_000]

METHODS = ["devroye", "pinv", "scipy"]

KS_ALPHA = 0.01


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_devroye(p, a, b, n, seed=42):
    """Time Devroye sampling.  Runs twice; reports the second (warm) time."""
    key = jax.random.PRNGKey(seed)
    # cold run (JIT compilation)
    s = gig_rvs_devroye(key, p, a, b, n)
    s.block_until_ready()
    # warm run
    key2 = jax.random.PRNGKey(seed + 1)
    t0 = time.perf_counter()
    samples = gig_rvs_devroye(key2, p, a, b, n)
    samples.block_until_ready()
    elapsed = time.perf_counter() - t0
    return np.asarray(samples), elapsed


def time_pinv(p, a, b, n, seed=42):
    """Time PINV: report setup and sampling separately (warm sampling)."""
    key = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()
    u_grid, x_grid = gig_build_pinv_table(p, a, b)
    t_setup = time.perf_counter() - t0

    # cold run
    s = gig_rvs_pinv(key, u_grid, x_grid, n)
    s.block_until_ready()
    # warm run
    key2 = jax.random.PRNGKey(seed + 1)
    t0 = time.perf_counter()
    samples = gig_rvs_pinv(key2, u_grid, x_grid, n)
    samples.block_until_ready()
    t_sample = time.perf_counter() - t0

    return np.asarray(samples), t_setup, t_sample


def time_scipy(p, a, b, n, seed=42):
    """Time SciPy geninvgauss."""
    b_sp = np.sqrt(a * b)
    scale = np.sqrt(b / a)
    t0 = time.perf_counter()
    samples = stats.geninvgauss.rvs(p=p, b=b_sp, scale=scale,
                                     size=n, random_state=seed)
    elapsed = time.perf_counter() - t0
    return samples, elapsed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def ks_test(samples, p, a, b):
    """Two-sided KS test against scipy.stats.geninvgauss CDF."""
    b_sp = np.sqrt(a * b)
    scale = np.sqrt(b / a)
    stat, pval = stats.kstest(
        np.asarray(samples),
        lambda x: stats.geninvgauss.cdf(x, p=p, b=b_sp, scale=scale),
    )
    return stat, pval


def moment_error(samples, gig):
    """Relative error of sample mean vs analytical mean."""
    true_mean = float(gig.mean())
    sample_mean = float(np.mean(samples))
    return abs(sample_mean - true_mean) / max(abs(true_mean), 1e-10)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def fmt_time(t):
    if t >= 1.0:
        return f"{t:.2f}s"
    if t >= 0.01:
        return f"{t*1e3:.0f}ms"
    return f"{t*1e3:.1f}ms"


def print_timing_table(results):
    W = 120
    print(f"\n{'=' * W}")
    print("  GIG RVS Benchmark — Timing")
    print(f"{'=' * W}")
    header = (
        f"  {'Parameters':<25} {'n':>8}  {'Devroye':>10}  "
        f"{'PINV setup':>10} {'PINV samp':>10} {'PINV total':>10}  "
        f"{'SciPy':>10}"
    )
    print(header)
    print(f"  {'-' * (W - 2)}")

    prev_params = None
    for r in results:
        if prev_params is not None and r["params"] != prev_params:
            print(f"  {'-' * (W - 2)}")
        prev_params = r["params"]
        pinv_total = r["pinv_setup"] + r["pinv_sample"]
        print(
            f"  {r['params']:<25} {r['n']:>8,}  {fmt_time(r['devroye']):>10}  "
            f"{fmt_time(r['pinv_setup']):>10} {fmt_time(r['pinv_sample']):>10} "
            f"{fmt_time(pinv_total):>10}  "
            f"{fmt_time(r['scipy']):>10}"
        )
    print(f"{'=' * W}")


def print_validation_table(val_results):
    W = 120
    print(f"\n{'=' * W}")
    print("  GIG RVS Benchmark — Validation (n=10,000)")
    print(f"{'=' * W}")
    header = (
        f"  {'Parameters':<25} {'Method':<10} "
        f"{'Sample Mean':>12} {'True Mean':>12} {'Rel Err':>10} "
        f"{'KS stat':>10} {'KS p-val':>10} {'Pass':>5}"
    )
    print(header)
    print(f"  {'-' * (W - 2)}")

    prev_params = None
    for r in val_results:
        if prev_params is not None and r["params"] != prev_params:
            print(f"  {'-' * (W - 2)}")
        prev_params = r["params"]
        ok = "Y" if r["ks_pval"] > KS_ALPHA else "N"
        print(
            f"  {r['params']:<25} {r['method']:<10} "
            f"{r['sample_mean']:>12.4f} {r['true_mean']:>12.4f} "
            f"{r['rel_err']:>10.4f} "
            f"{r['ks_stat']:>10.4f} {r['ks_pval']:>10.4f} {ok:>5}"
        )
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nnormix GIG RVS Benchmark")
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    # ---- JIT warm-up ----
    print("\nWarming up JIT...", end="", flush=True)
    key = jax.random.PRNGKey(999)
    _ = gig_rvs_devroye(key, 1.0, 1.0, 1.0, 10)
    _.block_until_ready()
    u, x = gig_build_pinv_table(1.0, 1.0, 1.0, n_grid=100)
    _ = gig_rvs_pinv(key, u, x, 10)
    _.block_until_ready()
    print(" done.", flush=True)

    # ---- Timing benchmark ----
    print("\nRunning timing benchmark...")
    timing_results = []

    for name, p, a, b in PARAM_SETS:
        for n in SAMPLE_SIZES:
            desc = f"{name}, n={n:,}"
            print(f"  {desc:<45}", end="", flush=True)

            _, t_dev = time_devroye(p, a, b, n)
            _, t_pinv_setup, t_pinv_sample = time_pinv(p, a, b, n)
            _, t_scipy = time_scipy(p, a, b, n)

            timing_results.append({
                "params": name, "n": n,
                "devroye": t_dev,
                "pinv_setup": t_pinv_setup,
                "pinv_sample": t_pinv_sample,
                "scipy": t_scipy,
            })
            print(
                f"  dev={fmt_time(t_dev)}, "
                f"pinv={fmt_time(t_pinv_setup)}+{fmt_time(t_pinv_sample)}, "
                f"scipy={fmt_time(t_scipy)}",
                flush=True,
            )

    print_timing_table(timing_results)

    # ---- Validation ----
    print("\nRunning validation (n=10,000)...")
    val_results = []
    n_val = 10_000

    for name, p, a, b in PARAM_SETS:
        gig = GIG(p=p, a=a, b=b)
        true_mean = float(gig.mean())

        for method in METHODS:
            samples = gig.rvs(n_val, seed=42, method=method)
            samples_np = np.asarray(samples)
            ks_stat, ks_pval = ks_test(samples_np, p, a, b)
            rel_err = moment_error(samples_np, gig)

            val_results.append({
                "params": name, "method": method,
                "sample_mean": float(np.mean(samples_np)),
                "true_mean": true_mean,
                "rel_err": rel_err,
                "ks_stat": ks_stat, "ks_pval": ks_pval,
            })

    print_validation_table(val_results)

    # ---- Summary ----
    n_fail = sum(1 for r in val_results if r["ks_pval"] <= KS_ALPHA)
    n_total = len(val_results)
    print(f"\nKS test: {n_total - n_fail}/{n_total} passed (alpha={KS_ALPHA})")
    if n_fail > 0:
        print("  FAILED:")
        for r in val_results:
            if r["ks_pval"] <= KS_ALPHA:
                print(f"    {r['params']} / {r['method']}: "
                      f"KS={r['ks_stat']:.4f}, p={r['ks_pval']:.4f}")


if __name__ == "__main__":
    main()
