"""
Incremental EM benchmark — Phase-5 closeout.

Phase-5 measurement (see ``docs/investigations/test_suite_performance_2026-04-28.md``).
Compares the legacy Python ``for`` loop in ``IncrementalEMFitter._fit_incremental_python``
against the JIT-friendly ``jax.lax.scan`` body in ``_fit_incremental_scan`` for the same
distribution, data, and pre-stacked PRNG keys.

The Python loop pays Python dispatch per outer step plus repeated tracing of the
M-step closure when JAX backends are used. The scan path traces the body once, so
the same cost gets amortised across ``max_steps`` outer iterations and the cached
call should approach the per-step floor of the underlying jitted M-step.

Measured cases match the heaviest pre-Phase-5 incremental EM tests:
    Robbins-Monro on VG, NInvG, NIG, GH with ``max_steps=20``, ``batch_size=50``,
    ``e_step_backend='jax'``, ``m_step_backend='jax'``.

Usage
-----
    uv run python benchmarks/bench_incremental_em.py
    uv run python benchmarks/bench_incremental_em.py --save
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

from normix import (
    GeneralizedHyperbolic,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)
from normix.fitting.em import IncrementalEMFitter, _materialize_incremental_subkeys
from normix.fitting.eta_rules import RobbinsMonroUpdate


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DISTRIBUTIONS = [
    ("VG",    VarianceGamma),
    ("NInvG", NormalInverseGamma),
    ("NIG",   NormalInverseGaussian),
    ("GH",    GeneralizedHyperbolic),
]

D = 3
N = 200
BATCH = 50
MAX_STEPS = 20
INNER_ITER = 1
TAU0 = 10.0
DATA_KEY = jax.random.PRNGKey(42)
RUN_KEY = jax.random.PRNGKey(0)


def _make_data() -> jax.Array:
    return jax.random.normal(DATA_KEY, shape=(N, D), dtype=jnp.float64) * 0.5


def _make_model(dist_cls, X: jax.Array):
    d = X.shape[1]
    mu = jnp.mean(X, axis=0)
    sigma = jnp.cov(X.T) + 1e-4 * jnp.eye(d)
    if dist_cls is VarianceGamma:
        return VarianceGamma.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, alpha=2.0, beta=1.0)
    if dist_cls is NormalInverseGamma:
        return NormalInverseGamma.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, alpha=3.0, beta=1.0)
    if dist_cls is NormalInverseGaussian:
        return NormalInverseGaussian.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, mu_ig=1.0, lam=1.0)
    if dist_cls is GeneralizedHyperbolic:
        return GeneralizedHyperbolic.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, p=-0.5, a=2.0, b=1.0)
    raise TypeError(dist_cls)


def _make_fitter() -> IncrementalEMFitter:
    return IncrementalEMFitter(
        eta_update=RobbinsMonroUpdate(tau0=TAU0),
        batch_size=BATCH,
        max_steps=MAX_STEPS,
        inner_iter=INNER_ITER,
        e_step_backend='jax',
        m_step_backend='jax',
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _block(result):
    """Block JAX async dispatch so wall-clock includes device transfer."""
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, (tuple, list)):
        for v in result:
            _block(v)


def _block_em_result(res) -> None:
    """Force completion of every leaf in an EMResult so timing is accurate."""
    leaves = jax.tree.leaves(res.model)
    for leaf in leaves:
        _block(leaf)
    _block(res.param_changes)


def measure_first_then_cached(make_call, n_cached: int = 3):
    """Return ``(first_call_s, cached_call_median_s)``.

    ``make_call`` is a zero-arg factory returning an ``EMResult``. The first
    invocation pays JAX compilation cost; subsequent invocations should hit the
    JAX cache. ``n_cached`` is small here because each call drives ~20 outer
    EM steps and is far heavier than a single solver call.
    """
    t0 = time.perf_counter()
    res = make_call()
    _block_em_result(res)
    t_first = time.perf_counter() - t0

    times = []
    for _ in range(n_cached):
        t0 = time.perf_counter()
        res = make_call()
        _block_em_result(res)
        times.append(time.perf_counter() - t0)
    return t_first, float(np.median(times))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _run_scan(fitter, model, X, n_obs, batch_sz, step_keys):
    return fitter._fit_incremental_scan(
        model, X, n_obs, batch_sz, step_keys)


def _run_python(fitter, model, X, n_obs, batch_sz, step_keys, dist_name):
    return fitter._fit_incremental_python(
        model, X, n_obs, batch_sz, step_keys, dist_name)


def _ll_diff(scan_res, py_res, X) -> float:
    """Sanity check: same RNG should produce identical fit; report the LL gap."""
    ll_s = float(scan_res.model.marginal_log_likelihood(X))
    ll_p = float(py_res.model.marginal_log_likelihood(X))
    return abs(ll_s - ll_p)


def bench_distribution(dist_name: str, dist_cls) -> dict:
    X = _make_data()
    n_obs = int(X.shape[0])
    bs = int(min(BATCH, n_obs))
    step_keys = _materialize_incremental_subkeys(RUN_KEY, MAX_STEPS)

    fitter = _make_fitter()
    base_model = _make_model(dist_cls, X)

    scan_call = lambda: _run_scan(
        fitter, base_model, X, n_obs, bs, step_keys)
    py_call = lambda: _run_python(
        fitter, base_model, X, n_obs, bs, step_keys, dist_name)

    py_first, py_cached = measure_first_then_cached(py_call)
    scan_first, scan_cached = measure_first_then_cached(scan_call)

    # Sanity: scan and python paths should agree on the same keys.
    res_s = scan_call()
    res_p = py_call()
    ll_gap = _ll_diff(res_s, res_p, X)

    return {
        "dist": dist_name,
        "max_steps": MAX_STEPS,
        "batch_size": bs,
        "n": n_obs,
        "d": int(X.shape[1]),
        "py_first_s": py_first,
        "py_cached_s": py_cached,
        "scan_first_s": scan_first,
        "scan_cached_s": scan_cached,
        "ll_gap": ll_gap,
    }


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_table(rows):
    W = 110
    hdr(
        f"IncrementalEMFitter — Python loop vs lax.scan "
        f"(Robbins-Monro, max_steps={MAX_STEPS}, n={N}, d={D}, batch={BATCH})",
        W,
    )
    print(
        f"  {'Dist':<6} "
        f"{'py 1st':>10} {'py cached':>12} "
        f"{'scan 1st':>10} {'scan cached':>14} "
        f"{'cached speedup':>16} "
        f"{'ll_gap':>12}"
    )
    sep(W)
    for r in rows:
        speedup = (r['py_cached_s'] / r['scan_cached_s']
                   if r['scan_cached_s'] > 0 else float('nan'))
        print(
            f"  {r['dist']:<6} "
            f"{fmt_time(r['py_first_s']):>10} "
            f"{fmt_time(r['py_cached_s']):>12} "
            f"{fmt_time(r['scan_first_s']):>10} "
            f"{fmt_time(r['scan_cached_s']):>14} "
            f"{speedup:>15.2f}× "
            f"{r['ll_gap']:>12.2e}"
        )
    print(f"{'=' * W}")


def print_summary(rows):
    W = 90
    hdr("Summary — cached-call wall time per fit (Phase 5 hot path)", W)
    if not rows:
        print(f"{'=' * W}")
        return
    for r in rows:
        speedup = (r['py_cached_s'] / r['scan_cached_s']
                   if r['scan_cached_s'] > 0 else float('nan'))
        compile_amort = (
            r['scan_first_s'] / r['scan_cached_s']
            if r['scan_cached_s'] > 0 else float('nan')
        )
        print(
            f"  {r['dist']:<6} "
            f"python {fmt_time(r['py_cached_s'])} → "
            f"scan {fmt_time(r['scan_cached_s'])} "
            f"({speedup:.2f}× faster, "
            f"first/cached = {compile_amort:.1f}×)"
        )
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Incremental EM JIT-scan benchmark")
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to benchmarks/results/")
    args = parser.parse_args()

    print("\nnormix Incremental EM Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}",
          flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    rows = []
    for name, cls in DISTRIBUTIONS:
        print(f"\nRunning {name}...", flush=True)
        rows.append(bench_distribution(name, cls))

    print_table(rows)
    print_summary(rows)

    if args.save:
        data = {
            "benchmark": "incremental_em",
            "config": {
                "n": N, "d": D, "batch_size": BATCH,
                "max_steps": MAX_STEPS, "inner_iter": INNER_ITER,
                "tau0": TAU0,
            },
            "rows": rows,
        }
        path = save_result("incremental_em", data)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
