"""Time float64 GIG inversion and fixed-count GH EM on the default device.

Usage: ``uv run python benchmarks/bench_gig_gpu_latency.py --out results.json``.
Set ``JAX_PLATFORMS=cpu`` or ``JAX_PLATFORMS=cuda,cpu`` before starting Python
to compare devices. First calls include compilation still needed in this
process; cached medians block after every call. Warm starts use the ASV
neighbour, not the exact solution. Batches cycle through the same targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import geninvgauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.utils import fmt_time, jax_timeit, system_info
from normix import GeneralizedHyperbolic
from normix.distributions.generalized_inverse_gaussian import GIG


_CASES = {
    "symmetric": (0.5, 1.0, 1.0),
    "asymmetric": (0.5, 10.0, 0.1),
    "inverse_asymmetric": (-1.0, 0.1, 10.0),
    "inverse_gaussian": (-0.5, 2.0, 1.0),
}


def _measure(fn: Callable, args: tuple, runs: int) -> tuple[dict, jax.Array]:
    jax.block_until_ready(args)
    start = time.perf_counter()
    solved = jax.block_until_ready(fn(*args))
    first = time.perf_counter() - start
    cached = jax_timeit(lambda: fn(*args), n_runs=runs, warmup=1)
    values = np.asarray(solved)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("GIG inversion returned nonfinite natural parameters")
    return {"first_s": first, "cached_median_s": cached}, solved


def _inverter(warm: bool, maxiter: int, tol: float) -> Callable:
    if warm:
        def invert(eta, theta0):
            return GIG.from_expectation(
                eta, theta0=theta0, maxiter=maxiter, tol=tol,
            ).natural_params()
    else:
        def invert(eta):
            return GIG.from_expectation(
                eta, maxiter=maxiter, tol=tol,
            ).natural_params()
    return invert


def _report(label: str, timing: dict) -> None:
    print(
        f"{label}: first={fmt_time(timing['first_s'])}, "
        f"cached median={fmt_time(timing['cached_median_s'])}",
        flush=True,
    )


def _em_benchmark(n: int, d: int, iterations: int) -> dict:
    rng = np.random.default_rng(136 + d)
    y = geninvgauss.rvs(0.5, 1.0, size=n, random_state=rng)
    gamma = np.linspace(-0.15, 0.15, d)
    x = y[:, None] * gamma + np.sqrt(y[:, None]) * rng.standard_normal((n, d))
    data = jnp.asarray(x, dtype=jnp.float64)
    model = GeneralizedHyperbolic.default_init(data)
    jax.block_until_ready((data, model))

    @jax.jit
    def step(current, observations):
        eta = current.e_step(observations, backend="jax").eta
        updated = current.m_step_normal(eta).m_step_subordinator(
            eta, backend="jax", method="newton", maxiter=20,
        )
        return updated.regularize_det_sigma_one()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model = jax.block_until_ready(step(model, data))
        times.append(time.perf_counter() - start)
        if not all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(model)):
            raise RuntimeError(f"GH EM failed after {len(times)} iterations")

    ll = float(jax.block_until_ready(model.marginal_log_likelihood(data)))
    if not np.isfinite(ll):
        raise RuntimeError("GH EM returned a nonfinite likelihood")
    result = {
        "n": n, "d": d, "iterations": iterations,
        "first_iteration_s": times[0],
        "cached_mean_per_iteration_s": float(np.mean(times[1:])),
        "cached_median_per_iteration_s": float(np.median(times[1:])),
        "iteration_times_s": times,
        "final_log_likelihood": ll,
        "subordinator_theta": np.asarray(model._joint.subordinator().natural_params()).tolist(),
    }
    print(
        f"GH EM n={n}, d={d}, iterations={iterations}: "
        f"first={fmt_time(times[0])}, "
        f"mean/iteration excluding first={fmt_time(result['cached_mean_per_iteration_s'])}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write timings, targets, and solved theta as JSON")
    parser.add_argument("--runs", type=int, default=10, help="Calls per cached median")
    parser.add_argument("--maxiter", type=int, default=500, help="GIG iteration budget")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--em-iterations", type=int, default=20)
    args = parser.parse_args()
    if args.runs < 1 or args.maxiter < 1 or args.em_iterations < 2:
        parser.error("runs and maxiter must be positive; em-iterations must be at least 2")

    info = system_info()
    info.update(backend=jax.default_backend(), x64=jax.config.jax_enable_x64)
    print(json.dumps(info, indent=2), flush=True)
    payload = {"system": info, "maxiter": args.maxiter, "tol": args.tol,
               "single": {}, "batch": {}, "em": []}
    targets, starts = [], []
    for label, (p, a, b) in _CASES.items():
        source = GIG(jnp.float64(p), jnp.float64(a), jnp.float64(b))
        eta = jax.block_until_ready(source.expectation_params())
        theta0 = GIG(jnp.float64(p + 0.25), jnp.float64(a * 1.5),
                     jnp.float64(b * 0.5)).natural_params()
        targets.append(eta)
        starts.append(theta0)
        row = {"classical": [p, a, b], "eta": np.asarray(eta).tolist(),
               "theta0": np.asarray(theta0).tolist()}
        for warm in (True, False):
            mode = "warm" if warm else "cold"
            fn = jax.jit(_inverter(warm, args.maxiter, args.tol))
            operands = (eta, theta0) if warm else (eta,)
            timing, theta = _measure(fn, operands, args.runs)
            timing["theta"] = np.asarray(theta).tolist()
            row[mode] = timing
            _report(f"GIG {label} {mode}", timing)
        payload["single"][label] = row

    targets = jnp.stack(targets)
    starts = jnp.stack(starts)
    for size in (1, 64, 1024):
        indices = jnp.arange(size) % len(_CASES)
        eta_batch, start_batch = targets[indices], starts[indices]
        row = {}
        for warm in (True, False):
            mode = "warm" if warm else "cold"
            fn = jax.jit(jax.vmap(_inverter(warm, args.maxiter, args.tol)))
            operands = (eta_batch, start_batch) if warm else (eta_batch,)
            timing, theta = _measure(fn, operands, args.runs)
            timing["theta"] = np.asarray(theta).tolist()
            row[mode] = timing
            _report(f"GIG batch={size} {mode}", timing)
        payload["batch"][str(size)] = row

    for n, d in ((500, 2), (10000, 10)):
        payload["em"].append(_em_benchmark(n, d, args.em_iterations))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
