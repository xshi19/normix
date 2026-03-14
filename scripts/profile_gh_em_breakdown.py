#!/usr/bin/env python3
"""
Profile where GH EM time is spent.

This script does not modify normix itself. It builds a synthetic dataset,
initializes a GeneralizedHyperbolic model, and then runs a small number of EM
iterations while timing:

1. Regularization
2. E-step
3. M-step total
4. GIG eta->theta update inside the M-step
5. Remaining M-step work (closed-form normal update + Cholesky)
6. Optional marginal logpdf pass (the extra work done when verbose >= 1)

It also prints compact cProfile summaries for:
- one E-step
- one isolated GIG eta->theta solve

Usage
-----
    python scripts/profile_gh_em_breakdown.py --n-samples 5000 --dim 8
    python scripts/profile_gh_em_breakdown.py --n-samples 5000 --dim 8 --input-backend jax
"""

from __future__ import annotations

import argparse
import cProfile
import copy
import io
import json
import os
import pstats
import sys
import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from normix.distributions.mixtures import GeneralizedHyperbolic
from normix.distributions.mixtures.generalized_hyperbolic import (
    REGULARIZATION_METHODS,
)
from normix.distributions.univariate import GeneralizedInverseGaussian


@dataclass
class IterationTiming:
    iteration: int
    regularize_pre_s: float
    e_step_s: float
    m_step_s: float
    gig_opt_s: float
    m_step_non_gig_s: float
    regularize_post_s: float
    logpdf_s: float
    total_s: float
    max_rel_change: float
    gig_calls: int
    scipy_minimize_calls: int
    scipy_nit: list[int]
    scipy_nfev: list[int]


def make_correlated_data(n_samples: int, dim: int, seed: int) -> np.ndarray:
    """Generate a moderately correlated dataset for repeatable timings."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    sigma = A @ A.T
    sigma /= np.mean(np.diag(sigma))
    sigma += 0.2 * np.eye(dim)
    mu = rng.normal(scale=0.5, size=dim)
    X = rng.multivariate_normal(mean=mu, cov=sigma, size=n_samples)

    if dim == 1:
        return X.reshape(-1, 1)
    return X


def apply_regularization(gh: GeneralizedHyperbolic, regularization: str) -> None:
    """Mirror the regularization step used inside GH.fit."""
    jt = gh._joint
    d = gh.d
    regularize_fn = REGULARIZATION_METHODS[regularization]
    mu = jt._mu
    gamma = jt._gamma
    sigma = jt._L_Sigma @ jt._L_Sigma.T
    p_val, a_val, b_val = jt._p, jt._a, jt._b
    log_det_sigma = jt.log_det_Sigma

    if regularization == "det_sigma_one":
        regularized = regularize_fn(
            mu, gamma, sigma, p_val, a_val, b_val, d,
            log_det_sigma=log_det_sigma,
        )
    else:
        regularized = regularize_fn(mu, gamma, sigma, p_val, a_val, b_val, d)

    a_reg = regularized["a"]
    b_reg = regularized["b"]
    if a_reg < 1e-6 or b_reg < 1e-6 or a_reg > 1e6 or b_reg > 1e6:
        regularized["a"] = np.clip(a_reg, 1e-6, 1e6)
        regularized["b"] = np.clip(b_reg, 1e-6, 1e6)

    gh._joint.set_classical_params(**regularized)


def initialize_model(
    X: np.ndarray,
    seed: int,
    regularization: str,
) -> tuple[GeneralizedHyperbolic, float]:
    """Initialize GH once, excluding later EM iterations."""
    gh = GeneralizedHyperbolic()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    t0 = time.perf_counter()
    gh._joint = gh._create_joint_distribution()
    gh._joint._d = X.shape[1]
    gh._initialize_params(X, random_state=seed)
    apply_regularization(gh, regularization)
    init_s = time.perf_counter() - t0
    return gh, init_s


def profile_summary(func, *args, lines: int = 15, **kwargs) -> str:
    """Return a short cumulative-time cProfile summary."""
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(lines)
    return stream.getvalue()


def maybe_measure_jax_array_to_numpy(X: np.ndarray) -> dict[str, Any]:
    """Check whether JAX is available and whether fit would coerce arrays."""
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    arr = jax.device_put(jnp.asarray(X))
    if hasattr(arr, "block_until_ready"):
        arr.block_until_ready()

    t0 = time.perf_counter()
    host = np.asarray(arr)
    if hasattr(arr, "block_until_ready"):
        arr.block_until_ready()
    transfer_s = time.perf_counter() - t0

    devices = []
    try:
        devices = [str(d) for d in jax.devices()]
    except Exception:
        pass

    return {
        "available": True,
        "jax_version": getattr(jax, "__version__", "unknown"),
        "devices": devices,
        "coerced_shape": list(host.shape),
        "np_asarray_s": transfer_s,
    }


def run_iteration(
    gh: GeneralizedHyperbolic,
    X: np.ndarray,
    iteration: int,
    regularization: str,
    *,
    fix_tail: bool,
) -> IterationTiming:
    """Run one GH EM iteration and attribute M-step time."""
    import normix.base.exponential_family as ef_mod

    prev_mu = gh._joint._mu.copy()
    prev_gamma = gh._joint._gamma.copy()
    prev_L = gh._joint._L_Sigma.copy()

    t_iter_0 = time.perf_counter()

    t0 = time.perf_counter()
    apply_regularization(gh, regularization)
    regularize_pre_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cond_exp = gh._conditional_expectation_y_given_x(X)
    e_step_s = time.perf_counter() - t0

    gig_opt_s = 0.0
    gig_calls = 0
    scipy_minimize_calls = 0
    scipy_nit: list[int] = []
    scipy_nfev: list[int] = []

    original_set_expectation = GeneralizedInverseGaussian.set_expectation_params
    original_minimize = ef_mod.minimize

    def timed_set_expectation(self, eta, theta0=None):
        nonlocal gig_opt_s, gig_calls
        t_inner_0 = time.perf_counter()
        try:
            return original_set_expectation(self, eta, theta0=theta0)
        finally:
            gig_opt_s += time.perf_counter() - t_inner_0
            gig_calls += 1

    def timed_minimize(*args, **kwargs):
        nonlocal scipy_minimize_calls
        scipy_minimize_calls += 1
        result = original_minimize(*args, **kwargs)
        scipy_nit.append(int(getattr(result, "nit", -1)))
        scipy_nfev.append(int(getattr(result, "nfev", -1)))
        return result

    GeneralizedInverseGaussian.set_expectation_params = timed_set_expectation
    ef_mod.minimize = timed_minimize
    try:
        t0 = time.perf_counter()
        gh._m_step(X, cond_exp, fix_tail=fix_tail, verbose=0)
        m_step_s = time.perf_counter() - t0
    finally:
        GeneralizedInverseGaussian.set_expectation_params = original_set_expectation
        ef_mod.minimize = original_minimize

    t0 = time.perf_counter()
    apply_regularization(gh, regularization)
    regularize_post_s = time.perf_counter() - t0

    max_rel_change = gh._check_convergence(
        prev_mu, prev_gamma, prev_L, verbose=0, iteration=iteration, X=X
    )

    t0 = time.perf_counter()
    _ = np.mean(gh.logpdf(X))
    logpdf_s = time.perf_counter() - t0

    total_s = time.perf_counter() - t_iter_0

    return IterationTiming(
        iteration=iteration,
        regularize_pre_s=regularize_pre_s,
        e_step_s=e_step_s,
        m_step_s=m_step_s,
        gig_opt_s=gig_opt_s,
        m_step_non_gig_s=max(m_step_s - gig_opt_s, 0.0),
        regularize_post_s=regularize_post_s,
        logpdf_s=logpdf_s,
        total_s=total_s,
        max_rel_change=float(max_rel_change),
        gig_calls=gig_calls,
        scipy_minimize_calls=scipy_minimize_calls,
        scipy_nit=scipy_nit,
        scipy_nfev=scipy_nfev,
    )


def run_benchmark(
    X: np.ndarray,
    seed: int,
    regularization: str,
    n_iters: int,
    *,
    fix_tail: bool,
) -> tuple[float, list[IterationTiming], str, str]:
    """Initialize GH and run several timed iterations."""
    gh, init_s = initialize_model(X, seed, regularization)

    e_profile = profile_summary(gh._conditional_expectation_y_given_x, X, lines=12)

    cond_exp = gh._conditional_expectation_y_given_x(X)
    sub_eta = np.array(
        [
            np.mean(cond_exp["E_log_Y"]),
            np.mean(cond_exp["E_inv_Y"]),
            np.mean(cond_exp["E_Y"]),
        ]
    )
    sub = copy.deepcopy(gh._joint.subordinator)
    gig_profile = profile_summary(
        sub.set_expectation_params,
        sub_eta,
        theta0=sub.natural_params,
        lines=12,
    )

    timings = []
    for iteration in range(1, n_iters + 1):
        timings.append(
            run_iteration(
                gh,
                X,
                iteration,
                regularization,
                fix_tail=fix_tail,
            )
        )

    return init_s, timings, e_profile, gig_profile


def print_run_summary(
    label: str,
    init_s: float,
    timings: list[IterationTiming],
) -> None:
    """Print aggregated timing information."""
    print(f"\n=== {label} ===")
    print(f"Initialization: {init_s:.4f} s")

    if not timings:
        return

    avg_total = mean(t.total_s for t in timings)
    avg_e = mean(t.e_step_s for t in timings)
    avg_m = mean(t.m_step_s for t in timings)
    avg_gig = mean(t.gig_opt_s for t in timings)
    avg_non_gig = mean(t.m_step_non_gig_s for t in timings)
    avg_logpdf = mean(t.logpdf_s for t in timings)
    avg_pre = mean(t.regularize_pre_s for t in timings)
    avg_post = mean(t.regularize_post_s for t in timings)

    print("Average per-iteration timings:")
    print(f"  total           : {avg_total:.4f} s")
    print(f"  e-step          : {avg_e:.4f} s  ({100 * avg_e / avg_total:.1f}%)")
    print(f"  m-step total    : {avg_m:.4f} s  ({100 * avg_m / avg_total:.1f}%)")
    print(f"  gig optimizer   : {avg_gig:.4f} s  ({100 * avg_gig / avg_total:.1f}%)")
    print(f"  m-step non-gig  : {avg_non_gig:.4f} s  ({100 * avg_non_gig / avg_total:.1f}%)")
    print(f"  regularize pre  : {avg_pre:.4f} s")
    print(f"  regularize post : {avg_post:.4f} s")
    print(f"  extra logpdf    : {avg_logpdf:.4f} s  (verbose >= 1 overhead)")

    all_nit = [nit for t in timings for nit in t.scipy_nit if nit >= 0]
    all_nfev = [nfev for t in timings for nfev in t.scipy_nfev if nfev >= 0]
    if all_nit:
        print(
            "  scipy minimize  : "
            f"{len(all_nit)} call(s), mean nit={mean(all_nit):.1f}, "
            f"mean nfev={mean(all_nfev):.1f}"
        )

    print("\nPer-iteration detail:")
    for timing in timings:
        print(
            f"  iter={timing.iteration:02d} "
            f"total={timing.total_s:.4f}s "
            f"e={timing.e_step_s:.4f}s "
            f"m={timing.m_step_s:.4f}s "
            f"gig={timing.gig_opt_s:.4f}s "
            f"logpdf={timing.logpdf_s:.4f}s "
            f"rel_change={timing.max_rel_change:.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regularization", type=str, default="det_sigma_one")
    parser.add_argument(
        "--input-backend",
        type=str,
        choices=("numpy", "jax"),
        default="numpy",
        help="Create the synthetic dataset as a NumPy array or a JAX device array.",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    X_numpy = make_correlated_data(args.n_samples, args.dim, args.seed)
    print(
        f"Dataset: n_samples={args.n_samples}, dim={args.dim}, seed={args.seed}, "
        f"regularization={args.regularization}, input_backend={args.input_backend}"
    )

    jax_info = maybe_measure_jax_array_to_numpy(X_numpy)
    if jax_info["available"]:
        print(
            "JAX available: "
            f"version={jax_info['jax_version']}, devices={jax_info['devices']}, "
            f"np.asarray(device_array)={jax_info['np_asarray_s']:.6f}s"
        )
    else:
        print(f"JAX check skipped: {jax_info['reason']}")

    X = X_numpy
    if args.input_backend == "jax":
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise RuntimeError(
                "--input-backend jax requested, but JAX is unavailable"
            ) from exc
        X = jax.device_put(jnp.asarray(X_numpy))
        if hasattr(X, "block_until_ready"):
            X.block_until_ready()

    full_init_s, full_timings, e_profile, gig_profile = run_benchmark(
        X,
        seed=args.seed,
        regularization=args.regularization,
        n_iters=args.iters,
        fix_tail=False,
    )
    fixed_init_s, fixed_timings, _, _ = run_benchmark(
        X,
        seed=args.seed,
        regularization=args.regularization,
        n_iters=args.iters,
        fix_tail=True,
    )

    print_run_summary("GH EM with tail update", full_init_s, full_timings)
    print_run_summary("GH EM with fix_tail=True", fixed_init_s, fixed_timings)

    if full_timings and fixed_timings:
        avg_full_total = mean(t.total_s for t in full_timings)
        avg_fixed_total = mean(t.total_s for t in fixed_timings)
        avg_full_m = mean(t.m_step_s for t in full_timings)
        avg_fixed_m = mean(t.m_step_s for t in fixed_timings)
        avg_full_gig = mean(t.gig_opt_s for t in full_timings)

        print("\n=== Comparison ===")
        print(
            f"Tail update adds about {avg_full_total - avg_fixed_total:.4f} s "
            f"per iteration overall."
        )
        print(
            f"M-step grows from {avg_fixed_m:.4f} s to {avg_full_m:.4f} s, "
            f"with measured GIG time {avg_full_gig:.4f} s."
        )

    print("\n=== cProfile: one E-step ===")
    print(e_profile)
    print("\n=== cProfile: one GIG eta->theta solve ===")
    print(gig_profile)

    if args.json_out:
        payload = {
            "args": vars(args),
            "jax_info": jax_info,
            "full_init_s": full_init_s,
            "fixed_init_s": fixed_init_s,
            "full_timings": [asdict(t) for t in full_timings],
            "fixed_timings": [asdict(t) for t in fixed_timings],
            "e_profile": e_profile,
            "gig_profile": gig_profile,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
