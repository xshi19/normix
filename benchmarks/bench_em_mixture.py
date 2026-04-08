"""
EM benchmark for all mixture distributions × backends × methods.

Runs EM (and optionally MCECM) to convergence with per-sub-step timing:
  E-step | M-step normal | M-step subordinator | regularize

Usage:
    uv run python benchmarks/bench_em_mixture.py
    uv run python benchmarks/bench_em_mixture.py --large
    uv run python benchmarks/bench_em_mixture.py --mcecm --n-stocks 50
    uv run python benchmarks/bench_em_mixture.py --save
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.utils import load_sp500_data, save_result, fmt_time, hdr, sep


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

from normix import (
    GeneralizedHyperbolic,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)

DISTRIBUTIONS = [
    ("VG", VarianceGamma),
    ("NInvG", NormalInverseGamma),
    ("NIG", NormalInverseGaussian),
    ("GH", GeneralizedHyperbolic),
]


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

@dataclass
class EMConfig:
    dist_name: str
    dist_cls: type
    algorithm: str      # 'em' or 'mcecm'
    e_backend: str
    m_backend: str
    m_method: str
    regularization: str


def build_configs(include_mcecm: bool = False) -> list[EMConfig]:
    """Build all meaningful benchmark configurations.

    For VG/NIG/NInvG the subordinator M-step is closed-form, so only the
    E-step backend matters.  For GH the M-step solver (lbfgs vs newton,
    cpu vs jax) is the bottleneck, so we test multiple combinations.
    """
    configs = []
    algos = ['em', 'mcecm'] if include_mcecm else ['em']

    for dist_name, dist_cls in DISTRIBUTIONS:
        reg = 'det_sigma_one' if dist_name == 'GH' else 'none'
        for algo in algos:
            # CPU E-step, CPU M-step, lbfgs (baseline for GH)
            configs.append(EMConfig(
                dist_name, dist_cls, algo, 'cpu', 'cpu', 'lbfgs', reg))

            # JAX E-step, CPU M-step, lbfgs
            configs.append(EMConfig(
                dist_name, dist_cls, algo, 'jax', 'cpu', 'lbfgs', reg))

            if dist_name == 'GH':
                # CPU E-step, CPU M-step, newton
                configs.append(EMConfig(
                    dist_name, dist_cls, algo, 'cpu', 'cpu', 'newton', reg))
                # CPU E-step, JAX M-step, newton
                configs.append(EMConfig(
                    dist_name, dist_cls, algo, 'cpu', 'jax', 'newton', reg))

    return configs


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class EMBenchResult:
    dist_name: str
    algorithm: str
    e_backend: str
    m_backend: str
    m_method: str
    n_iter: int
    converged: bool
    total_time: float
    avg_e_step: float
    avg_m_normal: float
    avg_m_sub: float
    avg_regularize: float
    final_ll: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core: manual EM loop with per-sub-step timing
# ---------------------------------------------------------------------------

def _regularize(model, cfg: EMConfig):
    if cfg.regularization == 'det_sigma_one':
        if hasattr(model, 'regularize_det_sigma_one'):
            return model.regularize_det_sigma_one()
    return model


def _param_change(model, prev_mu, prev_gamma, prev_L):
    """Max relative change in normal parameters."""
    eps = 1e-10
    mu = model._joint.mu
    gamma = model._joint.gamma
    L = model._joint.L_Sigma
    rel_mu = jnp.linalg.norm(mu - prev_mu) / jnp.maximum(
        jnp.linalg.norm(prev_mu), eps)
    rel_g = jnp.linalg.norm(gamma - prev_gamma) / jnp.maximum(
        jnp.linalg.norm(prev_gamma), eps)
    rel_L = jnp.linalg.norm(L - prev_L) / jnp.maximum(
        jnp.linalg.norm(prev_L), eps)
    return float(jnp.maximum(jnp.maximum(rel_mu, rel_g), rel_L))


def run_benchmark(cfg: EMConfig, X: jax.Array,
                  max_iter: int, tol: float) -> EMBenchResult:
    """Run EM to convergence with sub-step timing breakdown."""
    try:
        model = cfg.dist_cls.default_init(X)
    except Exception as e:
        return EMBenchResult(
            cfg.dist_name, cfg.algorithm, cfg.e_backend, cfg.m_backend,
            cfg.m_method, 0, False, 0, 0, 0, 0, 0, float('nan'),
            error=f"init: {e!s:.50}")

    e_times, mn_times, ms_times, reg_times = [], [], [], []
    n_iter = 0
    converged = False
    t_total = time.perf_counter()

    try:
        for i in range(max_iter):
            prev_mu = model._joint.mu
            prev_gamma = model._joint.gamma
            prev_L = model._joint.L_Sigma

            if cfg.algorithm == 'mcecm':
                # Cycle 1: E → M_normal → regularize
                t0 = time.perf_counter()
                eta = model.e_step(X, backend=cfg.e_backend)
                t_e1 = time.perf_counter() - t0

                t0 = time.perf_counter()
                model = model.m_step_normal(eta)
                t_mn = time.perf_counter() - t0

                t0 = time.perf_counter()
                model = _regularize(model, cfg)
                t_reg = time.perf_counter() - t0

                # Cycle 2: E → M_subordinator
                t0 = time.perf_counter()
                eta = model.e_step(X, backend=cfg.e_backend)
                t_e2 = time.perf_counter() - t0

                t0 = time.perf_counter()
                model = model.m_step_subordinator(
                    eta, backend=cfg.m_backend, method=cfg.m_method)
                t_ms = time.perf_counter() - t0

                e_times.append(t_e1 + t_e2)
                mn_times.append(t_mn)
                ms_times.append(t_ms)
                reg_times.append(t_reg)
            else:
                # EM: E → M_normal → M_subordinator → regularize
                t0 = time.perf_counter()
                eta = model.e_step(X, backend=cfg.e_backend)
                t_e = time.perf_counter() - t0

                t0 = time.perf_counter()
                model = model.m_step_normal(eta)
                t_mn = time.perf_counter() - t0

                t0 = time.perf_counter()
                model = model.m_step_subordinator(
                    eta, backend=cfg.m_backend, method=cfg.m_method)
                t_ms = time.perf_counter() - t0

                t0 = time.perf_counter()
                model = _regularize(model, cfg)
                t_reg = time.perf_counter() - t0

                e_times.append(t_e)
                mn_times.append(t_mn)
                ms_times.append(t_ms)
                reg_times.append(t_reg)

            max_change = _param_change(model, prev_mu, prev_gamma, prev_L)
            n_iter = i + 1
            if max_change < tol and i > 0:
                converged = True
                break

        total_time = time.perf_counter() - t_total
        ll = float(model.marginal_log_likelihood(X))

        return EMBenchResult(
            dist_name=cfg.dist_name,
            algorithm=cfg.algorithm,
            e_backend=cfg.e_backend,
            m_backend=cfg.m_backend,
            m_method=cfg.m_method,
            n_iter=n_iter,
            converged=converged,
            total_time=total_time,
            avg_e_step=float(np.mean(e_times)),
            avg_m_normal=float(np.mean(mn_times)),
            avg_m_sub=float(np.mean(ms_times)),
            avg_regularize=float(np.mean(reg_times)),
            final_ll=ll,
        )

    except Exception as e:
        return EMBenchResult(
            cfg.dist_name, cfg.algorithm, cfg.e_backend, cfg.m_backend,
            cfg.m_method, n_iter, False, time.perf_counter() - t_total,
            0, 0, 0, 0, float('nan'), error=str(e)[:60])


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_summary_table(results: list[EMBenchResult]):
    """Summary: convergence, total time, time/iter, final LL."""
    W = 120
    hdr("EM Benchmark — Summary", W)
    header = (
        f"  {'Dist':<6} {'Algo':<6} {'E':<5} {'M':<5} {'Method':<7} "
        f"{'Total':>8} {'Iters':>6} {'Per Iter':>10} {'Conv':>5} {'Final LL':>12}"
    )
    print(header)
    sep(W)

    prev_dist = None
    for r in results:
        if prev_dist is not None and r.dist_name != prev_dist:
            sep(W)
        prev_dist = r.dist_name

        if r.error:
            print(
                f"  {r.dist_name:<6} {r.algorithm:<6} {r.e_backend:<5} "
                f"{r.m_backend:<5} {r.m_method:<7} "
                f"{'ERROR':>8} {'-':>6} {'-':>10} {'N':>5} {r.error:>12}")
        else:
            t_per = fmt_time(r.total_time / max(r.n_iter, 1))
            conv = "Y" if r.converged else "N"
            print(
                f"  {r.dist_name:<6} {r.algorithm:<6} {r.e_backend:<5} "
                f"{r.m_backend:<5} {r.m_method:<7} "
                f"{fmt_time(r.total_time):>8} {r.n_iter:>6} "
                f"{t_per:>10} {conv:>5} {r.final_ll:>12.4f}")

    print(f"{'=' * W}")


def print_breakdown_table(results: list[EMBenchResult]):
    """Breakdown: average per-iteration sub-step times."""
    W = 120
    hdr("EM Benchmark — Per-Iteration Breakdown", W)
    header = (
        f"  {'Dist':<6} {'Algo':<6} {'E':<5} {'M':<5} {'Method':<7} "
        f"{'E-step':>10} {'M-norm':>10} {'M-sub':>10} {'Reg':>10} "
        f"{'%E':>5} {'%Mn':>5} {'%Ms':>5}"
    )
    print(header)
    sep(W)

    prev_dist = None
    for r in results:
        if r.error:
            continue
        if prev_dist is not None and r.dist_name != prev_dist:
            sep(W)
        prev_dist = r.dist_name

        total = r.avg_e_step + r.avg_m_normal + r.avg_m_sub + r.avg_regularize
        if total > 0:
            pct_e = 100 * r.avg_e_step / total
            pct_mn = 100 * r.avg_m_normal / total
            pct_ms = 100 * r.avg_m_sub / total
        else:
            pct_e = pct_mn = pct_ms = 0

        print(
            f"  {r.dist_name:<6} {r.algorithm:<6} {r.e_backend:<5} "
            f"{r.m_backend:<5} {r.m_method:<7} "
            f"{fmt_time(r.avg_e_step):>10} {fmt_time(r.avg_m_normal):>10} "
            f"{fmt_time(r.avg_m_sub):>10} {fmt_time(r.avg_regularize):>10} "
            f"{pct_e:>4.0f}% {pct_mn:>4.0f}% {pct_ms:>4.0f}%")

    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EM benchmark: all distributions × backends × methods")
    parser.add_argument("--n-stocks", type=int, default=20,
                        help="Number of stocks (default: 20)")
    parser.add_argument("--large", action="store_true",
                        help="Full S&P 500 data (all stocks, all history)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum EM iterations (default: 200)")
    parser.add_argument("--tol", type=float, default=1e-2,
                        help="Convergence tolerance (default: 1e-2)")
    parser.add_argument("--mcecm", action="store_true",
                        help="Include MCECM algorithm variants")
    parser.add_argument("--save", action="store_true",
                        help="Save results to benchmarks/results/")
    args = parser.parse_args()

    n_stocks = None if args.large else args.n_stocks

    print(f"\nnormix EM Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    try:
        X_raw = load_sp500_data(n_stocks)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)

    X = jnp.asarray(X_raw, dtype=jnp.float64)
    n, d = X.shape
    scale = "LARGE" if args.large else "standard"
    print(f"Data: {n} obs × {d} stocks ({scale})", flush=True)
    print(f"Settings: max_iter={args.max_iter}, tol={args.tol:.0e}, "
          f"mcecm={args.mcecm}", flush=True)

    configs = build_configs(include_mcecm=args.mcecm)
    results: list[EMBenchResult] = []

    for i, cfg in enumerate(configs, 1):
        tag = f"{cfg.dist_name} {cfg.algorithm} e={cfg.e_backend} m={cfg.m_backend}/{cfg.m_method}"
        print(f"  [{i:2d}/{len(configs)}] {tag:<45}", end="", flush=True)
        r = run_benchmark(cfg, X, args.max_iter, args.tol)
        results.append(r)
        if r.error:
            print(f"  ERROR: {r.error}", flush=True)
        else:
            print(f"  {fmt_time(r.total_time)}  ({r.n_iter} iters)", flush=True)

    print_summary_table(results)
    print_breakdown_table(results)

    if args.save:
        data = {
            "benchmark": "em_mixture",
            "args": {
                "n_stocks": d, "n_obs": n, "max_iter": args.max_iter,
                "tol": args.tol, "mcecm": args.mcecm, "large": args.large,
            },
            "results": [asdict(r) for r in results],
        }
        # dataclass field dist_cls is not JSON-serializable; drop it
        for row in data["results"]:
            row.pop("dist_cls", None)
        path = save_result("em_mixture", data)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
