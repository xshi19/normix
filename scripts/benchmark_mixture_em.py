"""
Mixture model EM benchmark — all distributions, all backend combinations.

Runs EM to convergence (tol=1e-2) for each combination and reports a table:

    Distribution | Step | Backend | Method | Total Time | Iterations | Time/Iter

Baseline setup: E-step=cpu, M-step=cpu/lbfgs.
Each row varies exactly one step from the baseline.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/benchmark_mixture_em.py [--n-stocks N]
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd

from normix import (
    GeneralizedHyperbolic,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)
from normix.fitting.em import BatchEMFitter, EMResult


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(n_stocks: int) -> np.ndarray:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"SP500 data not found at {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
    return df.values.astype(np.float64)[:, :n_stocks]


# ---------------------------------------------------------------------------
# Distribution factories
# ---------------------------------------------------------------------------

def make_model(cls, X):
    """Create default-initialised model from data."""
    return cls.default_init(X)


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
class BenchConfig:
    """One benchmark row configuration."""
    dist_name: str
    dist_cls: type
    step_varied: str        # "E-step", "M-step", or "baseline"
    e_backend: str
    m_backend: str
    m_method: str
    label: str


def build_configs():
    """Build all benchmark configurations.

    Baseline: E=cpu, M=cpu/lbfgs (default setup).
    Vary E-step: E=jax (rest stays baseline).
    Vary M-step backend: M=jax/newton (E stays baseline=cpu).
    Vary M-step method:  M=cpu/newton (only meaningful for GH where
        the subordinator M-step involves a GIG solver).
    """
    configs = []

    for dist_name, dist_cls in DISTRIBUTIONS:
        configs.append(BenchConfig(
            dist_name=dist_name, dist_cls=dist_cls,
            step_varied="both",
            e_backend="cpu", m_backend="cpu", m_method="lbfgs",
            label="cpu/lbfgs (baseline)",
        ))

        configs.append(BenchConfig(
            dist_name=dist_name, dist_cls=dist_cls,
            step_varied="E-step",
            e_backend="jax", m_backend="cpu", m_method="lbfgs",
            label="jax E",
        ))

        configs.append(BenchConfig(
            dist_name=dist_name, dist_cls=dist_cls,
            step_varied="M-step",
            e_backend="cpu", m_backend="jax", m_method="newton",
            label="jax/newton M",
        ))

        if dist_name == "GH":
            configs.append(BenchConfig(
                dist_name=dist_name, dist_cls=dist_cls,
                step_varied="M-step",
                e_backend="cpu", m_backend="cpu", m_method="newton",
                label="cpu/newton M",
            ))

    return configs


# ---------------------------------------------------------------------------
# Run single benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    dist_name: str
    step_varied: str
    e_backend: str
    m_backend: str
    m_method: str
    total_time: float
    n_iter: int
    time_per_iter: float
    converged: bool
    final_ll: float
    error: Optional[str] = None


def run_single(cfg: BenchConfig, X: jax.Array, max_iter: int, tol: float) -> BenchResult:
    """Run EM to convergence for one configuration."""

    regularization = 'det_sigma_one' if cfg.dist_name == 'GH' else 'none'

    try:
        model = make_model(cfg.dist_cls, X)

        fitter = BatchEMFitter(
            max_iter=max_iter,
            tol=tol,
            e_step_backend=cfg.e_backend,
            m_step_backend=cfg.m_backend,
            m_step_method=cfg.m_method,
            regularization=regularization,
            verbose=0,
        )

        result = fitter.fit(model, X)
        ll = float(result.model.marginal_log_likelihood(X))
        t_per_iter = result.elapsed_time / max(result.n_iter, 1)

        return BenchResult(
            dist_name=cfg.dist_name,
            step_varied=cfg.step_varied,
            e_backend=cfg.e_backend,
            m_backend=cfg.m_backend,
            m_method=cfg.m_method,
            total_time=result.elapsed_time,
            n_iter=result.n_iter,
            time_per_iter=t_per_iter,
            converged=result.converged,
            final_ll=ll,
        )

    except Exception as e:
        return BenchResult(
            dist_name=cfg.dist_name,
            step_varied=cfg.step_varied,
            e_backend=cfg.e_backend,
            m_backend=cfg.m_backend,
            m_method=cfg.m_method,
            total_time=0.0,
            n_iter=0,
            time_per_iter=0.0,
            converged=False,
            final_ll=float('nan'),
            error=str(e)[:60],
        )


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table(results: list[BenchResult]):
    """Print results as a formatted table.

    Columns:
      Distribution | Step Varied | E Backend | M Backend | M Method |
      Total Time   | Iterations  | Time/Iter | Conv      | Final LL
    """
    W = 125

    header = (
        f"  {'Dist':<6} {'Step Varied':<12} {'E Bknd':<7} {'M Bknd':<7} {'M Method':<9} "
        f"{'Total':>8} {'Iters':>6} {'Per Iter':>10} {'Conv':>5} {'Final LL':>11}"
    )

    print(f"\n{'=' * W}")
    print("  Mixture Model EM Benchmark")
    print(f"  Baseline: E=cpu, M=cpu/lbfgs. Each non-baseline row changes one step.")
    print(f"{'=' * W}")
    print(header)
    print(f"  {'-' * (W - 2)}")

    prev_dist = None
    for r in results:
        if prev_dist is not None and r.dist_name != prev_dist:
            print(f"  {'-' * (W - 2)}")
        prev_dist = r.dist_name

        if r.error:
            print(
                f"  {r.dist_name:<6} {r.step_varied:<12} {r.e_backend:<7} "
                f"{r.m_backend:<7} {r.m_method:<9} "
                f"{'ERROR':>8} {'-':>6} {'-':>10} "
                f"{'N':>5} {r.error:>11}"
            )
        else:
            conv_str = "Y" if r.converged else "N"
            if r.time_per_iter >= 1.0:
                t_per = f"{r.time_per_iter:.2f}s"
            elif r.time_per_iter >= 0.01:
                t_per = f"{r.time_per_iter:.3f}s"
            else:
                t_per = f"{r.time_per_iter*1e3:.1f}ms"
            print(
                f"  {r.dist_name:<6} {r.step_varied:<12} {r.e_backend:<7} "
                f"{r.m_backend:<7} {r.m_method:<9} "
                f"{r.total_time:>7.2f}s {r.n_iter:>6} "
                f"{t_per:>10} {conv_str:>5} {r.final_ll:>11.4f}"
            )

    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mixture model EM benchmark — all distributions × backends")
    parser.add_argument("--n-stocks", type=int, default=10,
                        help="Number of stocks from SP500 (default: 10)")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum EM iterations (default: 200)")
    parser.add_argument("--tol", type=float, default=1e-2,
                        help="Convergence tolerance (default: 1e-2)")
    args = parser.parse_args()

    print(f"\nnormix Mixture Model EM Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    try:
        X_raw = load_data(args.n_stocks)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        print("  Run: python scripts/download_sp500_data.py")
        sys.exit(1)

    X = jnp.asarray(X_raw, dtype=jnp.float64)
    n, d = X.shape
    print(f"Data: {n} observations × {d} stocks", flush=True)
    print(f"Settings: max_iter={args.max_iter}, tol={args.tol:.0e}", flush=True)

    configs = build_configs()
    results = []

    for i, cfg in enumerate(configs, 1):
        desc = f"{cfg.dist_name} {cfg.step_varied} {cfg.label}"
        print(f"  [{i:2d}/{len(configs)}] {desc:<40}", end="", flush=True)
        r = run_single(cfg, X, args.max_iter, args.tol)
        results.append(r)
        if r.error:
            print(f"  ERROR: {r.error}", flush=True)
        else:
            print(f"  {r.total_time:.2f}s  ({r.n_iter} iters)", flush=True)

    print_table(results)


if __name__ == "__main__":
    main()
