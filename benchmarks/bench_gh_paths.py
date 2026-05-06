"""
GH EM path comparison benchmark — Phase 6 closeout.

For each ``(n, d)`` regime, runs four GH fitting paths on the same synthetic
data and reports total wall time, per-iteration time, and final
log-likelihood:

    1. Batch EM,         CPU E + CPU M   (newton)
    2. Batch EM,         JAX E + JAX M   (newton, lax.scan, JIT-warm)
    3. Incremental EM,   JAX E + CPU M   (newton, Robbins-Monro)
    4. Incremental EM,   JAX E + JAX M   (newton, lax.scan, Robbins-Monro)

The benchmark is meant to answer Phase-6 questions about expected performance
regimes:

    - small ``n``, small ``d``: CPU often wins because Python/JAX dispatch
      and tiny GIG solver kernels dominate over the actual arithmetic.
    - moderate-to-large batch E-step over many observations: JAX/JAX may
      win because the E-step amortises kernel launch cost over many obs.
    - tiny GIG M-step solves: CPU SciPy or the cached JAX solver from
      Phase 4 are both fast; CPU often wins for ``d <= 3`` and small ``n``.

Each timing entry uses two runs: the first pays JAX compilation / SciPy
import cost, the second is the cached-call number that an outer training
loop would observe.

Usage
-----
    uv run python benchmarks/bench_gh_paths.py
    uv run python benchmarks/bench_gh_paths.py --save
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass, asdict

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.utils import save_result, fmt_time, hdr, sep

from normix import GeneralizedHyperbolic
from normix.fitting.em import BatchEMFitter, IncrementalEMFitter
from normix.fitting.eta_rules import RobbinsMonroUpdate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Regime:
    name: str
    n: int
    d: int

REGIMES = [
    Regime("small",  n=200,    d=2),
    Regime("medium", n=2000,   d=5),
    Regime("large",  n=10000,  d=10),
]

BATCH_MAX_ITER = 10
INCREMENTAL_STEPS = 20
INCREMENTAL_BATCH_FRAC = 0.25
TAU0 = 10.0


@dataclass
class PathResult:
    regime: str
    n: int
    d: int
    path: str            # short label
    e_backend: str
    m_backend: str
    fit_first_s: float
    fit_cached_s: float
    per_iter_first_s: float
    per_iter_cached_s: float
    n_iter: int
    final_ll: float
    error: str = ""


# ---------------------------------------------------------------------------
# Data and initial model
# ---------------------------------------------------------------------------

def _make_data(regime: Regime) -> jax.Array:
    """Heavy-tailed-ish synthetic data; same seed across paths for fairness."""
    key = jax.random.PRNGKey(2026)
    return jax.random.normal(key, (regime.n, regime.d), dtype=jnp.float64) * 0.3


def _make_init_model(X: jax.Array) -> GeneralizedHyperbolic:
    d = X.shape[1]
    mu = jnp.mean(X, axis=0)
    sigma = jnp.cov(X.T) + 1e-4 * jnp.eye(d)
    return GeneralizedHyperbolic.from_classical(
        mu=mu, gamma=jnp.zeros(d), sigma=sigma, p=-0.5, a=2.0, b=1.0)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _block(x):
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for v in x:
            _block(v)


def _block_em_result(res) -> None:
    leaves = jax.tree.leaves(res.model)
    for leaf in leaves:
        _block(leaf)
    if res.param_changes is not None:
        _block(res.param_changes)


def _measure_pair(make_call):
    """Return ``(t_first_s, t_cached_s, last_result)``.

    Runs ``make_call`` twice; first call pays compile/import overhead, second
    call is JIT-warm. Timing blocks on JAX async dispatch.
    """
    t0 = time.perf_counter()
    res = make_call()
    _block_em_result(res)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = make_call()
    _block_em_result(res)
    t_cached = time.perf_counter() - t0

    return t_first, t_cached, res


# ---------------------------------------------------------------------------
# Fit functions
# ---------------------------------------------------------------------------

def _batch_call(e_backend: str, m_backend: str):
    """Closure factory for batch GH EM."""
    def fit(model, X):
        fitter = BatchEMFitter(
            algorithm='em',
            max_iter=BATCH_MAX_ITER,
            tol=0.0,                 # disable early exit so all paths do same work
            verbose=0,
            regularization='det_sigma_one',
            e_step_backend=e_backend,
            m_step_backend=m_backend,
            m_step_method='newton',
        )
        return fitter.fit(model, X)
    return fit


def _incremental_call(e_backend: str, m_backend: str, batch_size: int, key):
    """Closure factory for Robbins-Monro incremental GH EM."""
    def fit(model, X):
        fitter = IncrementalEMFitter(
            eta_update=RobbinsMonroUpdate(tau0=TAU0),
            batch_size=batch_size,
            max_steps=INCREMENTAL_STEPS,
            inner_iter=1,
            verbose=0,
            regularization='det_sigma_one',
            e_step_backend=e_backend,
            m_step_backend=m_backend,
            m_step_method='newton',
        )
        return fitter.fit(model, X, key=key)
    return fit


# ---------------------------------------------------------------------------
# Run a single regime
# ---------------------------------------------------------------------------

def _path_iters(path: str) -> int:
    return BATCH_MAX_ITER if path.startswith("batch") else INCREMENTAL_STEPS


def run_regime(regime: Regime) -> list[PathResult]:
    X = _make_data(regime)
    init_model = _make_init_model(X)
    rng = jax.random.PRNGKey(0)
    bs = max(1, int(regime.n * INCREMENTAL_BATCH_FRAC))

    paths = [
        ("batch_cpu_cpu",   "cpu", "cpu", _batch_call("cpu", "cpu")),
        ("batch_jax_jax",   "jax", "jax", _batch_call("jax", "jax")),
        ("incr_jax_cpu",    "jax", "cpu",
         _incremental_call("jax", "cpu", bs, rng)),
        ("incr_jax_jax",    "jax", "jax",
         _incremental_call("jax", "jax", bs, rng)),
    ]

    results: list[PathResult] = []
    for label, eb, mb, fit_fn in paths:
        try:
            t_first, t_cached, res = _measure_pair(
                lambda: fit_fn(init_model, X))
            n_iter = res.n_iter or _path_iters(label)
            ll = float(res.model.marginal_log_likelihood(X))
            results.append(PathResult(
                regime=regime.name, n=regime.n, d=regime.d,
                path=label, e_backend=eb, m_backend=mb,
                fit_first_s=t_first,
                fit_cached_s=t_cached,
                per_iter_first_s=t_first / max(n_iter, 1),
                per_iter_cached_s=t_cached / max(n_iter, 1),
                n_iter=n_iter,
                final_ll=ll,
            ))
        except Exception as e:
            results.append(PathResult(
                regime=regime.name, n=regime.n, d=regime.d,
                path=label, e_backend=eb, m_backend=mb,
                fit_first_s=0.0, fit_cached_s=0.0,
                per_iter_first_s=0.0, per_iter_cached_s=0.0,
                n_iter=0, final_ll=float('nan'),
                error=f"{type(e).__name__}: {e!s:.80}",
            ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(rows: list[PathResult]):
    W = 120
    hdr("GH EM path comparison — first vs cached fit", W)
    print(
        f"  {'regime':<7} {'n':>6} {'d':>3} {'path':<16} "
        f"{'iters':>6} {'fit 1st':>10} {'fit cached':>12} "
        f"{'per-iter 1st':>14} {'per-iter cached':>17} "
        f"{'final LL':>12}"
    )
    sep(W)
    prev_regime = None
    for r in rows:
        if prev_regime is not None and r.regime != prev_regime:
            sep(W)
        prev_regime = r.regime
        if r.error:
            print(
                f"  {r.regime:<7} {r.n:>6} {r.d:>3} {r.path:<16} "
                f"{'-':>6} {'ERROR':>10} {'':>12} {'':>14} {'':>17} "
                f"{r.error:>12}"
            )
            continue
        print(
            f"  {r.regime:<7} {r.n:>6} {r.d:>3} {r.path:<16} "
            f"{r.n_iter:>6} "
            f"{fmt_time(r.fit_first_s):>10} "
            f"{fmt_time(r.fit_cached_s):>12} "
            f"{fmt_time(r.per_iter_first_s):>14} "
            f"{fmt_time(r.per_iter_cached_s):>17} "
            f"{r.final_ll:>12.4f}"
        )
    print(f"{'=' * W}")


def print_regime_winners(rows: list[PathResult]):
    """For each regime, name the cached-fit-time winner."""
    W = 90
    hdr("Cached-fit winner per regime (lower is better)", W)
    by_regime: dict[str, list[PathResult]] = {}
    for r in rows:
        if r.error:
            continue
        by_regime.setdefault(r.regime, []).append(r)
    for regime, group in by_regime.items():
        if not group:
            continue
        winner = min(group, key=lambda r: r.fit_cached_s)
        sample = group[0]
        print(
            f"  {regime:<7} (n={sample.n}, d={sample.d}): "
            f"{winner.path:<16} "
            f"cached={fmt_time(winner.fit_cached_s)} "
            f"per-iter={fmt_time(winner.per_iter_cached_s)}"
        )
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GH EM path comparison benchmark (Phase 6)")
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to benchmarks/results/")
    parser.add_argument(
        "--only-regime", type=str, default=None,
        help="Restrict to one regime: small | medium | large")
    args = parser.parse_args()

    print("\nnormix GH EM path benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    regimes = REGIMES
    if args.only_regime:
        regimes = [r for r in REGIMES if r.name == args.only_regime]
        if not regimes:
            print(f"  unknown regime {args.only_regime!r}; "
                  f"available: {', '.join(r.name for r in REGIMES)}")
            sys.exit(1)

    all_rows: list[PathResult] = []
    for regime in regimes:
        print(
            f"\nRegime {regime.name}: n={regime.n}, d={regime.d} ...",
            flush=True,
        )
        rows = run_regime(regime)
        for r in rows:
            tag = f"{r.path:<16}"
            if r.error:
                print(f"  {tag}  ERROR: {r.error}", flush=True)
            else:
                print(
                    f"  {tag}  cached={fmt_time(r.fit_cached_s)}  "
                    f"per-iter={fmt_time(r.per_iter_cached_s)}  "
                    f"LL={r.final_ll:.4f}",
                    flush=True,
                )
        all_rows.extend(rows)

    print_table(all_rows)
    print_regime_winners(all_rows)

    if args.save:
        data = {
            "benchmark": "gh_paths",
            "config": {
                "regimes": [asdict(r) for r in regimes],
                "batch_max_iter": BATCH_MAX_ITER,
                "incremental_max_steps": INCREMENTAL_STEPS,
                "incremental_batch_frac": INCREMENTAL_BATCH_FRAC,
                "tau0": TAU0,
            },
            "rows": [asdict(r) for r in all_rows],
        }
        path = save_result("gh_paths", data)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
