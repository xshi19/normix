"""Diagnostic: track the M-step denominator D = 1 - eta2*eta3 across EM iterations.

Background
----------
The closed-form normal-parameter M-step divides by ``D = 1 - E[1/Y]*E[Y]``
(``JointNormalMixture._mstep_normal_params``). By Cauchy-Schwarz this batch
quantity is always ``D <= 0`` (equality only in the degenerate / Gaussian
limit where the subordinator Y is constant). The guard floors at
``-SAFE_DENOMINATOR`` (sign-preserving; see finding B1 in
``dev-notes/plans/em_robustness_followups.md``).

This script replicates the notebook fit configuration and records D at every
EM iteration, reporting when |D| hits the floor or D > 0 from roundoff.

Usage
-----
    uv run python scripts/check_mstep_denominator.py
    uv run python scripts/check_mstep_denominator.py --families VG --max-iter 100
    uv run python scripts/check_mstep_denominator.py --max-stocks 50
"""
from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from normix import (  # noqa: E402
    GeneralizedHyperbolic,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)
from normix.fitting.em import _param_change  # noqa: E402
from normix.utils.constants import SAFE_DENOMINATOR  # noqa: E402

FAMILIES = {
    "VG": VarianceGamma,
    "NInvG": NormalInverseGamma,
    "NIG": NormalInverseGaussian,
    "GH": GeneralizedHyperbolic,
}


def subordinator_index(model) -> float:
    """A scalar 'degeneracy' indicator (the GIG-style index/shape).

    Large values => subordinator near-deterministic => near-Gaussian limit
    => D -> 0. Returns NaN if no obvious scalar shape is exposed.
    """
    for attr in ("alpha", "p", "lam"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            return float(jnp.ravel(jnp.asarray(val))[0])
    return float("nan")


def run_instrumented_em(
    cls,
    X: jax.Array,
    *,
    max_iter: int,
    tol: float,
    m_step_method: str,
) -> dict:
    """Replicate BatchEMFitter._em_step manually, recording D each iteration."""
    model = cls.default_init(X)

    rows = []
    n_iter = 0
    for i in range(max_iter):
        prev_params = model.em_convergence_params()

        eta = model.e_step(X, backend="jax")
        D = float(1.0 - eta.E_inv_Y * eta.E_Y)
        eta2 = float(eta.E_inv_Y)
        eta3 = float(eta.E_Y)
        idx = subordinator_index(model)

        # Replicate the post-B1 guard: safe_D = -max(|D|, floor).
        floored = abs(D) <= SAFE_DENOMINATOR
        positive = D > 0.0

        model = model.m_step(eta, backend="cpu", method=m_step_method)
        max_change = float(_param_change(model.em_convergence_params(), prev_params))

        rows.append(
            dict(
                it=i,
                D=D,
                eta2=eta2,
                eta3=eta3,
                prod=eta2 * eta3,
                index=idx,
                clipped=floored,
                positive=positive,
                max_change=max_change,
            )
        )
        n_iter = i + 1
        if max_change < tol and i > 0:
            break

    return {"rows": rows, "n_iter": n_iter, "model": model}


def summarize(name: str, out: dict) -> None:
    rows = out["rows"]
    Ds = np.array([r["D"] for r in rows])
    abs_min = np.min(np.abs(Ds))
    n_floor = sum(r["clipped"] for r in rows)
    n_pos = sum(r["positive"] for r in rows)

    print(f"\n{'='*78}")
    print(f"  {name}: {out['n_iter']} EM iterations")
    print(f"{'='*78}")
    print(f"  {'it':>3}  {'D':>14}  {'eta2*eta3':>14}  {'index':>10}  "
          f"{'|Δparams|':>11}  note")
    print(f"  {'-'*3}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*11}  ----")
    for r in rows:
        note = ""
        if r["positive"]:
            note = "D>0 (roundoff)"
        elif r["clipped"]:
            note = "|D| floored"
        print(f"  {r['it']:>3}  {r['D']:>14.6e}  {r['prod']:>14.10f}  "
              f"{r['index']:>10.3f}  {r['max_change']:>11.3e}  {note}")
    print(f"  {'-'*60}")
    print(f"  min |D| over run : {abs_min:.6e}   (floor magnitude = {SAFE_DENOMINATOR:.0e})")
    print(f"  iterations with |D| <= floor : {n_floor}")
    print(f"  iterations with D > 0 (roundoff)  : {n_pos}")
    verdict = ("Healthy: D well below floor (no near-Gaussian limit)."
               if (n_floor == 0 and n_pos == 0)
               else "Near-Gaussian limit: |D| at/below floor or D>0 from roundoff.")
    print(f"  NOTE: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/sp500_returns.csv")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--max-stocks", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    log_returns = pd.read_csv(args.data, index_col="Date", parse_dates=True)
    tickers = sorted(log_returns.columns.tolist())
    if args.max_stocks is not None:
        tickers = tickers[: args.max_stocks]
    data = log_returns[tickers].values
    n_train = len(data) // 2
    X = jnp.asarray(data[:n_train], dtype=jnp.float64)
    print(f"Data: {X.shape[0]} train samples x {X.shape[1]} stocks "
          f"(d/2+1 = {X.shape[1] / 2 + 1:.0f})")

    for name in args.families:
        method = "lbfgs" if name == "GH" else "newton"
        out = run_instrumented_em(
            FAMILIES[name], X, max_iter=args.max_iter, tol=args.tol,
            m_step_method=method,
        )
        summarize(name, out)


if __name__ == "__main__":
    main()
