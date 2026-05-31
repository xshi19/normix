"""Marimo notebook: VarianceGamma EM diverges to NaN on heavy-tailed data.

Run with:

    uv run marimo edit dev-notes/investigations/variance_gamma_em_nan.py

or, headless:

    uv run marimo run dev-notes/investigations/variance_gamma_em_nan.py

Summary of the finding is in ``variance_gamma_em_nan.md``.
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # VarianceGamma EM diverges to NaN on a heavy-tailed series

    **Claim under test** (from the docs-refactor plan): *"VarianceGamma's light
    tails diverge to `nan` on the kurtosis≈20 index series."*

    This notebook shows that explanation is **wrong**. The VG fit does not fail
    because its tails are too light — it fits competitively and the
    log-likelihood is still *improving* right up to the blow-up. The real cause
    is that the EM drives the **Gamma subordinator shape $\alpha$ below 1**,
    where the mixing density piles mass at $Y \approx 0$, the posterior
    inverse-variance weights $E[1/Y \mid x]$ explode, and the covariance M-step
    produces a `nan`.
    """)
    return


@app.cell
def _():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy import stats

    from normix import (
        VarianceGamma,
        NormalInverseGaussian,
        GeneralizedHyperbolic,
        Gamma,
    )

    # Equal-weight index proxy from the checked-in S&P 500 return panel.
    data_path = Path(__file__).resolve().parents[2] / "data" / "sp500_returns.csv"
    panel = pd.read_csv(data_path, index_col="Date", parse_dates=True).dropna(
        axis=1, how="any"
    )
    index_ret = panel.mean(axis=1).values.astype(np.float64)
    X = jnp.asarray(index_ret[: len(index_ret) // 2].reshape(-1, 1))
    return (
        Gamma,
        GeneralizedHyperbolic,
        NormalInverseGaussian,
        VarianceGamma,
        X,
        index_ret,
        jnp,
        np,
        stats,
    )


@app.cell
def _(index_ret, mo, stats):
    mo.md(
        f"""
        ## The data

        An equal-weighted index of the S&P 500 panel: **{len(index_ret)} daily
        returns**, excess kurtosis **{stats.kurtosis(index_ret):.1f}**, skew
        **{stats.skew(index_ret):.2f}** — strongly heavy-tailed and mildly
        left-skewed.
        """
    )
    return


@app.cell
def _(VarianceGamma, X, np):
    # Fit VG and inspect the log-likelihood history (verbose=1 records it).
    vg_init = VarianceGamma.default_init(X)
    res = vg_init.fit(X, max_iter=100, tol=1e-3, e_step_backend="cpu", verbose=1)
    ll = np.asarray(res.log_likelihoods)
    first_nan = int(np.argmax(np.isnan(ll))) if np.isnan(ll).any() else -1
    summary = {
        "converged": res.converged,
        "n_iter": res.n_iter,
        "ll_improves_before_blowup": ll[:first_nan].tolist()[:6] + ["..."],
        "first_nan_iteration": first_nan,
    }
    summary
    return (vg_init,)


@app.cell
def _(mo):
    mo.md(r"""
    The likelihood climbs ($3.14 \to 3.21$) for ~16 iterations, then every
    subsequent value is `nan`. The fit was *succeeding*. Let us trace the
    parameters per iteration to see what breaks.
    """)
    return


@app.cell
def _(X, np, vg_init):
    import pandas as _pd

    # Single-step EM to trace parameters as they approach the blow-up.
    _m = vg_init
    _trace_rows = []
    for _it in range(20):
        _m = _m.fit(X, max_iter=1, tol=0.0, e_step_backend="cpu").model
        _mll = float(_m.marginal_log_likelihood(X))
        _trace_rows.append(
            {
                "iter": _it + 1,
                "alpha": float(_m.joint.alpha),
                "sigma": float(_m.joint.L_Sigma[0, 0]) ** 2,
                "mll": _mll,
            }
        )
        if not np.isfinite(_mll):
            break
    trace = _pd.DataFrame(_trace_rows)
    trace
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two things happen together:

    - **$\alpha$ falls monotonically** through $1.0$ (around iteration 6) and
      keeps going to $\approx 0.69$.
    - **$\sigma$ turns `nan` at iteration 19**, while $\alpha$ is still a finite
      $\approx 0.69$. So the covariance update is the proximate failure.

    ## Root cause: the Gamma inverse moment $E[1/Y]$ diverges for $\alpha \le 1$

    The VG subordinator is $Y \sim \mathrm{Gamma}(\alpha, \beta)$, with

    $$
    E[1/Y] = \frac{\beta}{\alpha - 1}, \qquad \text{finite only for } \alpha > 1.
    $$

    The EM covariance M-step weights each observation by the posterior
    inverse-variance $E[1/Y \mid x]$. As $\alpha$ drops below 1 the mixing
    density concentrates near $Y = 0$, those weights grow without bound, and
    the weighted covariance eventually overflows to `nan`.
    """)
    return


@app.cell
def _(Gamma, jnp):
    # E[1/Y] for a Gamma subordinator at decreasing shape alpha.
    import pandas as _pd2

    _inv_rows = []
    for _a in [2.0, 1.5, 1.05, 0.95, 0.69]:
        _g = Gamma(alpha=jnp.array(_a), beta=jnp.array(1.0))
        _y = _g.rvs(2_000_000, seed=0)
        _inv_rows.append(
            {
                "alpha": _a,
                "analytic_E[1/Y]": 1.0 / (_a - 1) if _a > 1 else float("inf"),
                "MC_E[1/Y]": float(jnp.mean(1.0 / _y)),
            }
        )
    _pd2.DataFrame(_inv_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $E[1/Y]$ is finite and well-behaved for $\alpha > 1$, but the Monte-Carlo
    estimate explodes once $\alpha < 1$ (the analytic value is $+\infty$). This
    is exactly the regime the EM walks into.

    ## It is not "light tails": NIG and GH fit the same data fine
    """)
    return


@app.cell
def _(GeneralizedHyperbolic, NormalInverseGaussian, X):
    import pandas as _pd3

    nig = NormalInverseGaussian.default_init(X).fit(
        X, max_iter=80, tol=1e-3, e_step_backend="cpu"
    ).model
    gh = GeneralizedHyperbolic.default_init(X).fit(
        X, max_iter=80, tol=1e-3, e_step_backend="cpu"
    ).model
    _pd3.DataFrame(
        [
            {"model": "NormalInverseGaussian", "train_mll": float(nig.marginal_log_likelihood(X))},
            {"model": "GeneralizedHyperbolic", "train_mll": float(gh.marginal_log_likelihood(X))},
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    NIG and GH converge without trouble and reach a mean log-likelihood close to
    VG's last *finite* value ($\approx 3.21$). VG is **not** uncompetitive — its
    EM path is numerically fragile in the $\alpha < 1$ corner.

    ## Practical mitigations

    - **Stop earlier.** The likelihood is flat by iteration ~16; a tighter `tol`
      or smaller `max_iter` returns the good finite iterate before the blow-up.
    - **Use NIG or GH** on strongly heavy-tailed series; their subordinators
      (Inverse Gaussian / GIG) do not have the $\alpha \le 1$ inverse-moment
      singularity in the same way.

    ## Suggested library fix

    Guard the VG covariance M-step against the degenerate mixing regime — e.g.
    floor the Gamma shape at $\alpha > 1$ during the update, or clamp the
    posterior $E[1/Y \mid x]$ weights — so the fit stalls at the converged
    iterate instead of emitting `nan`. Tracked for the `em` / `variance_gamma`
    path.
    """)
    return


if __name__ == "__main__":
    app.run()
