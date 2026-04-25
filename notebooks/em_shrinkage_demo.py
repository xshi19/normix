"""Demo: Phase-1 EM API + Phase-2 shrinkage combinator on S&P 500 returns.

A marimo notebook companion to
`docs/design/em_covariance_extensions.md` (Phase 1, Phase 2) and
`docs/design/penalised_em.md`. Run with::

    uv run marimo edit notebooks/em_shrinkage_demo.py

or, for a static preview::

    uv run marimo run notebooks/em_shrinkage_demo.py
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(
    width="medium",
    app_title="normix — Phase 1 + 2 EM Demo (S&P 500)",
)


@app.cell
def _intro(mo):
    mo.md(r"""
    # normix EM extensions on S&P 500 returns

    This notebook demonstrates the Phase-1 (EM API generalisation) and
    Phase-2 (shrinkage combinator) work landed in
    `docs/design/em_covariance_extensions.md`. We use a high-dimensional,
    small-$n$ slice of S&P 500 daily returns — exactly the regime where
    the sample covariance is ill-conditioned and shrinkage is meant to
    help.

    **Outline**

    1. Data: a small training window (`n ≲ d`) over $d$ S&P 500 stocks.
    2. **Phase 1** — new EM API:
        * `NormalMixtureEta` in theory order $(s_1, …, s_6)$,
        * `EtaUpdateRule` / `AffineRule` two-layer abstraction,
        * generalised `affine_combine` (scalar / pytree / callable weights),
        * `MarginalMixture.em_convergence_params()`.
    3. **Phase 2** — `Shrinkage` combinator:
        * scalar $\tau$ with an isotropic prior,
        * per-field $\tau$ to shrink only $\Sigma$,
        * composition with `RobbinsMonroUpdate` (incremental EM),
        * test log-likelihood vs $\tau$ sweep.
    4. Diagnostic: condition number of $\Sigma$ and out-of-sample LL.

    References: `docs/theory/shrinkage.rst`,
    `docs/design/penalised_em.md`.
    """)
    return


@app.cell
def _imports():
    import os
    import time
    import warnings

    import numpy as np
    import pandas as pd
    import jax
    import jax.numpy as jnp
    import marimo as mo

    warnings.filterwarnings("ignore")
    jax.config.update("jax_enable_x64", True)

    from normix import (
        VarianceGamma,
        NormalInverseGamma,
        NormalInverseGaussian,
        GeneralizedHyperbolic,
        MultivariateNormal,
    )
    from normix.fitting import (
        BatchEMFitter,
        IncrementalEMFitter,
        EMResult,
        NormalMixtureEta,
        affine_combine,
        EtaUpdateRule,
        AffineRule,
        IdentityUpdate,
        RobbinsMonroUpdate,
        EWMAUpdate,
        SampleWeightedUpdate,
        AffineUpdate,
        Shrinkage,
        eta0_from_model,
        eta0_isotropic,
        eta0_diagonal,
        eta0_with_sigma,
    )
    from normix.mixtures.marginal import MarginalMixture, NormalMixture

    return (
        AffineRule,
        AffineUpdate,
        BatchEMFitter,
        EWMAUpdate,
        IdentityUpdate,
        IncrementalEMFitter,
        MarginalMixture,
        NormalMixture,
        NormalMixtureEta,
        RobbinsMonroUpdate,
        SampleWeightedUpdate,
        Shrinkage,
        VarianceGamma,
        affine_combine,
        eta0_diagonal,
        eta0_isotropic,
        jax,
        jnp,
        mo,
        np,
        os,
        pd,
        time,
    )


@app.cell
def _data_header(mo):
    mo.md(r"""
    ## 1. Load S&P 500 returns and pick a small-$n$ slice

    We deliberately use a **short training window** and a **moderately
    large dimension** so that the sample covariance is poorly
    conditioned. This is where shrinkage helps; with $n \gg d$ the
    unshrunk EM is already excellent.
    """)
    return


@app.cell
def _load_data(mo, os, pd):
    _candidates = [
        os.path.join("data", "sp500_returns.csv"),
        os.path.join("..", "data", "sp500_returns.csv"),
    ]
    data_path = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Could not find data/sp500_returns.csv. "
            "Run 'python scripts/download_sp500_data.py' from the repo root."
        )

    log_returns = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    mo.md(
        f"Loaded `{data_path}` — shape **{log_returns.shape}**, "
        f"range **{log_returns.index[0].date()}** to "
        f"**{log_returns.index[-1].date()}**."
    )
    return (log_returns,)


@app.cell
def _make_slice(jnp, log_returns, mo, np):
    d = 468
    n_train = 1500
    _seed = 0

    _rng = np.random.default_rng(_seed)
    _all_tickers = sorted(log_returns.columns.tolist())
    tickers = sorted(_rng.choice(_all_tickers, size=d, replace=False).tolist())

    _n_total = len(log_returns)
    if n_train + 250 > _n_total:
        n_train = _n_total - 250

    _train_df = log_returns[tickers].iloc[:n_train]
    _test_df = log_returns[tickers].iloc[n_train: n_train + 250]

    X_train = jnp.asarray(_train_df.values, dtype=jnp.float64)
    X_test = jnp.asarray(_test_df.values, dtype=jnp.float64)

    mo.md(
        f"- **d** = {d}, **n_train** = {X_train.shape[0]}, "
        f"**n_test** = {X_test.shape[0]}\n\n"
        f"- training: {_train_df.index[0].date()} → "
        f"{_train_df.index[-1].date()}\n\n"
        f"- testing : {_test_df.index[0].date()} → "
        f"{_test_df.index[-1].date()}\n\n"
        f"- ratio n/d = **{X_train.shape[0] / d:.2f}**"
    )
    return X_test, X_train, d


@app.cell
def _phase1_header(mo):
    mo.md(r"""
    ## 2. Phase 1 — generalised EM API

    Phase 1 reorders `NormalMixtureEta` to **theory order**
    $(s_1, …, s_6)$, introduces the two-layer rule abstraction
    (`EtaUpdateRule` → `AffineRule`), generalises `affine_combine` to
    accept scalar / pytree / callable weights, and adds
    `MarginalMixture.em_convergence_params()` so the fitter no longer
    hard-codes `(mu, gamma, L_Sigma)`.

    Below: each piece in the smallest possible cell.
    """)
    return


@app.cell
def _phase1_eta_fields(NormalMixtureEta, VarianceGamma, X_train, mo):
    model = VarianceGamma.default_init(X_train)
    eta = model.compute_eta_from_model()

    _field_summary = "\n".join(
        f"- `{name:12s}` shape={tuple(getattr(eta, name).shape) or '()'} "
        for name in (
            "E_inv_Y", "E_Y", "E_log_Y", "E_X", "E_X_inv_Y", "E_XXT_inv_Y",
        )
    )
    mo.md(
        "### 2.1 `NormalMixtureEta` in theory order\n\n"
        "Field order matches `docs/theory/shrinkage.rst` "
        "$(s_1, s_2, s_3, s_4, s_5, s_6)$ — shared with the planned "
        "`FactorMixtureStats` so per-field $\\tau$ ports across families.\n\n"
        f"{_field_summary}\n\n"
        f"Fields are JAX leaves: `isinstance(eta, NormalMixtureEta)` → "
        f"`{isinstance(eta, NormalMixtureEta)}`"
    )
    return eta, model


@app.cell
def _phase1_rule_hierarchy(
    AffineRule,
    AffineUpdate,
    EWMAUpdate,
    IdentityUpdate,
    RobbinsMonroUpdate,
    SampleWeightedUpdate,
    Shrinkage,
    eta,
    mo,
):
    _rules = [
        IdentityUpdate(),
        RobbinsMonroUpdate(tau0=10.0),
        EWMAUpdate(w=0.2),
        SampleWeightedUpdate(),
        AffineUpdate(b=0.9, c=0.1),
        Shrinkage(IdentityUpdate(), eta, tau=0.5),
    ]

    _rows = []
    for _r in _rules:
        _kind = (
            "AffineRule" if isinstance(_r, AffineRule)
            else "EtaUpdateRule"
        )
        _rows.append(
            f"| `{type(_r).__name__}` | `{_kind}` |"
        )

    mo.md(
        "### 2.2 Two-layer rule hierarchy\n\n"
        "`EtaUpdateRule` is the most general form $\\eta_t = "
        "\\mathrm{rule}(\\eta_{t-1}, \\hat\\eta)$ via `__call__`. "
        "`AffineRule` is the affine specialisation — its subclasses "
        "implement `weights(...)` and the base class runs the result "
        "through `affine_combine`.\n\n"
        "| Rule | Layer |\n"
        "|---|---|\n"
        + "\n".join(_rows) +
        "\n\nAll rules are `eqx.Module` pytrees: their hyperparameters are "
        "JAX leaves (JIT-compatible, differentiable)."
    )
    return


@app.cell
def _phase1_affine_combine(
    NormalMixtureEta,
    X_train,
    affine_combine,
    eta,
    jnp,
    mo,
    model,
    np,
):
    eta_new = model.e_step(X_train)

    _out_scalar = affine_combine(eta, eta_new, b=0.7, c=0.3)

    _w_pytree = NormalMixtureEta(
        E_inv_Y=jnp.float64(0.0),
        E_Y=jnp.float64(0.0),
        E_log_Y=jnp.float64(0.0),
        E_X=jnp.zeros_like(eta.E_X),
        E_X_inv_Y=jnp.zeros_like(eta.E_X_inv_Y),
        E_XXT_inv_Y=jnp.full_like(eta.E_XXT_inv_Y, 0.5),
    )
    _one_minus_w = NormalMixtureEta(
        E_inv_Y=jnp.float64(1.0),
        E_Y=jnp.float64(1.0),
        E_log_Y=jnp.float64(1.0),
        E_X=jnp.ones_like(eta.E_X),
        E_X_inv_Y=jnp.ones_like(eta.E_X_inv_Y),
        E_XXT_inv_Y=jnp.ones_like(eta.E_XXT_inv_Y) - _w_pytree.E_XXT_inv_Y,
    )

    _out_pytree = affine_combine(eta, eta_new, b=_one_minus_w, c=_w_pytree)

    _diff_scalar = float(jnp.linalg.norm(_out_scalar.E_X - eta_new.E_X))
    _diff_pytree_E_X = float(jnp.linalg.norm(_out_pytree.E_X - eta_new.E_X))
    _sigma_scalar_diag = np.array(_out_scalar.E_XXT_inv_Y).diagonal().mean()
    _sigma_pytree_diag = np.array(_out_pytree.E_XXT_inv_Y).diagonal().mean()

    mo.md(
        "### 2.3 `affine_combine` — scalar vs pytree weights\n\n"
        "Same call signature handles both forms; the second form lets the "
        "user weight individual sufficient statistics independently — "
        "the building block for Σ-only shrinkage.\n\n"
        f"- **scalar weights** (`b=0.7, c=0.3`): `||E_X - eta_new.E_X||` "
        f"= {_diff_scalar:.4f}\n"
        f"- **pytree weights** (zero on $s_1$–$s_5$, 0.5 on $s_6$): "
        f"`||E_X - eta_new.E_X||` = {_diff_pytree_E_X:.2e}\n\n"
        f"Mean diagonal of `E_XXT_inv_Y` after combine: "
        f"scalar = {_sigma_scalar_diag:.5f}, "
        f"pytree = {_sigma_pytree_diag:.5f}."
    )
    return


@app.cell
def _phase1_convergence(MarginalMixture, NormalMixture, mo, model):
    _conv = model.em_convergence_params()
    _leaf_shapes = [tuple(leaf.shape) for leaf in _conv]

    mo.md(
        "### 2.4 `em_convergence_params()` on `MarginalMixture`\n\n"
        "The fitter no longer hard-codes `(mu, gamma, L_Sigma)`. Each "
        "marginal exposes a pytree whose leaf-wise change measures EM "
        "convergence. For `NormalMixture` this is `(mu, gamma, L_Sigma)`; "
        "for the planned `FactorNormalMixture` it will be "
        r"$(\mu, \gamma, F F^\top + D)$ to sidestep $F$'s rotational gauge."
        "\n\n"
        f"- `isinstance(model, MarginalMixture)` → "
        f"**{isinstance(model, MarginalMixture)}**\n"
        f"- `isinstance(model, NormalMixture)` → "
        f"**{isinstance(model, NormalMixture)}**\n"
        f"- leaf shapes: **{_leaf_shapes}**"
    )
    return


@app.cell
def _phase2_header(mo):
    mo.md(r"""
    ## 3. Phase 2 — `Shrinkage(base, eta0, tau)` combinator

    The combinator wraps **any** base rule (affine or otherwise) and
    pulls each sufficient statistic toward a prior $\eta_0$. The
    scalar version reproduces the legacy `ShrinkageUpdate`
    numerically; the pytree version supports per-field $\tau$ — for
    instance shrinking only $\Sigma$.

    ### Notation

    Let the EM iterate (or mini-batch step) be indexed by $t$ and let
    $k \in \{1,\dots,6\}$ index the six sufficient-statistics fields
    of `NormalMixtureEta` — i.e. $s_1 = E[Y^{-1}]$, $s_2 = E[Y]$,
    $s_3 = E[\log Y]$, $s_4 = E[X]$, $s_5 = E[X/Y]$,
    $s_6 = E[XX^\top/Y]$.

    - $\hat\eta_k$ — the E-step's data-driven estimate of $s_k$
      (in batch EM: averaged over the full dataset; in online EM:
      averaged over the current mini-batch).
    - $\eta_{t,k}$ — the running estimate of $s_k$ kept by the rule
      after step $t$.
    - $\mathrm{base}(\eta_{t-1}, \hat\eta)_k$ — the *base rule*'s
      update for $s_k$ before shrinkage (e.g. $\hat\eta_k$ for
      `IdentityUpdate`, or a Robbins–Monro blend
      $b\,\eta_{t-1,k} + c\,\hat\eta_k$ for `RobbinsMonroUpdate`).
    - $\eta_{0,k}$ — the prior value for $s_k$.
    - $\tau_k \ge 0$ — the per-field shrinkage strength.

    ### Update formula

    `Shrinkage` produces, at every step and **independently for each
    field $k$**,

    $$
    \eta_{t,k} \;=\; \frac{\tau_k}{1+\tau_k}\,\eta_{0,k}
                  \;+\; \frac{1}{1+\tau_k}\,
                        \mathrm{base}(\eta_{t-1}, \hat\eta)_k.
    $$

    A *scalar* $\tau$ is the special case $\tau_1 = \dots = \tau_6 = \tau$
    (all six fields shrunk equally). A *pytree* $\tau$ allows a different
    $\tau_k$ for each field — setting $\tau_k = 0$ for $k \ne 6$ shrinks
    **only $\Sigma$** (§3.3 below).

    $\tau_k$ has a direct, interpretable meaning:

    - $\tau_k = 0$ → no shrinkage on $s_k$ (use the base rule's value).
    - $\tau_k = 1$ → 50/50 blend on $s_k$.
    - $\tau_k \to \infty$ → pure prior on $s_k$ ($\eta_{t,k} = \eta_{0,k}$).

    Equivalently $\tau_k$ is a *pseudo-count*: the prior contributes as
    much as $\tau_k \cdot n$ effective observations on $s_k$ against $n$
    real ones (see `docs/theory/shrinkage.rst` for the derivation).
    """)
    return


@app.cell
def _baseline_fit(
    BatchEMFitter,
    VarianceGamma,
    X_test,
    X_train,
    jnp,
    mo,
    time,
):
    init = VarianceGamma.default_init(X_train)
    _t0 = time.perf_counter()
    base_result = BatchEMFitter(
        max_iter=80, tol=1e-3,
        e_step_backend="cpu", m_step_backend="cpu",
    ).fit(init, X_train)
    _base_time = time.perf_counter() - _t0

    base_train_ll = float(base_result.model.marginal_log_likelihood(X_train))
    base_test_ll = float(base_result.model.marginal_log_likelihood(X_test))
    _base_sigma = base_result.model.sigma()
    base_cond = float(jnp.linalg.cond(_base_sigma))

    mo.md(
        "### 3.1 Baseline VG fit (no shrinkage)\n\n"
        f"- iterations: **{int(base_result.n_iter)}**, "
        f"converged: **{bool(base_result.converged)}**, "
        f"wall: **{_base_time:.2f}s**\n"
        f"- mean log-likelihood — train: **{base_train_ll:.4f}**, "
        f"test: **{base_test_ll:.4f}**\n"
        f"- $\\mathrm{{cond}}(\\Sigma)$: **{base_cond:.2e}**"
    )
    return base_cond, base_test_ll, base_train_ll, init


@app.cell
def _phase2_isotropic(
    BatchEMFitter,
    IdentityUpdate,
    Shrinkage,
    VarianceGamma,
    X_test,
    X_train,
    base_cond,
    base_test_ll,
    base_train_ll,
    eta0_isotropic,
    init,
    jnp,
    mo,
    time,
):
    _sigma2_data = float(jnp.var(X_train))

    eta0_iso = eta0_isotropic(
        VarianceGamma.default_init(X_train), sigma2=_sigma2_data)

    _rows = ["| τ | weight on η₀ | weight on η̂ | iters | "
             "train LL | test LL | cond(Σ) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    _rows.append(
        f"| 0 (baseline) | 0 | 1 | — | {base_train_ll:.4f} | "
        f"{base_test_ll:.4f} | {base_cond:.2e} |"
    )

    iso_results = {}
    for _tau in (0.05, 0.25, 1.0, 5.0):
        _t0 = time.perf_counter()
        _res = BatchEMFitter(
            eta_update=Shrinkage(IdentityUpdate(), eta0_iso, tau=_tau),
            max_iter=80, tol=1e-3,
            e_step_backend="cpu", m_step_backend="cpu",
        ).fit(init, X_train)
        _elapsed = time.perf_counter() - _t0
        _train_ll = float(_res.model.marginal_log_likelihood(X_train))
        _test_ll = float(_res.model.marginal_log_likelihood(X_test))
        _cond = float(jnp.linalg.cond(_res.model.sigma()))
        iso_results[_tau] = {
            "iters": int(_res.n_iter),
            "train_ll": _train_ll,
            "test_ll": _test_ll,
            "cond": _cond,
            "elapsed": _elapsed,
        }
        _w_prior = _tau / (1.0 + _tau)
        _w_data = 1.0 / (1.0 + _tau)
        _rows.append(
            f"| {_tau:g} | {_w_prior:.3f} | {_w_data:.3f} | "
            f"{int(_res.n_iter):d} | {_train_ll:.4f} | "
            f"{_test_ll:.4f} | {_cond:.2e} |"
        )

    mo.md(
        "### 3.2 Scalar τ — shrink **all six fields** toward an "
        "isotropic prior\n\n"
        "Here `Shrinkage(IdentityUpdate(), eta0_iso, tau=τ)` is run "
        "inside batch EM (`BatchEMFitter`), so $t$ indexes the EM "
        "iteration and $\\hat\\eta$ is the E-step on the **full** "
        "training set. Because $\\tau$ is a **scalar**, the same "
        "$\\tau_1 = \\dots = \\tau_6 = \\tau$ applies to every "
        "sufficient-statistics field $s_k$:\n\n"
        "$$\\eta_{t,k} = \\tfrac{\\tau}{1+\\tau}\\,\\eta_{0,k} "
        "+ \\tfrac{1}{1+\\tau}\\,\\hat\\eta_k, \\qquad k = 1,\\dots,6.$$\n\n"
        "**The prior $\\eta_0$ is `eta0_iso`**, built from a "
        "moment-of-data initialisation:\n\n"
        "- $s_1, s_2, s_3$: the gamma-subordinator expectations of "
        "`VarianceGamma.default_init(X_train)`.\n"
        "- $s_4 = \\mu_0 + \\gamma_0 E[Y]$, "
        "$\\;s_5 = \\mu_0 E[Y^{-1}] + \\gamma_0$ — built from "
        "`default_init`'s $(\\mu_0, \\gamma_0)$.\n"
        "- $s_6$: the $E[XX^\\top/Y]$ form whose extracted "
        "dispersion is $\\Sigma_0 = \\sigma^2 I_d$ with "
        f"$\\sigma^2 = \\mathrm{{Var}}(X) = {_sigma2_data:.4e}$.\n\n"
        "The actual matrix $\\Sigma_0$ that this prior implies — i.e. "
        "what the next cell prints — is therefore "
        "$\\sigma^2 I_d$. (Contrast §3.3, where a different prior is "
        "used **and** all $\\tau_k$ for $k \\ne 6$ are zeroed out so "
        "those prior values are effectively ignored.)\n\n"
        "Increasing $\\tau$ trades a tiny amount of in-sample LL for a "
        "**better-conditioned $\\Sigma$**. The test LL typically peaks "
        "at a moderate $\\tau$ in the small-$n$ regime.\n\n"
        + "\n".join(_rows)
    )
    return eta0_iso, iso_results


@app.cell
def _phase2_iso_target(mo):
    mo.md("""
    **Shrinkage target $\Sigma_0$ for §3.2** "
          "— `VarianceGamma.from_expectation(eta0_iso).sigma()`:
    """)
    return


@app.cell
def _phase2_iso_target_matrix(VarianceGamma, eta0_iso):
    VarianceGamma.from_expectation(eta0_iso).sigma()
    return


@app.cell
def _phase2_per_field(
    BatchEMFitter,
    IdentityUpdate,
    NormalMixtureEta,
    Shrinkage,
    VarianceGamma,
    X_test,
    X_train,
    base_cond,
    base_test_ll,
    base_train_ll,
    d,
    eta0_diagonal,
    init,
    jnp,
    mo,
    time,
):
    _diag_var = jnp.var(X_train, axis=0)
    eta0_diag = eta0_diagonal(
        VarianceGamma.default_init(X_train), diag=_diag_var)

    def _tau_sigma_only(value: float) -> NormalMixtureEta:
        return NormalMixtureEta(
            E_inv_Y=jnp.float64(0.0),
            E_Y=jnp.float64(0.0),
            E_log_Y=jnp.float64(0.0),
            E_X=jnp.zeros(d, dtype=jnp.float64),
            E_X_inv_Y=jnp.zeros(d, dtype=jnp.float64),
            E_XXT_inv_Y=jnp.full((d, d), value, dtype=jnp.float64),
        )

    _rows = ["| τ_Σ | weight on Σ₀ | weight on Σ̂ | "
             "iters | train LL | test LL | cond(Σ) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    _rows.append(
        f"| 0 (baseline) | 0 | 1 | — | {base_train_ll:.4f} | "
        f"{base_test_ll:.4f} | {base_cond:.2e} |"
    )

    sigma_only_results = {}
    for _tau_value in (0.25, 1.0, 4.0):
        _t0 = time.perf_counter()
        _res = BatchEMFitter(
            eta_update=Shrinkage(
                IdentityUpdate(), eta0_diag,
                tau=_tau_sigma_only(_tau_value)),
            max_iter=80, tol=1e-3,
            e_step_backend="cpu", m_step_backend="cpu",
        ).fit(init, X_train)
        _elapsed = time.perf_counter() - _t0
        _train_ll = float(_res.model.marginal_log_likelihood(X_train))
        _test_ll = float(_res.model.marginal_log_likelihood(X_test))
        _cond = float(jnp.linalg.cond(_res.model.sigma()))
        sigma_only_results[_tau_value] = {
            "train_ll": _train_ll, "test_ll": _test_ll, "cond": _cond,
            "elapsed": _elapsed, "iters": int(_res.n_iter),
        }
        _w_prior = _tau_value / (1.0 + _tau_value)
        _w_data = 1.0 / (1.0 + _tau_value)
        _rows.append(
            f"| {_tau_value:g} | {_w_prior:.3f} | {_w_data:.3f} | "
            f"{int(_res.n_iter):d} | {_train_ll:.4f} | "
            f"{_test_ll:.4f} | {_cond:.2e} |"
        )

    mo.md(
        "### 3.3 Per-field τ — shrink **only $s_6 = E[XX^\\top/Y]$**\n\n"
        "Same fitter as §3.2 (`BatchEMFitter` + `IdentityUpdate`), so "
        "$t$ is again the EM iteration and $\\hat\\eta$ comes from the "
        "full-data E-step. The difference is `tau`: instead of a scalar "
        "we pass a `NormalMixtureEta` pytree of $\\tau_k$ values\n\n"
        "$$\\tau_k = \\begin{cases} 0 & k \\in \\{1,2,3,4,5\\} \\\\ "
        "\\tau_\\Sigma & k = 6 \\end{cases}$$\n\n"
        "(`_tau_sigma_only(value)` builds exactly this pytree — every "
        "leaf zeroed out except `E_XXT_inv_Y`, which is filled with the "
        "scalar $\\tau_\\Sigma$). Plugging into the per-field formula\n\n"
        "$$\\eta_{t,k} = \\tfrac{\\tau_k}{1+\\tau_k}\\,\\eta_{0,k} "
        "+ \\tfrac{1}{1+\\tau_k}\\,\\hat\\eta_k$$\n\n"
        "the $\\tau_k = 0$ rows simplify to "
        "$\\eta_{t,k} = \\hat\\eta_k$ (data only — the prior values "
        "for $s_1,\\dots,s_5$ are multiplied by 0 and **never enter "
        "the update**). Only $s_6$ is blended:\n\n"
        "$$\\eta_{t,6} = \\tfrac{\\tau_\\Sigma}{1+\\tau_\\Sigma}\\,"
        "\\eta_{0,6} + \\tfrac{1}{1+\\tau_\\Sigma}\\,\\hat\\eta_6.$$\n\n"
        "Because $s_6 = \\Sigma + \\mu\\mu^\\top E[1/Y] + "
        "\\gamma\\gamma^\\top E[Y] + \\mu\\gamma^\\top + "
        "\\gamma\\mu^\\top$, this is equivalent to shrinking only the "
        "dispersion $\\Sigma$ (the $\\mu, \\gamma$ pieces of $s_6$ are "
        "consistent with the unshrunk $\\hat\\mu, \\hat\\gamma$). The "
        "prior matrix $\\Sigma_0 = \\mathrm{diag}(\\mathrm{Var}_j(X))$ "
        "is `_diag_var`; the next cell prints the full matrix that "
        "`eta0_diag` implies.\n\n"
        "Building block for Ledoit–Wolf-style targets (use "
        "`eta0_with_sigma` for an arbitrary $\\Sigma_0$).\n\n"
        + "\n".join(_rows)
    )
    return (eta0_diag,)


@app.cell
def _phase2_diag_target(mo):
    mo.md("""
    **Shrinkage target $\Sigma_0$ for §3.3** "
          "— `VarianceGamma.from_expectation(eta0_diag).sigma()`:
    """)
    return


@app.cell
def _phase2_diag_target_matrix(VarianceGamma, eta0_diag):
    VarianceGamma.from_expectation(eta0_diag).sigma()
    return


@app.cell
def _phase2_composition(
    IdentityUpdate,
    IncrementalEMFitter,
    RobbinsMonroUpdate,
    Shrinkage,
    VarianceGamma,
    X_test,
    X_train,
    base_test_ll,
    eta0_isotropic,
    init,
    jax,
    jnp,
    mo,
    time,
):
    _sigma2_data = float(jnp.var(X_train))
    eta0_compose = eta0_isotropic(
        VarianceGamma.default_init(X_train), sigma2=_sigma2_data)
    _key = jax.random.PRNGKey(0)

    _rows = ["| base rule | τ | weight on η₀ | weight on base | "
             "test LL | wall (s) |",
             "|---|---:|---:|---:|---:|---:|"]

    _runs = [
        ("Identity (batch)", IdentityUpdate(), 0.5, 60),
        ("Robbins–Monro (online)", RobbinsMonroUpdate(tau0=10.0), 0.0, 80),
        ("Robbins–Monro + shrink", RobbinsMonroUpdate(tau0=10.0), 0.5, 80),
    ]

    incr_results = {}
    for _name, _base, _tau_val, _max_steps in _runs:
        _rule = (Shrinkage(_base, eta0_compose, tau=_tau_val) if _tau_val > 0
                 else _base)
        _t0 = time.perf_counter()
        _res = IncrementalEMFitter(
            eta_update=_rule,
            batch_size=64,
            max_steps=_max_steps,
            e_step_backend="cpu",
            m_step_backend="cpu",
        ).fit(init, X_train, key=_key)
        _elapsed = time.perf_counter() - _t0
        _ll = float(_res.model.marginal_log_likelihood(X_test))
        incr_results[_name] = {"test_ll": _ll, "elapsed": _elapsed}
        _w_prior = _tau_val / (1.0 + _tau_val) if _tau_val > 0 else 0.0
        _w_base = 1.0 / (1.0 + _tau_val) if _tau_val > 0 else 1.0
        _rows.append(
            f"| {_name} | {_tau_val:g} | {_w_prior:.3f} | {_w_base:.3f} | "
            f"{_ll:.4f} | {_elapsed:.2f} |")

    mo.md(
        "### 3.4 Composing `Shrinkage` with online rules — "
        "**different setup from §3.2/§3.3**\n\n"
        "The fitter here is `IncrementalEMFitter`, not `BatchEMFitter`, "
        "so the meaning of $t$ and $\\hat\\eta$ changes:\n\n"
        "- $t$ now indexes the **mini-batch step** (one E-step per "
        "batch of size 64), not a full EM iteration over the dataset.\n"
        "- $\\hat\\eta_t$ is the E-step on the **$t$-th mini-batch** "
        "alone, not on the full $X_\\mathrm{train}$.\n"
        "- The `base` rule no longer needs to be the identity. For "
        "`RobbinsMonroUpdate(tau0=10)` the base produces its own "
        "running blend\n\n"
        "$$\\mathrm{base}(\\eta_{t-1}, \\hat\\eta)_k "
        "= b_t\\,\\eta_{t-1,k} + c_t\\,\\hat\\eta_{t,k},\\qquad "
        "c_t = \\tfrac{1}{\\tau_0 + t},\\;\\; b_t = 1 - c_t,$$\n\n"
        "  so a single mini-batch never overwrites the running estimate.\n\n"
        "The shrinkage step then applies the same per-field formula as "
        "before, but on top of `base`:\n\n"
        "$$\\eta_{t,k} = \\tfrac{\\tau_k}{1+\\tau_k}\\,\\eta_{0,k} "
        "+ \\tfrac{1}{1+\\tau_k}\\,"
        "\\mathrm{base}(\\eta_{t-1}, \\hat\\eta)_k.$$\n\n"
        "Here $\\tau$ is again a scalar, so the same $\\tau$ applies to "
        "all six fields. The columns 'weight on η₀' and 'weight on base' "
        "show $\\tau/(1+\\tau)$ and $1/(1+\\tau)$ for each row. The "
        "prior $\\eta_0$ = `eta0_compose` is the same isotropic prior "
        "as §3.2; the next cell prints the implied $\\Sigma_0$.\n\n"
        + "\n".join(_rows) +
        f"\n\nFor reference, the batch baseline test LL was "
        f"**{base_test_ll:.4f}**."
    )
    return (eta0_compose,)


@app.cell
def _phase2_compose_target(mo):
    mo.md("""
    **Shrinkage target $\Sigma_0$ for §3.4** "
          "— `VarianceGamma.from_expectation(eta0_compose).sigma()` "
          "(same isotropic prior as §3.2):
    """)
    return


@app.cell
def _phase2_compose_target_matrix(VarianceGamma, eta0_compose):
    VarianceGamma.from_expectation(eta0_compose).sigma()
    return


@app.cell
def _phase2_sweep_header(mo):
    mo.md(r"""
    ## 4. τ sweep — picking the regularisation strength

    Test log-likelihood vs $\tau$ for the isotropic prior. The peak
    of `test_ll` is a reasonable starting point for $\tau$ when no
    held-out validation set is available; in production prefer
    cross-validation.
    """)
    return


@app.cell
def _phase2_sweep(base_test_ll, iso_results, mo):
    _sorted_taus = sorted(iso_results.keys())
    _sweep = [
        (tau, iso_results[tau]["test_ll"], iso_results[tau]["cond"])
        for tau in _sorted_taus
    ]
    _table = "\n".join(
        f"| τ = {tau:g} | {ll:.4f} | {cond:.2e} |"
        for tau, ll, cond in _sweep
    )

    _best_tau, _best_ll, _best_cond = max(_sweep, key=lambda r: r[1])

    mo.md(
        "| τ | test LL | cond(Σ) |\n"
        "|---:|---:|---:|\n"
        f"{_table}\n\n"
        f"- **best τ for held-out LL**: τ = **{_best_tau:g}** "
        f"→ test LL = **{_best_ll:.4f}**, cond(Σ) = **{_best_cond:.2e}**.\n"
        f"- **baseline (τ = 0)**: test LL = **{base_test_ll:.4f}**.\n\n"
        "(Δ test LL > 0 means shrinkage helped on this slice.)"
    )
    return


@app.cell
def _summary(mo):
    mo.md(r"""
    ## 5. Takeaways

    - **Phase 1** keeps the EM API JIT-friendly and pytree-native:
        * `NormalMixtureEta` mirrors theory order so per-field
          shrinkage targets transfer unchanged to the planned factor
          family.
        * `EtaUpdateRule` / `AffineRule` separates the general form
          from the affine specialisation; non-affine rules
          (combinators, ML predictors) plug in without another API
          revision.
        * `MarginalMixture.em_convergence_params()` removes
          full-covariance assumptions from the fitter.
    - **Phase 2** is one combinator + four target builders:
        * `Shrinkage(base, eta0, tau)` with scalar OR pytree `tau`,
        * targets via `eta0_isotropic`, `eta0_diagonal`,
          `eta0_with_sigma`, `eta0_from_model`,
        * composes with `IdentityUpdate` for batch EM, with
          `RobbinsMonroUpdate` / `EWMAUpdate` /
          `SampleWeightedUpdate` for incremental EM.

    See `docs/design/penalised_em.md` for the API rationale and
    `docs/theory/shrinkage.rst` for the derivation.
    """)
    return


if __name__ == "__main__":
    app.run()
