# Finance Architecture: `normix.finance`

> **IN PROGRESS — Phase D done; Phase E mean-risk done (transaction costs pending); Phase F proposed.**
> Moved to `../plans/` on 2026-05-10 (previously in `../design/`).
> Cross-references to the EM / covariance work now point to the archived proposal.

**Date:** 2026-04-17 (status refreshed 2026-06-26)
**Status:** Phase D (portfolio projection + CVaR) and the **mean-risk** half of
Phase E (efficient surface + frontier reduction) are implemented and shipped;
see the records below. Phase E transaction costs and Phase F (diversification)
remain design sketches. The EM / covariance
prerequisites (Phases A–C / Phases 1–4 of the EM extensions plan) are in
`master`, and the EM fitter the finance layer builds on has since been hardened
(VG/NInvG prior-moment floors, posterior `b_post` floor, diverged guard — see
`../archive/design/em_robustness_followups.md`).
**Scope:** new top-level subpackage `normix/finance/`
**Theory:** `../../docs/theory/cvar_derivatives.md`,
`../../docs/theory/mean_risk_optimization.md`,
`../../docs/theory/transaction_costs.md`,
`../../docs/theory/diversification.md` (where applicable)
**Living EM design:** `../design/em_framework.md`,
`../design/mixtures.md`.

---

## Main Recommendation

Create a dedicated `normix.finance` package that depends on the
distribution layer, but keep finance concepts out of the probabilistic
core unless they are genuinely distribution-level operations.

The finance module should be built around one central fact:

- if `X` is a multivariate normal mixture, then `wᵀ X` is a univariate
  normal mixture with parameters
  `(wᵀ μ, wᵀ γ, sqrt(wᵀ Σ w))`
  and the same subordinator family.

That portfolio projection is the bridge between the scientific-computing
core and the finance layer.

## Finance Should Be a Downstream Layer

Recommended dependency direction:

```text
normix.distributions / normix.mixtures / normix.fitting
    -> normix.finance.projection
    -> normix.finance.risk
    -> normix.finance.optimization
    -> normix.finance.diversification
```

The core package should not import finance. This keeps the distribution
layer clean and avoids turning every modeling class into an application
class.

## Core Finance Abstraction: Portfolio Projection

Before adding risk measures or optimizers, add one explicit projection
object.

```python
class PortfolioProjection(eqx.Module):
    mu: jax.Array
    gamma: jax.Array
    sigma: jax.Array
    subordinator: ExponentialFamily
```

For a given multivariate normal-mixture model and weight vector `w`, this
object represents the univariate distribution of `wᵀ X`.

That one abstraction gives a clean foundation for:

- VaR and CVaR evaluation;
- CVaR gradient and Hessian formulas;
- mean-risk optimization in reduced coordinates;
- worst-case box models;
- transaction-cost approximations.

Whether the projection lives as a helper function or a method on
`MarginalMixture` is an API choice. Recommendation:

- start with `normix.finance.projection.project_portfolio(model, w)`;
- only move it into the core distribution API later if it becomes
  universally useful outside finance.

## Proposed Module Layout

```text
normix/finance/
├── __init__.py
├── projection.py        # PortfolioProjection, project_portfolio
├── risk.py              # RiskMeasure, CoherentRiskMeasure, CVaR
├── optimization.py      # mean-risk problems and efficient-surface solvers
├── transaction_costs.py # QP approximation builders and result objects
├── diversification.py   # ENB, minimum torsion, generalized ENB
└── reports.py           # shared result dataclasses
```

## Risk Measures

Recommended base interface:

```python
class RiskMeasure(eqx.Module):
    def value(self, projection: PortfolioProjection) -> jax.Array: ...
    def gradient(self, projection: PortfolioProjection) -> jax.Array: ...
    def hessian(self, projection: PortfolioProjection) -> jax.Array: ...
```

Then build:

- `CVaR(alpha)` as the first concrete implementation;
- optional later additions such as variance, standard deviation, or
  worst-case box risk.

Why this interface works well:

- it keeps the analytic formulas close to the univariate theory in
  `../../docs/theory/cvar_derivatives.md`;
- it separates univariate risk calculus from portfolio chain rules;
- it allows exact formulas first, with Monte Carlo fallbacks later.

For portfolio-level gradients and Hessians, add helper functions that
apply the chain rule from projection parameters back to weights using
`(μ, γ, Σ)`.

## Mean-Risk Optimization

The theory in `../../docs/theory/mean_risk_optimization.md` suggests two
layers:

1. a general portfolio problem API;
2. a specialized reduced solver for coherent risk measures under normal
   mixtures.

Recommended objects:

```python
class MeanRiskProblem(eqx.Module):
    model: ...
    risk: RiskMeasure
    target_return: jax.Array | None
    constraints: ...


class EfficientSurfaceResult(eqx.Module):
    weights: jax.Array
    mu_tilde: jax.Array
    gamma_tilde: jax.Array
    risk_value: jax.Array
```

Implementation advice:

- do not begin with a large generic optimization framework;
- first expose the reduced formulas and a small number of well-documented
  solvers;
- keep solver dependencies optional until the API is stable.

## Transaction Costs

The transaction-cost theory in `../../docs/theory/transaction_costs.md` is
best treated as an optimization layer, not as a risk-measure layer.

Recommended design:

- keep the risk measure responsible only for `value`, `gradient`, and
  `hessian`;
- let `transaction_costs.py` build the local quadratic approximation and
  the QP matrices;
- add a small result dataclass with both the approximate solution and
  the implied portfolio weights.

This keeps the package modular: one can reuse the same `CVaR` object
with or without transaction costs.

## ENB and Generalized ENB

ENB is not a coherent risk measure, so it should not live under
`risk.py`. It deserves its own diversification module.

Recommended structure:

```python
class DiversificationMeasure(eqx.Module):
    def evaluate(self, model, w): ...
```

Concrete implementations:

- `VarianceENB`
- `MinimumTorsion`
- `GeneralizedENB(risk_measure=CVaR(...))`

This matches the theory:

- variance-based ENB depends on covariance diagonalization;
- generalized ENB depends on the Hessian of squared risk, which is
  downstream of the risk-measure layer.

## What the Finance Layer Should Reuse From the Core

The finance code should reuse existing model functionality rather than
re-implement distribution logic.

Required core ingredients:

- `mean()` and `cov()` from multivariate mixtures;
- subordinator moments from the fitted model;
- portfolio projection into univariate normal-mixture form;
- optional future access to covariance operators instead of dense
  matrices (only relevant once the deferred `DispersionModel` work in
  `../design/mixtures.md` § 6.5 is undertaken).

This keeps finance as a thin application layer over the distribution
engine.

## Recommended Roadmap

### Phase D: Finance foundation — **Implemented (2026-05-17, refactored 2026-05-24)**

- `NormalMixture.project(w)` / `normix.finance.projection.project_portfolio` —
  univariate-mixture projection of `wᵀ X` as a `Univariate*` instance
  (`UnivariateVarianceGamma`, `UnivariateNormalInverseGamma`, etc.).
- `Univariate*` classes expose scalar `mean`/`var`/`std`, `pdf`/`log_prob`,
  `cdf`/`ppf` (deterministic PINV over the marginal Bessel density), and
  `(n,)`-shaped `rvs`.
- `normix.finance.risk.CVaR(alpha)` — value, scalar gradient and Hessian in
  `(μ̃, γ̃, σ̃)`, and JIT-able `gradient_w` / `hessian_w` via the portfolio
  chain rule. **Deterministic VaR** uses `Univariate*.ppf(α)`. **CVaR value
  and derivatives** use conditional Monte Carlo (CMC) over the subordinator
  `Y` with a CMC bisection quantile (`quantile_cmc`) so the same `Y` samples
  are reused across value and derivatives (common random numbers). See
  Asmussen & Glynn (2007) §V.4 and Glasserman (2004) §4.2 for the CMC
  framework; the CMC CDF estimator is Rao-Blackwellised over `Y`.
- `normix.finance.functional.WeightFunctional` — bundles `CVaR`, model, and
  fixed `Y` into JIT-able `w ↦ risk(w)` with `grad` and `hess` for Phase E
  optimisation.
- Demo: the published tutorial
  [`docs/tutorials/finance/04_cvar_optimization.md`](../../docs/tutorials/finance/04_cvar_optimization.md)
  fits a mixture, computes CVaR, verifies the analytic gradient/Hessian against
  finite differences, and takes a few gradient steps to reduce risk. (The older
  `notebooks/finance_phase_d_cvar_demo.ipynb` is superseded by this tutorial and
  is slated for deletion in docs-refactor Phase 4.)
- Data: the finance tutorials read from the committed panel
  [`data/sp500_returns.csv`](../../data/sp500_returns.csv) (which includes the
  DJ30 constituents — 29 of 30; WBA, dropped from the index in 2024, is absent).
  [`scripts/download_dj30.py`](../../scripts/download_dj30.py) remains an
  optional offline refresh but is not used in CI (needs `yfinance` + network).
- Tests: [`tests/finance/test_cvar.py`](../../tests/finance/test_cvar.py).

### Phase E: Portfolio optimization

- **Mean-risk optimization — Implemented (2026-06-26).**
  `normix.finance.optimization.MeanRiskProblem(model, risk)` implements the
  efficient-surface reduction from `../../docs/theory/mean_risk_optimization.md`:
  the 3×3 matrix `A = [μ γ e]ᵀ Σ⁻¹ [μ γ e]` (Cholesky solves), minimum-dispersion
  `weights(μ̃, γ̃) = Σ⁻¹[μ γ e] A⁻¹ [μ̃ γ̃ 1]ᵀ`, `dispersion`/`expected_return`/
  `min_variance_point`, the convex `efficient_surface` (Fig. 8), and the
  `efficient_frontier` (Fig. 9) via a vectorized golden-section minimisation
  along each return-constraint line. Result pytrees `EfficientSurface` /
  `EfficientFrontier`.
  - The surface grid needs a fast risk evaluation: `CVaR.value_reduced(μ̃, γ̃, σ̃, Y)`
    inverts the conditional-MC CDF inside an analytic bracket (no PINV/Bessel),
    so it is `jax.vmap`-able across the grid. `_mc.py` gained raw cores
    (`cdf_cmc_raw`, `quantile_cmc_raw`); the object-based `value`/`quantile_cmc`
    now delegate to them (behaviour unchanged).
  - Tutorial: [`docs/tutorials/finance/05_mean_risk_optimization.md`](../../docs/tutorials/finance/05_mean_risk_optimization.md)
    replicates [Shi2016] Figs. 8–9 (efficient surface + geometry) for GH and
    overlays the gauge-invariant efficient frontiers of VG / NIG / NInvG / GH.
  - Tests: [`tests/finance/test_optimization.py`](../../tests/finance/test_optimization.py).
- **Transaction costs — still proposed.** Add the local-quadratic / QP builders
  from `../../docs/theory/transaction_costs.md`; keep external solver
  dependencies optional until usage patterns settle.

### Phase F: Diversification analytics

- implement variance ENB and minimum torsion;
- implement generalized ENB based on squared coherent risk;
- add comparative examples showing when CVaR-based ENB differs from
  variance-based ENB.

(Phases A–C were the EM / covariance work, now done — see
`../archive/design/em_covariance_extensions.md`. The finance layer
benefits from the unified `MarginalMixture` interface and the
factor-analysis covariance family for high-dimensional portfolios.)
