# Finance Architecture: `normix.finance`

**Date:** 2026-04-17
**Status:** Proposed — design sketch, implementation deferred until the
EM extensions in `docs/design/em_covariance_extensions.md` are in.
**Scope:** new top-level subpackage `normix/finance/`
**Theory:** `docs/theory/cvar_derivatives.rst`,
`docs/theory/mean_risk_optimization.rst`,
`docs/theory/transaction_costs.rst`,
`docs/theory/diversification.rst` (where applicable)

> The previous combined "next-stage architecture" document has been split:
> the EM / covariance work lives in
> `docs/design/em_covariance_extensions.md`, and this document covers only
> the `normix.finance` layer.

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
  `docs/theory/cvar_derivatives.rst`;
- it separates univariate risk calculus from portfolio chain rules;
- it allows exact formulas first, with Monte Carlo fallbacks later.

For portfolio-level gradients and Hessians, add helper functions that
apply the chain rule from projection parameters back to weights using
`(μ, γ, Σ)`.

## Mean-Risk Optimization

The theory in `docs/theory/mean_risk_optimization.rst` suggests two
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

The transaction-cost theory in `docs/theory/transaction_costs.rst` is
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
  `docs/design/em_covariance_extensions.md` §8 is undertaken).

This keeps finance as a thin application layer over the distribution
engine.

## Recommended Roadmap

### Phase D: Finance foundation

- add `normix.finance.projection`;
- implement `CVaR` value and derivatives for portfolio projections;
- add basic reporting objects and examples.

### Phase E: Portfolio optimization

- implement mean-risk optimization with the efficient-surface reduction;
- add transaction-cost QP builders;
- keep external solver dependencies optional until usage patterns
  settle.

### Phase F: Diversification analytics

- implement variance ENB and minimum torsion;
- implement generalized ENB based on squared coherent risk;
- add comparative examples showing when CVaR-based ENB differs from
  variance-based ENB.

(Phases A–C are the EM / covariance work in
`docs/design/em_covariance_extensions.md` and should land first, since
the finance layer benefits from the unified `MarginalMixture` interface
and from factor-analysis covariance for high-dimensional portfolios.)
