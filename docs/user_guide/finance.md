# Finance

The `normix.finance` subpackage turns a fitted mixture into portfolio analytics.
Because a normal variance-mean mixture is conditionally Gaussian given the latent
$Y$, portfolio quantities — and crucially their gradients and Hessians — are
computable by a fast conditional Monte Carlo over $Y$ alone.

## Portfolio projection

Any linear combination $w^\top X$ of a mixture's assets is again a univariate
member of the *same* family, with parameters available in closed form. No
re-fitting is needed:

```python
from normix.finance import project_portfolio

proj = project_portfolio(model, w)     # a Univariate* distribution of wᵀX
proj.mean(); proj.std()
proj.ppf(0.05)                         # 5% quantile (a VaR level)
```

This makes it cheap to evaluate many candidate weightings against one fitted
model. See {doc}`../tutorials/finance/02_multivariate_stocks`.

## Tail risk: VaR and CVaR

`CVaR(alpha)` computes Value-at-Risk and Conditional Value-at-Risk at tail
probability `alpha`:

```python
from normix.finance import CVaR

cvar = CVaR(0.05)                       # 95% level
Y = proj.subordinator.rvs(100_000, seed=0)   # conditional-MC draws
var_95 = cvar.var(proj)                 # deterministic quantile
cvar_95 = cvar.value(proj, Y)           # conditional Monte Carlo over Y
```

Conditioning on $Y$ makes the estimator far lower-variance than sampling returns
directly.

## Differentiable risk

The payoff for the exponential-family structure is analytic risk sensitivities,
in both the projected scalar parametrization and the portfolio weights:

```python
cvar.gradient_scalar(proj, Y)   # ∂CVaR/∂(μ̃, γ̃, σ̃)
cvar.hessian_scalar(proj, Y)    # 3×3 Hessian
cvar.gradient_w(model, w, Y)    # ∇_w CVaR  (chain rule through the projection)
cvar.hessian_w(model, w, Y)     # weight-space Hessian
```

These match finite differences to machine precision and plug straight into
gradient- or Newton-based portfolio optimizers. The `WeightFunctional` helper
bundles a risk measure, model, and `Y` into a callable with `.grad` and `.hess`.
See {doc}`../tutorials/finance/04_cvar_optimization` for verification and a
worked CVaR-reduction loop.

## Scaling to many assets

At portfolio scale the factor mixtures replace a dense covariance with
$\Sigma = F F^\top + \operatorname{diag}(D)$, cutting covariance parameters from
$O(d^2)$ to $O(d r)$ and routing every solve through the Woodbury identity. The
GH tail behaviour is retained. See
{doc}`../tutorials/finance/03_factor_mixture_portfolios`.

## Further reading

The mathematical background — CVaR derivatives, mean–risk optimization,
transaction costs, and factor analysis — is developed in the
{doc}`theory notes <../theory/index>`.
