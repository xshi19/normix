# Finance

The `normix.finance` subpackage turns a fitted mixture into portfolio analytics.
Because a normal variance-mean mixture is conditionally Gaussian given the latent
$Y$, portfolio quantities — their gradients and Hessians — are
computed by quadrature over $Y$, with the conditional normal integrated exactly.

## Portfolio projection

Any linear combination $w^\top X$ of a mixture's assets is again a univariate
member of the *same* family, with parameters available in closed form. No
re-fitting is needed:

```python
proj = model.project(w)                # a Univariate* distribution of wᵀX
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

cvar = CVaR(0.05)                       # 5% tail probability
var_95 = cvar.var(proj)                 # deterministic quantile
cvar_95 = cvar.value(proj)
```

The default uses 512 weighted nodes. Supply a sample or a weighted measure
in the same positional slot to choose another scenario set:

```python
from normix.finance import DiscreteMeasure

Y = proj.subordinator().rvs(20_000, seed=0)
mc = cvar.value(proj, Y)
Q = proj.subordinator().quadrature(1024)
value = cvar.value(proj, Q)
resolution = cvar.step_resolution(proj, Q)
Q_hist = DiscreteMeasure.from_weights(Y, historical_weights)
```

The resolution diagnostic is a heuristic. Small values call for more nodes;
finite-moment requirements remain separate. See
{doc}`../tutorials/finance/08_quadrature_scenarios` and
{doc}`../design/quadrature_scenarios`.

## Differentiable risk

The payoff for the exponential-family structure is analytic risk sensitivities,
in both the projected scalar parametrization and the portfolio weights:

```python
cvar.gradient_scalar(proj)   # ∂CVaR/∂(μ̃, γ̃, σ̃)
cvar.hessian_scalar(proj)    # 3×3 Hessian
cvar.gradient_w(model, w)
cvar.hessian_w(model, w)
```

These are checked against finite differences and can be used in
gradient- or Newton-based portfolio optimizers. The `WeightFunctional` helper
bundles a risk measure, model, and a fixed scenario measure into a callable with `.grad` and `.hess`.
See {doc}`../tutorials/finance/04_cvar_optimization` for verification and a
worked CVaR-reduction loop.

## Mean-risk optimization

`MeanRiskProblem` solves the mean-risk problem
$\min_w \rho(w^\top X)$ s.t. $w^\top e = 1,\, E[w^\top X] \ge m$ by exploiting
the normal-mixture reduction to two coordinates
$(\tilde\mu, \tilde\gamma) = (w^\top\mu, w^\top\gamma)$:

```python
from normix.finance import MeanRiskProblem, CVaR

prob = MeanRiskProblem(model, CVaR(0.05))

prob.weights(mu_t, gamma_t)        # min-dispersion weights for a target (μ̃, γ̃)
prob.min_dispersion_point()        # reduced coords of the global min-dispersion portfolio

surface = prob.efficient_surface(mu_grid, gamma_grid)
frontier = prob.efficient_frontier(targets, gamma_bounds=(lo, hi))
frontier.weights                   # realised portfolio weights along the frontier
```

The `efficient_surface` is the convex surface of {ref}`Shi2016 <shi2016>`
(Fig. 8); the `efficient_frontier` is the classical risk–return frontier of
Fig. 9: it minimizes risk at each target expected return. See {doc}`../tutorials/finance/05_mean_risk_optimization` for a worked
replication across all four mixture families.

## Transaction costs

An $\ell_1$ turnover penalty breaks the two-dimensional reduction, so
rebalancing with costs uses a **local quadratic program** at the current
portfolio $w_0$:

```python
import jax.numpy as jnp
from normix.finance import TransactionCostProblem, CVaR

tc = TransactionCostProblem(model, CVaR(0.05), c1=5.0, c2=5e-2)
A = -jnp.eye(model.d)              # optional long-only
result = tc.solve(w0, A=A, b=jnp.zeros(model.d))

result.weights                     # w* (or w0 if the approx. gain is ≤ 0)
result.turnover                    # ‖w* − w0‖₁
result.qp.m_tilde, result.qp.H_tilde   # theory matrices for an external QP
```

The risk measure is unchanged — only `gradient_w` / `hessian_w` at $w_0$ enter
the QP. See {doc}`../tutorials/finance/06_transaction_costs`.

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
