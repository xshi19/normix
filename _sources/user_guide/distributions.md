# Distributions

normix implements the Generalized Hyperbolic (GH) family and its relatives as a
single, consistent set of exponential-family distributions. This guide is a map
of what is available and how to choose.

## The family at a glance

Every multivariate distribution here is a **normal variance-mean mixture**,

$$
X \mid Y \sim \mathcal{N}(\mu + \gamma\, Y,\; \Sigma\, Y), \qquad Y \sim \text{subordinator},
$$

and the choice of positive subordinator $Y$ names the member:

| Distribution | Subordinator | Subordinator parameters |
|---|---|---|
| `VarianceGamma` | Gamma | `alpha`, `beta` |
| `NormalInverseGamma` | Inverse Gamma | `alpha`, `beta` |
| `NormalInverseGaussian` | Inverse Gaussian | `mu_ig`, `lam` |
| `GeneralizedHyperbolic` | GIG | `p`, `a`, `b` |

The shared location/shape parameters are always $\mu$ (`mu`), $\gamma$
(`gamma`), and the covariance $\Sigma$ given through its Cholesky factor
`L_Sigma`. Because the `GIG` nests the Gamma, Inverse Gamma, and Inverse
Gaussian as limits, `GeneralizedHyperbolic` nests all the others — see
{doc}`../tutorials/core/02_gh_family_tour`.

## Subordinators and the Gaussian core

The building blocks are exponential families in their own right:

| Distribution | Parameters | Notes |
|---|---|---|
| `Gamma` | `alpha`, `beta` | closed-form moments and MLE |
| `InverseGamma` | `alpha`, `beta` | closed-form moments and MLE |
| `InverseGaussian` | `mu`, `lam` | closed-form moments and MLE |
| `GIG` / `GeneralizedInverseGaussian` | `p`, `a`, `b` | Bessel-valued log-partition |
| `MultivariateNormal` | `mu`, `L_Sigma` | the Gaussian core |

See {doc}`../tutorials/distributions/01_univariate_positive`,
{doc}`../tutorials/distributions/02_gig`, and
{doc}`../tutorials/distributions/03_multivariate_normal`.

## Three layers per mixture

Each mixture comes in several layers; reach for the one that matches your task:

- **Marginal** (e.g. `NormalInverseGaussian`) — the distribution of $X$. This is
  what you usually fit and evaluate: `pdf`, `log_prob`, `mean`, `cov`, `rvs`.
- **Joint** (e.g. `JointNormalInverseGaussian`, via `model.joint`) — the pair
  $(X, Y)$ with the latent subordinator, used by the EM E-step and accessible
  for `joint.rvs` and `joint.conditional_expectations`.
- **Univariate** (e.g. `UnivariateNormalInverseGaussian`) — the $d = 1$ case,
  which adds a scipy-style `cdf` and `ppf` for tail calculations.
- **Factor** (e.g. `FactorNormalInverseGaussian`) — a high-dimensional variant
  with a low-rank-plus-diagonal covariance $\Sigma = F F^\top +
  \operatorname{diag}(D)$; see {doc}`../tutorials/distributions/05_factor_mixtures`.

## Choosing a distribution

- **Light-to-moderate tails, symmetric or skewed:** `VarianceGamma` or
  `NormalInverseGamma`.
- **Heavy tails (financial returns):** `NormalInverseGaussian` is a robust
  default; `GeneralizedHyperbolic` adds a third shape parameter for the heaviest
  cases.
- **Unsure / want the most flexible model:** `GeneralizedHyperbolic` — it
  contains the others as special cases.
- **Many assets / dimensions:** the corresponding `Factor*` variant.
- **One-dimensional with CDF/quantile needs:** the `Univariate*` variant.

## A common interface

Whatever you pick, the API is the same:

```python
from normix import NormalInverseGaussian

model = NormalInverseGaussian.from_classical(
    mu=mu, gamma=gamma, sigma=Sigma, mu_ig=1.0, lam=1.5)

model.pdf(x)            # density at one point (vmap to batch)
model.mean(); model.cov()
model.rvs(1000, seed=0)
result = model.fit(X)   # EM; result.model is the fit
```

Construction also works from natural parameters (`from_natural`) and from
expectation parameters (`from_expectation`) — the three parametrizations are
explained in {doc}`exponential_family`.
