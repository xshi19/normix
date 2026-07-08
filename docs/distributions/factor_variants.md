---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 240
---

# Factor variants

Every mixture in this gallery has a **factor-analysis** sibling that replaces
the dense covariance with a low-rank-plus-diagonal structure

$$
\Sigma = F F^\top + \operatorname{diag}(D), \qquad
F \in \mathbb{R}^{d \times r},\; D \in \mathbb{R}^d_{>0}.
$$

This cuts the dispersion cost from $d(d+1)/2$ parameters to $d\,r + d$ and keeps
every solve $O(d r^2)$ via the Woodbury identity — the practical choice when
$d$ is large (many assets, many features). The four variants —
`FactorVarianceGamma`, `FactorNormalInverseGamma`, `FactorNormalInverseGaussian`,
and `FactorGeneralizedHyperbolic` — mirror their full-$\Sigma$ counterparts
exactly, differing only in how $\Sigma$ is stored.

## The covariance structure

A rank-$r$ block $F F^\top$ captures the dominant correlations; the diagonal
$D$ supplies asset-specific idiosyncratic variance and keeps $\Sigma$ full-rank.

```{code-cell} python
:tags: [hide-input]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import FactorNormalInverseGaussian
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)

rng = np.random.default_rng(0)
d, r = 12, 2
F = jnp.asarray(rng.normal(size=(d, r)) * 0.6)
D = jnp.asarray(np.abs(rng.normal(size=d)) * 0.3 + 0.3)

model = FactorNormalInverseGaussian.from_classical(
    mu=jnp.zeros(d), gamma=jnp.zeros(d), F=F, D=D, mu_ig=1.0, lam=1.5)

low_rank = np.asarray(model.F @ model.F.T)
Sigma = np.asarray(model.sigma())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
vmax = float(np.abs(Sigma).max())
for ax, M, title in [(a1, low_rank, r"low-rank $FF^\top$ (rank 2)"),
                     (a2, Sigma, r"$\Sigma = FF^\top + \mathrm{diag}(D)$")]:
    im = ax.imshow(M, cmap="BrBG", vmin=-vmax, vmax=vmax)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"Factor covariance (d = {d}, r = {r})")
plt.show()
```

## Parametrizations

Built from the shared location/shape block, the factor covariance, and the
family's own subordinator parameters:

| Symbol | Attribute | Meaning |
|---|---|---|
| $\mu$ | `mu` | location $(d,)$ |
| $\gamma$ | `gamma` | skewness $(d,)$ |
| $F$ | `F` | factor loadings $(d, r)$ |
| $D$ | `D` | idiosyncratic variances $(d,)$, positive |
| subordinator | `subordinator` | Gamma / InverseGamma / InverseGaussian / GIG |

The subordinator keyword arguments to `from_classical` match the dense
siblings: `alpha, beta` (VarianceGamma, NormalInverseGamma), `mu_ig, lam`
(NormalInverseGaussian), or `p, a, b` (GeneralizedHyperbolic). Note $F$ is
identifiable only up to an $r \times r$ rotation, so EM convergence is measured
on $\Sigma$, not on $F$.

## Quick usage

```{code-cell} python
print("d / r:", model.d, model.r)
print("mean:", np.asarray(model.mean())[:4], "...")
print("Σ diagonal:", np.asarray(jnp.diag(model.sigma()))[:4], "...")

X = model.rvs(2_000, seed=0)
# default_init takes the factor rank r; fit runs EM with Woodbury linear algebra
result = FactorNormalInverseGaussian.default_init(X, r=2).fit(X, max_iter=50, tol=1e-3)
print("converged:", result.converged, " fitted r:", result.model.r)
```

## See also

- API: {py:class}`~normix.distributions.variance_gamma.FactorVarianceGamma`,
  {py:class}`~normix.distributions.normal_inverse_gamma.FactorNormalInverseGamma`,
  {py:class}`~normix.distributions.normal_inverse_gaussian.FactorNormalInverseGaussian`,
  {py:class}`~normix.distributions.generalized_hyperbolic.FactorGeneralizedHyperbolic`
- Theory: {doc}`../theory/factor_analysis`
- Tutorials: {doc}`../tutorials/distributions/05_factor_mixtures`,
  {doc}`../tutorials/finance/03_factor_mixture_portfolios`
