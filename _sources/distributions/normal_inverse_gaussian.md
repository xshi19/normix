---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 240
---

# NormalInverseGaussian

The **Normal-Inverse Gaussian** (NIG) distribution is the normal variance-mean
mixture

$$
X \mid Y \sim \mathcal{N}(\mu + \gamma Y,\; \Sigma Y), \qquad
Y \sim \mathrm{InvGaussian}(\mu_{IG}, \lambda),
$$

with an {doc}`InverseGaussian <inverse_gaussian>` subordinator — the
{doc}`GIG <gig>` case $p = -\tfrac12,\; a = \lambda/\mu_{IG}^2,\; b = \lambda$.
It combines semi-heavy tails with tractable moments and is the standard robust
default for financial returns.

## Density gallery

The skewness $\gamma$ tilts the marginal; the InverseGaussian mixing gives it
its characteristic semi-heavy tails.

```{code-cell} python
:tags: [hide-input]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import NormalInverseGaussian, UnivariateNormalInverseGaussian
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=5, suppress=True)

x = jnp.linspace(-6.0, 6.0, 400)
fig, ax = plt.subplots()
for g in [-1.5, 0.0, 1.5]:
    nig = UnivariateNormalInverseGaussian.from_classical(
        mu=0.0, gamma=g, sigma=1.0, mu_ig=1.0, lam=1.5)
    ax.plot(np.asarray(x), np.asarray(jax.vmap(nig.pdf)(x)), label=rf"$\gamma$ = {g:g}")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_title(r"NormalInverseGaussian marginals ($\mu$=0, $\sigma$=1, $\mu_{IG}$=1, $\lambda$=1.5)")
ax.legend()
plt.show()
```

## Parametrizations

Built from the shared location/shape block and the InverseGaussian subordinator:

| Symbol | Attribute | Meaning |
|---|---|---|
| $\mu$ | `mu` | location $(d,)$ |
| $\gamma$ | `gamma` | skewness $(d,)$ |
| $\Sigma = L_\Sigma L_\Sigma^\top$ | `L_Sigma` | dispersion Cholesky $(d, d)$ |
| $\mu_{IG}$ | `mu_ig` | InverseGaussian mean |
| $\lambda$ | `lam` | InverseGaussian shape |

The marginal is not an exponential family, but the joint $(X, Y)$ is — its
natural parametrization and the EM machinery are in {doc}`../theory/gh`.

## Quick usage

```{code-cell} python
mu = jnp.array([0.0, 0.0])
gamma = jnp.array([0.3, -0.4])
Sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])

nig = NormalInverseGaussian.from_classical(mu=mu, gamma=gamma, sigma=Sigma, mu_ig=1.0, lam=1.5)
print("mean:", np.asarray(nig.mean()))
print("cov:\n", np.asarray(nig.cov()))

X = nig.rvs(2_000, seed=0)
result = NormalInverseGaussian.default_init(X).fit(X, max_iter=50, tol=1e-3)
print("converged:", bool(result.converged), "n_iter:", int(result.n_iter),
      "fitted gamma:", np.asarray(result.model.gamma))
```

The $d = 1$ sibling {py:class}`~normix.distributions.normal_inverse_gaussian.UnivariateNormalInverseGaussian`
adds `cdf` / `ppf` for VaR-style tail calculations; the
{doc}`FactorNormalInverseGaussian <factor_variants>` handles many assets.

## See also

- API: {py:class}`~normix.distributions.normal_inverse_gaussian.NormalInverseGaussian`
- Theory: {doc}`../theory/gh`
- Tutorials: {doc}`../tutorials/distributions/04_normal_mixtures`,
  {doc}`../tutorials/finance/01_univariate_index`
