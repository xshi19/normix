---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 240
---

# VarianceGamma

The **Variance Gamma** distribution is the normal variance-mean mixture

$$
X \mid Y \sim \mathcal{N}(\mu + \gamma Y,\; \Sigma Y), \qquad
Y \sim \mathrm{Gamma}(\alpha, \beta),
$$

with a {doc}`Gamma <gamma>` subordinator — equivalently the
{doc}`GIG <gig>` limit $p = \alpha,\; a = 2\beta,\; b \to 0$. It has lighter
tails than the other mixtures and no continuous component at the origin, making
it a natural first choice for mildly heavy-tailed, possibly skewed data.

## Density gallery

The skewness vector $\gamma$ tilts the density; at $\gamma = 0$ it is symmetric
about $\mu$. Below is the one-dimensional marginal.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import VarianceGamma, UnivariateVarianceGamma
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=5, suppress=True)

x = jnp.linspace(-6.0, 6.0, 400)
fig, ax = plt.subplots()
for g in [-1.5, 0.0, 1.5]:
    vg = UnivariateVarianceGamma.from_classical(
        mu=0.0, gamma=g, sigma=1.0, alpha=1.5, beta=1.0)
    ax.plot(np.asarray(x), np.asarray(jax.vmap(vg.pdf)(x)), label=rf"$\gamma$ = {g:g}")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_title(r"VarianceGamma marginals ($\mu$=0, $\sigma$=1, $\alpha$=1.5, $\beta$=1)")
ax.legend()
plt.show()
```

## Parametrizations

Built from the shared location/shape block and the Gamma subordinator:

| Symbol | Attribute | Meaning |
|---|---|---|
| $\mu$ | `mu` | location $(d,)$ |
| $\gamma$ | `gamma` | skewness $(d,)$ |
| $\Sigma = L_\Sigma L_\Sigma^\top$ | `L_Sigma` | dispersion Cholesky $(d, d)$ |
| $\alpha$ | `alpha` | Gamma shape |
| $\beta$ | `beta` | Gamma rate |

The marginal $f(x)$ is not itself an exponential family, but the joint
$(X, Y)$ is — its natural parametrization $\theta$ and the EM machinery are
derived in {doc}`../theory/gh`.

## Quick usage

```{code-cell} python
mu = jnp.array([0.0, 0.0])
gamma = jnp.array([0.3, -0.4])
Sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])

vg = VarianceGamma.from_classical(mu=mu, gamma=gamma, sigma=Sigma, alpha=1.5, beta=1.5)
print("mean:", np.asarray(vg.mean()))
print("cov:\n", np.asarray(vg.cov()))

X = vg.rvs(2_000, seed=0)                  # (n, d); vg.joint.rvs gives (X, Y)
result = VarianceGamma.default_init(X).fit(X, max_iter=50, tol=1e-3)
print("converged:", result.converged, " fitted gamma:", np.asarray(result.model.gamma))
```

The $d = 1$ sibling {py:class}`~normix.distributions.variance_gamma.UnivariateVarianceGamma`
adds a scipy-style `cdf` / `ppf`; the high-dimensional
{doc}`FactorVarianceGamma <factor_variants>` swaps $\Sigma$ for a factor model.

## See also

- API: {py:class}`~normix.distributions.variance_gamma.VarianceGamma`
- Theory: {doc}`../theory/gh`
- Tutorial: {doc}`../tutorials/distributions/04_normal_mixtures`
