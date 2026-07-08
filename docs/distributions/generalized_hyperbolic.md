---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 300
---

# GeneralizedHyperbolic

The **Generalized Hyperbolic** (GH) distribution is the most general member of
the family — the normal variance-mean mixture

$$
X \mid Y \sim \mathcal{N}(\mu + \gamma Y,\; \Sigma Y), \qquad
Y \sim \mathrm{GIG}(p, a, b),
$$

with a full {doc}`GIG <gig>` subordinator. Because the GIG nests the
{doc}`Gamma <gamma>`, {doc}`InverseGamma <inverse_gamma>`, and
{doc}`InverseGaussian <inverse_gaussian>` as limits, GH contains the
{doc}`VarianceGamma <variance_gamma>`, {doc}`NormalInverseGamma
<normal_inverse_gamma>`, and {doc}`NormalInverseGaussian
<normal_inverse_gaussian>` as special cases — reach for it when you want the
most flexible model.

## Density gallery

Beyond the shared location $\mu$, skewness $\gamma$, and dispersion $\Sigma$,
GH exposes the GIG shape $p$, which tunes the tail weight independently. Here we
vary $p$ at a fixed positive skew.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import GeneralizedHyperbolic, UnivariateGeneralizedHyperbolic
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=5, suppress=True)

x = jnp.linspace(-6.0, 6.0, 400)
fig, ax = plt.subplots()
for p in [-1.5, -0.5, 1.0, 3.0]:
    gh = UnivariateGeneralizedHyperbolic.from_classical(
        mu=0.0, gamma=0.5, sigma=1.0, p=p, a=1.0, b=1.0)
    ax.plot(np.asarray(x), np.asarray(jax.vmap(gh.pdf)(x)), label=f"p = {p:g}")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_title(r"GeneralizedHyperbolic marginals ($\mu$=0, $\gamma$=0.5, $\sigma$=1, $a$=$b$=1)")
ax.legend()
plt.show()
```

## Parametrizations

Built from the shared location/shape block and the GIG subordinator:

| Symbol | Attribute | Meaning |
|---|---|---|
| $\mu$ | `mu` | location $(d,)$ |
| $\gamma$ | `gamma` | skewness $(d,)$ |
| $\Sigma = L_\Sigma L_\Sigma^\top$ | `L_Sigma` | dispersion Cholesky $(d, d)$ |
| $p$ | `p` | GIG shape (any real) |
| $a$ | `a` | GIG rate ($> 0$) |
| $b$ | `b` | GIG rate ($> 0$) |

The marginal is not an exponential family, but the joint $(X, Y)$ is — its
natural parametrization $\theta$, the closed-form Bessel log-density, and the
EM/MCECM algorithms are derived in {doc}`../theory/gh` and
{doc}`../theory/em_algorithm`.

## Quick usage

```{code-cell} python
mu = jnp.array([0.0, 0.0])
gamma = jnp.array([0.3, -0.4])
Sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])

gh = GeneralizedHyperbolic.from_classical(mu=mu, gamma=gamma, sigma=Sigma, p=-0.5, a=1.0, b=1.0)
print("mean:", np.asarray(gh.mean()))
print("cov:\n", np.asarray(gh.cov()))

X = gh.rvs(2_000, seed=0)
# default_init warm-starts from the best of the NIG / VG / NInvG sub-model fits
result = GeneralizedHyperbolic.default_init(X).fit(X, max_iter=50, tol=1e-3)
print("converged:", result.converged, " fitted gamma:", np.asarray(result.model.gamma))
```

The $d = 1$ sibling {py:class}`~normix.distributions.generalized_hyperbolic.UnivariateGeneralizedHyperbolic`
adds `cdf` / `ppf`; the {doc}`FactorGeneralizedHyperbolic <factor_variants>`
scales to high dimensions.

## See also

- API: {py:class}`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic`
- Theory: {doc}`../theory/gh`, {doc}`../theory/em_algorithm`
- Tutorials: {doc}`../tutorials/core/02_gh_family_tour`,
  {doc}`../tutorials/distributions/04_normal_mixtures`
