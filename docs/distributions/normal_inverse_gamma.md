---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 240
---

# NormalInverseGamma

The **Normal-Inverse Gamma** distribution is the normal variance-mean mixture

$$
X \mid Y \sim \mathcal{N}(\mu + \gamma Y,\; \Sigma Y), \qquad
Y \sim \mathrm{InvGamma}(\alpha, \beta),
$$

with an {doc}`InverseGamma <inverse_gamma>` subordinator — equivalently the
{doc}`GIG <gig>` limit $a \to 0,\; p = -\alpha,\; b = 2\beta$. The polynomial
right tail of the mixing law gives the marginal genuinely heavy (power-law)
tails, heavier than the {doc}`VarianceGamma <variance_gamma>`.

## Density gallery

As with every mixture, $\gamma$ controls skewness about $\mu$.

```{code-cell} python
:tags: [hide-input]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import NormalInverseGamma, UnivariateNormalInverseGamma
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=5, suppress=True)

x = jnp.linspace(-6.0, 6.0, 400)
fig, ax = plt.subplots()
for g in [-1.5, 0.0, 1.5]:
    ninvg = UnivariateNormalInverseGamma.from_classical(
        mu=0.0, gamma=g, sigma=1.0, alpha=3.0, beta=2.0)
    ax.plot(np.asarray(x), np.asarray(jax.vmap(ninvg.pdf)(x)), label=rf"$\gamma$ = {g:g}")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_title(r"NormalInverseGamma marginals ($\mu$=0, $\sigma$=1, $\alpha$=3, $\beta$=2)")
ax.legend()
plt.show()
```

## Parametrizations

Built from the shared location/shape block and the InverseGamma subordinator:

| Symbol | Attribute | Meaning |
|---|---|---|
| $\mu$ | `mu` | location $(d,)$ |
| $\gamma$ | `gamma` | skewness $(d,)$ |
| $\Sigma = L_\Sigma L_\Sigma^\top$ | `L_Sigma` | dispersion Cholesky $(d, d)$ |
| $\alpha$ | `alpha` | InverseGamma shape |
| $\beta$ | `beta` | InverseGamma rate |

The marginal is not an exponential family, but the joint $(X, Y)$ is — its
natural parametrization and the EM machinery are in {doc}`../theory/gh`.

## Quick usage

Raw $(\gamma, \Sigma, \beta)$ are identified only up to the scale gauge
$Y \mapsto cY$ (see {doc}`../theory/em_algorithm`); compare the invariants
$\mu$, $\gamma E[Y]$, and $E[Y]\,\Sigma$ instead.

```{code-cell} python
mu = jnp.array([0.0, 0.0])
gamma = jnp.array([0.3, -0.4])
Sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])

ninvg = NormalInverseGamma.from_classical(mu=mu, gamma=gamma, sigma=Sigma, alpha=3.0, beta=2.0)
print("mean:", np.asarray(ninvg.mean()))
print("cov:\n", np.asarray(ninvg.cov()))

X = ninvg.rvs(2_000, seed=0)
result = NormalInverseGamma.default_init(X).fit(X, max_iter=50, tol=1e-3)
fit = result.model
ey = float(fit.joint.subordinator().mean())
print("converged:", bool(result.converged), "n_iter:", int(result.n_iter))
print("mu:", np.asarray(fit.mu))
print("gamma * E[Y]:", np.asarray(fit.gamma) * ey)   # true [0.3, -0.4] (E[Y]=1)
print("E[Y] * Sigma:\n", ey * np.asarray(fit.sigma()))
print("mean:", np.asarray(fit.mean()))
print("cov:\n", np.asarray(fit.cov()))
```

The $d = 1$ sibling {py:class}`~normix.distributions.normal_inverse_gamma.UnivariateNormalInverseGamma`
adds `cdf` / `ppf`; the {doc}`FactorNormalInverseGamma <factor_variants>` scales
to high dimensions.

## See also

- API: {py:class}`~normix.distributions.normal_inverse_gamma.NormalInverseGamma`
- Theory: {doc}`../theory/gh`
- Tutorial: {doc}`../tutorials/distributions/04_normal_mixtures`
