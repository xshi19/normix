---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 300
---

# Distribution gallery

One compact reference page per distribution — a density picture, the three
parametrizations, and a ten-line usage recipe. Think of it as the
`scipy.stats`-style index for the Generalized Hyperbolic family: a direct,
linkable answer to *"what does this distribution look like and how do I use
it?"*

The pages are ordered to narrate the family tree. The
{doc}`GIG <gig>` is the parent: it is an exponential family with a
Bessel-valued log-partition, and it nests the {doc}`Gamma <gamma>`,
{doc}`InverseGamma <inverse_gamma>`, and {doc}`InverseGaussian
<inverse_gaussian>` as boundary limits. Compounding a
{doc}`MultivariateNormal <multivariate_normal>` against one of these positive
subordinators produces the normal variance-mean mixtures — {doc}`VarianceGamma
<variance_gamma>`, {doc}`NormalInverseGamma <normal_inverse_gamma>`,
{doc}`NormalInverseGaussian <normal_inverse_gaussian>`, and the
all-encompassing {doc}`GeneralizedHyperbolic <generalized_hyperbolic>`. The
{doc}`factor variants <factor_variants>` swap the dense covariance for a
low-rank-plus-diagonal structure.

```{code-cell} python
:tags: [hide-input]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import (
    GIG, Gamma, InverseGamma, InverseGaussian, MultivariateNormal,
    UnivariateVarianceGamma, UnivariateNormalInverseGamma,
    UnivariateNormalInverseGaussian, UnivariateGeneralizedHyperbolic,
)
from normix.utils.plotting import set_theme, COLORS

set_theme()


def thumb(ax, dist, x, title):
    y = np.asarray(jax.vmap(dist.pdf)(x))
    xn = np.asarray(x)
    ax.plot(xn, y, color=COLORS["accent"], lw=1.8)
    ax.fill_between(xn, y, color=COLORS["accent"], alpha=0.12)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])


fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.0))
xp = jnp.linspace(1e-3, 6.0, 300)     # positive support
xm = jnp.linspace(-6.0, 6.0, 300)     # real line

# Row 1 — GIG and its positive Gamma / InverseGamma limits
thumb(axes[0, 0], GIG(p=jnp.array(1.5), a=jnp.array(2.0), b=jnp.array(1.5)), xp, "GIG")
thumb(axes[0, 1], Gamma(alpha=jnp.array(2.5), beta=jnp.array(1.5)), xp, "Gamma")
thumb(axes[0, 2], InverseGamma(alpha=jnp.array(3.0), beta=jnp.array(2.0)), xp, "InverseGamma")

# Row 2 — InverseGaussian, the Gaussian core, and the first mixture
thumb(axes[1, 0], InverseGaussian(mu=jnp.array(1.0), lam=jnp.array(2.0)), xp, "InverseGaussian")

axm = axes[1, 1]
mvn = MultivariateNormal.from_classical(jnp.zeros(2), jnp.array([[1.0, 0.6], [0.6, 1.0]]))
grid = jnp.linspace(-3.0, 3.0, 80)
GX, GY = jnp.meshgrid(grid, grid)
pts = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
Z = np.asarray(jax.vmap(mvn.pdf)(pts)).reshape(GX.shape)
axm.contourf(np.asarray(GX), np.asarray(GY), Z, levels=8, cmap="BuPu")
axm.set_title("MultivariateNormal", fontsize=10.5)
axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])

thumb(axes[1, 2],
      UnivariateVarianceGamma.from_classical(mu=0.0, gamma=0.8, sigma=1.0, alpha=1.5, beta=1.0),
      xm, "VarianceGamma")

# Row 3 — the remaining mixtures
thumb(axes[2, 0],
      UnivariateNormalInverseGamma.from_classical(mu=0.0, gamma=0.8, sigma=1.0, alpha=3.0, beta=2.0),
      xm, "NormalInverseGamma")
thumb(axes[2, 1],
      UnivariateNormalInverseGaussian.from_classical(mu=0.0, gamma=0.8, sigma=1.0, mu_ig=1.0, lam=1.5),
      xm, "NormalInverseGaussian")
thumb(axes[2, 2],
      UnivariateGeneralizedHyperbolic.from_classical(mu=0.0, gamma=0.8, sigma=1.0, p=-0.5, a=1.0, b=1.0),
      xm, "GeneralizedHyperbolic")

fig.suptitle("The Generalized Hyperbolic family", y=0.99, fontsize=14)
fig.tight_layout()
plt.show()
```

## The pages

| Distribution | Support | Role in the family |
|---|---|---|
| {doc}`GIG <gig>` | $(0, \infty)$ | Bessel-valued parent of the positive subordinators |
| {doc}`Gamma <gamma>` | $(0, \infty)$ | $b \to 0$ limit of GIG |
| {doc}`InverseGamma <inverse_gamma>` | $(0, \infty)$ | $a \to 0$ limit of GIG |
| {doc}`InverseGaussian <inverse_gaussian>` | $(0, \infty)$ | $p = -\tfrac12$ special case of GIG |
| {doc}`MultivariateNormal <multivariate_normal>` | $\mathbb{R}^d$ | the Gaussian core that gets mixed |
| {doc}`VarianceGamma <variance_gamma>` | $\mathbb{R}^d$ | mixture with a Gamma subordinator |
| {doc}`NormalInverseGamma <normal_inverse_gamma>` | $\mathbb{R}^d$ | mixture with an InverseGamma subordinator |
| {doc}`NormalInverseGaussian <normal_inverse_gaussian>` | $\mathbb{R}^d$ | mixture with an InverseGaussian subordinator |
| {doc}`GeneralizedHyperbolic <generalized_hyperbolic>` | $\mathbb{R}^d$ | mixture with a GIG subordinator — nests them all |
| {doc}`Factor variants <factor_variants>` | $\mathbb{R}^d$ | low-rank-plus-diagonal covariance for high dimensions |

```{toctree}
:hidden:

gig
gamma
inverse_gamma
inverse_gaussian
multivariate_normal
variance_gamma
normal_inverse_gamma
normal_inverse_gaussian
generalized_hyperbolic
factor_variants
```
