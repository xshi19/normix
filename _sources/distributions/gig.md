---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 180
---

# GIG

The **Generalized Inverse Gaussian** is the parent of normix's positive
distributions: an exponential family on $(0, \infty)$ whose log-partition is
Bessel-valued. Its density with parameters $(p, a, b)$ is

$$
f(x \mid p, a, b) = \frac{(a/b)^{p/2}}{2\,K_p(\sqrt{ab})}\,
x^{p-1}\exp\!\Big(-\tfrac12\big(a x + b/x\big)\Big), \qquad x > 0,
$$

where $K_p$ is the modified Bessel function of the second kind. It nests the
{doc}`Gamma <gamma>` ($b \to 0$), {doc}`InverseGamma <inverse_gamma>`
($a \to 0$), and {doc}`InverseGaussian <inverse_gaussian>` ($p = -\tfrac12$).

## Density gallery

The shape parameter $p$ controls the behaviour near the origin (monotone for
$p \le 1$, unimodal with an interior mode for $p > 1$), while $a$ and $b$ set
the scale and the two tails.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import GIG
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=5, suppress=True)

x = jnp.linspace(1e-3, 6.0, 400)
fig, ax = plt.subplots()
for p in [-1.0, 0.5, 1.5, 3.0]:
    g = GIG(p=jnp.array(p), a=jnp.array(2.0), b=jnp.array(1.5))
    ax.plot(np.asarray(x), np.asarray(jax.vmap(g.pdf)(x)), label=f"p = {p:g}")
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_title("GIG densities (a = 2, b = 1.5)")
ax.legend()
plt.show()
```

## Parametrizations

Stored attributes: `p`, `a`, `b`. As an exponential family with sufficient
statistic $t(x) = [\log x,\; 1/x,\; x]$:

| Parametrization | Value |
|---|---|
| Classical | $p \in \mathbb{R},\; a > 0,\; b > 0$ |
| Natural $\theta$ | $[\,p - 1,\; -b/2,\; -a/2\,]$ |
| Expectation $\eta = \nabla\psi$ | $[\,\mathbb{E}[\log X],\; \mathbb{E}[1/X],\; \mathbb{E}[X]\,]$ |

Inverting $\eta \mapsto \theta$ is a strictly convex but stiff Bessel problem,
solved by an $\eta$-rescaled multi-start Newton iteration. See {doc}`../theory/gig`
for the derivation.

## Quick usage

```{code-cell} python
gig = GIG(p=jnp.array(1.5), a=jnp.array(2.0), b=jnp.array(1.5))

print("mean/var/std:", float(gig.mean()), float(gig.var()), float(gig.std()))
print("pdf(1.0):    ", float(gig.pdf(jnp.array(1.0))))
print("5% quantile: ", float(gig.ppf(jnp.array(0.05))))

samples = gig.rvs(10_000, seed=0)          # Devroye exact sampler
fitted = GIG.fit_mle(samples)              # moment-match + multi-start Newton
print("fit (p,a,b): ", float(fitted.p), float(fitted.a), float(fitted.b))
```

## See also

- API: {py:class}`~normix.distributions.generalized_inverse_gaussian.GIG`
- Theory: {doc}`../theory/gig`
- Tutorial: {doc}`../tutorials/distributions/02_gig` — Bessel log-partition and
  the multi-start solver in depth
