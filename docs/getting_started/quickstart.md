---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 180
---

# Quickstart

Fit a Generalized Hyperbolic distribution to data and evaluate it — the whole
loop in a dozen lines.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)   # always enable float64 first
import jax.numpy as jnp

from normix import GeneralizedHyperbolic

# Some 3-D data (your returns, measurements, ...)
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (2000, 3))

# Initialize from data moments, then fit by EM
model = GeneralizedHyperbolic.default_init(X)
result = model.fit(X, max_iter=100, tol=1e-3)
fitted = result.model

print(f"converged: {result.converged} in {result.n_iter} iters")
```

The fit comes back inside an `EMResult`; `result.model` is the trained
distribution. Everything you would expect is one call away:

```{code-cell} python
# Log-density on a batch (the core acts on one observation; vmap to batch)
log_p = jax.vmap(fitted.log_prob)(X)

print("mean:\n", fitted.mean())
print("covariance:\n", fitted.cov())
print("mean log-likelihood:", float(fitted.marginal_log_likelihood(X)))

# Draw fresh samples from the fitted model
samples = fitted.rvs(5, seed=1)
print("samples shape:", samples.shape)
```

That is the entire workflow: **`default_init` → `fit` → use the model**. Swap
`GeneralizedHyperbolic` for any other family (`VarianceGamma`,
`NormalInverseGaussian`, …) and the code is identical.

## Where to next

- {doc}`first_model` — the same workflow, explained step by step.
- {doc}`../user_guide/distributions` — choosing the right distribution.
- {doc}`../tutorials/core/01_exponential_family` — the structure underneath it
  all.
