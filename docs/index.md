# normix

**normix** is a JAX package for Generalized Hyperbolic distributions and their
relatives, implemented as exponential families. It is built on
[Equinox](https://docs.kidger.site/equinox/) with float64 precision throughout,
and is JIT-compiled, differentiable, and `vmap`-compatible end to end.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from normix import GeneralizedHyperbolic

# Fit a Generalized Hyperbolic distribution to data via EM
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (1000, 3))
model = GeneralizedHyperbolic.default_init(X)
result = model.fit(X, max_iter=100)

# Evaluate the log-density (batched via vmap)
log_p = jax.vmap(result.model.log_prob)(X)
```

## Key features

- **Exponential-family structure** — three parametrizations (classical, natural
  $\theta$, expectation $\eta$) with automatic, lossless conversion.
- **The full GH family** — Gamma, Inverse Gamma, Inverse Gaussian, GIG, Variance
  Gamma, Normal-Inverse Gamma, Normal-Inverse Gaussian, and Generalized
  Hyperbolic, plus univariate and factor variants.
- **EM fitting** — batch and incremental/mini-batch EM with a hybrid CPU/JAX
  backend (scipy Bessel on the hot path).
- **Closed-form divergences** — squared Hellinger and KL with no Monte Carlo.
- **Finance toolkit** — portfolio projection and differentiable CVaR.
- **Immutable** — all distributions are `equinox.Module` pytrees.

## Where to start

- New here? Read {doc}`getting_started/install` then the
  {doc}`getting_started/quickstart`.
- Want the concepts? The {doc}`user_guide/exponential_family` guide explains the
  structure underneath everything.
- Prefer learning by example? The {doc}`tutorials/index` run end to end.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting_started/install
getting_started/quickstart
getting_started/first_model
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user_guide/distributions
user_guide/exponential_family
user_guide/em_fitting
user_guide/divergences
user_guide/finance
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

theory/index
design/index
api/index
changelog
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
