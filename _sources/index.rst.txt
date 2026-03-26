normix Documentation
====================

**normix** is a JAX package for Generalized Hyperbolic distributions and related
distributions, implemented as exponential families. Built on
`Equinox <https://docs.kidger.site/equinox/>`_ with Float64 precision throughout.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from normix import GeneralizedHyperbolic, BatchEMFitter

   # Fit Generalized Hyperbolic to data via EM
   key = jax.random.PRNGKey(0)
   X = jax.random.normal(key, (1000, 3))
   model = GeneralizedHyperbolic.default_init(X)
   result = model.fit(X, max_iter=100)

   # Evaluate log-density (batched via vmap)
   log_p = jax.vmap(result.model.log_prob)(X)


Key Features
------------

- **Exponential family structure** — three parametrizations (classical, natural,
  expectation) with automatic conversion via ``jax.grad``
- **Full GH distribution family** — Gamma, Inverse Gamma, Inverse Gaussian, GIG,
  Variance Gamma, Normal-Inverse Gamma, Normal-Inverse Gaussian, Generalized Hyperbolic
- **EM algorithm** — batch, online, and mini-batch EM with CPU/GPU hybrid backend
- **JAX-native** — JIT-compiled, differentiable, ``vmap``-compatible
- **Immutable** — all distributions are ``eqx.Module`` pytrees; M-step returns a new model
- **Hybrid CPU/JAX backend** — up to 15× faster EM via ``scipy`` Bessel evaluation on CPU

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   design
   architecture
   theory/index
   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
