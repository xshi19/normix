Quick Start
===========

Installation
------------

.. code-block:: bash

   # Using uv (recommended)
   uv sync

   # Or pip
   pip install -e .

normix requires **Python ≥ 3.12** and uses Float64 precision. The package
automatically sets ``jax.config.update("jax_enable_x64", True)`` on import.


Core Dependencies
-----------------

.. list-table::
   :header-rows: 1

   * - Package
     - Role
   * - ``jax``
     - Array computation, autodiff, JIT, vmap
   * - ``equinox``
     - Immutable pytree-based modules
   * - ``scipy``
     - CPU Bessel evaluation (EM hot path)
   * - ``jaxopt``
     - L-BFGS-B for GIG η→θ optimization


Univariate Distributions
------------------------

All univariate distributions are exponential families with three parametrizations:

.. code-block:: python

   import jax.numpy as jnp
   from normix import Gamma

   # Create from classical parameters
   dist = Gamma(alpha=jnp.array(2.0), beta=jnp.array(1.0))

   # Evaluate log-density on a single observation
   dist.log_prob(jnp.array(1.5))

   # Three parametrizations
   theta = dist.natural_params()       # natural parameters θ
   eta   = dist.expectation_params()   # expectation parameters η = E[t(X)]
   I     = dist.fisher_information()   # Fisher information I(θ) = ∇²ψ(θ)

   # Reconstruct from natural or expectation parameters
   dist2 = Gamma.from_natural(theta)
   dist3 = Gamma.from_expectation(eta)

   # Maximum likelihood estimation
   key = jax.random.PRNGKey(0)
   samples = dist.rvs(1000, seed=42)
   dist_mle = Gamma.fit_mle(samples)


Available univariate distributions:

.. list-table::
   :header-rows: 1

   * - Class
     - Parameters
     - Description
   * - ``Gamma``
     - ``alpha``, ``beta``
     - Shape α > 0, rate β > 0
   * - ``InverseGamma``
     - ``alpha``, ``beta``
     - Shape α > 0, rate β > 0
   * - ``InverseGaussian``
     - ``mu``, ``lam``
     - Mean μ > 0, shape λ > 0
   * - ``GIG``
     - ``p``, ``a``, ``b``
     - Generalized Inverse Gaussian


Multivariate Normal
-------------------

.. code-block:: python

   from normix import MultivariateNormal

   mu = jnp.zeros(3)
   L = jnp.eye(3)  # Cholesky factor of covariance
   dist = MultivariateNormal(mu=mu, L_Sigma=L)

   # Log-density (single observation), batch via vmap
   x = jnp.ones(3)
   dist.log_prob(x)
   log_probs = jax.vmap(dist.log_prob)(X)


Normal Variance-Mean Mixtures
-----------------------------

The GH distribution family is modelled as a normal variance-mean mixture:

.. math::

   X \mid Y \sim \mathcal{N}(\mu + \gamma Y,\, \Sigma Y), \quad Y \sim \text{subordinator}

Each mixture has a **marginal** class (what users interact with) and a **joint** class
(used internally for the EM E-step).

.. list-table::
   :header-rows: 1

   * - Marginal Class
     - Subordinator
     - Parameters
   * - ``VarianceGamma``
     - Gamma
     - ``mu``, ``gamma``, ``L_Sigma``, ``alpha``, ``beta``
   * - ``NormalInverseGamma``
     - InverseGamma
     - ``mu``, ``gamma``, ``L_Sigma``, ``alpha``, ``beta``
   * - ``NormalInverseGaussian``
     - InverseGaussian
     - ``mu``, ``gamma``, ``L_Sigma``, ``mu_ig``, ``lam``
   * - ``GeneralizedHyperbolic``
     - GIG
     - ``mu``, ``gamma``, ``L_Sigma``, ``p``, ``a``, ``b``


Fitting with EM
---------------

The simplest way to fit a distribution is via the ``fit`` convenience method:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from normix import GeneralizedHyperbolic

   # Generate sample data
   key = jax.random.PRNGKey(0)
   X = jax.random.normal(key, (1000, 5))

   # Initialize from data moments, then fit
   model = GeneralizedHyperbolic.default_init(X)
   result = model.fit(X, max_iter=200, tol=1e-4)

   # result is an EMResult with diagnostics
   print(f"Converged: {result.converged}")
   print(f"Iterations: {result.n_iter}")
   print(f"Time: {result.elapsed_time:.2f}s")

   # The fitted model
   fitted = result.model

For more control, use ``BatchEMFitter`` directly:

.. code-block:: python

   from normix.fitting.em import BatchEMFitter

   fitter = BatchEMFitter(
       max_iter=200,
       tol=1e-4,
       e_step_backend='cpu',    # CPU Bessel for speed
       m_step_backend='cpu',    # CPU solver for GIG η→θ
       verbose=1,               # print summary
   )
   result = fitter.fit(model, X)

The ``e_step_backend='cpu'`` option routes Bessel function evaluations through
``scipy.special.kve`` instead of JAX, yielding a ~15× speedup for large datasets.
See :doc:`design` for the rationale behind this hybrid approach.


Bessel Functions
----------------

normix provides a JIT-able, differentiable ``log_kv`` (log modified Bessel function
of the second kind):

.. code-block:: python

   from normix import log_kv

   # JAX backend (JIT-able, differentiable)
   log_kv(0.5, 2.0)

   # CPU backend (fast, for EM hot path)
   log_kv(0.5, 2.0, backend='cpu')

The JAX backend uses a 4-regime dispatch (Hankel, Olver, small-z, Gauss-Legendre)
with ``@jax.custom_jvp`` for exact derivatives.


Batching with vmap
------------------

All core methods operate on single observations. Use ``jax.vmap`` for batching:

.. code-block:: python

   # Log-density over a batch
   log_probs = jax.vmap(dist.log_prob)(X)

   # Sufficient statistics over a batch
   T = jax.vmap(type(dist).sufficient_statistics)(X)

This keeps the core implementation clean and lets JAX handle vectorization optimally.
