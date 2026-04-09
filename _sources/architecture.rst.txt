Architecture
============

This page describes the package structure and class hierarchy of normix.


Package Layout
--------------

.. code-block:: text

   normix/
   ├── __init__.py                  # Public API, enables float64
   ├── exponential_family.py        # ExponentialFamily(eqx.Module) base class
   ├── distributions/
   │   ├── gamma.py                 # Gamma(α, β)
   │   ├── inverse_gamma.py         # InverseGamma(α, β)
   │   ├── inverse_gaussian.py      # InverseGaussian(μ, λ)
   │   ├── generalized_inverse_gaussian.py  # GIG(p, a, b)
   │   ├── normal.py                # MultivariateNormal(μ, L_Σ)
   │   ├── variance_gamma.py        # VarianceGamma / JointVarianceGamma
   │   ├── normal_inverse_gamma.py  # NormalInverseGamma / JointNormalInverseGamma
   │   ├── normal_inverse_gaussian.py  # NormalInverseGaussian / JointNormalInverseGaussian
   │   └── generalized_hyperbolic.py   # GeneralizedHyperbolic / JointGeneralizedHyperbolic
   ├── mixtures/
   │   ├── joint.py                 # JointNormalMixture(ExponentialFamily)
   │   └── marginal.py              # NormalMixture (owns a JointNormalMixture)
   ├── fitting/
   │   ├── em.py                    # EMResult; BatchEMFitter, IncrementalEMFitter
   │   ├── eta.py                   # NormalMixtureEta, affine_combine
   │   ├── eta_rules.py             # EtaUpdateRule (eqx.Module) + concrete rules
   │   └── solvers.py               # Bregman divergence solvers (η→θ)
   └── utils/
       ├── bessel.py                # log_kv with custom JVP
       ├── constants.py             # Shared numerical constants
       ├── plotting.py              # Notebook plotting helpers
       └── validation.py            # EM validation helpers


Class Hierarchy
---------------

.. code-block:: text

   eqx.Module
   ├── ExponentialFamily (abstract)
   │   ├── Gamma
   │   ├── InverseGamma
   │   ├── InverseGaussian
   │   ├── GeneralizedInverseGaussian (alias: GIG)
   │   ├── MultivariateNormal
   │   └── JointNormalMixture (abstract)
   │       ├── JointVarianceGamma
   │       ├── JointNormalInverseGamma
   │       ├── JointNormalInverseGaussian
   │       └── JointGeneralizedHyperbolic
   │
   └── NormalMixture (abstract)
       ├── VarianceGamma
       ├── NormalInverseGamma
       ├── NormalInverseGaussian
       └── GeneralizedHyperbolic


ExponentialFamily
-----------------

All distributions with a density of the form

.. math::

   p(x \mid \theta) = h(x) \exp\!\bigl(\theta^T t(x) - \psi(\theta)\bigr)

subclass ``ExponentialFamily``. Subclasses implement four abstract methods:

.. list-table::
   :header-rows: 1

   * - Method
     - Purpose
   * - ``_log_partition_from_theta(theta)``
     - Log-partition function :math:`\psi(\theta)`
   * - ``natural_params()``
     - Compute :math:`\theta` from stored classical parameters
   * - ``sufficient_statistics(x)``
     - Compute :math:`t(x)` for a single observation
   * - ``log_base_measure(x)``
     - Compute :math:`\log h(x)`

Everything else is derived automatically:

- ``log_prob(x)`` = :math:`\log h(x) + t(x) \cdot \theta - \psi(\theta)`
- ``expectation_params()`` = :math:`\nabla\psi(\theta)` via ``jax.grad``
- ``fisher_information()`` = :math:`\nabla^2\psi(\theta)` via ``jax.hessian``


Constructors
^^^^^^^^^^^^

.. code-block:: python

   # From classical parameters (human-readable)
   dist = Gamma(alpha=jnp.array(2.0), beta=jnp.array(1.0))
   dist = Gamma.from_classical(alpha=2.0, beta=1.0)

   # From natural parameters θ
   dist = Gamma.from_natural(theta)

   # From expectation parameters η (may involve optimization for GIG)
   dist = Gamma.from_expectation(eta)

   # MLE: η̂ = mean t(xᵢ), then from_expectation
   dist = Gamma.fit_mle(X)

   # Warm-start fit from current instance
   dist = dist.fit(X)


Distributions
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Distribution
     - Stored Attributes
     - Notes
   * - ``Gamma``
     - ``alpha``, ``beta``
     - Shape, rate
   * - ``InverseGamma``
     - ``alpha``, ``beta``
     - Shape, rate
   * - ``InverseGaussian``
     - ``mu``, ``lam``
     - Mean, shape
   * - ``GIG``
     - ``p``, ``a``, ``b``
     - Generalized Inverse Gaussian
   * - ``MultivariateNormal``
     - ``mu``, ``L_Sigma``
     - Mean, Cholesky of covariance


Mixture Structure
-----------------

The GH family is modelled as a normal variance-mean mixture. The **joint**
distribution :math:`f(x, y)` is an exponential family. The **marginal**
distribution :math:`f(x)` is not.

.. code-block:: text

   JointNormalMixture(ExponentialFamily)     f(x, y)
       ├── JointVarianceGamma                Y ~ Gamma
       ├── JointNormalInverseGamma           Y ~ InverseGamma
       ├── JointNormalInverseGaussian        Y ~ InverseGaussian
       └── JointGeneralizedHyperbolic        Y ~ GIG

   NormalMixture(eqx.Module)                f(x) = ∫ f(x,y) dy
       ├── VarianceGamma
       ├── NormalInverseGamma
       ├── NormalInverseGaussian
       └── GeneralizedHyperbolic

``NormalMixture`` owns a ``JointNormalMixture``. The joint provides:

- ``conditional_expectations(x)`` — E[log Y|x], E[1/Y|x], E[Y|x] for the EM E-step
- ``_mstep_normal_params(eta)`` — closed-form M-step for μ, γ, L_Σ from ``NormalMixtureEta``

.. list-table::
   :header-rows: 1

   * - Marginal Class
     - Joint Class
     - Mixing Distribution
   * - ``VarianceGamma``
     - ``JointVarianceGamma``
     - :math:`Y \sim \text{Gamma}(\alpha, \beta)`
   * - ``NormalInverseGamma``
     - ``JointNormalInverseGamma``
     - :math:`Y \sim \text{InverseGamma}(\alpha, \beta)`
   * - ``NormalInverseGaussian``
     - ``JointNormalInverseGaussian``
     - :math:`Y \sim \text{InverseGaussian}(\mu, \lambda)`
   * - ``GeneralizedHyperbolic``
     - ``JointGeneralizedHyperbolic``
     - :math:`Y \sim \text{GIG}(p, a, b)`


EM Algorithm
------------

The EM fitters implement the expectation-maximisation algorithm.

.. code-block:: python

   from normix.fitting.em import BatchEMFitter, EMResult

   fitter = BatchEMFitter(max_iter=200, tol=1e-4)
   result = fitter.fit(model, X)

``EMResult`` contains:

- ``model`` — the fitted distribution
- ``n_iter`` — number of iterations
- ``converged`` — whether the algorithm converged
- ``elapsed_time`` — wall-clock seconds
- ``param_changes`` — per-iteration max relative parameter change
- ``log_likelihoods`` — per-iteration log-likelihood (optional)

**Available fitters:**

.. list-table::
   :header-rows: 1

   * - Fitter
     - Description
   * - ``BatchEMFitter``
     - Standard batch EM; supports ``lax.scan`` (JIT) or Python loop (CPU). Optional ``eta_update`` for shrinkage.
   * - ``IncrementalEMFitter``
     - Mini-batch / online / fine-tuning EM with pluggable ``EtaUpdateRule`` (Robbins-Monro, EWMA, etc.)
