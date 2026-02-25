Architecture
============

This page describes the high-level class hierarchy, key attributes, and methods
in ``normix``. It serves as a reference for both users and developers.

Class Hierarchy
---------------

.. code-block:: text

   Distribution (abstract)
   ├── ExponentialFamily (abstract)
   │   ├── Exponential
   │   ├── Gamma
   │   ├── InverseGamma
   │   ├── InverseGaussian
   │   ├── GeneralizedInverseGaussian
   │   ├── MultivariateNormal
   │   └── JointNormalMixture (abstract)
   │       ├── JointVarianceGamma
   │       ├── JointNormalInverseGamma
   │       ├── JointNormalInverseGaussian
   │       └── JointGeneralizedHyperbolic
   └── NormalMixture (abstract)
       ├── VarianceGamma
       ├── NormalInverseGamma
       ├── NormalInverseGaussian
       └── GeneralizedHyperbolic

There are two parallel branches:

- **ExponentialFamily** branch: every class stores its own named internal
  attributes and exposes classical, natural, and expectation parametrizations
  as lazy cached properties.
- **NormalMixture** branch: marginal distributions :math:`f(x)` that are
  *not* exponential families.  Each owns a ``JointNormalMixture`` instance
  (accessible via ``.joint``) and delegates parameter storage to it.


Design Principles
-----------------

1. **Named attributes are the single source of truth** -- there is no
   centralized ``_natural_params`` tuple.  Each subclass stores its own
   canonical representation using Greek-letter names matching the
   mathematical notation (e.g., ``_alpha`` / ``_beta`` for Gamma,
   ``_mu`` / ``_L_Sigma`` for MultivariateNormal).

2. **Cached properties are lazy** -- ``natural_params``, ``classical_params``,
   and ``expectation_params`` are computed on demand from internal state
   and invalidated when state changes.

3. **Setter paths are independent** -- ``_set_from_classical`` and
   ``_set_from_natural`` each write directly to named attributes; they
   never call each other.

4. **Cholesky-first storage** -- all covariance and precision matrices are
   stored and manipulated via Cholesky factors (``solve_triangular``,
   ``cho_solve``), avoiding ``np.linalg.inv``.

5. **EM fast path** -- ``JointNormalMixture._set_internal()`` accepts
   pre-computed Cholesky factors directly, avoiding redundant
   decompositions during the EM M-step.


ExponentialFamily
-----------------

All distributions with a density of the form

.. math::

   p(x|\theta) = h(x) \exp\!\bigl(\theta^T t(x) - \psi(\theta)\bigr)

inherit from ``ExponentialFamily``.

Internal attributes
~~~~~~~~~~~~~~~~~~~

``ExponentialFamily`` itself stores **no** parameter attributes.  Each
subclass defines its own (see table below).

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Subclass
     - Internal attributes
     - Description
   * - ``Exponential``
     - ``_lambda``
     - Rate :math:`\lambda`
   * - ``Gamma``
     - ``_alpha``, ``_beta``
     - Shape :math:`\alpha`, rate :math:`\beta`
   * - ``InverseGamma``
     - ``_alpha``, ``_beta``
     - Shape :math:`\alpha`, rate :math:`\beta`
   * - ``InverseGaussian``
     - ``_mu``, ``_lambda``
     - Mean :math:`\mu`, shape :math:`\lambda`
   * - ``GeneralizedInverseGaussian``
     - ``_p``, ``_a``, ``_b``
     - :math:`p`, :math:`a`, :math:`b`
   * - ``MultivariateNormal``
     - ``_mu``, ``_L_Sigma``
     - Mean :math:`\mu`, lower Cholesky of :math:`\Sigma`
   * - ``JointNormalMixture`` subclasses
     - ``_mu``, ``_gamma``, ``_L_Sigma``, + mixing params
     - See :ref:`joint-mixture-attrs`

Public API
~~~~~~~~~~

The following methods and properties are intended for end users.

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Method / Property
     - Kind
     - Description
   * - ``from_classical_params(**kw)``
     - classmethod
     - Factory: create instance from classical parameters
   * - ``from_natural_params(theta)``
     - classmethod
     - Factory: create instance from natural parameters
   * - ``from_expectation_params(eta)``
     - classmethod
     - Factory: create instance from expectation parameters
   * - ``set_classical_params(**kw)``
     - method
     - Set params from classical, returns ``self``
   * - ``set_natural_params(theta)``
     - method
     - Set params from natural, returns ``self``
   * - ``set_expectation_params(eta)``
     - method
     - Set params from expectation, returns ``self``
   * - ``classical_params``
     - cached property
     - Frozen dataclass of classical parameters
   * - ``natural_params``
     - cached property
     - Flat numpy array of natural parameters
   * - ``expectation_params``
     - cached property
     - Flat numpy array of expectation parameters
   * - ``logpdf(x)``
     - method
     - Log probability density
   * - ``pdf(x)``
     - method
     - Probability density
   * - ``fit(X)``
     - method
     - MLE via sufficient statistics, returns ``self``
   * - ``score(X)``
     - method
     - Mean log-likelihood (sklearn convention)
   * - ``fisher_information(theta)``
     - method
     - Fisher information matrix (Hessian of :math:`\psi`)

Subclass contract (internal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods that every ``ExponentialFamily`` subclass **must** implement.
These are ``@abstractmethod`` and are never called by users directly.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Purpose
   * - ``_set_from_classical(**kw)``
     - Store classical params as named attrs, set ``_fitted``, invalidate cache
   * - ``_set_from_natural(theta)``
     - Decompose ``theta`` into named attrs, set ``_fitted``, invalidate cache
   * - ``_compute_natural_params()``
     - Build flat ``theta`` from named attrs (backs ``natural_params``)
   * - ``_compute_classical_params()``
     - Build frozen dataclass from named attrs (backs ``classical_params``)
   * - ``_sufficient_statistics(x)``
     - Compute :math:`t(x)`
   * - ``_log_partition(theta)``
     - Compute :math:`\psi(\theta)`
   * - ``_log_base_measure(x)``
     - Compute :math:`\log h(x)`
   * - ``_get_natural_param_support()``
     - Return bounds for ``theta`` validation

Optional overrides (internal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods with default implementations that subclasses **may** override for
efficiency.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Method
     - Default behaviour
     - Overridden by
   * - ``_compute_expectation_params()``
     - Calls ``_natural_to_expectation(natural_params)``
     - All distributions (direct from internal attrs, avoids natural param round-trip)
   * - ``_natural_to_expectation(theta)``
     - Numerical gradient of ``_log_partition`` via ``scipy.differentiate``
     - Exponential, Gamma, InverseGamma, InverseGaussian, GIG, MultivariateNormal,
       all JointNormalMixture subclasses
   * - ``_expectation_to_natural(eta)``
     - L-BFGS-B optimisation
     - Exponential, Gamma, InverseGamma, InverseGaussian, GIG, MultivariateNormal
   * - ``fisher_information(theta)``
     - Numerical Hessian of ``_log_partition`` via ``scipy.differentiate``
     - Exponential, Gamma
   * - ``logpdf(x)`` / ``logpdf(x, y)``
     - Generic :math:`\log h + \theta^T t - \psi`
     - MultivariateNormal, JointNormalMixture (Cholesky-based)


.. _joint-mixture-attrs:

JointNormalMixture
------------------

``JointNormalMixture`` extends ``ExponentialFamily`` for the joint density
:math:`f(x, y)` where :math:`X|Y \sim N(\mu + \gamma Y,\; \Sigma Y)`.

Internal attributes
~~~~~~~~~~~~~~~~~~~

All subclasses share three normal-component attributes stored at the
``JointNormalMixture`` level, plus mixing-specific attributes in each child.

.. list-table::
   :header-rows: 1
   :widths: 15 15 40 30

   * - Attribute
     - Type
     - Description
     - Set by
   * - ``_mu``
     - ``NDArray``
     - Location vector, shape ``(d,)``
     - ``_store_normal_params``, ``_set_from_natural``, ``_set_internal``
   * - ``_gamma``
     - ``NDArray``
     - Skewness vector, shape ``(d,)``
     - ``_store_normal_params``, ``_set_from_natural``, ``_set_internal``
   * - ``_L_Sigma``
     - ``NDArray``
     - Lower Cholesky of :math:`\Sigma`, shape ``(d, d)``
     - ``_store_normal_params``, ``_set_from_natural``, ``_set_internal``
   * - ``_d``
     - ``int``
     - Dimension of :math:`X`
     - Inferred from ``_mu``

**Mixing-specific attributes (subclass level):**

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Subclass
     - Attributes
     - Mixing distribution
   * - ``JointVarianceGamma``
     - ``_alpha``, ``_beta``
     - :math:`Y \sim \text{Gamma}(\alpha, \beta)`
   * - ``JointNormalInverseGamma``
     - ``_alpha``, ``_beta``
     - :math:`Y \sim \text{InvGamma}(\alpha, \beta)`
   * - ``JointNormalInverseGaussian``
     - ``_delta``, ``_eta``
     - :math:`Y \sim \text{InvGaussian}(\delta, \eta)`
   * - ``JointGeneralizedHyperbolic``
     - ``_p``, ``_a``, ``_b``
     - :math:`Y \sim \text{GIG}(p, a, b)`

Cached properties
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Property
     - Description
     - Inherited from
   * - ``natural_params``
     - Flat natural parameter vector
     - ``ExponentialFamily``
   * - ``classical_params``
     - Frozen dataclass of classical parameters
     - ``ExponentialFamily``
   * - ``expectation_params``
     - Expectation parameter vector
     - ``ExponentialFamily``
   * - ``log_det_Sigma``
     - :math:`\log|\Sigma| = 2\sum_i \log L_{ii}`
     - ``JointNormalMixture``
   * - ``L_Sigma_inv``
     - :math:`L_\Sigma^{-1}` via ``solve_triangular``
     - ``JointNormalMixture``
   * - ``gamma_mahal_sq``
     - :math:`\gamma^T \Sigma^{-1} \gamma`
     - ``JointNormalMixture``

Public API (additional to ExponentialFamily)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Method / Property
     - Kind
     - Description
   * - ``d``
     - property
     - Dimension of :math:`X`
   * - ``rvs(size, random_state)``
     - method
     - Sample ``(X, Y)`` from the joint distribution
   * - ``mean()``
     - method
     - :math:`E[X]` and :math:`E[Y]`
   * - ``var()``
     - method
     - :math:`\text{Var}[X]` and :math:`\text{Var}[Y]`
   * - ``cov()``
     - method
     - :math:`\text{Cov}[X]` and :math:`\text{Var}[Y]`
   * - ``conditional_mean_x_given_y(y)``
     - method
     - :math:`E[X|Y=y] = \mu + \gamma y`
   * - ``conditional_cov_x_given_y(y)``
     - method
     - :math:`\text{Cov}[X|Y=y] = \Sigma y`

Subclass contract (internal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract methods that every ``JointNormalMixture`` child must implement.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Purpose
   * - ``_get_mixing_distribution_class()``
     - Return the mixing distribution class (e.g., ``Gamma``)
   * - ``_store_mixing_params(**kw)``
     - Store mixing params as named attrs (e.g., ``_delta``, ``_eta``)
   * - ``_store_mixing_params_from_theta(theta)``
     - Extract and store mixing params from the natural parameter vector
   * - ``_compute_mixing_theta(theta_4, theta_5)``
     - Build :math:`\theta_1, \theta_2, \theta_3` from mixing attrs and quadratic forms
   * - ``_create_mixing_distribution()``
     - Construct a fitted mixing distribution from named attrs
   * - ``_set_from_classical(**kw)``
     - Parse all classical params (normal + mixing), store, set fitted
   * - ``_compute_classical_params()``
     - Build frozen dataclass from all named attrs

**Implemented at JointNormalMixture level** (not overridden by children):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Purpose
   * - ``_store_normal_params(mu, gamma, sigma)``
     - Validate and store ``_mu``, ``_gamma``, ``_L_Sigma`` (Cholesky of sigma)
   * - ``_set_from_natural(theta)``
     - Extract normal params via Cholesky, delegate mixing to ``_store_mixing_params_from_theta``
   * - ``_set_internal(mu, gamma, L_sigma, **mixing_kw)``
     - EM fast path: store pre-computed values directly, zero decompositions
   * - ``_compute_natural_params()``
     - Build full theta via ``cho_solve`` from ``_mu``, ``_gamma``, ``_L_Sigma`` + ``_compute_mixing_theta``
   * - ``_extract_normal_params_with_cholesky(theta)``
     - Extract :math:`\mu, \gamma` from an arbitrary ``theta`` vector via Cholesky of :math:`\Lambda`
   * - ``_sufficient_statistics(x, y)``
     - Compute :math:`t(x, y)` for the joint exponential family
   * - ``_log_base_measure(x, y)``
     - Compute :math:`\log h(x, y)`

**Overridden by children** (distribution-specific):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Overriding subclasses
   * - ``_log_partition(theta)``
     - All four (VG, NInvG, NIG, GH) -- mixing-specific terms
   * - ``_natural_to_expectation(theta)``
     - All four -- analytical gradient
   * - ``_get_natural_param_support()``
     - All four -- mixing-specific bounds


NormalMixture
-------------

``NormalMixture`` extends ``Distribution`` (not ``ExponentialFamily``) for the
marginal density :math:`f(x) = \int f(x,y)\,dy`.  It owns a ``JointNormalMixture``
instance and delegates all parameter storage to it.

Internal attributes
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 60

   * - Attribute
     - Description
   * - ``_joint``
     - The underlying ``JointNormalMixture`` instance (source of truth for all parameters)

Public API
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Method / Property
     - Kind
     - Description
   * - ``from_classical_params(**kw)``
     - classmethod
     - Factory: create from classical parameters
   * - ``from_natural_params(theta)``
     - classmethod
     - Factory: create from natural parameters (of the joint)
   * - ``from_expectation_params(eta)``
     - classmethod
     - Factory: create from expectation parameters (of the joint)
   * - ``set_classical_params(**kw)``
     - method
     - Forward to ``_joint.set_classical_params``
   * - ``set_natural_params(theta)``
     - method
     - Forward to ``_joint.set_natural_params``
   * - ``set_expectation_params(eta)``
     - method
     - Forward to ``_joint.set_expectation_params``
   * - ``natural_params``
     - property
     - Delegates to ``_joint.natural_params``
   * - ``classical_params``
     - property
     - Delegates to ``_joint.classical_params``
   * - ``expectation_params``
     - property
     - Delegates to ``_joint.expectation_params``
   * - ``joint``
     - property
     - Access the underlying ``JointNormalMixture``
   * - ``d``
     - property
     - Dimension of :math:`X`
   * - ``mixing_distribution``
     - property
     - The mixing distribution of :math:`Y`, built from joint named attrs
   * - ``pdf(x)``
     - method
     - Marginal PDF :math:`f(x)`
   * - ``logpdf(x)``
     - method
     - Marginal log PDF :math:`\log f(x)`
   * - ``rvs(size, random_state)``
     - method
     - Sample :math:`X` only (marginal)
   * - ``pdf_joint(x, y)``
     - method
     - Convenience alias for ``joint.pdf(x, y)``
   * - ``logpdf_joint(x, y)``
     - method
     - Convenience alias for ``joint.logpdf(x, y)``
   * - ``rvs_joint(size, random_state)``
     - method
     - Sample ``(X, Y)`` from the joint
   * - ``mean()``
     - method
     - :math:`E[X]` of the marginal
   * - ``var()``
     - method
     - :math:`\text{Var}[X]` of the marginal
   * - ``cov()``
     - method
     - :math:`\text{Cov}[X]` of the marginal
   * - ``fit(X, **kwargs)``
     - method
     - EM algorithm, returns ``self``
   * - ``fit_complete(X, Y, **kwargs)``
     - method
     - Fit from complete ``(X, Y)`` data

Subclass contract (internal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Purpose
   * - ``_create_joint_distribution()``
     - Factory: return a new ``JointNormalMixture`` instance
   * - ``_marginal_logpdf(x)``
     - Compute the closed-form marginal :math:`\log f(x)` (Bessel functions)
   * - ``_conditional_expectation_y_given_x(x)``
     - Compute :math:`E[g(Y)|X=x]` for the EM E-step

**Which methods are overridden by each child** (the analytical closed forms
differ per mixing distribution):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Overriding subclasses
   * - ``_marginal_logpdf(x)``
     - All four (VG, NInvG, NIG, GH) -- different Bessel-function formulae
   * - ``_conditional_expectation_y_given_x(x)``
     - All four -- different GIG conditional distributions
   * - ``_initialize_params(X)``
     - All four -- method-of-moments initialisation for EM
   * - ``fit(X, **kwargs)``
     - All four -- EM algorithm with distribution-specific M-step


Parameter Flow
--------------

.. code-block:: text

   ┌────────────────────────────────────────────────────────────┐
   │  Setters (write to named attrs)                           │
   │                                                           │
   │  set_classical_params ──► _set_from_classical             │
   │  set_natural_params   ──► _set_from_natural               │
   │  set_expectation_params ► _expectation_to_natural         │
   │                            then _set_from_natural         │
   │  (EM fast path)       ──► _set_internal                   │
   └──────────────────┬─────────────────────────────────────────┘
                      │
                      ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Named Internal Attributes (source of truth)              │
   │                                                           │
   │  _mu, _gamma, _L_Sigma, _alpha, _beta, _delta, _eta, ... │
   └──────────────────┬─────────────────────────────────────────┘
                      │ _invalidate_cache()
                      ▼
   ┌────────────────────────────────────────────────────────────┐
   │  Cached Properties (lazy, computed on demand)             │
   │                                                           │
   │  classical_params  ◄── _compute_classical_params()        │
   │  natural_params    ◄── _compute_natural_params()          │
   │  expectation_params◄── _compute_expectation_params()      │
   │  log_det_Sigma, L_Sigma_inv, gamma_mahal_sq               │
   └────────────────────────────────────────────────────────────┘

All three setter paths are **independent**: ``_set_from_classical`` never calls
``_set_from_natural``.  Each writes directly to named attributes.

The EM fast path ``_set_internal(mu, gamma, L_sigma, **mixing_kw)`` accepts
pre-computed Cholesky factors, bypassing all matrix decompositions.


Concrete Distributions
----------------------

Univariate (ExponentialFamily)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Class
     - Classical params
     - Natural params
     - Expectation params
   * - ``Exponential``
     - :math:`\lambda`
     - :math:`\theta = -\lambda`
     - :math:`\eta = 1/\lambda`
   * - ``Gamma``
     - :math:`\alpha, \beta`
     - :math:`[\alpha-1, -\beta]`
     - :math:`[\psi(\alpha)-\log\beta,\; \alpha/\beta]`
   * - ``InverseGamma``
     - :math:`\alpha, \beta`
     - :math:`[\beta, -(\alpha+1)]`
     - :math:`[-\psi(\alpha)+\log\beta,\; \beta/(\alpha-1)]`
   * - ``InverseGaussian``
     - :math:`\mu, \lambda`
     - :math:`[-\lambda/(2\mu^2),\; -\lambda/2]`
     - :math:`[\mu,\; 1/\mu + 1/\lambda]`
   * - ``GIG``
     - :math:`p, a, b`
     - :math:`[p-1, -b/2, -a/2]`
     - :math:`[E[\log Y],\; E[1/Y],\; E[Y]]`

Multivariate (ExponentialFamily)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MultivariateNormal`` stores ``_mu`` (mean) and ``_L_Sigma`` (lower Cholesky of
:math:`\Sigma`).  Overrides ``logpdf`` with a Cholesky-based implementation
using ``solve_triangular``.

Mixture distributions
~~~~~~~~~~~~~~~~~~~~~

Each mixture has a **marginal** class (``NormalMixture`` subclass) and a
**joint** class (``JointNormalMixture`` subclass):

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Family
     - Marginal class
     - Joint class
     - Mixing distribution
   * - Variance Gamma
     - ``VarianceGamma``
     - ``JointVarianceGamma``
     - :math:`Y \sim \text{Gamma}(\alpha, \beta)`
   * - Normal-Inverse Gamma
     - ``NormalInverseGamma``
     - ``JointNormalInverseGamma``
     - :math:`Y \sim \text{InvGamma}(\alpha, \beta)`
   * - Normal-Inverse Gaussian
     - ``NormalInverseGaussian``
     - ``JointNormalInverseGaussian``
     - :math:`Y \sim \text{InvGaussian}(\delta, \eta)`
   * - Generalized Hyperbolic
     - ``GeneralizedHyperbolic``
     - ``JointGeneralizedHyperbolic``
     - :math:`Y \sim \text{GIG}(p, a, b)`

The marginal class is what users interact with.  It provides ``pdf``,
``logpdf``, ``rvs``, ``fit``, and delegates exponential family operations
to the joint via the ``.joint`` property.
