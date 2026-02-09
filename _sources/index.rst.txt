normix Documentation
====================

**normix** is a Python package for Generalized Hyperbolic distributions and related distributions,
implemented as exponential families with an sklearn-style API.

Overview
--------

``normix`` provides a comprehensive, production-ready implementation of the Generalized Hyperbolic (GH)
distribution family, including:

- **Univariate distributions**: Exponential, Gamma, Inverse Gamma, Generalized Inverse Gaussian (GIG), Inverse Gaussian
- **Multivariate distributions**: Multivariate Normal
- **Mixture distributions**: Generalized Hyperbolic (GH), Normal Inverse Gaussian (NIG), Variance Gamma (VG), Normal Inverse Gamma (NInvG)

Key Features
------------

All distributions are implemented as **exponential families** with support for:

- Three parametrizations: **classical**, **natural**, and **expectation** parameters
- **sklearn-style API**: ``fit()`` returns self, method chaining supported
- Efficient **EM algorithms** for parameter estimation
- Joint distributions :math:`f(x,y)` and marginal distributions :math:`f(x)`

Mathematical Background
-----------------------

Exponential Families
~~~~~~~~~~~~~~~~~~~~

Distributions in exponential family form have the probability density:

.. math::

   p(x|\theta) = h(x) \exp(\theta^T t(x) - \psi(\theta))

where:

- :math:`\theta`: natural parameters (vector)
- :math:`t(x)`: sufficient statistics (vector)
- :math:`\psi(\theta)`: log partition function (cumulant generating function)
- :math:`h(x)`: base measure

**Key properties:**

- Expectation parameters: :math:`\eta = \nabla\psi(\theta) = E[t(X)]`
- Fisher information: :math:`I(\theta) = \nabla^2\psi(\theta) = \text{Cov}[t(X)]`
- MLE in closed form: :math:`\hat{\eta} = \frac{1}{n}\sum_{i=1}^n t(x_i)`

Generalized Hyperbolic as Normal Mixture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GH distribution can be represented as:

.. math::

   X|Y \sim N(\mu + \Gamma Y, \Sigma Y)

   Y \sim \text{GIG}(\lambda, \chi, \psi)

The marginal distribution :math:`f(x)` has a closed form involving modified Bessel functions
of the second kind :math:`K_\lambda(z)`.

Installation
------------

.. code-block:: bash

   pip install -e .

For development:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Example
-------------

.. code-block:: python

   from normix.distributions.univariate import Gamma
   import numpy as np

   # Create distribution from classical parameters
   dist = Gamma.from_classical_params(shape=2.0, rate=1.0)

   # Generate samples
   samples = dist.rvs(size=1000)

   # Fit distribution to data
   fitted_dist = Gamma().fit(samples)

   # Access parameters in different forms
   print(fitted_dist.classical_params)
   print(fitted_dist.natural_params)
   print(fitted_dist.expectation_params)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   demos
   theory/index
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
