API Reference
=============

.. toctree::
   :maxdepth: 1
   :hidden:

   distributions
   mixtures
   fitting
   finance
   utils

Reference pages, one per subpackage:

- :doc:`distributions` — the nine GH-family distributions (univariate,
  multivariate, and their joint/marginal/univariate-wrapper/factor variants)
- :doc:`mixtures` — the ``JointNormalMixture`` / ``NormalMixture`` /
  ``FactorNormalMixture`` base classes that every mixture distribution builds on
- :doc:`fitting` — EM fitters, solvers, and the incremental-EM
  :math:`\eta`-update machinery
- :doc:`finance` — portfolio projection, risk measures, and mean-risk
  optimization
- :doc:`utils` — Bessel functions, constants, sampling, and plotting helpers

Base Classes
------------

.. module:: normix.exponential_family

.. autoclass:: normix.exponential_family.ExponentialFamily
   :members:
   :undoc-members:
   :show-inheritance:

Divergences
-----------

.. automodule:: normix.divergences
   :members:
   :undoc-members:
