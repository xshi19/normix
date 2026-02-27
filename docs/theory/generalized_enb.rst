Generalized Effective Number of Bets
====================================

This section extends the ENB framework from :doc:`enb` to general convex
risk measures by diagonalizing the Hessian of the risk function.

Homogeneous Functions and Risk Decomposition
--------------------------------------------

**Definition.** A function :math:`r : \mathbb{R}^n \to \mathbb{R}` is
:math:`\tau`-**homogeneous** if :math:`r(tw) = t^\tau r(w)` for all
:math:`w \in \mathbb{R}^n` and :math:`t > 0`.

**Proposition** (Tasche [Tasche1999b]_). Let :math:`r` be a totally
differentiable, :math:`\tau`-homogeneous function with :math:`\tau \neq 0`.
Then:

1. :math:`\partial r / \partial w_k` is :math:`(\tau - 1)`-homogeneous.
2. Euler's theorem: :math:`\tau \, r(w) = w^\top \nabla r(w) = \sum_{k=1}^n w_k \, \frac{\partial r}{\partial w_k}(w)`.

This decomposes the risk :math:`r(w)` into marginal contributions
:math:`\frac{w_k}{\tau} \frac{\partial r}{\partial w_k}(w)`, analogous to
the variance case where :math:`\tau = 2`.

Local Diagonalization via the Hessian
-------------------------------------

For a general convex risk function, the marginal contributions
:math:`\frac{\partial r}{\partial w_k}` are not independent. To extract
locally independent contributions, we use the Taylor expansion:

.. math::

   r(w + \Delta w) \approx r(w) + \Delta w^\top \nabla r(w)
   + \frac{1}{2} \Delta w^\top H_r(w) \, \Delta w,

where :math:`H_r(w)` is the (positive semi-definite) Hessian matrix.

Let :math:`T(w)` diagonalize the Hessian: :math:`T(w) \, H_r(w) \, T(w)^\top
= D(w)`, and set :math:`v = (T(w)^\top)^{-1} w`. Using
:math:`(\tau - 1) \nabla r(w) = H_r(w) w` (from differentiating Euler's
identity), we obtain:

.. math::

   (\tau - 1) \, T(w) \nabla r(w) = D(w) \, v.

For :math:`\tau > 1`:

.. math::

   r(w) = \frac{1}{\tau} w^\top \nabla r(w)
   = \frac{1}{\tau(\tau - 1)} v^\top D(w) \, v
   = \sum_{k=1}^n \frac{d_k(w) \, v_k^2}{\tau(\tau - 1)},

where :math:`d_k(w)` are the diagonal entries of :math:`D(w)`. Since
:math:`H_r(w)` is positive semi-definite, we have :math:`d_k(w) \geq 0`,
which ensures that the risk contributions are non-negative when
:math:`\tau > 1`.

Local Independence
~~~~~~~~~~~~~~~~~~

The Taylor expansion in the transformed coordinates is:

.. math::

   r(w + \Delta w) &\approx r(w) + \frac{1}{\tau - 1} \Delta v^\top D(w) \, v
   + \frac{1}{\tau} \Delta v^\top D(w) \, \Delta v,

where :math:`\Delta v = (T(w)^\top)^{-1} \Delta w`. Each component of
:math:`\Delta v` has an approximately independent contribution to the change
:math:`r(w + \Delta w) - r(w)`, justifying the decomposition.

Generalized ENB
---------------

The generalized ENB is defined as:

.. math::

   p_k(w) &= \frac{d_k(w) \, v_k^2}{\tau(\tau - 1) \, r(w)},
   \quad k = 1, \ldots, n, \\
   N(w) &= \exp\!\left(-\sum_{k=1}^n p_k(w) \log p_k(w)\right).

Unlike the variance case, the Hessian :math:`H_r(w)` depends on :math:`w`,
so the transformation :math:`T(w)` must be recomputed for each portfolio.
However, the structural results from :doc:`enb` still apply:

Let

.. math::

   C(w) = \operatorname{diag}(H_r(w))^{-1/2} \, H_r(w) \,
   \operatorname{diag}(H_r(w))^{-1/2}

be the "correlation" of the Hessian, with eigendecomposition
:math:`C(w) = U(w) \, S(w) \, U(w)^\top`. Then :math:`T(w)` has the
representation:

.. math::

   T(w) = D^{1/2} V \, S(w)^{-1/2} U(w)^\top
   \operatorname{diag}(H_r(w))^{-1/2},

where :math:`D` is diagonal and :math:`V` is orthogonal. The ENB is again
independent of :math:`D`, and the **constrained minimum torsion**
transformation is:

.. math::

   T_{MT}(w) = U(w) \, S(w)^{-1/2} U(w)^\top
   \operatorname{diag}(H_r(w))^{-1/2}.

Application to Coherent Risk Measures
-------------------------------------

Given a coherent risk measure :math:`\rho`, define
:math:`r_\rho(w) = \rho(w^\top X)`. By positive homogeneity and
subadditivity, :math:`r_\rho` is convex and **1-homogeneous**
(:math:`\tau = 1`).

Since the generalized ENB requires :math:`\tau > 1`, we work with the
**squared risk** :math:`r_\rho^2(w)`, which is 2-homogeneous. This is
analogous to using variance (the square of standard deviation) in the
original ENB framework.

The Hessian of :math:`r_\rho^2(w)` can be computed from the CVaR gradient
and Hessian formulas in :doc:`cvar_derivatives`:

.. math::

   H_{r_\rho^2}(w) = 2 \nabla r_\rho(w) \, \nabla r_\rho(w)^\top
   + 2 \, r_\rho(w) \, H_{r_\rho}(w).

The CVaR-based ENB can reveal tail-risk concentrations that the
variance-based ENB misses. For example, in a portfolio of independent assets
where one has heavier tails, the variance-based ENB treats all assets equally
(since the covariance is diagonal), while the CVaR-based ENB correctly
assigns a larger risk contribution to the heavy-tailed asset.

References
----------

.. [Tasche1999b] Tasche, D. (1999). Risk contributions and performance
   measurement. Report of the Lehrstuhl für mathematische Statistik, TU München.
