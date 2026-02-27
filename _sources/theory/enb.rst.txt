Effective Number of Bets and Minimum Torsion
============================================

This section reviews the effective number of bets (ENB) and the minimum
torsion approach for measuring portfolio diversification, following
[Meucci2010]_ and [Meucci2014]_.

Variance-Based Risk Decomposition
----------------------------------

Let :math:`X \in \mathbb{R}^n` be a random vector of asset returns with
covariance :math:`\Sigma`, and let :math:`w \in \mathbb{R}^n` with
:math:`w^\top \mathbf{e} = 1` be the portfolio weights. The portfolio
variance is:

.. math::

   r_{\operatorname{Var}}(w) = w^\top \Sigma w.

The gradient is :math:`\nabla r_{\operatorname{Var}}(w) = 2 \Sigma w`, so the
marginal contribution of asset :math:`k` to variance is
:math:`w_k (\Sigma w)_k`. These contributions are generally **not**
independent.

Uncorrelated Factor Decomposition
---------------------------------

To obtain independent contributions, we seek an invertible matrix
:math:`T \in \mathbb{R}^{n \times n}` such that :math:`T \Sigma T^\top = D`
is diagonal. Let :math:`Y = TX` be the transformed returns and
:math:`v = (T^\top)^{-1} w` the adjusted weights. Then:

.. math::

   w^\top X = v^\top Y, \qquad
   w^\top \Sigma w = v^\top D v = \sum_{k=1}^n d_k v_k^2,

where :math:`d_k` are the diagonal entries of :math:`D`. The risk
contributions :math:`d_k v_k^2` are now independent.

Effective Number of Bets
------------------------

The **normalized risk contributions** form a discrete distribution:

.. math::
   :label: enb-weights

   p_k = \frac{d_k v_k^2}{w^\top \Sigma w}, \quad k = 1, \ldots, n.

Since :math:`p_k \geq 0` and :math:`\sum_k p_k = 1`, the **ENB** is defined
as the exponential entropy:

.. math::

   N = \exp\!\left(-\sum_{k=1}^n p_k \log p_k\right).

The ENB ranges from 1 (risk concentrated in one factor) to :math:`n`
(equally distributed among all factors). Equivalently,
:math:`-\log N` is proportional to the KL divergence between
:math:`\{p_k\}` and the uniform distribution.

Characterizing Diagonalizations
-------------------------------

Let :math:`C = U S U^\top` be the eigendecomposition of the correlation matrix
:math:`C = \operatorname{diag}(\Sigma)^{-1/2} \Sigma \,
\operatorname{diag}(\Sigma)^{-1/2}`, where :math:`U` is orthogonal and
:math:`S` is diagonal.

**Proposition.** Let :math:`\Sigma` be positive definite and :math:`T` be
invertible with :math:`T \Sigma T^\top = D` diagonal. Then there exists an
orthogonal matrix :math:`V` such that:

.. math::
   :label: T-representation

   T = D^{1/2} V S^{-1/2} U^\top
   \operatorname{diag}(\Sigma)^{-1/2}.

**Lemma.** The ENB :math:`N` is independent of the choice of :math:`D`.

*Proof.* Let :math:`u = V S^{-1/2} U^\top
\operatorname{diag}(\Sigma)^{1/2} w` and :math:`v = D^{-1/2} u`. Then
:math:`p_k = u_k^2 / (w^\top \Sigma w)`, which does not depend on
:math:`D`. :math:`\square`

Therefore, it suffices to choose only the orthogonal matrix :math:`V`.
Since :math:`V` is a rotation, it can map the vector
:math:`S^{-1/2} U^\top \operatorname{diag}(\Sigma)^{1/2} w` to any
direction. In particular:

- Choosing :math:`V` so that all :math:`v_k` are equal gives :math:`N = n` (maximal diversification).
- Choosing :math:`V` to concentrate on one component gives :math:`N = 1`.

Thus, the diagonalization :math:`T` must be chosen carefully.

Minimum Torsion
---------------

The **minimum torsion** approach [Meucci2014]_ selects :math:`T` to minimize
the change from the original weights. The rationale is that if :math:`w` is
close to equally weighted, then :math:`v = (T^\top)^{-1} w` should also be
close to equally weighted.

The degree of change is measured by the **normalized tracking error**:

.. math::

   \operatorname{NTE}(T) = \sqrt{\frac{1}{n}
   \operatorname{tr}\!\left(\operatorname{diag}(\Sigma)^{-1/2}
   (T - I) \Sigma (T - I)^\top
   \operatorname{diag}(\Sigma)^{-1/2}\right)}.

Using representation :eq:`T-representation`, the minimization problem becomes:

.. math::

   \min_{D, V} \; \operatorname{tr}\!\left(D
   - 2 D^{1/2} V S^{1/2} U^\top\right)
   \quad \text{s.t.} \quad
   D \text{ diagonal}, \; V \text{ orthogonal}.

If :math:`D` is fixed to the identity matrix, the optimal solution is simply
:math:`V^* = U`, giving the **constrained minimum torsion** transformation:

.. math::

   T_{MT} = U S^{-1/2} U^\top
   \operatorname{diag}(\Sigma)^{-1/2}.

For the general case (unconstrained :math:`D`), an iterative algorithm that
converges rapidly is described in [Meucci2014]_.

References
----------

.. [Meucci2010] Meucci, A. (2010). Managing diversification.
   *Risk Magazine*, 22(5), 74-79.

.. [Meucci2014] Meucci, A., Santangelo, A., & Deguest, R. (2014).
   Measuring portfolio diversification based on optimized uncorrelated
   factors. *SSRN Electronic Journal*.
