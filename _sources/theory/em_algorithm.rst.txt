EM Algorithm for Generalized Hyperbolic Distributions
=====================================================

The Expectation-Maximization (EM) algorithm is a classical iterative method for
fitting data with hidden (latent) variables [Dempster1977]_. This section
describes the EM algorithm for the Generalized Hyperbolic (GH) distribution,
following the framework in [Hu2005]_ with some modifications.

Overview
--------

The key insight is that while the marginal GH distribution :math:`f(x)` is not
an exponential family, the joint distribution :math:`f(x, y)` of the observed
variable :math:`X` and the latent mixing variable :math:`Y` **is** an exponential
family. This makes the EM algorithm particularly elegant.

Our approach differs from previous works in several ways:

1. We use general convex optimization to solve the GIG MLE directly without
   fixing the parameter :math:`p` (called :math:`\lambda` in some references).

2. Constraints on GIG parameters are unnecessary since the optimization is
   stable under the Hellinger distance.

3. The best way to regularize GH parameters is to fix :math:`|\Sigma| = 1`,
   and this can be done at the end of each EM iteration without affecting
   convergence.

Conditional Distribution of Y given X
-------------------------------------

Recall that a GH random vector :math:`X` can be expressed as:

.. math::

   X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z

where :math:`Z \sim N(0, \Sigma)` is independent of :math:`Y \sim \text{GIG}(p, a, b)`.

Given the joint density :math:`f(x, y)`, we can compute the conditional density
of :math:`Y` given :math:`X`:

.. math::

   f(y | x, \theta) &= \frac{f(x, y | \theta)}{f(x | \theta)} \\
   &\propto y^{p - 1 - d/2} \exp\left(-\frac{1}{2}(x-\mu-\gamma y)^\top \Sigma^{-1}
   (x-\mu-\gamma y) y^{-1} - \frac{1}{2}(b y^{-1} + a y)\right) \\
   &\propto y^{p - 1 - d/2} \exp\left(-\frac{1}{2}\left(b + (x-\mu)^\top \Sigma^{-1}(x-\mu)\right) y^{-1}
   - \frac{1}{2}\left(a + \gamma^\top \Sigma^{-1} \gamma\right) y\right)

where :math:`\theta = (\mu, \gamma, \Sigma, p, a, b)` denotes the full parameter set.

This is a GIG distribution with parameters:

.. math::

   Y | X = x \sim \text{GIG}\left(p - \frac{d}{2}, \,
   a + \gamma^\top \Sigma^{-1} \gamma, \,
   b + (x-\mu)^\top \Sigma^{-1}(x-\mu)\right)

Conditional Expectations
------------------------

Using the GIG moment formula, we obtain the conditional expectations needed for
the E-step:

.. math::
   :label: gig-moment-cond

   E[Y^\alpha | X = x, \theta] = \left(\sqrt{\frac{b + (x-\mu)^\top \Sigma^{-1}(x-\mu)}
   {a + \gamma^\top \Sigma^{-1} \gamma}}\right)^\alpha
   \frac{K_{p - d/2 + \alpha}\left(\sqrt{(b + q(x))(a + \tilde{q})}\right)}
   {K_{p - d/2}\left(\sqrt{(b + q(x))(a + \tilde{q})}\right)}

where:

- :math:`q(x) = (x-\mu)^\top \Sigma^{-1}(x-\mu)` is the squared Mahalanobis distance
- :math:`\tilde{q} = \gamma^\top \Sigma^{-1} \gamma`

For the log moment:

.. math::

   E[\log Y | X = x, \theta] = \left.\frac{\partial}{\partial \alpha}
   E[Y^\alpha | X = x, \theta]\right|_{\alpha=0}

This derivative can be computed numerically.

The EM Algorithm
----------------

Let :math:`x_1, \ldots, x_n \in \mathbb{R}^d` be i.i.d. sample data. Given initial
parameters :math:`\theta_0 = (\mu_0, \gamma_0, \Sigma_0, p_0, a_0, b_0)`, the EM
algorithm iterates between two steps.

E-Step
~~~~~~

The :math:`(k+1)`-th E-step computes the average conditional expectations of the
sufficient statistics:

.. math::
   :label: e-step

   \hat{\eta}_1^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[Y^{-1} | X = x_j, \theta_k] \\
   \hat{\eta}_2^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[Y | X = x_j, \theta_k] \\
   \hat{\eta}_3^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[\log Y | X = x_j, \theta_k] \\
   \hat{\eta}_4^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j \\
   \hat{\eta}_5^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j E[Y^{-1} | X = x_j, \theta_k] \\
   \hat{\eta}_6^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j x_j^\top E[Y^{-1} | X = x_j, \theta_k]

where the conditional expectations are computed using :eq:`gig-moment-cond`.

Note that :math:`(\hat{\eta}_1^{(k)}, \hat{\eta}_2^{(k)}, \hat{\eta}_3^{(k)})` are
estimates of the GIG expectation parameters, and
:math:`(\hat{\eta}_4^{(k)}, \hat{\eta}_5^{(k)}, \hat{\eta}_6^{(k)})` are estimates
of the normal component expectation parameters.

M-Step
~~~~~~

The M-step solves the optimization problem:

.. math::

   \theta_{k+1} = \arg\max_\theta \sum_{j=1}^n E[\log f(X, Y | \theta) | X = x_j, \theta_k]

This is equivalent to maximizing the joint GH log-likelihood at the estimated
expectation parameters:

.. math::

   \theta_{k+1} = \arg\max_\theta L_{\text{GH}}(\mu, \gamma, \Sigma, p, a, b |
   \hat{\eta}_1^{(k)}, \hat{\eta}_2^{(k)}, \hat{\eta}_3^{(k)},
   \hat{\eta}_4^{(k)}, \hat{\eta}_5^{(k)}, \hat{\eta}_6^{(k)})

The closed-form solutions are (see :doc:`gh` for derivation):

.. math::
   :label: m-step

   \mu_{k+1} &= \frac{\hat{\eta}_4^{(k)} - \hat{\eta}_2^{(k)} \hat{\eta}_5^{(k)}}
   {1 - \hat{\eta}_1^{(k)} \hat{\eta}_2^{(k)}} \\
   \gamma_{k+1} &= \frac{\hat{\eta}_5^{(k)} - \hat{\eta}_1^{(k)} \hat{\eta}_4^{(k)}}
   {1 - \hat{\eta}_1^{(k)} \hat{\eta}_2^{(k)}} \\
   \Sigma_{k+1} &= \hat{\eta}_6^{(k)} - \hat{\eta}_5^{(k)} \mu_{k+1}^\top
   - \mu_{k+1} (\hat{\eta}_5^{(k)})^\top + \hat{\eta}_1^{(k)} \mu_{k+1} \mu_{k+1}^\top
   - \hat{\eta}_2^{(k)} \gamma_{k+1} \gamma_{k+1}^\top \\
   (p_{k+1}, a_{k+1}, b_{k+1}) &= \arg\max_{p, a, b}
   L_{\text{GIG}}(p, a, b | \hat{\eta}_1^{(k)}, \hat{\eta}_2^{(k)}, \hat{\eta}_3^{(k)})

The first three equations have closed-form solutions, while the GIG parameters
require numerical optimization.

Parameter Regularization
------------------------

The GH model is not identifiable since the parameter sets
:math:`(\mu, \gamma/c, \Sigma/c, p, c \cdot b, a/c)` give the same distribution
for any :math:`c > 0`. A good way to regularize is to fix :math:`|\Sigma| = 1`.

Instead of adding a constraint to the optimization, we can rescale parameters
at the end of each M-step:

.. math::

   (\mu_k, \gamma_k, \Sigma_k, p_k, a_k, b_k) \rightarrow
   \left(\mu_k, |\Sigma_k|^{-1/d} \gamma_k, |\Sigma_k|^{-1/d} \Sigma_k,
   p_k, |\Sigma_k|^{-1/d} a_k, |\Sigma_k|^{1/d} b_k\right)

This rescaling does not affect the convergence of the EM algorithm:

**Proposition.** If we write the :math:`(k+1)`-th iteration as a function
:math:`f`, i.e.,

.. math::

   (\mu_{k+1}, \gamma_{k+1}, \Sigma_{k+1}, p_{k+1}, a_{k+1}, b_{k+1})
   = f(\mu_k, \gamma_k, \Sigma_k, p_k, a_k, b_k)

then for any :math:`c > 0`:

.. math::

   (\mu_{k+1}, c \gamma_{k+1}, c \Sigma_{k+1}, p_{k+1}, a_{k+1}/c, c \, b_{k+1})
   = f(\mu_k, c \gamma_k, c \Sigma_k, p_k, a_k/c, c \, b_k)

*Proof.* A direct computation using :eq:`gig-moment-cond` shows that:

.. math::

   E[Y^\alpha | X = x, \tilde{\theta}_k] &= E[Y^\alpha | X = x, \theta_k] / c^\alpha \\
   E[\log Y | X = x, \tilde{\theta}_k] &= E[\log Y | X = x, \theta_k] - \log c

where :math:`\tilde{\theta}_k = (\mu_k, c\gamma_k, c\Sigma_k, p_k, a_k/c, c \, b_k)`.

Thus if :math:`\hat{\eta}_1^{(k)}, \ldots, \hat{\eta}_6^{(k)}` are the E-step
outputs given :math:`\theta_k`, then
:math:`c \hat{\eta}_1^{(k)}, \hat{\eta}_2^{(k)}/c, \hat{\eta}_3^{(k)} - \log c,
\hat{\eta}_4^{(k)}, c \hat{\eta}_5^{(k)}, c \hat{\eta}_6^{(k)}` are the
corresponding outputs given :math:`\tilde{\theta}_k`. The result follows by
applying these to :eq:`m-step`. ∎

MCECM Algorithm
---------------

An alternative is the Multi-Cycle Expectation Conditional Maximization (MCECM)
algorithm [McNeil2010]_. Unlike the EM algorithm which updates all parameters
via :eq:`m-step`, MCECM proceeds in two cycles:

**Cycle 1:** Compute :math:`\mu_{k+1}`, :math:`\gamma_{k+1}`, :math:`\Sigma_{k+1}`
from the first three equations in :eq:`m-step`, then set
:math:`\Sigma_{k+1} \leftarrow \Sigma_{k+1} / |\Sigma_{k+1}|^{1/d}`.

**Cycle 2:** Set :math:`\tilde{\theta}_{k+1} = (\mu_{k+1}, \gamma_{k+1}, \Sigma_{k+1}, p_k, a_k, b_k)`
and recompute the GIG expectation parameters:

.. math::

   \tilde{\eta}_1^{(k+1)} &= \frac{1}{n} \sum_{j=1}^n E[Y^{-1} | X = x_j, \tilde{\theta}_{k+1}] \\
   \tilde{\eta}_2^{(k+1)} &= \frac{1}{n} \sum_{j=1}^n E[Y | X = x_j, \tilde{\theta}_{k+1}] \\
   \tilde{\eta}_3^{(k+1)} &= \frac{1}{n} \sum_{j=1}^n E[\log Y | X = x_j, \tilde{\theta}_{k+1}]

Then update the GIG parameters:

.. math::

   (p_{k+1}, a_{k+1}, b_{k+1}) = \arg\max_{p, a, b}
   L_{\text{GIG}}(p, a, b | \tilde{\eta}_1^{(k+1)}, \tilde{\eta}_2^{(k+1)}, \tilde{\eta}_3^{(k+1)})

Both algorithms converge to the MLE and have similar computational efficiency.

Special Cases
-------------

For special cases of the GH distribution, the EM algorithm simplifies because
the mixing distribution has fewer parameters and the M-step has closed-form
solutions.

Variance Gamma (VG)
~~~~~~~~~~~~~~~~~~~

The Variance Gamma distribution uses a **Gamma** mixing distribution:
:math:`Y \sim \text{Gamma}(\alpha, \beta)`. The GIG reduces to Gamma when
:math:`b \to 0`.

**Sufficient statistics for Y:**

.. math::

   t_Y(y) = (\log y, \, y)

**Expectation parameters:**

.. math::

   \eta_1 &= E[\log Y] = \psi(\alpha) - \log \beta \\
   \eta_2 &= E[Y] = \alpha / \beta

where :math:`\psi` is the digamma function.

**E-step:** The conditional distribution :math:`Y | X = x` is GIG with
:math:`b = (x-\mu)^\top \Sigma^{-1}(x-\mu)`, so we still need to compute
GIG conditional expectations. However, the M-step simplifies.

**M-step for Gamma parameters:** Given :math:`\hat{\eta}_1 = E[\log Y]` and
:math:`\hat{\eta}_2 = E[Y]`, we solve:

.. math::

   \psi(\alpha) - \log(\alpha / \hat{\eta}_2) = \hat{\eta}_1

This is a single-variable equation that can be solved efficiently using
Newton's method:

.. math::

   \alpha^{(t+1)} = \alpha^{(t)} - \frac{\psi(\alpha^{(t)}) - \log \alpha^{(t)} - (\hat{\eta}_1 - \log \hat{\eta}_2)}
   {\psi'(\alpha^{(t)}) - 1/\alpha^{(t)}}

Then :math:`\beta = \alpha / \hat{\eta}_2`.

Normal-Inverse Gaussian (NIG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Normal-Inverse Gaussian distribution uses an **Inverse Gaussian** mixing
distribution: :math:`Y \sim \text{IG}(\mu_Y, \lambda)`. This corresponds to
GIG with :math:`p = -1/2`.

**Sufficient statistics for Y:**

.. math::

   t_Y(y) = (y, \, y^{-1})

**Expectation parameters:**

.. math::

   \eta_1 &= E[Y] = \mu_Y \\
   \eta_2 &= E[Y^{-1}] = 1/\mu_Y + 1/\lambda

**M-step for Inverse Gaussian parameters:** Given :math:`\hat{\eta}_1 = E[Y]`
and :math:`\hat{\eta}_2 = E[Y^{-1}]`, the closed-form solution is:

.. math::

   \mu_Y &= \hat{\eta}_1 \\
   \lambda &= \frac{1}{\hat{\eta}_2 - 1/\hat{\eta}_1}

This is fully analytical with no optimization required.

Normal-Inverse Gamma (NInvG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Normal-Inverse Gamma distribution uses an **Inverse Gamma** mixing
distribution: :math:`Y \sim \text{InvGamma}(\alpha, \beta)`. This corresponds
to GIG with :math:`a \to 0` (or equivalently, :math:`p < 0` with :math:`a = 0`).

**Sufficient statistics for Y:**

.. math::

   t_Y(y) = (\log y, \, y^{-1})

**Expectation parameters:**

.. math::

   \eta_1 &= E[\log Y] = \log \beta - \psi(\alpha) \\
   \eta_2 &= E[Y^{-1}] = \alpha / \beta

**M-step for Inverse Gamma parameters:** Given :math:`\hat{\eta}_1 = E[\log Y]`
and :math:`\hat{\eta}_2 = E[Y^{-1}]`, we solve:

.. math::

   \log(\alpha / \hat{\eta}_2) - \psi(\alpha) = \hat{\eta}_1

This is again a single-variable equation solved by Newton's method:

.. math::

   \alpha^{(t+1)} = \alpha^{(t)} - \frac{\log \alpha^{(t)} - \psi(\alpha^{(t)}) - (\hat{\eta}_1 + \log \hat{\eta}_2)}
   {\frac{1}{\alpha^{(t)}} - \psi'(\alpha^{(t)})}

Then :math:`\beta = \alpha / \hat{\eta}_2`.

Summary of Special Cases
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: EM Algorithm Simplifications for Special Cases
   :header-rows: 1
   :widths: 20 25 25 30

   * - Distribution
     - Mixing Dist.
     - Sufficient Stats
     - M-Step Complexity
   * - GH (general)
     - GIG(:math:`p, a, b`)
     - :math:`(\log y, y^{-1}, y)`
     - 3D optimization
   * - Variance Gamma
     - Gamma(:math:`\alpha, \beta`)
     - :math:`(\log y, y)`
     - 1D Newton
   * - Normal-Inv Gaussian
     - InvGauss(:math:`\mu, \lambda`)
     - :math:`(y, y^{-1})`
     - Closed-form
   * - Normal-Inv Gamma
     - InvGamma(:math:`\alpha, \beta`)
     - :math:`(\log y, y^{-1})`
     - 1D Newton

The Normal-Inverse Gaussian case is particularly attractive because the M-step
is fully analytical. The Variance Gamma and Normal-Inverse Gamma cases require
only 1D optimization (Newton's method), which is much faster and more stable
than the 3D optimization required for the general GH case.

Numerical Considerations
------------------------

High-Dimensional Issues
~~~~~~~~~~~~~~~~~~~~~~~

When the dimension :math:`d` is large (e.g., 500), computing the modified Bessel
functions in :eq:`gig-moment-cond` can be challenging. For large :math:`d`,
:math:`K_{p + d/2 + \alpha}` may overflow or underflow. This is addressed using
log-space computations in the :mod:`normix.utils.bessel` module.

Matrix Conditioning
~~~~~~~~~~~~~~~~~~~

The formula for :math:`\Sigma_{k+1}` in :eq:`m-step` has the same numerical
issues as the sample covariance: the condition number can be huge when the
sample size is relatively small. Shrinkage estimators (such as penalized
likelihood methods) can help improve the conditioning of :math:`\Sigma`.

Implementation in normix
----------------------

In ``normix``, the EM algorithm is implemented in the :meth:`fit` method of
:class:`~normix.base.NormalMixture` subclasses. The key methods are:

- :meth:`_conditional_expectation_y_given_x`: Computes :math:`E[Y^{-1}|X]`,
  :math:`E[Y|X]`, :math:`E[\log Y|X]` for the E-step
- :meth:`joint.set_expectation_params`: Sets parameters from expectation
  parameters for the M-step
- :meth:`joint._expectation_to_natural`: Converts expectation to natural
  parameters (solves the GIG optimization)

References
----------

.. [Dempster1977] Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).
   Maximum likelihood from incomplete data via the EM algorithm.
   *Journal of the Royal Statistical Society: Series B*, 39(1), 1-38.
