Shrinkage with Penalized Likelihood
====================================

By setting :math:`|\Sigma| = 1` we ensure that the covariance matrix is
numerically invertible in each iteration of the EM algorithm. However, this
does not guarantee that the matrix inversion is well-conditioned. The formula
for :math:`\Sigma_{k+1}` in :eq:`m-step` has the same problem as the sample
covariance: the condition number of :math:`\Sigma_k` can be huge when the
sample size is relatively small. Shrinkage estimators based on penalized
likelihood can improve the conditioning.

Penalized Likelihood Framework
------------------------------

Consider a general exponential family with density:

.. math::

   f(x, y | \theta) = h(x, y) \exp\left(\theta^\top t(x, y) - \psi(\theta)\right),

where

.. math::

   \psi(\theta) = \log \int h(x, y) \exp\left(\theta^\top t(x, y)\right) dx \, dy.

The Kullback-Leibler divergence between two members of this family is:

.. math::

   D_{KL}(\theta_1 \| \theta_2) = \psi(\theta_2) - \psi(\theta_1)
   + \eta_1^\top (\theta_1 - \theta_2),

where :math:`\eta_1 = E[t(X, Y) | \theta_1] = \nabla\psi(\theta_1)`. This is
also the Bregman divergence with potential function :math:`\psi`.

Assume that :math:`x` is observable while :math:`y` is hidden. Given a sample
:math:`x_1, \ldots, x_n` and a prior parameter :math:`\theta_0`, the
**penalized likelihood** maximization is:

.. math::

   \max_\theta \frac{1}{n} \sum_{j=1}^n \log f(x_j | \theta)
   - \tau \, D_{KL}(\theta_0 \| \theta),

where :math:`f(x|\theta) = \int f(x,y|\theta) \, dy` is the marginal density
and :math:`\tau \geq 0` controls the amount of shrinkage. When :math:`\tau = 0`
this reduces to standard maximum likelihood. As :math:`\tau` increases, the
solution is pulled toward :math:`\theta_0`.

Penalized EM Algorithm
----------------------

This problem can be solved iteratively by the following EM algorithm:

.. math::

   \theta_{k+1} = \arg\max_\theta \frac{1}{n} \sum_{j=1}^n
   E\!\left[\log f(X, Y | \theta) \mid X = x_j, \theta_k\right]
   - \tau \, D_{KL}(\theta_0 \| \theta).

One can show that the penalized likelihood is non-decreasing:

.. math::

   &\frac{1}{n} \sum_{j=1}^n \log f(x_j | \theta_{k+1})
   - \tau \, D_{KL}(\theta_0 \| \theta_{k+1}) \\
   &\geq \frac{1}{n} \sum_{j=1}^n \log f(x_j | \theta_k)
   - \tau \, D_{KL}(\theta_0 \| \theta_k),

since the difference includes a non-negative KL divergence between
conditional distributions :math:`f(y|x,\theta_k)` and
:math:`f(y|x,\theta_{k+1})`.

Using the exponential family representation, the penalized M-step reduces to:

.. math::

   \theta_{k+1} = \arg\max_\theta \,
   \theta^\top \left(\frac{1}{n} \sum_{j=1}^n
   E[t(X, Y) | x_j, \theta_k] + \tau \, \eta_0\right)
   - (1 + \tau) \psi(\theta),

where :math:`\eta_0 = E[t(X, Y) | \theta_0]` are the expectation parameters
at the prior.

Shrunk Sufficient Statistics
----------------------------

As a result, the E-step computes **shrunk** sufficient statistics by
taking a convex combination of the conditional expectations and the prior
expectations:

.. math::

   \hat{\eta}_1^{(k)} &= \frac{1}{(1+\tau)n} \sum_{j=1}^n
   E[Y^{-1} | X = x_j, \theta_k] + \frac{\tau}{1+\tau} E[Y^{-1} | \theta_0] \\
   \hat{\eta}_2^{(k)} &= \frac{1}{(1+\tau)n} \sum_{j=1}^n
   E[Y | X = x_j, \theta_k] + \frac{\tau}{1+\tau} E[Y | \theta_0] \\
   \hat{\eta}_3^{(k)} &= \frac{1}{(1+\tau)n} \sum_{j=1}^n
   E[\log Y | X = x_j, \theta_k] + \frac{\tau}{1+\tau} E[\log Y | \theta_0] \\
   \hat{\eta}_4^{(k)} &= \frac{1}{(1+\tau)n} \sum_{j=1}^n x_j
   + \frac{\tau}{1+\tau} E[X | \theta_0] \\
   \hat{\eta}_5^{(k)} &= \frac{1}{(1+\tau)n} \sum_{j=1}^n
   x_j \, E[Y^{-1} | X = x_j, \theta_k]
   + \frac{\tau}{1+\tau} E[X Y^{-1} | \theta_0] \\
   \hat{\eta}_6^{(k)} &= \frac{1}{(1+\tau)n} \sum_{j=1}^n
   x_j x_j^\top E[Y^{-1} | X = x_j, \theta_k]
   + \frac{\tau}{1+\tau} E[X X^\top Y^{-1} | \theta_0]

where the **prior expectations** are computed from the GH expectation
parameters (see :eq:`gh-expectation`):

.. math::

   E[Y^\alpha | \theta_0] &= \left(\sqrt{\frac{b_0}{a_0}}\right)^\alpha
   \frac{K_{p_0 + \alpha}(\sqrt{a_0 b_0})}{K_{p_0}(\sqrt{a_0 b_0})} \\
   E[\log Y | \theta_0] &= \left.\frac{\partial}{\partial \alpha}
   E[Y^\alpha | \theta_0]\right|_{\alpha=0} \\
   E[X | \theta_0] &= \mu_0 + \gamma_0 \, E[Y | \theta_0] \\
   E[X Y^{-1} | \theta_0] &= \mu_0 \, E[Y^{-1} | \theta_0] + \gamma_0 \\
   E[X X^\top Y^{-1} | \theta_0] &= \Sigma_0
   + \mu_0 \mu_0^\top E[Y^{-1} | \theta_0]
   + \gamma_0 \gamma_0^\top E[Y | \theta_0]
   + \mu_0 \gamma_0^\top + \gamma_0 \mu_0^\top

The penalized maximum likelihood thus amounts to a **linear shrinkage** of the
conditional expectation parameters toward the prior :math:`\theta_0`. The
M-step is identical to the standard EM algorithm (see :eq:`m-step`).

Furthermore, the linear relationship between :math:`\Sigma_{k+1}` and
:math:`\hat{\eta}_6^{(k)}` implies that :math:`\Sigma_{k+1}` is shrunk
toward :math:`\Sigma_0` directly. By choosing an appropriate
:math:`\Sigma_0` (e.g., a well-conditioned target), one can improve the
conditioning of :math:`\Sigma_{k+1}` at each iteration.
