Online EM Algorithm
===================

In this section we review the online EM algorithm for exponential families
with hidden data [Cappe2009]_, and apply it to the Generalized Hyperbolic
distribution.

Online EM for Exponential Families
----------------------------------

Consider the exponential family with density:

.. math::

   f(x, y | \theta) = h(x, y)
   \exp\!\left(\theta^\top t(x, y) - \psi(\theta)\right),

where :math:`x` is observable, :math:`y` is hidden, and

.. math::

   \psi(\theta) = \log \int h(x, y)
   \exp\!\left(\theta^\top t(x, y)\right) dx \, dy.

The expectation parameter is :math:`\eta = \nabla\psi(\theta)`, and the
Legendre dual of :math:`\psi` is
:math:`\phi(\eta) = \eta^\top \theta - \psi(\theta)`,
with :math:`\theta = \nabla\phi(\eta)`.

Regret Framework
~~~~~~~~~~~~~~~~

The parameter estimation process can be viewed as an online game. At time
:math:`t-1` we make a prediction :math:`\theta_{t-1}`. At time :math:`t` the
environment reveals an observation :math:`x_t`, and the loss is
:math:`l(x_t, \theta_{t-1}) = -\log f(x_t | \theta_{t-1})`. The **regret**
from :math:`t = 1` to :math:`T` is:

.. math::
   :label: regret-def

   r_T(\theta_0, \ldots, \theta_{T-1}) =
   -\sum_{t=1}^T \log f(x_t | \theta_{t-1})
   - \min_\theta \left(-\sum_{t=1}^T \log f(x_t | \theta)
   + \tau_0 \, D_\psi(\theta \| \theta_0)\right),

where :math:`D_\psi(\theta \| \theta_0) = \psi(\theta) - \psi(\theta_0)
- \nabla\psi(\theta_0)^\top (\theta - \theta_0)` is the Bregman divergence,
which coincides with the KL divergence :math:`D_{KL}(\theta_0 \| \theta)`.

Update Rules
~~~~~~~~~~~~

The online EM algorithm updates the expectation parameters as follows.
Starting from :math:`\eta_0 = \nabla\psi(\theta_0)`:

.. math::
   :label: online-em-update

   \eta_t &= \eta_{t-1} + \tau_t^{-1}
   \left(\bar{t}(x_t | \theta_{t-1}) - \eta_{t-1}\right), \\
   \theta_t &= \nabla\phi(\eta_t),

where

.. math::

   \bar{t}(x_t | \theta_{t-1}) := E[t(X, Y) | X = x_t, \theta_{t-1}]

is the conditional expectation of the sufficient statistics, and
:math:`\tau_t = \tau_0 + t` is the step size.

Regret Bound
~~~~~~~~~~~~~

**Theorem.** With :math:`\tau_t = \tau_0 + t`, the regret
:eq:`regret-def` of the online EM algorithm :eq:`online-em-update` is:

.. math::
   :label: regret-decomp

   r_T = \sum_{t=1}^T \tau_t \, D_\psi(\theta_{t-1} \| \theta_t)
   + \sum_{t=1}^T D_{KL}(x_t, \theta_{t-1} \| x_t, \theta_{ML})
   - \tau_T \, D_\phi(\eta_T \| \eta_{ML}),

where :math:`\theta_{ML}` is the penalized MLE:

.. math::

   \theta_{ML} = \arg\min_\theta \left(-\sum_{t=1}^T \log f(x_t | \theta)
   + \tau_0 \, D_\psi(\theta \| \theta_0)\right),

:math:`\eta_{ML} = \nabla\psi(\theta_{ML})`, and
:math:`D_{KL}(x_t, \theta_{t-1} \| x_t, \theta_{ML})` is the KL divergence
between the conditional densities :math:`f(y | x_t, \theta_{t-1})` and
:math:`f(y | x_t, \theta_{ML})`.

*Proof.* The first part of the regret is:

.. math::

   -\sum_{t=1}^T \log f(x_t | \theta_{t-1})
   &= -\sum_{t=1}^T E[\log f(X, Y | \theta_{t-1}) | X = x_t, \theta_{t-1}] \\
   &\quad + \sum_{t=1}^T E[\log f(Y | X, \theta_{t-1}) | X = x_t, \theta_{t-1}].

The conditional expectation of the joint log-density satisfies:

.. math::

   &\sum_{t=1}^T \left(\psi(\theta_{t-1})
   - \theta_{t-1}^\top \bar{t}(x_t | \theta_{t-1})\right) \\
   &= \sum_{t=1}^T \left(-\phi(\eta_{t-1})
   - \tau_t \theta_{t-1}^\top (\eta_t - \eta_{t-1})\right) \\
   &= \sum_{t=1}^T \tau_t D_\phi(\eta_t \| \eta_{t-1})
   + \tau_0 \phi(\eta_0) - \tau_T \phi(\eta_T) \\
   &= \sum_{t=1}^T \tau_t D_\psi(\theta_{t-1} \| \theta_t)
   + \tau_0 \phi(\eta_0) - \tau_T \phi(\eta_T),

using :math:`\phi(\eta) = \theta^\top \eta - \psi(\theta)` and
:math:`D_\phi(\eta_t \| \eta_{t-1}) = D_\psi(\theta_{t-1} \| \theta_t)`.

The key identity for the accumulated conditional expectations is:

.. math::

   \eta_T = \tau_T^{-1} \left(\tau_0 \eta_0
   + \sum_{t=1}^T \bar{t}(x_t | \theta_{t-1})\right).

Applying this to the second part of the regret:

.. math::

   &-\sum_{t=1}^T E[\log f(X, Y | \theta_{ML}) | X = x_t, \theta_{t-1}]
   + \tau_0 \, D_\psi(\theta_{ML} \| \theta_0) \\
   &= \tau_0 \phi(\eta_0) - \tau_T \phi(\eta_T)
   + \tau_T D_\phi(\eta_T \| \eta_{ML}).

Combining both parts yields :eq:`regret-decomp`. :math:`\square`

Application to the GH Distribution
-----------------------------------

For the joint GH distribution :eq:`gh-joint`, which is an exponential family,
the online EM algorithm updates the sufficient statistics as follows. Given
a sequence of observations :math:`x_1, x_2, \ldots, x_T`:

.. math::

   s_1^{(t+1)} &= s_1^{(t)} + \tau_{t+1}^{-1}
   \left(E[Y^{-1} | X = x_{t+1}, \theta_t] - s_1^{(t)}\right), \\
   s_2^{(t+1)} &= s_2^{(t)} + \tau_{t+1}^{-1}
   \left(E[Y | X = x_{t+1}, \theta_t] - s_2^{(t)}\right), \\
   s_3^{(t+1)} &= s_3^{(t)} + \tau_{t+1}^{-1}
   \left(E[\log Y | X = x_{t+1}, \theta_t] - s_3^{(t)}\right), \\
   s_4^{(t+1)} &= s_4^{(t)} + \tau_{t+1}^{-1}
   \left(x_{t+1} - s_4^{(t)}\right), \\
   s_5^{(t+1)} &= s_5^{(t)} + \tau_{t+1}^{-1}
   \left(x_{t+1} \, E[Y^{-1} | X = x_{t+1}, \theta_t] - s_5^{(t)}\right), \\
   s_6^{(t+1)} &= s_6^{(t)} + \tau_{t+1}^{-1}
   \left(x_{t+1} x_{t+1}^\top E[Y^{-1} | X = x_{t+1}, \theta_t]
   - s_6^{(t)}\right),

where the conditional expectations are given by :eq:`gig-moment-cond`.

The parameters :math:`\theta_t = (\mu_t, \gamma_t, \Sigma_t, p_t, a_t, b_t)`
are then recovered from the sufficient statistics using the same formulas as
the M-step (see :eq:`m-step`):

.. math::

   \mu_t &= \frac{s_4^{(t)} - s_2^{(t)} s_5^{(t)}}
   {1 - s_1^{(t)} s_2^{(t)}}, \\
   \gamma_t &= \frac{s_5^{(t)} - s_1^{(t)} s_4^{(t)}}
   {1 - s_1^{(t)} s_2^{(t)}}, \\
   \Sigma_t &= s_6^{(t)} - s_5^{(t)} \mu_t^\top
   - \mu_t (s_5^{(t)})^\top + s_1^{(t)} \mu_t \mu_t^\top
   - s_2^{(t)} \gamma_t \gamma_t^\top,

and

.. math::

   (p_t, a_t, b_t) = \arg\max_{p, a, b}
   L_{GIG}(p, a, b \,|\, s_1^{(t)}, s_2^{(t)}, s_3^{(t)}).

The online EM algorithm has comparable convergence rate to the batch EM
algorithm, but each step processes only a single observation, making it
much faster per iteration. The step size :math:`\tau_t = \tau_0 + t`
corresponds to starting with an initial weight of :math:`\tau_0` "pseudo-observations."

Limitations for Curved Exponential Families
-------------------------------------------

The online EM algorithm of [Cappe2009]_ may not converge for **curved
exponential families** of the form:

.. math::

   f(x, y | u) = h(x, y) \exp\!\left(\theta(u)^\top t(x, y)
   - \psi(\theta(u))\right),

where :math:`\theta(u)` is a nonlinear mapping projecting the parameter
:math:`u` to a higher-dimensional space. The batch EM algorithm for curved
exponential families solves:

.. math::

   \nabla\theta(u)^\top \left(\frac{1}{n} \sum_{j=1}^n
   E[t(X, Y) | x_j, \theta(u)] - \nabla\psi(\theta(u))\right) = 0,

while the online EM tends to solve:

.. math::

   \frac{1}{n} \sum_{j=1}^n E[t(X, Y) | x_j, \theta(u)]
   - \nabla\psi(\theta(u)) = 0,

which may not have a solution when :math:`\dim(u) < \dim(\theta)`. Therefore,
the online EM algorithm cannot be directly applied to the factor analysis
model described in :doc:`factor_analysis`.

References
----------

.. [Cappe2009] Cappé, O. & Moulines, E. (2009). On-line expectation-maximization
   algorithm for latent data models. *Journal of the Royal Statistical Society:
   Series B*, 71(3), 593-613.
