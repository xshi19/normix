The Generalized Hyperbolic Distribution
=======================================

In this section we briefly review the basic properties of the Generalized
Hyperbolic (GH) distribution.

Definition as Normal Mixture
----------------------------

Let :math:`Y` be a GIG random variable with parameters :math:`(p, a, b)`, and
:math:`Z` be an independent Gaussian random vector with zero mean and covariance
:math:`\Sigma`. Then the random vector:

.. math::
   :label: gh-def

   X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z

follows the **generalized hyperbolic distribution** with parameters
:math:`(\mu, \gamma, \Sigma, p, a, b)`, where :math:`\mu, \gamma \in \mathbb{R}^d`
and :math:`\Sigma \in \mathbb{R}^{d \times d}` is a positive definite matrix.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

- :math:`\mu`: location parameter
- :math:`\gamma`: skewness parameter
- :math:`\Sigma`: models the dependency structure of the multivariate distribution
- :math:`(p, a, b)`: GIG parameters controlling the heavy-tailedness

In general, many multivariate heavy-tailed distributions can be defined by
:eq:`gh-def` given some non-negative random variable :math:`Y`. These distributions
are usually called **normal mixture** or **Gaussian mixture** distributions.

.. note::

   In many references, "normal mixture" refers to a discrete mixture of normal
   densities. In ``pygh``, we use "normal mixture" to refer to any random variable
   that can be expressed by :eq:`gh-def`.

Joint GH Distribution
---------------------

The joint distribution of :math:`X` and :math:`Y` is crucial in analyzing the GH
distribution. We call the distribution of :math:`(X, Y)` the **joint-GH distribution**.
Its density function is:

.. math::
   :label: gh-joint

   f(x, y | \mu, \gamma, \Sigma, p, a, b) &= \frac{1}{\sqrt{(2\pi)^d |\Sigma|}}
   \frac{(a/b)^{p/2}}{2 K_p(\sqrt{ab})} y^{p - 1 - d/2} \\
   &\quad \times \exp\left( -\frac{1}{2}(x - \mu - \gamma y)^\top \Sigma^{-1}
   (x - \mu - \gamma y) y^{-1} - \frac{1}{2}(b y^{-1} + a y) \right),

for :math:`y > 0`.

Marginal GH Density
-------------------

Integrating out :math:`y` from :eq:`gh-joint`, we obtain the marginal distribution
of :math:`x`, which is the GH density function:

.. math::
   :label: gh-marginal

   f(x | \mu, \gamma, \Sigma, p, a, b) &= \int_0^\infty f(x, y | \mu, \gamma, \Sigma, p, a, b) \, dy \\
   &= c \frac{K_{p - d/2}\left(\sqrt{(b + q(x))(a + \gamma^\top \Sigma^{-1} \gamma)}\right)}
   {\left(\sqrt{(b + q(x))(a + \gamma^\top \Sigma^{-1} \gamma)}\right)^{d/2 - p}}
   e^{(x - \mu)^\top \Sigma^{-1} \gamma},

where :math:`q(x) = (x - \mu)^\top \Sigma^{-1} (x - \mu)` is the Mahalanobis distance, and

.. math::

   c = \frac{(a/b)^{p/2} (a + \gamma^\top \Sigma^{-1} \gamma)^{d/2 - p}}
   {(2\pi)^{d/2} |\Sigma|^{1/2} K_p(\sqrt{ab})}.

Alternative Parameterization
----------------------------

Similar to the GIG distribution, one can use another parameterization with
:math:`\delta = \sqrt{b/a}` and :math:`\eta = \sqrt{ab}`:

.. math::

   f(x | \mu, \gamma, \Sigma, p, \delta, \eta) = c
   \frac{K_{p - d/2}\left(\sqrt{(\eta + q_\delta(x))(\eta + \tilde{q}_\delta)}\right)}
   {\left(\sqrt{(\eta + q_\delta(x))(\eta + \tilde{q}_\delta)}\right)^{d/2 - p}}
   e^{(x - \mu)^\top (\delta \Sigma)^{-1} \delta \gamma},

where :math:`q_\delta(x) = (x - \mu)^\top (\delta \Sigma)^{-1} (x - \mu)`,
:math:`\tilde{q}_\delta = (\delta \gamma)^\top (\delta \Sigma)^{-1} (\delta \gamma)`, and

.. math::

   c = \frac{(\eta + \tilde{q}_\delta)^{d/2 - p}}{(2\pi)^{d/2} |\delta \Sigma|^{1/2} K_p(\eta)}.

Model Identifiability
---------------------

From the above representation, one can observe that the GH model is **not regular**
since the parameter sets :math:`(\mu, \gamma/c, \Sigma/c, p, c\delta, \eta)` give the
same distribution for any :math:`c > 0`. Therefore, the Fisher information matrix of
the GH distribution is singular.

There are several ways to regularize the GH family:

1. **Set** :math:`\delta = 1` (simplest approach)
2. **Set** :math:`b = 1` in the EM algorithm [Protassov2004]_
3. **Fix** :math:`b` when :math:`p > -1` and fix :math:`a` when :math:`p < 1` [Hu2005]_
4. **Fix the determinant** :math:`|\Sigma| = 1` [McNeil2010]_

.. note::

   When the dimension :math:`d` is high, fixing :math:`|\Sigma| = 1` is recommended.
   Since :math:`|\Sigma/c| = |\Sigma|/c^d`, any small perturbation of the matrix scale
   will make :math:`|\Sigma|` change dramatically when :math:`d` is large. The matrix
   inversion would be intractable if :math:`|\Sigma|` is too large or too small.

Exponential Family Form
-----------------------

The joint-GH distribution :eq:`gh-joint` belongs to the exponential family with density:

.. math::

   f(x, y|\theta) = h(x, y) \exp\left(\theta^\top t(x, y) - \psi(\theta)\right)

**Sufficient Statistics:**

.. math::

   t(x, y) = \begin{pmatrix} \log y \\ y^{-1} \\ y \\ x \\ x y^{-1} \\ x x^\top y^{-1} \end{pmatrix}

where :math:`t_1, t_2, t_3 \in \mathbb{R}`, :math:`t_4, t_5 \in \mathbb{R}^d`, and
:math:`t_6 \in \mathbb{R}^{d \times d}`.

**Natural Parameters:**

The natural parameters are derived from the classical parameters
:math:`(\mu, \gamma, \Sigma, p, a, b)`. By expanding the exponent in :eq:`gh-joint`:

.. math::

   &-\frac{1}{2}(x - \mu - \gamma y)^\top \Sigma^{-1} (x - \mu - \gamma y) y^{-1}
   - \frac{1}{2}(b y^{-1} + a y) \\
   &= -\frac{1}{2} x^\top \Sigma^{-1} x \, y^{-1}
   + x^\top \Sigma^{-1} \mu \, y^{-1}
   + x^\top \Sigma^{-1} \gamma \\
   &\quad - \frac{1}{2} \mu^\top \Sigma^{-1} \mu \, y^{-1}
   - \mu^\top \Sigma^{-1} \gamma
   - \frac{1}{2} \gamma^\top \Sigma^{-1} \gamma \, y \\
   &\quad - \frac{1}{2} b \, y^{-1} - \frac{1}{2} a \, y

we identify the natural parameters:

.. math::
   :label: gh-natural-params

   \theta_1 &= p - 1 - \frac{d}{2} \quad \text{(coefficient of } \log y \text{)} \\
   \theta_2 &= -\frac{1}{2}\left(b + \mu^\top \Sigma^{-1} \mu\right) \quad \text{(coefficient of } y^{-1} \text{)} \\
   \theta_3 &= -\frac{1}{2}\left(a + \gamma^\top \Sigma^{-1} \gamma\right) \quad \text{(coefficient of } y \text{)} \\
   \theta_4 &= \Sigma^{-1} \gamma \quad \text{(coefficient of } x \text{)} \\
   \theta_5 &= \Sigma^{-1} \mu \quad \text{(coefficient of } x y^{-1} \text{)} \\
   \theta_6 &= -\frac{1}{2} \Sigma^{-1} \quad \text{(coefficient of } x x^\top y^{-1} \text{)}

**Base Measure:**

.. math::

   h(x, y) = \frac{1}{(2\pi)^{d/2}} \mathbf{1}_{y > 0}

**Log Partition Function:**

.. math::
   :label: gh-log-partition

   \psi(\theta) = \frac{1}{2} \log|\Sigma| + \log 2 + \log K_p(\sqrt{ab})
   + \frac{p}{2} \log\left(\frac{b}{a}\right) + \mu^\top \Sigma^{-1} \gamma

**Expectation Parameters:**

The expectation parameters :math:`\eta = \nabla\psi(\theta) = E[t(X, Y)]` are:

.. math::
   :label: gh-expectation

   \eta_1 &:= E[\log Y] = \left.\frac{\partial}{\partial \alpha}
   \left(\sqrt{\frac{b}{a}}\right)^\alpha \frac{K_{p+\alpha}(\sqrt{ab})}{K_p(\sqrt{ab})}
   \right|_{\alpha=0} \\
   \eta_2 &:= E[Y^{-1}] = \sqrt{\frac{a}{b}} \frac{K_{p-1}(\sqrt{ab})}{K_p(\sqrt{ab})} \\
   \eta_3 &:= E[Y] = \sqrt{\frac{b}{a}} \frac{K_{p+1}(\sqrt{ab})}{K_p(\sqrt{ab})} \\
   \eta_4 &:= E[X] = \mu + \gamma \eta_3 \\
   \eta_5 &:= E[X Y^{-1}] = \mu \eta_2 + \gamma \\
   \eta_6 &:= E[X X^\top Y^{-1}] = \Sigma + \mu \mu^\top \eta_2 + \gamma \gamma^\top \eta_3
   + \mu \gamma^\top + \gamma \mu^\top

where :math:`\eta_1, \eta_2, \eta_3 \in \mathbb{R}`, :math:`\eta_4, \eta_5 \in \mathbb{R}^d`, and
:math:`\eta_6 \in \mathbb{R}^{d \times d}`.

Note that :math:`\eta_1, \eta_2, \eta_3` are exactly the expectation parameters of the GIG
random variable :math:`Y`.

Recovering Parameters from Expectations
---------------------------------------

Given all the expectation parameters, we can recover the original parameters as follows:

.. math::
   :label: gh-m-step

   \mu &= \frac{\eta_4 - \eta_3 \eta_5}{1 - \eta_2 \eta_3}, \\
   \gamma &= \frac{\eta_5 - \eta_2 \eta_4}{1 - \eta_2 \eta_3}, \\
   \Sigma &= \eta_6 - \eta_5 \mu^\top - \mu \eta_5^\top + \eta_2 \mu \mu^\top - \eta_3 \gamma \gamma^\top, \\
   (p, a, b) &= \arg\max_{p,a,b} L_{\text{GIG}}(p, a, b | \eta_1, \eta_2, \eta_3),

where :math:`L_{\text{GIG}}` is the GIG log-likelihood function given in
:eq:`gig-loglik`.

These equations form the **M-step** in the EM algorithm, where the expectations in
:eq:`gh-expectation` are replaced by conditional expectations.

Hellinger Distance
------------------

While there is no analytical formulation of the Hellinger distance between two GH
distributions, the Hellinger distance of the joint-GH distributions is tractable.

**Proposition.** Let :math:`\theta_1 = (\mu_1, \gamma_1, \Sigma_1, p_1, a_1, b_1)` and
:math:`\theta_2 = (\mu_2, \gamma_2, \Sigma_2, p_2, a_2, b_2)` be the parameters of two
joint-GH distributions. The squared Hellinger distance between the two distributions is:

.. math::

   H_{\text{JGH}}^2(\theta_1 \| \theta_2) = 1 -
   \frac{|\Sigma_1 \Sigma_2|^{1/4}}{|\bar{\Sigma}|^{1/2}}
   \frac{(a_1/b_1)^{p_1/4} (a_2/b_2)^{p_2/4}}
   {\sqrt{K_{p_1}(\sqrt{a_1 b_1}) K_{p_2}(\sqrt{a_2 b_2})}}
   \frac{K_{\bar{p}}(\sqrt{\bar{b}' \bar{a}'})}{(\bar{a}'/\bar{b}')^{\bar{p}/2}}
   e^{-\frac{1}{4} \Delta\mu^\top \bar{\Sigma}^{-1} \Delta\gamma},

where:

- :math:`\Delta\mu = \mu_1 - \mu_2`, :math:`\Delta\gamma = \gamma_1 - \gamma_2`
- :math:`\bar{\Sigma} = (\Sigma_1 + \Sigma_2)/2`
- :math:`\bar{p} = (p_1 + p_2)/2`, :math:`\bar{a} = (a_1 + a_2)/2`, :math:`\bar{b} = (b_1 + b_2)/2`
- :math:`\bar{b}' = \bar{b} + \frac{1}{4} \Delta\mu^\top \bar{\Sigma}^{-1} \Delta\mu`
- :math:`\bar{a}' = \bar{a} + \frac{1}{4} \Delta\gamma^\top \bar{\Sigma}^{-1} \Delta\gamma`

If :math:`\mu_1 = \mu_2`, :math:`\gamma_1 = \gamma_2`, and :math:`\Sigma_1 = \Sigma_2`,
then :math:`H_{\text{JGH}}(\theta_1 \| \theta_2) = H_{\text{GIG}}(p_1, a_1, b_1 \| p_2, a_2, b_2)`.

Although :math:`H_{\text{JGH}}` differs from the Hellinger distance of the marginal GH,
it provides an upper bound, so we can use it to measure how close two GH distributions are.

Numerical Stability
-------------------

The computation of :eq:`gh-m-step` is not stable in terms of relative error under
certain conditions. However, it is relatively stable in terms of the Hellinger distance.

Numerical experiments show that:

- The relative errors of :math:`\mu` and :math:`\gamma` are around machine epsilon
- The relative error of some GIG parameters (especially when :math:`|p|` is large) can be large
- However, the Hellinger distance between true and estimated parameters remains small

This behavior is consistent with the ill-conditioning of the GIG optimization problem
discussed in the :doc:`gig` section.

Special Cases
-------------

Several important distributions are special cases of the GH family:

- **Normal-Inverse Gaussian (NIG)**: :math:`p = -1/2`
- **Variance-Gamma (VG)**: :math:`b \to 0` (Gamma mixing)
- **Normal-Inverse Gamma (NInvG)**: :math:`a \to 0` (Inverse-Gamma mixing)
- **Student-t**: :math:`p < 0`, :math:`a = 0`, :math:`\gamma = 0`
- **Hyperbolic**: :math:`p = 1`

These are implemented as separate classes in ``pygh``:

- :class:`~pygh.distributions.mixtures.NormalInverseGaussian`
- :class:`~pygh.distributions.mixtures.VarianceGamma`
- :class:`~pygh.distributions.mixtures.NormalInverseGamma`

References
----------

.. [Protassov2004] Protassov, R. S. (2004). EM-based maximum likelihood parameter estimation for multivariate generalized hyperbolic distributions.

.. [Hu2005] Hu, W. (2005). Calibration of multivariate generalized hyperbolic distributions using the EM algorithm.

.. [McNeil2010] McNeil, A. J., Frey, R., & Embrechts, P. (2010). Quantitative Risk Management. Princeton University Press.
