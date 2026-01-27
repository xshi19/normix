The Generalized Inverse Gaussian Distribution
=============================================

In this section we review the definition and statistical properties of the
generalized inverse Gaussian (GIG) distribution.

Definition
----------

The generalized inverse Gaussian distribution is a continuous probability
distribution with the density function:

.. math::
   :label: gig-pdf

   f(x|p,a,b) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})} x^{p-1}
   \exp\left(-\frac{1}{2}(b x^{-1} + a x)\right), \quad x > 0,

where :math:`K_p(\cdot)` is the modified Bessel function of the second kind
and the parameters :math:`(p, a, b)` satisfy:

.. math::

   \begin{cases}
   b > 0, \, a \geq 0 & \text{if } p < 0 \\
   b > 0, \, a > 0 & \text{if } p = 0 \\
   b \geq 0, \, a > 0 & \text{if } p > 0
   \end{cases}

Throughout this package we assume that :math:`a > 0` and :math:`b > 0` for simplicity.

Alternative Parameterization
----------------------------

Another useful way to parameterize the GIG distribution is to set
:math:`\delta = \sqrt{b/a}` and :math:`\eta = \sqrt{ab}`. In that case the
density function can be written as:

.. math::
   :label: gig-pdf-alt

   f(x|p, \delta, \eta) = \frac{\delta^p}{2K_p(\eta)} x^{p-1}
   \exp\left(-\frac{\eta}{2}(\delta x^{-1} + \delta^{-1} x)\right), \quad x > 0.

Note that :math:`\delta` serves as a scale parameter of the GIG distribution.

Moment Generating Function
--------------------------

The moment generating function of a GIG distributed random variable :math:`X` is given by:

.. math::

   E[e^{uX}] = \left(\sqrt{\frac{a}{a-2u}}\right)^p
   \frac{K_p(\sqrt{b(a-2u)})}{K_p(\sqrt{ab})}
   = \left(\sqrt{\frac{\eta}{\eta-2\delta u}}\right)^p
   \frac{K_p(\sqrt{\eta^2-2\delta u})}{K_p(\eta)}.

Moments
-------

The moments of the GIG distribution have a particularly elegant form:

.. math::
   :label: gig-moments

   E[X^\alpha] = \left(\sqrt{\frac{b}{a}}\right)^\alpha
   \frac{K_{p+\alpha}(\sqrt{ab})}{K_p(\sqrt{ab})}
   = \delta^\alpha \frac{K_{p+\alpha}(\eta)}{K_p(\eta)}.

This formula is implemented in the :meth:`~normix.distributions.univariate.GeneralizedInverseGaussian.moment_alpha` method.

Exponential Family Form
-----------------------

The GIG distribution belongs to the exponential family with density:

.. math::

   f(x|\theta) = h(x) \exp\left(\theta^\top t(x) - \psi(\theta)\right)

**Sufficient Statistics:**

.. math::

   t(x) = \begin{pmatrix} \log x \\ x^{-1} \\ x \end{pmatrix}

**Natural Parameters:**

The natural parameters :math:`\theta = (\theta_1, \theta_2, \theta_3)` are derived from
the classical parameters :math:`(p, a, b)`:

.. math::
   :label: gig-natural-params

   \theta_1 &= p - 1 \quad \text{(unbounded)} \\
   \theta_2 &= -\frac{b}{2} < 0 \\
   \theta_3 &= -\frac{a}{2} < 0

The inverse transformation is:

.. math::

   p = \theta_1 + 1, \quad b = -2\theta_2, \quad a = -2\theta_3

**Base Measure:**

.. math::

   h(x) = \mathbf{1}_{x > 0}

**Log Partition Function:**

.. math::
   :label: gig-log-partition

   \psi(\theta) = \log 2 + \log K_p(\sqrt{ab}) + \frac{p}{2} \log\left(\frac{b}{a}\right)

where :math:`p = \theta_1 + 1`, :math:`a = -2\theta_3`, and :math:`b = -2\theta_2`.

**Expectation Parameters:**

The expectation parameters :math:`\eta = \nabla\psi(\theta) = E[t(X)]` are:

.. math::
   :label: gig-expectation-params

   \eta_1 &= E[\log X] = \frac{\partial \psi}{\partial \theta_1} \\
   \eta_2 &= E[X^{-1}] = \sqrt{\frac{a}{b}} \frac{K_{p-1}(\sqrt{ab})}{K_p(\sqrt{ab})} \\
   \eta_3 &= E[X] = \sqrt{\frac{b}{a}} \frac{K_{p+1}(\sqrt{ab})}{K_p(\sqrt{ab})}

Unfortunately we do not have an analytical formula for :math:`\eta_1 = E[\log X]`.
In practice it can only be approximated numerically.

Maximum Likelihood Estimation
-----------------------------

Given the expectation parameters :math:`(\eta_1, \eta_2, \eta_3)`, computing the
natural parameters :math:`(p, a, b)` by solving the above equations is a challenging
problem. Let :math:`x_1, x_2, \ldots, x_n` be a sequence of sample data, then the
maximum likelihood estimator (MLE) of GIG is given by:

.. math::
   :label: gig-mle

   (\hat{p}, \hat{a}, \hat{b}) = \arg\max_{p,a,b} L_{GIG}(p, a, b | \hat{\eta}_1, \hat{\eta}_2, \hat{\eta}_3),

where :math:`L_{GIG}` is the log-likelihood function (excluding constants):

.. math::
   :label: gig-loglik

   L_{GIG}(p, a, b | \eta_1, \eta_2, \eta_3) =
   -\frac{1}{2} b \hat{\eta}_1 - \frac{1}{2} a \hat{\eta}_2 + p \hat{\eta}_3
   + \frac{p}{2} \log(a/b) - \log(K_p(\sqrt{ab})),

and the sufficient statistics are:

- :math:`\hat{\eta}_1 = \frac{1}{n} \sum_{k=1}^n x_k^{-1}`
- :math:`\hat{\eta}_2 = \frac{1}{n} \sum_{k=1}^n x_k`
- :math:`\hat{\eta}_3 = \frac{1}{n} \sum_{k=1}^n \log(x_k)`

One can verify that the optimal solution :math:`(\hat{p}, \hat{a}, \hat{b})` must
satisfy :eq:`gig-expectation-params` where :math:`(\eta_1, \eta_2, \eta_3)` are
replaced by :math:`(\hat{\eta}_1, \hat{\eta}_2, \hat{\eta}_3)`.

Numerical Challenges
--------------------

As discussed in [Jorgensen2012]_, there is no analytical expression for
:math:`\hat{p}` or even its partial derivatives. Most literature suggests fixing
:math:`p` when maximizing the log-likelihood function.

Even when :math:`p` is fixed, [Hu2005]_ reports that when :math:`|p|` is large
(say, above 10), there might be no solution for the first two equations in
:eq:`gig-expectation-params`.

Hellinger Distance
------------------

To measure estimation errors, one good choice is the Hellinger distance between
the true and estimated parameters.

**Proposition.** Let :math:`(p_1, a_1, b_1)` and :math:`(p_2, a_2, b_2)` be the
parameters of two GIG distributions. The squared Hellinger distance between the
two distributions is given by:

.. math::

   H_{GIG}^2(p_1, a_1, b_1 \| p_2, a_2, b_2) = 1 -
   \frac{(a_1/b_1)^{p_1/4} (a_2/b_2)^{p_2/4}}
        {\sqrt{K_{p_1}(\sqrt{a_1 b_1}) K_{p_2}(\sqrt{a_2 b_2})}}
   \frac{K_{\bar{p}}(\sqrt{\bar{a}\bar{b}})}{(\bar{a}/\bar{b})^{\bar{p}/2}},

where :math:`\bar{p} = (p_1 + p_2)/2`, :math:`\bar{a} = (a_1 + a_2)/2`, and
:math:`\bar{b} = (b_1 + b_2)/2`.

Special Cases
-------------

There are several important special cases of the GIG distribution:

- **Inverse Gaussian (IG)**: when :math:`p = -1/2`
- **Gamma**: when :math:`p > 0` and :math:`b \to 0`, giving :math:`\text{Gamma}(p, a/2)`
- **Inverse Gamma**: when :math:`p < 0` and :math:`a \to 0`, giving :math:`\text{InvGamma}(-p, b/2)`

These special cases are implemented as separate classes in ``normix``:

- :class:`~normix.distributions.univariate.InverseGaussian`
- :class:`~normix.distributions.univariate.Gamma`
- :class:`~normix.distributions.univariate.InverseGamma`

References
----------

.. [Jorgensen2012] Jørgensen, B. (2012). *Statistical Properties of the Generalized Inverse Gaussian Distribution*. Springer.
