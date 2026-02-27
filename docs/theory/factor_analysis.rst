Factor Analysis for Generalized Hyperbolic Distributions
========================================================

Another way to improve the conditioning of :math:`\Sigma` is the factor
analysis approach [Tortora2013]_. Instead of estimating a full covariance
matrix, we assume the structure :math:`\Sigma = F F^\top + D` where
:math:`F \in \mathbb{R}^{d \times r}` with :math:`r < d` and
:math:`D \in \mathbb{R}^{d \times d}` is a positive definite diagonal matrix.

This is equivalent to saying that a GH random vector :math:`X` can be
expressed as:

.. math::

   X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}(F Z + \varepsilon),

where :math:`Z \in \mathbb{R}^r \sim N(0, I)` and
:math:`\varepsilon \sim N(0, D)` are independent.

Joint Distribution
------------------

The conditional distribution of :math:`X` given :math:`Y` and :math:`Z`
is :math:`N(\mu + \gamma Y + \sqrt{Y} F Z, \, D Y)`. The joint distribution
of :math:`(X, Y, Z)` is:

.. math::
   :label: fa-joint

   &f(x, y, z | \mu, \gamma, F, D, p, a, b) \\
   &= \frac{1}{\sqrt{(2\pi)^{d+r} |D|}}
   \frac{(a/b)^{p/2}}{2 K_p(\sqrt{ab})}
   y^{p - 1 - d/2} \\
   &\quad \times \exp\!\left(-\frac{1}{2} z^\top z
   - \frac{1}{2}(b \, y^{-1} + a \, y)
   - \frac{1}{2}(x - \mu - \gamma y - F \sqrt{y} \, z)^\top
   D^{-1}(x - \mu - \gamma y - F \sqrt{y} \, z) \, y^{-1}\right)

for :math:`y > 0`.

Curved Exponential Family
-------------------------

The density :eq:`fa-joint` belongs to a **curved exponential family**:

.. math::

   f(x, y, z) = h(x, y, z) \exp\!\left(\theta(u)^\top t(x, y, z)
   - \psi(\theta(u))\right),

where :math:`\theta(\cdot)` is a nonlinear mapping from the parameter
:math:`u = (\mu, \gamma, F, D, p, a, b)` to a higher-dimensional space.

The sufficient statistics :math:`t(x, y, z)` consist of ten components:

.. math::

   s_1 = y^{-1}, \quad s_2 = y, \quad s_3 = \log y, \quad
   s_4 = x, \quad s_5 = x y^{-1}, \quad s_6 = x x^\top y^{-1},

.. math::

   s_7 = x z^\top y^{-1/2}, \quad s_8 = z y^{-1/2}, \quad
   s_9 = z y^{1/2}, \quad s_{10} = z z^\top.

Log-Likelihood Function
-----------------------

The log-likelihood function for the factor analysis model is:

.. math::

   L_{FA}(u | s) &= -\frac{1}{2} \log|D|
   - \frac{1}{2} \mu^\top D^{-1} \mu \, s_1
   - \frac{1}{2} \gamma^\top D^{-1} \gamma \, s_2
   + \gamma^\top D^{-1} s_4
   + \mu^\top D^{-1} s_5 \\
   &\quad - \frac{1}{2} \operatorname{tr}(D^{-1} s_6)
   + \operatorname{tr}(F^\top D^{-1} s_7)
   - \mu^\top D^{-1} F s_8
   - \gamma^\top D^{-1} F s_9 \\
   &\quad - \frac{1}{2} \operatorname{tr}(F^\top D^{-1} F s_{10})
   - \mu^\top D^{-1} \gamma
   + L_{GIG}(p, a, b | s_1, s_2, s_3),

where :math:`L_{GIG}` is the GIG log-likelihood defined in :eq:`gig-loglik`.

M-Step: Closed-Form Solutions
-----------------------------

Setting the partial derivatives of :math:`L_{FA}` to zero yields analytic
formulas for the M-step. Define:

.. math::
   :label: fa-aux

   q_1 &= s_8^\top s_{10}^{-1} s_8 - s_1, \\
   q_2 &= s_9^\top s_{10}^{-1} s_8 - 1, \\
   q_3 &= s_9^\top s_{10}^{-1} s_9 - s_2, \\
   q_4 &= s_7^\top s_{10}^{-1} s_8 - s_5, \\
   q_5 &= s_7^\top s_{10}^{-1} s_9 - s_4.

Then the parameter updates are:

.. math::
   :label: fa-mstep

   \mu &= \frac{q_2 \, q_5 - q_3 \, q_4}{q_2^2 - q_1 \, q_3}, \\
   \gamma &= \frac{q_2 \, q_4 - q_1 \, q_5}{q_2^2 - q_1 \, q_3}, \\
   F &= (s_7 - \mu \, s_8^\top - \gamma \, s_9^\top) \, s_{10}^{-1}, \\
   D &= \operatorname{diag}\!\big(
   s_1 \mu \mu^\top + s_2 \gamma \gamma^\top
   - s_4 \gamma^\top - \gamma s_4^\top
   - s_5 \mu^\top - \mu s_5^\top + s_6 \\
   &\qquad - s_7 F^\top - F s_7^\top
   + F s_8 \mu^\top + \mu (F s_8)^\top
   + F s_9 \gamma^\top + \gamma (F s_9)^\top \\
   &\qquad + F s_{10} F^\top
   + \mu \gamma^\top + \gamma \mu^\top\big), \\
   (p, a, b) &= \arg\max_{p, a, b} L_{GIG}(p, a, b | s_1, s_2, s_3).

Conditional Expectations for the E-Step
---------------------------------------

Integrating :eq:`fa-joint` over :math:`z`, the joint distribution of
:math:`(X, Y)` has covariance structure :math:`\Sigma = F F^\top + D`.
Therefore the conditional distribution of :math:`Y` given :math:`X` is:

.. math::

   Y | X = x \sim \operatorname{GIG}\!\left(p - \frac{d}{2}, \,
   a + \gamma^\top (F F^\top + D)^{-1} \gamma, \,
   b + (x - \mu)^\top (F F^\top + D)^{-1} (x - \mu)\right),

and the conditional moments :math:`E[Y^\alpha | X, u]` and
:math:`E[\log Y | X, u]` are computed using :eq:`gig-moment-cond`.

The conditional distribution of :math:`(X, Z)` given :math:`Y` is Gaussian:

.. math::

   \begin{pmatrix} (X - \mu - \gamma Y)/\sqrt{Y} \\ Z \end{pmatrix}
   \Bigg| Y, u
   \sim N\!\left(0, \begin{pmatrix}
   F F^\top + D & F \\ F^\top & I
   \end{pmatrix}\right).

Define :math:`\beta = F^\top (F F^\top + D)^{-1}`. The conditional
expectations of the latent factor :math:`Z` are:

.. math::

   E[Z Y^{-1/2} | X, u] &= \beta(X - \mu) E[Y^{-1} | X, u] - \beta \gamma, \\
   E[Z Y^{1/2} | X, u] &= \beta(X - \mu) - \beta \gamma \, E[Y | X, u], \\
   E[Z Z^\top | X, u] &= I - \beta F
   + \beta(X - \mu)(X - \mu)^\top \beta^\top E[Y^{-1} | X, u] \\
   &\quad - \beta(X - \mu)\gamma^\top \beta^\top
   - \beta \gamma (X - \mu)^\top \beta^\top
   + \beta \gamma \gamma^\top \beta^\top E[Y | X, u].

E-Step
------

Given i.i.d. samples :math:`x_1, \ldots, x_n` and current parameters
:math:`u_k = (\mu_k, \gamma_k, F_k, D_k, p_k, a_k, b_k)`, the E-step
computes all ten sufficient statistics. The first six are the same as the
standard EM algorithm (see :eq:`e-step`):

.. math::

   s_1^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[Y^{-1} | X = x_j, u_k], \\
   s_2^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[Y | X = x_j, u_k], \\
   s_3^{(k)} &= \frac{1}{n} \sum_{j=1}^n E[\log Y | X = x_j, u_k], \\
   s_4^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j, \\
   s_5^{(k)} &= \frac{1}{n} \sum_{j=1}^n x_j \, E[Y^{-1} | X = x_j, u_k], \\
   s_6^{(k)} &= \frac{1}{n} \sum_{j=1}^n
   x_j x_j^\top E[Y^{-1} | X = x_j, u_k].

The remaining four are determined by the first six using
:math:`\beta_k = F_k^\top (F_k F_k^\top + D_k)^{-1}`:

.. math::

   s_7^{(k)} &= (s_6^{(k)} - s_5^{(k)} \mu_k^\top
   - s_4^{(k)} \gamma_k^\top) \beta_k^\top, \\
   s_8^{(k)} &= \beta_k (s_5^{(k)} - \mu_k \, s_1^{(k)} - \gamma_k), \\
   s_9^{(k)} &= \beta_k (s_4^{(k)} - \mu_k - \gamma_k \, s_2^{(k)}), \\
   s_{10}^{(k)} &= I - \beta_k F_k
   + \beta_k \big(s_6^{(k)} - s_5^{(k)} \mu_k^\top
   - \mu_k (s_5^{(k)})^\top + \mu_k \mu_k^\top s_1^{(k)} \\
   &\quad - (s_4^{(k)} - \mu_k) \gamma_k^\top
   - \gamma_k (s_4^{(k)} - \mu_k)^\top
   + \gamma_k \gamma_k^\top s_2^{(k)}\big) \beta_k^\top.

The M-step then applies :eq:`fa-mstep` and :eq:`fa-aux` with
:math:`s = (s_1^{(k)}, \ldots, s_{10}^{(k)})`.

References
----------

.. [Tortora2013] Tortora, C., McNicholas, P. D., & Browne, R. P. (2013).
   Mixtures of multivariate generalized hyperbolic distributions.
