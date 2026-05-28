CVaR Derivatives for Normal Mixture Distributions
=================================================

This section computes the first and second derivatives of CVaR for normal
mixture distributions, following [RauHasanov2004]_ and [Tasche1999]_.

General CVaR Derivatives
------------------------

Let :math:`X = (X_1, \ldots, X_n)` be a random vector with portfolio weights
:math:`w \in \mathbb{R}^n`, and define:

.. math::

   r_{\operatorname{VaR}_\alpha}(w) &:= \operatorname{VaR}_\alpha(w^\top X), \\
   r_{\operatorname{CVaR}_\alpha}(w) &:= \operatorname{CVaR}_\alpha(w^\top X).

**Assumption.** Let :math:`p(x_1 | x_2, \ldots, x_n)` be the conditional
density of :math:`X_1` given :math:`X_2, \ldots, X_n`. Assume:

1. :math:`y \mapsto p(y | x_2, \ldots, x_n)` is continuous.
2. :math:`(y, w) \mapsto E[p(w_1^{-1}(y - \sum_{l=2}^n w_l X_l) | X_2, \ldots, X_n)]` is finite and continuous.
3. The density at the VaR quantile is strictly positive.
4. :math:`(y, w) \mapsto E[X_j \, p(w_1^{-1}(y - \sum_{l=2}^n w_l X_l) | X_2, \ldots, X_n)]` is finite and continuous.
5. :math:`(y, w) \mapsto E[X_j X_k \, p(w_1^{-1}(y - \sum_{l=2}^n w_l X_l) | X_2, \ldots, X_n)]` is finite and continuous.

First Derivatives
~~~~~~~~~~~~~~~~~

Under conditions 1--4 above, for :math:`w_1 \neq 0`:

.. math::

   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial w_j}(w)
   &= -\frac{E\!\left[X_j \, p\!\left(w_1^{-1}\!\left(
   -r_{\operatorname{VaR}_\alpha}(w) - \sum_{l=2}^n w_l X_l\right)
   \middle| X_2, \ldots, X_n\right)\right]}
   {E\!\left[p\!\left(w_1^{-1}\!\left(
   -r_{\operatorname{VaR}_\alpha}(w) - \sum_{l=2}^n w_l X_l\right)
   \middle| X_2, \ldots, X_n\right)\right]},
   \quad j = 2, \ldots, n, \\
   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial w_1}(w)
   &= w_1^{-1} \left(r_{\operatorname{VaR}_\alpha}(w)
   - \sum_{j=2}^n w_j
   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial w_j}(w)\right),

and

.. math::
   :label: cvar-gradient

   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial w_j}(w)
   = -E[X_j \mid w^\top X \leq -r_{\operatorname{VaR}_\alpha}(w)]
   = -\alpha^{-1} E\!\left[X_j \,
   \mathbf{1}_{\{w^\top X \leq -r_{\operatorname{VaR}_\alpha}(w)\}}\right],

for :math:`j = 1, \ldots, n`.

Second Derivatives
~~~~~~~~~~~~~~~~~~

Under the full assumption, for :math:`j, k = 2, \ldots, n`:

.. math::
   :label: cvar-hessian

   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial w_j \partial w_k}(w)
   = \frac{1}{\alpha |w_1|}
   E\!\left[X_k \left(
   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial w_j}(w)
   + X_j\right) p\!\left(w_1^{-1}\!\left(
   -r_{\operatorname{VaR}_\alpha}(w)
   - \sum_{l=2}^n w_l X_l\right)
   \middle| X_2, \ldots, X_n\right)\right],

and for :math:`j = 1, \ldots, n`:

.. math::

   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial w_j \partial w_1}(w)
   = -w_1^{-1} \sum_{k=2}^n w_k
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial w_j \partial w_k}(w).

This second equation follows from the 1-homogeneity of CVaR:
:math:`\sum_{k=1}^n w_k \,
\partial^2 r_{\operatorname{CVaR}_\alpha} / \partial w_j \partial w_k = 0`.

Application to Univariate Normal Mixtures
-----------------------------------------

The univariate normal mixture :eq:`nm-def` can be viewed as a "portfolio"
with two risky assets. Define:

.. math::

   r_{\operatorname{VaR}_\alpha}(\mu, \gamma, \sigma) &:=
   \operatorname{VaR}_\alpha(\mu + \gamma Y + \sigma \sqrt{Y} Z), \\
   r_{\operatorname{CVaR}_\alpha}(\mu, \gamma, \sigma) &:=
   \operatorname{CVaR}_\alpha(\mu + \gamma Y + \sigma \sqrt{Y} Z),

where :math:`Z \sim N(0, 1)` and :math:`\sigma > 0`. Denote the standard
normal density by :math:`\varphi` and CDF by :math:`\Phi`.

First Derivatives
~~~~~~~~~~~~~~~~~

.. math::

   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial \mu} &= -1, \\
   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial \gamma}
   &= -\frac{E\!\left[\sqrt{Y} \, \varphi\!\left(
   \frac{-r_{\operatorname{VaR}_\alpha} - \mu - \gamma Y}
   {\sigma \sqrt{Y}}\right)\right]}
   {E\!\left[\frac{1}{\sqrt{Y}} \, \varphi\!\left(
   \frac{-r_{\operatorname{VaR}_\alpha} - \mu - \gamma Y}
   {\sigma \sqrt{Y}}\right)\right]}, \\
   \frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial \sigma}
   &= \sigma^{-1} \left(r_{\operatorname{VaR}_\alpha} + \mu
   - \gamma \frac{\partial r_{\operatorname{VaR}_\alpha}}
   {\partial \gamma}\right),

and

.. math::
   :label: cvar-nm-grad

   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial \mu} &= -1, \\
   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial \gamma}
   &= -\alpha^{-1} E\!\left[Y \, \Phi\!\left(
   \frac{-r_{\operatorname{VaR}_\alpha} - \mu - \gamma Y}
   {\sigma \sqrt{Y}}\right)\right], \\
   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial \sigma}
   &= \sigma^{-1} \left(r_{\operatorname{CVaR}_\alpha} + \mu
   - \gamma \frac{\partial r_{\operatorname{CVaR}_\alpha}}
   {\partial \gamma}\right).

Second Derivatives
~~~~~~~~~~~~~~~~~~

.. math::
   :label: cvar-nm-hessian

   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}{\partial \mu^2}
   = \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial \mu \, \partial \gamma}
   = \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial \mu \, \partial \sigma} &= 0, \\
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}{\partial \gamma^2}
   &= \frac{1}{\alpha \sigma} E\!\left[\sqrt{Y} \, \varphi\!\left(
   \frac{-r_{\operatorname{VaR}_\alpha} - \mu - \gamma Y}
   {\sigma \sqrt{Y}}\right)
   \left(\frac{\partial r_{\operatorname{VaR}_\alpha}}{\partial \gamma}
   + Y\right)\right], \\
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial \gamma \, \partial \sigma}
   &= -\frac{\gamma}{\sigma}
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}{\partial \gamma^2}, \\
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}{\partial \sigma^2}
   &= -\frac{\gamma}{\sigma}
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial \gamma \, \partial \sigma}.

All derivatives can be computed via Monte Carlo by generating i.i.d. samples
of the mixing variable :math:`Y`.

Portfolio CVaR Gradient and Hessian
-----------------------------------

For a portfolio :math:`w`, using :eq:`nm-portfolio` we can write
:math:`r_{\operatorname{CVaR}_\alpha}(w) =
r_{\operatorname{CVaR}_\alpha}(w^\top \mu, w^\top \gamma,
\sqrt{w^\top \Sigma w})`. The chain rule gives:

.. math::

   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial w_j}(w)
   = -\mu_j + \gamma_j \frac{\partial r_{\operatorname{CVaR}_\alpha}}
   {\partial \gamma}
   + \frac{(\Sigma w)_j}{\sqrt{w^\top \Sigma w}}
   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial \sigma},

where the partial derivatives on the right are evaluated at
:math:`(w^\top \mu, w^\top \gamma, \sqrt{w^\top \Sigma w})`.

The **Hessian matrix** is:

.. math::

   H_{r_{\operatorname{CVaR}_\alpha}}(w)
   &= \gamma \gamma^\top
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}{\partial \gamma^2}
   + (w^\top \Sigma w)^{-1/2}
   (\gamma w^\top \Sigma + \Sigma w \, \gamma^\top)
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}
   {\partial \gamma \, \partial \sigma} \\
   &\quad + (w^\top \Sigma w)^{-1}
   \Sigma w \, w^\top \Sigma
   \frac{\partial^2 r_{\operatorname{CVaR}_\alpha}}{\partial \sigma^2} \\
   &\quad + (w^\top \Sigma w)^{-3/2}
   (\Sigma \, w^\top \Sigma w - \Sigma w \, w^\top \Sigma)
   \frac{\partial r_{\operatorname{CVaR}_\alpha}}{\partial \sigma}.

References
----------

.. [RauHasanov2004] Rau-Bredow, H. (2004). Value-at-risk, expected shortfall,
   and marginal risk contribution. In *Risk Measures for the 21st Century*,
   Wiley.

.. [Tasche1999] Tasche, D. (1999). Risk contributions and performance
   measurement. Report of the Lehrstuhl für mathematische Statistik, TU München.
