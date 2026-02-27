Mean-Risk Optimization for Normal Mixture Distributions
======================================================

This section develops the mean-risk portfolio optimization framework for
normal mixture distributions. The key insight is that the normal mixture
structure :eq:`gh-def` enables a dimension reduction from :math:`d` assets
to a two-dimensional problem.

Coherent Risk Measures
----------------------

**Definition.** Let :math:`(\Omega, \mathcal{F}, \mathbb{P})` be a
probability space and :math:`\mathcal{L}(\Omega, \mathcal{F})` the set of
real-valued random variables. A **coherent risk measure** is a function
:math:`\rho : \mathcal{L} \to \mathbb{R}` satisfying [Artzner1999]_:

1. *Monotonicity:* If :math:`X \leq Y`, then :math:`\rho(X) \geq \rho(Y)`.
2. *Translation invariance:* :math:`\rho(X + c) = \rho(X) - c` for all :math:`c \in \mathbb{R}`.
3. *Positive homogeneity:* :math:`\rho(\lambda X) = \lambda \rho(X)` for all :math:`\lambda \geq 0`.
4. *Subadditivity:* :math:`\rho(X + Y) \leq \rho(X) + \rho(Y)`.

**Definition.** For a continuous random variable :math:`X` and
:math:`\alpha \in (0, 1)`:

.. math::

   \operatorname{VaR}_\alpha(X) &:= -\inf\{x \in \mathbb{R} :
   \mathbb{P}(X \leq x) > \alpha\}, \\
   \operatorname{CVaR}_\alpha(X) &:= -E[X \mid X \leq
   -\operatorname{VaR}_\alpha(X)].

VaR is widely used but is **not** coherent (it lacks subadditivity). CVaR
is coherent.

Risk Monotonicity for Normal Mixtures
-------------------------------------

Recall that a normal mixture random vector can be written as:

.. math::
   :label: nm-def

   X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z,

where :math:`\mu, \gamma \in \mathbb{R}^d`, :math:`Y \geq 0` is a univariate
random variable, and :math:`Z \sim N(0, \Sigma)` is independent of :math:`Y`.

Consider the univariate case :math:`d = 1` with :math:`Z \sim N(0, \sigma^2)`,
:math:`\sigma > 0`.

**Theorem 1.** Let :math:`\rho` be a coherent risk measure that depends only
on the distribution. If :math:`X` follows :eq:`nm-def`, then:

1. :math:`\mu \mapsto \rho(X)` is decreasing.
2. :math:`\gamma \mapsto \rho(X)` is non-increasing.
3. :math:`\sigma \mapsto \rho(X)` is non-decreasing on :math:`\mathbb{R}^+`.

*Proof.*

(i) By translation invariance:
:math:`\rho(X) = \rho(\gamma Y + \sqrt{Y} \sigma Z) - \mu`, which is
decreasing in :math:`\mu`.

(ii) For any :math:`\Delta\gamma \geq 0`:

.. math::

   \rho((\gamma + \Delta\gamma) Y + \sqrt{Y} \sigma Z)
   &\leq \rho(\gamma Y + \sqrt{Y} \sigma Z)
   + \rho(\Delta\gamma \, Y) \\
   &\leq \rho(\gamma Y + \sqrt{Y} \sigma Z),

since :math:`\rho(\Delta\gamma \, Y) \leq \rho(0) = 0` by monotonicity
(:math:`\Delta\gamma \, Y \geq 0`).

(iii) The map :math:`\sigma \mapsto \rho(\gamma Y + \sigma \sqrt{Y} Z)` is
convex:

.. math::

   \rho(\gamma Y + (a \sigma_1 + (1-a) \sigma_2) \sqrt{Y} Z)
   \leq a \, \rho(\gamma Y + \sigma_1 \sqrt{Y} Z)
   + (1-a) \, \rho(\gamma Y + \sigma_2 \sqrt{Y} Z),

and symmetric about zero (since
:math:`\gamma Y + \sigma \sqrt{Y} Z \stackrel{d}{=}
\gamma Y - \sigma \sqrt{Y} Z`). A convex symmetric function is
non-decreasing on :math:`\mathbb{R}^+`. :math:`\square`

Intuitively, :eq:`nm-def` can be viewed as a portfolio with a risk-free
component :math:`\mu`, a non-negative-return asset with weight :math:`\gamma`,
and a risky asset with weight :math:`\sigma`. Any coherent risk measure
prefers large :math:`\mu` and :math:`\gamma` and small :math:`\sigma`.

Portfolio Return as Normal Mixture
----------------------------------

In the :math:`d`-dimensional case, a portfolio with weight
:math:`w \in \mathbb{R}^d` (:math:`w^\top \mathbf{e} = 1`) has return:

.. math::
   :label: nm-portfolio

   w^\top X \stackrel{d}{=} w^\top \mu
   + w^\top \gamma \, Y + \sqrt{w^\top \Sigma w \, Y} \, Z,

where :math:`Z \sim N(0, 1)`. The expected return is:

.. math::

   E[w^\top X] = w^\top \mu + w^\top \gamma \, E[Y].

Mean-Risk Optimization
----------------------

The generalized mean-risk optimization problem is:

.. math::
   :label: mean-risk-opt

   \min_w \; \rho(w^\top X) \quad
   \text{s.t.} \quad w^\top \mathbf{e} = 1, \quad
   E[w^\top X] \geq m,

where :math:`m \in \mathbb{R}`.

Dimension Reduction via the Efficient Surface
----------------------------------------------

**Proposition.** The solution of :eq:`mean-risk-opt` is:

.. math::

   w^* = \Sigma^{-1} [\mu \; \gamma \; \mathbf{e}] \, A^{-1}
   [\tilde{\mu}^* \; \tilde{\gamma}^* \; 1]^\top,

where :math:`\tilde{\mu}^*, \tilde{\gamma}^* \in \mathbb{R}` solve the
two-dimensional problem:

.. math::

   \min_{\tilde{\mu}, \tilde{\gamma}} \;
   \rho\!\left(\tilde{\mu} + \tilde{\gamma} \, Y
   + \sqrt{g(\tilde{\mu}, \tilde{\gamma}) \, Y} \, Z\right)
   \quad \text{s.t.} \quad
   \tilde{\mu} + \tilde{\gamma} \, E[Y] \geq m,

with

.. math::

   g(\tilde{\mu}, \tilde{\gamma}) = [\tilde{\mu}, \tilde{\gamma}, 1] \,
   A^{-1} [\tilde{\mu}, \tilde{\gamma}, 1]^\top, \qquad
   A = [\mu \; \gamma \; \mathbf{e}]^\top \Sigma^{-1}
   [\mu \; \gamma \; \mathbf{e}].

*Proof.* Define :math:`w^*(\tilde{\mu}, \tilde{\gamma})` as the solution of:

.. math::
   :label: constrained-opt

   w^*(\tilde{\mu}, \tilde{\gamma}) := \arg\min_w \rho(w^\top X)
   \quad \text{s.t.} \quad w^\top \mathbf{e} = 1, \;
   w^\top \mu = \tilde{\mu}, \; w^\top \gamma = \tilde{\gamma}.

Then :eq:`mean-risk-opt` is equivalent to optimizing over
:math:`(\tilde{\mu}, \tilde{\gamma})` with constraint
:math:`\tilde{\mu} + \tilde{\gamma} E[Y] \geq m`.

By Theorem 1 and :eq:`nm-portfolio`, when :math:`w^\top \mu` and
:math:`w^\top \gamma` are fixed, :math:`\rho(w^\top X)` is non-decreasing
in :math:`w^\top \Sigma w`. Therefore :eq:`constrained-opt` reduces to:

.. math::

   w^*(\tilde{\mu}, \tilde{\gamma}) = \arg\min_w w^\top \Sigma w

with the same constraints. By Lagrange multipliers:

.. math::

   w^*(\tilde{\mu}, \tilde{\gamma}) = \Sigma^{-1}
   [\mu \; \gamma \; \mathbf{e}] \, A^{-1}
   [\tilde{\mu} \; \tilde{\gamma} \; 1]^\top.

Substituting into :eq:`nm-portfolio` completes the proof. :math:`\square`

This reduces the :math:`d`-dimensional problem to a two-dimensional one in
:math:`(\tilde{\mu}, \tilde{\gamma})`. The surface
:math:`(\tilde{\mu}, \tilde{\gamma}) \mapsto \rho` is the **efficient surface**,
generalizing the classical efficient frontier.

Worst-Case Risk Measures
------------------------

**Definition.** Let :math:`\mathcal{P}` be a set of probability distributions.
The **worst-case** coherent risk measure is:

.. math::

   \rho^*(w) := \sup_{f \in \mathcal{P}} \rho(w^\top X).

For the **box uncertainty set** of normal mixture models:

.. math::

   \mathcal{P} = \{f_X(\cdot | \mu, \gamma, \Sigma) :
   \underline{\mu} \preceq \mu \preceq \overline{\mu}, \;
   \underline{\gamma} \preceq \gamma \preceq \overline{\gamma}, \;
   \underline{\Sigma} \preceq \Sigma \preceq \overline{\Sigma}, \;
   f_Y \text{ fixed}\},

where :math:`\preceq` denotes element-wise inequality, the following holds:

**Proposition.** For any :math:`w \in \mathbb{R}^d_+`:

.. math::

   \rho^*(w) = \rho\!\left(w^\top \underline{\mu}
   + w^\top \underline{\gamma} \, Y
   + \sqrt{w^\top \overline{\Sigma} w \, Y} \, Z\right).

This follows directly from Theorem 1: the worst case uses the smallest
:math:`\mu`, smallest :math:`\gamma`, and largest :math:`\Sigma`.

References
----------

.. [Artzner1999] Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999).
   Coherent measures of risk. *Mathematical Finance*, 9(3), 203-228.
