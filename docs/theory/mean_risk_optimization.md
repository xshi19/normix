# Mean-Risk Optimization for Normal Mixture Distributions

This section develops the mean-risk portfolio optimization framework for
normal mixture distributions. The key insight is that the normal mixture
structure {eq}`gh-def` enables a dimension reduction from $d$ assets
to a two-dimensional problem.

## Coherent Risk Measures

**Definition.** Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a
probability space and $\mathcal{L}(\Omega, \mathcal{F})$ the set of
real-valued random variables. A **coherent risk measure** is a function
$\rho : \mathcal{L} \to \mathbb{R}$ satisfying [Artzner1999](#artzner1999):

1. *Monotonicity:* If $X \leq Y$, then $\rho(X) \geq \rho(Y)$.
2. *Translation invariance:* $\rho(X + c) = \rho(X) - c$ for all $c \in \mathbb{R}$.
3. *Positive homogeneity:* $\rho(\lambda X) = \lambda \rho(X)$ for all $\lambda \geq 0$.
4. *Subadditivity:* $\rho(X + Y) \leq \rho(X) + \rho(Y)$.

**Definition.** For a continuous random variable $X$ and
$\alpha \in (0, 1)$:

```{math}
\operatorname{VaR}_\alpha(X) &:= -\inf\{x \in \mathbb{R} :
\mathbb{P}(X \leq x) > \alpha\}, \\
\operatorname{CVaR}_\alpha(X) &:= -E[X \mid X \leq
-\operatorname{VaR}_\alpha(X)].
```

VaR is widely used but is **not** coherent (it lacks subadditivity). CVaR
is coherent.

## Risk Monotonicity for Normal Mixtures

Recall that a normal mixture random vector can be written as:

```{math}
:label: nm-def

X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z,
```

where $\mu, \gamma \in \mathbb{R}^d$, $Y \geq 0$ is a univariate
random variable, and $Z \sim N(0, \Sigma)$ is independent of $Y$.

Consider the univariate case $d = 1$ with $Z \sim N(0, \sigma^2)$,
$\sigma > 0$.

**Theorem 1.** Let $\rho$ be a coherent risk measure that depends only
on the distribution. If $X$ follows {eq}`nm-def`, then:

1. $\mu \mapsto \rho(X)$ is decreasing.
2. $\gamma \mapsto \rho(X)$ is non-increasing.
3. $\sigma \mapsto \rho(X)$ is non-decreasing on $\mathbb{R}^+$.

*Proof.*

(i) By translation invariance:
$\rho(X) = \rho(\gamma Y + \sqrt{Y} \sigma Z) - \mu$, which is
decreasing in $\mu$.

(ii) For any $\Delta\gamma \geq 0$:

```{math}
\rho((\gamma + \Delta\gamma) Y + \sqrt{Y} \sigma Z)
&\leq \rho(\gamma Y + \sqrt{Y} \sigma Z)
+ \rho(\Delta\gamma \, Y) \\
&\leq \rho(\gamma Y + \sqrt{Y} \sigma Z),
```

since $\rho(\Delta\gamma \, Y) \leq \rho(0) = 0$ by monotonicity
($\Delta\gamma \, Y \geq 0$).

(iii) The map $\sigma \mapsto \rho(\gamma Y + \sigma \sqrt{Y} Z)$ is
convex:

```{math}
\rho(\gamma Y + (a \sigma_1 + (1-a) \sigma_2) \sqrt{Y} Z)
\leq a \, \rho(\gamma Y + \sigma_1 \sqrt{Y} Z)
+ (1-a) \, \rho(\gamma Y + \sigma_2 \sqrt{Y} Z),
```

and symmetric about zero (since
$\gamma Y + \sigma \sqrt{Y} Z \stackrel{d}{=}
\gamma Y - \sigma \sqrt{Y} Z$). A convex symmetric function is
non-decreasing on $\mathbb{R}^+$. $\square$

Intuitively, {eq}`nm-def` can be viewed as a portfolio with a risk-free
component $\mu$, a non-negative-return asset with weight $\gamma$,
and a risky asset with weight $\sigma$. Any coherent risk measure
prefers large $\mu$ and $\gamma$ and small $\sigma$.

## Portfolio Return as Normal Mixture

In the $d$-dimensional case, a portfolio with weight
$w \in \mathbb{R}^d$ ($w^\top \mathbf{e} = 1$) has return:

```{math}
:label: nm-portfolio

w^\top X \stackrel{d}{=} w^\top \mu
+ w^\top \gamma \, Y + \sqrt{w^\top \Sigma w \, Y} \, Z,
```

where $Z \sim N(0, 1)$. The expected return is:

```{math}
E[w^\top X] = w^\top \mu + w^\top \gamma \, E[Y].
```

## Mean-Risk Optimization

The generalized mean-risk optimization problem is:

```{math}
:label: mean-risk-opt

\min_w \; \rho(w^\top X) \quad
\text{s.t.} \quad w^\top \mathbf{e} = 1, \quad
E[w^\top X] \geq m,
```

where $m \in \mathbb{R}$.

## Dimension Reduction via the Efficient Surface

**Proposition.** The solution of {eq}`mean-risk-opt` is:

```{math}
w^* = \Sigma^{-1} [\mu \; \gamma \; \mathbf{e}] \, A^{-1}
[\tilde{\mu}^* \; \tilde{\gamma}^* \; 1]^\top,
```

where $\tilde{\mu}^*, \tilde{\gamma}^* \in \mathbb{R}$ solve the
two-dimensional problem:

```{math}
\min_{\tilde{\mu}, \tilde{\gamma}} \;
\rho\!\left(\tilde{\mu} + \tilde{\gamma} \, Y
+ \sqrt{g(\tilde{\mu}, \tilde{\gamma}) \, Y} \, Z\right)
\quad \text{s.t.} \quad
\tilde{\mu} + \tilde{\gamma} \, E[Y] \geq m,
```

with

```{math}
g(\tilde{\mu}, \tilde{\gamma}) = [\tilde{\mu}, \tilde{\gamma}, 1] \,
A^{-1} [\tilde{\mu}, \tilde{\gamma}, 1]^\top, \qquad
A = [\mu \; \gamma \; \mathbf{e}]^\top \Sigma^{-1}
[\mu \; \gamma \; \mathbf{e}].
```

*Proof.* Define $w^*(\tilde{\mu}, \tilde{\gamma})$ as the solution of:

```{math}
:label: constrained-opt

w^*(\tilde{\mu}, \tilde{\gamma}) := \arg\min_w \rho(w^\top X)
\quad \text{s.t.} \quad w^\top \mathbf{e} = 1, \;
w^\top \mu = \tilde{\mu}, \; w^\top \gamma = \tilde{\gamma}.
```

Then {eq}`mean-risk-opt` is equivalent to optimizing over
$(\tilde{\mu}, \tilde{\gamma})$ with constraint
$\tilde{\mu} + \tilde{\gamma} E[Y] \geq m$.

By Theorem 1 and {eq}`nm-portfolio`, when $w^\top \mu$ and
$w^\top \gamma$ are fixed, $\rho(w^\top X)$ is non-decreasing
in $w^\top \Sigma w$. Therefore {eq}`constrained-opt` reduces to:

```{math}
w^*(\tilde{\mu}, \tilde{\gamma}) = \arg\min_w w^\top \Sigma w
```

with the same constraints. By Lagrange multipliers:

```{math}
w^*(\tilde{\mu}, \tilde{\gamma}) = \Sigma^{-1}
[\mu \; \gamma \; \mathbf{e}] \, A^{-1}
[\tilde{\mu} \; \tilde{\gamma} \; 1]^\top.
```

Substituting into {eq}`nm-portfolio` completes the proof. $\square$

This reduces the $d$-dimensional problem to a two-dimensional one in
$(\tilde{\mu}, \tilde{\gamma})$. The surface
$(\tilde{\mu}, \tilde{\gamma}) \mapsto \rho$ is the **efficient surface**,
generalizing the classical efficient frontier.

## Worst-Case Risk Measures

**Definition.** Let $\mathcal{P}$ be a set of probability distributions.
The **worst-case** coherent risk measure is:

```{math}
\rho^*(w) := \sup_{f \in \mathcal{P}} \rho(w^\top X).
```

For the **box uncertainty set** of normal mixture models:

```{math}
\mathcal{P} = \{f_X(\cdot | \mu, \gamma, \Sigma) :
\underline{\mu} \preceq \mu \preceq \overline{\mu}, \;
\underline{\gamma} \preceq \gamma \preceq \overline{\gamma}, \;
\underline{\Sigma} \preceq \Sigma \preceq \overline{\Sigma}, \;
f_Y \text{ fixed}\},
```

where $\preceq$ denotes element-wise inequality, the following holds:

**Proposition.** For any $w \in \mathbb{R}^d_+$:

```{math}
\rho^*(w) = \rho\!\left(w^\top \underline{\mu}
+ w^\top \underline{\gamma} \, Y
+ \sqrt{w^\top \overline{\Sigma} w \, Y} \, Z\right).
```

This follows directly from Theorem 1: the worst case uses the smallest
$\mu$, smallest $\gamma$, and largest $\Sigma$.

## References

```{eval-rst}
.. [Artzner1999] Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999).
   Coherent measures of risk. *Mathematical Finance*, 9(3), 203-228.
```
