# Generalized Effective Number of Bets

This section extends the ENB framework from {doc}`enb` to general convex
risk measures by diagonalizing the Hessian of the risk function.

## Homogeneous Functions and Risk Decomposition

**Definition.** A function $r : \mathbb{R}^n \to \mathbb{R}$ is
$\tau$-**homogeneous** if $r(tw) = t^\tau r(w)$ for all
$w \in \mathbb{R}^n$ and $t > 0$.

**Proposition** (Tasche [Tasche1999b](#tasche1999b)). Let $r$ be a totally
differentiable, $\tau$-homogeneous function with $\tau \neq 0$.
Then:

1. $\partial r / \partial w_k$ is $(\tau - 1)$-homogeneous.
2. Euler's theorem: $\tau \, r(w) = w^\top \nabla r(w) = \sum_{k=1}^n w_k \, \frac{\partial r}{\partial w_k}(w)$.

This decomposes the risk $r(w)$ into marginal contributions
$\frac{w_k}{\tau} \frac{\partial r}{\partial w_k}(w)$, analogous to
the variance case where $\tau = 2$.

## Local Diagonalization via the Hessian

For a general convex risk function, the marginal contributions
$\frac{\partial r}{\partial w_k}$ are not independent. To extract
locally independent contributions, we use the Taylor expansion:

```{math}
r(w + \Delta w) \approx r(w) + \Delta w^\top \nabla r(w)
+ \frac{1}{2} \Delta w^\top H_r(w) \, \Delta w,
```

where $H_r(w)$ is the (positive semi-definite) Hessian matrix.

Let $T(w)$ diagonalize the Hessian: $T(w) \, H_r(w) \, T(w)^\top
= D(w)$, and set $v = (T(w)^\top)^{-1} w$. Using
$(\tau - 1) \nabla r(w) = H_r(w) w$ (from differentiating Euler's
identity), we obtain:

```{math}
(\tau - 1) \, T(w) \nabla r(w) = D(w) \, v.
```

For $\tau > 1$:

```{math}
r(w) = \frac{1}{\tau} w^\top \nabla r(w)
= \frac{1}{\tau(\tau - 1)} v^\top D(w) \, v
= \sum_{k=1}^n \frac{d_k(w) \, v_k^2}{\tau(\tau - 1)},
```

where $d_k(w)$ are the diagonal entries of $D(w)$. Since
$H_r(w)$ is positive semi-definite, we have $d_k(w) \geq 0$,
which ensures that the risk contributions are non-negative when
$\tau > 1$.

### Local Independence

The Taylor expansion in the transformed coordinates is:

```{math}
r(w + \Delta w) &\approx r(w) + \frac{1}{\tau - 1} \Delta v^\top D(w) \, v
+ \frac{1}{\tau} \Delta v^\top D(w) \, \Delta v,
```

where $\Delta v = (T(w)^\top)^{-1} \Delta w$. Each component of
$\Delta v$ has an approximately independent contribution to the change
$r(w + \Delta w) - r(w)$, justifying the decomposition.

## Generalized ENB

The generalized ENB is defined as:

```{math}
p_k(w) &= \frac{d_k(w) \, v_k^2}{\tau(\tau - 1) \, r(w)},
\quad k = 1, \ldots, n, \\
N(w) &= \exp\!\left(-\sum_{k=1}^n p_k(w) \log p_k(w)\right).
```

Unlike the variance case, the Hessian $H_r(w)$ depends on $w$,
so the transformation $T(w)$ must be recomputed for each portfolio.
However, the structural results from {doc}`enb` still apply:

Let

```{math}
C(w) = \operatorname{diag}(H_r(w))^{-1/2} \, H_r(w) \,
\operatorname{diag}(H_r(w))^{-1/2}
```

be the "correlation" of the Hessian, with eigendecomposition
$C(w) = U(w) \, S(w) \, U(w)^\top$. Then $T(w)$ has the
representation:

```{math}
T(w) = D^{1/2} V \, S(w)^{-1/2} U(w)^\top
\operatorname{diag}(H_r(w))^{-1/2},
```

where $D$ is diagonal and $V$ is orthogonal. The ENB is again
independent of $D$, and the **constrained minimum torsion**
transformation is:

```{math}
T_{MT}(w) = U(w) \, S(w)^{-1/2} U(w)^\top
\operatorname{diag}(H_r(w))^{-1/2}.
```

## Application to Coherent Risk Measures

Given a coherent risk measure $\rho$, define
$r_\rho(w) = \rho(w^\top X)$. By positive homogeneity and
subadditivity, $r_\rho$ is convex and **1-homogeneous**
($\tau = 1$).

Since the generalized ENB requires $\tau > 1$, we work with the
**squared risk** $r_\rho^2(w)$, which is 2-homogeneous. This is
analogous to using variance (the square of standard deviation) in the
original ENB framework.

The Hessian of $r_\rho^2(w)$ can be computed from the CVaR gradient
and Hessian formulas in {doc}`cvar_derivatives`:

```{math}
H_{r_\rho^2}(w) = 2 \nabla r_\rho(w) \, \nabla r_\rho(w)^\top
+ 2 \, r_\rho(w) \, H_{r_\rho}(w).
```

The CVaR-based ENB can reveal tail-risk concentrations that the
variance-based ENB misses. For example, in a portfolio of independent assets
where one has heavier tails, the variance-based ENB treats all assets equally
(since the covariance is diagonal), while the CVaR-based ENB correctly
assigns a larger risk contribution to the heavy-tailed asset.

## References

```{eval-rst}
.. [Tasche1999b] Tasche, D. (1999). Risk contributions and performance
   measurement. Report of the Lehrstuhl für mathematische Statistik, TU München.
```
