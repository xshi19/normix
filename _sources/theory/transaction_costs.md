# Portfolio Optimization with Transaction Costs

This section extends the mean-risk optimization framework to include
transaction costs by using a quadratic programming approximation.

## Problem Formulation

Consider the $d$-dimensional portfolio optimization problem:

```{math}
:label: tc-opt

\max_{w \in \mathbb{R}^d} \;
w^\top m - c_1 \, r(w) - c_2 \|w - w_0\|_1
\quad \text{s.t.} \quad
w^\top \mathbf{e} = 1, \;
A w \leq b,
```

where $w$ is the portfolio weight, $m \in \mathbb{R}^d$ is the
expected return, $r : \mathbb{R}^d \to \mathbb{R}$ is a convex
non-negative risk function, $w_0$ is the current portfolio weight
satisfying the constraints, and $c_1, c_2 > 0$ are constants.

The three terms in the objective are:

- $w^\top m$: expected return,
- $c_1 \, r(w)$: risk penalty,
- $c_2 \|w - w_0\|_1$: transaction cost (turnover penalty).

When $r(w) = \rho(w^\top X)$ for a coherent risk measure $\rho$
and normal mixture returns $X$, the dimension reduction of
{doc}`mean_risk_optimization` cannot be applied directly due to the
transaction cost term.

## Quadratic Approximation

When transaction costs constrain the solution to be close to $w_0$, we
can approximate the convex risk function by its Taylor expansion:

```{math}
r(w) \approx r(w_0) + (w - w_0)^\top \nabla r(w_0)
+ \frac{1}{2} (w - w_0)^\top H_r(w_0) (w - w_0),
```

where $\nabla r$ is the gradient and $H_r$ is the Hessian of
$r$. The approximate optimization problem becomes:

```{math}
:label: tc-approx

\max_{w \in \mathbb{R}^d} \;
w^\top (m - c_1 \nabla r(w_0))
- \frac{c_1}{2} (w - w_0)^\top H_r(w_0) (w - w_0)
- c_2 \|w - w_0\|_1
\quad \text{s.t.} \quad
w^\top \mathbf{e} = 1, \;
A w \leq b.
```

## Reduction to Quadratic Programming

Problem {eq}`tc-approx` is convex but non-smooth at $w_0$. It can be
reformulated as a quadratic program by introducing buy and sell variables.
Let $v = (v^+; v^-)$ where $v^+, v^- \in \mathbb{R}^d$ with
$v^+, v^- \geq 0$ and $w = w_0 + v^+ - v^-$. Then:

```{math}
:label: tc-qp

\max_{v \geq 0} \;
v^\top \tilde{m} - \frac{c_1}{2} v^\top \tilde{H} v
\quad \text{s.t.} \quad
v^\top \tilde{\mathbf{e}} = 0, \;
\tilde{A} v \leq \tilde{b},
```

where

```{math}
\tilde{m} = \begin{pmatrix}
m - c_1 \nabla r(w_0) - c_2 \mathbf{e} \\
-m + c_1 \nabla r(w_0) - c_2 \mathbf{e}
\end{pmatrix}, \quad
\tilde{H} = \begin{pmatrix}
H_r(w_0) & -H_r(w_0) \\
-H_r(w_0) & H_r(w_0)
\end{pmatrix},
```

```{math}
\tilde{\mathbf{e}} = \begin{pmatrix}
\mathbf{e} \\ -\mathbf{e}
\end{pmatrix}, \quad
\tilde{A} = (A \; -A), \quad
\tilde{b} = b - A w_0.
```

If $v^*$ is the solution of {eq}`tc-qp`, then the optimal portfolio
weight is:

```{math}
w^* = w_0 + (I \; -I) \, v^*.
```

```{note}

The matrix $\tilde{H}$ is not full rank, so {eq}`tc-qp` is not
strictly convex. In practice, a small regularization of the zero
eigenvalues may be needed for certain QP solvers. One should verify that
the objective at $w^*$ exceeds the objective at $w_0$;
otherwise, holding the current position is preferable.
```

The gradient $\nabla r(w_0)$ and Hessian $H_r(w_0)$ for CVaR
risk can be computed using the formulas in {doc}`cvar_derivatives`. The QP
formulation is typically orders of magnitude faster than solving the
original non-smooth convex problem.

Executable walkthrough:
{doc}`../tutorials/finance/06_transaction_costs`.
