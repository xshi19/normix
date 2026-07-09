# Effective Number of Bets and Minimum Torsion

This section reviews the effective number of bets (ENB) and the minimum
torsion approach for measuring portfolio diversification, following
[Meucci2010](#meucci2010) and [Meucci2014](#meucci2014).

## Variance-Based Risk Decomposition

Let $X \in \mathbb{R}^n$ be a random vector of asset returns with
covariance $\Sigma$, and let $w \in \mathbb{R}^n$ with
$w^\top \mathbf{e} = 1$ be the portfolio weights. The portfolio
variance is:

```{math}
r_{\operatorname{Var}}(w) = w^\top \Sigma w.
```

The gradient is $\nabla r_{\operatorname{Var}}(w) = 2 \Sigma w$, so the
marginal contribution of asset $k$ to variance is
$w_k (\Sigma w)_k$. These contributions are generally **not**
independent.

## Uncorrelated Factor Decomposition

To obtain independent contributions, we seek an invertible matrix
$T \in \mathbb{R}^{n \times n}$ such that $T \Sigma T^\top = D$
is diagonal. Let $Y = TX$ be the transformed returns and
$v = (T^\top)^{-1} w$ the adjusted weights. Then:

```{math}
w^\top X = v^\top Y, \qquad
w^\top \Sigma w = v^\top D v = \sum_{k=1}^n d_k v_k^2,
```

where $d_k$ are the diagonal entries of $D$. The risk
contributions $d_k v_k^2$ are now independent.

## Effective Number of Bets

The **normalized risk contributions** form a discrete distribution:

```{math}
:label: enb-weights

p_k = \frac{d_k v_k^2}{w^\top \Sigma w}, \quad k = 1, \ldots, n.
```

Since $p_k \geq 0$ and $\sum_k p_k = 1$, the **ENB** is defined
as the exponential entropy:

```{math}
N = \exp\!\left(-\sum_{k=1}^n p_k \log p_k\right).
```

The ENB ranges from 1 (risk concentrated in one factor) to $n$
(equally distributed among all factors). Equivalently,
$-\log N$ is proportional to the KL divergence between
$\{p_k\}$ and the uniform distribution.

## Characterizing Diagonalizations

Let $C = U S U^\top$ be the eigendecomposition of the correlation matrix
$C = \operatorname{diag}(\Sigma)^{-1/2} \Sigma \,
\operatorname{diag}(\Sigma)^{-1/2}$, where $U$ is orthogonal and
$S$ is diagonal.

**Proposition.** Let $\Sigma$ be positive definite and $T$ be
invertible with $T \Sigma T^\top = D$ diagonal. Then there exists an
orthogonal matrix $V$ such that:

```{math}
:label: T-representation

T = D^{1/2} V S^{-1/2} U^\top
\operatorname{diag}(\Sigma)^{-1/2}.
```

**Lemma.** The ENB $N$ is independent of the choice of $D$.

*Proof.* Let $u = V S^{-1/2} U^\top
\operatorname{diag}(\Sigma)^{1/2} w$ and $v = D^{-1/2} u$. Then
$p_k = u_k^2 / (w^\top \Sigma w)$, which does not depend on
$D$. $\square$

Therefore, it suffices to choose only the orthogonal matrix $V$.
Since $V$ is a rotation, it can map the vector
$S^{-1/2} U^\top \operatorname{diag}(\Sigma)^{1/2} w$ to any
direction. In particular:

- Choosing $V$ so that all $v_k$ are equal gives $N = n$ (maximal diversification).
- Choosing $V$ to concentrate on one component gives $N = 1$.

Thus, the diagonalization $T$ must be chosen carefully.

## Minimum Torsion

The **minimum torsion** approach [Meucci2014](#meucci2014) selects $T$ to minimize
the change from the original weights. The rationale is that if $w$ is
close to equally weighted, then $v = (T^\top)^{-1} w$ should also be
close to equally weighted.

The degree of change is measured by the **normalized tracking error**:

```{math}
\operatorname{NTE}(T) = \sqrt{\frac{1}{n}
\operatorname{tr}\!\left(\operatorname{diag}(\Sigma)^{-1/2}
(T - I) \Sigma (T - I)^\top
\operatorname{diag}(\Sigma)^{-1/2}\right)}.
```

Using representation {eq}`T-representation`, the minimization problem becomes:

```{math}
\min_{D, V} \; \operatorname{tr}\!\left(D
- 2 D^{1/2} V S^{1/2} U^\top\right)
\quad \text{s.t.} \quad
D \text{ diagonal}, \; V \text{ orthogonal}.
```

If $D$ is fixed to the identity matrix, the optimal solution is simply
$V^* = U$, giving the **constrained minimum torsion** transformation:

```{math}
T_{MT} = U S^{-1/2} U^\top
\operatorname{diag}(\Sigma)^{-1/2}.
```

For the general case (unconstrained $D$), an iterative algorithm that
converges rapidly is described in [Meucci2014](#meucci2014).

## References

```{eval-rst}
.. [Meucci2010] Meucci, A. (2010). Managing diversification.
   *Risk Magazine*, 22(5), 74-79.
```

```{eval-rst}
.. [Meucci2014] Meucci, A., Santangelo, A., & Deguest, R. (2014).
   Measuring portfolio diversification based on optimized uncorrelated
   factors. *SSRN Electronic Journal*.
```
