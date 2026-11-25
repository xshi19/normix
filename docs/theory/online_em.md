# Online EM Algorithm

This page states the sequential updates normix computes: the statistic
recursion, the implemented step sizes, the generalized-hyperbolic
M-step on the running state, and the constrained M-step for a curved
family. The aggregation identity for the index $t = 1,\ldots,T$, the
Bregman regret identity, and the comparison of the constrained and
ambient stationary equations are derived in the
[online-EM note](https://xshi19.github.io/math/normix-theory/normix-online-em/).
The method is {ref}`Cappe2009 <cappe2009>`.

normix writes a density as $f(x\mid\theta)$. On that site the same
function is $p_\theta(x)$.

The Legendre dual on this page is
$\varphi(\eta) = \langle\theta, \eta\rangle - \psi(\theta)$, so
$\theta = \nabla\varphi(\eta)$ and $\eta = \nabla\psi(\theta)$.
The online-EM note writes this dual as $\psi^*$.
The note writes a classical or constrained parameter as $\vartheta$.
This page uses $\theta$ for the natural parameter. A factor model keeps
the symbol $u$ from {doc}`factor_analysis`.

**See also:** executable walkthrough in
{doc}`../tutorials/em/02_incremental_em`.

## Sequential update

For an observation $x_t$, write

```{math}
\bar t(x_t) = E[t(X, Y) \mid X = x_t]
```

at the parameter available when $x_t$ arrives.
{class}`~normix.fitting.em.IncrementalEMFitter` keeps the running state
in a {class}`~normix.fitting.eta.NormalMixtureEta`, starting from
{meth}`~normix.mixtures.marginal.NormalMixture.compute_eta_from_model`.
One step applies

```{math}
\eta \leftarrow a + b\,\eta + c\,\bar t,
```

with weights from
{meth}`~normix.fitting.eta_rules.AffineRule.weights`.
For the first four rules below, $a = 0$.

- {class}`~normix.fitting.eta_rules.IdentityUpdate`: $b = 0$, $c = 1$.
  The running state is replaced by the new average.
- {class}`~normix.fitting.eta_rules.RobbinsMonroUpdate`: at step
  $t = 0, 1, 2, \ldots$,

  ```{math}
  c = \frac{1}{\tau_0 + t}, \qquad b = 1 - c,
  ```

  with $\tau_0 + t \ne 0$. The class default is $\tau_0 = 10$, so the
  first weight is $1/10$.
  {class}`~normix.fitting.em.IncrementalEMFitter` uses that default
  when no rule is passed, and it passes $t = 0$ on the first batch.
  This is the Robbins–Monro schedule
  {ref}`RobbinsMonro1951 <robbinsmonro1951>` at that index.
- {class}`~normix.fitting.eta_rules.SampleWeightedUpdate`: if $n$ is the
  cumulative count before the batch and $m$ is the batch size,

  ```{math}
  b = \frac{n}{n+m}, \qquad c = \frac{m}{n+m},
  ```

  and then $n \leftarrow n + m$. The count starts at $0$, so the first
  batch has $b = 0$ and $c = 1$ and replaces the initial state.
  Every batch satisfies $(n+m)\eta_{\mathrm{new}} = n\,\eta + m\,\bar t$
  with the count from before the batch.
- {class}`~normix.fitting.eta_rules.EWMAUpdate`: $b = 1 - w$ and
  $c = w$, with default $w = 0.1$. The weight does not depend on $t$.
- {class}`~normix.fitting.eta_rules.AffineUpdate` stores constant
  $(a, b, c)$.

If `inner_iter` is $1$, $\bar t$ is the E-step average on the current
mini-batch. If `inner_iter` is greater than $1$, the fitter runs that
many E-step, M-step, and regularization cycles on the mini-batch and
sets $\bar t$ to the expectation parameters of the fitted model. The
fitter then applies {eq}`m-step` to the updated state, followed by the
same regularization options as
{class}`~normix.fitting.em.BatchEMFitter`. The default is `none`.
Each batch is drawn from the sample after the sample mean has been
removed. That mean is added back to $\mu$ at the end. $\gamma$,
$\Sigma$, and the mixing parameters are unchanged by the shift.

The online-EM note indexes the reciprocal schedule by
$t = 1,\ldots,T$, so its first weight is $1/(\tau_0+1)$.
Its aggregation identity is for that index. See the
[online-EM note](https://xshi19.github.io/math/normix-theory/normix-online-em/#posterior-statistics-and-the-sequential-update).
The same weight families are set out as update rules in the
[EM-framework note](https://xshi19.github.io/math/normix-theory/normix-em-framework/#two-layers-of-update-rules).
{class}`~normix.fitting.eta_rules.Shrinkage` composes any of these rules
with the penalty in {doc}`shrinkage`.

The Bregman regret identity for the note's index is derived in the
[online-EM note](https://xshi19.github.io/math/normix-theory/normix-online-em/#a-bregman-regret-identity).

## Generalized hyperbolic mixtures

The conditional moments on a batch are {eq}`gig-moment-cond`.
The six blocks are the averages {eq}`e-step`, computed on the batch
rather than on the full sample, and stored in that order.
The running state uses the same order. The parameter update is
{eq}`m-step` at the running state. The GIG arguments are that state's
log moment, inverse moment, and first moment, in that order. For the
joint GH family this is the classical form of $\theta = \nabla\varphi(\eta)$.
The batch formulas and the boundary-moment checks are in the
[online-EM note](https://xshi19.github.io/math/normix-theory/normix-online-em/#application-to-generalized-hyperbolic-mixtures).
The fitter options are described in {doc}`../design/em_framework`.

## Curved families

A curved family has classical parameter $u$ and natural map
$\theta(u)$,

```{math}
f(x, y \mid u) = h(x, y)\exp\!\left(
\langle\theta(u), t(x, y)\rangle - \psi(\theta(u))\right).
```

The M-step on a running state $\eta$ is

```{math}
u \in \arg\max_u \left\{
\langle\theta(u), \eta\rangle - \psi(\theta(u))\right\}.
```

At an interior stationary point,

```{math}
D\theta(u)^\top \bigl(\eta - \nabla\psi(\theta(u))\bigr) = 0.
```

{meth}`~normix.mixtures.factor.FactorNormalMixture.m_step` evaluates
{eq}`fa-mstep` for $(\mu, \gamma, F, D)$. The mixing block uses the
moment order of {eq}`m-step`: log moment, inverse moment, then first
moment.
{class}`~normix.fitting.em.IncrementalEMFitter` calls that method on a
factor model. The comparison with the ambient equation
$\eta = \nabla\psi(\theta(u))$ is in the
[online-EM note](https://xshi19.github.io/math/normix-theory/normix-online-em/#curved-families-need-a-constrained-m-step).
