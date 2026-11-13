# EM Algorithm for Generalized Hyperbolic Distributions

This page states the EM formulas that normix computes for a generalized
hyperbolic mixture: the posterior moments, the averaged E-step, the
M-step map, the scale normalization, the ECM cycles, and the mixing
updates for variance-gamma, normal-inverse Gaussian, and
normal-inverse gamma. The complete-data likelihood, the equivariance
argument, and the likelihood inequality are derived in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/).
The algorithm is due to {ref}`Dempster1977 <dempster1977>`. The GH
specialization follows {ref}`Hu2005 <hu2005>`.

normix writes a density as $f(x\mid\theta)$. On that site the same
function is $p_\theta(x)$.

The EM note writes the classical tuple $(\mu, \gamma, \Sigma, p, a, b)$
as $\vartheta$. This page uses $\theta$ for the natural parameter of the
joint family, as in {doc}`gh`. The expectation parameter is
$\eta = \nabla\psi(\theta)$.

**See also:** executable walkthroughs in
{doc}`../tutorials/em/01_batch_em` and
{doc}`../tutorials/em/04_em_vs_mcecm`.

## Conditional distribution

Let $X$ follow {eq}`gh-def`, with $Y \sim \mathrm{GIG}(p, a, b)$ and
$\Sigma$ positive definite. For $a > 0$ and $b > 0$,

```{math}
Y \mid X = x \sim \mathrm{GIG}\!\left(
p - \frac{d}{2},\;
a + \gamma^\top\Sigma^{-1}\gamma,\;
b + q(x)\right),
```

where $q(x) = (x-\mu)^\top\Sigma^{-1}(x-\mu)$.
{meth}`~normix.mixtures.joint.JointNormalMixture.conditional_expectations`
evaluates the posterior moments of this law. The expansion of the joint
density is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#conditional-distribution-of-the-mixing-variable).

## Conditional moments

```{math}
:label: gig-moment-cond

E[Y^\alpha \mid X = x]
= \left(
\sqrt{\frac{b + q(x)}{a + \gamma^\top\Sigma^{-1}\gamma}}
\right)^\alpha
\frac{K_{p - d/2 + \alpha}\!\left(\sqrt{(b + q(x))(a + \tilde q)}\right)}
{K_{p - d/2}\!\left(\sqrt{(b + q(x))(a + \tilde q)}\right)},
```

where $\tilde q = \gamma^\top\Sigma^{-1}\gamma$. This is {eq}`gig-moments`
at the posterior parameters. The log moment is the derivative at
$\alpha = 0$. With $A = a + \tilde q$ and $B = b + q(x)$,

```{math}
E[\log Y \mid X = x]
= \tfrac12\log(B/A)
+ \left.\partial_\nu\log K_\nu(\sqrt{AB})\right|_{\nu = p - d/2}.
```

On a variance-gamma, normal-inverse-Gaussian, or normal-inverse-gamma
model the same ratio is evaluated at the posterior parameters in the
sections below.

## E-step

Let $x_1,\ldots,x_n \in \mathbb{R}^d$.
{meth}`~normix.mixtures.marginal.NormalMixture.e_step` returns the
averages below, stored on
{class}`~normix.fitting.eta.NormalMixtureEta`:

```{math}
:label: e-step

\hat s_1^{(k)} &= \frac1n\sum_{j=1}^n E[Y^{-1}\mid X = x_j], \\
\hat s_2^{(k)} &= \frac1n\sum_{j=1}^n E[Y\mid X = x_j], \\
\hat s_3^{(k)} &= \frac1n\sum_{j=1}^n E[\log Y\mid X = x_j], \\
\hat s_4^{(k)} &= \frac1n\sum_{j=1}^n x_j, \\
\hat s_5^{(k)} &= \frac1n\sum_{j=1}^n x_j\, E[Y^{-1}\mid X = x_j], \\
\hat s_6^{(k)} &= \frac1n\sum_{j=1}^n x_j x_j^\top E[Y^{-1}\mid X = x_j].
```

The conditional moments are {eq}`gig-moment-cond`. The joint sufficient
statistic in {eq}`gh-expectation` is ordered
$(\log Y,\, Y^{-1},\, Y,\, X,\, XY^{-1},\, XX^\top Y^{-1})$.
The EM note uses that order for $\widehat\eta$.
This page keeps the stored order {eq}`e-step`.
The GIG solver receives $(\hat s_3^{(k)}, \hat s_1^{(k)}, \hat s_2^{(k)})$.
The reason for these six blocks is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#e-step-in-one-statistic-order).

## M-step

For a full exponential family the M-step maximizes
$\langle\theta, \eta\rangle - \psi(\theta)$, with $\eta$ in the
sufficient-statistic order of {eq}`gh-expectation`.
The Bregman form of that inversion is in the
[exponential-family note](https://xshi19.github.io/math/normix-theory/normix-exponential-family-core/#from-complete-data-to-em).
{meth}`~normix.mixtures.marginal.NormalMixture.m_step` calls
{meth}`~normix.mixtures.marginal.NormalMixture.from_expectation` and
applies

```{math}
:label: m-step

\mu_{k+1} &= \frac{\hat s_4^{(k)} - \hat s_2^{(k)}\hat s_5^{(k)}}
{1 - \hat s_1^{(k)}\hat s_2^{(k)}}, \\
\gamma_{k+1} &= \frac{\hat s_5^{(k)} - \hat s_1^{(k)}\hat s_4^{(k)}}
{1 - \hat s_1^{(k)}\hat s_2^{(k)}}, \\
\Sigma_{k+1} &= \hat s_6^{(k)} - \hat s_5^{(k)}\mu_{k+1}^\top
- \mu_{k+1}(\hat s_5^{(k)})^\top
+ \hat s_1^{(k)}\mu_{k+1}\mu_{k+1}^\top
- \hat s_2^{(k)}\gamma_{k+1}\gamma_{k+1}^\top, \\
(p_{k+1}, a_{k+1}, b_{k+1}) &= \arg\max_{p,a,b}
L_{\mathrm{GIG}}(p, a, b \mid \hat s_3^{(k)}, \hat s_1^{(k)}, \hat s_2^{(k)}),
```

where $L_{\mathrm{GIG}}$ is {eq}`gig-loglik`.
In the order of {eq}`gh-m-step`,
$(\eta_1,\eta_2,\eta_3,\eta_4,\eta_5,\eta_6)
= (\hat s_3^{(k)}, \hat s_1^{(k)}, \hat s_2^{(k)},
\hat s_4^{(k)}, \hat s_5^{(k)}, \hat s_6^{(k)})$.

A non-degenerate mixing law has $\hat s_1^{(k)}\hat s_2^{(k)} > 1$.
The recovered $\Sigma_{k+1}$ is the covariance of a normal law, so it
has to be positive definite. The mixing maximizer has to lie in the
subordinator's domain. For generalized hyperbolic that domain is

```{math}
\hat s_2^{(k)} > 0,\quad
\hat s_1^{(k)} > 0,\quad
\hat s_1^{(k)}\hat s_2^{(k)} > 1,\quad
-\log\hat s_1^{(k)} < \hat s_3^{(k)} < \log\hat s_2^{(k)}.
```

These bounds are necessary. Equality is a point-mass boundary, and the
inequalities do not guarantee an interior generalized hyperbolic law.

{meth}`~normix.mixtures.marginal.NormalMixture.m_step_normal` updates
only $(\mu, \gamma, \Sigma)$.
{meth}`~normix.mixtures.marginal.NormalMixture.m_step_subordinator`
updates only the mixing law.
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.from_expectation`
is the three-parameter inversion. The algebra of {eq}`m-step` is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#m-step-as-an-expectation-to-parameter-map).

## Scale normalization

For every $s > 0$ the parameter sets

```{math}
(\mu,\;\gamma,\;\Sigma,\;p,\;a,\;b)
\quad\text{and}\quad
(\mu,\;\gamma/s,\;\Sigma/s,\;p,\;a/s,\;bs)
```

give the same marginal law of $X$. The map is the replacement
$Y \mapsto sY$. The concentration $\omega = \sqrt{ab}$ is invariant,
and the scale $\delta = \sqrt{b/a}$ is multiplied by $s$.
The Fisher information $I(\theta) = \nabla^2\psi(\theta)$ is singular on
this orbit. The equivariance argument is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#scale-normalization-and-equivariance).
The same orbit is stated in {doc}`gh`.

{meth}`~normix.mixtures.marginal.NormalMixture.regularize_det_sigma`
chooses $s = \exp((\log|\Sigma| - \tau)/d)$ and applies the map.
{meth}`~normix.mixtures.marginal.NormalMixture.regularize_det_sigma_one`
is the case $\tau = 0$, so $|\Sigma| = 1$ and $s = |\Sigma|^{1/d}$:

```{math}
(\mu, \gamma, \Sigma, p, a, b)
\mapsto
\left(\mu,\;
|\Sigma|^{-1/d}\gamma,\;
|\Sigma|^{-1/d}\Sigma,\;
p,\;
|\Sigma|^{-1/d}a,\;
|\Sigma|^{1/d}b\right).
```

The mixing parameters travel with $s$ as follows.

- Generalized hyperbolic: $(a, b) \mapsto (a/s,\, bs)$, and $p$ is fixed.
  {meth}`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic.regularize_a_eq_b`
  uses $s = \sqrt{a/b}$, which sets $a = b = \omega$.
- Variance-gamma, with rate $\beta$: $(\alpha, \beta) \mapsto (\alpha,\, \beta/s)$.
- Normal-inverse gamma, with rate $\beta$: $(\alpha, \beta) \mapsto (\alpha,\, s\beta)$.
- Normal-inverse Gaussian: $(\mu_{IG}, \lambda) \mapsto (s\mu_{IG},\, s\lambda)$.
  {meth}`~normix.distributions.normal_inverse_gaussian.NormalInverseGaussian.regularize_a_eq_b`
  uses $s = 1/\mu_{IG}$, which sets $\mu_{IG} = 1$.

Variance-gamma and normal-inverse gamma leave
{meth}`~normix.mixtures.marginal.NormalMixture.regularize_a_eq_b`
unchanged, because one GIG rate is zero.
On a factor model the same $s$ acts by $F \mapsto F/\sqrt{s}$ and
$D \mapsto D/s$, through
{meth}`~normix.mixtures.factor.FactorNormalMixture.regularize_det_sigma`.

## ECM cycles

{class}`~normix.fitting.em.BatchEMFitter` runs one of two cycles.
Multi-cycle ECM (MCECM) is the conditional-maximization scheme of
{ref}`MengRubin1993 <mengrubbin1993>`, as used for GH mixtures in
{ref}`McNeil2010 <mcneil2010>`.

- `algorithm='em'`: E-step, then {eq}`m-step`, then the selected
  regularization.
- `algorithm='mcecm'`: E-step, then
  {meth}`~normix.mixtures.marginal.NormalMixture.m_step_normal`, then
  the selected regularization, then a second E-step, then
  {meth}`~normix.mixtures.marginal.NormalMixture.m_step_subordinator`.

The regularization is `none` (the default), `det_sigma_one`,
`det_sigma_x`, or `a_eq_b`. `det_sigma_one` applies the coupled map
above, including $\gamma$ and the mixing parameters.
When a statistic rule is set, it is applied to the first E-step average
only. The fitter subtracts the sample mean from the data and from
$\mu$ before these updates, and adds that mean back to $\mu$.
$\gamma$, $\Sigma$, and the mixing parameters are unchanged by the
shift. A shrinkage target equal to the initial model's expectation
parameters is expressed in the centered coordinates. The penalty itself
is {doc}`shrinkage`.

The ascent argument for an exact conditional maximization is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#ecm-and-multiple-cycles).
An increase of the complete-data objective does not by itself select a
global maximum of the observed likelihood. The fitter loop is described
in {doc}`../design/em_framework`.

## Special mixing families

The normal block remains {eq}`m-step`. Only the mixing update changes.
The three boundary fits are summarized in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#special-mixing-families).
The docstrings of
{class}`~normix.distributions.gamma.Gamma` and
{class}`~normix.distributions.inverse_gamma.InverseGamma` write the
digamma as $\psi$. On this page and in the EM note, $\psi_0$ is the
digamma and $\psi_1 = \psi_0'$ is the trigamma, so $\psi$ remains the
log-partition. The shared Newton step, for a scalar target, is

```{math}
\alpha \leftarrow \alpha -
\frac{\psi_0(\alpha) - \log\alpha - \mathrm{target}}
{\psi_1(\alpha) - 1/\alpha}.
```

### Variance-gamma

$Y \sim \mathrm{Gamma}(\alpha, \beta)$ with rate $\beta$, the limit
$b \to 0$ ({ref}`MadanSeneta1990 <madan1990>`). The moments used by the
update are

```{math}
E[\log Y] = \psi_0(\alpha) - \log\beta, \qquad E[Y] = \alpha/\beta.
```

{meth}`~normix.distributions.gamma.Gamma.from_expectation` receives
$(\hat s_3^{(k)}, \hat s_2^{(k)})$ and solves

```{math}
\psi_0(\alpha) - \log\alpha = \hat s_3^{(k)} - \log\hat s_2^{(k)},
\qquad \beta = \alpha/\hat s_2^{(k)}.
```

The target in the Newton step is
$\hat s_3^{(k)} - \log\hat s_2^{(k)}$. A finite gamma moment satisfies
$\hat s_2^{(k)} > 0$ and $\hat s_3^{(k)} < \log\hat s_2^{(k)}$.
The posterior used by the E-step is

```{math}
Y \mid X = x \sim \mathrm{GIG}\!\left(
\alpha - \tfrac{d}{2},\;
2\beta + \gamma^\top\Sigma^{-1}\gamma,\;
q(x)\right).
```

The case $q(x) = 0$ is discussed in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#gamma-mixing-variance-gamma).

### Normal-inverse Gaussian

$Y \sim \mathrm{IG}(\mu_{IG}, \lambda)$, the case $p = -1/2$
({ref}`BarndorffNielsen1997 <barndorffnielsen1997>`). The moments are

```{math}
E[Y] = \mu_{IG}, \qquad E[Y^{-1}] = 1/\mu_{IG} + 1/\lambda.
```

{meth}`~normix.distributions.inverse_gaussian.InverseGaussian.from_expectation`
receives $(\hat s_2^{(k)}, \hat s_1^{(k)})$ and returns

```{math}
\mu_{IG} = \hat s_2^{(k)}, \qquad
\lambda = \frac{1}{\hat s_1^{(k)} - 1/\hat s_2^{(k)}},
```

which requires $\hat s_2^{(k)} > 0$ and
$\hat s_1^{(k)} > 1/\hat s_2^{(k)}$.
The posterior is GIG of order $-1/2 - d/2$, with rates
$\lambda/\mu_{IG}^2 + \gamma^\top\Sigma^{-1}\gamma$ and $\lambda + q(x)$.
The closed form is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#inverse-gaussian-mixing-nig).

### Normal-inverse gamma

$Y \sim \mathrm{InvGamma}(\alpha, \beta)$ with rate $\beta$, the limit
$a \to 0$. The moments used by the update are

```{math}
E[\log Y] = \log\beta - \psi_0(\alpha), \qquad E[Y^{-1}] = \alpha/\beta.
```

{meth}`~normix.distributions.inverse_gamma.InverseGamma.from_expectation`
receives $(-\hat s_1^{(k)}, \hat s_3^{(k)})$ and solves

```{math}
\psi_0(\alpha) - \log\alpha = -\hat s_3^{(k)} - \log\hat s_1^{(k)},
\qquad \beta = \alpha/\hat s_1^{(k)}.
```

The target in the Newton step is
$-\hat s_3^{(k)} - \log\hat s_1^{(k)}$. A finite inverse-gamma moment
of this pair satisfies $\hat s_1^{(k)} > 0$ and
$\hat s_3^{(k)} > -\log\hat s_1^{(k)}$.
The posterior used by the E-step is

```{math}
Y \mid X = x \sim \mathrm{GIG}\!\left(
-\alpha - \tfrac{d}{2},\;
\gamma^\top\Sigma^{-1}\gamma,\;
2\beta + q(x)\right).
```

The scalar solve is in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#inverse-gamma-mixing).

| Marginal family | Mixing statistics | Mixing update |
|---|---|---|
| Generalized hyperbolic | $(\log Y,\, Y^{-1},\, Y)$ | {eq}`gig-loglik` in $(\hat s_3, \hat s_1, \hat s_2)$ |
| Variance-gamma | $(\log Y,\, Y)$ | digamma root, then $\beta = \alpha/\hat s_2$ |
| Normal-inverse Gaussian | $(Y,\, Y^{-1})$ | closed form for $(\mu_{IG}, \lambda)$ |
| Normal-inverse gamma | $(\log Y,\, Y^{-1})$ | digamma root, then $\beta = \alpha/\hat s_1$ |

## Numerical remarks

The conditioning of $\Sigma_{k+1}$ in {eq}`m-step` is the subject of
{doc}`shrinkage`. Overflow of the posterior Bessel ratio, and the
distinction between an exact $Q$ increase and a global maximum, are
discussed in the
[EM note](https://xshi19.github.io/math/normix-theory/normix-em-algorithm/#numerical-considerations-and-implementation).
Direct maximization of the observed likelihood is compared in
{doc}`../design/why_not_gradient_descent` and the
[gradient-descent note](https://xshi19.github.io/math/normix-theory/normix-why-not-gradient-descent/).
