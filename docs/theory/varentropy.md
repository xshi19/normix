# Entropy, Varentropy, and Rényi Entropy

This page states the entropy, varentropy, and Rényi formulas that
normix computes for exponential families and for joint normal
variance-mean mixtures. The escort differentiation, the
$\chi^2$ decomposition, and the general base-measure covariance formula
are derived in the
[varentropy note](https://xshi19.github.io/math/normix-theory/normix-varentropy/).
For the structural properties of varentropy see
{ref}`Stankyavichyus2026 <stankyavichyus2026>`.

normix writes a density as $f(x\mid\theta)$. On that site the same
function is $p_\theta(x)$.

The surprisal is $\mathcal{I}(X) = -\log f(X\mid\theta)$. The Fisher
information is $I(\theta) = \nabla^2\psi(\theta)$. These are different
objects: $\mathcal{I}(X)$ is a random variable, and $I(\theta)$ is a
matrix.

**See also:** Monte Carlo checks in
{doc}`../tutorials/stats/03_varentropy`.

## Definitions

Let $X$ have density $f$ with respect to a reference measure $\mu(dx)$.

```{math}
:label: ve-defs

H = \mathbb{E}[-\log f(X)], \qquad
V_H = \operatorname{Var}[-\log f(X)].
```

$H$ is the differential entropy and $V_H$ is the varentropy. The
varentropy needs only $\mathcal{I}(X) \in L^2$, so it stays finite for
laws whose fourth moment diverges.

The Rényi entropy of order $\alpha > 0$, $\alpha \ne 1$
({ref}`Renyi1961 <renyi1961>`), is

```{math}
:label: ve-renyi

H_\alpha = \frac{1}{1-\alpha} \log \int f(x)^\alpha \, \mu(dx),
```

and $H_\alpha \to H$ as $\alpha \to 1$.

## Density power

```{math}
:label: ve-R

R(\alpha) = \log \int f(x)^\alpha \, \mu(dx), \qquad R(1) = 0.
```

After the reflection $\alpha \mapsto 1-\alpha$, $R$ is the cumulant-generating function of the surprisal, so

```{math}
:label: ve-cumulants

H = -R'(1), \qquad V_H = R''(1),
```

and $H_\alpha = R(\alpha)/(1-\alpha)$. The expansion

```{math}
H_\alpha = H - \tfrac{1}{2} V_H\,(\alpha - 1) + \mathcal{O}\!\left((\alpha-1)^2\right)
```

is the slope of the Rényi spectrum at $\alpha = 1$. The
cumulant-generator identification is in the varentropy note.
{meth}`~normix.exponential_family.ExponentialFamily.entropy`,
{meth}`~normix.exponential_family.ExponentialFamily.varentropy`, and
{meth}`~normix.exponential_family.ExponentialFamily.renyi`
differentiate
{meth}`~normix.exponential_family.ExponentialFamily.log_density_power`.

## Exponential family

For

```{math}
f(x\mid\theta) = h(x)\exp\{\theta^\top t(x) - \psi(\theta)\}
```

with $\log h \equiv b_0$ on the support,

```{math}
:label: ve-R-ef

R(\alpha) = (\alpha - 1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta),
```

when $\alpha\theta$ lies in the natural domain. Otherwise
$R(\alpha) = +\infty$, and $H_\alpha = R(\alpha)/(1-\alpha)$ inherits
the sign of $1-\alpha$. For a gamma law of shape $s$ the condition is
$\alpha > 0$ and $\alpha(s-1)+1 > 0$. For an inverse-gamma law of
shape $s$ it is $\alpha(s+1)-1 > 0$, which already forces $\alpha > 0$.

With $\eta = \nabla\psi(\theta)$,

```{math}
:label: ve-ef-formulas

H = \psi(\theta) - \theta^\top \eta - b_0, \qquad
V_H = \theta^\top I(\theta)\,\theta, \qquad
H_\alpha = \frac{(\alpha-1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta)}{1-\alpha}.
```

The quadratic form holds because
$\mathcal{I}(X) - H = -\theta^\top\{t(X) - \eta\}$. This covers
{class}`~normix.distributions.gamma.Gamma`,
{class}`~normix.distributions.inverse_gamma.InverseGamma`, and
{class}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian`
($b_0 = 0$), and
{class}`~normix.distributions.normal.MultivariateNormal`
($\log h \equiv 0$, $V_H = d/2$).

```{note}
{class}`~normix.distributions.inverse_gaussian.InverseGaussian`
has $\log h(x) = -\tfrac12\log(2\pi) - \tfrac32\log x$, which is not
constant. The general covariance correction is in the varentropy note.
normix evaluates the inverse-Gaussian quantities on the embedding
$\mathrm{GIG}(-\tfrac12, \lambda/\mu^2, \lambda)$, where $\log h \equiv 0$
and $\log x$ is a sufficient statistic.
```

## GIG

Let $Y \sim \mathrm{GIG}(p, a, b)$ with density {eq}`gig-pdf` and
$\theta = (p-1,\,-b/2,\,-a/2)$. Substituting the GIG log-partition into
{eq}`ve-R-ef` gives, with $z = \sqrt{ab}$,

```{math}
:label: ve-gig-R

R(\alpha) = (1-\alpha)\log 2 + \frac{1-\alpha}{2}\log\!\frac{b}{a}
+ \log K_{1 + \alpha(p-1)}(\alpha z) - \alpha \log K_p(z).
```

On the math site $\sqrt{ab}$ is the concentration $\omega$, and the
operator below is written $\mathcal{L}$. Here $z = \omega$ and

```{math}
L = (p-1)\,\partial_p + z\,\partial_z.
```

Differentiating along the escort path produces

```{math}
:label: ve-gig

\boxed{\;
V_H\{\mathrm{GIG}(p,a,b)\} = (L^2 - L)\log K_p(z),
\qquad z = \sqrt{ab}.
\;}
```

Expanded, this is

```{math}
V_H = (p-1)^2 \partial_p^2 \log K_p(z)
+ 2(p-1)\,z\,\partial_p\partial_z \log K_p(z)
+ z^2 \partial_z^2 \log K_p(z),
```

which equals $\theta^\top I(\theta)\,\theta$. The entropy is

```{math}
:label: ve-gig-entropy

H\{\mathrm{GIG}(p,a,b)\} = \log\{2 K_p(z)\} + \tfrac12\log\!\frac{b}{a}
- L\log K_p(z).
```

For $a, b > 0$ every moment of $Y$ is finite, and $V_H$ stays finite,
including in the inverse-gamma limit $a \to 0$ where the kurtosis
diverges. The tail comparison is in the
[GIG note](https://xshi19.github.io/math/normix-theory/normix-generalized-inverse-gaussian/).

## Joint law

For the joint law {eq}`gh-joint`, conditionally on $Y$, the quadratic
form in the Gaussian exponent is $\chi^2_d$ and independent of $Y$.
That decomposition is in the varentropy note. It gives

```{math}
:label: ve-joint

\boxed{\;
V_H(X, Y) = \frac{d}{2}
+ \operatorname{Var}\!\left[\mathcal{I}_Y + \frac{d}{2}\log Y\right],
\;}
```

where $\mathcal{I}_Y = -\log f_Y(Y)$ is the subordinator surprisal.
The Gaussian layer contributes $d/2$. The parameters $\mu$, $\gamma$,
and $\Sigma$ change the joint entropy only through constants and
means. They drop out of $V_H$ except through the dimension $d$.

For a GIG subordinator, $t(Y) = (\log Y,\, Y^{-1},\, Y)$ and
$\mathcal{I}_Y = \psi_{\mathrm{GIG}}(\theta) - \theta^\top t(Y)$, so

```{math}
\mathcal{I}_Y + \tfrac{d}{2}\log Y
= \psi_{\mathrm{GIG}}(\theta)
- \left(\theta - \left(\tfrac{d}{2},\, 0,\, 0\right)^\top\right)^\top t(Y).
```

The variance of this shifted statistic is

```{math}
:label: ve-joint-gig

\boxed{\;
V_H(X, Y) = \frac{d}{2} + (L_d^2 - L_d)\log K_p(z),
\qquad
L_d = \left(p - 1 - \tfrac{d}{2}\right)\partial_p + z\,\partial_z,
\qquad z = \sqrt{ab}.
\;}
```

For $d = 0$ this is {eq}`ve-gig`. The shift
$p - 1 \mapsto p - 1 - d/2$ is the natural coordinate $\theta_1$ in
{eq}`gh-natural-params`. Variance-gamma ($b \to 0$) and normal-inverse
gamma ($a \to 0$) evaluate $R(\alpha)$ from their own subordinator
log-partitions. The normal-inverse Gaussian joint fixes $p = -1/2$ in
its log-partition, so the density-power integral is evaluated on the
GH embedding, where $p$ moves with $\alpha$.
