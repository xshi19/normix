# The Generalized Inverse Gaussian Distribution

This page states the generalized inverse Gaussian (GIG) formulas that
normix computes. The moment generating function, the tail expansion, the
density in scale-concentration coordinates, and the Bessel form of the
Hellinger distance are derived in the
[GIG note](https://xshi19.github.io/math/normix-theory/normix-generalized-inverse-gaussian/).
A standard reference for the distribution is
{ref}`Jorgensen2012 <jorgensen2012>`.

normix writes a density as $f(x\mid\theta)$. On that site the same
function is $p_\theta(x)$.

**See also:** executable walkthrough in
{doc}`../tutorials/distributions/02_gig`.

## Parameters

normix stores $(p, a, b)$. The parameter domain is

```{math}
\begin{cases}
b > 0,\; a \ge 0 & \text{if } p < 0, \\
b > 0,\; a > 0 & \text{if } p = 0, \\
b \ge 0,\; a > 0 & \text{if } p > 0.
\end{cases}
```

The open interior $a > 0$, $b > 0$ is the Bessel case. The edges
$a = 0$ and $b = 0$ are the inverse-gamma and gamma laws below.

{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.from_natural`
rejects a positive rate coordinate. Zero rates are the boundary
embeddings below. The constructors are `__init__(p, a, b)`,
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.from_natural`,
and
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.from_expectation`.
They do not take a scale-concentration pair.

For $a > 0$ and $b > 0$, the scale and the concentration are
$\delta = \sqrt{b/a}$ and $\omega = \sqrt{ab}$, so $a = \omega/\delta$
and $b = \omega\delta$.
Here $\eta$ is reserved for the expectation parameter $\nabla\psi$.
The [GIG note](https://xshi19.github.io/math/normix-theory/normix-generalized-inverse-gaussian/)
uses the same $\delta$ and $\omega$.

## Density

```{math}
:label: gig-pdf

f(x\mid p,a,b) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})} x^{p-1}
\exp\left(-\frac{1}{2}(b x^{-1} + a x)\right), \quad x > 0,
```

where $K_p$ is the modified Bessel function of the second kind
({ref}`DLMF <dlmf>` §10.32). On a boundary, read {eq}`gig-pdf` as the
limit named above. Do not substitute $a = 0$ or $b = 0$ into the
Bessel ratio.

## Moments

For $a > 0$ and $b > 0$,

```{math}
:label: gig-moments

E[X^r] = \left(\sqrt{\frac{b}{a}}\right)^r
\frac{K_{p+r}(\sqrt{ab})}{K_p(\sqrt{ab})}.
```

The ratio is finite for every real $r$, so every moment exists.
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.mean`
and
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.var`
use this formula, with $\mathrm{Var}[X] = E[X^2] - E[X]^2$.
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.raw_moment`
returns the same ratio and requires $a, b > 0$.

At a boundary the moment domain changes.

- Gamma, $p > 0$ and $b \to 0$: $E[X^r] < \infty$ if and only if
  $r > -p$.
- Inverse gamma, $p < 0$ and $a \to 0$: $E[X^r] < \infty$ if and only
  if $r < -p$.

The tail expansion that produces these domains is in the GIG note.

## Exponential family

```{math}
f(x\mid\theta) = h(x) \exp\left(\theta^\top t(x) - \psi(\theta)\right)
```

```{math}
t(x) = \begin{pmatrix} \log x \\ x^{-1} \\ x \end{pmatrix},
\qquad
h(x) = \mathbf{1}_{x > 0}.
```

```{math}
:label: gig-natural-params

\theta_1 &= p - 1, \\
\theta_2 &= -\frac{b}{2} \le 0, \\
\theta_3 &= -\frac{a}{2} \le 0.
```

Strict inequality is the interior. The inverse map is
$p = \theta_1 + 1$, $b = -2\theta_2$, $a = -2\theta_3$.

```{math}
:label: gig-log-partition

\psi(\theta) = \log 2 + \log K_p(\sqrt{ab}) + \frac{p}{2} \log\left(\frac{b}{a}\right),
```

with $p$, $a$, and $b$ as above. This display is the interior formula.
On a gamma or inverse-gamma boundary, $\psi$ is the log-partition of that
boundary law. The expectation parameter
$\eta = \nabla\psi(\theta) = E[t(X)]$ is

```{math}
:label: gig-expectation-params

\eta_1 &= E[\log X] = \frac{\partial \psi}{\partial \theta_1}
= \frac12\log\frac{b}{a}
+ \left.\partial_\nu \log K_\nu(\sqrt{ab})\right|_{\nu=p}, \\
\eta_2 &= E[X^{-1}] = \sqrt{\frac{a}{b}} \frac{K_{p-1}(\sqrt{ab})}{K_p(\sqrt{ab})}, \\
\eta_3 &= E[X] = \sqrt{\frac{b}{a}} \frac{K_{p+1}(\sqrt{ab})}{K_p(\sqrt{ab})}.
```

The order derivative is not an elementary expression. Its equality with
$\partial\psi/\partial\theta_1$ is derived in the GIG note.
{meth}`~normix.exponential_family.ExponentialFamily.expectation_params`
returns $(\eta_1, \eta_2, \eta_3)$ in that order.

## Likelihood

Given $(\eta_1, \eta_2, \eta_3)$, the classical parameters solve
{eq}`gig-expectation-params`. For a sample $x_1,\ldots,x_n$ the maximum
likelihood estimator is

```{math}
:label: gig-mle

(\hat{p}, \hat{a}, \hat{b}) = \arg\max_{p,a,b} L_{GIG}(p, a, b | \hat{\eta}_1, \hat{\eta}_2, \hat{\eta}_3),
```

where the log-likelihood, up to an additive constant, is

```{math}
:label: gig-loglik

L_{GIG}(p, a, b | \hat{\eta}_1, \hat{\eta}_2, \hat{\eta}_3) =
(p-1) \hat{\eta}_1 - \frac{1}{2} b \hat{\eta}_2 - \frac{1}{2} a \hat{\eta}_3
+ \frac{p}{2} \log(a/b) - \log(K_p(\sqrt{ab})).
```

The arguments stay in the order $(\log X, X^{-1}, X)$:

- $\hat{\eta}_1 = \frac{1}{n} \sum_{k=1}^n \log(x_k)$
- $\hat{\eta}_2 = \frac{1}{n} \sum_{k=1}^n x_k^{-1}$
- $\hat{\eta}_3 = \frac{1}{n} \sum_{k=1}^n x_k$

An interior critical point satisfies {eq}`gig-expectation-params` at the
empirical moments. There is no closed form for $\hat{p}$
({ref}`Jorgensen2012 <jorgensen2012>`).
{ref}`Hu2005 <hu2005>` reports that the moment equations for $\eta_2$ and
$\eta_3$ can fail to have a solution when $|p|$ is large.
{meth}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian.from_expectation`
is the numerical inversion.

## Hellinger distance

For two GIG laws with natural parameters $\theta$ and $\varphi$, the
squared Hellinger distance that
{meth}`~normix.exponential_family.ExponentialFamily.squared_hellinger`
evaluates is

```{math}
H^2 = 1 - \exp\left(
\psi\left(\frac{\theta+\varphi}{2}\right)
- \frac{\psi(\theta)+\psi(\varphi)}{2}
\right).
```

The Bessel-ratio form of the same affinity is derived in the GIG note
({ref}`Shi2016 <shi2016>`).

## Boundary laws

- Inverse Gaussian: $p = -1/2$, stored as
  {class}`~normix.distributions.inverse_gaussian.InverseGaussian`.
  {meth}`~normix.distributions.inverse_gaussian.InverseGaussian.to_gig`
  maps $(\mu, \lambda)$ to $\mathrm{GIG}(-1/2,\, \lambda/\mu^2,\, \lambda)$.
- Gamma: $p > 0$ and $b \to 0$, giving $\mathrm{Gamma}(p, a/2)$, stored as
  {class}`~normix.distributions.gamma.Gamma`.
  {meth}`~normix.distributions.gamma.Gamma.to_gig`
  maps $(\alpha, \beta)$ to $\mathrm{GIG}(\alpha,\, 2\beta,\, 0)$.
- Inverse gamma: $p < 0$ and $a \to 0$, giving $\mathrm{InvGamma}(-p, b/2)$,
  stored as
  {class}`~normix.distributions.inverse_gamma.InverseGamma`.
  {meth}`~normix.distributions.inverse_gamma.InverseGamma.to_gig`
  maps $(\alpha, \beta)$ to $\mathrm{GIG}(-\alpha,\, 0,\, 2\beta)$.
