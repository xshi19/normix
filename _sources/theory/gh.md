# The Generalized Hyperbolic Distribution

This page states the generalized hyperbolic (GH) formulas that normix
computes. The marginal integral, the scale orbit, the natural-parameter
expansion, and the Bessel form of the joint Hellinger distance are
derived in the
[GH note](https://xshi19.github.io/math/normix-theory/normix-generalized-hyperbolic/).
The family was introduced by
{ref}`BarndorffNielsen1977 <barndorffnielsen1977>`.

normix writes a density as $f(x\mid\theta)$. On that site the same
function is $p_\theta(x)$.

## Definition

Let $Y$ be GIG with parameters $(p, a, b)$, and let $Z$ be an
independent Gaussian vector with mean zero and covariance $\Sigma$. Then

```{math}
:label: gh-def

X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z
```

is generalized hyperbolic with parameters $(\mu, \gamma, \Sigma, p, a, b)$.
Here $\mu, \gamma \in \mathbb{R}^d$ and $\Sigma$ is positive definite.
$\Sigma$ is the covariance of $Z$. The constructor
{meth}`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic.from_classical`
takes $(\mu, \gamma, \Sigma, p, a, b)$ and stores the Cholesky factor of
$\Sigma$.

- $\mu$: location
- $\gamma$: skewness
- $\Sigma$: dispersion
- $(p, a, b)$: GIG parameters of $Y$

Any law of the form {eq}`gh-def`, for a nonnegative $Y$, is a **normal
mixture** in normix.

```{note}
In many references, "normal mixture" means a discrete mixture of normal
densities. In normix it means a random vector that can be written as
{eq}`gh-def`.
```

## Joint density

The joint law of $(X, Y)$ is the joint GH distribution:

```{math}
:label: gh-joint

f(x, y \mid \mu, \gamma, \Sigma, p, a, b) &= \frac{1}{\sqrt{(2\pi)^d |\Sigma|}}
\frac{(a/b)^{p/2}}{2 K_p(\sqrt{ab})} y^{p - 1 - d/2} \\
&\quad \times \exp\left( -\frac{1}{2}(x - \mu - \gamma y)^\top \Sigma^{-1}
(x - \mu - \gamma y) y^{-1} - \frac{1}{2}(b y^{-1} + a y) \right),
```

for $y > 0$.

## Marginal density

Integrating out $y$ gives the marginal density. The calculation is in
the GH note. The density that
{meth}`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic.log_prob`
evaluates is

```{math}
:label: gh-marginal

\begin{aligned}
f(x \mid \mu, \gamma, \Sigma, p, a, b)
&= (2\pi)^{-d/2}\,|\Sigma|^{-1/2}\,
\frac{(a/b)^{p/2}}{K_p(\sqrt{ab})}
\left(\frac{A}{Q+b}\right)^{(d/2-p)/2} \\
&\quad \times
K_{p-d/2}\!\left(\sqrt{A(Q+b)}\right)
\exp\!\left(\gamma^\top\Sigma^{-1}(x-\mu)\right).
\end{aligned}
```

where $Q = (x-\mu)^\top\Sigma^{-1}(x-\mu)$ and
$A = a + \gamma^\top\Sigma^{-1}\gamma$.

The conditional law used by the E-step is

```{math}
Y \mid X = x \sim \mathrm{GIG}\!\left(p - \tfrac{d}{2},\;
a + \gamma^\top\Sigma^{-1}\gamma,\;
b + Q\right).
```

## Parameters and scale

For every $c > 0$ the parameter sets

```{math}
(\mu,\; \gamma,\; \Sigma,\; p,\; a,\; b)
\quad\text{and}\quad
(\mu,\; \gamma/c,\; \Sigma/c,\; p,\; a/c,\; bc)
```

give the same marginal law of $X$. The product $ab$ is invariant, so
the concentration $\omega = \sqrt{ab}$ is invariant, while the scale
$\delta = \sqrt{b/a}$ is multiplied by $c$. The Fisher information of
the unrestricted parameter is singular. The derivation of the orbit is
in the GH note.

Two representatives are implemented:

- {meth}`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic.regularize_a_eq_b`
  sets $a = b = \omega$.
- {meth}`~normix.mixtures.marginal.NormalMixture.regularize_det_sigma_one`
  sets $|\Sigma| = 1$, the constraint in
  {ref}`McNeil2010 <mcneil2010>`.
  {meth}`~normix.mixtures.marginal.NormalMixture.regularize_det_sigma`
  sets $\log|\Sigma|$ to a chosen target.

Since $|\Sigma/c| = |\Sigma|/c^d$, a small change in the matrix scale
moves $|\Sigma|$ sharply when $d$ is large. Fixing $|\Sigma| = 1$ does
not control the condition number of $\Sigma$. The mechanisms are
described in {doc}`../design/em_framework`. Other normalizations used
in the literature are in the GH note.

## Moments

```{math}
:label: gh-mean-cov

E[X] &= \mu + \gamma\,E[Y], \\
\mathrm{Cov}[X] &= E[Y]\,\Sigma + \mathrm{Var}[Y]\,\gamma\gamma^\top.
```

These identities use the moments of $Y$ that they display. When
$\gamma_i = 0$, the drift terms vanish, so $E[X_i] = \mu_i$ as soon as
$E[\sqrt{Y}] < \infty$, and $\mathrm{Cov}[X] = E[Y]\,\Sigma$ needs only
$E[Y] < \infty$. For an inverse-gamma subordinator (the Student-$t$
case $\nu = 2\alpha$) that is $\alpha > 1/2$ for the mean and
$\alpha > 1$ for the covariance. When $\gamma_i \ne 0$ the mean needs
$E[Y]$ and the covariance needs $\mathrm{Var}[Y]$. Cauchy
($\alpha = 1/2$, $\gamma = 0$) has no mean: the integral is $+\infty$,
not $\mu$. A gamma subordinator has every positive moment, so
variance-gamma does not inherit the inverse-gamma table. A zero
coordinate of $\gamma$ is not multiplied by an infinite moment of $Y$.

Conditioning on $Y$ gives the third and fourth central moments. The
expansion is in the GH note. For coordinate $i$, with
$m_k = E[Y^k]$ from {eq}`gig-moments` or from the boundary law,

```{math}
:label: gh-skew-kurt

\mu_3(X_i) &= \gamma_i^3\,\mu_3(Y) + 3\,\gamma_i\,\Sigma_{ii}\,\mathrm{Var}[Y], \\
\mu_4(X_i) &= \gamma_i^4\,\mu_4(Y)
  + 6\,\gamma_i^2\,\Sigma_{ii}\,E[(Y-m_1)^2 Y]
  + 3\,\Sigma_{ii}^2\,E[Y^2].
```

Skewness is $\mu_3/\sigma^3$ and excess kurtosis is
$\mu_4/\sigma^4 - 3$. When $\gamma = 0$ the skewness is $0$ and the
excess kurtosis is $3\,E[Y^2]/E[Y]^2 - 3$, which equals $3/\alpha$ for
variance-gamma and $3/(\alpha-2)$ for normal-inverse gamma
($\alpha > 2$). The $t_3$ law (normal-inverse gamma, $\gamma = 0$,
$\alpha = 1.5$) has no fourth moment, so the excess kurtosis is
$+\infty$. When $\gamma \ne 0$, the term $\gamma_i^4\mu_4(Y)$ requires
$E[Y^4] < \infty$ (inverse-gamma shape $\alpha > 4$).

{meth}`~normix.mixtures.marginal.NormalMixture.skewness` and
{meth}`~normix.mixtures.marginal.NormalMixture.kurtosis` return these
values, component-wise for $d > 1$.

## Exponential family

The joint law {eq}`gh-joint` is an exponential family

```{math}
f(x, y\mid\theta) = h(x, y) \exp\left(\theta^\top t(x, y) - \psi(\theta)\right).
```

Expanding the joint exponent produces the coordinates below. The
expansion is in the GH note.

```{math}
t(x, y) = \begin{pmatrix} \log y \\ y^{-1} \\ y \\ x \\ x y^{-1} \\ x x^\top y^{-1} \end{pmatrix},
\qquad
h(x, y) = (2\pi)^{-d/2}\,\mathbf{1}_{y > 0}.
```

The blocks have sizes $1$, $1$, $1$, $d$, $d$, and $d \times d$. In the
parameter array the last block is stored as a vector of length $d^2$:
the coefficient of $\mathrm{vec}(xx^\top y^{-1})$ is
$-\tfrac12\mathrm{vec}(\Sigma^{-1})$.

```{math}
:label: gh-natural-params

\theta_1 &= p - 1 - \frac{d}{2}, \\
\theta_2 &= -\frac{1}{2}\left(b + \mu^\top \Sigma^{-1} \mu\right), \\
\theta_3 &= -\frac{1}{2}\left(a + \gamma^\top \Sigma^{-1} \gamma\right), \\
\theta_4 &= \Sigma^{-1} \gamma, \\
\theta_5 &= \Sigma^{-1} \mu, \\
\theta_6 &= -\frac{1}{2} \Sigma^{-1}.
```

```{math}
:label: gh-log-partition

\psi(\theta) = \frac{1}{2} \log|\Sigma| + \log 2 + \log K_p(\sqrt{ab})
+ \frac{p}{2} \log\left(\frac{b}{a}\right) + \mu^\top \Sigma^{-1} \gamma.
```

```{math}
:label: gh-expectation

\eta_1 &:= E[\log Y] = \left.\frac{\partial}{\partial r}
\left(\sqrt{\frac{b}{a}}\right)^r \frac{K_{p+r}(\sqrt{ab})}{K_p(\sqrt{ab})}
\right|_{r=0}, \\
\eta_2 &:= E[Y^{-1}] = \sqrt{\frac{a}{b}} \frac{K_{p-1}(\sqrt{ab})}{K_p(\sqrt{ab})}, \\
\eta_3 &:= E[Y] = \sqrt{\frac{b}{a}} \frac{K_{p+1}(\sqrt{ab})}{K_p(\sqrt{ab})}, \\
\eta_4 &:= E[X] = \mu + \gamma \eta_3, \\
\eta_5 &:= E[X Y^{-1}] = \mu \eta_2 + \gamma, \\
\eta_6 &:= E[X X^\top Y^{-1}] = \Sigma + \mu \mu^\top \eta_2 + \gamma \gamma^\top \eta_3
+ \mu \gamma^\top + \gamma \mu^\top.
```

Here $\eta = \nabla\psi(\theta) = E[t(X, Y)]$. The first three coordinates
are the GIG expectation parameters of $Y$. On a variance-gamma,
normal-inverse-gamma, or normal-inverse-Gaussian joint, the same block
is returned with the gamma, inverse-gamma, or inverse-Gaussian moments
of $Y$. Those entries are not the partial derivatives of $\psi$ after the
frozen coordinate ($b = 0$, $a = 0$, or $p = -1/2$) has been removed.
Where a moment of $Y$ does not exist, the entry is $+\infty$.

## M-step

```{math}
:label: gh-m-step

\mu &= \frac{\eta_4 - \eta_3 \eta_5}{1 - \eta_2 \eta_3}, \\
\gamma &= \frac{\eta_5 - \eta_2 \eta_4}{1 - \eta_2 \eta_3}, \\
\Sigma &= \eta_6 - \eta_5 \mu^\top - \mu \eta_5^\top + \eta_2 \mu \mu^\top - \eta_3 \gamma \gamma^\top, \\
(p, a, b) &= \arg\max_{p,a,b} L_{\text{GIG}}(p, a, b \mid \eta_1, \eta_2, \eta_3),
```

where $L_{\text{GIG}}$ is {eq}`gig-loglik`. These are the updates
{meth}`~normix.mixtures.marginal.MarginalMixture.m_step` applies, with
the expectations in {eq}`gh-expectation` replaced by conditional
expectations. The derivation is in the GH note. The EM write-up is
{doc}`em_algorithm`.

## Hellinger distance

There is no closed form asserted here for the Hellinger distance
between two marginal GH laws. For two joint laws with natural
parameters $\theta$ and $\varphi$,
{meth}`~normix.exponential_family.ExponentialFamily.squared_hellinger`
evaluates

```{math}
H_{\mathrm{JGH}}^2 = 1 - \exp\left(
\psi\left(\frac{\theta+\varphi}{2}\right)
- \frac{\psi(\theta)+\psi(\varphi)}{2}
\right).
```

The average is in natural coordinates. When the normal parameters
agree, this reduces to the GIG Hellinger distance. Marginalization
gives $H_X^2 \le H_{\mathrm{JGH}}^2$. The marginal method returns the
joint distance, which is an upper bound on the marginal distance. The
Bessel-ratio expansion and the comparison are derived in the GH note
({ref}`Shi2016 <shi2016>`).

## Special cases

- Normal-inverse Gaussian ({ref}`BarndorffNielsen1997 <barndorffnielsen1997>`):
  $p = -1/2$, class
  {class}`~normix.distributions.normal_inverse_gaussian.NormalInverseGaussian`.
- Variance-gamma ({ref}`MadanSeneta1990 <madan1990>`): $b \to 0$, class
  {class}`~normix.distributions.variance_gamma.VarianceGamma`.
- Normal-inverse gamma: $a \to 0$, class
  {class}`~normix.distributions.normal_inverse_gamma.NormalInverseGamma`.
- Student-$t$: $p < 0$, $a = 0$, $\gamma = 0$.
- Hyperbolic: order $p = 1$. This is not a separate class.
