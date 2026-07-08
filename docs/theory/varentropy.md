# Entropy, Varentropy, and Rényi Entropy

This section derives the key information-theoretic quantities implemented in
``normix`` — the differential entropy, the **varentropy**, and the Rényi
entropy — for the exponential-family components and for the joint normal
variance-mean mixtures. We derive only the formulas used by the code; for the
general structural decomposition of varentropy and its properties we refer to
[Stankyavichyus2026](#stankyavichyus2026).

**See also:** Monte Carlo validation and fat-tail comparisons in
{doc}`../tutorials/stats/03_varentropy`.

## Definitions

Let $X$ have density $p(x)$ with respect to a reference measure
$\mu(dx)$. The **information content** (surprisal) is
$\mathcal{I}(X) = -\log p(X)$. Its mean is the **differential entropy**
and its variance is the **varentropy**:

```{math}
:label: ve-defs

H = \mathbb{E}[-\log p(X)], \qquad
V_H = \operatorname{Var}[-\log p(X)].
```

Entropy measures the average surprisal; varentropy measures how much the
surprisal fluctuates around that average. Unlike the kurtosis, the varentropy
requires only $\mathcal{I}(X) \in L^2$, so it stays finite for many
heavy-tailed laws whose fourth moment diverges.

The **Rényi entropy** of order $\alpha > 0$, $\alpha \neq 1$, is

```{math}
:label: ve-renyi

H_\alpha = \frac{1}{1-\alpha} \log \int p(x)^\alpha \, \mu(dx),
```

and $H_\alpha \to H$ as $\alpha \to 1$.

## Density-Power Route

All three quantities are governed by the **log density-power integral**

```{math}
:label: ve-R

R(\alpha) = \log \int p(x)^\alpha \, \mu(dx), \qquad R(1) = 0.
```

Writing $\mathcal{I} = -\log p(X)$, the function $R(1-s)$ is the
cumulant-generating function of $\mathcal{I}$, so differentiating at
$\alpha = 1$ gives

```{math}
:label: ve-cumulants

H = -R'(1), \qquad V_H = R''(1),
```

while the Rényi entropy is $H_\alpha = R(\alpha)/(1-\alpha)$. The
first-order expansion

```{math}
H_\alpha = H - \tfrac{1}{2} V_H\,(\alpha - 1) + \mathcal{O}\!\left((\alpha-1)^2\right)
```

shows that the varentropy is (twice the negative of) the slope of the Rényi
spectrum at $\alpha = 1$.

## Exponential Family

For a ``normix`` exponential family
$p(x\mid\theta) = h(x)\exp\{\theta^\top t(x) - \psi(\theta)\}$
whose base measure is **constant on its support**, $\log h(x) \equiv b_0$,
the density power stays in the family with natural parameter
$\alpha\theta$:

```{math}
:label: ve-R-ef

R(\alpha) = (\alpha - 1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta).
```

Substituting into {eq}`ve-cumulants` and using $\eta = \nabla\psi(\theta)$
and the Fisher information $I(\theta) = \nabla^2\psi(\theta)$ gives closed
forms in terms of the log-partition triad:

```{math}
:label: ve-ef-formulas

H = \psi(\theta) - \theta^\top \eta - b_0, \qquad
V_H = \theta^\top I(\theta)\,\theta, \qquad
H_\alpha = \frac{(\alpha-1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta)}{1-\alpha}.
```

The varentropy identity $V_H = \theta^\top I(\theta)\,\theta$ holds because
the centered information content
$\mathcal{I}(X) - H = -\theta^\top\{t(X) - \eta\}$ lies exactly in the
span of the score. This covers {class}`~normix.distributions.gamma.Gamma`,
{class}`~normix.distributions.inverse_gamma.InverseGamma`,
{class}`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian`
(all with $b_0 = 0$) and
{class}`~normix.distributions.normal.MultivariateNormal` (with
$\log h \equiv 0$ and $V_H = d/2$).

```{note}

When $\log h(x)$ is **not** constant on the support — as for
{class}`~normix.distributions.inverse_gaussian.InverseGaussian`, whose
$\log h(x) = -\tfrac12\log(2\pi) - \tfrac32\log x$ — formula
{eq}`ve-ef-formulas` acquires base-measure covariance terms (see
[Stankyavichyus2026](#stankyavichyus2026)). ``normix`` avoids them by evaluating the quantities
on the exact GIG embedding $\mathrm{GIG}(-\tfrac12, \lambda/\mu^2,
\lambda)$, where $\log h \equiv 0$ and the $\log x$ term becomes
a sufficient statistic.
```

## Varentropy of the GIG Distribution

Let $Y \sim \mathrm{GIG}(p, a, b)$ with density {eq}`gig-pdf` and natural
parameters $\theta = [p-1,\,-b/2,\,-a/2]$. Raising the density to the
power $\alpha$ keeps it proportional to a GIG density with parameters
$(1 + \alpha(p-1),\, \alpha a,\, \alpha b)$ — the escort-closure property.
Using the Bessel integral
$\int_0^\infty y^{q-1} e^{-(uy + v/y)/2}\,dy = 2\,(v/u)^{q/2} K_q(\sqrt{uv})$
gives, with $z = \sqrt{ab}$,

```{math}
:label: ve-gig-R

R(\alpha) = (1-\alpha)\log 2 + \frac{1-\alpha}{2}\log\!\frac{b}{a}
+ \log K_{1 + \alpha(p-1)}(\alpha z) - \alpha \log K_p(z).
```

Define $F(p, z) = \log K_p(z)$ and the first-order differential operator

```{math}
:label: ve-L

L = (p-1)\,\partial_p + z\,\partial_z.
```

Along the escort path $\alpha \mapsto (1 + \alpha(p-1),\, \alpha z)$ we have
$\frac{d}{d\alpha}F = LF$ and, because $L$ has variable coefficients,
$\frac{d^2}{d\alpha^2}F = (L^2 - L)F$ at $\alpha = 1$. All remaining
terms of {eq}`ve-gig-R` are linear in $\alpha$, so from
{eq}`ve-cumulants`,

```{math}
:label: ve-gig

\boxed{\;
V_H\{\mathrm{GIG}(p,a,b)\} = (L^2 - L)\log K_p(z),
\qquad z = \sqrt{ab}.
\;}
```

Expanded, this is the pure second-order form

```{math}
V_H = (p-1)^2 F_{pp} + 2(p-1)\,z\,F_{pz} + z^2 F_{zz},
```

which coincides with the Fisher quadratic form
$\theta^\top I(\theta)\,\theta$ of {eq}`ve-ef-formulas`. The entropy
follows from $H = -R'(1)$:

```{math}
:label: ve-gig-entropy

H\{\mathrm{GIG}(p,a,b)\} = \log\{2 K_p(z)\} + \tfrac12\log\!\frac{b}{a}
- L\log K_p(z).
```

Because the fixed argument of $K_p$ is exponentially thin in both tails
whenever $a, b > 0$ (see {doc}`gig`), $V_H$ remains finite over the
entire parameter range, including the inverse-gamma limit $a \to 0$ where
the kurtosis diverges.

## Joint Varentropy of Normal Variance-Mean Mixtures

Consider the joint law of {eq}`gh-joint` for a general subordinator,

```{math}
X \mid Y = y \sim \mathcal{N}_d(\mu + \gamma y,\ \Sigma y), \qquad
Y \sim g_\vartheta.
```

Conditionally on $Y$, the quadratic form
$Q = (X - \mu - \gamma Y)^\top \Sigma^{-1}(X - \mu - \gamma Y)/Y$ is
$\chi^2_d$ and independent of $Y$, so

```{math}
-\log p(X \mid Y) = C_\Sigma + \tfrac{d}{2}\log Y + \tfrac12 Q, \qquad
C_\Sigma = \tfrac{d}{2}\log(2\pi) + \tfrac12\log|\Sigma|.
```

Writing $\mathcal{I}_Y = -\log g_\vartheta(Y)$ for the subordinator
surprisal, the joint information content is
$\mathcal{I}_{X,Y} = C_\Sigma + \mathcal{I}_Y + \tfrac{d}{2}\log Y +
\tfrac12 Q$. Since $\operatorname{Var}(\tfrac12 Q) = d/2$ and
$Q \perp Y$,

```{math}
:label: ve-joint

\boxed{\;
V_H(X, Y) = \frac{d}{2}
+ \operatorname{Var}\!\left[\mathcal{I}_Y + \frac{d}{2}\log Y\right].
\;}
```

The Gaussian layer contributes exactly $d/2$; the parameters
$\mu, \gamma, \Sigma$ enter the joint entropy only through constants and
means, and drop out of the varentropy entirely (they affect it only through the
dimension $d$).

For a GIG subordinator, $\mathcal{I}_Y = \psi_{\mathrm{GIG}}(\theta) -
\theta^\top t(Y)$ with $t(Y) = [\log Y,\, Y^{-1},\, Y]$, so adding
$\tfrac{d}{2}\log Y$ shifts only the coefficient of $\log Y$:

```{math}
\mathcal{I}_Y + \tfrac{d}{2}\log Y
= \psi_{\mathrm{GIG}}(\theta) - (\theta - \delta_d)^\top t(Y),
\qquad \delta_d = (\tfrac{d}{2},\, 0,\, 0)^\top.
```

Its variance is the shifted Fisher quadratic form, and translating to classical
coordinates gives the operator $L_d$ of {eq}`ve-L` with the order
shifted by $-d/2$:

```{math}
:label: ve-joint-gig

\boxed{\;
V_H(X, Y) = \frac{d}{2} + (L_d^2 - L_d)\log K_p(z),
\qquad
L_d = \left(p - 1 - \tfrac{d}{2}\right)\partial_p + z\,\partial_z.
\;}
```

For $d = 0$ this reduces to the GIG varentropy {eq}`ve-gig`. The
operator shift $p - 1 \mapsto p - 1 - d/2$ is exactly the effect of the
conditional Gaussian volume factor $Y^{-d/2}$, and mirrors the natural
parameter $\theta_1 = p - 1 - d/2$ of the joint exponential family
{eq}`gh-natural-params`. The Variance-Gamma, Normal-Inverse Gamma, and
Normal-Inverse Gaussian joints are the $b \to 0$, $a \to 0$, and
$p = -1/2$ cases; each is computed on its own subordinator's
log-partition, with the Normal-Inverse Gaussian evaluated through its GH
embedding so that the order $p$ varies faithfully along the density-power
path.

## References

```{eval-rst}
.. [Stankyavichyus2026] Stankyavichyus, A. V. (2026). *Varentropy: Overview,
   Computational Routes, and Structural Decomposition*. Preprint.
```
