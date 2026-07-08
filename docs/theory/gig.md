# The Generalized Inverse Gaussian Distribution

In this section we review the definition and statistical properties of the
generalized inverse Gaussian (GIG) distribution.

**See also:** executable walkthrough in
{doc}`../tutorials/distributions/02_gig`.

## Definition

The generalized inverse Gaussian distribution is a continuous probability
distribution with the density function:

```{math}
:label: gig-pdf

f(x|p,a,b) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})} x^{p-1}
\exp\left(-\frac{1}{2}(b x^{-1} + a x)\right), \quad x > 0,
```

where $K_p(\cdot)$ is the modified Bessel function of the second kind
and the parameters $(p, a, b)$ satisfy:

```{math}
\begin{cases}
b > 0, \, a \geq 0 & \text{if } p < 0 \\
b > 0, \, a > 0 & \text{if } p = 0 \\
b \geq 0, \, a > 0 & \text{if } p > 0
\end{cases}
```

Throughout this package we assume that $a > 0$ and $b > 0$ for simplicity.

## Alternative Parameterization

Another useful way to parameterize the GIG distribution is to set
$\delta = \sqrt{b/a}$ and $\eta = \sqrt{ab}$. In that case the
density function can be written as:

```{math}
:label: gig-pdf-alt

f(x|p, \delta, \eta) = \frac{\delta^p}{2K_p(\eta)} x^{p-1}
\exp\left(-\frac{\eta}{2}(\delta x^{-1} + \delta^{-1} x)\right), \quad x > 0.
```

Note that $\delta$ serves as a scale parameter of the GIG distribution.

## Moment Generating Function

The moment generating function of a GIG distributed random variable $X$ is given by:

```{math}
E[e^{uX}] = \left(\sqrt{\frac{a}{a-2u}}\right)^p
\frac{K_p(\sqrt{b(a-2u)})}{K_p(\sqrt{ab})}
= \left(\sqrt{\frac{\eta}{\eta-2\delta u}}\right)^p
\frac{K_p(\sqrt{\eta^2-2\delta u})}{K_p(\eta)}.
```

## Moments

The moments of the GIG distribution have a particularly elegant form:

```{math}
:label: gig-moments

E[X^\alpha] = \left(\sqrt{\frac{b}{a}}\right)^\alpha
\frac{K_{p+\alpha}(\sqrt{ab})}{K_p(\sqrt{ab})}
= \delta^\alpha \frac{K_{p+\alpha}(\eta)}{K_p(\eta)}.
```

This formula is implemented in the {meth}`~normix.distributions.univariate.GeneralizedInverseGaussian.moment_alpha` method.

## Tail Behavior

The GIG density combines the power-law factor $x^{p-1}$ with a two-sided
exponential factor. For $a > 0$ and $b > 0$ the tail asymptotics are

```{math}
:label: gig-tails

f(x) &\sim C\, x^{p-1}\, e^{-a x / 2}, \qquad x \to \infty, \\
f(x) &\sim C\, x^{p-1}\, e^{-b / (2x)}, \qquad x \to 0^+,
```

where $C$ collects the normalising constants. The right tail decays
**exponentially** (a *semi-heavy* tail), while the left tail is suppressed
*super-exponentially* by the factor $e^{-b/(2x)}$. Consequently, whenever
$a, b > 0$, **every moment is finite**,

```{math}
E[X^\alpha] < \infty \quad \text{for all } \alpha \in \mathbb{R},
```

in agreement with the closed form {eq}`gig-moments`, which is well defined for
every real order $\alpha$.

The order $p$ controls only the polynomial prefactor; the exponential
factors dominate both tails. Genuine power-law (Pareto-type) heaviness therefore
appears only at the boundary of the parameter space, where one of the two
exponential factors degenerates:

- **Inverse-gamma limit** ($a \to 0$, $p < 0$): the right cutoff
  $e^{-a x / 2}$ vanishes and $f(x) \sim C\, x^{p-1}$, a power law
  with tail index $-p$. Positive moments then satisfy
  $E[X^\alpha] < \infty$ if and only if $\alpha < -p$.
- **Gamma limit** ($b \to 0$, $p > 0$): the origin factor
  $e^{-b / (2x)}$ vanishes, $f(x) \sim C\, x^{p-1}$ as
  $x \to 0^+$, and the right tail stays exponential. All positive moments
  remain finite, while inverse moments $E[X^{-\alpha}]$ are finite if and
  only if $\alpha < p$.

## Skewness and Kurtosis

Because {eq}`gig-moments` supplies every raw moment $m_k = E[X^k]$ in
closed form, the standardised moments follow directly. With $\mu = m_1$
and $\sigma^2 = m_2 - m_1^2$, the third and fourth central moments are

```{math}
\mu_3 &= m_3 - 3 m_1 m_2 + 2 m_1^3, \\
\mu_4 &= m_4 - 4 m_1 m_3 + 6 m_1^2 m_2 - 3 m_1^4,
```

and the skewness and excess kurtosis are

```{math}
:label: gig-kurtosis

\gamma_1 = \frac{\mu_3}{\sigma^3}, \qquad
\gamma_2 = \frac{\mu_4}{\sigma^4} - 3 .
```

For $a, b > 0$ both are finite (all four moments exist), and the GIG is
right-skewed and leptokurtic over the usual parameter range.

The kurtosis, however, requires a finite fourth moment, so it becomes unbounded
precisely in the heavy-tailed inverse-gamma limit. As $a \to 0$ one has
$X \to \text{InvGamma}(\alpha, b/2)$ with $\alpha = -p$, and

```{math}
\gamma_2 = \frac{6(5\alpha - 11)}{(\alpha - 3)(\alpha - 4)}, \qquad \alpha > 4,
```

which diverges as $\alpha \to 4^+$ and is undefined for
$\alpha \le 4$. The kurtosis therefore fails exactly where the right tail
is heaviest, which motivates tail diagnostics that remain finite without a
fourth moment (for example the *varentropy* $\operatorname{Var}[-\log f(X)]$,
which stays finite throughout this limit).

## Exponential Family Form

The GIG distribution belongs to the exponential family with density:

```{math}
f(x|\theta) = h(x) \exp\left(\theta^\top t(x) - \psi(\theta)\right)
```

**Sufficient Statistics:**

```{math}
t(x) = \begin{pmatrix} \log x \\ x^{-1} \\ x \end{pmatrix}
```

**Natural Parameters:**

The natural parameters $\theta = (\theta_1, \theta_2, \theta_3)$ are derived from
the classical parameters $(p, a, b)$:

```{math}
:label: gig-natural-params

\theta_1 &= p - 1 \quad \text{(unbounded)} \\
\theta_2 &= -\frac{b}{2} < 0 \\
\theta_3 &= -\frac{a}{2} < 0
```

The inverse transformation is:

```{math}
p = \theta_1 + 1, \quad b = -2\theta_2, \quad a = -2\theta_3
```

**Base Measure:**

```{math}
h(x) = \mathbf{1}_{x > 0}
```

**Log Partition Function:**

```{math}
:label: gig-log-partition

\psi(\theta) = \log 2 + \log K_p(\sqrt{ab}) + \frac{p}{2} \log\left(\frac{b}{a}\right)
```

where $p = \theta_1 + 1$, $a = -2\theta_3$, and $b = -2\theta_2$.

**Expectation Parameters:**

The expectation parameters $\eta = \nabla\psi(\theta) = E[t(X)]$ are:

```{math}
:label: gig-expectation-params

\eta_1 &= E[\log X] = \frac{\partial \psi}{\partial \theta_1} \\
\eta_2 &= E[X^{-1}] = \sqrt{\frac{a}{b}} \frac{K_{p-1}(\sqrt{ab})}{K_p(\sqrt{ab})} \\
\eta_3 &= E[X] = \sqrt{\frac{b}{a}} \frac{K_{p+1}(\sqrt{ab})}{K_p(\sqrt{ab})}
```

Unfortunately we do not have an analytical formula for $\eta_1 = E[\log X]$.
In practice it can only be approximated numerically.

## Maximum Likelihood Estimation

Given the expectation parameters $(\eta_1, \eta_2, \eta_3)$, computing the
natural parameters $(p, a, b)$ by solving the above equations is a challenging
problem. Let $x_1, x_2, \ldots, x_n$ be a sequence of sample data, then the
maximum likelihood estimator (MLE) of GIG is given by:

```{math}
:label: gig-mle

(\hat{p}, \hat{a}, \hat{b}) = \arg\max_{p,a,b} L_{GIG}(p, a, b | \hat{\eta}_1, \hat{\eta}_2, \hat{\eta}_3),
```

where $L_{GIG}$ is the log-likelihood function (excluding constants):

```{math}
:label: gig-loglik

L_{GIG}(p, a, b | \eta_1, \eta_2, \eta_3) =
-\frac{1}{2} b \hat{\eta}_1 - \frac{1}{2} a \hat{\eta}_2 + p \hat{\eta}_3
+ \frac{p}{2} \log(a/b) - \log(K_p(\sqrt{ab})),
```

and the sufficient statistics are:

- $\hat{\eta}_1 = \frac{1}{n} \sum_{k=1}^n x_k^{-1}$
- $\hat{\eta}_2 = \frac{1}{n} \sum_{k=1}^n x_k$
- $\hat{\eta}_3 = \frac{1}{n} \sum_{k=1}^n \log(x_k)$

One can verify that the optimal solution $(\hat{p}, \hat{a}, \hat{b})$ must
satisfy {eq}`gig-expectation-params` where $(\eta_1, \eta_2, \eta_3)$ are
replaced by $(\hat{\eta}_1, \hat{\eta}_2, \hat{\eta}_3)$.

## Numerical Challenges

As discussed in [Jorgensen2012](#jorgensen2012), there is no analytical expression for
$\hat{p}$ or even its partial derivatives. Most literature suggests fixing
$p$ when maximizing the log-likelihood function.

Even when $p$ is fixed, [Hu2005](#hu2005) reports that when $|p|$ is large
(say, above 10), there might be no solution for the first two equations in
{eq}`gig-expectation-params`.

## Hellinger Distance

To measure estimation errors, one good choice is the Hellinger distance between
the true and estimated parameters.

**Proposition.** Let $(p_1, a_1, b_1)$ and $(p_2, a_2, b_2)$ be the
parameters of two GIG distributions. The squared Hellinger distance between the
two distributions is given by:

```{math}
H_{GIG}^2(p_1, a_1, b_1 \| p_2, a_2, b_2) = 1 -
\frac{(a_1/b_1)^{p_1/4} (a_2/b_2)^{p_2/4}}
     {\sqrt{K_{p_1}(\sqrt{a_1 b_1}) K_{p_2}(\sqrt{a_2 b_2})}}
\frac{K_{\bar{p}}(\sqrt{\bar{a}\bar{b}})}{(\bar{a}/\bar{b})^{\bar{p}/2}},
```

where $\bar{p} = (p_1 + p_2)/2$, $\bar{a} = (a_1 + a_2)/2$, and
$\bar{b} = (b_1 + b_2)/2$.

## Special Cases

There are several important special cases of the GIG distribution:

- **Inverse Gaussian (IG)**: when $p = -1/2$
- **Gamma**: when $p > 0$ and $b \to 0$, giving $\text{Gamma}(p, a/2)$
- **Inverse Gamma**: when $p < 0$ and $a \to 0$, giving $\text{InvGamma}(-p, b/2)$

These special cases are implemented as separate classes in ``normix``:

- {class}`~normix.distributions.univariate.InverseGaussian`
- {class}`~normix.distributions.univariate.Gamma`
- {class}`~normix.distributions.univariate.InverseGamma`

## References

```{eval-rst}
.. [Jorgensen2012] Jørgensen, B. (2012). *Statistical Properties of the Generalized Inverse Gaussian Distribution*. Springer.
```
