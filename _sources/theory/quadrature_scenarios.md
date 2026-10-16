# Quadrature scenarios for mixture CVaR

We derive deterministic scenarios for the normal variance-mean mixtures of
{doc}`gh`. The portfolio notation and loss convention follow
{doc}`mean_risk_optimization`; the derivative formulas are developed in
{doc}`cvar_derivatives`. A worked comparison is in
{doc}`../tutorials/finance/08_quadrature_scenarios`.

## Portfolio returns and losses

Let $\mathbf X=\mu+\gamma Y+\sqrt{Y}Z$, where $Y>0$,
$Z\sim N(0,\Sigma)$ is independent of $Y$, and $\Sigma\succ0$.
For portfolio weights $w\ne0$, define

```{math}
X=w^\top\mathbf X
\stackrel{d}{=}\tilde\mu+\tilde\gamma Y+\tilde\sigma\sqrt{Y}\,Z_0,
\qquad
\tilde\mu=w^\top\mu,\quad
\tilde\gamma=w^\top\gamma,\quad
\tilde\sigma=\sqrt{w^\top\Sigma w},
```

where $Z_0\sim N(0,1)$ is independent of $Y$. The portfolio loss is $L=-X$.
For tail probability $\alpha\in(0,1)$, write $x_\alpha=F_X^{-1}(\alpha)$.
Then $\operatorname{VaR}_\alpha=-x_\alpha$ and
$\operatorname{CVaR}_\alpha=-E[X\mid X\le x_\alpha]$. Positive risk denotes a loss.

Given $Y=y$, put $m_y=\tilde\mu+\tilde\gamma y$,
$s_y=\tilde\sigma\sqrt y$, and $z_y(x)=(x-m_y)/s_y$.
With $\Phi$ and $\varphi$ denoting the standard normal CDF and density,

```{math}
F_X(x)=E[\Phi(z_Y(x))],\qquad
f_X(x)=E[\varphi(z_Y(x))/s_Y].
```

The CDF is continuous and strictly increasing by dominated convergence and
strict increase of every conditional CDF. Its quantile is unique. Density
and derivative formulas below assume that the displayed expectations are
finite at the quantile. The density can be unbounded at the location for
some Gamma mixing laws; continuity of the CDF does not require a bounded density.

## Rockafellar–Uryasev and conditioning

**Proposition.** Suppose $E|L|<\infty$ and $F_L$ is continuous. The function

```{math}
G(t)=t+\alpha^{-1}E[(L-t)^+]
```

is convex, has derivative $G'(t)=1-\alpha^{-1}\mathbb P(L>t)$, and attains
its minimum at the $(1-\alpha)$-quantiles of $L$. Its minimum is CVaR
({ref}`RockafellarUryasev2000 <rockafellaruryasev2000>`).

*Proof.* Each function $t\mapsto(\ell-t)^+$ is convex. Also,
$E[(L-t)^+]=\int_t^\infty\mathbb P(L>u)\,du$, so differentiation gives
the stated derivative. At a minimizer $t_*$, $\mathbb P(L>t_*)=\alpha$;
expanding the positive part gives
$G(t_*)=\alpha^{-1}E[L\mathbf1_{\{L>t_*\}}]$. $\square$

For loss distributions with atoms, the same minimum defines expected
shortfall with the appropriate fraction of mass at the quantile
({ref}`RockafellarUryasev2002 <rockafellaruryasev2002>`). Here only $Y$ will
be discretized: the normal component keeps the return distribution continuous.

Substituting $t=-x$ gives the return form

```{math}
:label: quad-ru
\operatorname{RU}(x)=-x+\alpha^{-1}E[(x-X)^+],\qquad
\operatorname{RU}'(x)=-1+F_X(x)/\alpha.
```

**Lemma.** For $V\sim N(m,s^2)$, $s>0$, and $z=(x-m)/s$,

```{math}
E[(x-V)^+]=(x-m)\Phi(z)+s\varphi(z),\qquad
E[V\mathbf1_{\{V\le x\}}]=m\Phi(z)-s\varphi(z).
```

*Proof.* Write $V=m+sZ_0$ and integrate
$\int_{-\infty}^z(z-u)\varphi(u)\,du$.
Since $\varphi'(u)=-u\varphi(u)$, the integral equals
$z\Phi(z)+\varphi(z)$. The second identity follows by the same integration. $\square$

Thus conditioning on $Y$ integrates the normal variable exactly. Define

```{math}
h_x(y)=(x-m_y)\Phi(z_y(x))+s_y\varphi(z_y(x)).
```

The tower property gives $\operatorname{RU}(x)=-x+\alpha^{-1}E[h_x(Y)]$.
At $F_X(x_\alpha)=\alpha$ this becomes

```{math}
:label: quad-cvar
\operatorname{CVaR}_\alpha
=-\alpha^{-1}E[m_Y\Phi(z_Y(x_\alpha))-s_Y\varphi(z_Y(x_\alpha))].
```

Whenever second moments exist, conditioning also reduces Monte Carlo variance:
$\operatorname{Var}(E[(x-X)^+\mid Y])\le\operatorname{Var}((x-X)^+)$.
The conditional integrand is smooth, even though the positive part has a kink.

### The envelope and parameter derivatives

Let $\theta$ be a model parameter and assume differentiation under the
expectation is valid. At the minimizing root, $\partial_x\operatorname{RU}=0$;
hence

```{math}
\frac{d\operatorname{CVaR}_\alpha}{d\theta}
=\left.\partial_\theta\operatorname{RU}(x;\theta)\right|_{x=x_\alpha}.
```

The root can therefore be held fixed for a first derivative. The expression
$C(x)=-\alpha^{-1}E[m_Y\Phi(z_Y)-s_Y\varphi(z_Y)]$ instead satisfies
$C(x)=\operatorname{RU}(x)-x(F_X(x)-\alpha)/\alpha$.
Its derivative at a frozen root contains the extra term
$-x\partial_\theta F_X/\alpha$. Equality of values at a root does not imply
equality of their frozen-root derivatives.

Differentiating the conditional positive part gives

```{math}
\partial_{\tilde\mu}\operatorname{CVaR}_\alpha=-1,\qquad
\partial_{\tilde\gamma}\operatorname{CVaR}_\alpha=-\alpha^{-1}E[Y\Phi(z_Y)],\qquad
\partial_{\tilde\sigma}\operatorname{CVaR}_\alpha=\alpha^{-1}E[\sqrt Y\varphi(z_Y)].
```

The analytic Hessian and the chain rule to $w$ are in
{doc}`cvar_derivatives`, following {ref}`Tasche1999 <tasche1999>` and
{ref}`RauHasanov2004 <rauhasanov2004>`. A frozen-root *second* derivative
omits the motion of the minimizer. It must not replace that Hessian.

## Scenarios as a probability measure

**Definition.** A scenario measure is
$Q=\sum_{j=1}^N p_j\delta_{Y_j}$, with $Y_j>0$, $p_j\ge0$, and $\sum_jp_j=1$.
Write $E_Q[g]=\sum_jp_jg(Y_j)$ and let
$X_Q=\tilde\mu+\tilde\gamma Y_Q+\tilde\sigma\sqrt{Y_Q}Z_0$, where $Y_Q\sim Q$.
An i.i.d. sample has $p_j=1/N$. Quadrature supplies deterministic nodes and
nonuniform probabilities. Weighted historical scenarios are another instance.

**Proposition.** Every conditional CDF, CVaR, gradient, and Hessian identity
above holds exactly for $X_Q$ with $E$ replaced by $E_Q$. In particular,
$F_Q$ is continuous and strictly increasing. If
$q_j=m_{Y_j}+s_{Y_j}\Phi^{-1}(\alpha)$ for the positive-mass atoms, then

```{math}
F_Q(\min_j q_j)\le\alpha\le F_Q(\max_j q_j).
```

*Proof.* $X_Q$ is a probability mixture of normals. At the minimum component
quantile every component CDF is at most $\alpha$, and at the maximum every
component CDF is at least $\alpha$. Taking a convex combination preserves both
inequalities. All conditional identities follow from the same tower property. $\square$

CVaR remains coherent and convex in $w$ for this fixed law
({ref}`Artzner1999 <artzner1999>`). The efficient-surface reduction of
{doc}`mean_risk_optimization` therefore holds for $X_Q$ as well. Negative
weights need not define a CDF and destroy this argument.

**Proposition (error transfer).** If both risks are finite, then

```{math}
|\operatorname{CVaR}_\alpha(X_Q)-\operatorname{CVaR}_\alpha(X)|
\le\alpha^{-1}\max_{x\in\{x_\alpha,x_{\alpha,Q}\}}
|(E_Q-E)[h_x(Y)]|.
```

*Proof.* Evaluate $\operatorname{RU}_Q$ at the minimizer of
$\operatorname{RU}$ to bound one difference; interchange the two functions
for the reverse inequality. $\square$

To first order the quantile error is
$x_{\alpha,Q}-x_\alpha\simeq-(F_Q-F_X)(x_\alpha)/f_X(x_\alpha)$.
CVaR stationarity removes this root error to first order, whereas derivative
errors also depend on the root shift.

For i.i.d. nodes, $\operatorname{Var}(E_Q[g])=\operatorname{Var}(g(Y))/N$ when
that variance exists. The usual $N^{-1/2}$ rate does not use smoothness. For
inverse-gamma shape at most two and negative skew, the second moment of the
CVaR integrand diverges; this rate is then unjustified.

## Gauss–Legendre quadrature

The Legendre polynomial
$P_M(t)=(2^M M!)^{-1}(d/dt)^M(t^2-1)^M$ is orthogonal to every polynomial
of degree below $M$ on $[-1,1]$. Its $M$ roots $t_j$ are distinct and interior
({ref}`DLMF <dlmf>`, [§18.3](https://dlmf.nist.gov/18.3)).
Let $\ell_j(t)=\prod_{k\ne j}(t-t_k)/(t_j-t_k)$ and
$w_j=\int_{-1}^1\ell_j(t)\,dt$.

**Proposition.** The rule $\sum_jw_jh(t_j)$ is exact for polynomials of
degree at most $2M-1$. Its weights satisfy

```{math}
w_j=\int_{-1}^1\ell_j(t)^2\,dt
=\frac{2}{(1-t_j^2)[P_M'(t_j)]^2}>0,\qquad \sum_jw_j=2.
```

*Proof.* Divide a polynomial $q$ of degree at most $2M-1$ as $q=P_Mr+s$,
with $\deg r,\deg s<M$. Orthogonality gives $\int q=\int s$, and $q=s$
at the nodes. Interpolation integrates $s$ exactly. Applying this result to
$\ell_j^2$ proves positivity; applying it to 1 gives the sum. The closed
form follows from the Legendre Christoffel–Darboux identity and
$(1-t^2)P_M'(t)=M(P_{M-1}(t)-tP_M(t))$
({ref}`DLMF <dlmf>`, [§18.2](https://dlmf.nist.gov/18.2) and
[§18.9](https://dlmf.nist.gov/18.9)). $\square$

No $M$-node rule can integrate every degree-$2M$ polynomial: the nonnegative
polynomial $\prod_j(t-t_j)^2$ vanishes at all nodes and has positive integral.
For $M=2$, $t_j=\pm1/\sqrt3$ and $w_j=1$. Cubics integrate exactly;
the error on $t^4$ is $2/5-2/9=8/45$.

For $h\in C^{2M}[-1,1]$, the remainder is

```{math}
\int_{-1}^1h(t)\,dt-\sum_jw_jh(t_j)
=\frac{2^{2M+1}(M!)^4}{(2M+1)[(2M)!]^3}h^{(2M)}(\xi)
```

for some $\xi\in(-1,1)$ ({ref}`DavisRabinowitz1984 <davisrabinowitz1984>`).
To obtain the constant, subtract the Hermite interpolant matching $h,h'$
at the roots. Bound its remainder above and below by the extreme values
of $h^{(2M)}$ times $\prod_j(t-t_j)^2/(2M)!$, then use continuity.
The leading coefficient of $P_M$ is $(2M)!/(2^M(M!)^2)$ and
$\int_{-1}^1P_M^2=2/(2M+1)$.

More generally, if $E_n(h)$ is the best uniform polynomial approximation
error of degree $n$, positivity and exactness imply
$|\int h-\sum_jw_jh(t_j)|\le4E_{2M-1}(h)$.
Analyticity in a Bernstein ellipse with parameter $\rho>1$ gives geometric
convergence in $M$; the bound also depends on the maximum magnitude in that
ellipse ({ref}`Trefethen2013 <trefethen2013>`). Entire integrands can still
require many nodes when they vary rapidly near the real axis.

## Log coordinates, windows, and heavy tails

Set $s=\log y$. Then $f_s(s)=f_Y(e^s)e^s$ and all four mixing laws have

```{math}
\log f_s(s)=\nu s-Ae^s-Be^{-s}+\text{constant}.
```

Here $A,B\ge0$. We use $a_Y$ for Gamma or inverse-gamma shape to distinguish
it from the tail probability $\alpha$.

| Law | $\nu$ | $A$ | $B$ | Positive moments |
|---|---:|---:|---:|---|
| Gamma$(a_Y,\beta)$ | $a_Y$ | $\beta$ | 0 | all |
| InverseGamma$(a_Y,\beta)$ | $-a_Y$ | 0 | $\beta$ | $k<a_Y$ |
| InverseGaussian$(\mu_Y,\lambda)$ | $-1/2$ | $\lambda/(2\mu_Y^2)$ | $\lambda/2$ | all |
| GIG$(p,a,b)$, $a,b>0$ | $p$ | $a/2$ | $b/2$ | all |

The log density is strictly concave. At its mode $s_0$, let
$A_c=Ae^{s_0}$, $B_c=Be^{-s_0}$, and $u=s-s_0$. Then

```{math}
A_c-B_c=\nu,\qquad A_cB_c=AB,\qquad
g_c(u)=\nu u-A_c\operatorname{expm1}(u)-B_c\operatorname{expm1}(-u).
```

A quadrature rule on a finite window, including its change-of-variable
Jacobian in $\omega_j>0$, yields
$Y_j=e^{s_0+u_j}$ and $p_j\propto\omega_j e^{g_c(u_j)}$.
Normalization removes the unknown density constant, so no Bessel normalizer
is needed. The sinh coordinate $u=c\sinh t$, with
$c=(A_c+B_c)^{-1/2}$, concentrates nodes near the mode.

### Truncation with moment tilts

For an integrand growing as $y^k$, the relevant exponent is $g_c(u)+ku$.
A right endpoint where this exponent has dropped by $P$ controls that
moment's tail, provided the moment exists. Density truncation alone does
not control $Y$-weighted tail integrals.

**Lemma.** Let $\psi(s)=\log f_s(s)$. If $\lambda_b=\psi'(b)+k<0$, then

```{math}
\int_b^\infty e^{ks}f_s(s)\,ds
\le \frac{e^{kb}f_s(b)}{|\lambda_b|}.
```

*Proof.* Concavity places $\psi(s)+ks$ below its tangent at $b$.
Exponentiate that tangent and integrate. $\square$

For inverse-gamma shape $a_Y$, the right tail in $s$ decays like
$e^{-a_Ys}$; a $k$-tilted window has length roughly $P/(a_Y-k)$.
The $k=1$ tilt covers CVaR and its skew derivative for negative skew.
At zero skew the skew Hessian requires $E[Y^{3/2}]$. Gamma with small shape
has a long *left* exponential tail. Interior GIG and inverse Gaussian have
double-exponential walls in $s$; near the inverse-gamma boundary a long
exponential stretch precedes the right wall.

Writing $k_+=\sup\{k\ge0:E[Y^k]<\infty\}$, the inverse-gamma domain
conditions give:

| Quantity | Divergence condition | Value |
|---|---|---|
| CVaR | $\tilde\gamma<0$, $k_+\le1$; or $\tilde\gamma=0$, $k_+\le1/2$ | $+\infty$ |
| $\partial_{\tilde\gamma}\operatorname{CVaR}$ | $\tilde\gamma\le0$, $k_+\le1$ | $-\infty$ |
| $\partial^2_{\tilde\gamma}\operatorname{CVaR}$ | $\tilde\gamma=0$, $k_+\le3/2$ | $+\infty$ |

Every finite rule has finite moments. Increasing its node count cannot
establish existence of moments of the original law. These domain conditions
must accompany any truncation-based computation.

## The near-step regime

If $y_c=(x_\alpha-\tilde\mu)/\tilde\gamma>0$, put $s_c=\log y_c$.
Differentiation gives

```{math}
\left.\frac{d z_{e^s}(x_\alpha)}{ds}\right|_{s=s_c}
=-\frac{\tilde\gamma\sqrt{y_c}}{\tilde\sigma},\qquad
\rho_{\mathrm{loc}}=\frac{|\tilde\gamma|\sqrt{y_c}}{\tilde\sigma},\qquad
\delta_s=1/\rho_{\mathrm{loc}}.
```

Thus the conditional CDF changes over log width $\delta_s$. The density
factor in the Hessian is a narrow peak on the same scale. A fixed rule can
resolve the CVaR value while missing most of this peak. For inverse-gamma
mixing and negative skew, small tail probabilities move $y_c$ outward;
in the drift-dominated limit $\rho_{\mathrm{loc}}\asymp\alpha^{-1/(2a_Y)}$.

The ratio $R=\delta_s/h(s_c)$, where $h(s_c)$ is the local log-node spacing,
measures resolution. It is a heuristic, not a bound. Agreement of two node
counts can also miss an unresolved peak. Increasing the node count or
placing panel boundaries near $s_c$ improves resolution. Boundaries that
move with $x$ or $w$ no longer define one fixed discrete law over an
optimization surface; convexity then holds only up to integration error.

Finally, the component-quantile bracket can be enormous when a log window
contains very large $Y_j$. A monotone coordinate $x=c+h\sinh t$, $h>0$,
compresses the bracket before bisection. This changes neither the root nor
the bracketing proof. The center and scale must themselves remain near the
relevant probability mass: a truncated mean of an infinite-mean law does
not supply such a center.
