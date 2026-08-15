# Subordinator-tracking portfolios

In a generalized hyperbolic (GH) model the latent subordinator $Y$ is
the volatility clock: conditionally on $Y=y$, returns are Gaussian with
covariance $y\Sigma$ and mean $\mu + \gamma y$. The Bayes estimate
$E[Y\mid X]$ is already computed in the EM E-step. This note asks a
different question: can a *portfolio* $w^\top X$ be dominated by $Y$,
so that the portfolio itself reveals the clock?

The answer in the model is yes, and the analysis is closed-form. The
optimal direction is $w \propto \Sigma^{-1}\gamma$. The achievable
signal-to-noise is the scalar $\tilde q = \gamma^\top\Sigma^{-1}\gamma$
that already appears in the GH marginal density and the conditional GIG
posterior. "Dominated by $Y$" can be made precise in three inequivalent
ways. We did not find this *tracking* reading in the literature, but
each ingredient exists separately.

Background: {doc}`../theory/gh`, {doc}`../theory/gig`,
{doc}`../theory/em_algorithm`, {doc}`../theory/mean_risk_optimization`.
The S&P 500 measurement is {doc}`subordinator_tracking_empirics`.

## Setup

Let $X\in\mathbb{R}^d$ be a normal mean–variance mixture
{ref}`BarndorffNielsen1977 <barndorffnielsen1977>`
{ref}`McNeil2010 <mcneil2010>`

$$
X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}\, Z,
\qquad Z\sim N(0,\Sigma)\perp Y\ge 0,
$$

with $\mu,\gamma\in\mathbb{R}^d$ and $\Sigma\succ 0$. The mixing law
of $Y$ is GIG$(p,a,b)$ for GH, Gamma for variance-gamma (VG)
{ref}`MadanSeneta1990 <madan1990>`, InverseGaussian for
normal-inverse-Gaussian (NIG)
{ref}`BarndorffNielsen1997 <barndorffnielsen1997>`, or InverseGamma
for normal-inverse-gamma (NInvG). Write

$$
q(x) = (x-\mu)^\top\Sigma^{-1}(x-\mu),
\qquad
\tilde q = \gamma^\top\Sigma^{-1}\gamma,
\qquad
\mu_3 = E\bigl[(Y-E[Y])^3\bigr]
$$

for the Mahalanobis radius, the $\gamma$-energy in the $\Sigma$ metric,
and the third central moment of $Y$. The model has the scaling gauge
$(\gamma,\Sigma,Y)\mapsto(\gamma/c,\,\Sigma/c,\,cY)$; quantities below
are flagged when they are invariant.

Three standing facts.

1. **Projection.** For any $w$, the portfolio $P_w = w^\top X$ is a
   univariate mixture with the same $Y$ and parameters
   $(w^\top\mu,\, w^\top\gamma,\, w^\top\Sigma w)$
   {ref}`Blaesild1981 <blaesild1981>`. This is
   {eq}`nm-portfolio` in {doc}`../theory/mean_risk_optimization`,
   implemented as {py:meth}`~normix.mixtures.marginal.NormalMixture.project`.
2. **Posterior.** $Y\mid X=x$ is GIG with updated parameters
   $(p-d/2,\, a+\tilde q,\, b+q(x))$
   ({doc}`../theory/em_algorithm`). The posterior depends on $x$ only
   through $q(x)$. The linear statistic
   $s(x)=\gamma^\top\Sigma^{-1}(x-\mu)$ enters the joint density as a
   $y$-free tilt $e^{s(x)}$ and cancels from the conditional.
3. **Bayes optimality.** $E[Y\mid X]$ minimises mean-squared error
   among measurable functions of $X$. Normix computes it via
   {py:meth}`~normix.mixtures.joint.JointNormalMixture.conditional_expectations`.

## The tracking portfolio

Assume $\gamma\neq 0$ and take any $w$ with $m=w^\top\gamma\neq 0$.
Define the **tracker**

$$
\hat Y_w
= \frac{P_w - w^\top\mu}{w^\top\gamma}
= Y + \sqrt{Y}\,\varepsilon_w,
\qquad
\varepsilon_w\sim N(0,t_w)\perp Y,
\qquad
t_w = \frac{w^\top\Sigma w}{(w^\top\gamma)^2}.
$$

Conditionally on $Y$, $\hat Y_w\sim N(Y,\, t_w Y)$: an unbiased
observation of $Y$ whose noise variance is proportional to the signal
level. The mean-squared error is therefore

$$
E\bigl[(\hat Y_w - Y)^2\bigr]
= E\bigl[\mathrm{Var}(\hat Y_w\mid Y)\bigr]
= E[Y]\, t_w.
$$

**Proposition 1 (optimal direction).** $t_w \ge 1/\tilde q$ for all
$w$, with equality if and only if $w\propto\Sigma^{-1}\gamma$.

*Proof.* Cauchy–Schwarz in the $\Sigma$ inner product:

$$
(w^\top\gamma)^2
= \bigl((\Sigma^{1/2}w)^\top(\Sigma^{-1/2}\gamma)\bigr)^2
\le (w^\top\Sigma w)\,\tilde q,
$$

with equality iff $\Sigma^{1/2}w$ is parallel to
$\Sigma^{-1/2}\gamma$. $\square$

Two remarks on the informal recipe "maximise $w^\top\gamma$, minimise
$w^\top\Sigma w$".

- The Pareto frontier of that bi-objective problem is exactly the ray
  $\{c\,\Sigma^{-1}\gamma : c>0\}$. No other direction improves one
  objective without hurting the other.
- The degree-1 ratio $w^\top\Sigma w/w^\top\gamma$ is *not* a
  dominance measure: scaling $w$ down sends it to $0$. The
  scale-invariant objective is $t_w$. The degree-1 ratio does have an
  exact meaning — it is four times the expected pathwise floor of the
  portfolio below its location (Proposition 2).

Fix the normalisation $m=1$:

$$
w^\star = \frac{\Sigma^{-1}\gamma}{\tilde q},
\qquad
\hat Y = \frac{s(X)}{\tilde q},
\qquad
E\bigl[(\hat Y-Y)^2\bigr] = \frac{E[Y]}{\tilde q}.
$$

Computing $w^\star$ from a fitted model is two triangular solves
against the stored Cholesky factor `L_Sigma`.

### Two invariant signal-to-noise ratios

Under the gauge $c$: $\tilde q\mapsto\tilde q/c$,
$E[Y]\mapsto c\,E[Y]$, $\mathrm{Var}(Y)\mapsto c^2\mathrm{Var}(Y)$.
Two dimensionless, gauge-invariant ratios control everything below:

$$
\kappa_{\mathrm{lev}} = \tilde q\, E[Y]
\qquad\text{(level SNR)},
\qquad
\kappa = \frac{\tilde q\,\mathrm{Var}(Y)}{E[Y]}
\qquad\text{(fluctuation SNR)}.
$$

Their ratio $\kappa/\kappa_{\mathrm{lev}} = \mathrm{Var}(Y)/E[Y]^2$ is
the squared coefficient of variation of $Y$. Then

$$
\frac{E[(\hat Y-Y)^2]}{E[Y]^2} = \frac{1}{\kappa_{\mathrm{lev}}},
\qquad
\frac{E[(\hat Y-Y)^2]}{\mathrm{Var}(Y)} = \frac{1}{\kappa}.
$$

$\kappa_{\mathrm{lev}}\gg 1$ means the portfolio *level* is dominated
by the subordinator term. $\kappa\gg 1$ means the tracker resolves
$Y$'s *fluctuations*, which is the estimation-relevant statement. In
the identifiability gauge $E[Y]=1$ they collapse to
$\kappa_{\mathrm{lev}}=\tilde q$ and $\kappa=\tilde q\,\mathrm{Var}(Y)$.

The correlation of the tracker with $Y$ follows at once from the
fluctuation SNR:

$$
\mathrm{corr}(\hat Y, Y) = \sqrt{\frac{\kappa}{1+\kappa}}.
$$

At $\kappa=0.05$ this is about $0.22$; at $\kappa=1$ it is
$1/\sqrt{2}\approx 0.71$. Daily-equity $\kappa$ in the $10^{-2}$
range is a prior, not a theorem — see the empirics.

### The tracker as one synthetic asset

$P^\star=(w^\star)^\top X$ is the univariate mixture with parameters
$((w^\star)^\top\mu,\, 1,\, 1/\tilde q)$. The $d=1$ posterior on the
tracker alone is

$$
Y\mid\hat Y=\hat y
\sim \mathrm{GIG}\Bigl(p-\tfrac12,\; a+\tilde q,\; b+\tilde q\,\hat y^2\Bigr),
$$

to be compared with the full-information posterior
$\mathrm{GIG}(p-d/2,\, a+\tilde q,\, b+q(x))$. The $a$-updates agree.
The information lost by compressing $X$ into the scalar $\hat Y$ sits
in the $b$-slot and the GIG order.

Decompose $x-\mu$ into its $\gamma$-component and its
$\Sigma$-orthogonal complement:

$$
q(x)
= \frac{s(x)^2}{\tilde q} + q_\perp(x)
= \tilde q\,\hat y^2 + q_\perp(x),
\qquad q_\perp(x)\ge 0.
$$

The full posterior sees the *squared tracker* plus the orthogonal
Mahalanobis radius $q_\perp$ — a realized-dispersion statistic that is
informative about $Y$ even when $\gamma=0$. Two channels:

- **Linear (drift).** The tracker, strength $\tilde q$. Requires
  $\gamma\neq 0$. Tradable: it is a portfolio.
- **Quadratic (dispersion).** The radius, strength of order $d$ (one
  $\chi^2_1$ observation of scale $Y$ per orthogonal direction,
  visible in the posterior order $p-d/2$). Alive even for elliptical
  models. Not a linear payoff — in market terms it is a
  variance-swap-like object.

## Three estimators of $Y$

| Estimator | Form | Needs | MSE / $\mathrm{Var}(Y)$ |
|---|---|---|---|
| Tracker | $\hat Y = s(X)/\tilde q$ | direction $\Sigma^{-1}\gamma$, location $\mu$ | $1/\kappa$ |
| Linear Bayes | $E[Y] + \frac{\kappa}{1+\kappa}(\hat Y-E[Y])$ | also $E[Y]$, $\mathrm{Var}(Y)$ | $1/(1+\kappa)$ |
| Posterior mean | $E[Y\mid X]$ | full model | $\le 1/(1+\kappa)$ |

The linear-Bayes row is the best affine function of $X$. With
$\mathrm{Cov}(X)=E[Y]\,\Sigma + \mathrm{Var}(Y)\,\gamma\gamma^\top$
and $\mathrm{Cov}(X,Y)=\mathrm{Var}(Y)\,\gamma$, the Sherman–Morrison
formula gives

$$
\mathrm{Cov}(X)^{-1}\gamma
= \frac{\Sigma^{-1}\gamma}{E[Y] + \mathrm{Var}(Y)\,\tilde q}
$$

— the same direction again — and the usual credibility shrinkage
$\kappa/(1+\kappa)$ toward $E[Y]$, with MSE
$\mathrm{Var}(Y)/(1+\kappa)$. The posterior mean improves on every
affine rule because it also uses $q_\perp$. It has no closed form
(Bessel ratios), but
$E[\mathrm{Var}(Y\mid X)]\le \mathrm{Var}(Y)/(1+\kappa)$ is a usable
upper bound. Even at $\kappa=0$ the posterior remains consistent for
$Y$ as $d\to\infty$ under the model.

The tracker therefore costs a factor $(1+\kappa)/\kappa$ in MSE
relative to linear Bayes. That factor is negligible when
$\kappa\gg 1$ and ruinous when $\kappa\ll 1$: unbiasedness is
expensive when the signal is weak. The tracker's compensations: it is
a portfolio (tradable, holdable, a return series), it is linear (no
Bessel evaluations), and it needs only the direction
$\Sigma^{-1}\gamma$ — not the mixing law, and not even $\tilde q$ if
only relative movements matter.

## Three meanings of "dominated by $Y$"

**(i) Observation-level.** In $\hat Y_w = Y + \sqrt{Y}\,\varepsilon_w$,
the signal exceeds the noise on the event $Y\gg t_w$. For the optimal
tracker, $t_w=1/\tilde q$. Typical observations are
subordinator-dominated iff $\kappa_{\mathrm{lev}}\gg 1$.

**(ii) $L^2$ / distributional.** Along any sequence of models with
$\tilde q\to\infty$ and the mixing law of $Y$ held fixed,
$\hat Y\to Y$ in $L^2$, so the law of the portfolio converges to the
mixing law shifted by the location:

- VG: $P^\star - (w^\star)^\top\mu \Rightarrow \mathrm{Gamma}$;
- NIG: $\Rightarrow$ InverseGaussian;
- GH: $\Rightarrow$ GIG.

The portfolio *becomes* its mixing distribution. At finite $\tilde q$
the tracker is the univariate mixture with residual variance
$1/\tilde q$: the minimal Gaussian blur of $Y$ achievable by any
portfolio.

**(iii) Pathwise floor.**

**Proposition 2.** For $m=w^\top\gamma>0$ and
$\varepsilon\sim N(0,t_w)$,

$$
P_w - w^\top\mu
= m\bigl(Y + \sqrt{Y}\,\varepsilon\bigr)
\ge -\frac{m\,\varepsilon^2}{4}
$$

pathwise, because $y+\sqrt{y}\,\varepsilon \ge -\varepsilon^2/4$ for
all $y\ge 0$ (minimise the quadratic in $\sqrt{y}$). Hence the
expected worst-case shortfall below location is at most
$m\,t_w/4$, which is the degree-1 ratio
$w^\top\Sigma w/(4\,w^\top\gamma)$. For the optimal tracker the left
tail below location is Gaussian-thin:

$$
P\bigl(P^\star-(w^\star)^\top\mu \le -c\bigr)
\le 2\Phi\bigl(-2\sqrt{c\tilde q}\bigr),
\qquad c>0,
$$

since the event requires $\varepsilon^2\ge 4c$. The tracker is the
model's maximally asymmetric portfolio: GIG (semi-heavy, $e^{-ay/2}$)
upside inherited from $Y$, downside below location bounded by
$\chi^2_1/(4\tilde q)$. Shorting it is the model's linear
"short-volatility" trade.

**No-near-arbitrage.** As $\tilde q\to\infty$ the tracker converges to
a non-negative payoff above location — a limiting free lunch unless
$(w^\star)^\top\mu$ is pushed negative. In an economy where such
portfolios are priced, either $\tilde q$ stays moderate or the
location term compensates. Empirically relevant $\kappa$ should
therefore be $O(1)$ or smaller. That is a prior about markets, not a
theorem.

## The same direction from four angles

The ray $\Sigma^{-1}\gamma$ is canonical for the family, not just for
tracking.

1. **Max-SNR** (Proposition 1).
2. **Best linear predictor** of $Y$ given $X$ (Sherman–Morrison above).
3. **Maximal skewness.** Portfolio skewness depends on $w$ only through
   $t=w^\top\Sigma w/(w^\top\gamma)^2$. For $m>0$,

   $$
   \mathrm{skew}(P_w)
   = \frac{\mu_3 + 3t\,\mathrm{Var}(Y)}
          {\bigl(\mathrm{Var}(Y) + t\,E[Y]\bigr)^{3/2}}.
   $$

   Differentiating in $t$,

   $$
   \frac{d}{dt}\mathrm{skew} \gtrless 0
   \iff
   t \lessgtr t^\dagger
   := \frac{2\,\mathrm{Var}(Y)}{E[Y]} - \frac{\mu_3}{\mathrm{Var}(Y)}.
   $$

   Skewness is unimodal on the feasible interval
   $[1/\tilde q,\infty)$. The maximiser is $w^\star$ if and only if
   $t^\dagger\le 1/\tilde q$; in particular for all $\tilde q$ whenever

   $$
   E[Y]\,\mu_3 \ge 2\,\mathrm{Var}(Y)^2
   $$

   (equivalently $\kappa_1\kappa_3\ge 2\kappa_2^2$ in cumulants of
   $Y$). This holds with *equality* for Gamma mixing (VG) — both sides
   equal $2\alpha^2/\beta^4$ — and strictly for InverseGaussian
   ($E[Y]\,\mu_3 = 3\,\mathrm{Var}(Y)^2$) and InverseGamma
   ($E[Y]\,\mu_3 = \frac{4(\alpha-2)}{\alpha-3}\mathrm{Var}(Y)^2 > 2\,\mathrm{Var}(Y)^2$
   for $\alpha>3$). For general GIG we use the inequality as a
   per-fit check; Cauchy–Schwarz on the Lévy measure gives only the
   weaker $\kappa_1\kappa_3\ge\kappa_2^2$. When
   $t^\dagger>1/\tilde q$, the maximiser sits at the interior point
   $t=t^\dagger$ and is a whole cone of weights, so max-skewness and
   max-SNR decouple.
4. **Markowitz tangency under a pure volatility premium.** In the
   model, $E[X]-\mu = \gamma\,E[Y]$: all expected return beyond the
   location is compensation for subordinator exposure. If the location
   is common and riskless, $\mu=r\mathbf{1}$, the tangency weights
   {ref}`Markowitz1952 <markowitz1952>` are
   $\mathrm{Cov}(X)^{-1}(E[X]-r\mathbf{1})\propto\Sigma^{-1}\gamma$.
   In general this is the skewness–variance fund of
   {ref}`MenciaSentana2009 <mencasentana2009>`: riskless +
   mean–variance fund + a third fund that loads on the skewness
   vector. That third fund is the minimum-dispersion portfolio per
   unit of $w^\top\gamma$ — the tracker, subject to their budget
   constraint.

Item 4 is already in the efficient-surface reduction of
{doc}`../theory/mean_risk_optimization`: portfolios are parametrised
by $(\tilde\mu,\tilde\gamma)$ with minimal dispersion
$g(\tilde\mu,\tilde\gamma)$. The constrained tracker is the
$\tilde\gamma$-extreme of that surface.

## When can $\tilde q$ be large?

Cross-sectional growth of $\tilde q_d=\gamma_d^\top\Sigma_d^{-1}\gamma_d$
depends on how skewness aligns with the covariance factors.

**Equicorrelation.** Let
$\Sigma=\sigma^2\bigl[(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top\bigr]$
with $0<\rho<1$, and split $\gamma=g\mathbf{1}+\delta$ with
$\mathbf{1}^\top\delta=0$. Then

$$
\tilde q_d
= \frac{g^2 d}{\sigma^2(1-\rho+\rho d)}
+ \frac{\lVert\delta\rVert^2}{\sigma^2(1-\rho)}
\;\xrightarrow[d\to\infty]{}\;
\frac{g^2}{\sigma^2\rho}
+ \lim_d\frac{\lVert\delta\rVert^2}{\sigma^2(1-\rho)}.
$$

- If skewness loads only on the market ($\delta=0$) — the usual
  first-order description of equities, where the index carries the
  leverage asymmetry — $\tilde q_d$ **saturates** at
  $g^2/(\sigma^2\rho)$. Diversification cannot separate $Y$ from
  market noise, because the market factor is exactly the direction
  where the Gaussian component refuses to average out. The
  $d\to\infty$ tracker is the index itself, and the ceiling is the
  index's own $\kappa$.
- Unbounded recovery needs *idiosyncratic skewness dispersion*:
  $\lVert\delta\rVert^2\asymp d$. Then the tracker is a long–short
  in the $\delta$ direction, and recovery of $Y$ is a
  law-of-large-numbers effect across names.

**Contrast with the Bayes channel.** Conditionally on $Y=y$, the
whitened residuals
$\Sigma^{-1/2}(X-\mu-\gamma y)/\sqrt{y}$ are i.i.d. standard normal
in every direction. The Mahalanobis radius per dimension therefore
concentrates at $Y$ as $d\to\infty$, even for bounded $\tilde q$ and
even for $\gamma=0$. The quadratic channel does not saturate. It does
need a well-estimated $\Sigma^{-1}$ (serious at large $d$; see
{doc}`../theory/shrinkage` and {doc}`../theory/factor_analysis`), it
is quadratic in returns rather than a position, and it leans on exact
conditional Gaussianity. The tracker degrades more gracefully on all
three counts — when the linear signal is actually there.

## Open problems

(a) **Time series.** Per-period $Y_t$ with dependence (subordinated
Lévy or stochastic-volatility dynamics). Does the tracker plus online
EM ({doc}`../theory/online_em`, {ref}`Cappe2009 <cappe2009>`) yield a
real-time activity index comparable to realized variance? The empirics
answer this in the negative on daily US large caps: the object that
tracks volatility is $q_\perp$, not $\hat Y$.

(b) **Nonlinear tradable payoffs.** Among payoffs $\phi(w^\top X)$
(options on the tracker), what closes the gap to the quadratic
channel?

(c) **Several subordinators.** Common plus idiosyncratic clocks
{ref}`Semeraro2008 <semeraro2008>`. Which linear combinations track
the *common* clock, and is the answer again a generalised eigenvector
problem?

A GIG cumulant sweep for $\kappa_1\kappa_3\ge 2\kappa_2^2$ is also
open. The Gamma boundary is tight; the four GH fits in the empirics
all satisfy $t^\dagger\le 1/\tilde q$.

## Literature

We did not find the statement "the minimum-variance-per-unit-skewness
portfolio of a normal mean–variance mixture is an optimal linear
estimator of the latent subordinator, with conditional law
$N(Y,\,Y/\tilde q)$". The surrounding pieces sit in five strands.

**GH structure and the posterior.**
{ref}`BarndorffNielsen1977 <barndorffnielsen1977>` introduced GH;
{ref}`Blaesild1981 <blaesild1981>` proved closure under affine maps;
{ref}`MadanSeneta1990 <madan1990>` (VG) and
{ref}`BarndorffNielsen1997 <barndorffnielsen1997>` (NIG) are the
workhorse special cases; {ref}`McNeil2010 <mcneil2010>` Ch. 6 is the
standard mixture treatment. The GIG posterior and its Bessel-ratio
moments drive the EM algorithms of
{ref}`Protassov2004 <protassov2004>` and {ref}`Hu2005 <hu2005>` —
normix's E-step. What EM uses as a latent-variable imputation, this
note reads as a volatility measurement.

**Portfolio selection in mixtures.**
{ref}`MenciaSentana2009 <mencasentana2009>` is the closest work: for
location-scale mixtures of normals, any portfolio law is determined
by (mean, variance, skewness), the mean–variance–skewness frontier is
closed-form, and its efficient part is spanned by three funds, the
third loading on the skewness vector. Section 5(4) above is the
observation that their skewness fund *is* a mimicking portfolio for
the latent mixing variable — an interpretation they do not pursue
(their $\xi$ is integrated out, not estimated). The repo's own
mean-risk reduction ({doc}`../theory/mean_risk_optimization`, after
{ref}`Shi2016 <shi2016>`) contains the same
$(\tilde\mu,\tilde\gamma,\tilde\sigma)$ geometry.

**Skewness-maximising projections.**
{ref}`Loperfido2010 <loperfido2010>` posed projection pursuit by
skewness for skew-normal families;
{ref}`ArevalilloNavarro2020 <arevalillo2020>`
{ref}`ArevalilloNavarro2021 <arevalillo2021>` prove that for scale
mixtures of skew-normal vectors the max-skewness direction is
proportional to the shape vector (scaled by the scatter inverse).
Structurally identical to the $t^\dagger$ calculation, with a
truncated-normal latent instead of a subordinator. The
feasibility boundary $t^\dagger\le 1/\tilde q$ and the Gamma case
sitting exactly on it appear to be new for GH-type mixtures. The
canonical-form literature for skew-elliptical families
{ref}`AzzaliniCapitanio2014 <azzalini2014>` makes the same point that
one linear combination carries all the asymmetry.

**Mimicking / tracking portfolios.**
{ref}`HubermanKandelStambaugh1987 <huberman1987>` characterise
portfolios that can replace factors in pricing relations;
{ref}`BreedenGibbonsLitzenberger1989 <breeden1989>` build the
consumption-mimicking maximum-correlation portfolio;
{ref}`Lamont2001 <lamont2001>` tracks macro variables. The
subordinator tracker is a maximum-correlation mimicking portfolio
where the "factor" is the model's own latent activity. The mixture
structure supplies in closed form what that literature estimates by
regression.

**Time-change recovery and traded volatility.**
{ref}`Clark1973 <clark1973>` began the subordinated-returns program;
{ref}`AneGeman2000 <anegeman2000>` claimed recovery of the
transaction clock's moments (contested:
{ref}`MurphyIzzeldin2010 <murphy2010>`; see also
{ref}`RichardsonSmith1994 <richardson1994>`). Realized variance
estimates the integrated clock from high-frequency data
{ref}`BarndorffNielsenShephard2002 <barndorffnielsenshephard2002>` —
the quadratic channel, done across time instead of across assets.
Variance swaps {ref}`CarrWu2009 <carrwu2009>` are the traded
quadratic instrument. Multivariate subordination with common plus
idiosyncratic clocks
{ref}`Semeraro2008 <semeraro2008>`
{ref}`LucianoSemeraro2010 <lucianosemeraro2010>`
{ref}`LucianoMarenaSemeraro2016 <luciano2016>`
{ref}`BallottaBonfiglioli2016 <ballotta2016>` is the natural setting
for open problem (c).
