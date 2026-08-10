# Subordinator-tracking portfolios in normal mean–variance mixtures

**Status:** research note, 2026-08-10. Math and literature only — no implementation.
**Question posed:** in GH/VG/NIG models the subordinator $Y$ is hidden; besides the
Bayes estimate $E[Y \mid X]$, can a portfolio weight $w$ that maximises
$w^\top\gamma$ while minimising $w^\top\Sigma w$ produce a portfolio *dominated* by
$Y$, so that the portfolio itself reveals the hidden volatility clock?
**Answer:** yes, and the analysis is clean. The optimal direction is
$w \propto \Sigma^{-1}\gamma$, the achievable signal-to-noise is governed by the
scalar $\tilde q = \gamma^\top\Sigma^{-1}\gamma$ that already appears in the GH
marginal density and the conditional GIG posterior, and "dominated by $Y$" can be
made precise in three inequivalent ways (§4). We did not find this *tracking*
interpretation in the literature, but each ingredient exists separately (§8).

Theory background: [`docs/theory/gh.md`](../../docs/theory/gh.md),
[`docs/theory/gig.md`](../../docs/theory/gig.md),
[`docs/theory/em_algorithm.md`](../../docs/theory/em_algorithm.md),
[`docs/theory/mean_risk_optimization.md`](../../docs/theory/mean_risk_optimization.md).

## 1. Setup and notation

Throughout, $X$ is a $d$-dimensional normal mean–variance mixture,

$$
X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}\, Z,
\qquad Z \sim N(0, \Sigma) \perp Y \ge 0,
$$

with $\mu, \gamma \in \mathbb{R}^d$, $\Sigma \succ 0$, and subordinator $Y$
(GIG$(p,a,b)$ for GH; Gamma for VG; InverseGaussian for NIG; InverseGamma for
NInvG). We write

$$
e = E[Y], \quad v = \operatorname{Var}(Y), \quad \mu_3 = E[(Y-e)^3], \qquad
q(x) = (x-\mu)^\top\Sigma^{-1}(x-\mu), \qquad
\tilde q = \gamma^\top\Sigma^{-1}\gamma,
$$

matching the notation of the GH theory page. Three standing facts from the
existing docs:

1. **Projection.** For any $w \in \mathbb{R}^d$ the portfolio
   $P_w = w^\top X$ is a *univariate* normal mixture with the same subordinator
   and parameters $(\tilde\mu, \tilde\gamma, \tilde\sigma^2)
   = (w^\top\mu,\; w^\top\gamma,\; w^\top\Sigma w)$
   (equation `nm-portfolio` in `mean_risk_optimization.md`; closure of the GH
   family under affine maps is due to Blæsild 1981). Implemented as
   `NormalMixture.project(w)`.
2. **Posterior.** $Y \mid X = x \sim \mathrm{GIG}\bigl(p - \tfrac{d}{2},\;
   a + \tilde q,\; b + q(x)\bigr)$ (`em_algorithm.md`). Note the posterior
   depends on $x$ only through the quadratic statistic $q(x)$: the linear
   statistic $s(x) = \gamma^\top\Sigma^{-1}(x-\mu)$ enters the joint density as
   a $y$-free tilt $e^{s(x)}$ and cancels from the conditional.
3. **Bayes optimality.** $E[Y\mid X]$ minimises mean-squared error among all
   measurable functions of $X$; normix computes it via
   `JointNormalMixture.conditional_expectations` ($\alpha = 1$ in the
   conditional GIG moment formula).

The model has the scaling gauge $(\gamma, \Sigma, Y) \to (\gamma/c, \Sigma/c, cY)$;
we flag which quantities are gauge-invariant as they appear.

## 2. The tracking portfolio

Assume $\gamma \ne 0$ and take any $w$ with $m = w^\top\gamma \neq 0$. Define the
**tracker**

$$
\hat Y_w \;=\; \frac{P_w - w^\top\mu}{w^\top\gamma}
\;=\; Y + \sqrt{Y}\,\varepsilon_w,
\qquad
\varepsilon_w \sim N(0, t_w) \perp Y,
\qquad
t_w = \frac{w^\top\Sigma w}{(w^\top\gamma)^2}.
$$

Conditionally on $Y$, $\hat Y_w \sim N(Y,\, t_w Y)$: a conditionally unbiased
observation of $Y$ whose noise variance is proportional to the signal level.
Its mean-squared error is

$$
E\bigl[(\hat Y_w - Y)^2\bigr] = E\bigl[\operatorname{Var}(\hat Y_w \mid Y)\bigr]
= e\, t_w .
$$

**Proposition 1 (optimal direction).** $t_w \ge 1/\tilde q$ for all $w$, with
equality iff $w \propto \Sigma^{-1}\gamma$.

*Proof.* Cauchy–Schwarz in the $\Sigma$ inner product:
$(w^\top\gamma)^2 = \bigl((\Sigma^{1/2}w)^\top(\Sigma^{-1/2}\gamma)\bigr)^2
\le (w^\top\Sigma w)\,\tilde q$, equality iff
$\Sigma^{1/2}w \parallel \Sigma^{-1/2}\gamma$. $\square$

Two remarks on the original formulation.

- *"Maximise $w^\top\gamma$, minimise $w^\top\Sigma w$."* The Pareto frontier of
  this bi-objective problem is exactly the ray $\{c\,\Sigma^{-1}\gamma : c > 0\}$:
  Proposition 1 says no other direction improves one objective without hurting
  the other. So the informal recipe and the tracker coincide.
- *"The ratio $w^\top\Sigma w / w^\top\gamma$."* This ratio is homogeneous of
  degree 1 in $w$ (scaling $w$ down sends it to $0$), so it cannot by itself
  measure dominance. The scale-invariant objective is
  $t_w = w^\top\Sigma w/(w^\top\gamma)^2$. The degree-1 ratio does have an exact
  meaning, though — it is four times the expected worst-case shortfall of the
  portfolio below its location (Proposition 2 below), which is presumably the
  intuition behind it.

We fix the normalisation $m = 1$ from now on:

$$
w^\star = \frac{\Sigma^{-1}\gamma}{\tilde q}, \qquad
\hat Y = \hat Y_{w^\star} = \frac{s(X)}{\tilde q}, \qquad
s(x) = \gamma^\top\Sigma^{-1}(x - \mu),
\qquad
E\bigl[(\hat Y - Y)^2\bigr] = \frac{e}{\tilde q}.
$$

Computing $w^\star$ from a fitted model is two triangular solves against the
stored Cholesky factor `L_Sigma`.

### Two invariant signal-to-noise ratios

Under the gauge $c$: $\tilde q \to \tilde q/c$, $e \to ce$, $v \to c^2 v$. Two
dimensionless, gauge-invariant ratios control everything below:

$$
\kappa_{\mathrm{lev}} = \tilde q\, e
\qquad\text{(level SNR)}, \qquad\qquad
\kappa = \frac{\tilde q\, v}{e}
\qquad\text{(fluctuation SNR)},
$$

with $\kappa / \kappa_{\mathrm{lev}} = v/e^2$ the squared coefficient of
variation of $Y$. Then

$$
\frac{E[(\hat Y - Y)^2]}{E[Y]^2} = \frac{1}{\kappa_{\mathrm{lev}}},
\qquad\qquad
\frac{E[(\hat Y - Y)^2]}{\operatorname{Var}(Y)} = \frac{1}{\kappa}.
$$

$\kappa_{\mathrm{lev}} \gg 1$ means the portfolio *level* is dominated by the
subordinator term (the dominance question as posed); $\kappa \gg 1$ means the
tracker resolves $Y$'s *fluctuations*, which is the estimation-relevant
statement. In the normalisation $E[Y] = 1$ used for identifiability they are
$\kappa_{\mathrm{lev}} = \tilde q$ and $\kappa = \tilde q\,\operatorname{Var}(Y)$.

### The tracker as one synthetic asset

$P^\star = (w^\star)^\top X$ is the univariate mixture with parameters
$(\tilde\mu, \tilde\gamma, \tilde\sigma^2) = ((w^\star)^\top\mu,\; 1,\; 1/\tilde q)$.
Applying the $d = 1$ posterior formula to it:

$$
Y \mid \hat Y = \hat y \;\sim\; \mathrm{GIG}\Bigl(p - \tfrac12,\;
a + \tilde q,\; b + \tilde q\,\hat y^2\Bigr),
$$

to be compared with the full-information posterior
$\mathrm{GIG}\bigl(p - \tfrac{d}{2},\, a + \tilde q,\, b + q(x)\bigr)$. The
$a$-updates agree exactly; the information lost by compressing $X$ into the
scalar $\hat Y$ sits in the $b$-slot and the order. Decomposing $x - \mu$ into
its $\gamma$-component and its $\Sigma$-orthogonal complement gives

$$
q(x) = \frac{s(x)^2}{\tilde q} + q_\perp(x)
= \tilde q\,\hat y^2 + q_\perp(x), \qquad q_\perp(x) \ge 0,
$$

so the full posterior sees the *squared tracker* plus the orthogonal Mahalanobis
radius $q_\perp$ — a realized-dispersion statistic that is informative about $Y$
even when $\gamma = 0$. The two channels for learning $Y$ are therefore:

- **linear (drift) channel** — the tracker, strength $\tilde q$; requires
  $\gamma \ne 0$; tradable;
- **quadratic (dispersion) channel** — the radius, strength $\sim d$ (one
  $\chi^2_1$ observation of scale $Y$ per orthogonal direction, visible in the
  posterior order $p - d/2$); alive even for elliptical models; *not* a
  portfolio payoff — in market terms it is a variance-swap-like object, not a
  linear position.

## 3. Three estimators of $Y$

| estimator | form | needs | MSE / $\operatorname{Var}(Y)$ |
|---|---|---|---|
| tracker (conditionally unbiased) | $\hat Y = s(X)/\tilde q$ | direction $\Sigma^{-1}\gamma$, location $\mu$ | $1/\kappa$ |
| linear Bayes (credibility) | $e + \frac{\kappa}{1+\kappa}\,(\hat Y - e)$ | additionally $e, v$ | $1/(1+\kappa)$ |
| posterior mean (MMSE) | $E[Y \mid X]$, Bessel ratio in $q(x)$ | full model | $\le 1/(1+\kappa)$ |

The linear-Bayes row is the best affine function of $X$: with
$\operatorname{Cov}(X) = e\Sigma + v\gamma\gamma^\top$ and
$\operatorname{Cov}(X, Y) = v\gamma$, Sherman–Morrison gives
$\operatorname{Cov}(X)^{-1}\gamma = \Sigma^{-1}\gamma/(e + v\tilde q)$ — *the
same direction again* — and the usual credibility shrinkage
$\kappa/(1+\kappa)$ toward the prior mean, with
$\mathrm{MSE} = v/(1+\kappa)$. The MMSE bound follows since the posterior mean
improves on every affine rule; it has no closed form (Bessel ratios), but
$E[\operatorname{Var}(Y \mid X)] \le v/(1+\kappa)$ is a usable upper bound.

So the hierarchy is: the tracker costs a factor $\tfrac{1+\kappa}{\kappa}$ in
MSE relative to linear Bayes (negligible when $\kappa \gg 1$, ruinous when
$\kappa \ll 1$ — unbiasedness is expensive when the signal is weak), and linear
Bayes is dominated by the posterior mean, which exploits the quadratic channel.
The tracker's compensations: it is a *portfolio* (tradable, holdable,
back-testable as a return series), it is linear (no Bessel evaluations), and it
requires only the direction $\Sigma^{-1}\gamma$ — not the subordinator law, not
even $\tilde q$ if only relative movements matter.

## 4. What "dominated by the subordinator" means

Three precise versions.

**(i) Observation-level dominance.** In
$\hat Y_w = Y + \sqrt{Y}\varepsilon_w$, the signal exceeds the noise on the
event $Y \gg t_w$ (for the optimal tracker, $t_w = 1/\tilde q$); with
$E[Y] = 1$, typical observations are subordinator-dominated iff
$\kappa_{\mathrm{lev}} = \tilde q \gg 1$.

**(ii) $L^2$ / distributional convergence.** Along any sequence of models with
$\tilde q \to \infty$ (fixed subordinator law), $\hat Y \to Y$ in $L^2$, hence
the law of the tracker converges to the law of $Y$ shifted by the location:

- VG: $P^\star - (w^\star)^\top\mu \Rightarrow \mathrm{Gamma}(\alpha, \beta)$,
- NIG: $\Rightarrow \mathrm{InverseGaussian}(\mu_{\mathrm{ig}}, \lambda)$,
- GH: $\Rightarrow \mathrm{GIG}(p, a, b)$.

The portfolio *becomes* its mixing distribution. At finite $\tilde q$ the
tracker is the univariate mixture with $\tilde\sigma^2 = 1/\tilde q$, i.e. the
minimal Gaussian blur of the subordinator achievable by any portfolio.

**(iii) A pathwise floor.** This is the sharpest and least obvious statement.

**Proposition 2 (portfolio floor).** For any $w$ with $m = w^\top\gamma > 0$,
writing $\varepsilon \sim N(0, t_w)$ for the tracker noise,

$$
P_w - w^\top\mu \;=\; m\bigl(Y + \sqrt{Y}\,\varepsilon\bigr)
\;\ge\; -\,\frac{m\,\varepsilon^2}{4}
\qquad \text{pathwise,}
$$

since $y + \sqrt y\,\varepsilon \ge -\varepsilon^2/4$ for all $y \ge 0$
(minimise the quadratic in $\sqrt y$). Consequently

$$
E\Bigl[\,\text{worst-case shortfall below location}\,\Bigr]
\;\le\; \frac{m\, t_w}{4} \;=\; \frac{1}{4}\,
\frac{w^\top\Sigma w}{w^\top\gamma},
$$

which is exactly the degree-1 ratio from the original question, and for the
optimal tracker the left tail below location is *Gaussian-thin*:

$$
P\bigl(P^\star - (w^\star)^\top\mu \le -c\bigr)
\;\le\; P\bigl(\varepsilon^2 \ge 4c\bigr)
= 2\,\Phi\bigl(-2\sqrt{c\,\tilde q}\bigr), \qquad c > 0.
$$

So the tracker is the model's maximally asymmetric portfolio: GIG (semi-heavy,
$e^{-ay/2}$) upside inherited from $Y$, but downside below location bounded by
$\tfrac{1}{4\tilde q}\chi^2_1$. Shorting it is the model's pure
"short-volatility" trade — small steady collection against a semi-heavy left
tail — the linear-portfolio analogue of selling variance swaps.

**No-near-arbitrage bound.** As $\tilde q \to \infty$ the tracker's return
converges to $(w^\star)^\top\mu + Y \ge (w^\star)^\top\mu$: a limiting free
lunch unless the location term compensates. In an economy where such portfolios
are priced, either $\tilde q$ stays moderate or $(w^\star)^\top\mu$ is pushed
negative (the cost of holding the volatility-revealing asset). This is a
testable restriction linking $\mu$, $\gamma$, $\Sigma$, and it says the
empirically relevant regime is probably $\kappa = O(1)$: *partial* revelation
of $Y$, not dominance. Fitted daily-equity $\gamma$'s are small; measuring
$\hat\kappa$ on real panels is the first empirical task (§7).

## 5. The same direction from four angles

The direction $\Sigma^{-1}\gamma$ is canonical for the family, not just for
tracking:

1. **Max-SNR tracker** (Proposition 1).
2. **Best linear predictor** of $Y$ given $X$ (§3, Sherman–Morrison).
3. **Maximal-skewness projection.** From the moment formulas on the GH page,
   the portfolio's third central moment and variance depend on $w$ only through
   $(m, s^2) = (w^\top\gamma, w^\top\Sigma w)$, and for $m > 0$, with
   $t = s^2/m^2$,

   $$
   \operatorname{skew}(P_w)
   = \frac{\mu_3 + 3tv}{(v + te)^{3/2}},
   \qquad
   \frac{d}{dt}\operatorname{skew} \gtrless 0
   \iff t \lessgtr t^\dagger := \frac{2v}{e} - \frac{\mu_3}{v}.
   $$

   Skewness is unimodal in $t$ on $[1/\tilde q, \infty)$, so the
   maximal-skewness portfolio is $w \propto \Sigma^{-1}\gamma$ **iff**
   $t^\dagger \le 1/\tilde q$; in particular for all $\tilde q$ whenever

   $$
   e\,\mu_3 \ge 2v^2
   \qquad (\text{equivalently } \kappa_1\kappa_3 \ge 2\kappa_2^2
   \text{ in cumulants of } Y).
   $$

   This holds with *equality* for Gamma mixing (VG) — direct computation, both
   sides $2\alpha^2/\beta^4$ — and strictly for InverseGaussian
   ($e\mu_3 = 3v^2$) and InverseGamma ($e\mu_3 = \tfrac{4(\alpha-2)}{\alpha-3}v^2
   > 2v^2$ for $\alpha > 3$). For general GIG$(p,a,b)$ we conjecture it but have
   not proved it; Cauchy–Schwarz applied to the Lévy measure of $Y$ (GIG is
   infinitely divisible) gives only the weaker $\kappa_1\kappa_3 \ge \kappa_2^2$.
   A numerical sweep over $(p, a, b)$ would settle it (§7). When
   $t^\dagger > 1/\tilde q$, the skewness maximiser sits at interior
   $t = t^\dagger$ and is non-unique — a whole cone of weights — so
   max-skewness and max-SNR genuinely decouple there.
4. **Tangency portfolio under a pure volatility premium.** In the model,
   $E[X] - \mu = \gamma e$: all expected return beyond the location is
   compensation for subordinator exposure. If the location is common and
   riskless, $\mu = r\mathbf{1}$, then the Markowitz tangency weights are
   $\operatorname{Cov}(X)^{-1}(E[X] - r\mathbf 1) \propto \Sigma^{-1}\gamma$:
   every mean–variance investor already holds the subordinator tracker as their
   risky fund. In general ($\mu \ne r\mathbf 1$) the two funds differ, which is
   exactly the three-fund separation of Mencía & Sentana (2009): riskless +
   mean–variance fund + a skewness–variance fund; the third fund is, in our
   notation, the minimum-dispersion portfolio per unit of $w^\top\gamma$ —
   i.e. the tracker direction subject to their budget constraint.

Item 4 connects to machinery already in normix: the efficient-surface reduction
in `mean_risk_optimization.md` parametrises portfolios by
$(\tilde\mu, \tilde\gamma)$ with minimal dispersion
$g(\tilde\mu, \tilde\gamma)$; the constrained tracker is the
$\tilde\gamma$-extreme of that surface and `MeanRiskProblem.weights` already
computes it once the return constraint is dropped and $\tilde\gamma$ is pinned.

## 6. High-dimensional behaviour: when can $\tilde q$ be large?

Cross-sectional growth of $\tilde q_d = \gamma_d^\top\Sigma_d^{-1}\gamma_d$ is
the whole game, and the answer depends on how skewness aligns with the
covariance factors.

**Equicorrelation example.** Let
$\Sigma = \sigma^2\bigl[(1-\rho)I_d + \rho\mathbf1\mathbf1^\top\bigr]$,
$0 < \rho < 1$, and split $\gamma = g\mathbf 1 + \delta$ with
$\mathbf1^\top\delta = 0$. Then

$$
\tilde q_d
= \frac{g^2\, d}{\sigma^2(1 - \rho + \rho d)}
+ \frac{\lVert\delta\rVert^2}{\sigma^2(1-\rho)}
\;\xrightarrow[d\to\infty]{}\;
\frac{g^2}{\sigma^2\rho}
+ \lim_d \frac{\lVert\delta\rVert^2}{\sigma^2(1-\rho)} .
$$

- If skewness loads only on the market direction ($\delta = 0$) — the typical
  first-order description of equities, where the index carries the
  leverage/volatility-feedback asymmetry — $\tilde q_d$ **saturates** at
  $g^2/(\sigma^2\rho)$. Diversification cannot separate $Y$ from market noise,
  because the market factor is exactly the direction where the Gaussian
  component refuses to average out.
- $\tilde q_d \to \infty$ requires *idiosyncratic skewness dispersion*:
  $\lVert\delta\rVert^2 \asymp d$, i.e. cross-sectional variation in $\gamma_i$
  orthogonal to the dominant covariance factors. Then the tracker is a
  long–short portfolio in the $\delta$ direction, and recovery of $Y$ is a
  law-of-large-numbers effect across names.

**Contrast with the Bayes channel.** Conditional on $Y = y$, the posterior mode
solves $(a + \tilde q)\,y_\ast^2 - 2(p - \tfrac d2 - 1)y_\ast - (b + q(X)) = 0$
with $E[q(X) \mid Y = y] = y^2\tilde q + y d$, giving $y_\ast \to y$ as
$d \to \infty$ *even for bounded $\tilde q$* (and even for $\gamma = 0$): the
whitened residuals $\Sigma^{-1/2}(X - \mu - \gamma Y)/\sqrt{Y}$ are i.i.d.
standard normal in every direction, so the Mahalanobis radius per dimension
concentrates at $Y$ regardless of factor structure. The quadratic channel does
not saturate — but it needs the true $\Sigma^{-1}$ (a serious estimation
problem at large $d$; cf. [`docs/theory/shrinkage.md`](../../docs/theory/shrinkage.md)
and [`docs/theory/factor_analysis.md`](../../docs/theory/factor_analysis.md)),
it is quadratic in returns rather than a position, and it leans on exact
conditional Gaussianity. The tracker degrades gracefully on all three counts.

## 7. Findings, suggestions, open problems

**Findings.**

1. Mathematically the idea works, and the right objective is scale-invariant:
   maximise $(w^\top\gamma)^2 / (w^\top\Sigma w)$, solved by
   $w \propto \Sigma^{-1}\gamma$ with optimum $\tilde q$. The degree-1 ratio
   $w^\top\Sigma w / w^\top\gamma$ from the original question is (four times)
   the expected pathwise floor of the portfolio below its location
   (Proposition 2), not a dominance measure.
2. Dominance is governed by two gauge-invariant numbers:
   $\kappa_{\mathrm{lev}} = \tilde q\,E[Y]$ (levels) and
   $\kappa = \tilde q\operatorname{Var}(Y)/E[Y]$ (fluctuations). Tracking MSEs:
   $1/\kappa$ (unbiased tracker), $1/(1+\kappa)$ (linear Bayes, same direction,
   credibility shrinkage), $\le 1/(1+\kappa)$ (posterior mean).
3. As $\tilde q \to \infty$ the tracker's law converges to the subordinator's
   law (VG$\to$Gamma, NIG$\to$IG, GH$\to$GIG), and its downside below location
   is bounded by $\chi^2_1/(4\tilde q)$ — a near-arbitrage in the limit, so
   equilibrium pricing should keep $\kappa$ moderate. Expect partial
   revelation, not dominance, in fitted equity models.
4. The Bayes posterior uses only the quadratic statistic $q(x)$; the tracker is
   the optimal *linear* (hence tradable) compression, and
   $q(x) = \tilde q\hat y^2 + q_\perp(x)$ cleanly separates the drift channel
   (strength $\tilde q$, dies at $\gamma = 0$) from the dispersion channel
   (strength $d$, survives at $\gamma = 0$, saturation-free but non-tradable
   and $\Sigma$-hungry).
5. The direction $\Sigma^{-1}\gamma$ is simultaneously max-SNR, best-linear-
   predictor, max-skewness (for VG/NIG/NInvG mixing always; for GIG under a
   conjectured cumulant inequality $\kappa_1\kappa_3 \ge 2\kappa_2^2$), and the
   tangency portfolio when the entire risk premium is subordinator compensation.
6. With a strong common covariance factor carrying the skewness, $\tilde q_d$
   saturates as $d \to \infty$ (equicorrelation bound $g^2/(\sigma^2\rho)$);
   unbounded recovery requires idiosyncratic skewness dispersion.

**Suggestions for normix** (all deferred; none require new theory):

- Report $\tilde q$, $\kappa_{\mathrm{lev}}$, $\kappa$ as fitted-model
  diagnostics (two triangular solves plus subordinator moments already
  exposed). Natural home: a small `normix.finance` diversification-adjacent
  helper or a `NormalMixture` method; decide via the architect workflow if
  promoted.
- A tracker-vs-Bayes notebook: simulate VG/NIG/GH, compare $\hat Y$, linear
  Bayes, and `conditional_expectations` against the true path of $Y$; verify
  the $1/\kappa$, $1/(1+\kappa)$ MSE laws; then compute $\hat\kappa$ on
  `data/sp500_returns.csv` to locate real equity panels on the
  revelation spectrum.
- Settle the GIG cumulant inequality $\kappa_1\kappa_3 \ge 2\kappa_2^2$
  (numerical sweep over $(p,a,b)$; the Gamma boundary case suggests it is
  tight exactly in the VG limit).
- Estimation of the *direction* is the practical bottleneck: $\gamma$ is the
  weakest-identified parameter (skewness converges slowly). Shrinkage on
  $\gamma$, or restricting $\gamma$ to a factor structure, would stabilise
  $\Sigma^{-1}\gamma$; ties into the existing shrinkage/factor-analysis work.
- CVaR of $\pm P^\star$ via the existing `project(w)` + `CVaR` machinery
  quantifies the asymmetry in Proposition 2 (short tracker = model-consistent
  short-vol trade).

**Open problems.** (a) Time-series version: per-period $Y_t$ with dependence
(subordinated Lévy or SV dynamics) — does the tracker plus online EM
([`docs/theory/online_em.md`](../../docs/theory/online_em.md)) yield a
real-time activity index comparable to realized variance? (b) Optimal *nonlinear tradable* payoffs: among payoffs
$\phi(w^\top X)$ (options on the tracker), what closes the gap to the quadratic
channel? (c) Multi-subordinator models (common + idiosyncratic clocks à la
Semeraro): which linear combinations track the *common* clock, and is the
answer again a generalised eigenvector problem?

## 8. Literature

We did not find the specific statement "the minimum-variance-per-unit-skewness
portfolio of a normal mean–variance mixture is an optimal linear estimator of
the latent subordinator, with conditional law $N(Y, Y/\tilde q)$". The
surrounding pieces are all known, in five separate strands:

**GH structure and the conditional posterior.** Barndorff-Nielsen (1977)
introduced the GH family; Blæsild (1981) proved closure under affine maps
(the projection lemma); Madan & Seneta (1990) (VG) and Barndorff-Nielsen (1997)
(NIG) are the special cases; McNeil, Frey & Embrechts (2010, Ch. 6) is the
standard mixture treatment. The GIG posterior $Y \mid X$ and its Bessel-ratio
moments drive the EM algorithms of Protassov (2004) and Hu (2005) — normix's
E-step. What EM uses per-observation as a latent-variable imputation, this note
reads as a *volatility measurement*.

**Portfolio selection in mixture models.** Mencía & Sentana (2009) is the
closest work: for location-scale mixtures of normals, any portfolio law is
determined by (mean, variance, skewness), the mean–variance–skewness frontier
is closed-form, and its efficient part is spanned by three funds, the third
loading on the skewness vector. Our §5(4) is the observation that their
skewness fund *is* a mimicking portfolio for the latent mixing variable — an
interpretation they do not pursue (their $\xi$ is integrated out, not
estimated). The repo's own mean-risk reduction
(`docs/theory/mean_risk_optimization.md`, after [Shi2016]) contains the same
$(\tilde\mu, \tilde\gamma, \tilde\sigma)$ geometry.

**Skewness-maximising projections.** Loperfido (2010) posed projection pursuit
by skewness for skew-normal families; Arevalillo & Navarro (2020; 2021) prove
that for scale mixtures of skew-normal vectors the max-skewness direction is
proportional to the shape vector (scaled by the scatter inverse), via
third-cumulant eigenproblems. Structurally identical to our §5(3) — their
latent variable is a truncated normal, ours a subordinator — but we are not
aware of the result stated for GH-type mean–variance mixtures, where the
feasibility boundary $t^\dagger \le 1/\tilde q$ appears and the Gamma case
sits exactly on it. The canonical-form literature for skew-elliptical families
(Azzalini & Capitanio 2014, §5) makes the same point that one linear
combination carries all the asymmetry.

**Mimicking / tracking portfolios.** Huberman, Kandel & Stambaugh (1987)
characterise portfolios that can replace factors in pricing relations; Breeden,
Gibbons & Litzenberger (1989) build the consumption-mimicking maximum-
correlation portfolio; Lamont (2001) tracks macro variables. The subordinator
tracker is exactly a maximum-correlation mimicking portfolio where the
"factor" is the model's own latent activity variable — the mixture structure
supplies what that literature estimates by regression, and the answer is
closed-form.

**Time-change recovery and traded volatility.** Clark (1973) began the
subordinated-returns program; Ané & Geman (2000) claimed recovery of the
transaction clock's moments (contested: Murphy & Izzeldin 2010; see also
Richardson & Smith 1994 for moment-based recovery of latent information flow).
Realized variance estimates the integrated clock from high-frequency data
(Barndorff-Nielsen & Shephard 2002) — the quadratic channel of §2, done across
time instead of across assets. Variance swaps (Carr & Wu 2009) are the traded
quadratic instrument. On the model side, multivariate subordination with
common + idiosyncratic clocks (Semeraro 2008; Luciano & Semeraro 2010; Luciano,
Marena & Semeraro 2016; Ballotta & Bonfiglioli 2016) is the natural setting for
open problem (c).

## References

- Ané, T. & Geman, H. (2000). Order flow, transaction clock, and normality of
  asset returns. *Journal of Finance*, 55(5), 2259–2284.
- Arevalillo, J. M. & Navarro, H. (2020). Data projections by skewness
  maximization under scale mixtures of skew-normal vectors. *Advances in Data
  Analysis and Classification*, 14(2), 435–461.
- Arevalillo, J. M. & Navarro, H. (2021). Skewness-based projection pursuit as
  an eigenvector problem in scale mixtures of skew-normal distributions.
  *Symmetry*, 13(6), 1056.
- Azzalini, A. & Capitanio, A. (2014). *The Skew-Normal and Related Families*.
  Cambridge University Press.
- Ballotta, L. & Bonfiglioli, E. (2016). Multivariate asset models using Lévy
  processes and applications. *European Journal of Finance*, 22(13), 1320–1350.
- Barndorff-Nielsen, O. E. (1977). Exponentially decreasing distributions for
  the logarithm of particle size. *Proceedings of the Royal Society A*, 353,
  401–419.
- Barndorff-Nielsen, O. E. (1997). Normal inverse Gaussian distributions and
  stochastic volatility modelling. *Scandinavian Journal of Statistics*, 24(1),
  1–13.
- Barndorff-Nielsen, O. E. & Shephard, N. (2002). Econometric analysis of
  realized volatility and its use in estimating stochastic volatility models.
  *Journal of the Royal Statistical Society: Series B*, 64(2), 253–280.
- Blæsild, P. (1981). The two-dimensional hyperbolic distribution and related
  distributions, with an application to Johannsen's bean data. *Biometrika*,
  68(1), 251–263.
- Breeden, D. T., Gibbons, M. R. & Litzenberger, R. H. (1989). Empirical tests
  of the consumption-oriented CAPM. *Journal of Finance*, 44(2), 231–262.
- Carr, P. & Wu, L. (2009). Variance risk premiums. *Review of Financial
  Studies*, 22(3), 1311–1341.
- Clark, P. K. (1973). A subordinated stochastic process model with finite
  variance for speculative prices. *Econometrica*, 41(1), 135–155.
- Huberman, G., Kandel, S. & Stambaugh, R. F. (1987). Mimicking portfolios and
  exact arbitrage pricing. *Journal of Finance*, 42(1), 1–9.
- Hu, W. (2005). Calibration of multivariate generalized hyperbolic
  distributions using the EM algorithm. PhD thesis. (Also
  `docs/references.md` → Hu2005.)
- Lamont, O. A. (2001). Economic tracking portfolios. *Journal of
  Econometrics*, 105(1), 161–184.
- Loperfido, N. (2010). Canonical transformations of skew-normal variates.
  *TEST*, 19, 146–165.
- Luciano, E. & Semeraro, P. (2010). Multivariate time changes for Lévy asset
  models: characterization and calibration. *Journal of Computational and
  Applied Mathematics*, 233.
- Luciano, E., Marena, M. & Semeraro, P. (2016). Dependence calibration and
  portfolio fit with factor-based subordinators. *Quantitative Finance*, 16(7),
  1037–1052.
- Madan, D. B. & Seneta, E. (1990). The variance gamma (V.G.) model for share
  market returns. *Journal of Business*, 63(4), 511–524.
- McNeil, A. J., Frey, R. & Embrechts, P. (2010). *Quantitative Risk
  Management*. Princeton University Press. (`docs/references.md` → McNeil2010.)
- Mencía, J. & Sentana, E. (2009). Multivariate location-scale mixtures of
  normals and mean-variance-skewness portfolio allocation. *Journal of
  Econometrics*, 153(2), 105–121.
- Murphy, A. & Izzeldin, M. (2010). Recovering the moments of information flow
  and the normality of asset returns. *Applied Financial Economics*, 20.
- Protassov, R. S. (2004). EM-based maximum likelihood parameter estimation for
  multivariate generalized hyperbolic distributions. *Statistics and
  Computing*, 14. (`docs/references.md` → Protassov2004.)
- Richardson, M. & Smith, T. (1994). A direct test of the mixture of
  distributions hypothesis: measuring the daily flow of information. *Journal
  of Financial and Quantitative Analysis*, 29(1), 101–116.
- Semeraro, P. (2008). A multivariate variance gamma model for financial
  applications. *International Journal of Theoretical and Applied Finance*,
  11(1), 1–18.
- Shi, X. (2016). *Generalized Hyperbolic Distributions and Related Topics*.
  PhD thesis. (`docs/references.md` → Shi2016.)
