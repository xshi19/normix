# Subordinator tracking in normal mean–variance mixtures

**A research report.** Theory 2026-08-10; S&P 500 empirics Phases 0–3, 2026-08-13.
Working notes: [`subordinator_tracking_portfolio.md`](subordinator_tracking_portfolio.md)
(full derivations and literature),
[`subordinator_tracking_sp500_plan.md`](subordinator_tracking_sp500_plan.md),
[`subordinator_tracking_sp500_results.md`](subordinator_tracking_sp500_results.md).
Code: `notebooks/subordinator_tracking/` (public API only; `normix` unchanged).

## Abstract

In a multivariate normal mean–variance mixture the latent subordinator $Y$
is not a portfolio. The linear functional of $X$ with minimal noise per
unit of $Y$-loading is nonetheless unique: $w^\star \propto \Sigma^{-1}\gamma$,
with signal-to-noise $\tilde q = \gamma^\top\Sigma^{-1}\gamma$. We make
"dominated by $Y$" precise (observation-level, $L^2$, and a pathwise
floor), place the tracker in a three-estimator hierarchy against the
Bayes posterior, and identify the same direction as max-SNR, best linear
predictor, max-skewness (for VG/NIG/NInvG; GH on a cumulant inequality),
and the tangency portfolio when the entire risk premium is subordinator
compensation.

On the S&P 500 current-constituent panel (2015–2026, $d\le 468$) the
mathematics is not contradicted — the MSE laws hold in simulation, and
$t^\dagger \le 1/\tilde q$ on every fitted VG/NIG/GH model — but the
linear channel is absent. Fitted $\kappa = \tilde q\,\mathrm{Var}(Y)/E[Y]$
sits at or below a day-wise sign-flip null at every $d$; $\tilde q_d$
grows with dimension only because the null does. The posterior mean
$E[Y\mid X]$, which uses the quadratic statistic $q(x)$, tracks realized
volatility (corr $0.58$ with 21-day RV; $0.93$ with cross-sectional
dispersion). Online EM does not turn the tracker into a clock. The
empirical conclusion is one the theory already allowed: daily equity
$\gamma$ is too small for a *portfolio* to reveal $Y$, while the
non-tradable quadratic channel still can.

## 1. Setup

Let $X\in\mathbb{R}^d$ be a normal mean–variance mixture

$$
X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}\, Z,
\qquad Z\sim N(0,\Sigma)\perp Y\ge 0,
$$

with $\mu,\gamma\in\mathbb{R}^d$, $\Sigma\succ 0$, and $Y$ GIG (GH),
Gamma (VG), InverseGaussian (NIG), or InverseGamma (NInvG). Write
$e=E[Y]$, $v=\mathrm{Var}(Y)$, $\mu_3=E[(Y-e)^3]$,
$q(x)=(x-\mu)^\top\Sigma^{-1}(x-\mu)$, and
$\tilde q=\gamma^\top\Sigma^{-1}\gamma$. The model has the scaling gauge
$(\gamma,\Sigma,Y)\mapsto(\gamma/c,\Sigma/c,cY)$.

Three standing facts:

1. For any $w$, $P_w=w^\top X$ is a univariate mixture with the same $Y$
   and parameters $(w^\top\mu,\,w^\top\gamma,\,w^\top\Sigma w)$
   (Blæsild 1981).
2. $Y\mid X=x \sim \mathrm{GIG}(p-d/2,\, a+\tilde q,\, b+q(x))$. The
   posterior depends on $x$ only through $q(x)$; the linear statistic
   $s(x)=\gamma^\top\Sigma^{-1}(x-\mu)$ cancels.
3. $E[Y\mid X]$ is MMSE among measurable functions of $X$.

## 2. The tracking portfolio

Assume $\gamma\neq 0$ and $m=w^\top\gamma\neq 0$. The **tracker**

$$
\hat Y_w = \frac{P_w - w^\top\mu}{w^\top\gamma}
= Y + \sqrt{Y}\,\varepsilon_w,
\qquad \varepsilon_w\sim N(0,t_w)\perp Y,
\qquad t_w = \frac{w^\top\Sigma w}{(w^\top\gamma)^2}
$$

is conditionally unbiased for $Y$, with
$E[(\hat Y_w-Y)^2]=e\,t_w$.

**Proposition 1.** $t_w \ge 1/\tilde q$ for all $w$, with equality iff
$w\propto\Sigma^{-1}\gamma$.

*Proof.* Cauchy–Schwarz in the $\Sigma$ inner product:
$(w^\top\gamma)^2 = \bigl((\Sigma^{1/2}w)^\top(\Sigma^{-1/2}\gamma)\bigr)^2
\le (w^\top\Sigma w)\,\tilde q$. $\square$

The Pareto frontier of "maximise $w^\top\gamma$, minimise $w^\top\Sigma w$"
is exactly this ray. The degree-1 ratio $w^\top\Sigma w/w^\top\gamma$ is
homogeneous of degree 1 and cannot measure dominance; it is four times
the expected pathwise floor of Proposition 2 below.

Normalising $m=1$,

$$
w^\star = \frac{\Sigma^{-1}\gamma}{\tilde q},
\qquad
\hat Y = \frac{s(X)}{\tilde q},
\qquad
E[(\hat Y-Y)^2] = \frac{e}{\tilde q}.
$$

Two gauge-invariant SNRs:

$$
\kappa_{\mathrm{lev}} = \tilde q\, e
\quad\text{(levels)},
\qquad
\kappa = \frac{\tilde q\, v}{e}
\quad\text{(fluctuations)}.
$$

Relative to $E[Y]^2$ the tracker MSE is $1/\kappa_{\mathrm{lev}}$; relative
to $\mathrm{Var}(Y)$ it is $1/\kappa$. In the identifiability gauge $e=1$,
$\kappa_{\mathrm{lev}}=\tilde q$ and $\kappa=\tilde q\,v$.

Decompose $q(x)=\tilde q\,\hat y^2 + q_\perp(x)$ with $q_\perp\ge 0$. The
full posterior sees the squared tracker plus the orthogonal Mahalanobis
radius. Two channels:

- **linear (drift)** — the tracker, strength $\tilde q$; requires
  $\gamma\neq 0$; tradable;
- **quadratic (dispersion)** — the radius, strength $\sim d$; alive at
  $\gamma=0$; not a linear payoff.

## 3. Three estimators of $Y$

| estimator | form | MSE / $\mathrm{Var}(Y)$ |
|---|---|---|
| tracker | $\hat Y=s(X)/\tilde q$ | $1/\kappa$ |
| linear Bayes | $e+\frac{\kappa}{1+\kappa}(\hat Y-e)$ | $1/(1+\kappa)$ |
| posterior mean | $E[Y\mid X]$ | $\le 1/(1+\kappa)$ |

Linear Bayes is the best affine rule: $\mathrm{Cov}(X,Y)=v\gamma$ and
Sherman–Morrison put $\mathrm{Cov}(X)^{-1}\gamma$ on the same ray
$\Sigma^{-1}\gamma$. The posterior mean also uses $q_\perp$, so it
improves on every affine rule; even at $\kappa=0$ it remains consistent
for $Y$ as $d\to\infty$ under the model.

The tracker costs a factor $(1+\kappa)/\kappa$ in MSE relative to linear
Bayes. Its compensations: it is a portfolio, it is linear, and it needs
only the direction $\Sigma^{-1}\gamma$.

## 4. Three meanings of "dominated by $Y$"

**(i) Observation-level.** In $\hat Y_w=Y+\sqrt{Y}\varepsilon_w$, typical
observations are subordinator-dominated iff $\kappa_{\mathrm{lev}}\gg 1$.

**(ii) $L^2$ / distributional.** Along $\tilde q\to\infty$ with the
subordinator law fixed, $\hat Y\to Y$ in $L^2$, so the portfolio law
converges to the mixing law (VG$\to$Gamma, NIG$\to$IG, GH$\to$GIG).

**(iii) Pathwise floor.**

**Proposition 2.** For $m=w^\top\gamma>0$ and $\varepsilon\sim N(0,t_w)$,

$$
P_w - w^\top\mu = m\bigl(Y+\sqrt{Y}\,\varepsilon\bigr)
\ge -\frac{m\,\varepsilon^2}{4}
$$

pathwise, since $y+\sqrt{y}\,\varepsilon\ge -\varepsilon^2/4$ for all
$y\ge 0$. Hence
$E[\text{worst-case shortfall below location}]\le m t_w/4$, which is
the degree-1 ratio, and for the optimal tracker

$$
P\bigl(P^\star-(w^\star)^\top\mu \le -c\bigr)
\le 2\Phi\bigl(-2\sqrt{c\tilde q}\bigr), \qquad c>0.
$$

As $\tilde q\to\infty$ the tracker converges to a non-negative payoff
above location — a limiting free lunch unless $(w^\star)^\top\mu$ is
pushed negative. Empirically relevant $\kappa$ should therefore be
$O(1)$ or smaller. That is a prior, not a theorem about markets.

## 5. The same direction from four angles

1. Max-SNR (Proposition 1).
2. Best linear predictor of $Y$ given $X$ (Sherman–Morrison).
3. **Maximal skewness.** Portfolio skewness depends on $w$ only through
   $t=w^\top\Sigma w/(w^\top\gamma)^2$, and
   $\mathrm{skew}(t)=(\mu_3+3tv)/(v+te)^{3/2}$ is unimodal on
   $[1/\tilde q,\infty)$. The maximiser is $w^\star$ iff
   $t^\dagger:=2v/e-\mu_3/v \le 1/\tilde q$. This holds with equality for
   Gamma mixing (VG) and strictly for InverseGaussian and InverseGamma
   ($\alpha>3$). For general GIG we use it as a per-fit check.
4. Markowitz tangency when $E[X]-\mu=\gamma e$ is the whole risk premium
   and $\mu=r\mathbf{1}$: weights $\propto\Sigma^{-1}\gamma$. In general
   this is Mencía & Sentana's (2009) skewness–variance fund.

## 6. When can $\tilde q$ be large?

Under equicorrelation $\Sigma=\sigma^2[(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top]$
and $\gamma=g\mathbf{1}+\delta$ with $\mathbf{1}^\top\delta=0$,

$$
\tilde q_d
\to \frac{g^2}{\sigma^2\rho} + \lim_d \frac{\lVert\delta\rVert^2}{\sigma^2(1-\rho)}.
$$

Market-only skewness ($\delta=0$) saturates. Unbounded recovery needs
idiosyncratic skewness dispersion $\lVert\delta\rVert^2\asymp d$. The
quadratic channel does not saturate: whitened residuals are i.i.d.
$N(0,Y)$ in every direction, so $q(X)/d$ concentrates at $Y$ even for
bounded $\tilde q$ and even for $\gamma=0$.

## 7. Empirical design

Panel: `data/sp500_returns.csv`, daily log returns 2015-12-15 → 2026-02-09
($T=2552$), $d=468$ current constituents. Nested random subsets
$d\in\{5,10,25,50,100,200,468\}$, five seeds; primary $d=50$ seed 0.
No winsorisation; 12 observations with $|r|>0.5$ kept. Survivorship
bias is accepted and stated.

Primary model: NIG with `regularization='a_eq_b'` (gauge $E[Y]=1$),
EM `tol=1e-5`. Secondary: GH as nested continuation from the NIG
embedding at $d\le 50$; VG with `alpha_min='density'` at $d\le 10$.
Null for $\tilde q$: day-wise sign-flip of demeaned returns (kills odd
joint moments, preserves $\Sigma$). Online EM: Cappé–Moulines EWMA on
$\eta$, no `regularize_a_eq_b`; report gauge-invariant $\kappa_t$.

Pre-registered hypotheses:

- **H1.** Fitted $\kappa\ll 1$; daily equity portfolios are residual-dominated.
  Back-of-envelope from univariate skew/kurtosis: $\kappa\sim 10^{-2}$.
- **H2.** $\tilde q_d$ saturates because $\gamma$ aligns with the market
  factor; any excess over $\kappa_{\mathrm{index}}$ is $\lVert\delta\rVert^2$.
- **H3.** For VG/NIG, $t^\dagger\le 0$, so model max-skewness is $w^\star$;
  sample maximisers overfit.
- **H4.** Short EWMA half-lives track regimes faster but inflate
  $\hat\kappa_t$ and churn $w^\star_t$.

## 8. Phase 0 — synthetic validation

Generator: the Phase 1 NIG at $d=50$ ($e=1$, $\tilde q=0.0749$,
$v=0.638$, $\kappa=0.0478$), with $\gamma\mapsto c\gamma$ for
$c\in\{0,1,3,10\}$. $T=2552$, $R=20$.

Tracker MSE $e/\tilde q$ and linear-Bayes MSE $v/(1+\kappa)$ hold to
$<1\%$ relative error. $\mathrm{corr}(\hat Y,Y)=\sqrt{\kappa/(1+\kappa)}$
is within $2\%$ at $\kappa\ge 0.43$ and $6.5\%$ at $\kappa=0.048$
(sample $0.228$ vs $0.214$). The posterior mean, even at $c=0$, has
$\mathrm{corr}(E[Y\mid X],Y)=0.95$ and MSE $0.09\times$ the linear-Bayes
bound: at $d=50$ the quadratic channel saturates.

Null floor of $\hat{\tilde q}$ at $c=0$, $T=2552$: mean $0.013/0.026/0.058$
at $d=10/25/50$, with 95% quantiles $0.021/0.040/0.078$. The generator's
own $\tilde q=0.075$ sits on the $d=50$ 95% quantile. $\hat{\tilde q}$ is
upward-biased; at $c=1$ the bias is the whole signal. Direction cosine
is $0.76$ at $\kappa=0.05$ and $0.98$ at $\kappa=4.8$.

Online rehearsal (slow $\gamma$-rotation of $\pi/4$, IG scale jump
$\times 3$ at $T/2$): short $h$ inflates $\kappa_t$ ($14\times$ at
$h=21$) and *worsens* direction recovery. Smoothed $h=21$ tracker matches
EWMA RV (both $\approx 0.46$) and does not beat it. A frozen true-model
posterior still has corr $0.96$.

## 9. Phase 1 — static S&P 500 (H1, H3)

Sign-flip tests on nested seed-0 universes:

| $d$ | $\hat{\tilde q}$ | null 95% | $p$ | $\hat\kappa$ |
|---|---|---|---|---|
| 5 | 0.0043 | 0.0098 | 0.41 | 0.0082 |
| 10 | 0.018 | 0.018 | 0.078 | 0.020 |
| 25 | 0.036 | 0.040 | 0.12 | 0.029 |
| 50 | 0.075 | 0.085 | 0.20 | 0.048 |
| 100 | 0.127 | 0.164 | 0.52 | 0.071 |
| 200 | 0.219 | 0.298 | 0.81 | 0.115 |
| 468 | 0.509 | 0.808 | 1.00 | 0.233 |

H1 holds. Equal-weight univariate NIG gives
$\kappa_{\mathrm{ew50}}=0.0093$, $\kappa_{\mathrm{ew468}}=0.0125$,
matching the $10^{-2}$ envelope. Panel $\hat\kappa$ exceeds the index
but not the null; at $d=468$, $p=1$. GH $\kappa_{\mathrm{lev}}$ agrees
with NIG at $d=50$ ($0.069$ vs $0.075$); extra GIG parameters do not
create a linear signal.

The tracker at $d=50$ is consistent with the i.i.d. mixture
($\mathrm{Var}(\hat Y)=14.67$ vs $13.99$; ACF$_1=0.012$) and is
**uncorrelated with realized vol** (corr $0.07$). The posterior mean
has corr $0.58$ with 21-day RV, $0.93$ with cross-sectional dispersion,
and $0.99$ with $q_\perp$. Sample channel split matches a draw from the
fitted model. The i.i.d. misspecification the plan expected in $\hat Y$
lives in $E[Y\mid X]$, not in the tracker.

H3, model side: $t^\dagger<0$ for every NIG and GH fit; $t^\dagger\approx 0$
for VG. The GIG inequality holds on these four GH points. Sample side:
direct skew maximisation (20 L-BFGS starts) has cosine $0.13$ ($d=10$)
and $-0.20$ ($d=50$) with $w^\star$; in-sample skew $8.3$ at $d=50$
collapses out of sample. Equal-weight / min-var / PC1 all have negative
sample skew (index leverage). The tracker is the only listed portfolio
with positive sample skew, and it is not the sample maximiser.

Anatomy: long/short, 24/26 names, $\mathrm{corr}(w,\beta)=0.065$.
Location $(w^\star)^\top\mu=-0.987$ against sample $E[P^\star]=0.013$, so
$E[\hat Y]=1=e$ — the location eats the $Y$-premium. Block-bootstrap
cosine 5/50/95 $=0.58/0.73/0.82$: a moderately tight cone around a
direction the sign-flip calls odd-moment noise. Proposition 2's bound is
vacuous ($\tilde q$ small).

## 10. Phase 2 — dimension sweep (H2)

Five nested seeds. $d=468$ is the full panel, so seeds coincide.

| $d$ | mean $\hat{\tilde q}$ | sign-flip 95% | mean $\hat\kappa$ | $\kappa_{\mathrm{index}}$ | PC1 share of $\tilde q$ |
|---|---|---|---|---|---|
| 5 | 0.012 | 0.010 | 0.014 | 0.009 | 0.39 |
| 10 | 0.020 | 0.018 | 0.019 | 0.011 | 0.27 |
| 25 | 0.035 | 0.040 | 0.025 | 0.013 | 0.14 |
| 50 | 0.060 | 0.085 | 0.039 | 0.012 | 0.086 |
| 100 | 0.107 | 0.164 | 0.061 | 0.013 | 0.045 |
| 200 | 0.217 | 0.298 | 0.111 | 0.013 | 0.019 |
| 468 | 0.509 | 0.808 | 0.233 | 0.012 | 0.007 |

$\hat{\tilde q}_d$ grows, roughly linearly, and does not saturate. That
is the shape of the note's idiosyncratic-$\delta$ branch, but every
point sits at or below the sign-flip floor. $\kappa_{\mathrm{index}}$ is
flat at $\sim 0.01$. PC1's share of $\tilde q$ *falls* with $d$. The
equicorrelation ceiling $g^2/(\bar\sigma^2\rho)$ is $0.004$–$0.011$.
$\lVert\delta\rVert^2/\lVert\gamma\rVert^2$ rises $0.18\to 0.41$: extra
$\tilde q$ lives in $\delta$, which is where $\gamma$-estimation noise
lives.

H2 is false in both intended readings. There is no saturating
market-skewness signal, and the growing $\tilde q_d$ is not recoverable
idiosyncratic skewness.

## 11. Phase 3 — online EM (H4)

Warm start on 2016–2017 ($d=50$): $\tilde q=0.368$, $\kappa=0.182$,
already higher than the full-sample fit. Online period 2017-12-14 →
2026-02-09. No re-gauging.

| $h$ | $\tau$ | mean $\kappa_t$ | mean $e_t$ | $\cos_{21}$ | corr$(\hat Y,\mathrm{RV})$ | corr$(q_\perp,\mathrm{RV})$ |
|---|---|---|---|---|---|---|
| 21 | 0 | 1.06 | 4772 | 0.59 | $-0.14$ | 0.14 |
| 63 | 0 | 0.41 | 27 | 0.79 | $-0.04$ | 0.01 |
| 252 | 0 | 0.11 | 1.9 | 0.94 | $-0.03$ | 0.55 |
| 504 | 0 | 0.084 | 1.6 | 0.98 | $-0.01$ | 0.60 |
| 21 | 0.1 | 0.52 | 1.3 | 0.47 | $-0.22$ | 0.54 |
| $1/t$ | 0 | 0.079 | 1.5 | 0.99 | $0.00$ | 0.61 |

The $1/t$ schedule ends at $\kappa=0.051$ vs static $0.048$. Short $h$
inflates gauge-invariant $\kappa_t$ ($22\times$ at $h=21$) and explodes
the gauge ($\bar e_t=4772$), as the plan's §12 warning described.
Shrinkage $\tau=0.1$ pins the gauge and rescues $q_\perp$ (corr $0.54$)
without rescuing $\hat Y$. Long $h$ or $1/t$: $\kappa_t$ near static,
stable $w^\star$, and $q_\perp$ matches the static posterior as a vol
proxy (corr $0.55$–$0.61$ with 21-day RV; $0.90$–$0.94$ with
cross-sectional dispersion).

EWMA of $m_t^2$ at $h=21$ has corr $0.66$ with 21-day RV. Unshrunk
$h=21$ $q_\perp$ loses that comparison. The quadratic channel is
competitive only once $n_{\mathrm{eff}}$ is hundreds of days — i.e.
once the online model is close to the static fit. The linear tracker
is uncorrelated with RV at every $h$.

H4 is right about estimation noise and wrong as a recipe for extracting
a linear clock. Open problem (a) of the theory note, on this panel: the
tracker plus online EM is not a real-time activity index comparable to
realized variance. $q_\perp$ is, and it does not need $\gamma_t$.

## 12. Conclusions

The linear algebra is clean and the simulation laws hold. On daily
US large-cap returns the drift channel is not there.

1. **H1 accepted.** $\kappa\sim 10^{-2}$ to $10^{-1}$ in point estimates,
   indistinguishable from a sign-flip null. A portfolio cannot dominate
   $Y$. Relative tracker MSE $1/\kappa$ is $20$–$100$.
2. **H2 rejected.** $\tilde q_d$ grows rather than saturates, but the
   growth is the null. PC1 does not carry $\tilde q$. Cross-section
   does not buy a linear clock.
3. **H3 split.** Model max-skewness is $w^\star$ on every fit, including
   GH. Sample third moments do not recover it.
4. **H4 split.** Short half-lives inflate $\kappa_t$ and churn $w^\star$;
   they do not produce a better $\hat Y$. The object that tracks
   volatility is $q_\perp$, at long memory.

The theory note's split between tradable-linear and non-tradable-quadratic
channels is the result. Mencía & Sentana's skewness fund is, on this
panel, a noisy long–short with no clock content. The EM E-step's
$E[Y\mid X]$ remains a usable latent-vol diagnostic, as the finance
tutorials already treat it.

None of this is a failure of the mathematics. Fitted daily-equity
$\gamma$ is small, the sign-flip says it is consistent with zero, and
Proposition 1 then says no linear payoff can reveal $Y$. That is
partial revelation in the $\kappa=O(10^{-2})$ regime the no-near-arbitrage
heuristic predicted, taken all the way to "no linear revelation."

## 13. Limitations

Current-constituent panel (survivorship); large caps only; single common
subordinator; i.i.d. GH (the ACF of $\hat Y$ being $\approx 0$ is
consistent with that, the ACF of $E[Y\mid X]$ is not). Full-$\Sigma$ at
$d=468$ has $T/d\approx 5.5$. Sign-flip nulls at $d\ge 100$ use $B=20$.
GH was a nested continuation, not a cold start, and was not run at
$d>50$. VIX and high-frequency RV were not used; panel-based proxies
suffice for the mechanism question. Multi-subordinator models and
nonlinear payoffs (theory note open problems (b), (c)) were out of scope.

## 14. Promotion candidates

Report $\tilde q$ and $\kappa$ on fitted models only with a matched null
floor. A tracker-vs-Bayes notebook is still worth teaching the channel
split; the equity punchline is $q_\perp$, not $w^\star$. No change to
`normix`.

## References

Full bibliography in [`subordinator_tracking_portfolio.md`](subordinator_tracking_portfolio.md) §8.
Primary: Barndorff-Nielsen (1977, 1997); Blæsild (1981); Madan & Seneta
(1990); Protassov (2004); Hu (2005); Mencía & Sentana (2009); McNeil,
Frey & Embrechts (2010); Shi (2016).
