# Subordinator tracking on the S&P 500

Phases 0–3, 2026-08-13. Derivations:
{doc}`subordinator_tracking`. Code lives in
`notebooks/subordinator_tracking/` and uses only the public API;
`normix` is unchanged.

The mathematics says a unique portfolio direction is the least-noisy
linear readout of the latent clock $Y$, and that a second,
non-tradable quadratic statistic can read $Y$ even when that
direction carries no signal. This page measures both claims on daily
US large-cap returns. The algebra holds. The linear channel is not
there. The quadratic channel is.

## What was asked

Two study modes, written down before the fits.

1. **Static.** Fit one distribution to the full history. Report the
   signal-to-noise numbers $\tilde q$ and $\kappa$, test them against
   a null that kills odd moments, and check whether the model's
   max-skewness portfolio is the tracker.
2. **Dynamic.** Update the model daily with EWMA-weighted online EM
   and rebalance $w^\star_t$. The question is extraction of a latent
   clock from existing data, not prediction. Evaluation is in-sample
   by design.

Four pre-registered hypotheses.

- **H1 (partial revelation).** Fitted $\kappa\ll 1$: daily equity
  portfolios are residual-dominated. A back-of-envelope from
  univariate skew and excess kurtosis in the gauge $E[Y]=1$,

  $$
  \kappa
  \approx \frac{\mathrm{skew}(X)^2}{3\,\mathrm{kurt}(X)},
  $$

  with daily-index $\mathrm{skew}\approx -0.5$ and excess kurtosis
  $5$–$15$, gives $\kappa\sim 10^{-2}$. Per-day tracker correlation
  would then be $\sqrt{\kappa/(1+\kappa)}\approx 0.1$. The open
  question is how much the cross-section adds.
- **H2 (saturation).** $\tilde q_d$ saturates as $d$ grows because
  fitted $\gamma$ aligns with the market covariance factor. Under
  market-only skewness the $d\to\infty$ tracker is the index, so the
  ceiling is the index's own $\kappa$. Any excess must come from
  idiosyncratic skewness dispersion $\lVert\delta\rVert^2$.
- **H3 (max-skew = tracker).** For fitted VG/NIG,
  $t^\dagger\le 0\le 1/\tilde q$ analytically, so the model's
  max-skewness portfolio is exactly $w^\star$. For GH the inequality
  is checked numerically. On *sample* third moments, the direct skew
  maximiser should point near $\hat\Sigma^{-1}\hat\gamma$ up to
  estimation noise.
- **H4 (responsiveness).** Short EWMA half-lives track volatility
  regimes faster but inject noise into $\hat\gamma_t$ — direction
  churn, turnover, and upward-biased $\hat\kappa_t$ — while long
  half-lives approach the static fit. There should be an interior
  optimum tied to the persistence of $Y_t$.

**Verdicts, up front.** H1 accepted. H2 rejected in both of its
intended readings. H3 split: true on every fitted model, false on
sample third moments. H4 split: short half-lives do inflate $\kappa_t$
and churn $w^\star$; they do not produce a better $\hat Y$.

## Data and cohorts

**Panel.** `data/sp500_returns.csv`: daily log returns,
2015-12-15 → 2026-02-09 ($T=2552$ days), $d=468$ current
constituents with near-complete history. The window covers the 2018
Volmageddon, the 2020-03 COVID crash, and the 2022 hiking cycle.

**Hygiene.** No winsorisation — tails are the signal. Twelve
observations with $|r|>0.5$ were flagged and kept. No zero-variance
columns. Survivorship bias is accepted: this is a current-constituent
panel of large caps, not an investable point-in-time universe. Fine
for a mechanism study; not a trading backtest.

**Nested universes.** For each random seed we draw a permutation of
the 468 tickers and take prefixes of size
$d\in\{5,10,25,50,100,200,468\}$. Nesting means the size-$10$ set
contains the size-$5$ set, and so on, so $\tilde q_d$ comparisons
along a seed are monotone in the name list rather than in a fresh
draw. Five seeds. At $d=468$ every seed is the full panel, so the
five runs coincide.

**Primary working universe.** $d=50$, seed 0. Small enough that a
full $\Sigma$ is well conditioned ($T/d\approx 51$), large enough
that the quadratic channel has something to do. Headline claims use
this universe; the $d$-sweep is the robustness check. Fits at
$d=468$ have $T/d\approx 5.5$ and are stress cases, not the basis of
the punchline.

**Market proxy.** Equal-weight panel mean
$m_t=\frac1d\sum_i x_{t,i}$. A univariate NIG fit of $m_t$ is the
index baseline $\kappa_{\mathrm{index}}$ — the $\delta=0$ ceiling
from the equicorrelation calculation.

**What "cohort" means here.** A cohort is one $(d,\mathrm{seed})$
universe: a fixed list of names, the full $T=2552$ days on those
names, one fitted model (and, in Phase 3, one online path). Phase 1
reports seed 0 at every $d$. Phase 2 averages the five seeds and
puts a standard error on $\hat{\tilde q}$. Phase 3 is seed 0,
$d=50$ only, plus a secondary $d=20$ unshrunk check.

## Models and fitting

**Primary: NIG**,
{py:class}`~normix.distributions.normal_inverse_gaussian.NormalInverseGaussian`.
No unbounded-likelihood boundary (unlike VG, whose density requires
Gamma shape $\alpha>d/2$), closed-form InverseGaussian M-step, and
`regularization='a_eq_b'` pins $\mu_{\mathrm{IG}}=1$, which is
exactly the identifiability gauge $E[Y]=1$. Then
$\kappa_{\mathrm{lev}}=\tilde q$ and $\kappa=\tilde q\,\mathrm{Var}(Y)$
directly.

**Secondary: GH** at $d\le 50$, started from the fitted NIG's exact
GH embedding (`to_generalized_hyperbolic`) and then freeing
$(p,a,b)$. Nested continuation, not a cold start. The same
`regularization='a_eq_b'` flag now pins $a=b$ on the GIG, which is
*not* $E[Y]=1$ unless $p=-1/2$. Report $\kappa_{\mathrm{lev}}$ and
$\kappa$, not raw $\tilde q$. **VG** only at $d\le 10$, with
`alpha_min='density'` so the marginal stays bounded
($\alpha\ge d/2+\varepsilon$). That clamp is a different estimand
from unconstrained VG; see the family comparison below.

**EM.** {py:class}`~normix.fitting.em.BatchEMFitter`, `default_init`
cold start, `tol=1e-5`. The package default $10^{-3}$ stops after one
iteration on this panel: daily-equity $\gamma$ is small enough that
$\mathrm{rms}(\Delta\gamma)/(1+\mathrm{rms}(\gamma))$ already meets
it. $\kappa$ and $\kappa_{\mathrm{lev}}$ are gauge-invariant, so the
regularisation does not create them.

**Online loop (Phase 3).** Hand-rolled from the public E-step / M-step
and {py:class}`~normix.fitting.eta_rules.EWMAUpdate`.
{py:class}`~normix.fitting.em.IncrementalEMFitter` draws random
batches, so it is the wrong tool for a chronological clock.
{ref}`Cappe2009 <cappe2009>` online EM with a constant step — the
EWMA regime that tracks slowly-varying parameters rather than
converging. No `regularize_a_eq_b` inside the loop: re-gauging every
step would mix $Y$-gauges inside the EWMA $\eta$ recursion. Report
gauge-invariant $\kappa_t$; $\tilde q_t$ is not comparable across
half-lives without a gauge.

## Metrics glossary

Every number in the tables below is one of these.

### Signal-to-noise

| Symbol | Meaning |
|---|---|
| $\tilde q=\gamma^\top\Sigma^{-1}\gamma$ | Energy of $\gamma$ in the $\Sigma$ metric. Scale-dependent: under $Y\mapsto cY$ it maps to $\tilde q/c$. Quoted in the $E[Y]=1$ gauge that `'a_eq_b'` pins for NIG. |
| $\kappa_{\mathrm{lev}}=\tilde q\,E[Y]$ | Level SNR. Tracker MSE relative to $E[Y]^2$ is $1/\kappa_{\mathrm{lev}}$. Gauge-invariant. |
| $\kappa=\tilde q\,\mathrm{Var}(Y)/E[Y]$ | Fluctuation SNR. Tracker MSE relative to $\mathrm{Var}(Y)$ is $1/\kappa$. Gauge-invariant. This is the headline "can a portfolio read $Y$?" number. |
| $\sqrt{\kappa/(1+\kappa)}$ | Model correlation $\mathrm{corr}(\hat Y,Y)$ if the mixture is well specified and $Y$ were observed. |
| $1/\kappa$ | Relative tracker MSE. At $\kappa=0.05$ this is $20$: the unbiased portfolio is twenty times noisier than $Y$'s own variance. |
| $\kappa_{\mathrm{index}}$ | $\kappa$ from a univariate NIG fit of the equal-weight mean $m_t$. The $\delta=0$ ceiling. |

### Estimators of $Y$

| Object | What it is |
|---|---|
| $\hat Y_t=(w^\star)^\top(x_t-\hat\mu)$ | Linear tracker. A portfolio, after subtracting location. Conditionally unbiased for $Y$ in the model. |
| Linear Bayes | $E[Y]+\frac{\kappa}{1+\kappa}(\hat Y-E[Y])$. Best affine rule. Same direction, shrunk toward the prior mean. |
| $E[Y\mid x_t]$ | Posterior mean from the fitted joint. Uses $q(x_t)$, hence both the squared tracker and $q_\perp$. MMSE among functions of $X$. |
| $q_\perp(x_t)=q(x_t)-\tilde q\,\hat Y_t^2$ | Orthogonal Mahalanobis radius. The part of the quadratic statistic that is invisible to any portfolio. |

On real data $Y_t$ is latent, so we cannot score $\hat Y$ against
truth. We score it against volatility *proxies* (next block) and
against the model's own moment laws (variance of $\hat Y$ should be
$\mathrm{Var}(Y)+E[Y]/\tilde q$; ACF of $\hat Y$ should be $0$ under
i.i.d. mixing).

### Volatility proxies

| Proxy | Construction | Role |
|---|---|---|
| 21-day RV | Centered 21-day realized variance of $m_t$ | Slow, model-free clock. The comparison the plan treats as primary. |
| EWMA RV | EWMA of $m_t^2$ at a matched half-life | Fairness check: same smoother family as the online model. |
| Cross-sectional dispersion | $\frac1d\sum_i(x_{t,i}-m_t)^2/\hat\sigma_i^2$ | Model-free cousin of $q_\perp$. A day when names disagree after scaling is a high-$Y$ day under the mixture. |

Correlations are Pearson unless stated. A number such as
"corr$(\hat Y,\mathrm{RV})=0.07$" means the linear tracker does not
move with the 21-day RV series. "corr$(E[Y\mid X],\mathrm{RV})=0.58$"
means the posterior does.

### Nulls and geometry

| Object | Meaning |
|---|---|
| Sign-flip null | For each day $t$, multiply the *whole* demeaned cross-section by an independent $S_t=\pm 1$. Kills every odd joint moment (including all skewness) and preserves $\Sigma$ and the mixing structure. Refitting produces the distribution of $\hat{\tilde q}>0$ that estimation noise alone can generate. The 95% quantile of that distribution is the floor in the tables; the $p$-value is the fraction of null fits with $\hat{\tilde q}$ at least as large as the real fit. |
| $c=0$ synthetic floor | Phase 0 analogue: draw from the fitted NIG with $\gamma$ set to $0$, refit, collect $\hat{\tilde q}$. Same idea, with known truth. |
| Direction cosine | $\cos\angle(\hat w^\star, w^\star)$ in the $\Sigma$ metric, or $\cos\angle(w^\star_t, w^\star_{t-21})$ for stability. $1$ is parallel, $0$ is orthogonal, negative is flipped. |
| Block-bootstrap cone | Moving-block bootstrap with 21-day blocks, to respect volatility clustering. The 5/50/95 percentiles of the cosine of $\hat w^\star$ against the full-sample $w^\star$ are a confidence cone on the direction. |
| PC1 share of $\tilde q$ | Write $\Sigma=\sum_k\lambda_k u_k u_k^\top$ and $\tilde q=\sum_k(u_k^\top\gamma)^2/\lambda_k$. The $k=1$ term is the fraction of $\tilde q$ that lives on the market eigenvector. High and stable $\Rightarrow$ market-aligned skewness (H2's saturation story). Falling with $d$ $\Rightarrow$ the extra $\tilde q$ is in small-$\lambda$ directions, which is also where $\gamma$-estimation noise lives. |
| $g\mathbf{1}+\delta$ split | $\gamma=g\mathbf{1}+\delta$ with $\mathbf{1}^\top\delta=0$. Market loading versus idiosyncratic residual. $\lVert\delta\rVert^2/\lVert\gamma\rVert^2$ rising with $d$ is the idiosyncratic-dispersion branch — or noise. |
| $t^\dagger$ | $2\mathrm{Var}(Y)/E[Y]-\mu_3/\mathrm{Var}(Y)$. Model max-skewness equals $w^\star$ iff $t^\dagger\le 1/\tilde q$. |
| $n_{\mathrm{eff}}$ | EWMA effective sample size $(2-w)/w\approx 2.9\,h$ for half-life $h$. A full $\Sigma$ at $d=50$ wants $n_{\mathrm{eff}}\gtrsim 150$; half-lives below $\sim 63$ days are under-determined without shrinkage. |
| Turnover | Daily unit-gross turnover of $w^\star_t$. How much the tracker portfolio is being rewritten. |
| $\tau$ | Shrinkage weight toward a static $\eta_0$ in the online $\eta$ update. $\tau=0$ is pure EWMA; $\tau=0.1$ pins the gauge and conditions $\Sigma_t$. |

## Phase 0 — synthetic validation

Truth is available. Generator: the Phase 1 NIG at $d=50$
($E[Y]=1$, $\tilde q=0.0749$, $\mathrm{Var}(Y)=0.638$,
$\kappa=0.0478$), with $\gamma\mapsto c\gamma$ for
$c\in\{0,1,3,10\}$. That sweeps
$\kappa\in\{0,\,0.0478,\,0.430,\,4.78\}$. $T=2552$, $R=20$ i.i.d.
draws. MSE and correlation laws use the true parameters (no refit).
$\hat{\tilde q}$ and direction recovery use cold-start EM.

### MSE and correlation laws (true parameters)

| $c$ | $\kappa$ | Tracker MSE rel. err. | Linear-Bayes MSE rel. err. | $\mathrm{corr}(\hat Y,Y)$ rel. err. | Post. MSE $/$ linear-Bayes bound | $\mathrm{corr}(E[Y\mid X],Y)$ |
|---|---|---|---|---|---|---|
| 0 | 0 | — | 1.1% | — | 0.089 | 0.954 |
| 1 | 0.0478 | 0.63% | 0.84% | 6.5% | 0.095 | 0.954 |
| 3 | 0.430 | 0.74% | 0.10% | 1.6% | 0.125 | 0.956 |
| 10 | 4.78 | 0.19% | 0.06% | 0.05% | 0.336 | 0.970 |

**What worked.** Tracker MSE $E[Y]/\tilde q$ and linear-Bayes MSE
$\mathrm{Var}(Y)/(1+\kappa)$ hold to $<1\%$ relative error. The
correlation law $\sqrt{\kappa/(1+\kappa)}$ is within $2\%$ at
$\kappa\ge 0.43$.

**What is already the empirics' punchline, in simulation.** Even at
$c=0$ (no drift channel) the posterior mean has
$\mathrm{corr}(E[Y\mid X],Y)=0.95$ and MSE $0.09\times$ the
linear-Bayes bound. At $d=50$ the quadratic channel saturates. A
*portfolio* cannot dominate $Y$ in this regime; $E[Y\mid X]$ still
can.

**A small miss.** At the weakest nonzero point ($c=1$, the
generator's own $\kappa$) sample $\mathrm{corr}(\hat Y,Y)$ is
$0.228$ vs $0.214$ ($6.5\%$ relative). Twenty replications of $2552$
days: this is a small systematic, not Monte Carlo noise. The $5\%$
acceptance flag fails only there.

### Estimation noise of $\hat{\tilde q}$

Null floor at $c=0$, $T=2552$, $R=20$:

| $d$ | Mean $\hat{\tilde q}_0$ | 95% quantile | Std |
|---|---|---|---|
| 10 | 0.0130 | 0.0206 | 0.0043 |
| 25 | 0.0261 | 0.0395 | 0.0091 |
| 50 | 0.0578 | 0.0775 | 0.0100 |

The floor grows with $d$. At $d=50$ the generator's own
$\tilde q=0.0749$ sits on the null 95% quantile: a full-history NIG
fit on this panel is *not* distinguishable from $\gamma=0$.

Bias at $T=2552$, $d=50$:

| $c$ | True $\tilde q$ | Mean $\hat{\tilde q}$ | Mean $\cos\angle$ |
|---|---|---|---|
| 0 | 0 | 0.058 | — |
| 1 | 0.075 | 0.138 | 0.76 |
| 3 | 0.674 | 0.750 | 0.94 |
| 10 | 7.49 | 7.90 | 0.98 |

**What failed as a point estimate.** $\hat{\tilde q}$ is
upward-biased. At $c=1$ the bias is the whole signal: you would
report $\tilde q\approx 0.14$ for a true $0.075$. Direction recovery
is still usable at $\kappa=0.05$ given $T=2552$ (cosine $0.76$) and
essentially exact at $\kappa=4.8$. Cutting $T$ to $500$ drops the
$c=1$ cosine to $0.40$.

Every later $\hat{\tilde q}$ is therefore reported against this
floor (or its sign-flip sibling), not as a raw SNR.

### Online EM rehearsal (H4, with known $Y$)

Path: $\gamma_t$ rotates by $\pi/4$ in the $\Sigma$ metric over
$T=2552$; InverseGaussian scale jumps $\times 3$ at $t=T/2$. Oracle
start at the true $t=0$ model. Filtered tracker (using
$\theta_{t-1}$ on $x_t$) versus true $Y$. EWMA realized variance of
the equal-weight mean is the RV baseline. True $\kappa=0.096$ on
this path.

| $h$ | $\tau$ | $n_{\mathrm{eff}}$ | corr$(\hat Y^{\mathrm{filt}},Y)$ | After $h_s=21$ | corr RV | Mean $\kappa_t$ | Mean $\cos\angle$ | Turnover |
|---|---|---|---|---|---|---|---|---|
| 21 | 0 | 61 | 0.32 | 0.46 | 0.47 | 1.36 | 0.18 | 0.084 |
| 21 | 0.1 | 61 | 0.14 | 0.09 | 0.47 | 0.50 | 0.47 | 0.139 |
| 63 | 0 | 182 | 0.26 | 0.30 | 0.48 | 0.64 | 0.31 | 0.054 |
| 252 | 0 | 727 | 0.30 | 0.38 | 0.46 | 0.23 | 0.57 | 0.024 |

**What H4 got right.** Short $h$ inflates gauge-invariant $\kappa_t$
($14\times$ at $h=21$) and raises turnover.

**What H4 got wrong as a recipe.** Short $h$ does *not* recover the
rotating direction — mean cosine is *worse* ($0.18$ vs $0.57$ at
$h=252$). The rotation is slow ($\pi/4$ over ten years); estimation
noise in $\hat\gamma_t$ dominates the tracking benefit. Shrinkage
$\tau=0.1$ at $h=21$ buys cosine ($0.47$) and spends $Y$-tracking
and turnover. The smoothed $h=21$ tracker matches EWMA RV (both
$\approx 0.46$) and does not beat it. A frozen true-model
*posterior* on the same path still has corr $0.96$ — again the
quadratic channel, which does not need $\gamma_t$.

In-sample $\hat Y$ (using $\theta_t$ on $x_t$) has corr $0.63$–$0.89$.
That is same-day overfit. The filtered numbers above are the honest
ones.

## Phase 1 — static S&P 500 (H1, H3)

Nested seed-0 universes. NIG at all $d$; GH nested continuation at
$d\le 50$; VG at $d\le 10$. Sign-flip $B=50$ for $d\le 50$, $B=20$
for $d\in\{100,200,468\}$.

### H1 — no linear extraction

| $d$ | $\hat{\tilde q}$ | Null 95% | $p$ | $\hat\kappa$ | $\sqrt{\kappa/(1+\kappa)}$ | $1/\kappa$ |
|---|---|---|---|---|---|---|
| 5 | 0.0043 | 0.0098 | 0.41 | 0.0082 | 0.090 | 122 |
| 10 | 0.018 | 0.018 | 0.078 | 0.020 | 0.139 | 51 |
| 25 | 0.036 | 0.040 | 0.12 | 0.029 | 0.167 | 35 |
| 50 | 0.075 | 0.085 | 0.20 | 0.048 | 0.214 | 21 |
| 100 | 0.127 | 0.164 | 0.52 | 0.071 | 0.258 | 14 |
| 200 | 0.219 | 0.298 | 0.81 | 0.115 | 0.321 | 8.7 |
| 468 | 0.509 | 0.808 | 1.00 | 0.233 | 0.435 | 4.3 |

Read a row as: "the fit produced this $\tilde q$; a sign-flip that
kills all skewness produces a larger $\tilde q$ this often; the
implied fluctuation SNR and the correlation you would get *if* $Y$
were observed and the model were true."

Equal-weight univariate NIG:
$\kappa_{\mathrm{ew50}}=0.0093$, $\kappa_{\mathrm{ew468}}=0.0125$.
The $d=50$ panel $\hat\kappa=0.048$ exceeds the index but sits
inside the sign-flip floor. $\hat{\tilde q}$ grows with $d$; the
null grows faster. At $d=468$, $p=1$: the fit is *less* skewed than
a typical sign-flip.

**H1 holds.** Point estimates sit in $10^{-2}$ to $10^{-1}$ and are
indistinguishable from odd-moment noise. Secondary GH/VG fits do
not create a linear signal either; the VG $\kappa$ gap is a
constraint, not leftover scale — next subsection.

### Family comparison — why VG $\kappa$ is smaller

The H1 table is NIG. The question is whether the secondary families
disagree on $\gamma$-energy or on $\mathrm{Var}(Y)$ after the
scaling gauge is removed, and whether a smaller VG $\kappa$ is an
EM or simulation artifact.

The mixture has the orbit
$(\gamma,\Sigma,Y)\mapsto(\gamma/c,\Sigma/c,cY)$. Under it
$\tilde q\mapsto\tilde q/c$, $e\mapsto ce$, $v\mapsto c^2 v$, so
raw $\tilde q$ and raw $e$ are not comparable across families.
Two invariants factor $\kappa$:

$$
\kappa_{\mathrm{lev}} = \tilde q\, e,
\qquad
\mathrm{cv}^2 = v/e^2,
\qquad
\kappa = \kappa_{\mathrm{lev}}\,\mathrm{cv}^2.
$$

$\kappa_{\mathrm{lev}}$ is $\tilde q$ in the $e=1$ gauge (scale-free
$\gamma$-energy). $\mathrm{cv}^2$ is $\mathrm{Var}(Y)$ in that same
gauge. `'a_eq_b'` pins those gauges differently:
NIG gets $e=\mu_{\mathrm{IG}}=1$; GH gets $a=b$, hence
$e=K_{p+1}(a)/K_p(a)$ ($0.10$ at $d=50$, $p\approx-2.47$); VG is
a degenerate GIG ($b=0$) and the flag is a no-op.

Seed-0 secondary fits on the same universes as the H1 table:

| $d$ | Family | $\tilde q$ | $e$ | $\kappa_{\mathrm{lev}}$ | $\mathrm{cv}^2$ | $\kappa$ |
|---|---|---|---|---|---|---|
| 5 | NIG | 0.0043 | 1 | 0.0043 | 1.91 | 0.0082 |
| 5 | GH | 0.014 | 0.25 | 0.0034 | 5.24 | 0.018 |
| 5 | VG | 0.0085 | 0.99 | 0.0084 | 0.385 | 0.0032 |
| 10 | NIG | 0.018 | 1 | 0.018 | 1.11 | 0.020 |
| 10 | GH | 0.105 | 0.14 | 0.014 | 3.39 | 0.049 |
| 10 | VG | 0.056 | 0.85 | 0.048 | 0.196 | 0.0094 |
| 50 | NIG | 0.075 | 1 | 0.075 | 0.638 | 0.048 |
| 50 | GH | 0.688 | 0.10 | 0.069 | 1.39 | 0.097 |

**Not a leftover scale.** After the $e=1$ gauge, GH and NIG
$\gamma$-energy agree at every overlapping $d$
($0.0034$ vs $0.0043$, $0.014$ vs $0.018$, $0.069$ vs $0.075$).
Directions agree too: $\cos\angle(\gamma_{\mathrm{NIG}},\gamma_{\mathrm{GH}})\ge 0.999$
in the NIG $\Sigma$-metric;
$\cos\angle(\gamma_{\mathrm{NIG}},\gamma_{\mathrm{VG}})\ge 0.988$.
The families are reading the same skewness axis.

**Not a simulation issue.** Phase 0 draws were NIG-only. These
numbers are full-history EM on the same real panel. Mean
log-likelihood at $d=10$: GH $27.06$, NIG $27.05$, VG $26.73$.
VG is a *worse* fit, not a different random draw.

**GH $\kappa$ is not higher because $e=0.10$.** $\kappa$ is
gauge-invariant. The $e=0.10$ vs $1$ contrast is the $a=b$ gauge.
GH $\kappa$ is higher because GIG puts more relative variance on
$Y$ ($\mathrm{cv}^2=1.39$ vs NIG $0.64$ at $d=50$).
$\kappa_{\mathrm{lev}}$ agrees; extra $(p,a,b)$ flexibility
does not create a linear signal.

**VG $\kappa$ is smaller because the density clamp binds.**
Gamma mixing has $\mathrm{cv}^2=1/\alpha$.
`alpha_min='density'` is $\alpha\ge d/2+\varepsilon$
($\varepsilon=0.1$), so
$\mathrm{cv}^2\le 2/(d+0.2)$. The fitted $\alpha$ sits
*exactly* on that floor: $2.6$ at $d=5$, $5.1$ at $d=10$.
Hence $\mathrm{cv}^2=0.385$ and $0.196$, against NIG $1.91$
and $1.11$. VG then inflates $\kappa_{\mathrm{lev}}$
($0.048$ vs NIG $0.018$ at $d=10$) to recover some skewness;
$\kappa=\kappa_{\mathrm{lev}}\,\mathrm{cv}^2$ still comes out
about half of NIG. Matching NIG's $\mathrm{cv}^2\approx 1.11$
at $d=10$ would need $\alpha\approx 0.9$, which is illegal for
a bounded VG density ($d/2=5$).

Cold-start VG with `alpha_min=None` on the same universes
drops below the floor and the $\gamma$-energy lines up with
NIG:

| $d$ | $\alpha$ | vs $d/2$ | $\kappa_{\mathrm{lev}}$ | $\mathrm{cv}^2$ | $\kappa$ |
|---|---|---|---|---|---|
| 5 | 1.16 | $<2.5$ | 0.0046 (NIG $0.0043$) | 0.86 | 0.0040 |
| 10 | 1.56 | $<5$ | 0.022 (NIG $0.018$) | 0.64 | 0.014 |

After the gauge, unconstrained VG $\gamma$-energy matches NIG and
$\mathrm{Var}(Y)$ is in the same order. The residual $\mathrm{cv}^2$
gap (Gamma vs InverseGaussian tails) is a family difference, not a
scale bug. Those fits have an unbounded density at $x=\mu$, which
is why the study clamped $\alpha$. The reported
$\kappa_{\mathrm{VG}}=0.009$ at $d=10$ is the clamped estimand.
Compare it to NIG at the same $d$ ($\kappa=0.020$), not to the
$d=50$ NIG headline.

None of the three families clears the sign-flip floor. Extra
subordinator flexibility still does not create a linear signal.

### Tracker versus quadratic channel ($d=50$)

| | Sample | Model / simulation |
|---|---|---|
| $\mathrm{Var}(\hat Y)$ | 14.67 | $\mathrm{Var}(Y)+E[Y]/\tilde q=13.99$ |
| $\mathrm{skew}(\hat Y)$ | 0.79 | 0.51 |
| ACF$_1$, ACF$_{21}$ of $\hat Y$ | 0.012, 0.005 | 0 |
| $\mathrm{corr}(\hat Y,\,E[Y\mid X])$ | 0.203 | 0.205 |
| $\mathrm{corr}(E[Y\mid X],\,q_\perp)$ | 0.989 | 0.998 |
| $\mathrm{corr}(E[Y\mid X],\,\text{x-sec. disp.})$ | 0.933 | — |
| $\mathrm{corr}(E[Y\mid X],\,\text{21-day RV})$ | 0.580 | — |
| $\mathrm{corr}(\hat Y,\,\text{21-day RV})$ | 0.071 | — |

**What worked (model side).** The linear tracker is consistent with
the i.i.d. mixture: variance, ACF near zero, and the split between
$\hat Y$ and $E[Y\mid X]$ match a draw from the fitted model.

**What failed (as a clock).** $\hat Y$ is uncorrelated with realized
vol. The posterior mean is essentially $q_\perp$
(corr $0.99$) and *is* the volatility clock (corr $0.58$ with 21-day
RV, $0.93$ with cross-sectional dispersion).

The i.i.d. misspecification the plan expected to see in $\hat Y$
(clustered tracker residuals) is not there. It lives in
$E[Y\mid X]$, which inherits the persistence of realized vol. Phase 3
therefore scores the quadratic channel against RV, and does not
expect a clustered $\hat Y_t$.

### H3 — max-skewness

$t^\dagger<0$ for every NIG and GH fit; $t^\dagger\approx 0$ for VG
(Gamma equality). Model max-skewness is $w^\star$ in all cases,
including the GIG check on these four GH fits.

Sample third moments do not recover it. Direct skew maximisation,
20 L-BFGS starts:

| $d$ | In-sample max skew | Tracker sample skew | $\cos\angle$ | OOS max | OOS tracker |
|---|---|---|---|---|---|
| 10 | 3.24 | 0.59 | 0.13 | $-0.023$ | 0.24 |
| 50 | 8.31 | 0.79 | $-0.20$ | 0.20 | $-0.047$ |

Equal-weight, min-variance, and PC1 all have $w^\top\gamma<0$ and
*negative* sample skew (index leverage). The tracker is the only
listed portfolio with positive sample skew, and it is not the sample
maximiser. In-sample skew $8.3$ at $d=50$ collapses out of sample.
Third moments overfit.

**H3 split.** True on the model. False as a sample-moment recipe.

### Anatomy of $w^\star$ at $d=50$

Unit-gross long/short, 24 long / 26 short, long share $0.45$,
Herfindahl $0.032$. Correlation of weights with market beta is
$0.065$ — not a market tilt. $\mathrm{corr}(\hat Y, m_t)=-0.37$.
Location $(w^\star)^\top\mu=-0.987$ against sample
$E[P^\star]=0.013$, so $E[\hat Y]=1=E[Y]$: the location term eats
the $Y$-premium and the raw tracker return is near zero. That is the
no-near-arbitrage sign from Proposition 2, in a regime where the
proposition's tail bound is vacuous ($\tilde q$ is small:
$P(\hat Y\le -1)=0.28$ vs bound $0.58$).

Block-bootstrap cosine 5/50/95 $=0.58/0.73/0.82$: a moderately tight
cone around a direction the sign-flip calls odd-moment noise. CVaR
at $5\%$ on the $\hat Y$ scale is $7.50$ long / $9.14$ short
(std $\approx 3.8$). Mild asymmetry, Gaussian residual, not a GIG
floor.

Split $\gamma=g\mathbf{1}+\delta$: $g=-8.5\cdot 10^{-4}$,
$\lVert\delta\rVert^2=2.1\cdot 10^{-5}$ vs
$\lVert\gamma\rVert^2=5.7\cdot 10^{-5}$. PC1 carries $5.6\%$ of
$\tilde q$. Not market-aligned skewness — consistent with isotropic
estimation noise.

## Phase 2 — dimension sweep (H2)

Five nested seeds. $d=468$ is the full panel, so seeds coincide.

| $d$ | Mean $\hat{\tilde q}$ | SE | Sign-flip 95% | Mean $\hat\kappa$ | Mean $\kappa_{\mathrm{index}}$ | PC1 share | Small-$\lambda$ 10% |
|---|---|---|---|---|---|---|---|
| 5 | 0.012 | 0.006 | 0.010 | 0.014 | 0.009 | 0.39 | 0.024 |
| 10 | 0.020 | 0.004 | 0.018 | 0.019 | 0.011 | 0.27 | 0.024 |
| 25 | 0.035 | 0.007 | 0.040 | 0.025 | 0.013 | 0.14 | 0.075 |
| 50 | 0.060 | 0.015 | 0.085 | 0.039 | 0.012 | 0.086 | 0.085 |
| 100 | 0.107 | 0.014 | 0.164 | 0.061 | 0.013 | 0.045 | 0.080 |
| 200 | 0.217 | 0.011 | 0.298 | 0.111 | 0.013 | 0.019 | 0.095 |
| 468 | 0.509 | 0 | 0.808 | 0.233 | 0.012 | 0.007 | 0.106 |

$\hat{\tilde q}_d$ grows, roughly linearly in $d$, and **does not
saturate**. That is the shape of the idiosyncratic-$\delta$ branch
in the theory note — except every point sits at or below the
sign-flip floor. (The $d=5,10$ means slightly exceed the *seed-0*
null; they do not exceed a seed-matched null, and PC1 shares at
$d=5$ range $0.001$–$0.76$ across seeds.) The index ceiling
$\kappa_{\mathrm{index}}\approx 0.01$ is flat in $d$.

Attribution is the opposite of market-aligned skewness: PC1's share
of $\tilde q$ *falls* from $0.39$ to $0.007$. The equicorrelation
ceiling $g^2/(\bar\sigma^2\rho)$ is $0.004$–$0.011$, the same order
as $\kappa_{\mathrm{index}}$.
$\lVert\delta\rVert^2/\lVert\gamma\rVert^2$ rises $0.18\to 0.41$:
the extra $\tilde q$ lives in $\delta$, which is where
$\gamma$-estimation noise lives. The small-$\lambda$ $10\%$ column
is the share of $\tilde q$ on the weakest tenth of the spectrum; it
rises, slowly, as one would expect if the "signal" is noise in
ill-conditioned directions.

**H2 is false in both intended readings.** There is no saturating
market-skewness signal, and the growing $\tilde q_d$ is not
recoverable idiosyncratic skewness. It is the $d$-dependent null.
Cross-section does not buy a linear clock.

## Phase 3 — online EM (H4)

Warm start: NIG on 2016–2017 (504 days), $d=50$ seed 0. Warm-window
$\tilde q=0.368$, $\kappa=0.182$ — already higher than the
full-sample $0.075/0.048$. Online period 2017-12-14 → 2026-02-09
($T=2048$). No re-gauging. Half-life $h$ with
$w=1-2^{-1/h}$; $1/t$ is the consistency check
({doc}`../theory/online_em`).

| $h$ | $\tau$ | $n_{\mathrm{eff}}$ | Mean $\kappa_t$ | Mean $E[Y]_t$ | Turnover | $\cos_{21}$ | corr$(\hat Y,\mathrm{RV})$ | corr$(q_\perp,\mathrm{RV})$ | corr$(q_\perp,\mathrm{disp})$ |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 0 | 61 | 1.06 | 4772 | 0.084 | 0.59 | $-0.14$ | 0.14 | 0.61 |
| 63 | 0 | 182 | 0.41 | 27 | 0.055 | 0.79 | $-0.04$ | 0.01 | 0.44 |
| 126 | 0 | 364 | 0.19 | 3.3 | 0.039 | 0.88 | $-0.05$ | 0.37 | 0.80 |
| 252 | 0 | 727 | 0.11 | 1.9 | 0.027 | 0.94 | $-0.03$ | 0.55 | 0.90 |
| 504 | 0 | 1454 | 0.084 | 1.6 | 0.017 | 0.98 | $-0.01$ | 0.60 | 0.93 |
| 21 | 0.1 | 61 | 0.52 | 1.3 | 0.123 | 0.47 | $-0.22$ | 0.54 | 0.89 |
| $1/t$ | 0 | 2552 | 0.079 | 1.5 | 0.010 | 0.99 | $0.00$ | 0.61 | 0.94 |

The $1/t$ schedule ends at $\kappa=0.051$ vs static $0.048$ —
consistency holds. A secondary $d=20$ unshrunk run:
$h=21$ $q_\perp$ corr with RV $=-0.05$; $h=252$ gives $0.50$. Same
pattern, smaller panel.

**What H4 got right.** Short $h$ inflates gauge-invariant $\kappa_t$
($22\times$ at $h=21$) and explodes the gauge itself
($\overline{E[Y]}_t=4772$). Direction cosine over 21 days drops to
$0.59$. The linear tracker stays uncorrelated (or anti-correlated)
with RV at every $h$.

**What H4 got wrong as a clock recipe.** Shrinkage $\tau=0.1$ at
$h=21$ pins the gauge ($\overline{E[Y]}=1.3$) and rescues $q_\perp$
(corr $0.54$) at the cost of turnover. It does not rescue $\hat Y$.
Long $h$ or $1/t$: $\kappa_t$ near the static value, stable
$w^\star$, and $q_\perp$ matches the static posterior as a vol proxy
(corr with 21-day RV $0.55$–$0.61$; with cross-sectional dispersion
$0.90$–$0.94$). Smoothing $q_\perp$ at half-life $5$ lifts RV-corr
to $\sim 0.71$–$0.75$.

**Fairness versus RV.** EWMA of $m_t^2$ at $h=21$ has corr $0.66$
with 21-day RV (same window family). Unshrunk $h=21$ $q_\perp$ loses
that comparison ($0.14$). The model's quadratic channel is
competitive only once $n_{\mathrm{eff}}$ is hundreds of days — once
the online model is close to the static fit.

Open problem (a) of the theory note, on this panel: the tracker plus
online EM is **not** a real-time activity index comparable to
realized variance. $q_\perp$ is, and it does not need $\gamma_t$.

## Worked / did not work

| Claim | Result |
|---|---|
| Tracker MSE $=E[Y]/\tilde q$ in simulation | Holds to $<1\%$. |
| Linear-Bayes MSE $=\mathrm{Var}(Y)/(1+\kappa)$ | Holds to $<1\%$. |
| $\mathrm{corr}(\hat Y,Y)=\sqrt{\kappa/(1+\kappa)}$ | Holds at $\kappa\ge 0.43$; $6.5\%$ high at the panel's own $\kappa\approx 0.05$. |
| Posterior mean tracks $Y$ at $d=50$ even if $\gamma=0$ | Holds (corr $0.95$ in simulation). |
| $\hat{\tilde q}$ is a usable point SNR | Fails. Upward-biased; at the panel's $\kappa$ the bias is the whole signal. Always report a matched null. |
| Fitted $\kappa\ll 1$ on daily equities (H1) | Holds. Indistinguishable from sign-flip. |
| Cross-section lifts $\kappa$ above the index (H2, first reading) | Point estimates do; the null does too, faster. |
| $\tilde q_d$ saturates on the market factor (H2, second reading) | Fails. Grows linearly; PC1 share *falls*. |
| Growing $\tilde q_d$ is recoverable idiosyncratic skewness | Fails. It is the $d$-dependent null. |
| Model max-skewness $=w^\star$ (H3, model) | Holds on every VG/NIG/GH fit, including the GIG check. |
| Sample skew maximiser recovers $w^\star$ (H3, sample) | Fails. Cosine $0.13$ / $-0.20$; OOS collapse. |
| Tracker has positive sample skew; index portfolios do not | Holds, and is the only listed portfolio that does. |
| Short online half-life inflates $\kappa_t$ and churns $w^\star$ (H4, noise) | Holds, including gauge explosion at $h=21$, $\tau=0$. |
| Short half-life extracts a better linear clock (H4, recipe) | Fails. $\hat Y$ uncorrelated with RV at every $h$. |
| $q_\perp$ / $E[Y\mid X]$ tracks realized vol | Holds, once $n_{\mathrm{eff}}$ is hundreds of days (corr $0.55$–$0.61$ with 21-day RV; $0.90$–$0.94$ with dispersion). |
| Online EM + tracker is a real-time activity index | Fails on this panel. |
| Proposition 2 tail bound is informative at fitted $\tilde q$ | Fails (vacuous). Location sign is consistent with no-near-arbitrage. |
| Extra GIG parameters (GH vs NIG) create a linear signal | Fails. $\kappa_{\mathrm{lev}}$ agrees; GH $\kappa$ is higher via $\mathrm{cv}^2$, not via the $a=b$ gauge $e=0.10$. |
| VG $\kappa$ smaller than NIG is leftover scale or a simulation bug | Fails as a diagnosis. $\kappa_{\mathrm{lev}}$ and $\gamma$-direction agree once the gauge is removed; the gap is the binding `alpha_min='density'` clamp ($\alpha=d/2+\varepsilon$). Unconstrained VG recovers NIG $\gamma$-energy and a closer $\kappa$, at the cost of an unbounded density. |

None of this is a failure of the mathematics. Fitted daily-equity
$\gamma$ is small, the sign-flip says it is consistent with zero, and
Proposition 1 then says no linear payoff can reveal $Y$. That is
partial revelation in the $\kappa=O(10^{-2})$ regime the
no-near-arbitrage heuristic predicted, taken all the way to "no
linear revelation."

Mencía & Sentana's skewness fund is, on this panel, a noisy
long–short with no clock content. The EM E-step's $E[Y\mid X]$
remains a usable latent-vol diagnostic, as the finance tutorials
already treat it.

## Limitations

Current-constituent panel (survivorship); large caps only; a single
common subordinator; i.i.d. GH (the ACF of $\hat Y$ being
$\approx 0$ is consistent with that, the ACF of $E[Y\mid X]$ is
not). Full-$\Sigma$ at $d=468$ has $T/d\approx 5.5$. Sign-flip
nulls at $d\ge 100$ use $B=20$. GH was a nested continuation, not a
cold start, and was not run at $d>50$. VG was $d\le 10$ only, and
the reported $\kappa$ uses `alpha_min='density'` (the unconstrained
MLE wants $\alpha<d/2$). VIX and high-frequency RV
were not used; panel-based proxies suffice for the mechanism
question. Multi-subordinator models and nonlinear payoffs (open
problems (b), (c) in {doc}`subordinator_tracking`) were out of
scope.

## What this is not asking `normix` to do

Report $\tilde q$ and $\kappa$ on fitted models only with a matched
null floor. A tracker-versus-Bayes notebook is still worth teaching
the channel split; the equity punchline is $q_\perp$, not $w^\star$.
No package change. The linear tracker is a clean theoretical object
that this panel does not support as a data-analysis tool.
