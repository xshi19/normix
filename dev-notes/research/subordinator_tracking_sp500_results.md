# Subordinator tracking on S&P 500 — empirical findings

**Published MyST:** `docs/research/subordinator_tracking_empirics.md`.
**Status:** Phases 0–3 done (2026-08-13). Report:
[`subordinator_tracking_report.md`](subordinator_tracking_report.md).
**Plan:** [`subordinator_tracking_sp500_plan.md`](subordinator_tracking_sp500_plan.md).
**Code:** `notebooks/subordinator_tracking/` (`lib.py`, `00_*.py`, `01_static_sp500.py`).
**Cache (gitignored):** `notebooks/subordinator_tracking/_cache/` (fits, tables, figures).

Generator: NIG, `regularization='a_eq_b'`, nested $d=50$ subset of
`data/sp500_returns.csv` (seed 0; 2552 days). EM `tol=1e-5` — the package
default $10^{-3}$ stops after one iteration because daily-equity $\gamma$ is
small enough that $\mathrm{rms}(\Delta\gamma)/(1+\mathrm{rms}(\gamma))$ already
meets it.

Fitted generator ($E[Y]=1$ gauge): $\tilde q = 0.0749$, $v=0.638$,
$\kappa=0.0478$, $\kappa_{\mathrm{lev}}=0.0749$, $t^\dagger=-0.638\le 1/\tilde q$
(NIG identity $e\mu_3=3v^2$). $\gamma$-scales $c\in\{0,1,3,10\}$ give
$\kappa\in\{0,\,0.0478,\,0.430,\,4.78\}$.

## Phase 0 — synthetic validation

$T=2552$, $d=50$, $R=20$ i.i.d. draws from the scaled generator. True
parameters (no refit) for the MSE/corr laws; cold-start EM for $\hat{\tilde q}$
and direction.

### MSE / correlation laws (true $\theta$)

| $c$ | $\kappa$ | MSE$_{\mathrm{trk}}$ relerr | MSE$_{\mathrm{lin}}$ relerr | $\mathrm{corr}(\hat Y,Y)$ relerr | MSE$_{\mathrm{post}}/(v/(1+\kappa))$ | $\mathrm{corr}(E[Y\mid X],Y)$ |
|---|---|---|---|---|---|---|
| 0 | 0 | — | 1.1% | — | 0.089 | 0.954 |
| 1 | 0.0478 | 0.63% | 0.84% | 6.5% | 0.095 | 0.954 |
| 3 | 0.430 | 0.74% | 0.10% | 1.6% | 0.125 | 0.956 |
| 10 | 4.78 | 0.19% | 0.06% | 0.05% | 0.336 | 0.970 |

Tracker MSE $e/\tilde q$ and linear-Bayes MSE $v/(1+\kappa)$ hold to $<1\%$.
The correlation law $\sqrt{\kappa/(1+\kappa)}$ is within 2% at $\kappa\ge 0.43$;
at the weakest point ($c=1$) the sample correlation is 0.228 vs 0.214
(6.5% relative — a small systematic, not Monte Carlo: 20×2552 observations).
The 5% acceptance flag fails only there.

The posterior mean is the story the linear tracker is not: even at $\kappa=0$
(no drift channel) $\mathrm{corr}(E[Y\mid X],Y)=0.95$ and MSE is $0.09\times$
the linear-Bayes bound. At $d=50$ the quadratic channel saturates. This is
exactly the note's §2 split, and it supports the prior that a *portfolio*
cannot dominate $Y$ on daily equities while $E[Y\mid X]$ still can.

### Estimation noise of $\hat{\tilde q}$ (cold-start EM)

Null floor at $c=0$, $T=2552$ ($R=20$):

| $d$ | mean $\hat{\tilde q}_0$ | 95% quantile | std |
|---|---|---|---|
| 10 | 0.0130 | 0.0206 | 0.0043 |
| 25 | 0.0261 | 0.0395 | 0.0091 |
| 50 | 0.0578 | 0.0775 | 0.0100 |

The floor grows with $d$. At $d=50$ the generator's own $\tilde q=0.0749$ sits
on the null 95% quantile: a full-history NIG fit on this panel is *not*
distinguishable from $\gamma=0$. Phase 1 should treat $\hat{\tilde q}$ as a
test against this floor, not as a point SNR.

Bias at $T=2552$, $d=50$:

| $c$ | true $\tilde q$ | mean $\hat{\tilde q}$ | mean $\cos\angle$ |
|---|---|---|---|
| 0 | 0 | 0.058 | — |
| 1 | 0.075 | 0.138 | 0.76 |
| 3 | 0.674 | 0.750 | 0.94 |
| 10 | 7.49 | 7.90 | 0.98 |

$\hat{\tilde q}$ is upward-biased; the bias is the whole signal at $c=1$.
Direction recovery is usable at $\kappa=0.05$ given $T=2552$ (cosine 0.76)
and essentially exact at $\kappa=4.8$. Cutting $T$ to 500 drops the $c=1$
cosine to 0.40.

### Online EM rehearsal (H4)

Path: $\gamma_t$ rotates by $\pi/4$ in the $\Sigma$-metric over $T=2552$;
IG scale jumps $\times 3$ at $t=T/2$. Oracle start at the true $t=0$ model
(no re-gauging). Filtered tracker vs true $Y$; EWMA realized variance of the
equal-weight mean as the RV baseline.

| $h$ | $\tau$ | $n_{\mathrm{eff}}$ | $\mathrm{corr}(\hat Y^{\mathrm{filt}},Y)$ | $\mathrm{corr}$ after $h_s=21$ | corr RV | mean $\kappa_t$ (true 0.096) | mean $\cos\angle$ | turnover |
|---|---|---|---|---|---|---|---|---|
| 21 | 0 | 61 | 0.32 | 0.46 | 0.47 | 1.36 | 0.18 | 0.084 |
| 21 | 0.1 | 61 | 0.14 | 0.09 | 0.47 | 0.50 | 0.47 | 0.139 |
| 63 | 0 | 182 | 0.26 | 0.30 | 0.48 | 0.64 | 0.31 | 0.054 |
| 252 | 0 | 727 | 0.30 | 0.38 | 0.46 | 0.23 | 0.57 | 0.024 |

H4 as stated is only half right.

- Short $h$ does inflate the *gauge-invariant* $\kappa_t$ (14× at $h=21$,
  2.4× at $h=252$) and raises turnover. $\tilde q_t$ itself is not comparable
  across $h$ without a gauge; $\kappa_t$ is the right diagnostic.
- Short $h$ does *not* recover the rotating direction — mean cosine is
  *worse* (0.18 vs 0.57). The rotation is slow ($\pi/4$ over ten years);
  estimation noise in $\hat\gamma_t$ dominates the tracking benefit.
  Shrinkage $\tau=0.1$ at $h=21$ buys cosine (0.47) and spends $Y$-tracking
  and turnover; it is not a free regulariser.
- Smoothed $h=21$ tracker matches EWMA RV (both $\approx 0.46$) and does not
  beat it. A frozen true-model *posterior* on the same path still has
  corr $0.96$ — again the quadratic channel, which does not need $\gamma_t$.

In-sample $\hat Y$ (using $\theta_t$ on $x_t$) has corr 0.63–0.89; that is
the same-day overfit the plan flagged. Filtered numbers above are the honest
ones.

### Phase 0 implications for Phases 1–3

1. **H1 is the default.** Fitted $\kappa\approx 0.05$ on the $d=50$ panel,
   and that $\tilde q$ is inside the $c=0$ 95% null. Expect the static
   full-history SNR table to say "no linear extraction."
2. **Report every $\hat{\tilde q}$ against the floor in the table above**
   (and its $T$-matched siblings). Phase 2's $d$-sweep will otherwise
   read noise growth as $\tilde q_d$ growth.
3. **The quadratic channel works at this $d$** even when the tracker does
   not. Phase 1 should still plot $E[Y\mid X]$ vs realized vol; that is
   the measurement the note's Bayes row predicts, not the portfolio.
4. **Online EM: prefer $h\sim 252$ for the direction, $h\sim 21$ plus
   post-hoc smoothing of $\hat Y$ for the clock.** Do not read $\tilde q_t$
   as SNR without a gauge; use $\kappa_t$. Shrinkage toward a static
   $\eta_0$ fights the jump — if used, shrink $\Sigma$ only (per-field
   $\tau$), not $\gamma$.
5. **EM `tol`:** keep $10^{-5}$ for this study.

### Gotchas found in the harness

- Calling `subordinator().raw_moments` (Bessel) on every fitted model
  recompiles `log_kv` and OOMs around ~150 EM refits. NIG path now uses
  closed-form IG moments (`mean`/`var`, $\mu_3=3v^2/e$).
- `IncrementalEMFitter` was not used (random batches); the loop is
  `e_step` → EWMA → `m_step`, JIT-ed, no `regularize_a_eq_b`.

## Phase 1 — static S&P 500

Nested seed-0 universes; NIG at all $d$; GH nested continuation from the
NIG embedding at $d\le 50$; VG with `alpha_min='density'` at $d\le 10$.
Sign-flip null: $B=50$ for $d\le 50$, $B=20$ for $d\in\{100,200,468\}$.
Hygiene: 12 days with $|r|>0.5$ (kept); no zero-variance columns.

### H1 — no linear extraction

| $d$ | $\hat{\tilde q}$ | null 95% | $p$ | $\hat\kappa$ | $\sqrt{\kappa/(1+\kappa)}$ | $1/\kappa$ |
|---|---|---|---|---|---|---|
| 5 | 0.0043 | 0.0098 | 0.41 | 0.0082 | 0.090 | 122 |
| 10 | 0.018 | 0.018 | 0.078 | 0.020 | 0.139 | 51 |
| 25 | 0.036 | 0.040 | 0.12 | 0.029 | 0.167 | 35 |
| 50 | 0.075 | 0.085 | 0.20 | 0.048 | 0.214 | 21 |
| 100 | 0.127 | 0.164 | 0.52 | 0.071 | 0.258 | 14 |
| 200 | 0.219 | 0.298 | 0.81 | 0.115 | 0.321 | 8.7 |
| 468 | 0.509 | 0.808 | 1.00 | 0.233 | 0.435 | 4.3 |

Equal-weight univariate NIG: $\kappa_{\mathrm{ew50}}=0.0093$,
$\kappa_{\mathrm{ew468}}=0.0125$. The $d=50$ panel $\hat\kappa=0.048$
exceeds the index but sits inside the sign-flip floor. $\hat{\tilde q}$
grows with $d$; the null grows faster. At $d=468$, $p=1$: the fit is
*less* skewed than a typical sign-flip. H1 holds.

GH $\kappa_{\mathrm{lev}}$ at $d=50$ is $0.069$ vs NIG $0.075$ (gauge-invariant
level SNR agrees). GH $\kappa=0.097$ is higher via $\mathrm{cv}^2=1.39$ vs
$0.64$, not via $e=0.10$ (`a_eq_b` on GH is $a=b$, not $E[Y]=1$). VG
$\kappa=0.009$ at $d=10$ is the binding `alpha_min='density'` clamp
($\alpha=5.1=d/2+\varepsilon$, $\mathrm{cv}^2=0.196$ vs NIG $1.11$);
unconstrained VG recovers $\kappa_{\mathrm{lev}}\approx$ NIG and
$\kappa=0.014$. Same-$d$ comparison, not $d=50$. Extra subordinator
flexibility does not create a linear signal. Details:
[`docs/research/subordinator_tracking_empirics.md`](../../docs/research/subordinator_tracking_empirics.md)
(family comparison).

### Tracker vs quadratic channel ($d=50$)

| | sample | model / sim |
|---|---|---|
| $\mathrm{Var}(\hat Y)$ | 14.67 | $v+e/\tilde q=13.99$ |
| $\mathrm{skew}(\hat Y)$ | 0.79 | 0.51 |
| ACF$_1$, ACF$_{21}$ of $\hat Y$ | 0.012, 0.005 | 0 |
| $\mathrm{corr}(\hat Y, E[Y\mid X])$ | 0.203 | 0.205 |
| $\mathrm{corr}(E[Y\mid X], q_\perp)$ | 0.989 | 0.998 |
| $\mathrm{corr}(E[Y\mid X], \mathrm{x\text{-}sec.\ disp.})$ | 0.933 | — |
| $\mathrm{corr}(E[Y\mid X], 21\mathrm{d\ RV})$ | 0.580 | — |
| $\mathrm{corr}(\hat Y, 21\mathrm{d\ RV})$ | 0.071 | — |

The linear tracker is consistent with the i.i.d. mixture (moments, ACF,
channel split) and is **uncorrelated with realized vol**. The posterior
mean is essentially $q_\perp$ and **is** the volatility clock. Phase 0's
quadratic-channel story on synthetic data repeats on the panel.

The ACF of $\hat Y$ being ~0 is a plan adjustment: i.i.d. misspecification
does *not* show up in the tracker. It shows up in $E[Y\mid X]$. Phase 3
should score the quadratic channel against RV, not expect a clustered
$\hat Y_t$.

### H3 — max-skewness

$t^\dagger < 0$ for every NIG and GH fit; $t^\dagger \approx 0$ for VG
(Gamma equality). Model max-skewness $= w^\star$ in all cases, including
the GIG conjecture on these four GH fits.

Sample third moments do not recover it. Direct skew maximisation
(20 L-BFGS starts):

| $d$ | in-sample max skew | tracker sample skew | $\cos\angle$ | OOS max | OOS tracker |
|---|---|---|---|---|---|
| 10 | 3.24 | 0.59 | 0.13 | $-0.023$ | 0.24 |
| 50 | 8.31 | 0.79 | $-0.20$ | 0.20 | $-0.047$ |

Equal-weight / min-var / PC1 all have $w^\top\gamma < 0$ and *negative*
sample skew (index leverage). The tracker is the only listed portfolio
with positive sample skew, but it is not the sample maximiser — third
moments overfit.

### Anatomy ($d=50$)

Unit-gross long/short, 24 long / 26 short, long share $0.45$, HHI $0.032$.
$\mathrm{corr}(w, \beta_{\mathrm{mkt}})=0.065$. $\mathrm{corr}(\hat Y, m_t)=-0.37$.
Location $(w^\star)^\top\mu=-0.987$; sample $E[P^\star]=0.013$, so
$E[\hat Y]=1=e$. The location term eats the $Y$-premium; raw tracker
return is near zero. Block-bootstrap cosine 5/50/95 =
$0.58/0.73/0.82$ — a moderately tight cone around a direction the
sign-flip says is odd-moment noise. Prop. 2 bound is vacuous
($\tilde q$ small: $P(\hat Y\le -1)=0.28$ vs bound $0.58$).
CVaR$_{5\%}$ long/short tracker $= 7.50 / 9.14$ on the $\hat Y$ scale
(std $\approx 3.8$). Mild asymmetry, Gaussian residual, not a GIG floor.

$\gamma = g\mathbf 1 + \delta$: $g=-8.5\cdot 10^{-4}$,
$\lVert\delta\rVert^2 = 2.1\cdot 10^{-5}$ vs $\lVert\gamma\rVert^2=5.7\cdot 10^{-5}$.
PC1 carries $5.6\%$ of $\tilde q$; the rest is spread. Not
market-aligned skewness — consistent with isotropic estimation noise.

### Phase 1 implications for 2–3

1. Treat $\hat{\tilde q}_d$ growth as a race against the null, not as
   diversification of $Y$. Phase 2 overlay is the sign-flip 95% curve
   above; 5-seed error bars on the fit.
2. Eigen-attribution should load on small $\lambda$ if the "signal" is
   $\gamma$-noise.
3. Phase 3: keep the $h$-grid as a *parameter*-tracking study. Add the
   filtered posterior / $q_\perp$ vs RV. Do not expect $\hat Y_t$ to
   be a vol index.

## Phase 2 — dimension sweep (H2)

5 nested seeds. $d=468$ is the full panel, so all seeds coincide.

| $d$ | mean $\hat{\tilde q}$ | se | sign-flip 95% | mean $\hat\kappa$ | mean $\kappa_{\mathrm{index}}$ | PC1 share | small-$\lambda$ 10% |
|---|---|---|---|---|---|---|---|
| 5 | 0.012 | 0.006 | 0.010 | 0.014 | 0.009 | 0.39 | 0.024 |
| 10 | 0.020 | 0.004 | 0.018 | 0.019 | 0.011 | 0.27 | 0.024 |
| 25 | 0.035 | 0.007 | 0.040 | 0.025 | 0.013 | 0.14 | 0.075 |
| 50 | 0.060 | 0.015 | 0.085 | 0.039 | 0.012 | 0.086 | 0.085 |
| 100 | 0.107 | 0.014 | 0.164 | 0.061 | 0.013 | 0.045 | 0.080 |
| 200 | 0.217 | 0.011 | 0.298 | 0.111 | 0.013 | 0.019 | 0.095 |
| 468 | 0.509 | 0 | 0.808 | 0.233 | 0.012 | 0.007 | 0.106 |

$\hat{\tilde q}_d$ grows, roughly linearly in $d$, and **does not saturate**.
That looks like the note's idiosyncratic-$\delta$ branch — except every
point sits at or below the sign-flip floor (the $d=5,10$ means slightly
exceed the *seed-0* null; they do not exceed a seed-matched null, and
PC1 shares at $d=5$ range $0.001$–$0.76$ across seeds). The index
ceiling $\kappa_{\mathrm{index}}\approx 0.01$ is flat in $d$.

Attribution is the opposite of market-aligned skewness: PC1's share of
$\tilde q$ *falls* from $0.39$ to $0.007$. The equicorrelation ceiling
$g^2/(\bar\sigma^2\rho)$ is $0.004$–$0.011$, the same order as
$\kappa_{\mathrm{index}}$. $\lVert\delta\rVert^2/\lVert\gamma\rVert^2$
rises $0.18\to 0.41$: the extra $\tilde q$ lives in $\delta$, which is
where $\gamma$-estimation noise lives.

**H2 verdict.** The hypothesis as stated is false in both of its
intended readings. There is no saturating market-skewness signal, and
the growing $\tilde q_d$ is not recoverable idiosyncratic skewness — it
is the $d$-dependent null. Cross-section does not buy a linear clock.

## Phase 3 — online EM on real data

Warm start: NIG on 2016–2017 (504 days), $d=50$ seed 0.
Warm-window $\tilde q=0.368$, $\kappa=0.182$ — higher than the
full-sample $0.075/0.048$. Online period: 2017-12-14 → 2026-02-09
($T=2048$). No re-gauging.

| $h$ | $\tau$ | $n_{\mathrm{eff}}$ | mean $\kappa_t$ | mean $e_t$ | turnover | $\cos_{21}$ | corr$(\hat Y,\mathrm{RV})$ | corr$(q_\perp,\mathrm{RV})$ | corr$(q_\perp,\mathrm{disp})$ |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 0 | 61 | 1.06 | 4772 | 0.084 | 0.59 | $-0.14$ | 0.14 | 0.61 |
| 63 | 0 | 182 | 0.41 | 27 | 0.055 | 0.79 | $-0.04$ | 0.01 | 0.44 |
| 126 | 0 | 364 | 0.19 | 3.3 | 0.039 | 0.88 | $-0.05$ | 0.37 | 0.80 |
| 252 | 0 | 727 | 0.11 | 1.9 | 0.027 | 0.94 | $-0.03$ | 0.55 | 0.90 |
| 504 | 0 | 1454 | 0.084 | 1.6 | 0.017 | 0.98 | $-0.01$ | 0.60 | 0.93 |
| 21 | 0.1 | 61 | 0.52 | 1.3 | 0.123 | 0.47 | $-0.22$ | 0.54 | 0.89 |
| $1/t$ | 0 | 2552 | 0.079 | 1.5 | 0.010 | 0.99 | $0.00$ | 0.61 | 0.94 |

$1/t$ terminal $\kappa=0.051$ vs static $0.048$ — consistency check holds.
$d=20$ unshrunk: $h=21$ $q_\perp$ corr with RV $=-0.05$; $h=252$ gives $0.50$.

H4 is right about noise, wrong as a recipe for the linear clock.

- Short $h$ inflates gauge-invariant $\kappa_t$ ($22\times$ at $h=21$) and
  explodes the gauge itself ($\bar e_t=4772$). Direction cosine over 21
  days drops to $0.59$. The linear tracker stays uncorrelated (or
  anti-correlated) with RV at every $h$.
- Shrinkage $\tau=0.1$ at $h=21$ pins the gauge ($\bar e=1.3$) and
  rescues $q_\perp$ (corr $0.54$) at the cost of turnover. It does not
  rescue $\hat Y$.
- Long $h$ or $1/t$: $\kappa_t$ near the static value, stable $w^\star$,
  and $q_\perp$ matches the static posterior as a vol proxy
  (corr with 21-day RV $0.55$–$0.61$; with cross-sectional dispersion
  $0.90$–$0.94$). Smoothing $q_\perp$ at $h_s=5$ lifts RV-corr to
  $\sim 0.71$–$0.75$.
- Fairness vs RV: EWMA of $m_t^2$ at $h=21$ has corr $0.66$ with 21-day
  RV (same window family). Unshrunk $h=21$ $q_\perp$ loses that
  comparison ($0.14$). The model's quadratic channel is competitive
  only once $n_{\mathrm{eff}}$ is hundreds of days — i.e. once it is
  close to the static fit.

Open problem (a) of the theory note, on this panel: the tracker plus
online EM is **not** a real-time activity index comparable to realized
variance. The posterior / $q_\perp$ is, and it does not need $\gamma_t$.

## Promotion candidates

- $\tilde q/\kappa$ fitted-model diagnostics **only** with a matched
  null floor (sign-flip or $c=0$ synthetic). A bare $\tilde q$ will be
  misread as SNR.
- Tracker-vs-Bayes tutorial: the punchline on daily equities is the
  quadratic channel, not the portfolio.
- No package change. The linear tracker is a clean theoretical object
  that this panel does not support as a data-analysis tool.
