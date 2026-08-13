# Empirical plan: subordinator tracking on the S&P 500 panel

**Status:** plan, 2026-08-13. Phases 0–1 done; Phase 2 next.
Findings in [`subordinator_tracking_sp500_results.md`](subordinator_tracking_sp500_results.md).
**Theory:** [`subordinator_tracking_portfolio.md`](subordinator_tracking_portfolio.md);
its notation is reused without redefinition: $e = E[Y]$, $v = \operatorname{Var}(Y)$,
$\mu_3 = E[(Y-e)^3]$, $\tilde q = \gamma^\top\Sigma^{-1}\gamma$,
$\kappa_{\mathrm{lev}} = \tilde q\,e$, $\kappa = \tilde q\,v/e$,
$w^\star = \Sigma^{-1}\gamma/\tilde q$, $\hat Y = (w^\star)^\top(X-\mu)$,
$t^\dagger = 2v/e - \mu_3/v$.
**Constraint:** the normix package is **not modified**. All research code lives in
`notebooks/` (jupytext percent files + one shared plain-Python helper module),
using only the public API. Anything worth promoting into `normix.finance` is a
*deliverable of the findings note*, decided afterwards.

## 1. Questions and hypotheses

Two study modes, as posed:

1. **Static** — fit one distribution to the full history, run the SNR analysis
   ($\tilde q$, $\kappa_{\mathrm{lev}}$, $\kappa$), and check the max-skewness
   portfolio (Phases 1–2).
2. **Dynamic** — update the model daily with EWMA-weighted online EM and
   rebalance $w^\star_t$ along the way, studying how the EM's responsiveness
   affects the extracted volatility path (Phase 3). Evaluation is in-sample by
   design: the goal is extraction of the latent clock from existing data, not
   prediction.

Pre-registered hypotheses, with a back-of-envelope prior for H1:

- **H1 (partial revelation).** Fitted $\kappa \ll 1$: daily equity portfolios are
  dominated by the Gaussian residual, not the skewness term. Univariate
  calibration in the gauge $e = 1$: for small $\gamma$,
  $\operatorname{skew}(X) \approx 3\gamma v/\sigma$ and excess
  $\operatorname{kurt}(X) \approx 3v$, so

  $$
  \kappa = \tilde q\,v = \frac{\operatorname{skew}(X)^2}{9v}
  = \frac{\operatorname{skew}(X)^2}{3\,\operatorname{kurt}(X)}.
  $$

  Daily index returns with $\operatorname{skew} \approx -0.5$ and excess
  kurtosis $5$–$15$ give $\kappa \sim 10^{-2}$, i.e. per-day tracker correlation
  $\sqrt{\kappa/(1+\kappa)} \approx 0.1$ and $\kappa_{\mathrm{lev}} \sim 10^{-2}$:
  levels overwhelmingly noise-dominated. The open question is how much the
  cross-section adds.
- **H2 (saturation).** $\tilde q_d$ saturates as $d$ grows because fitted
  $\gamma$ aligns with the market covariance factor (note §6). Under
  market-only skewness ($\delta = 0$) the $d\to\infty$ tracker is the index
  itself, so the ceiling is $\kappa_\infty \approx \kappa_{\mathrm{index}}$;
  any excess must come from idiosyncratic skewness dispersion
  $\lVert\delta\rVert^2$. Measuring that excess is the Phase 2 deliverable.
- **H3 (max-skew = tracker).** For fitted VG/NIG, $t^\dagger \le 0 \le 1/\tilde q$
  analytically, so the model's max-skewness portfolio is exactly $w^\star$; for
  fitted GH the inequality $t^\dagger \le 1/\tilde q$ is checked numerically.
  On sample moments, the direct skewness maximiser should point near
  $\hat\Sigma^{-1}\hat\gamma$ up to (large) estimation noise.
- **H4 (responsiveness trade-off).** Short EWMA half-lives track volatility
  regimes faster but inject estimation noise into $\hat\gamma_t$ — direction
  churn, turnover, and upward-biased $\hat{\tilde q}_t$ — while long half-lives
  approach the static fit. There is an interior optimum tied to the
  persistence of $Y_t$. Rough scaling: per-day $\kappa \sim 10^{-2}$, but the
  tracker's noise is i.i.d. across days while $Y_t$ is persistent, so smoothing
  over $n_{\mathrm{eff}}$ days lifts the effective SNR to
  $\approx n_{\mathrm{eff}}\,\kappa\,(\text{signal retention})$ — order 1 for a
  one-month window. Verified on synthetic data in Phase 0 before being trusted
  on real data.

## 2. Data

`data/sp500_returns.csv`: daily log returns, 2015-12-15 → 2026-02-09
($T = 2552$ days), $d = 468$ current constituents with near-complete history
(see `scripts/download_sp500_data.py`). Covers the 2018 Volmageddon, the
2020-03 COVID crash, and the 2022 hiking cycle — good variation in the latent
clock.

- **Universes.** Nested random subsets $d \in \{5, 10, 25, 50, 100, 200, 468\}$,
  5 seeds each (nesting makes $\tilde q_d$ monotone comparisons meaningful per
  seed). Primary working universe: $d = 50$, seed 0.
- **Market proxy.** Equal-weight panel mean $m_t = \tfrac1d\sum_i x_{t,i}$;
  univariate fits of $m_t$ give the index baseline $\kappa_{\mathrm{index}}$.
- **Hygiene.** No winsorisation (tails are the signal). Flag $|r| > 0.5$
  returns and zero-variance columns for manual inspection; drop nothing
  silently.
- **Known biases.** Current-constituent panel → survivorship bias; all large
  caps. Acceptable for a mechanism study; stated in the limitations section of
  the findings note.

## 3. Model and estimation conventions

- **Primary model: NIG** (`NormalInverseGaussian`). No unbounded-likelihood
  boundary (unlike VG, whose density requires the Gamma shape
  $\alpha > d/2$ — untenable beyond small $d$), closed-form IG M-step (no
  Bessel solve), and `regularization='a_eq_b'` pins $\mu_{\mathrm{IG}} = 1$,
  i.e. exactly the identifiability gauge $E[Y] = 1$, so
  $\kappa_{\mathrm{lev}} = \tilde q$ and $\kappa = \tilde q\,v$ directly.
- **Secondary: GH** (`GeneralizedHyperbolic`, `regularization='a_eq_b'`) at
  $d \le 50$ (plan originally said $d \le 100$; GH M-step is a GIG solve, so
  $d=100$ is deferred unless the $d=50$ continuation is cheap). Initialisation
  is the fitted NIG's exact GH embedding (`to_generalized_hyperbolic`), then
  free $(p,a,b)$ — nested continuation, not `default_init`. **VG** only at
  $d \le 10$, with `alpha_min='density'`.
- **Fitting.** `BatchEMFitter` per the finance tutorials
  (`docs/tutorials/finance/02_multivariate_stocks.md`); `default_init` cold
  start; `e_step_backend='cpu'` where profiling favours it at large $T\,d$.
  Use `tol=1e-5` (not the package default $10^{-3}$): daily-equity $\gamma$ is
  small enough that one EM step already meets $10^{-3}$. $\kappa$ and
  $\kappa_{\mathrm{lev}}$ are gauge-invariant, so regularisation
  affects them not at all; $\tilde q$ is quoted in the $E[Y] = 1$ gauge that
  `'a_eq_b'` pins for NIG.
- **Statistical error.** Two resampling tools, reused across phases:
  - *Sign-flip null* for $\tilde q$: $x_t \mapsto \bar x + S_t\,(x_t - \bar x)$
    with $\bar x$ the sample mean and i.i.d. $S_t = \pm 1$ per **day** (one
    sign for the whole cross-section). Kills all odd joint moments, preserves $\Sigma$ and the
    mixing structure; refitting gives the null distribution of
    $\hat{\tilde q} > 0$ produced by pure estimation noise. This is the
    yardstick for "is the skewness signal real", and (Phase 2) the noise floor
    that mechanically grows with $d$.
  - *Moving-block bootstrap* (21-day blocks) for confidence cones on the
    direction $w^\star$ and on $\hat\kappa$, respecting volatility clustering.

## 4. Derived quantities (implementation formulas)

Everything below uses public API only; no reimplementation of model math.

| Quantity | Computation |
|---|---|
| $\Sigma^{-1}\gamma$, $\tilde q$ | two triangular solves against `model.L_Sigma`; $\tilde q = \gamma^\top(\Sigma^{-1}\gamma)$ |
| $w^\star$ | $\Sigma^{-1}\gamma/\tilde q$ (gauge $w^{\star\top}\gamma = 1$); unit-gross rescale for turnover/weights reporting |
| tracker $\hat Y_t$ | $(w^\star)^\top(x_t - \hat\mu)$ |
| $e, v, \mu_3$ | `model.subordinator().raw_moments([1,2,3])` |
| $\kappa_{\mathrm{lev}}, \kappa, t^\dagger$ | from the above |
| linear Bayes | $e + \tfrac{\kappa}{1+\kappa}(\hat Y_t - e)$ |
| posterior mean $E[Y\mid x_t]$ | `jax.vmap(model.joint.conditional_expectations)(X)["E_Y"]` |
| quadratic split | $q(x_t)$ Mahalanobis via `L_Sigma`; $q_\perp = q - \tilde q\,\hat Y_t^2$ |
| model skew of $P_w$ | `model.project(w).skewness()`; the $t$-curve $(\mu_3 + 3tv)/(v+te)^{3/2}$ for figures |
| left-tail bound (Prop 2) | $P(P^\star - \text{loc} \le -c) \le 2\Phi(-2\sqrt{c\tilde q})$ vs empirical tail |
| synthetic truth | `model.joint.rvs(n, seed)` returns $(X, Y)$ |

Volatility proxies on real data (where $Y_t$ is latent): centered 21-day
realized variance of $m_t$; EWMA realized variance at matched half-life;
cross-sectional dispersion $\tfrac1d\sum_i (x_{t,i}-m_t)^2/\hat\sigma_i^2$
(a model-free cousin of the quadratic channel). Correlations reported as
Pearson and Spearman, on levels and logs.

## 5. Phase 0 — synthetic validation (truth available)

Calibrate the pipeline where $Y_t$ is observable, so real-data numbers can be
interpreted. Generator: the Phase 1 NIG fit at $d = 50$ (realistic regime),
with $\gamma$ scaled by $c \in \{0, 1, 3, 10\}$ to sweep
$\kappa \in \{0, \sim 10^{-2}, \sim 10^{-1}, \sim 1\}$; $T = 2552$ to match;
$\ge 20$ replications.

Checks, each against its closed-form law:

1. **MSE laws.** Tracker $e/\tilde q$; linear Bayes $v/(1+\kappa)$; posterior
   mean $\le v/(1+\kappa)$ (note §3 table). Acceptance: within Monte Carlo
   error (~5% relative at this design).
2. **Correlation law.** $\operatorname{corr}(\hat Y, Y) = \sqrt{\kappa/(1+\kappa)}$.
3. **Estimation noise of $\hat{\tilde q}$.** Refit on finite samples across
   $(d, T)$; measure the null floor $\hat{\tilde q}_{\,0}(d, T)$ at $c = 0$ and
   the bias at $c > 0$. This curve is reused as the Phase 1 significance
   yardstick and the Phase 2 noise-floor overlay.
4. **Direction recovery.** $\cos\angle(\hat w^\star, w^\star)$ vs $\kappa$ and $T$.
5. **Online EM rehearsal.** Time-varying synthetic truth (regime switch in the
   subordinator scale and a slow rotation of $\gamma_t$); run the Phase 3 loop
   and verify the responsiveness trade-off (H4) is visible and the smoothing
   scaling argument holds, before applying either to real data.

## 6. Phase 1 — static full-history study (the "simplest way")

One NIG (and GH) fit per universe on all 2552 days.

1. **Headline SNR table** — per model × $d$: $\tilde q$, $e$, $v$, $v/e^2$,
   $\kappa_{\mathrm{lev}}$, $\kappa$, $\sqrt{\kappa/(1+\kappa)}$, tracker
   relative MSE $1/\kappa$, sign-flip null 95% quantile of $\hat{\tilde q}$ and
   the implied p-value. This is the direct test of H1.
2. **Tracker time series.** $\hat Y_t$, linear Bayes, $E[Y\mid x_t]$, and the
   volatility proxies on one axis (COVID zoom inset). Moment consistency:
   sample $\operatorname{Var}(\hat Y_t)$ vs $v + e/\tilde q$; sample skew of
   $\hat Y_t$ vs the model value; sample ACF of $\hat Y_t$ (the i.i.d. model
   says zero — the observed clustering quantifies the misspecification that
   Phase 3 addresses).
3. **Channel comparison.** $\operatorname{corr}(\hat Y_t, E[Y\mid x_t])$ and
   $\operatorname{corr}(\hat Y_t, q_\perp/(d-1))$ vs their model-implied
   values — the orthogonal decomposition makes the second pair's noises
   independent, so the product law is testable.
4. **Max-skewness portfolio (H3).**
   - $t^\dagger$ vs $1/\tilde q$ per fitted model (analytic for VG/NIG,
     numeric for GH — also a data point for the note's GIG cumulant
     conjecture).
   - Skew placement figure: the model curve
     $\operatorname{skew}(t)$ with portfolios placed on it ($w^\star$,
     equal-weight, min-variance, PC1, random directions, best single names) —
     model-implied vs sample skew per portfolio (third-moment goodness of
     fit).
   - Direct sample-skew maximisation at $d \in \{10, 50\}$ (autodiff ascent on
     $w \mapsto \widehat{\operatorname{skew}}(w^\top x)$, ~20 starts):
     cosine similarity to $\hat\Sigma^{-1}\hat\gamma$, plus a split-sample
     honesty check (maximise on the first half, evaluate skew on the second;
     compare against the tracker direction estimated on the first half). The
     sample maximiser will overfit — the split quantifies by how much, and the
     sign-flip null gives its overfitting yardstick.
5. **Portfolio anatomy.** $w^\star$ weights vs market beta, gross/net exposure,
   top names, correlation of tracker returns with $m_t$; realized mean vs
   location $(w^\star)^\top\hat\mu$ (the no-near-arbitrage sign check from
   note §4). Left-tail calibration plot against the Gaussian-thin bound.

## 7. Phase 2 — dimension sweep: does $\tilde q_d$ saturate? (H2)

1. $\hat{\tilde q}_d$ and $\hat\kappa_d$ vs $d$ over the nested universes
   (5 seeds → error bars), **with the Phase 1 sign-flip 95% overlay**
   (and the Phase 0 synthetic $c=0$ floor at $d\le 50$). Estimation noise
   grows with $d$; Phase 1 seed 0 already has $\hat{\tilde q}$ *below* the
   null at $d\ge 100$. Fake growth must be subtracted before calling the
   curve saturating or not.
2. **Ceiling comparison.** Univariate $\kappa_{\mathrm{index}}$ from the
   equal-weight market fit = the $\delta = 0$ ceiling; the gap
   $\hat\kappa_d - \hat\kappa_{\mathrm{index}}$ is the cross-sectional gain.
3. **Attribution.** Eigen-decompose $\hat\Sigma = \sum_k \lambda_k u_k u_k^\top$
   and stack $\tilde q = \sum_k (u_k^\top\hat\gamma)^2/\lambda_k$ by component:
   does the market PC carry the skewness (→ saturation) or do the small-
   $\lambda$ directions contribute (→ idiosyncratic dispersion,
   $\tilde q_d$ growth)? Equicorrelation-style split
   $\hat\gamma = \hat g\mathbf 1 + \hat\delta$ reported alongside.

## 8. Phase 3 — online EM dynamic tracker (the "advanced way")

Chronological loop, hand-rolled from public API (`IncrementalEMFitter` samples
random batches, so it is not reused; the rules and eta pytrees are):

```
warm start: batch fit on days 1..504 (2016–2017); eta = model.compute_eta_from_model()
for each day t in 505..T:
    eta_hat = model.e_step(x_t[None, :])                    # one-day batch
    eta     = EWMAUpdate(w)(eta, eta_hat, ...)               # (1-w)·eta + w·eta_hat
    model   = model.m_step(eta)                              # online EM: theta_t = ∇φ(eta_t)
    record q̃_t, κ_t, w*_t, Ŷ_t (from model_t: in-sample; from model_{t-1}: filtered)
```

This is Cappé–Moulines online EM with a constant step
(`docs/theory/online_em.md`) — the EWMA regime that tracks slowly-varying
parameters rather than converging.

- **Half-life grid.** $h \in \{21, 63, 126, 252, 504\}$ days,
  $w = 1 - 2^{-1/h}$. Consistency check: swapping `EWMAUpdate` for
  `SampleWeightedUpdate` (the $1/t$ schedule, which the regret analysis in
  `docs/theory/online_em.md` ties to the penalised MLE) should land near the
  Phase 1 static fit after the full pass. EWMA effective sample size is
  $n_{\mathrm{eff}} = (2-w)/w \approx 2.9\,h$; a full $\Sigma$ needs
  $n_{\mathrm{eff}} \gtrsim 3d$, so at $d = 50$ half-lives below ~63 days
  require `Shrinkage(EWMAUpdate(w), eta0, tau)` with $\tau > 0$ toward the
  static-fit `eta0_from_model` target to keep $\hat\Sigma_t$ conditioned;
  elsewhere $\tau \in \{0, 0.1\}$ is a sensitivity axis. Secondary universe
  $d = 20$ runs unshrunk everywhere.
- **Two smoothing knobs, kept separate.** $h$ smooths the *parameters*
  (how fast the model adapts); a post-hoc EWMA with half-life $h_s \in \{1, 5, 21\}$
  smooths the extracted *state* $\hat Y_t$. Conflating them would confuse
  parameter drift with state noise; the responsiveness question (H4) is about
  $h$, reported at each $h_s$.
- **Metrics vs $h$** (the responsiveness frontier):
  1. parameter paths $\tilde q_t$, $\kappa_t$ and direction stability —
     $\cos\angle(w^\star_t, w^\star_{t-21})$, daily unit-gross turnover;
  2. extraction quality — corr of (smoothed) $\hat Y_t$ **and of the
     filtered posterior / $q_\perp$** with the realized proxies of §4,
     benchmarked against plain EWMA realized variance *at the same
     half-life*. Phase 1: static $\hat Y$ has ACF $\approx 0$ and
     $\mathrm{corr}$ with 21-day RV $=0.07$; $E[Y\mid X]$ has corr $0.58$.
     The linear tracker is not the clock; the quadratic channel is.
  3. skewness capture — sample skew of the rebalanced tracker return series
     $\{w^{\star\top}_t x_t\}$ (and the filtered variant) vs $h$, against the
     static Phase 1 value; descriptive financial stats (mean vs location,
     vol, drawdowns) for the short-vol interpretation;
  4. noise bias — $\hat{\tilde q}_t$ against the sign-flip null at matched
     $n_{\mathrm{eff}}$ (short $h$ should show visible inflation, per H4).
- **Event reads.** 2018-02, 2020-03, 2022: do $\kappa_t$ spike, does
  $w^\star_t$ rotate, and how fast does each $h$ recover?

## 9. Code layout, caching, compute

```
notebooks/subordinator_tracking/
├── lib.py                     # tracker/κ math, universes, sign-flip null, block
│                              # bootstrap, online-EM loop, proxies, npz caching
├── 00_synthetic_validation.py # Phase 0   (jupytext percent, paired .ipynb gitignored)
├── 01_static_sp500.py         # Phase 1
├── 02_dimension_sweep.py      # Phase 2
├── 03_online_em.py            # Phase 3
└── _cache/                    # fitted models / sweep results (.npz), gitignored
```

Two `.gitignore` lines accompany this layout (`notebooks/**/*.ipynb`,
`notebooks/subordinator_tracking/_cache/`) — repo config, not package code.
Long jobs (sign-flip refits, the $d = 468$ fits, the $h$-grid) cache to
`_cache/` keyed by a config hash so notebooks re-run in minutes. Rough budget
(CPU): batch NIG fit at $d = 50$ seconds, at $d = 468$ minutes; sign-flip
$B = 100$ at $d \le 50$ ~minutes; Phase 3 grid (5 half-lives × ~2050 days,
closed-form NIG M-step) ~minutes per config. GH variants cost more via the GIG
M-step solve; run them once, cached.

## 10. Deliverables and order

1. `00` → `01` → `02` → `03`, in that order (Phase 0's noise floor and
   rehearsal are inputs to the others; the $d = 50$ real-data fit that seeds
   Phase 0's generator is computed once and shared with Phase 1 via the
   cache). No hard gates: even
   $\hat{\tilde q} \approx$ null everywhere is a clean answer to H1 ("no
   linear extraction; quadratic channel only"), and Phase 3 remains
   informative as a parameter-tracking study.
2. Findings land in `subordinator_tracking_sp500_results.md` (headline κ
   table, saturation verdict with attribution, responsiveness frontier,
   H1–H4 verdicts, limitations), written as each phase completes.
3. A short "promotion candidates" list at the end of the findings note —
   e.g. $\tilde q/\kappa$ fitted-model diagnostics, a tracker-vs-Bayes
   tutorial — for a separate decision (architect workflow) before anything
   touches `normix/`.

## 11. Out of scope

Package changes; trading backtests with costs (the return-series stats are
descriptive); external data (VIX comparison optional later — the panel-based
proxies suffice); multi-subordinator models and nonlinear payoffs (note §7
open problems (b), (c)); the GIG cumulant-inequality sweep, except the free
per-fit $t^\dagger$ check in Phase 1.

## 12. Risks and gotchas

- **$\gamma$ is the weakest-identified parameter** (note §7); $(\mu, \gamma)$
  trade off along a ridge in-sample. Always report direction cones (block
  bootstrap), never a bare $w^\star$.
- **Estimation noise inflates $\hat{\tilde q}$** with $d$ and with short
  half-lives. Every $\tilde q$ number in Phases 1–3 is reported against the
  matched null floor; Phase 2 conclusions are drawn only net of it.
- **Gauge drift in the online loop.** The E-step statistics are expressed in
  the current model's $Y$-gauge; re-gauging (`regularize_a_eq_b`) every step
  would mix gauges inside the EWMA η recursion. Run the raw chain and report
  gauge-invariant quantities ($\tilde q_t$ is not; $\kappa_t$,
  $\kappa_{\mathrm{lev},t}$ are); apply the $E[Y] = 1$ gauge only when
  plotting $Y$-level paths.
- **VG boundary**: density unbounded for Gamma shape $\alpha \le d/2$ — VG is
  a small-$d$ curiosity here, not a workhorse.
- **i.i.d. misspecification** is a feature to measure (ACF of $\hat Y_t$), not
  an error to hide; it is the bridge from Phase 1 to Phase 3.
- **Full-$\Sigma$ conditioning**: $T/d \approx 5.5$ at $d = 468$; those fits
  are stress cases for the sweep, not the basis of headline claims. A
  `FactorNormalMixture` variant (γ stabilised by factor structure, note §7)
  is a natural follow-up if the sweep says idiosyncratic dispersion matters —
  but online EM does not apply to the factor family
  (`docs/theory/online_em.md`, curved-family limitation), so it stays a
  static-study extension.
