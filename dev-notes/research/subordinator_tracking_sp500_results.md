# Subordinator tracking on S&P 500 — empirical findings

**Status:** Phase 0 done (2026-08-12). Phases 1–3 not started.
**Plan:** [`subordinator_tracking_sp500_plan.md`](subordinator_tracking_sp500_plan.md).
**Code:** `notebooks/subordinator_tracking/` (`lib.py`, `00_synthetic_validation.py`).
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

## Phase 1 — static S&P 500 (not started)

## Phase 2 — dimension sweep (not started)

## Phase 3 — online EM on real data (not started)

## Promotion candidates

Deferred until Phase 1 exists. Phase 0 already suggests $\tilde q/\kappa$
diagnostics are only useful with a null floor attached.
