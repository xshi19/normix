# Gauges for normal mean–variance mixtures

**Status: study plan, 2026-08-19.** Companion to
{doc}`subordinator_tracking_empirics`, whose VG/NIG/GH $\kappa$
comparison motivated it. Planned code: `notebooks/gauges/` (public
API only; no package change is presumed).

Every normal mean–variance mixture

$$
X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}\,Z,
\qquad Z\sim N(0,\Sigma)\perp Y\ge 0,
$$

carries a one-parameter family of observationally equivalent
parametrizations: for $s>0$,

$$
(\mu,\;\gamma,\;\Sigma,\;Y)\;\longmapsto\;
(\mu,\;\gamma/s,\;\Sigma/s,\;sY)
$$

leaves the law of $X$ unchanged. Fitted $(\gamma,\Sigma)$ and the
subordinator moments $E[Y]$, $\mathrm{Var}(Y)$ are therefore quoted
in a *gauge* — a rule that picks one representative per orbit. This
page asks which gauges make two comparisons meaningful:

1. **Distribution-invariant.** Fits of GH, VG, NIG, NInvG to the
   *same data* should display $\mu$, $\gamma$, $\Sigma$ on the same
   scale.
2. **Dimension-invariant.** A fit at $d=100$ should display $E[Y]$,
   $\mathrm{Var}(Y)$ on the same scale as a fit to a $d=50$ column
   subsample — and as the ensemble of $100$ univariate ($d=1$) fits,
   with the multivariate estimate sitting inside the ensemble spread
   rather than on a different scale.

It then separates two places a gauge can act: *inside EM* (re-gauge
each iteration, as `regularization=` does now) versus *post
estimation* (fit in any gauge, re-gauge once for display), states
hypotheses about when the choice affects estimation at all, and
pre-registers a numerical plan.

## 1. The orbit and its invariants

$\mu$ is untouched by the orbit: location is identified, and every
gauge below reports the same $\mu$. The one-dimensional freedom
lives in $(\gamma, \Sigma, Y)$. A maximal invariant is

$$
\bar\gamma = E[Y]\,\gamma,
\qquad
\bar\Sigma = E[Y]\,\Sigma,
\qquad
\bar Y = Y / E[Y],
$$

i.e. the parameters of the representative with $E[\bar Y]=1$,
together with the *shape* of the normalized subordinator. The
observable moments are functions of invariants only:

$$
E[X] = \mu + \bar\gamma,
\qquad
\mathrm{Cov}(X) = \bar\Sigma
  + \mathrm{cv}^2\,\bar\gamma\bar\gamma^\top,
\qquad
\mathrm{cv}^2 = \frac{\mathrm{Var}(Y)}{E[Y]^2} = \mathrm{Var}(\bar Y),
$$

and the tracking-page SNRs are
$\kappa_{\mathrm{lev}}=\bar\gamma^\top\bar\Sigma^{-1}\bar\gamma$ and
$\kappa=\kappa_{\mathrm{lev}}\,\mathrm{cv}^2$.

A gauge is a *section*: a scale functional $T(\theta)>0$ with
$T(g_s\theta) = s\,T(\theta)$, applied as $s = 1/T(\theta)$ so that
the representative satisfies $T=1$. Candidates, existing and
proposed:

| Gauge | Condition | Reads off | Exists |
|---|---|---|---|
| `'a_eq_b'` | $a=b$ on the GIG | subordinator coordinates | GH, NIG (no-op for VG, NInvG) |
| `'det_sigma_one'` | $\lvert\Sigma\rvert=1$ | $\Sigma$ | always |
| `'det_sigma_x'` | $\log\lvert\Sigma\rvert$ = initial model's | $\Sigma$ | always |
| $E[Y]=1$ (proposed; `pin_mean_y`) | $E[Y]=1$ | subordinator law | needs $E[Y]<\infty$ (NInvG: $\alpha>1$) |
| geometric (proposed) | $\exp E[\log Y]=1$ | subordinator law | always |
| median (proposed) | $\mathrm{med}(Y)=1$ | subordinator law | always |
| trace (considered) | $\mathrm{tr}(\Sigma)=d$ | $\Sigma$ | always |

Call a section *$Y$-side* if its condition involves only the law of
$Y$ ($a=b$, $E[Y]=1$, geometric, median) and *$\Sigma$-side* if it
involves $\Sigma$ (both determinant modes, trace).

## 2. Which sections satisfy which invariance

### 2.1 Distribution invariance

Fits of two families to the same data land on nearby laws of $X$
but on *different orbits in different parameter spaces*. The gauged
$(\gamma,\Sigma)$ are functions of the fitted invariants
$(\bar\gamma, \bar\Sigma, \mathrm{cv}^2)$; two families display
comparable parameters exactly when the section is pinned by
functionals of the $X$-law that both families are forced to match.

$E[X]$ and $\mathrm{Cov}(X)$ are matched by every reasonable fit,
and they identify $\bar\gamma$ (through the skewness direction and
$E[X]-\mu$) and $\bar\Sigma = \mathrm{Cov}(X) -
\mathrm{cv}^2\bar\gamma\bar\gamma^\top$ up to a correction that is
second order in $\bar\gamma$ (negligible on equity panels where
$\kappa_{\mathrm{lev}}\sim 10^{-2}$). Hence:

- **$E[Y]=1$: distribution-invariant.** The displayed
  $(\bar\gamma,\bar\Sigma)$ are the moment-pinned invariants
  themselves. There is a second, EM-specific reason it is the
  natural $Y$-side choice: at an EM fixed point each family matches
  the *prior* subordinator moments in its sufficient set to the
  posterior averages, and $E[Y]$ is in that set for VG
  ($E[\log Y], E[Y]$), NIG ($E[Y], E[1/Y]$), and GH (all three).
  Only NInvG ($E[1/Y], E[\log Y]$) treats $E[Y]$ as an
  extrapolation — and its $E[Y]=\beta/(\alpha-1)$ may not exist
  ($\alpha\le 1$; `normix` floors the denominator at
  `ALPHA_MOMENT_MARGIN`, so a pinned "$E[Y]$" would silently use
  the floored surrogate). The geometric gauge $\exp E[\log Y]=1$
  is the mirror image: matched for VG, NInvG, GH, extrapolated for
  NIG, and it always exists.
- **$\lvert\Sigma\rvert=1$, $\mathrm{tr}(\Sigma)=d$:
  distribution-invariant.** Both are functionals of $\bar\Sigma$
  alone.
- **$a=b$: not distribution-invariant.** The condition lives in
  GIG coordinates; the implied mean $E[Y]=K_{p+1}(a)/K_p(a)$
  depends on the index $p$, which differs across families
  ($p=-1/2$ for NIG, free for GH, degenerate for VG/NInvG).
  Measured on the S&P panel: after `'a_eq_b'`, $E[Y]=1$ for NIG
  but $0.25/0.14/0.10$ for GH at $d=5/10/50$
  ({doc}`subordinator_tracking_empirics`, family-comparison
  section).

One caveat applies to all of them: distribution invariance holds
only to the extent the families fit comparable laws. A clamped VG
(`alpha_min='density'`) is a different estimand by construction and
will disagree on $\mathrm{cv}^2$ under any gauge.

### 2.2 Dimension invariance

For $S\subseteq\{1,\dots,d\}$ the sub-vector $X_S$ is a mixture
with the *same* $Y$ and restricted $(\mu_S,\gamma_S,\Sigma_{SS})$
— closure under affine maps,
{ref}`Blaesild1981 <blaesild1981>`. This gives a clean criterion:

**Proposition.** A section commutes with coordinate marginalization
— gauging the $d$-model and then restricting equals restricting and
then gauging — if and only if its condition is a functional of the
law of $Y$ alone. $Y$-side sections pick the same $s$ for the model
and every sub-model; a $\Sigma$-side section picks
$s=T(\Sigma_{SS})$, which moves with $S$.

So $E[Y]=1$, geometric, median, and also $a=b$ are
dimension-invariant at the population level; the determinant and
trace gauges are not. The failure of the determinant gauge is not
small. Under equicorrelation
$\Sigma = \sigma^2[(1-\rho)I + \rho\mathbf 1\mathbf 1^\top]$, the
$E[Y]$ displayed by the $\lvert\Sigma\rvert=1$ gauge is

$$
E[Y]\Big|_{\det}
= \det(\bar\Sigma)^{1/d}
= \sigma^2\,(1-\rho)^{(d-1)/d}\bigl(1+(d-1)\rho\bigr)^{1/d}
\;\xrightarrow{d\to\infty}\; \sigma^2(1-\rho):
$$

at $\rho=0.35$ the same data displays an $E[Y]$ about $32\%$
smaller at $d=100$ than at $d=1$ — pure gauge artifact. The trace
gauge is exact *in expectation* under uniformly random column
subsets ($E[\mathrm{tr}(\Sigma_{SS})/k] = \mathrm{tr}(\Sigma)/d$)
but noisy per draw.

Two caveats separate the population statement from what the
proposed experiment will see.

First, once a $Y$-side gauge is fixed, $E[Y]\equiv 1$ across $d$ is
trivial, and the entire content of dimension invariance moves into
$\mathrm{Var}(Y)=\mathrm{cv}^2$ — which is *gauge-free*. Whether
$\mathrm{cv}^2$ is stable across $d$ is then a statistical question
about the data and the model, not about the gauge. The existing
NIG fits already answer it for this panel: in the $E[Y]=1$ gauge,

| $d$ | 5 | 10 | 25 | 50 | 100 | 200 | 468 |
|---|---|---|---|---|---|---|---|
| $\mathrm{cv}^2$ | 1.91 | 1.11 | 0.79 | 0.64 | 0.56 | 0.52 | 0.46 |

a monotone decline, while the univariate NIG fits of the
equal-weight index give $\mathrm{cv}^2 = 2.19$ ($d=50$ universe)
and $2.32$ (full panel) — *above every multivariate point*. The
mechanism: each coordinate carries common plus idiosyncratic
excess kurtosis; a univariate fit reads the total, while the
$d$-dimensional fit identifies $Y$ from cross-sectional
co-movement only, and the posterior concentrates via $q(x)/d$. On
data with idiosyncratic volatility clocks, the shared-$Y$ model
*should* report $\mathrm{cv}^2$ falling with $d$. No gauge can
remove this; a dimension-invariant gauge is what makes it visible
as a model property instead of a scale artifact.

Second, VG under `alpha_min='density'` has
$\mathrm{cv}^2 = 1/\alpha \le 2/(d+0.2)$: the clamp forces a $1/d$
decay of the estimand itself. Cross-$d$ VG comparisons are
structurally broken under the $d$-aware clamp; a $d$-independent
absolute `alpha_min` restores comparability of the estimand but
abandons density boundedness for $d > 2\,\alpha_{\min}$.

### 2.3 Summary

| Section | Distribution-invariant | Dimension-invariant |
|---|---|---|
| $E[Y]=1$ | yes (moment-pinned; matched for VG/NIG/GH) | yes ($Y$-side) |
| geometric $\exp E[\log Y]=1$ | yes, weaker (matched for VG/NInvG/GH) | yes ($Y$-side) |
| median$(Y)=1$ | weakest (not moment-matched) | yes ($Y$-side) |
| `'a_eq_b'` | **no** (GH vs NIG $E[Y]$: $0.10$ vs $1$) | yes ($Y$-side) |
| `'det_sigma_one'` / `'det_sigma_x'` | yes | **no** ($\approx 1-\rho$ drift) |
| $\mathrm{tr}(\Sigma)=d$ | yes | in expectation only |
| `'none'` | no (init- and path-dependent) | no |

$E[Y]=1$ is the only candidate satisfying both, with the NInvG
existence caveat; the geometric gauge is the robust fallback that
always exists. Both are cheap post-fit rescales along the existing
orbit (`NormalMixture._rescale`).

## 3. In EM or post estimation?

The batch EM map is *gauge-equivariant*. Under $g_s$ the posterior
is a relabeling, $Y\mid x \mapsto sY\mid x$
(GIG$(p-\tfrac d2,\,a+\tilde q,\,b+q(x))$ with
$a\mapsto a/s$, $b\mapsto sb$, $\tilde q\mapsto\tilde q/s$,
$q(x)\mapsto s\,q(x)$), and every family's M-step commutes with the
orbit because the subordinator families are closed under
$Y\mapsto sY$. Hence, in exact arithmetic, re-gauging after each
M-step does not change the sequence of *orbits* EM visits — only
the representative reported at the end. **Gauging inside batch EM
and gauging once after it are the same estimator.** Equivalently:
the orbit is a zero-information direction of the marginal
likelihood; batch EM neither pushes along it nor restores it.

The equivalence leaks through exactly four channels, which is
where the hypotheses live:

1. **Absolute floors and clamps are not equivariant.**
   `B_POST_FLOOR` ($b_{\mathrm{post}}\ge 10^{-6}$), `SIGMA_REG`,
   `GIG_CLAMP_LO/HI` are fixed constants while $q(x)$ and the GIG
   parameters scale with the gauge. On daily equities the
   $E[Y]=1$ gauge puts $q(x)\approx d$ (Mahalanobis scale); the
   $\lvert\Sigma\rvert=1$ gauge puts $q(x)\sim 10^{-4}d$ — four
   orders closer to the $b_{\mathrm{post}}$ floor. VG, the only
   family whose E-step actually leans on that floor, is the family
   where the in-EM gauge can plausibly change the fitted *orbit*.
2. **The stopping rule is gauge-dependent.** Convergence is
   measured on changes of $(\mu,\gamma,L_\Sigma)$
   (`em_convergence_params`). A gauged loop quotients out the flat
   direction; an ungauged loop lets float-level drift along the
   orbit contaminate the metric. This changes *when EM stops*, not
   where the orbit goes.
3. **Data scale interacts with (1).** Fitting returns in natural
   units versus percent ($\times 100$) moves every floor's bite
   point. Gauge-invariants should be unchanged; any change
   quantifies channel (1).
4. **Online EM averages η across iterations.** The EWMA recursion
   $\eta_t = (1-w)\eta_{t-1} + w\,t(x_t)$
   ({ref}`Cappe2009 <cappe2009>`) is an average of sufficient
   statistics *in a fixed gauge*. Re-gauging the model each step
   without transforming the running $\eta$ mixes incompatible
   scales — that is why the tracking study ran Phase 3 without
   re-gauging. But with no re-gauge the stochastic updates
   random-walk along the zero-information direction: at $h=21$ the
   gauge exploded to $\overline{E[Y]}_t = 4772$. Neither option is
   right. The consistent third option transforms the running η
   with the same orbit map: under $Y\mapsto sY$,

   $$
   \eta_1 \mapsto \eta_1 + \log s,\quad
   \eta_2 \mapsto \eta_2/s,\quad
   \eta_3 \mapsto s\,\eta_3,\quad
   \eta_4 \mapsto \eta_4,\quad
   \eta_5 \mapsto \eta_5/s,\quad
   \eta_6 \mapsto \eta_6/s,
   $$

   for $\eta = (E[\log Y], E[1/Y], E[Y], E[X], E[X/Y],
   E[XX^\top/Y])$. This is the one place a gauge plausibly
   *improves estimation* rather than reporting, and the one
   concrete package-change candidate.

## 4. Hypotheses

- **G1 (distribution invariance).** Under the $E[Y]=1$ display
  gauge, NIG/GH (and unclamped VG) fits of the same universe agree
  on $\gamma$ and $\Sigma$ to within estimation error (target:
  $\le 10\%$ relative $\ell_2$/Frobenius at $d\le 10$); under
  `'a_eq_b'` the displayed $E[Y]$ disperses by $\ge 5\times$
  (already observed). $\lvert\Sigma\rvert=1$ also aligns families.
- **G2 (dimension invariance of sections).** On the same data,
  $E[Y]$ displayed by the $\lvert\Sigma\rvert=1$ gauge drifts
  across nested $d$ by a factor consistent with the
  equicorrelation prediction $\approx(1-\rho)$; $Y$-side gauges
  show no drift by construction.
- **G3 (well-specified unbiasedness).** On synthetic single-clock
  data (fitted NIG generator, matched $T$), $\mathrm{cv}^2$
  estimates are unbiased across $d\in\{1,\dots,100\}$ and the
  $d=100$ estimate falls inside the interquartile range of the
  $100$ univariate estimates.
- **G4 (real-data decline is misspecification).** On the S&P
  panel the univariate ensemble of $\mathrm{cv}^2$ sits above the
  multivariate estimate at every $d$, and the multivariate
  $\mathrm{cv}^2(d)$ declines monotonically. The ensemble
  mid-range is *not* recovered by any gauge; the gap is an
  idiosyncratic-clock measure.
- **G5 (VG clamp).** With `alpha_min='density'`, VG
  $\mathrm{cv}^2$ tracks the forced ceiling $2/(d+0.2)$ across
  $d$; with a fixed absolute `alpha_min`, the cross-$d$ profile
  parallels NIG's.
- **G6 (batch equivalence).** For NIG/GH/NInvG, fitting with
  `'none'`, `'a_eq_b'`, or `'det_sigma_one'` and re-gauging
  post-fit gives identical gauge-invariants
  ($\kappa_{\mathrm{lev}}$, $\mathrm{cv}^2$, mean log-likelihood)
  up to float noise (target: $10^{-8}$ relative for NIG's
  closed-form M-step, $10^{-6}$ for GH's solver path).
- **G7 (the leaks).** VG violates G6 measurably through
  `B_POST_FLOOR` when the in-EM gauge compresses $q(x)$
  ($\lvert\Sigma\rvert=1$ on raw returns); iteration counts differ
  across in-EM gauges through the stopping metric even where
  invariants agree; rescaling the data $\times 100$ leaves
  invariants unchanged except where floors bind.
- **G8 (online EM).** Per-step re-gauging *with* the η transform
  of §3(4) reproduces the no-regauge orbit path (gauge-invariant
  $\kappa_t$ within Monte Carlo bands) while keeping
  $\overline{E[Y]}_t\in[0.5,2]$; per-step re-gauging *without*
  the transform biases $\kappa_t$.

## 5. Study plan

Each phase states its completion criterion. All code in
`notebooks/gauges/` on the public API; NIG/GH/VG (+NInvG where
noted); the S&P panel and universes of
{doc}`subordinator_tracking_empirics`.

**Phase A — equivariance audit (G6, G7).** Synthetic NIG/GH data,
$d\in\{2,10\}$, one seed. Fit from a common `default_init` under
each in-EM gauge; record per-iteration *invariants*
($\kappa_{\mathrm{lev}}$, $\mathrm{cv}^2$, mean log-likelihood) and
floor-hit counters ($b_{\mathrm{post}}$ at floor, GIG clamps).
*Done when:* invariant trajectories overlaid per family with
agreement tolerances measured, and VG's floor-driven divergence (or
its absence) quantified.

**Phase B — distribution invariance (G1).** Real panel,
$d\in\{5,10,25,50\}$, seed 0: NIG, GH, VG (clamped and absolute
`alpha_min`), NInvG. Display each fit under `'a_eq_b'`,
$\lvert\Sigma\rvert=1$, $E[Y]=1$, geometric. Report cross-family
relative distances of $\gamma$ (ℓ2), $\Sigma$ (Frobenius and
log-det), and the displayed $E[Y]$ table. *Done when:* the §2.3
table's distribution-invariance column is confirmed or amended
with measured numbers.

**Phase C — dimension invariance (G2–G5).** (i) Synthetic: draw
from the fitted $d=100$ NIG generator, $T=2552$, $R=10$; fit at
nested $d\in\{1,5,10,25,50,100\}$ and fit all $100$ univariate
margins on one replicate; compare $\mathrm{cv}^2$ point estimates
to truth and the univariate ensemble quantiles. (ii) Real panel:
same design, NIG (+GH at $d\le 50$; VG both clamp variants);
additionally display $E[Y]$ under $\lvert\Sigma\rvert=1$ across
$d$ to measure the G2 drift against the equicorrelation
prediction. *Done when:* synthetic bias/SE table across $d$ and
the real-data ensemble-vs-multivariate figure exist, with the G4
gap quantified.

**Phase D — scale robustness (G7).** Refit Phase B's NIG/GH/VG at
$d\in\{10,50\}$ with returns in natural units and $\times 100$.
Compare gauge-invariants and floor-hit counters. *Done when:* the
invariants' sensitivity to data scale is tabulated per family ×
in-EM gauge, and any discrepancy is attributed to a named
constant.

**Phase E — online EM gauge handling (G8).** Rerun the Phase 3
online loop ($d=50$, $h\in\{21,63\}$) under three policies: no
re-gauge (baseline), per-step re-gauge without η transform,
per-step re-gauge with the §3(4) η transform. Track
$\overline{E[Y]}_t$, gauge-invariant $\kappa_t$, direction cosine,
and terminal-vs-static consistency. *Done when:* the three
policies are compared on one figure and G8 is accepted or
rejected.

## 6. Decision gates

- If G6 holds and G7 is confined to VG: the in-EM `regularization`
  flag is a reporting/stopping convenience for batch EM. Keep
  `'a_eq_b'` as a cheap dimension-invariant loop gauge, add a
  `'mean_y_one'` mode (or promote `pin_mean_y`) as the recommended
  *display* gauge, and standardize all cross-family and cross-$d$
  tables on it.
- If G7 shows VG orbit-level sensitivity: recommend fitting VG in
  the gauge that keeps $q(x) = O(d)$ (the $E[Y]=1$ scale) and
  record it as a design row.
- If G8 holds: the η-consistent re-gauge is the package-change
  candidate — an online-EM option that re-gauges model and running
  η together, closing the Phase 3 gauge-explosion problem without
  biasing the recursion.
- NInvG: if the $E[Y]=1$ gauge is adopted, its display must fall
  back to the geometric gauge when $\alpha\le 1+$ margin rather
  than silently using the floored moment.

## 7. What this is not asking `normix` to do

No change to fitting defaults before Phases A–E report. The
invariants $\kappa_{\mathrm{lev}}$, $\kappa$, $\mathrm{cv}^2$ need
no gauge at all and remain the preferred comparison quantities;
the gauge question is about displaying $(\gamma, \Sigma, E[Y],
\mathrm{Var}(Y))$ themselves, about warm starts that cross
families or dimensions, and about the online-EM η recursion.
