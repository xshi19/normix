# EM Robustness Follow-Ups (post `b_post`-floor review)

**Date:** 2026-06-10
**Status:** In progress. **Phase 1 (docs) and Phase 2 bug fixes done:**
- **M1, M2, D1** — tech-note §6 corrected (`fab19be`, PR #49): case split at
  $\nu=0$, cap $(d-2\alpha)/b_{\min}$ linear in $d$ and uniform in
  $a_{\text{post}}$, verification table + `kve` repro, NC $\omega$-floor
  recommendation deleted, $4.4\times10^{23}$ footnoted as clamp-dependent, and
  the F5 gauge note added.
- **B1** — sign-preserving M-step denominator (`0d5ea29`, PR #50): floor at
  $-\max(|D|, \texttt{SAFE\_DENOMINATOR})$ in `JointNormalMixture._mstep_normal_params`.
- **B2** — distribution-specific $(\alpha-1)$ denominator floor for VG/NInvG
  prior moments (`5fa19f7`, PR #51) with the revised design in §3.2 — the
  original lifted-GIG `expectation_params` sketch was rejected; see §3.2.
- **D2** — `normix/` docstring/comment `dev-notes/` cross-link violations fixed;
  `scripts/check_doc_links.sh` green.

Remaining items (R1–R3, F1–F5, the bulk of T1–T7, D3, D4) not yet implemented.
**Origin:** post-merge review of the `b_post` floor fix
(`5a0ecfb`, PR #45). The review verified the math of
[`../tech_notes/vg_em_inverse_moment_singularity.md`](../tech_notes/vg_em_inverse_moment_singularity.md)
analytically and numerically, audited the E-step/M-step code paths, and ran
`tests/test_variance_gamma.py` + `tests/test_em_regression.py` (32 passed).
**Verdict:** the methodology is sound and the floor is the right fix, well
placed. This plan tracks (a) one substantive correction to the tech note,
(b) two latent bugs found during the review, (c) a consolidation of the
posterior-GIG machinery, (d) EM-fitter hardening, and (e) the test and doc
work for each.

**Companion docs:**
[`../tech_notes/vg_em_inverse_moment_singularity.md`](../tech_notes/vg_em_inverse_moment_singularity.md),
[`../investigations/variance_gamma_em_nan.md`](../investigations/variance_gamma_em_nan.md),
`docs/theory/em_algorithm.rst`.
**Code:** `normix/mixtures/{joint,marginal,factor}.py`,
`normix/distributions/{variance_gamma,normal_inverse_gamma,normal_inverse_gaussian,generalized_hyperbolic,generalized_inverse_gaussian,gamma,inverse_gamma}.py`,
`normix/fitting/em.py`, `normix/utils/constants.py`.

---

## 1. Findings summary

| ID | Type | Finding | Priority | Effort | Status |
|----|------|---------|----------|--------|--------|
| M1 | Math/doc error | Tech note §6 mis-derives the capped moment for $p_{\text{post}}<0$ and reverses the $b$-floor vs $\omega$-floor comparison. The adopted $b$-floor is actually the *more* uniform regularizer. | P1 | doc only | ✅ `fab19be` |
| M2 | Doc nit | §5's "$E[Y^{-1}\mid x]\approx 4.4\times10^{23}$ at the exact mode ($q(x)=0$)" cannot follow from $q=0$ literally; it reflects internal `TINY` clamps / float spacing in the repro. | P3 | doc only | ✅ `fab19be` |
| B1 | Bug | `JointNormalMixture._mstep_normal_params`: `safe_D` floors $D = 1-\eta_2\eta_3$ at $+10^{-10}$, but $D \le 0$ always (Cauchy–Schwarz), so the floor flips the sign of $\mu, \gamma$ in the near-Gaussian limit. | P1 | ~5 LOC | ✅ `0d5ea29` |
| B2 | Bug | `VarianceGamma._subordinator_expectations` returns $E[1/Y]=\beta/(\alpha-1)$, which is **negative** for $\alpha<1$ (true moment is $+\infty$); same for NInvG's $E[Y]$. Feeds `compute_eta_from_model` → incremental-EM warm starts and shrinkage. Now *more* reachable because the floored batch EM produces finite models with $\alpha<1$. Fix: distribution-specific denominator floor on the single divergent moment (keep the other two exact); **not** via `to_gig().expectation_params()`, which corrupts the exact moments — see §3.2. | P1 | ~15 LOC | ✅ `5fa19f7` |
| R1 | Refactor | `B_POST_FLOOR` is applied at four call sites (`joint.py`, `marginal.py` CPU, `factor.py` ×2). Collapse to one floored helper per hierarchy. | P2 | ~15 LOC | ☐ todo |
| R2 | Refactor | All eight `_posterior_gig_params` overrides (4 joint + 4 factor) are the same map in GIG coordinates: $(p_{\text{gig}}-\tfrac d2,\ a_{\text{gig}}+w_2,\ b_{\text{gig}}+z_2)$ of `subordinator().to_gig()`. Replace with one base implementation per hierarchy. | P2 | −80 LOC net | ☐ todo |
| R3 | Cleanup | `_PARAM_EPS = 1e-10` defined locally in `em.py` (violates the constants rule); dead `hasattr(j, '_posterior_gig_params')` guard in `NormalMixture._e_step_subordinator_cpu` (method is abstract on the base, always present). | P3 | ~10 LOC | ☐ todo |
| F1 | Framework | `BatchEMFitter` has no finite guard: a NaN iterate silently runs to `max_iter` and returns a NaN model (the original failure mode). Add fail-fast / keep-last-finite. | P1 | ~40 LOC | ☐ todo |
| F2 | Framework | `EMResult.log_likelihoods` is `None` unless `verbose ≥ 1` — diagnostics coupled to printing. Add `track_ll`. | P2 | ~30 LOC | ☐ todo |
| F3 | Framework | `self._target_log_det` set inside `fit` makes the fitter stateful / non-reentrant. Thread it through as a local. | P2 | ~15 LOC | ☐ todo |
| F4 | Framework (design-gated) | The likelihood is still unbounded after the floor; EM can park near the spike ($\alpha < d/2$, $\mu$ on a data point). Expose an optional $\alpha$ lower bound (ghyp "fix-$\lambda$" analogue) as a user-facing estimand control. | P3 | ~40 LOC | ☐ todo |
| F5 | Doc note | `B_POST_FLOOR` is not gauge-invariant: under the $Y \to sY$ rescale ($\Sigma \to \Sigma/s$), $b_{\text{post}} = q(x)$ scales with $s$, so a fixed $10^{-6}$ binds with different strength under `det_sigma_one` vs `none`. Document. | P3 | doc only | ✅ `fab19be` |
| T1–T6 | Testing | High-d EM, quantitative cap, dormancy, small-$\alpha$ MCECM/incremental, monotone LL, `safe_D` sign — see §4. | P1–P2 | ~250 LOC | ◐ partial (B1/B2 regressions landed) |
| D1, D2 | Docs | Tech note §6 rewrite (D1); `joint.py`/`constants.py` cross-link violations (D2) — see §5. | P1 | doc only | ✅ `fab19be` (D1), this PR (D2) |
| D3, D4 | Docs | Post-implementation design/architecture rows (D3); optional public EM paragraph (D4) — see §5. | P2–P3 | doc only | ☐ todo |

---

## 2. The math correction (M1)

§6 of the tech note claims the capped moment behaves like
$E[Y^{-1}\mid x] \lesssim C\,a_{\text{post}}^{p_{\text{post}}}\,b_{\min}^{p_{\text{post}}-1}$
("$\sim 10^9$ at $d=1$; larger at high $d$") and recommends switching to the
Nitithumbundit–Chan $\omega$-floor for high-dimensional VG because it "caps
$E[Y^{-1}\mid x]$ uniformly in $a_{\text{post}}$". Both statements are wrong;
the formula is valid only for $0 < p_{\text{post}} < 1$.

**Correct small-$\omega$ asymptotics of the capped moment.** Write
$\nu = p_{\text{post}}$, $\omega = \sqrt{a_{\text{post}} b_{\min}}$:

$$
E[Y^{-1}\mid x]\Big|_{b=b_{\min}} \;\sim\;
\begin{cases}
\dfrac{\Gamma(1-\nu)}{\Gamma(\nu)}\, 2^{1-2\nu}\,
a_{\text{post}}^{\nu}\, b_{\min}^{\nu-1}, & 0<\nu<1,\\[2ex]
\dfrac{2|\nu|}{b_{\min}} \;=\; \dfrac{d-2\alpha}{b_{\min}}, & \nu<0,
\end{cases}
$$

and for large $\omega$ (floor effectively non-binding),
$E[Y^{-1}\mid x] \approx \sqrt{a_{\text{post}}/b_{\min}}$. So the worst-case
cap is $\max\!\big((d-2\alpha)/b_{\min},\ \sqrt{a_{\text{post}}/b_{\min}}\big)$
— **linear in $d$ and uniform in $a_{\text{post}}$** in the singular regime.
The $\omega$-floor instead gives $b \ge \Delta^2/a_{\text{post}}$, hence a cap
$2|\nu|\,a_{\text{post}}/\Delta^2$ that **grows linearly in
$a_{\text{post}}$** — the uniformity claim is backwards.

**Numerical verification** (scipy `kve`, $b_{\min}=10^{-6}$, $\Delta=10^{-3}$):

| Case | Tech-note formula | Exact |
|---|---|---|
| $\nu=0.2,\ a=2$ ($d=1,\ \alpha=0.7$) | $2.8\times10^4$ | $2.99\times10^4$ ✓ |
| $\nu=-0.5,\ a=2$ ($d=1,\ \alpha\to0$) | $7.1\times10^8$ | $1.0\times10^6$ |
| $\nu=-49.5,\ a=2$ ($d=100,\ \alpha=0.5$) | astronomically large | $9.9\times10^7 = 99/b_{\min}$ |
| $a$-scaling, $\nu=-0.5$, $a\in[0.1,100]$ | — | $b$-floor cap $\approx 10^6$ (flat); $\omega$-floor cap $10^5\to10^8$ |

Repro:

```python
import numpy as np
from scipy.special import kve
E_inv_Y = lambda p, a, b: np.sqrt(a/b) * kve(p-1, np.sqrt(a*b)) / kve(p, np.sqrt(a*b))
```

**Consequence:** the design decision (constant $b$-floor) stands, on *stronger*
grounds than the note claims — the "if high-dimensional VG ever matters,
switch to the NC $\omega$-floor" escape hatch should be deleted, not kept.

---

## 3. Proposed design

### 3.1 B1 — sign-preserving M-step denominator

For each observation, $E[Y^{-1}\mid x_i]\,E[Y\mid x_i] \ge 1$ (Jensen). For the
batch averages, by Cauchy–Schwarz
$\eta_2\eta_3 = \bar u\,\bar v \ge \big(\overline{\sqrt{uv}}\big)^2 \ge 1$, so
$D = 1-\eta_2\eta_3 \le 0$ always (equality iff the posterior is degenerate —
the Gaussian limit). The current floor replaces a tiny negative $D$ with
$+10^{-10}$, flipping the sign of both $\mu$ and $\gamma$. Fix in
`JointNormalMixture._mstep_normal_params`:

```python
safe_D = jnp.where(D < 0.0, -1.0, 1.0) * jnp.maximum(jnp.abs(D), SAFE_DENOMINATOR)
```

(`D = 0` maps to $-$`SAFE_DENOMINATOR`, the theoretically correct side; a
positive $D$ — only possible from user-supplied or corrupted $\eta$ — keeps
its sign rather than being silently negated.)

### 3.2 B2 — prior subordinator expectations at $\alpha \le 1$

`compute_eta_from_model` reconstructs $\eta$ at the model's own parameters,
but for VG with $\alpha \le 1$ the exact $E[1/Y]$ does not exist, and the
closed form $\beta/(\alpha-1)$ silently returns a *negative* number (verified:
$-5.0$ at $\alpha=0.8$). Symmetrically, NInvG's $E[Y] = \beta/(\alpha-1)$
breaks for InverseGamma $\alpha \le 1$.

**Design (revised): regularize only the single divergent moment with a
distribution-specific denominator floor — do *not* route through
`to_gig().expectation_params()`.** This supersedes the earlier "floor the
whole $\eta$ through the lifted GIG" sketch; that version was rejected for two
reasons:

1. **It corrupts the two moments that are already exact and finite.** Of the
   three subordinator moments, only $\beta/(\alpha-1)$ is problematic. For VG
   $E[\log Y] = \psi(\alpha) - \log\beta$ and $E[Y] = \alpha/\beta$ are exact
   $\forall\,\alpha>0$; for NInvG $E[\log Y] = \log\beta - \psi(\alpha)$ and
   $E[1/Y] = \alpha/\beta$ are exact $\forall\,\alpha>0$. Replacing *all three*
   with the lifted-GIG `expectation_params()` (which uses 5 Bessel evaluations
   and a finite-difference $\partial_p \log K_p$) injects avoidable error into
   the exact moments. Measured at $b_{\min}=10^{-6}$:

   | $\alpha$ | $E[\log Y]$ exact | GIG path | abs. err | $E[Y]$ exact | GIG path | abs. err |
   |---|---|---|---|---|---|---|
   | $0.5$ | $-3.0621$ | $-3.0339$ | $2.8\times10^{-2}$ | $0.1667$ | $0.1671$ | $4.1\times10^{-4}$ |
   | $0.2$ | $-5.2890$ | $-4.2968$ | $\mathbf{9.9\times10^{-1}}$ | $0.2000$ | $0.2150$ | $1.5\times10^{-2}$ |

   The $\sim 1.0$ absolute error in $E[\log Y]$ at $\alpha=0.2$ is unacceptable
   for moments that have a one-line exact form.

2. **It is more complex, not simpler.** Routing the special cases through the
   general GIG Bessel machinery contradicts the design philosophy
   (*"special-case distributions must use their own analytical formulas rather
   than routing through the general computation"*). The b-floor consistency
   argument does not require the *moment* to equal the lifted GIG moment — it
   only requires a finite, positive, continuous surrogate.

**Adopted fix.** Keep the exact closed forms for the two always-finite
moments; floor only the $(\alpha-1)$ denominator of the divergent one:

```python
# VarianceGamma._subordinator_expectations (and Factor sibling)
E_log_Y = digamma(alpha) - log(beta)                          # exact
E_inv_Y = beta / jnp.maximum(alpha - 1.0, ALPHA_MOMENT_MARGIN) # floored
E_Y     = alpha / beta                                         # exact

# NormalInverseGamma._subordinator_expectations (and Factor sibling)
E_log_Y = log(beta) - digamma(alpha)                          # exact
E_inv_Y = alpha / beta                                         # exact
E_Y     = beta / jnp.maximum(alpha - 1.0, ALPHA_MOMENT_MARGIN) # floored
```

- For $\alpha > 1 + \varepsilon$ the result equals the exact $\beta/(\alpha-1)$.
- For $\alpha \le 1 + \varepsilon$ the moment is capped at $\beta/\varepsilon$:
  finite, positive, and **continuous** at $\alpha = 1+\varepsilon$ (both sides
  give $\beta/\varepsilon$). `jnp.maximum` is JIT-safe with no branch.
- This caps the inverse moment at a modest value ($\beta/0.1 = 10\beta$),
  which is *safer* downstream (no $\sim10^4$–$10^5$ blow-up in
  $E[XX^\top/Y]$ that the lifted-GIG value would produce) while still being a
  legitimate regularization in the divergent regime.
- `NormalInverseGaussian` is unchanged: its InverseGaussian subordinator
  ($\mathrm{GIG}(p{=}-\tfrac12)$) has finite positive moments for all
  parameters, so it already routes through GIG correctly with no singularity.
- New constant `ALPHA_MOMENT_MARGIN` (`0.1`) in `normix/utils/constants.py`
  with doc rows in `ARCHITECTURE.md` and `coding-conventions.mdc`.

Cost: zero Bessel evaluations — two `digamma` calls and a `jnp.maximum`, same
as before the fix.

### 3.3 R1 + R2 — one posterior-GIG map per hierarchy

All eight `_posterior_gig_params` overrides encode the identical conjugacy in
GIG coordinates. With the exact embeddings already on every subordinator
(`Gamma.to_gig`, `InverseGamma.to_gig`, `InverseGaussian.to_gig`), plus a
trivial identity `GeneralizedInverseGaussian.to_gig()` (returns `self`), the
map collapses to **one non-abstract base implementation per hierarchy**:

```python
# JointNormalMixture (FactorNormalMixture analogous, with self.subordinator)
def _posterior_gig_params(self, z2, w2):
    """Pure prior-to-posterior conjugacy map, uniform across families."""
    gig = self.subordinator().to_gig()
    return gig.p - self.d / 2.0, gig.a + w2, gig.b + z2

def _floored_posterior_gig_params(self, z2, w2):
    """E-step entry point: posterior map + B_POST_FLOOR (single chokepoint)."""
    p, a, b = self._posterior_gig_params(z2, w2)
    return p, a, jnp.maximum(b, B_POST_FLOOR)
```

- Deletes the 8 subclass overrides
  (`JointVarianceGamma`, `JointNormalInverseGamma`, `JointNormalInverseGaussian`,
  `JointGeneralizedHyperbolic` + the 4 `Factor*` siblings); their per-family
  posterior formulas move into the base docstring as a table.
- `_posterior_gig_params` stays **pure** (posterior-parameter unit tests
  unaffected); the floor moves from four scattered call sites into the two
  `_floored_*` definitions (R1). All four E-step paths
  (`JointNormalMixture._compute_posterior_expectations`,
  `NormalMixture._e_step_subordinator_cpu`,
  `FactorNormalMixture._conditional_expectations`,
  `FactorNormalMixture._e_step_subordinator_cpu`) call the floored entry point.
- Broadcasting is unchanged: the GIG embedding parameters are scalars, the
  quad forms may be `(n,)` arrays (CPU batch paths).
- This is not a closed-form-specialization loss (the algebra is identical;
  Bessel work unchanged), so it is consistent with the design philosophy:
  it removes mathematical duplication rather than delegating to a more
  general computation.
- Behavior-preserving: validated by an e_step regression on fixed seeds
  across all four families × {jax, cpu} before/after (see T3).

### 3.4 F1–F3 — fitter hardening

**F1 (fail-fast / keep-last-finite).** A single NaN leaf makes
`_param_change` NaN, so one scalar test suffices. Python loop: check
`jnp.isfinite(max_change)` after `_step`; on failure revert to the previous
model, record divergence, break. Scan path: extend the carry,

```python
finite = jnp.isfinite(max_change)
keep_old = converged | diverged | ~finite
mdl_out = jax.tree.map(lambda n, o: jnp.where(keep_old, o, n), mdl_new, mdl)
diverged = diverged | ~finite
```

Add `diverged: bool = False` to `EMResult` (frozen dataclass, defaulted —
existing constructor sites unchanged) and a `verbose` warning line. This
converts any future degeneracy from silent NaN corruption (the original
19-iteration failure mode) into a clean stop with the last finite iterate.

**F2 (decouple LL tracking).** `BatchEMFitter(track_ll: bool = False)`.
Python loop: compute LL when `track_ll or verbose >= 1`. Scan path:
`track_ll` is a static Python bool — when set, compute
`marginal_log_likelihood` inside the body and emit it as a scan output. The
monotone-LL regression test (T5) builds on this.

**F3 (stateless fitter).** Replace the `self._target_log_det` attribute with
a local computed in `fit` and passed to `_regularize(model, target_log_det)`
(and through the step helpers). `BatchEMFitter` becomes reentrant; behavior
identical.

### 3.5 F4 — optional $\alpha$ lower bound (design-gated)

The floor makes iterates finite but the VG likelihood is still unbounded
($\alpha < d/2$ with $\mu$ on a data point). Mirror the `ghyp` fix-$\lambda$
option as an *opt-in* estimand control:

- `Gamma.from_expectation(..., alpha_min: float | None = None)` clamps the
  digamma-Newton result (`jnp` path and CPU path).
- `VarianceGamma.fit(..., alpha_min=None)` forwards through
  `m_step → _subordinator_from_eta`. Default `None` = no behavior change.
- Document the recommended choice $\alpha_{\min} = d/2 + 1 + \varepsilon$
  (keeps $E[Y^{-1}\mid x]$ classically bounded) vs. $d/2$ (keeps the density
  bounded) in the tech note.

Requires a short design discussion before implementation (API surface:
kwarg vs. fitter config) — add a row to `../design/design.md` when decided.

### 3.6 R3 — cleanups

- Move `_PARAM_EPS` to `normix/utils/constants.py` as `PARAM_CHANGE_EPS`
  (update the constants tables in `ARCHITECTURE.md` and
  `coding-conventions.mdc` per the agent-maintenance skill).
- Delete the `hasattr(j, '_posterior_gig_params')` guard in
  `NormalMixture._e_step_subordinator_cpu` (the method is abstract on
  `JointNormalMixture`; after R2 it is concrete on the base — either way the
  guard is dead).

---

## 4. Test plan

| ID | File | Test | Asserts | Marker |
|----|------|------|---------|--------|
| T1 | `test_variance_gamma.py` | `TestInverseMomentSingularityVG::test_em_no_overflow_high_dim` — parametrize $d \in \{5, 10\}$ (and $d=50$ as `slow`), data from VG $\alpha=1.0$ with the sample mean appended as a near-mode observation, both E-step backends | all params finite, final LL $\ge$ init LL. Rationale: threshold $\alpha \le d/2+1$ means the default init $\alpha=2$ is *already* in the divergent regime for every $d \ge 2$; current coverage is $d=1$ only. | `slow` for $d=50$ |
| T1b | `test_factor_mixture.py` | factor variant: `FactorVarianceGamma`, $d=10$, $r=2$, near-mode observation | finiteness + LL improvement | — |
| T2 | `test_variance_gamma.py` | quantitative cap: $E[1/Y \mid x{=}\mu]$ at the floor matches the §2 asymptotics — $\alpha=0.7$ ($\nu=0.2$ branch, expect $\approx 2.99\times10^4$ for $a_{\text{post}}=2$) and $\alpha=0.2$ ($\nu<0$ branch, expect $\approx(d-2\alpha)/b_{\min}$) | `rtol=0.05`. Guards the floor's *value*, not just `isfinite` (an `isfinite`-only test cannot catch a floor-constant regression). | — |
| T3 | `test_em_regression.py` (or new `test_posterior_gig.py`) | dormancy + refactor regression: for GH/NIG/NInvG with typical priors ($b_{\text{prior}} \gg 10^{-6}$), `conditional_expectations(x=mu)` equals the unfloored GIG moments exactly; e_step outputs for all 4 families × {jax, cpu} match pre-R2 golden values on fixed seeds | `rtol=1e-12` dormancy; `rtol=1e-12` refactor parity | — |
| T4 | `test_incremental_em.py`, `test_mcecm.py` | small-$\alpha$ coverage (all current EM-path tests use $\alpha \in [2.0, 2.5]$): `compute_eta_from_model` finite with $\eta_2, \eta_3 > 0$ at $\alpha \in \{1.0, 0.8, 0.2\}$ (B2), **and** the two always-finite moments stay bit-exact (VG: $E[\log Y]=\psi(\alpha)-\log\beta$, $E[Y]=\alpha/\beta$; NInvG: $E[\log Y]=\log\beta-\psi(\alpha)$, $E[1/Y]=\alpha/\beta$) — guards against a regression to the lossy GIG path; `IncrementalEMFitter` and `algorithm='mcecm'` on heavy-peaked VG data ($\alpha_{\text{true}}=0.7$) stay finite | finiteness, positivity of $\eta_2, \eta_3$, exactness of the closed-form moments (`rel=1e-12`) | — |
| T5 | `test_em_regression.py` | monotone LL: heavy-peaked $d=1$ VG case with `track_ll=True` (F2) | $\Delta\text{LL} \ge -10^{-8}$ per iteration — the defining EM invariant; the original failure was "LL improving, then NaN" | — |
| T6 | `test_jax_distributions.py` (or alongside M-step tests) | `safe_D` sign (B1): synthetic $\eta$ with $D = -10^{-12}$ recovers $\mu, \gamma$ with the correct sign (compare against $D=-10^{-8}$ reference); property check $1 - \eta_2\eta_3 \le 0$ for `e_step` output across all four families | sign correctness; Cauchy–Schwarz property | — |
| T7 | `test_em_regression.py` | F1 guard: a fitter step forced to NaN (e.g. init with absurd params) returns `diverged=True` with all-finite model params | last-finite semantics | — |

---

## 5. Docs plan

| ID | File | Change |
|----|------|--------|
| D1 ✅ `fab19be` | `../tech_notes/vg_em_inverse_moment_singularity.md` | **Done.** Rewrote the §6 "trade-off" paragraph with the corrected asymptotics of §2 above (case split at $\nu=0$, cap $(d-2\alpha)/b_{\min}$ linear in $d$, uniform in $a_{\text{post}}$); **deleted** the "switch to the NC $\omega$-floor at high $d$" recommendation (it is the weaker floor for large $a_{\text{post}}$); added the verification table + `kve` repro snippet (M1). Footnoted the $4.4\times10^{23}$ anecdote as clamp-dependent (M2). Added the F5 gauge note: $b_{\text{post}}$ scales under the $Y \to sY$ regularization gauge, so the fixed $10^{-6}$ binds differently under `det_sigma_one` vs `none`. |
| D2 ✅ (this PR) | `normix/mixtures/joint.py`, `normix/utils/constants.py` | **Done.** `scripts/check_doc_links.sh` had failed on master (introduced by `5a0ecfb`): `joint.py` (docstring) and `constants.py` (comment) referenced `dev-notes/tech_notes/...`, violating `docs-cross-links.mdc`. Fix applied: in `joint.py` replaced with a `:doc:`../docs/theory/em_algorithm`` reference to the public theory page; in `constants.py` dropped the path (the tech-note pointer lives in `ARCHITECTURE.md`'s constants table). Checker now green. |
| D3 | `../design/design.md`, `../ARCHITECTURE.md` | After implementation: decision rows for "posterior GIG map via `to_gig` embedding (R2)", "prior moments via distribution-specific $(\alpha-1)$ denominator floor at $\alpha \le 1$ (B2) — *not* the lifted-GIG `expectation_params`, which corrupts the exact $E[\log Y], E[Y]$ moments", and F4 if adopted; update the constants tables (`PARAM_CHANGE_EPS`, `ALPHA_MOMENT_MARGIN`) and the E-step paragraph in `ARCHITECTURE.md`; then archive this plan to `../archive/design/` per `maintain-design-docs.mdc`. |
| D4 (optional) | `docs/theory/em_algorithm.rst` | Short public paragraph on the degenerate-subordinator regularization (the floor currently exists only in internal docs). Decide whether the rationale is user-facing enough to promote. |

---

## 6. Phasing

```
Phase 1 (docs)        M1, M2, D1, D2            — ✅ done (no behavior change)
Phase 2 (bug fixes)   B1, B2, T2, T4, T6        — ✅ B1, B2 done; quantitative-cap/MCECM tests (T2, partial T4/T6) still open
Phase 3 (consolidate) R2, R1, R3, GIG.to_gig, T3, T1, T1b   — ☐ todo
Phase 4 (fitter)      F1, F2, F3, T5, T7        — ☐ todo (EMResult gains `diverged`)
Phase 5 (design-gated) F4 (+ design.md row), D3, D4         — ☐ todo
```

- Phases are independently mergeable; Phase 3 depends on nothing in Phase 2
  but T3's golden values must be captured *before* the R2 refactor lands.
- T1/T2 only need the existing floor — they can land any time after Phase 1.
- Phase 5 requires explicit design sign-off (API surface for `alpha_min`,
  whether D4 is promoted).

**Exit criteria per phase:** full `uv run pytest tests/` green; Phase 3
additionally requires bit-level (or `rtol=1e-12`) e_step parity across all
four families × both backends; Phase 4 requires the NaN-injection test (T7)
to demonstrate keep-last-finite semantics on both loop paths.

---

## References

- Review basis: commit `5a0ecfb` *fix(em): floor posterior b_post to prevent
  VG EM overflow (#45)*.
- T. Nitithumbundit, J. S. K. Chan (2015), arXiv:1504.01239 — §3.3.1
  ($\Delta$-region bound; compared against in §2/M1).
- D. Lüthi, W. Breymann, `ghyp` R package — fix-$\lambda$ option (F4 analogue).
- DLMF §10.30 — small-argument asymptotics of $K_\nu$ used in §2.
