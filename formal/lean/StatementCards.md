# Statement cards for W2 M0

Six English proposition cards, one per checklist group in [`05_math_checklist.md`](../../dev-notes/research/formal_verification/05_math_checklist.md). IDs come from that inventory. Template: report 01, section 6.4. W2 M0 implementation **merged** in PR #161 (`80073f3`): 31 exported theorems across three modules. Every card remains **statement-drafted**, with human statement review **pending**. Merge status does not mean `statement-reviewed`.

Historical source baseline: `51e053c57237a29c71b56d3a2a86988277477bd9`; current correspondence: master `d557399905c533b3f15825941be6f04a6e897584`. Compilation and zero-proof-hole evidence come from the merged build record; this documentation update reruns no Lean build.

Shared assurance record for every card: **math theorem** — checked, statement review pending; **program-translation certificate** — not covered; **approximation-error bound** — not covered; **floating-point guarantee** — not covered. All claims use exact real arithmetic. Future acceptance records must retain all four labels; see the [roadmap](../../dev-notes/research/formal_verification/04_integrated_roadmap.md#execution-and-mathematical-boundaries).

Human-review statements, definitions, and domains before assigning proof work. AI edits proofs only against that specification. Changes to definitions, assumptions, domains, or source mappings require visible revisions and renewed human review.

Lean names are fully qualified in `AxiomAudit.lean`.

Current Python correspondence (2026-10-31): `GIG.from_natural` rejects positive
rate coordinates with `eqx.error_if` and calls `_from_natural_unchecked` on
accepted inputs, including zero rates. Acceptance is a rate-sign check, not a
normalizability theorem for every boundary tuple. The unchanged Lean clamp model
describes the historical baseline. It agrees algebraically on nonpositive rates;
it does not model runtime rejection, JAX transformations, or floating-point behavior.

## EF-01, EF-02, EF-03: GIG parameter round trips

| Field | Content |
|---|---|
| Statement in prose | The natural-parameter map $(p,a,b)\mapsto(p-1,-b/2,-a/2)$ and its unclamped inverse $(\theta_0,\theta_1,\theta_2)\mapsto(\theta_0+1,-2\theta_2,-2\theta_1)$ compose to the identity in both orders, for all real inputs. Each coordinate identity is also stated separately. |
| Lean signature | `Normix.Distributions.gig_b_round_trip (b : ℝ) : -2 * (-b / 2) = b`; `gig_a_round_trip`; `gig_p_round_trip (p : ℝ) : (p - 1) + 1 = p`; `gig_from_natural_unclamped_round_trip (q : GigParams) : gigFromNaturalUnclamped (gigNatural q) = q`; `gig_natural_unclamped_round_trip (t : GigNatural) : gigNatural (gigFromNaturalUnclamped t) = t` |
| Assumptions | None (real arithmetic). Validity ($a,b>0$) is a separate card (EF-06). |
| Why this is the intended spec | The separate coordinate identities fix the sign, factor 2, and order $(\theta_0,\theta_1,\theta_2)$ paired with $(\log y,1/y,y)$. Both round trips are also checked; they alone cannot exclude a consistently wrong permutation. |
| Covered normix function | `normix/distributions/generalized_inverse_gaussian.py`: `natural_params` (`[p - 1, -b/2, -a/2]`) and `_from_natural_unchecked`. Public `from_natural` delegates on accepted inputs; its rejection path is outside the proof. |
| Risks of a weakened spec | Testing positive inputs only would lose the universal identity. Equating the unrestricted algebraic model with the current constructor omits input rejection (see EF-06). Round trips alone do not check the conventions against the density exponent. |
| Evidence and review | Formula correspondence: local check. Algebra and Lean compilation: merged PR #161 build record. Implementation correspondence beyond the formulas: to verify. Review: pending. |

## EF-04: Gamma classical/natural round trip, mean and variance positivity

| Field | Content |
|---|---|
| Statement in prose | The map $(\alpha,\beta)\mapsto(\alpha-1,-\beta)$ and its inverse $(\theta_1,\theta_2)\mapsto(\theta_1+1,-\theta_2)$ compose to the identity in both orders for all reals. For $\alpha>0,\beta>0$, mean $\alpha/\beta$ and variance $\alpha/\beta^2$ are positive, and for $\beta\ne0$ the variance equals mean divided by $\beta$. |
| Lean signature | `gamma_from_natural_round_trip`, `gamma_natural_round_trip`, `gamma_var_pos (p : GammaParams) : p.Valid → 0 < gammaVar p`, `gamma_mean_pos`, `gamma_var_eq_mean_div_beta (p) : p.beta ≠ 0 → gammaVar p = gammaMean p / p.beta` |
| Assumptions | Round trips: none. Positivity: `p.Valid` (`0 < alpha ∧ 0 < beta`). |
| Why this is the intended spec | `gammaMean` and `gammaVar` are the closed formulas of `Gamma.mean` and `Gamma.var`; the card does not claim they are the moments of the Gamma law (that needs the distribution instance, DM-01/DM-02). |
| Covered normix function | `normix/distributions/gamma.py`: `natural_params`, `from_natural`, `mean`, `var`. |
| Risks of a weakened spec | Positivity alone permits wrong positive scale factors. W3 M1 must connect actual `Gamma.var`, with both parameters varying, to W2's `gammaVar`, $\alpha/\beta^2$, then derive positivity. Distributional moments remain DM-01/DM-02; constructor clamps (`THETA_FLOOR`) are excluded. |
| Evidence and review | Local check for the formulas; Lean compilation checked. Distributional meaning of the formulas: not covered. Review: pending. |

## EF-06: validity preservation and the historical GIG clamp

| Field | Content |
|---|---|
| Statement in prose | A GIG parameter triple has $a>0,b>0$ iff its natural image has $\theta_1<0,\theta_2<0$; the same holds for the unclamped and the clamped inverse. The clamped inverse always has $a,b\ge0$. The clamp on $b$ (resp. $a$) is inactive exactly when $\theta_1\le0$ (resp. $\theta_2\le0$), so clamped and unclamped inverses agree on the valid natural domain, and `gigFromNatural (gigNatural q) = q` for valid $q$. |
| Lean signature | `gig_natural_valid_iff`, `gig_from_natural_unclamped_valid_iff`, `gig_from_natural_valid_iff`, `gig_from_natural_b_nonneg`, `gig_from_natural_a_nonneg`, `gig_b_clamp_inactive_iff`, `gig_a_clamp_inactive_iff`, `gig_from_natural_eq_unclamped (t) : t.Valid → gigFromNatural t = gigFromNaturalUnclamped t`, `gig_from_natural_round_trip (q) : q.Valid → gigFromNatural (gigNatural q) = q`, `gig_natural_round_trip` |
| Assumptions | `GigParams.Valid` is `0 < a ∧ 0 < b`; `GigNatural.Valid` is `theta1 < 0 ∧ theta2 < 0`. $p$ is unrestricted. |
| Why this is the intended spec | It separates the formula from the historical constructor clamp and states exactly where they coincide. The valid domain is the GIG interior; Gamma and InverseGamma boundary cases ($a=0$ or $b=0$) are excluded. |
| Covered normix function | `GeneralizedInverseGaussian.natural_params` and `_from_natural_unchecked` algebra. Public `from_natural` rejects positive rate coordinates with `eqx.error_if`, without a tolerance. Historical clamp theorems do not cover that rejection. |
| Risks of a weakened spec | The strict interior excludes zero rates accepted by Python. A sign check does not establish all boundary laws. A certificate for `_from_natural_unchecked` must name that method and cannot silently claim the public error path. Other constructor floors need separate correspondence. |
| Evidence and review | Historical clamp model: Lean checked. Current rejection and transformation behavior: `tests/test_gig_expectation_domain.py`. Runtime validation is not proved in Lean. Review: pending. |

## EF-07: normalization of finite positive exponential weights

| Field | Content |
|---|---|
| Statement in prose | For a nonempty finite index set and any real log weights, $p_i=\exp(\ell_i-\log\sum_j e^{\ell_j})$ is positive, equals $e^{\ell_i}/\sum_j e^{\ell_j}$, and the $p_i$ sum to one. |
| Lean signature | `Normix.Numerics.normalized_weights_pos (logW : Fin n → ℝ) (i) : 0 < normalizedWeights logW i`; `normalized_weights_eq_div [NeZero n]`; `normalized_weights_sum [NeZero n] (logW : Fin n → ℝ) : ∑ i, normalizedWeights logW i = 1` |
| Assumptions | `[NeZero n]` (nonempty) for the identity and the sum; positivity holds without it. |
| Why this is the intended spec | `normalizedWeights` mirrors `p = exp(log_w - logsumexp(log_w))`, with logsumexp modeled as the log of a real sum, not a stabilized floating-point routine. |
| Covered normix function | `normix/utils/bessel.py`: `_moments_from_weights` (weight normalization step). |
| Risks of a weakened spec | Underflow, `-inf` log weights, and numerically stabilized logsumexp are not modeled; positivity is exact only for finite reals. |
| Evidence and review | Local check for the formula; Lean checked. Review: pending. |

## EF-08: fixed-geometry derivative of the finite log partition

| Field | Content |
|---|---|
| Statement in prose | For positive base weights $w_i$ and real locations $x_i$ on a nonempty index set, $\psi(\theta)=\log\sum_i w_ie^{\theta x_i}$ is differentiable at every real $\theta$ and $\psi'(\theta)=\sum_i q_i(\theta)x_i$ with $q_i(\theta)=w_ie^{\theta x_i}/\sum_jw_je^{\theta x_j}$. |
| Lean signature | `Normix.Numerics.fixed_geometry_log_partition_has_deriv_at [NeZero n] (w x : Fin n → ℝ) (hw : ∀ i, 0 < w i) (theta : ℝ) : HasDerivAt (fixedGeometryLogPartition w x) (∑ i, fixedGeometryWeights w x theta i * x i) theta` |
| Assumptions | $w_i>0$, nonempty index set; nodes and weights are fixed, independent of $\theta$. |
| Why this is the intended spec | The derivative object is the function with geometry frozen, as the checklist requires; it is not a statement about the quadrature that moves its window with the argument. The result is a real-analysis derivative, not a statement about JAX autodiff. |
| Covered normix function | Mathematical model of `_moments_from_weights` / `log_kv_moments` with frozen nodes (one statistic coordinate only). |
| Risks of a weakened spec | One dimension only; multidimensional EF-08 and Hessian/covariance EF-10 are the remaining W2 M1 target (5–10 days). Equality with JAX AD under frozen geometry needs W3; Bessel uses `stop_gradient`, not a custom JVP. |
| Evidence and review | Derivation and Lean check. Review: pending. |

## EF-09: centered covariance is a positive-semidefinite Gram matrix

| Field | Content |
|---|---|
| Statement in prose | For finite statistic vectors $s_i\in\mathbb R^d$ and weights $p_i$, let $m=\sum_ip_is_i$ and $C_{ab}=\sum_ip_i(s_{ia}-m_a)(s_{ib}-m_b)$. Then $v^\top Cv=\sum_ip_i(\sum_jv_j(s_{ij}-m_j))^2$ for every $v$ and arbitrary real weights; for nonnegative weights $C$ is symmetric and positive semidefinite. If $\sum_ip_i=1$, then $C_{ab}=\sum_ip_is_{ia}s_{ib}-m_am_b$. |
| Lean signature | `Normix.Numerics.centered_gram_identity`, `centered_gram_nonneg (p s) (hp : ∀ i, 0 ≤ p i) (v) : 0 ≤ ∑ a, ∑ b, v a * centeredCov p s a b * v b`, `centered_cov_symm`, `centered_cov_pos_semidef (hp) : (centeredCov p s).PosSemidef`, `centered_cov_eq_raw_sub (hp : ∑ i, p i = 1)` |
| Assumptions | Nonnegative weights for semidefiniteness; unit total weight only for the raw-moment identity. No claim of strict positive definiteness. |
| Why this is the intended spec | `centeredCov` is defined by the centered Gram sum; the positive-semidefinite conclusion is proved via the sum-of-squares identity, not assumed. |
| Covered normix function | `normix/utils/bessel.py`: `_moments_from_weights` (`cov = R^T R`, `R_i = sqrt(p_i)(s_i - mean)`), `BesselMoments.cov`. The symmetrization `0.5 (cov + cov^T)` is a no-op here because `cov` is symmetric. The specific statistic $s=(x,\mathrm{expm1}(-x),\mathrm{expm1}(x))$ is not instantiated. |
| Risks of a weakened spec | Floating-point semidefiniteness (eigenvalue signs) is not covered. Strict positivity needs separate assumptions on the statistic vectors. Equality with the matrix returned by the Python function needs the W3 bridge. |
| Evidence and review | Local check for the structure; Lean checked. Review: pending. |
