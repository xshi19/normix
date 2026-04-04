# Package Improvement Roadmap

Based on [package_review_2026-03-30](../reviews/package_review_2026-03-30.md).

## Issues and Recommendations

### Summary Table

| ID | Type | Issue / Recommendation | Priority | Difficulty | Est. LOC | Impact | Prerequisites | Multi-phase |
|----|------|------------------------|----------|------------|----------|--------|---------------|-------------|
| B1 | Bug | ~~`InverseGaussian.cdf` returns NaN in high-shape regimes (`lam/mu` large).~~ **Fixed:** log-space second term via `log_ndtr`; moderate + extreme CDF tests vs SciPy (`02d2a8e`, `0c9f5c5`). | ✅ | Medium | ~30 | Done. | None | No |
| B2 | Bug | ~~`OnlineEMFitter`/`MiniBatchEMFitter` algorithmically mis-specified.~~ **Fixed:** replaced by `IncrementalEMFitter` with correct Robbins-Monro, EWMA, etc. | ✅ | High | ~450 | Done. | D1. | Yes |
| B3 | Bug | ~~`BatchEMFitter._fit_scan` double-counts the terminating iteration in `n_iter`.~~ **Fixed:** explicit iteration counter in scan carry (`02d2a8e`). | ✅ | Low | ~15 | Done. | None | No |
| B4 | Bug | ~~`JointNormalInverseGamma._subordinator_log_partition` passes wrong sign convention, returns NaN.~~ **Fixed:** deleted along with all four `_subordinator_log_partition` implementations (Phase 4 — method was never called). | ✅ | Low | ~10 | Done. | None | No |
| B5 | Bug | ~~`InverseGaussian` module docstring has wrong sign for the log-partition formula.~~ **Fixed:** corrected ψ sign (`02d2a8e`). | ✅ | Trivial | ~2 | Done. | None | No |
| D1 | Design | ~~Decide the fate of `OnlineEMFitter` / `MiniBatchEMFitter`.~~ **Resolved:** replaced by `IncrementalEMFitter` + `EtaUpdateRule`. | ✅ | — | 0 | Done. | None | No |
| D2 | Design | ~~Decide public status of joint distribution classes~~ **Resolved:** public `ExponentialFamily` joints; `log_prob` / `sufficient_statistics` use flat `concat(x,[y])`; `from_natural` on joints still unimplemented (optional follow-up). GIG-based joints use the same `-b/2`, `-a/2` scaling on `1/y`, `y` as standalone GIG. Documented in `docs/design/design.md`. | ✅ | — | 0 | Done. | None | No |
| D3 | Design | ~~Rationalize `MultivariateNormal` relative to the rest of the package.~~ **Done:** Made full `ExponentialFamily` subclass; added `_log_partition_from_theta`, `natural_params`, `sufficient_statistics`, `log_base_measure`, `from_natural`, `mean`, `cov`, `rvs`; full EF round-trip tests added (Phase 7). | ✅ | High | ~80–150 | Done. | D2. | Yes |
| D4 | Design | ~~Evaluate `jaxopt` dependency risk — upstream is unmaintained and emitting warnings.~~ **Resolved:** jaxopt kept for LBFGS/BFGS (only JAX-native quasi-Newton with reparameterization); `DeprecationWarning` suppressed at import; migration path to optax/optimistix documented in `docs/design/design.md` § jaxopt migration (Phase 7). | ✅ | High | ~100–200 | Done. | None | Yes |
| C1 | Code style | Public type annotations incomplete: missing return types on `log_kv`, several GH/VG/NIG methods, `NormalMixture.joint`. | P3 | Low | ~30 | **Low** — IDE support, maintainability. | None | No |
| C2 | Code style | `jax.config.update("jax_enable_x64", True)` repeated across many modules instead of centralized at package init. | P3 | Low | ~20 | **Low** — hygiene; reduces confusion about when x64 is active. | None | No |
| C3 | Code style | ~~Dead / drifted code pockets: unused `_subordinator_log_partition` implementations, stale compatibility leftovers.~~ **Fixed:** all four `_subordinator_log_partition` implementations and the abstract declaration removed (Phase 4). | ✅ | Low–Med | ~30–60 | Done. | D2. | No |
| C4 | Code style | GH `default_init` is heavy and exception-swallowing; trades robustness for opacity. | P3 | Medium | ~30 | **Low** — user-facing robustness, but works in practice. | None | No |
| T1 | Testing | ~~Nearly half the test suite (246/506) is skipped — tests target old API.~~ **Fixed:** all 246 skipped tests rewritten for current API; suite: 598 passed, 0 skipped (Phase 3). | ✅ | High | ~300–500 | Done. | None | Yes |
| T2 | Testing | ~~Add mathematical invariants test layer: ∇ψ = η, Hessian SPD, density vs. SciPy.~~ **Done:** invariants added to `test_exponential_family.py` and `test_distributions_vs_scipy.py`; covers Gamma, InverseGamma, InverseGaussian, GIG (Phase 3). | ✅ | Medium | ~100–150 | Done. | None | No |
| T3 | Testing | ~~Add exponential-family round-trip tests for joint distribution classes.~~ **Done:** `TestJointExponentialFamilyRoundTrip` covers all four joints in `test_jax_distributions.py` (Phase 4). | ✅ | Medium | ~60–100 | Done. | D2. | No |
| T4 | Testing | ~~Add extreme-parameter tests for all functions using exponentials outside log-space.~~ **Done:** `test_extreme_parameters.py` covers large/small shape, near-boundary, overflow/underflow (Phase 3). | ✅ | Medium | ~50–80 | Done. | B1. | No |
| T5 | Testing | ~~Add tests for `OnlineEMFitter`/`MiniBatchEMFitter`.~~ **Done:** `test_incremental_em.py` (29 tests). | ✅ | Medium | ~350 | Done. | B2. | No |
| T6 | Testing | Add more property-based and edge-case tests for GIG. | P3 | Medium | ~60–80 | **Medium** — GIG is the most numerically critical module. | None | No |
| DOC1 | Docs | ~~Fix broken `README.md` examples~~ **Fixed:** imports, EM kwargs, layout (`e8db92a`). | ✅ | Low | ~30 | Done. | None | No |
| DOC2 | Docs | ~~Fix `normix/__init__.py` docstring~~ **Fixed:** `default_init` + instance `model.fit(X)` pattern (`e8db92a`). | ✅ | Low | ~10 | Done. | None | No |
| DOC3 | Docs | ~~Update `docs/theory/em_algorithm.rst` — still references old methods.~~ **Fixed:** "Implementation in normix" section rewritten to current `e_step`/`m_step`/`conditional_expectations`/`solve_bregman` API (Phase 4). | ✅ | Medium | ~30–50 | Done. | None | No |
| DOC4 | Docs | Add executable doc checks in CI: README code blocks, module doctest snippets, notebook smoke checks. | P2 | Medium | ~50–80 | **Medium** — prevents future documentation drift. | DOC1, DOC2 (examples must be correct first). | Yes |
| DOC5 | Docs | ~~Mark historical design notes and investigations as historical.~~ **Done:** `solver_redesign.md` status updated from "Proposal (v2)" to "Implemented" (Phase 4). | ✅ | Low | ~20 | Done. | None | No |
| DOC6 | Docs | Add executable documentation examples for `log_kv`. | P3 | Low | ~15 | **Low** — improves onboarding for the most central utility. | None | No |
| PKG1 | Packaging | `jax[cuda12]` is a hard runtime dependency — unusually opinionated for a library. Make base install CPU-neutral. | P2 | Low | ~10 | **High** — blocks CPU-only users and inflates install size. | None | No |
| PKG2 | Packaging | `matplotlib` is in core dependencies but only used in notebooks/plotting. Move to optional extras. | P2 | Low | ~10 | **Medium** — smaller core install footprint. | None | No |

### Counts by Priority (original review)

| Priority | Bugs | Design | Code Style | Testing | Docs | Packaging | **Total** |
|----------|------|--------|------------|---------|------|-----------|-----------|
| P1       | 2    | 1      | —          | 2       | 2    | —         | **7**     |
| P2       | 2    | 1      | 1          | 3       | 2    | 2         | **11**    |
| P3       | 1    | 2      | 3          | 1       | 2    | —         | **9**     |
| **Total** | **5** | **4** | **4**     | **6**   | **6** | **2**   | **27**    |

**Resolved as of Phase 7:** B1–B5, D1–D4, C3, T1–T5, DOC1–DOC3, DOC5 (22/27). Remaining open: C1, C2, C4, T6, DOC4, DOC6, PKG1, PKG2.

---

## Improvement Roadmap

### Dependency Graph

```
D1 ──► B2 ──► T5
D2 ──► D3, T3, C3
B1 ──► T4
DOC1, DOC2 ──► DOC4
```

All other items are independent.

---

### Phase 0 — Design Decisions ✅ DONE

**Goal:** Make the two blocking architectural decisions so downstream work can proceed.

| Item | Action | Status |
|------|--------|--------|
| **D1** | Decide: implement real stochastic EM, or deprecate `OnlineEMFitter`/`MiniBatchEMFitter`. | ✅ Replaced by `IncrementalEMFitter` + `EtaUpdateRule` |
| **D2** | Decide: joint classes are full public exponential-family objects, or internal helpers. | ✅ Public `ExponentialFamily`; see `docs/design/design.md` § Joint classes as public exponential families (D2). |

**Exit criteria:** Each decision documented in `docs/design/design.md` with rationale. (D1, D2 satisfied.)

---

### Phase 1 — Critical Bugs & Broken Docs ✅ DONE

**Goal:** Fix all P1 correctness bugs and restore a working first-use experience.

**Completed.** Code and tests: `02d2a8e` (IG CDF log-space rewrite, IG docstring, `_fit_scan` `n_iter`) and `0c9f5c5` (IG CDF moderate + extreme tests vs SciPy). README and package docstring: `e8db92a`.

| Item | Description | Status |
|------|-------------|--------|
| **B1** | `InverseGaussian.cdf` via `log_ndtr`; moderate + extreme tests. | ✅ Done |
| **B5** | IG module docstring ψ sign. | ✅ Done |
| **B3** | `_fit_scan` iteration counter in scan carry (correct `n_iter`). | ✅ Done |
| **DOC1** | `README.md` examples runnable. | ✅ Done |
| **DOC2** | `normix/__init__.py` quick-start docstring. | ✅ Done |

**Exit criteria (met):** `InverseGaussian(mu=1.0, lam=1000.0).cdf(1.0)` is correct; README snippets copy-paste; scan path reports correct `n_iter`.

---

### Phase 2 — Fitter Repair (depends on D1) ✅ DONE

**Goal:** Make the EM fitter family correct and tested.

**Completed.** D1 decision: replace `OnlineEMFitter`/`MiniBatchEMFitter` with
`IncrementalEMFitter` + pluggable `EtaUpdateRule` (eqx.Module) η-update rules.

| Item | Description | Status |
|------|-------------|--------|
| **B2** | `IncrementalEMFitter` replaces old fitters; `BatchEMFitter` gains optional `eta_update`. `NormalMixtureEta` pytree + `affine_combine` + 6 concrete rules (Identity, RobbinsMonro, SampleWeighted, EWMA, Shrinkage, Affine). | ✅ Done |
| **T5** | `tests/test_incremental_em.py`: 29 tests covering `compute_eta_from_model`, `affine_combine`, rule contracts, all fitter variants, shrinkage batch EM, pytree leaves. | ✅ Done |
| **D1** | Documented in `docs/design/design.md` § EM Framework. | ✅ Done |

---

### Phase 3 — Test Migration & Coverage ✅ DONE

**Goal:** Dramatically reduce the skipped test count and defend the public API surface.

**Completed.** All 246 skipped tests rewrote for the current API. Suite went from
`299 passed, 246 skipped` → `586 passed, 0 skipped`. New `test_extreme_parameters.py`
covers T4 extreme-parameter regimes across all distributions.

| Item | Description | Status |
|------|-------------|--------|
| **T1** | Rewrote 10 skipped test files to current API: `test_distributions_vs_scipy.py`, `test_variance_gamma.py`, `test_normal_inverse_gamma.py`, `test_normal_inverse_gaussian.py`, `test_generalized_hyperbolic.py`, `test_em_regression.py`, `test_multivariate_rvs.py`, `test_exponential_family.py`, `test_gig_special_cases.py`, `test_high_dimensional.py`, `test_sp500_distribution_validation.py`. | ✅ Done |
| **T2** | Mathematical invariants (∇ψ = η, Hessian SPD) added to `test_exponential_family.py` and `test_distributions_vs_scipy.py`. Covers Gamma, InverseGamma, InverseGaussian, GIG. | ✅ Done |
| **T4** | Extreme-parameter tests in new `test_extreme_parameters.py`: large/small shape, near-boundary, overflow/underflow for all univariate and mixture distributions. | ✅ Done |

**Exit criteria (met):** Skipped count = 0 (target was < 50); all EF distributions have invariant tests; no NaN in extreme-parameter tests.

---

### Phase 4 — API Consistency & Dead Code (D2 resolved) ✅ DONE

**Goal:** Align the exponential-family contract, clean dead code, tighten the public surface.

**Completed.** All five items delivered.

| Item | Description | Status |
|------|-------------|--------|
| **B4** | Deleted `JointNormalInverseGamma._subordinator_log_partition` (wrong sign convention, never called). | ✅ Done |
| **C3** | Removed `_subordinator_log_partition` from all four joint subclasses and the abstract declaration from `JointNormalMixture`; method was never called anywhere. | ✅ Done |
| **T3** | Added `TestJointExponentialFamilyRoundTrip` to `test_jax_distributions.py`: EF contract (log_partition self-consistency, expectation_params = ∇ψ), `rvs` finiteness, and `conditional_expectations` sign correctness for all four joints. | ✅ Done |
| **DOC3** | Updated `docs/theory/em_algorithm.rst` "Implementation in normix" section: replaced stale `_conditional_expectation_y_given_x`, `joint.set_expectation_params`, `joint._expectation_to_natural` with current `e_step`, `m_step`, `conditional_expectations`, `solve_bregman`. | ✅ Done |
| **DOC5** | Updated `docs/design/solver_redesign.md` status from "Proposal (v2)" to "Implemented". | ✅ Done |

**Exit criteria (met):** No dead `_subordinator_log_partition` code; theory docs match current API; new T3 tests pass.

---

### Phase 5 — Code Hygiene

**Goal:** Polish code style and centralize configuration.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **C1** | Add missing return type annotations on all public methods. | ~30 |
| **C2** | Centralize `jax_enable_x64` to a single entry point. | ~20 |
| **C4** | Improve GH `default_init` error handling transparency. | ~30 |
| **T6** | Add GIG property-based and edge-case tests. | ~60–80 |
| **DOC6** | Add executable examples for `log_kv`. | ~15 |

**Total estimated LOC:** ~155–175
**Exit criteria:** `rg "jax_enable_x64"` returns exactly one hit (in `__init__.py`); all public methods have return annotations; GIG edge-case test suite passes.

---

### Phase 6 — Packaging & CI

**Goal:** Make the package installable and testable with minimal friction.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **PKG1** | Make base install CPU-neutral; move `jax[cuda12]` to optional extra. | ~10 |
| **PKG2** | Move `matplotlib` to optional extras. | ~10 |
| **DOC4** | Add CI job: execute README code blocks, module doctests, notebook smoke checks. | ~50–80 |

**Total estimated LOC:** ~70–100
**Exit criteria:** `pip install normix` works on CPU-only machines; CI catches broken examples automatically.

---

### Phase 7 — Long-term ✅ DONE

**Goal:** Rationalize `MultivariateNormal` as a first-class distribution and document the jaxopt dependency situation.

**Completed.**

| Item | Description | Status |
|------|-------------|--------|
| **D3** | `MultivariateNormal` promoted to full `ExponentialFamily` subclass: `_log_partition_from_theta` (analytic via Cholesky of Λ), `natural_params`, `sufficient_statistics`, `log_base_measure`, `from_natural`, `mean`, `cov`, `rvs`. `log_prob` keeps efficient Cholesky override. EF round-trip tests added to `test_jax_distributions.py`. | ✅ Done |
| **D4** | jaxopt risk evaluated: currently the only JAX-native quasi-Newton optimizer compatible with reparameterized LBFGS/BFGS; migration plan (optax `scale_by_lbfgs` loop or Optimistix once box-constraint support matures) documented in `docs/design/design.md` § jaxopt migration. `DeprecationWarning` suppressed in `normix/__init__.py`. | ✅ Done |

**Exit criteria (met):** `MultivariateNormal` passes EF round-trip tests; jaxopt warning absent at import; migration plan documented.

---

## Phase Summary

| Phase | Theme | Est. LOC | Key Dependencies |
|-------|-------|----------|------------------|
| 0 | Design decisions | 0 | — |
| 1 | Critical bugs & docs | ~90 | — — **✅ DONE** |
| 2 | Fitter repair | ~210–400 | D1 — **✅ DONE** |
| 3 | Test migration & coverage | ~450–730 | Phase 1 (B1) — **✅ DONE** |
| 4 | API consistency & dead code | ~150–240 | D2 — **✅ DONE** |
| 5 | Code hygiene | ~155–175 | — |
| 6 | Packaging & CI | ~70–100 | DOC1, DOC2 |
| 7 | Long-term | ~180–350 | D2 — **✅ DONE** |

Phases 1, 5, and 6 are independent of the design decisions and can proceed in parallel.
Phase 4 (API consistency) was blocked on D2; D2 is resolved. Phase 2 was blocked on D1; D1 is resolved.
Phase 3 is complete: 0 skipped tests, all EF invariants covered, extreme-parameter tests pass.
Phase 4 is complete: dead `_subordinator_log_partition` removed, joint EF round-trip tests added, theory docs updated to current API.
Phase 7 is complete: `MultivariateNormal` is now a full `ExponentialFamily`; jaxopt migration plan documented and warning suppressed.
