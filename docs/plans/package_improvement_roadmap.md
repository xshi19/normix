# Package Improvement Roadmap

Based on [package_review_2026-03-30](../reviews/package_review_2026-03-30.md).

## Issues and Recommendations

### Summary Table

| ID | Type | Issue / Recommendation | Priority | Difficulty | Est. LOC | Impact | Prerequisites | Multi-phase |
|----|------|------------------------|----------|------------|----------|--------|---------------|-------------|
| B1 | Bug | ~~`InverseGaussian.cdf` returns NaN in high-shape regimes (`lam/mu` large).~~ **Fixed:** log-space second term via `log_ndtr`; moderate + extreme CDF tests vs SciPy (`02d2a8e`, `0c9f5c5`). | ✅ | Medium | ~30 | Done. | None | No |
| B2 | Bug | ~~`OnlineEMFitter`/`MiniBatchEMFitter` algorithmically mis-specified.~~ **Fixed:** replaced by `IncrementalEMFitter` with correct Robbins-Monro, EWMA, etc. | ✅ | High | ~450 | Done. | D1. | Yes |
| B3 | Bug | ~~`BatchEMFitter._fit_scan` double-counts the terminating iteration in `n_iter`.~~ **Fixed:** explicit iteration counter in scan carry (`02d2a8e`). | ✅ | Low | ~15 | Done. | None | No |
| B4 | Bug | `JointNormalInverseGamma._subordinator_log_partition` passes wrong sign convention, returns NaN. | P2 | Low | ~10 | **Low** — currently dead code, but a maintenance hazard. Fix or delete. | None | No |
| B5 | Bug | ~~`InverseGaussian` module docstring has wrong sign for the log-partition formula.~~ **Fixed:** corrected ψ sign (`02d2a8e`). | ✅ | Trivial | ~2 | Done. | None | No |
| D1 | Design | ~~Decide the fate of `OnlineEMFitter` / `MiniBatchEMFitter`.~~ **Resolved:** replaced by `IncrementalEMFitter` + `EtaUpdateRule`. | ✅ | — | 0 | Done. | None | No |
| D2 | Design | ~~Decide public status of joint distribution classes~~ **Resolved:** public `ExponentialFamily` joints; `log_prob` / `sufficient_statistics` use flat `concat(x,[y])`; `from_natural` on joints still unimplemented (optional follow-up). GIG-based joints use the same `-b/2`, `-a/2` scaling on `1/y`, `y` as standalone GIG. Documented in `docs/design/design.md`. | ✅ | — | 0 | Done. | None | No |
| D3 | Design | Rationalize `MultivariateNormal` relative to the rest of the package: add `rvs`, `mean`/`cov`, make it `ExponentialFamily`, or document as a lightweight helper. | P3 | High | ~80–150 | **Medium** — API inconsistency; users see it as a peer but it lacks standard methods. | D2 (public API boundary decision). | Yes |
| D4 | Design | Evaluate `jaxopt` dependency risk — upstream is unmaintained and emitting warnings. Plan migration path. | P3 | High | ~100–200 | **Medium** — long-term dependency risk; not an immediate blocker. | None | Yes |
| C1 | Code style | Public type annotations incomplete: missing return types on `log_kv`, several GH/VG/NIG methods, `NormalMixture.joint`. | P3 | Low | ~30 | **Low** — IDE support, maintainability. | None | No |
| C2 | Code style | `jax.config.update("jax_enable_x64", True)` repeated across many modules instead of centralized at package init. | P3 | Low | ~20 | **Low** — hygiene; reduces confusion about when x64 is active. | None | No |
| C3 | Code style | Dead / drifted code pockets: unused `_subordinator_log_partition` implementations, stale compatibility leftovers. | P2 | Low–Med | ~30–60 | **Medium** — maintenance hazard; confuses contributors. | D2 (need to know which code is "dead" vs. planned). | No |
| C4 | Code style | GH `default_init` is heavy and exception-swallowing; trades robustness for opacity. | P3 | Medium | ~30 | **Low** — user-facing robustness, but works in practice. | None | No |
| T1 | Testing | Nearly half the test suite (246/506) is skipped — tests target old API. Migrate to exercise current public surface. | P1 | High | ~300–500 | **High** — green suite materially overstates confidence; large parts of the API are weakly defended. | None | Yes |
| T2 | Testing | Add mathematical invariants test layer: ∇ψ = η, Hessian SPD, density vs. SciPy across moderate and extreme regimes. | P2 | Medium | ~100–150 | **Medium** — catches regressions in exponential-family contract. | None | No |
| T3 | Testing | Add exponential-family round-trip tests for joint distribution classes (`from_natural` / `from_expectation` if implemented). | P2 | Medium | ~60–100 | **Medium** — validates the contract advertised in docs. | D2 (public status decision). | No |
| T4 | Testing | Add extreme-parameter tests for all functions using exponentials outside log-space. | P2 | Medium | ~50–80 | **Medium** — catches overflow/underflow in production regimes. | B1 (IG CDF fix first). | No |
| T5 | Testing | ~~Add tests for `OnlineEMFitter`/`MiniBatchEMFitter`.~~ **Done:** `test_incremental_em.py` (29 tests). | ✅ | Medium | ~350 | Done. | B2. | No |
| T6 | Testing | Add more property-based and edge-case tests for GIG. | P3 | Medium | ~60–80 | **Medium** — GIG is the most numerically critical module. | None | No |
| DOC1 | Docs | ~~Fix broken `README.md` examples~~ **Fixed:** imports, EM kwargs, layout (`e8db92a`). | ✅ | Low | ~30 | Done. | None | No |
| DOC2 | Docs | ~~Fix `normix/__init__.py` docstring~~ **Fixed:** `default_init` + instance `model.fit(X)` pattern (`e8db92a`). | ✅ | Low | ~10 | Done. | None | No |
| DOC3 | Docs | Update `docs/theory/em_algorithm.rst` — still references old methods (`joint.set_expectation_params`, `_expectation_to_natural`). | P2 | Medium | ~30–50 | **Medium** — theory docs no longer describe current code. | None | No |
| DOC4 | Docs | Add executable doc checks in CI: README code blocks, module doctest snippets, notebook smoke checks. | P2 | Medium | ~50–80 | **Medium** — prevents future documentation drift. | DOC1, DOC2 (examples must be correct first). | Yes |
| DOC5 | Docs | Mark historical design notes and investigations as historical if they no longer describe current API. | P3 | Low | ~20 | **Low** — reduces confusion about which docs are current. | None | No |
| DOC6 | Docs | Add executable documentation examples for `log_kv`. | P3 | Low | ~15 | **Low** — improves onboarding for the most central utility. | None | No |
| PKG1 | Packaging | `jax[cuda12]` is a hard runtime dependency — unusually opinionated for a library. Make base install CPU-neutral. | P2 | Low | ~10 | **High** — blocks CPU-only users and inflates install size. | None | No |
| PKG2 | Packaging | `matplotlib` is in core dependencies but only used in notebooks/plotting. Move to optional extras. | P2 | Low | ~10 | **Medium** — smaller core install footprint. | None | No |

### Counts by Priority

| Priority | Bugs | Design | Code Style | Testing | Docs | Packaging | **Total** |
|----------|------|--------|------------|---------|------|-----------|-----------|
| P1       | 2    | 1      | —          | 2       | 2    | —         | **7**     |
| P2       | 2    | 1      | 1          | 3       | 2    | 2         | **11**    |
| P3       | 1    | 2      | 3          | 1       | 2    | —         | **9**     |
| **Total** | **5** | **4** | **4**      | **6**   | **6** | **2**    | **27**    |

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

### Phase 0 — Design Decisions

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

### Phase 3 — Test Migration & Coverage

**Goal:** Dramatically reduce the skipped test count and defend the public API surface.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **T1** | Migrate skipped tests to current API. Prioritize: public API examples → multivariate normal → joint EF round-trips → EM regressions. | ~300–500 |
| **T2** | Add mathematical invariants test layer (∇ψ = η, Hessian SPD, density vs. SciPy). | ~100–150 |
| **T4** | Add extreme-parameter tests for functions using exponentials outside log-space. | ~50–80 |

**Total estimated LOC:** ~450–730
**Exit criteria:** Skipped count drops below 50; all exponential-family distributions have invariant tests; no NaN in extreme-parameter tests.

---

### Phase 4 — API Consistency & Dead Code (D2 resolved)

**Goal:** Align the exponential-family contract, clean dead code, tighten the public surface.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **B4** | Fix or delete `JointNormalInverseGamma._subordinator_log_partition`. | ~10 |
| **C3** | Remove dead/drifted code identified during D2 decision. | ~30–60 |
| **T3** | Add round-trip tests for joint classes (`from_natural` / `from_expectation` if implemented); partial: `test_jax_distributions` checks `log_prob` vs `log_prob_joint` for all four joints. | ~60–100 |
| **DOC3** | Update `docs/theory/em_algorithm.rst` to current method names. | ~30–50 |
| **DOC5** | Mark historical design notes as historical. | ~20 |

**Total estimated LOC:** ~150–240
**Exit criteria:** No `NotImplementedError` in public-facing methods (or documented as internal); theory docs match current code; no dead code flagged in review.

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

### Phase 7 — Long-term (opportunistic)

| Item | Description |
|------|-------------|
| **D3** | Rationalize `MultivariateNormal`: add `rvs`, `mean`/`cov`, possibly make `ExponentialFamily`. |
| **D4** | Evaluate `jaxopt` migration path (optax, custom solver, etc.). |

These are larger architectural efforts that can be tackled opportunistically or as dedicated projects.

---

## Phase Summary

| Phase | Theme | Est. LOC | Key Dependencies |
|-------|-------|----------|------------------|
| 0 | Design decisions | 0 | — |
| 1 | Critical bugs & docs | ~90 | — |
| 2 | Fitter repair | ~210–400 | D1 — **✅ DONE** |
| 3 | Test migration & coverage | ~450–730 | Phase 1 (B1) |
| 4 | API consistency & dead code | ~150–240 | D2 |
| 5 | Code hygiene | ~155–175 | — |
| 6 | Packaging & CI | ~70–100 | DOC1, DOC2 |
| 7 | Long-term | ~180–350 | D2 |

Phases 1, 5, and 6 are independent of the design decisions and can proceed in parallel.
Phase 4 (API consistency) was blocked on D2; D2 is resolved. Phase 2 was blocked on D1; D1 is resolved.
Phase 3 benefits from Phase 1 fixes but can start in parallel for the test migration (T1) work.
