# Package Improvement Roadmap

Based on [package_review_2026-03-30](../reviews/package_review_2026-03-30.md).

## Issues and Recommendations

### Summary Table

| ID | Type | Issue / Recommendation | Priority | Difficulty | Est. LOC | Impact | Prerequisites | Multi-phase |
|----|------|------------------------|----------|------------|----------|--------|---------------|-------------|
| B1 | Bug | `InverseGaussian.cdf` returns NaN in high-shape regimes (`lam/mu` large). Rewrite second term in log-space using `log_ndtr`. | P1 | Medium | ~30 | **High** — correctness bug in a public method; affects scientific users in concentrated regimes. | None | No |
| B2 | Bug | `OnlineEMFitter` does not use its computed `step`; `MiniBatchEMFitter` runs ordinary batch E/M on each minibatch instead of Robbins-Monro averaging; `tau0` unused; `OnlineEMFitter` returns `converged=True` unconditionally. | P1 | High | ~150–300 | **High** — algorithmically mis-specified; users get different estimators than documented. | Design decision: implement properly vs. deprecate/remove (D1). | Yes |
| B3 | Bug | `BatchEMFitter._fit_scan` double-counts the terminating iteration in `n_iter`. | P2 | Low | ~15 | **Medium** — convergence diagnostics are inaccurate; scan vs. loop paths report different counts. | None | No |
| B4 | Bug | `JointNormalInverseGamma._subordinator_log_partition` passes wrong sign convention, returns NaN. | P2 | Low | ~10 | **Low** — currently dead code, but a maintenance hazard. Fix or delete. | None | No |
| B5 | Bug | `InverseGaussian` module docstring has wrong sign for the log-partition formula. Code is correct. | P3 | Trivial | ~2 | **Low** — misleads readers; no runtime effect. | None | No |
| D1 | Design | Decide the fate of `OnlineEMFitter` / `MiniBatchEMFitter`: implement genuine stochastic EM, or deprecate/remove. | P1 | — | 0 | **High** — blocks B2 resolution. | None | No |
| D2 | Design | Decide public status of joint distribution classes: full public exponential-family objects (finish `from_natural` etc.) or internal helpers (document as such). | P2 | — | 0 | **High** — blocks multiple downstream tasks (T3, C3). | None | No |
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
| T5 | Testing | Add dedicated tests for `OnlineEMFitter` / `MiniBatchEMFitter`: verify `tau0` dependence, compare trajectories to batch EM. | P1 | Medium | ~60–100 | **High** — validates that stochastic fitters work as advertised. | B2 (fitter implementation). | No |
| T6 | Testing | Add more property-based and edge-case tests for GIG. | P3 | Medium | ~60–80 | **Medium** — GIG is the most numerically critical module. | None | No |
| DOC1 | Docs | Fix broken `README.md` examples: missing imports, wrong method names (`m_step_solver` → `m_step_backend`/`m_step_method`), `fit` called as classmethod. | P1 | Low | ~30 | **High** — first-use experience is currently broken. | None | No |
| DOC2 | Docs | Fix `normix/__init__.py` docstring: `GeneralizedHyperbolic.fit(X, key=key, ...)` is shown as a classmethod but `fit` is an instance method. | P1 | Low | ~10 | **High** — package-level docstring is wrong. | None | No |
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

### Phase 0 — Design Decisions (no code changes)

**Goal:** Make the two blocking architectural decisions so downstream work can proceed.

| Item | Action |
|------|--------|
| **D1** | Decide: implement real stochastic EM, or deprecate `OnlineEMFitter`/`MiniBatchEMFitter`. |
| **D2** | Decide: joint classes are full public exponential-family objects, or internal helpers. |

**Exit criteria:** Each decision documented in `docs/design/design.md` with rationale.

---

### Phase 1 — Critical Bugs & Broken Docs

**Goal:** Fix all P1 correctness bugs and restore a working first-use experience.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **B1** | Rewrite `InverseGaussian.cdf` in log-space (`log_ndtr`). Add extreme-regime tests. | ~30 |
| **B5** | Fix IG docstring sign error. | ~2 |
| **DOC1** | Fix all `README.md` examples: add imports, fix method names. | ~30 |
| **DOC2** | Fix `normix/__init__.py` docstring (instance method, not classmethod). | ~10 |
| **B3** | Fix `_fit_scan` iteration accounting (store explicit stop index in scan carry). | ~15 |

**Total estimated LOC:** ~90
**Exit criteria:** `InverseGaussian(mu=1.0, lam=1000.0).cdf(1.0)` returns correct value; all README snippets are copy-paste runnable; scan path reports correct `n_iter`.

---

### Phase 2 — Fitter Repair (depends on D1)

**Goal:** Make the EM fitter family correct and tested.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **B2** | Implement or remove `OnlineEMFitter`/`MiniBatchEMFitter` per D1 decision. | ~150–300 |
| **T5** | Add dedicated stochastic-fitter tests (or removal tests if deprecated). | ~60–100 |

**Total estimated LOC:** ~210–400
**Exit criteria:** If kept, stochastic fitters pass trajectory comparison tests against batch EM. If removed, all references cleaned up and `__init__` exports updated.

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

### Phase 4 — API Consistency & Dead Code (depends on D2)

**Goal:** Align the exponential-family contract, clean dead code, tighten the public surface.

| Item | Description | Est. LOC |
|------|-------------|----------|
| **B4** | Fix or delete `JointNormalInverseGamma._subordinator_log_partition`. | ~10 |
| **C3** | Remove dead/drifted code identified during D2 decision. | ~30–60 |
| **T3** | Add round-trip tests for joint classes (if D2 = public). | ~60–100 |
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
| 2 | Fitter repair | ~210–400 | D1 |
| 3 | Test migration & coverage | ~450–730 | Phase 1 (B1) |
| 4 | API consistency & dead code | ~150–240 | D2 |
| 5 | Code hygiene | ~155–175 | — |
| 6 | Packaging & CI | ~70–100 | DOC1, DOC2 |
| 7 | Long-term | ~180–350 | D2 |

Phases 1, 5, and 6 are independent of the design decisions and can proceed in parallel.
Phases 2 and 4 are blocked until D1 and D2 are resolved respectively.
Phase 3 benefits from Phase 1 fixes but can start in parallel for the test migration (T1) work.
