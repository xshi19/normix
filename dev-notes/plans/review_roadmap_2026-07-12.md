# Package Review Roadmap (2026-07-12)

> **ACTIVE — not started (2026-07-16).**
> Based on [package_review_2026-07-12](../reviews/package_review_2026-07-12.md)
> (1018 fast tests green; the mathematical core is verified correct — every
> re-derived density, gradient, Hessian, and M-step formula matches the
> implementation). The findings are edge-hardening, hygiene, efficiency, and
> docs; nothing at the level of a wrong density or broken estimator.
> Predecessor: the 2026-03-30 review roadmap, all 27 items resolved, archived
> at [`../archive/plans/package_improvement_roadmap.md`](../archive/plans/package_improvement_roadmap.md).

## Conventions

- ID prefixes: **B** correctness, **D** design/API, **C** dead code,
  **E** efficiency, **DOC** stale docs/rules, **W** website, **T** tests,
  **F** features. `§` cites the review section carrying the evidence.
- Priority: **P1** — silent wrong numbers, actively misleading agent rules,
  or large cheap wins; **P2** — should do; **P3** — opportunistic.
- Measured numbers from the review are embedded as test targets so
  implementers don't re-derive them.

## Item Catalog

### Correctness (B) — review §1

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| B1 | 1.1, 8.1 | Cross-family `squared_hellinger(p, q)` / `kl_divergence(p, q)` evaluate `type(p)`'s ψ at *q*'s θ; the special-case ψs reinterpret θ under their own constraint, so mismatched families return plausible wrong numbers. Measured: `H²(NIG, GH_p=0.7) = 0.074043` vs truth `0.017019` (the reverse direction is correct because GH's ψ reads any θ). | Per **DEC-1**: in Tier 2/3 dispatch, when `type(p) is not type(q)`, lift both operands via `to_generalized_hyperbolic()` and compare in GH coordinates; raise when no lift exists. Touches `normix/divergences.py` Tier 3, `ExponentialFamily.squared_hellinger`/`kl_divergence`, and the `NormalMixture` delegating methods. Regression tests: the measured example asserts ≈ 0.017019 in *both* argument orders; exact-embedding case stays 0. DEC-1 must also scope the univariate EF level: Gamma vs InverseGamma θ share shape `(2,)`, so the same silent-wrong-number mode exists there (lift to GIG, or raise). ~50 LOC + tests. | P1 |
| B2 | 1.2 | `renyi` value at α = 1 is right but `jax.grad` through the `jnp.where` singularity guard returns 0.0 there; truth is −V_H/2 (Gamma(2,1): −0.3225). | Taylor branch `H − V_H(α−1)/2` inside an \|1−α\| < ε window with both `jnp.where` branches NaN-free (double-where), or a `jax.custom_jvp`. Regression test: `jax.grad(dist.renyi)(1.0) ≈ -dist.varentropy()/2` (lands with T4). ~15 LOC. | P1 |
| B3 | 1.3 | `GIG.fisher_information(backend='jax')` mixed entries H₁₂/H₁₃ are 4.5%/2.2% off at (p=0.7, a=1.4, b=0.9) — the integer-shift FD for ∂²log K_ν/∂ν∂z is documented for the Newton solver but leaks into a public API with no caveat. | Either FD the mixed term with ±`BESSEL_EPS_V` (4 extra Bessel evals, orders p±ε±1), or document the accuracy asymmetry on `fisher_information` and point to `backend='cpu'` / `jax.hessian`. Pick during implementation; interrogate review on the PR (GIG Hessian). ~25 LOC. | P2 |
| B4 | 1.4 | `GIG.cdf`/`ppf` call `float(self.p)` to pick the degenerate-limit branch → `jax.jit(g.cdf)` raises; contradicts the broadly advertised JIT-compatibility. | `lax.cond` on the degeneracy test (both branches traceable), or document the exception on the methods and in `ARCHITECTURE.md`. The `Univariate*` mixin `cdf`/`ppf` are pure JAX and unaffected. ~30 LOC. | P2 |
| B5 | 1.5 | `BatchEMFitter._fit_loop` refuses to converge at iteration 1 (`max_change < tol and i > 0`; `converged` requires `n_iter > 1`) while the `lax.scan` path can converge at step 1 — same fitter, two stopping semantics. | Unify; test that both paths agree on a fit that converges in one step. Interrogate review (EM fitter). ~10 LOC. | P2 |
| B6 | 1.5 | `EMResult` type drift on the scan path: `converged` is a JAX bool array, `n_iter` an int32 array against an `int` annotation; `result.converged is True` fails. | Cast to Python `bool`/`int` at the result boundary; test both paths. ~10 LOC. | P2 |
| B7 | 1.5 | `quantile_cmc` brackets ±5·std around the PINV ppf; if the CMC root falls outside (tiny `len(Y)`, extreme α), bisection silently returns an endpoint. | Bracket width tied to α, or an endpoint assertion. ~10 LOC + test. | P3 |
| B8 | 1.5 | `_UnivariateNormalMixtureMixin.log_prob` on a batched `(n,)` input fails loudly but with a cryptic `solve_triangular` shape error. | Eager `ndim` check with a "use `jax.vmap`" hint. ~8 LOC + test. | P3 |

### Design / API consistency (D) — review §2

All four need their Phase-0 decision (design.md row) before code.

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| D1 | 2 | Three spellings for "the subordinator": `model.joint.subordinator()` (method), `factor_model.subordinator` (field), `univariate.subordinator` (property) — a recurring doc/tutorial wart. | Uniform accessor on `MarginalMixture` per **DEC-2**. Constraint: an eqx field and a method cannot share the name `subordinator` on `FactorNormalMixture`, so uniformity implies renaming the field or wrapping it. Update tutorials/docs that spell it. ~40 LOC. | P2 |
| D2 | 2 | `MultivariateNormal.sigma` is a property while `NormalMixture.sigma()` is a method — same name, different call syntax, adjacent classes. | Align per **DEC-2** (coding conventions prefer methods for computed matrices). ~10 LOC. | P3 |
| D3 | 2 | VG/NInvG/GH `fit()` overrides exist only to flip default backends, re-list every parameter, and *narrow* the API (no `track_ll`, `eta_update`, `m_step_kwargs` without dropping to `BatchEMFitter`). | Per **DEC-3**: a `_fit_defaults()` classmethod consumed by the base `fit(**kwargs)` — one signature, pass-through preserved. ~60 LOC. | P2 |
| D4 | 2 | `finance/projection.py` is a 30-line module wrapping `model.project(w)` in a free function — "no wrapper that only forwards". | Per **DEC-4**: fold into `finance/__init__` or drop in favour of the method; update the finance tutorials and `finance_architecture.md` references. ~30 LOC. | P2 |

### Dead code (C) — review §3

One sweep PR; all trivially removable.

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| C1 | 3 | 15 ruff `F401`/`F811`/`F841` findings — unused imports/locals and a stale `# noqa: F811` (full table in review §3: `mixtures/joint.py`, `marginal.py`, `factor.py`, `generalized_hyperbolic.py`, `generalized_inverse_gaussian.py`, `fitting/eta.py`, `solvers.py`, `finance/projection.py`, `utils/validation.py`). | Remove all. Optionally add `ruff check --select F401,F811,F841` to CI to lock in (ruff is not yet a dev dep — decide in the PR). | P1 |
| C2 | 3 | `_gig_rvs_pinv` (`distributions/generalized_inverse_gaussian.py:174`) is a pure alias of `rvs_pinv` — violates the no-aliases convention. | Inline at the single call site (line 589). | P1 |
| C3 | 2, 3 | `MultivariateNormal.sample` (`distributions/normal.py:258`) is marked "Legacy API — prefer rvs" and referenced nowhere. | Delete. | P1 |
| C4 | 3 | `data/gig_test.csv` tracked but referenced by no test or script (verified: only the review mentions it). | `git rm`. | P1 |
| C5 | 3 | `BatchEMFitter._force_nonfinite_at_step` looks dead but is a deliberate hook for `test_em_regression.py`. | One comment pointing at the test. (Review non-issue otherwise; `bregman_objective` is exercised by tests — no action.) | P3 |

### Numerical efficiency (E) — review §4, ordered by expected impact

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| E1 | 4.1 | E-step aggregation materialises an `(n, d, d)` tensor: `jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y)` then `mean(axis=0)` — ~8 GB transient at n=100k, d=100. | Contract directly: `jnp.einsum('ni,nj,n->ij', ...) / n` in `NormalMixture._aggregate_eta` (`mixtures/marginal.py:433`) and `FactorNormalMixture._aggregate_stats` (`mixtures/factor.py:359`). Mathematically identical; existing EM tests guard it. ~4 LOC. | P1 |
| E2 | 4.2, 8.2 | CVaR triple-computes the VaR bisection: `build_quadratic_approximation` calls `value_w`/`gradient_w`/`hessian_w`, each re-projecting, rebuilding the PINV table, and re-running the 60-step CMC bisection for the same `(w, Y)`. | Fused `value_grad_hess_w` sharing one quantile solve — ~3× cut on the dominant cost of `TransactionCostProblem.solve` and any SQP-style use. ~80 LOC + tests pinning equality with the unfused path. | P2 |
| E3 | 4.3, 8.3 | PINV tables rebuilt on every `cdf`/`ppf` call (`_UnivariateNormalMixtureMixin._pinv_grids`, `GIG.cdf/ppf`, `InverseGaussian.ppf`): 4000 Bessel-heavy `log_prob` evals each time; modules are immutable so self-caching is out. | Per **DEC-5**: a small frozen quantile-table object (or documented `u_grid, x_grid = dist._pinv_grids()` reuse) so repeated quantile workloads amortise the build. ~60 LOC. | P3 |
| E4 | 4.4 | `log_kv` custom JVP always pays the ∂ν finite difference (2 extra Bessel evals) even when only the z-tangent is nonzero. | `jax.custom_jvp(..., symbolic_zeros=True)`; verify JVP correctness for both tangent patterns. Interrogate review (bessel.py, `custom_jvp`). ~15 LOC. | P2 |
| E5 | 4.5 | `_newton_digamma` (`distributions/gamma.py:204`) and `gammaincinv` (`utils/gammainc.py`) always run 50 `fori_loop` iterations (converged in <10). | `while_loop` with a tolerance, or cap at 20 — trims the VG/NInvG jax-backend M-step and Gamma/InvGamma `ppf`. ~20 LOC. | P3 |
| E6 | 4.6 | `GIG.var()` builds the full 7-Bessel Hessian for entry [2,2] (closed form: 3 Bessel evals). `NormalInverseGaussian._subordinator_expectations` routes E[Y] and E[1/Y] through the 5-Bessel GIG path though both are closed-form for InverseGaussian (only E[log Y] needs Bessel). | Closed forms; tests pin equality with the current values. ~30 LOC. | P2 |
| E7 | 4.7 | `MeanRiskProblem._reduced()` re-runs the Cholesky solves on every `weights`/`dispersion`/`risk_at` call; `A_inv` and `Sinv_M` are w-independent. | Precompute at construction (eqx fields). ~30 LOC. | P3 |

### Stale docs / rules (DOC) — review §5

Misleading agent-facing rules cost every future session; all fixes are edits.

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| DOC1 | 5 | `AGENTS.md` lists `tfp` (Bessel only) and `optax` (optional) as deps — twice (header line 4 + Core Dependencies section). Neither is imported anywhere in `normix/` nor in `pyproject.toml`. | Fix both lines to the real dep set (`jax`, `equinox`, `jaxopt`, `numpy`, `scipy`; `matplotlib` optional). | P1 |
| DOC2 | 5 | `design.md` Core Dependencies table still lists `tensorflow_probability.substrates.jax` (`log_bessel_kve` only); the JAX backend is pure-JAX. S8's "Pure-JAX (4 regimes)" row is already correct. | Fix the deps table row. | P1 |
| DOC3 | 5 | `.cursor/rules/project-overview.mdc`: Bessel said to live in `_bessel.py` (actual: `normix/utils/bessel.py`); storage table uses `L` where the convention mandates `L_Sigma`; lists the stale tfp dep. | Fix all three. | P1 |
| DOC4 | 5 | Constants tables drifted from `normix/utils/constants.py`: `coding-conventions.mdc` documents "backward-compatibility aliases" `GIG_EPS_V_HESS`/`GIG_EPS_NP` that don't exist; both it and the `ARCHITECTURE.md` table are missing `GIG_THETA_PERTURB`, `GIG_CLAMP_LO/HI`, `GIG_P_MAX`, `SIGMA_INIT_REG`, `BESSEL_SMALLZ_THRESHOLD` (verified), and the rule additionally lacks `B_POST_FLOOR`, `ALPHA_MIN_MARGIN`, `D_FLOOR`. (The review's `PARAM_CHANGE_EPS` mention is already resolved — removed from `constants.py` in #75.) | Sync both tables against `constants.py` (the canonical source). | P1 |
| DOC5 | 5 | `.cursor/rules/maintain-theory-docs.mdc` marks `transaction_costs.md` "(theory-only for now)" — tutorial `finance/06_transaction_costs` exists and is cross-linked. | Fix. | P2 |
| DOC6 | 5 | Public docstrings: `normix/utils/plotting.py:75` references `incerto-wiki/docs/design/VISUAL_STYLE.md` (different repo, meaningless to users); `mixtures/marginal.py` module docstring cross-references `docs/design/mixtures.md § 6` as a raw path (autodoc-visible; the cross-links rule requires Sphinx roles). | Fix both docstrings per `.cursor/rules/docs-cross-links.mdc`. | P2 |
| DOC7 | 5, 8.5 | `README.md`: distribution tables omit the `Univariate*` and `Factor*` families entirely; "For local development: `uv sync` … `pip install -e .`" — the pip step is redundant under uv; still no sentence that `from_natural`/`from_expectation` are intentionally unsupported on concrete joints (open since the 2026-04-04 review §3). | Fix tables + install text; add the one sentence and close that item for good. | P2 |

### Documentation website (W) — review §6

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| W1 | 6.1 | Broken API roles in theory pages: `theory/gig.md:72,286-288` references `normix.distributions.univariate.*` (module doesn't exist) and `GeneralizedInverseGaussian.moment_alpha` (method doesn't exist); `theory/gh.md:282-284` references `normix.distributions.mixtures.*` (wrong module path). Silent dead links under `nitpicky=False`. | Point the roles at the real paths/symbols. | P1 |
| W2 | 6.2 | Undefined citations: `[Hu2005]` cited on `theory/gig.md` and `theory/em_algorithm.md` but only defined on `theory/gh.md`; `[Shi2016]` used on `tutorials/em/04`, `tutorials/finance/05`, `user_guide/finance.md` without local definitions — masked by `suppress_warnings=['ref.citation']`. | Centralise citations in one included references page; drop the suppression. | P1 |
| W3 | 6.3 | `tutorials/finance/05_mean_risk_optimization.md:325` says ENB "remains on the roadmap"; `theory/enb.md` and `theory/generalized_enb.md` exist. | Reword to link the theory pages (an ENB *tutorial* is still future work — see finance plan Phase F). | P2 |
| W4 | 6.4 | Copy-paste hazards: `user_guide/finance.md:92-98` uses `jnp` without importing it; `user_guide/em_fitting.md:23-27` uses `NormalInverseGaussian` without an import; `CVaR(0.05)` commented as "95% level" (confidence-vs-tail wording). | Fix snippets and wording. | P2 |
| W5 | 6.5 | Navigation polish: Reference toctree uses `maxdepth: 1` (theory/design/API subpages invisible in the sidebar); landing "User guide" card deep-links to `exponential_family` instead of a hub; `docs/changelog.md` renders a duplicated "Changelog" heading (wrapper + included file). | Fix all three. | P2 |
| W6 | 6.6 | `docs/design/*.md` use raw relative markdown links (`../api/index`, `../theory/...`) instead of `{doc}` roles, bypassing Sphinx validation — the project's own cross-link rule. | Convert to `{doc}` roles. | P2 |
| W7 | 6.7, 8.6 | Both classes of docs bug found here (W1, W2) fail silently in CI. | Enable `nitpicky` (or a `{py:*}`-role check) in docs CI — **after** W1/W2/W6, likely with a small `nitpick_ignore` allowlist for external references. | P2 |

### Test suite (T) — review §7

| ID | § | Finding | Fix | Pri |
|----|---|---------|-----|-----|
| T1 | 7, 8.7 | No external reference for multivariate marginal densities — "the single highest-value guard this suite currently lacks". | Promote the review's verification harness into `tests/test_marginal_quadrature.py` marked `contract`: marginal `log_prob` vs trapezoid `∫ f(x,y) dy` on a log grid, all four families, d=2, including `x=μ` (review achieved ≤ 4e-11 on a 20k-point grid; a coarser grid with ≤ 1e-8 tolerance keeps it fast). ~120 LOC. | P1 |
| T2 | 7 | No `tests/conftest.py`: 32 files repeat the x64 toggle; SP500 loaders duplicated across three files with two CSV paths. | Add `conftest.py` with the session-level x64 setup and shared data fixtures; delete the duplicates. Net-negative LOC. | P2 |
| T3 | 7 | Marker drift: `smoke` defined but never used; a `gpu` marker on a CPU-backend test (`test_incremental_em.py:485`); a 4000-sample × 120-iteration factor-EM test and the d=50 high-dimensional tests unmarked (fast suite pays). | Fix markers (`--strict-markers` is already on); drop `smoke` from `pyproject.toml` or start using it; mark the heavy tests `slow`/`stress`. | P2 |
| T4 | 7 | Untested utilities: `gammaincinv` (vs `scipy.special.gammaincinv`), `rvs_pinv`/`build_pinv_table` directly, `quantile_cmc`, and the `renyi`-gradient edge case (§1.2). | Add unit tests (the `renyi` one lands with B2; the `quantile_cmc` bracket test with B7). ~80 LOC. | P2 |
| T5 | 7 | EM workflow tests lean on finiteness (`test_em_regression.py:173`, `test_sp500_distribution_validation.py:67`) with loose recovery tolerances (`test_factor_mixture.py:277` rtol=0.5); `test_mcecm.py`'s docstring claims MCECM ≡ EM at the MLE but no test compares them. | Tighten where cheap; add an MCECM-vs-EM at-the-MLE comparison test. ~60 LOC. | P3 |

### Features (F) — review §8, deduplicated

§8.1 → B1, §8.2 → E2, §8.3 → E3, §8.5 → DOC7, §8.6 → W7, §8.7 → T1.
Genuinely new:

| ID | § | Item | Pri |
|----|---|------|-----|
| F1 | 8.4 | `skewness()` / `kurtosis()` on marginal mixtures — closed forms exist for the GH family; standard finance need. Rounds out the scipy-like moment surface (`mean`/`cov`/`var`/`std` exist). Needs formulas in `docs/theory/`, implementation on `NormalMixture` + `Univariate*`, tests vs sample moments / quadrature. ~120 LOC. | P3 |

## Counts

| Priority | B | D | C | E | DOC | W | T | F | **Total** |
|----------|---|---|---|---|-----|---|---|---|-----------|
| P1 | 2 | — | 4 | 1 | 4 | 2 | 1 | — | **14** |
| P2 | 4 | 3 | — | 3 | 3 | 5 | 3 | — | **21** |
| P3 | 2 | 1 | 1 | 3 | — | — | 1 | 1 | **9** |
| **Total** | **8** | **4** | **5** | **7** | **7** | **7** | **5** | **1** | **44** |

## Dependency Graph

```
DEC-1 ──► B1                      W1, W2, W6 ──► W7
DEC-2 ──► D1, D2                  B2 ──► T4 (renyi regression lands with the fix)
DEC-3 ──► D3                      B7 ──► T4 (quantile_cmc bracket test)
DEC-4 ──► D4                      Phase 4 (tests) ──► safer Phase 5/6 refactors (soft)
DEC-5 ──► E3
```

Everything else is independent; Phases 1, 2, and 7 touch no `normix/` logic.

## Phases

### Phase 0 — Design decisions

Five decisions, each landing as a `../design/design.md` row before its code:

| ID | Decision | Gates | Proposal on the table |
|----|----------|-------|----------------------|
| DEC-1 | Cross-family divergence dispatch: lift-to-GH vs raise; Tier 2 or Tier 3 ownership; boundary handling | B1 | Review recommends lifting via `to_generalized_hyperbolic()` (turns a footgun into a feature); raise when no lift exists. Wrinkle: VG/NInvG (and Gamma/InverseGamma at EF level) sit on the GIG boundary where GH's ψ degenerates — the lifts take `boundary_eps`, so decide exact-vs-ε semantics |
| DEC-2 | One spelling for computed accessors: `subordinator` and `sigma` across `MarginalMixture` / `FactorNormalMixture` / `Univariate*` / `MultivariateNormal` | D1, D2 | Method everywhere (conventions avoid `@property` for computed matrices); resolve the factor-field name collision |
| DEC-3 | `fit()` defaults mechanism | D3 | `_fit_defaults()` classmethod consumed by the base `fit(**kwargs)` |
| DEC-4 | `finance/projection.py` placement | D4 | Fold into `finance/__init__` or drop in favour of `model.project(w)` |
| DEC-5 | Quantile-table reuse shape | E3 | Frozen table object vs documented `_pinv_grids()` reuse |

DEC-1 is the only decision blocking a P1 item — take it first. DEC-2 and
DEC-5 shape public API / a new abstraction; run them through the architect
skill if contested. **Exit:** design.md rows recorded.

### Phase 1 — Hygiene sweep (C1–C5)

One PR, pure deletions plus one comment.
**Exit:** `ruff check --select F401,F811,F841 normix/` clean;
`rg "_gig_rvs_pinv|def sample" normix/` shows neither; `data/gig_test.csv`
gone; fast suite green.

### Phase 2 — Agent docs & rules sync (DOC1–DOC7)

Two docs commits: internal rules/dev-notes (DOC1–DOC5), public docstrings +
README (DOC6, DOC7).
**Exit:** `rg "tfp|tensorflow|optax" AGENTS.md .cursor/rules/ dev-notes/design/design.md`
returns only historical/archive mentions; constants tables match
`constants.py` name-for-name; `bash scripts/check_doc_links.sh` OK.

### Phase 3 — Correctness (B1–B8)

TDD per item (failing test first — all eight have crisp reproductions).
Interrogate review on the B3 (GIG Hessian) and B5/B6 (EM fitter) PRs.
**Exit:** cross-family H² symmetric and equal to the GH-lifted truth;
`jax.grad(renyi)` at exactly 1.0 equals −varentropy/2; loop and scan paths
agree on step-1 convergence and return Python `bool`/`int`;
`jax.jit(gig.cdf)` either works or the exception is documented in the
docstring and `ARCHITECTURE.md`.

### Phase 4 — Test-suite hardening (T1–T5)

T1 first — it is independent and the highest-value guard; land it before the
Phase 5/6 refactors it protects.
**Exit:** `uv run pytest tests/ -m contract` includes the quadrature suite,
green at ≤ 1e-8; `rg -l "jax_enable_x64" tests/` returns only `conftest.py`;
`--strict-markers` passes with no unused markers; fast-suite runtime not
regressed (heavy tests now deselected by the default marker expression).

### Phase 5 — Efficiency (E1, E2, E4–E7; E3 after DEC-5)

Benchmark before/after (`benchmarks/run_all.py` + `compare.py`); E2 may need
a small CVaR benchmark if none covers it. Interrogate review on the E4 PR
(`bessel.py`, `custom_jvp`).
**Exit:** aggregation einsum contracts to `'ni,nj,n->ij'` (no `(n, d, d)`
intermediate); fused CVaR path runs one quantile solve where three ran, with
value/gradient/Hessian equal to the unfused path; `slow or stress` suite
green; no benchmark regression.

### Phase 6 — API consistency (D1–D4, after Phase 0 + Phase 4)

Public-API refactors, guarded by the hardened suite. Pre-1.0: change the
code, no compatibility shims.
**Exit:** one subordinator/sigma spelling everywhere (grep the three old
spellings in `normix/`, `docs/`, tutorials); `fit()` has one signature with
`track_ll`/`eta_update`/`m_step_kwargs` reachable on VG/NInvG/GH; no
forward-only projection module; `finance_architecture.md` updated.

### Phase 7 — Website (W1–W6, then W7)

Fix content first, then turn on enforcement.
**Exit:** `uv run make -C docs html` green with `nitpicky` (or the role
check) enabled and `suppress_warnings=['ref.citation']` removed; linkcheck
green; sidebar shows Reference subpages.

### Phase 8 — Features (F1)

`skewness()`/`kurtosis()` with theory-page derivations and moment tests.
**Exit:** values match sample moments / quadrature across the four families;
gallery/API docs updated.

## Phase Summary

| Phase | Theme | Items | Blocked by |
|-------|-------|-------|------------|
| 0 | Design decisions | DEC-1…5 | — |
| 1 | Hygiene sweep | C1–C5 | — |
| 2 | Agent docs & rules sync | DOC1–DOC7 | — |
| 3 | Correctness | B1–B8 | DEC-1 (B1 only) |
| 4 | Test hardening | T1–T5 | — (B2/B7 tests land with fixes) |
| 5 | Efficiency | E1, E2, E4–E7; E3 | DEC-5 (E3 only) |
| 6 | API consistency | D1–D4 | DEC-2/3/4; prefer after Phase 4 |
| 7 | Website | W1–W7 | W7 after W1/W2/W6 |
| 8 | Features | F1 | — |

Suggested order: Phases 1 + 2 immediately (independent, zero-risk); DEC-1 →
B1 and B2 next (the Medium-severity item and the cheapest wrong number);
T1 + E1 early (highest-value guard, biggest cheap win); then the remainder of
Phases 3–5, Phase 6 once decisions and tests are in, Phase 7 in one docs
pass, Phase 8 last.

## Execution Notes

- One logical change per commit, conventional-commit scoped
  (`.cursor/skills/git-conventions/`); Phase 1 and Phase 2 are each one PR.
- Interrogate skill is mandatory for PRs touching `normix/utils/bessel.py`
  (E4), the GIG solver surface (B3), or the EM fitter (B5, B6, E1).
- TDD skill for every B item; the review's measured numbers above are the
  failing-test targets.
- Per agent-maintenance: DEC rows go to `../design/design.md`; constants-table
  edits (DOC4) treat `constants.py` as canonical; plan-phase completions
  update this file.

## Non-Items

- Review §8.8 (env provisioning): `uv sync` worked out of the box — nothing
  to do.
- Review §3 non-issues: `bregman_objective` is exercised by tests;
  `_force_nonfinite_at_step` is a deliberate test hook (C5 adds the pointer
  comment only).
