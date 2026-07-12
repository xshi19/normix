# normix Package Review (2026-07-12)

## Scope

Full-package review requested along seven axes:

1. code errors, numerical issues, wrong math formulas
2. design vs the principles in `AGENTS.md` / `dev-notes/design/design.md`
3. dead code, unnecessary helpers, redundant logic
4. numerical efficiency
5. out-of-date inline comments and docstrings
6. documentation website structure, friendliness, errors
7. other improvements and feature ideas

## Verification Performed

- Fast test suite: `uv run pytest tests/` → **1018 passed, 68 deselected**
  (11:14 min; single warning is the known jaxopt deprecation inside a
  solver test that constructs jaxopt before the package-level filter).
- Independent numerical verification (scripts, not committed):
  - All four marginal log-densities (VG, NIG, NInvG, GH, d=2, including
    `x=μ`) vs trapezoid integration of the joint over `y` on a 20k-point
    log grid: agreement to **≤ 4e-11** (mostly ≤ 4e-15).
  - Joint `log_prob` (EF form `θᵀt − ψ`) vs direct
    `log_prob_joint(x, y)` for all four families: ≤ 2e-15.
  - `GIG._grad_log_partition` (Bessel ratios) vs quadrature moments:
    ≤ 1e-10 relative.
  - `log_kv` vs mpmath at regime boundaries (small-z with |v|≤0.5,
    Hankel/Olver thresholds, v=100 z=1): worst abs error 7e-9 at
    (v=0.51, z=1e-8), typically ≤ 1e-12.
  - Entropy vs scipy for Gamma / InverseGamma / InverseGaussian: exact
    to print precision.
  - Hand re-derivation: GIG θ-space Hessian (all six entries), the
    closed-form M-step (μ, γ, Σ with the −E[Y]γγᵀ substitution), the VG
    and NIG/GH marginal-density normalisations, the GIG η-rescaling
    direction, CVaR value / gradient / Hessian (verified against
    translation-invariance and positive-homogeneity Euler identities),
    and the transaction-cost QP blocks vs `docs/theory/transaction_costs.md`.
    All correct.
- `bash scripts/check_doc_links.sh` → OK.
- `ruff check --select F401,F811,F841` → 15 findings (below).
- Docs and test-suite deep dives delegated to read-only subagent surveys.

## Overall Assessment

The package is in very good shape. The mathematical core is *verified
correct* — every closed-form density, gradient, Hessian, and M-step
formula I re-derived or numerically cross-checked agrees with the
implementation, including the tricky near-boundary regimes (`x=μ`,
degenerate GIG limits, Bessel regime boundaries). Both issues from the
2026-04-04 follow-up review are fixed (`IncrementalEMFitter` now returns
`converged=None`; import-time x64 forcing is now a warning). The design
matches the stated philosophy closely: the triad, solver separation,
immutability, unbatched core, and "special cases use their own formulas"
are all consistently realised.

The findings below are mostly edge-case correctness traps, hygiene, and
efficiency opportunities — nothing at the level of a wrong density or a
broken estimator.

---

## 1. Correctness / numerical findings

### 1.1 Cross-family divergences silently return wrong numbers

**Severity: Medium** (silent wrong result on a plausible call)

`squared_hellinger(p, q)` / `kl_divergence(p, q)` evaluate
`type(p)._log_partition_from_theta` at *q's* natural parameters. The
joint-θ layouts are compatible across families, but each special-case ψ
reinterprets θ under its own family constraint:

- `JointNormalInverseGaussian._log_partition_from_theta` fixes `p=−1/2`
  and ignores `θ₁`;
- `JointVarianceGamma` / `JointNormalInverseGamma` recover only their own
  (α, β) from θ, implicitly assuming `b=0` / `a=0`.

Measured example (same μ, γ, Σ):

```text
H²(NIG, GH-lifted-same-distribution) = 0.000000   (correct, exact-embedding case)
H²(GH_p=0.7, NIG)                    = 0.017019   (correct: GH ψ reads any θ)
H²(NIG, GH_p=0.7)                    = 0.074043   (WRONG; truth 0.017019)
```

The Tier-3 docstring does say "must be the same type (or share the same
log-partition)", but nothing enforces it and the wrong number looks
plausible. Since `to_generalized_hyperbolic()` exists for all three
special cases, the fix is cheap and turns a footgun into a feature:
in Tier 3 (or Tier 2), when `type(p) is not type(q)`, lift both marginals
(or joints) to GH before dispatching; or at minimum raise on mismatched
families.

### 1.2 `renyi` gradient is wrong at exactly α = 1

**Severity: Low** (value correct; derivative at one point wrong)

`ExponentialFamily.renyi` guards the removable singularity with
`jnp.where(one_minus == 0.0, entropy, R(α)/safe)`. The *value* at α=1 is
right, but `jax.grad` through this `jnp.where` returns **0.0** at exactly
α=1, whereas the true derivative is −V_H/2:

```text
Gamma(2,1): varentropy = 0.6449 → d(renyi)/dα|₁ should be −0.3225
jax.grad at 1.0       :  0.0          (wrong)
jax.grad at 1.0+1e-7  : −0.32323      (correct)
```

Fix options: switch the guard to a Taylor branch
(`H − V_H(α−1)/2` inside an |1−α| < ε window, keeping both branches
NaN-free), or a `jax.custom_jvp`. Also worth a regression test.

### 1.3 `GIG.fisher_information(backend='jax')` mixed entries are ~2–4.5% off

**Severity: Low** (documented for the solver, but leaks into a public API)

`GIG._hessian_log_partition` approximates the mixed derivative
`∂²log K_ν/∂ν∂z` by an integer-shift FD (reusing p±1, p±2). At
(p=0.7, a=1.4, b=0.9) the H₁₂/H₁₃ entries are off by 4.5%/2.2% vs
`jax.hessian` (and vs the *more accurate* CPU central-FD Tier-3
override). For the Newton solver this only affects the step, not the
solution — that is documented in `_hessian_log_partition` — but
`fisher_information()` is public and returns the coarse matrix by
default with no caveat in its own docstring. Suggested: either FD the
mixed term with ±`BESSEL_EPS_V` (4 extra Bessel evals: orders
p±ε±1), or document the accuracy asymmetry on `fisher_information`
(cpu backend or `jax.hessian` for high-accuracy needs).

### 1.4 `GIG.cdf` / `GIG.ppf` are not JIT-safe

**Severity: Low** (loud failure, but contradicts stated contracts)

Both call `float(self.p)` etc. to pick the degenerate-limit branch, so
`jax.jit(g.cdf)` raises `TypeError: unhashable type: ArrayImpl`
(confirmed). The architecture docs advertise JIT-compatibility broadly;
either document the exception or use `lax.cond` on the degeneracy test.
(The `Univariate*` mixin `cdf`/`ppf` are pure JAX and fine.)

### 1.5 Smaller nits

- **Loop/scan convergence asymmetry** — `BatchEMFitter._fit_loop` refuses
  to converge at iteration 1 (`max_change < tol and i > 0`;
  `converged` additionally requires `n_iter > 1`), while the
  `lax.scan` path can converge at step 1. Same fitter, two stopping
  semantics.
- **`EMResult` type drift on the scan path** — `converged` is returned as
  a JAX bool array while `diverged` is cast to `bool` and `n_iter` is an
  int32 array against an `int` annotation. Cosmetic, but
  `result.converged is True` fails on the scan path.
- **`quantile_cmc` bracket** — ±5·std around the PINV ppf; if the CMC
  root falls outside (tiny `len(Y)`, extreme α), bisection silently
  returns an endpoint. Consider a width tied to α or an assertion.
- **`_UnivariateNormalMixtureMixin.log_prob`** on a batched `(n,)` input
  raises a shape error from `solve_triangular` (good: loud, not silent),
  but the message is cryptic; a `d=1` check with a "use jax.vmap" hint
  would be friendlier.

## 2. Design vs AGENTS.md / design.md

Verdict: strong match. Specifically checked and confirmed:

- **Triad + solver separation (F3–F5, S2)** — grad/hess passed to
  `solve_bregman` are always θ-space; the φ↔θ chain rule lives only in
  `_jax_newton_raw`. No distribution knows about reparametrisation.
- **Special cases use own formulas (Simplicity §)** — VG / NInvG / NIG
  marginal `log_prob` all use their specialised Bessel forms (verified
  numerically), never routing through GH; NIG uses the closed
  `K_{−1/2}` form.
- **Immutability, `replace(**)` facade, no isinstance dispatch** in the
  core (`from_expectation`'s pytree-vs-array `isinstance` is a documented
  input-form dispatch, not distribution-type dispatch — acceptable).
- **E12 posterior-GIG consolidation** — `_posterior_gig_params` /
  `_floored_posterior_gig_params` appear exactly twice (joint + factor
  bases) with `to_gig()` supplying coordinates; no subclass overrides.

Frictions worth a decision:

- **Three spellings for "the subordinator"**:
  `model.joint.subordinator()` (method, NormalMixture),
  `factor_model.subordinator` (field), `univariate.subordinator`
  (property). A uniform accessor on `MarginalMixture` (e.g. a
  `subordinator()` method everywhere, or property everywhere) would
  remove a recurring doc/tutorial wart.
- **`MultivariateNormal.sigma` is a property while `NormalMixture.sigma()`
  is a method** — same name, different call syntax, adjacent classes.
- **`MultivariateNormal.sample(key, shape)`** is marked "Legacy API —
  prefer rvs" and is referenced nowhere; per the "no redundant methods"
  rule it should be deleted.
- **`fit()` overrides** (VG/NInvG/GH) exist only to flip default backends
  and re-list every parameter; they also *narrow* the API (no
  `track_ll`, `eta_update`, `m_step_kwargs` without dropping to
  `BatchEMFitter`). A `_fit_defaults()` classmethod consumed by the base
  `fit(**kwargs)` would keep one signature and preserve pass-through.
- **`finance/projection.py`** is a 30-line module wrapping
  `model.project(w)` in a free function. It matches the plan doc, but
  under "no wrapper that only forwards" it could be folded into
  `finance/__init__` or dropped in favour of the method.

## 3. Dead code / redundant logic

Ruff (`F401/F811/F841`) plus manual findings — all trivially removable:

| Location | Item |
|---|---|
| `mixtures/joint.py:75-83` | unused `numpy`, `equinox`, `LOG_EPS` imports |
| `mixtures/joint.py:479` | unused re-import of `NormalMixtureEta` carrying a stale `# noqa: F811` |
| `mixtures/marginal.py:23` | unused `numpy` import |
| `mixtures/marginal.py:189` | unused `JointNormalMixture` import inside `NormalMixture.__init__` |
| `mixtures/factor.py:433` | unused local `d = s4.shape[0]` |
| `distributions/generalized_hyperbolic.py:28,30` | unused `Optional`, `eqx` |
| `distributions/generalized_inverse_gaussian.py:70` | `bregman_objective` imported but unused |
| `distributions/generalized_inverse_gaussian.py:775` | `import numpy as np` inside `_initial_guesses` shadows module import |
| `fitting/eta.py:23` | unused `Optional` |
| `fitting/solvers.py:47` | unused `List` |
| `finance/projection.py:19` | unused `jnp` |
| `utils/validation.py:8-9` | unused `jax`, `jnp` |
| `distributions/generalized_inverse_gaussian.py:174` | `_gig_rvs_pinv` is a pure alias of `rvs_pinv` — violates the "no unnecessary aliases" convention |
| `distributions/normal.py:258` | `MultivariateNormal.sample` legacy method, unused |
| `data/gig_test.csv` | tracked but referenced by no test or script |

Non-issues checked: `bregman_objective` itself is exercised by tests;
`BatchEMFitter._force_nonfinite_at_step` is a deliberate test hook used
by `test_em_regression.py` (fine, maybe worth a comment pointing at the
test).

## 4. Numerical efficiency

Ordered by expected impact:

1. **E-step aggregation materialises an `(n, d, d)` tensor.**
   `NormalMixture._aggregate_eta` and `FactorNormalMixture._aggregate_stats`
   compute `jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y)` then `mean(axis=0)`.
   Contract directly — `jnp.einsum('ni,nj,n->ij', X, X, E_inv_Y) / n` —
   to avoid O(n·d²) peak memory (n=100k, d=100 ⇒ ~8 GB transient today).
2. **CVaR triple-computes the VaR bisection.**
   `build_quadratic_approximation` calls `value_w`, `gradient_w`,
   `hessian_w`, each of which independently re-projects, rebuilds the
   PINV table (`ppf`), and re-runs the 60-step CMC bisection for the same
   `(w, Y)`. A fused `value_grad_hess_w` sharing one quantile solve would
   cut the dominant cost ~3×.
3. **PINV tables are rebuilt on every `cdf`/`ppf` call**
   (`_UnivariateNormalMixtureMixin._pinv_grids`, `GIG.cdf/ppf`,
   `InverseGaussian.ppf`): 4000 `log_prob` evaluations (Bessel-heavy)
   each time. Modules are immutable so self-caching is out, but a small
   frozen "quantile table" object (or documented
   `u_grid, x_grid = dist._pinv_grids()` reuse) would help repeated
   quantile workloads.
4. **`log_kv` custom JVP always pays the ∂ν finite difference** (2 extra
   Bessel evaluations) even when only the z-tangent is nonzero.
   `jax.custom_jvp(..., symbolic_zeros=True)` (JAX ≥ 0.4.14) would skip
   it for z-only differentiation.
5. **Fixed-iteration Newton loops**: `_newton_digamma` and `gammaincinv`
   always run 50 `fori_loop` iterations (converged in <10). A
   `while_loop` with a tolerance, or just 20 iterations, trims the
   VG/NInvG jax-backend M-step and Gamma/InvGamma `ppf`.
6. **`GIG.var()`** builds the full 7-Bessel Hessian for entry [2,2]; the
   closed form needs 3 Bessel evaluations. Similarly
   `NormalInverseGaussian._subordinator_expectations` routes E[Y] and
   E[1/Y] through the 5-Bessel GIG path although both are closed-form
   for InverseGaussian (only E[log Y] needs Bessel).
7. **`MeanRiskProblem._reduced()`** re-runs the Cholesky solves on every
   `weights`/`dispersion`/`risk_at` call; the pieces (`A_inv`,
   `Sinv_M`) are w-independent and could be precomputed fields.

## 5. Out-of-date comments, docstrings, rules

- **`AGENTS.md` header**: "Deps: `jax`, `equinox`, `jaxopt`, `tfp`
  (Bessel only), `optax` (optional)" — **tfp and optax are gone**
  (no import anywhere in `normix/`; not in `pyproject.toml`). The first
  of the two "## Design Philosophy" sections is also a mislabeled
  duplicate heading that actually lists dependencies.
- **`dev-notes/design/design.md` Core Dependencies table** still lists
  `tensorflow_probability.substrates.jax` (`log_bessel_kve` only) — the
  JAX backend is now pure-JAX and tfp-free. S8's "Pure-JAX (4 regimes)"
  row is correct; the deps table isn't.
- **`.cursor/rules/project-overview.mdc`**: says Bessel lives in
  `_bessel.py` (actual: `normix/utils/bessel.py`); storage table uses
  `L` where the code (and coding-conventions rule) mandate `L_Sigma`;
  lists the stale tfp dependency.
- **`.cursor/rules/coding-conventions.mdc` constants table** drifted from
  `normix/utils/constants.py`: lists `PARAM_CHANGE_EPS` and the
  "backward-compatibility aliases" `GIG_EPS_V_HESS`/`GIG_EPS_NP`, none of
  which exist; missing `B_POST_FLOOR`, `ALPHA_MIN_MARGIN`, `D_FLOOR`,
  `SIGMA_INIT_REG`, `GIG_THETA_PERTURB`, `GIG_CLAMP_LO/HI`, `GIG_P_MAX`,
  `BESSEL_SMALLZ_THRESHOLD`.
- **`dev-notes/ARCHITECTURE.md` constants table** (rule: "must match
  constants.py") is missing `GIG_THETA_PERTURB`, `GIG_CLAMP_LO/HI`,
  `GIG_P_MAX`, `SIGMA_INIT_REG`, `BESSEL_SMALLZ_THRESHOLD`.
- **`.cursor/rules/maintain-theory-docs.mdc`** marks
  `transaction_costs.md` "(theory-only for now)" — tutorial
  `finance/06_transaction_costs` exists and is cross-linked.
- **`normix/utils/plotting.py:75`** references
  `incerto-wiki/docs/design/VISUAL_STYLE.md` — a path in a different
  repository, meaningless to normix users (public docstring).
- **`mixtures/marginal.py` module docstring** cross-references
  `docs/design/mixtures.md § 6` as a raw path (autodoc-visible; the
  docs-cross-links rule requires Sphinx roles in docstrings).
- **`README.md`**: distribution tables omit the `Univariate*` and
  `Factor*` families entirely (significant public API); "For local
  development: `uv sync` … `pip install -e .`" — the pip step is
  redundant under uv.

## 6. Documentation website

Structure is good: Getting started → User guide → Distribution gallery →
Tutorials (core/distributions/EM/stats/finance, sensibly ordered) →
Reference (theory/design/API/changelog). No orphan pages; the gallery is
a genuinely nice discovery surface; theory↔tutorial cross-links are
consistent. Issues found (file:line from the docs survey):

1. **Broken API roles in theory pages** —
   `docs/theory/gig.md:72,286-288` references
   `normix.distributions.univariate.*` (module doesn't exist) and
   `GeneralizedInverseGaussian.moment_alpha` (method doesn't exist);
   `docs/theory/gh.md:282-284` references
   `normix.distributions.mixtures.*` (wrong module path). With
   `nitpicky=False` these fail silently as dead links.
2. **Undefined citations** — `[Hu2005]` cited on `theory/gig.md:253` and
   `theory/em_algorithm.md:6` but only defined on `theory/gh.md`;
   `[Shi2016]` used on `tutorials/em/04`, `tutorials/finance/05`,
   `user_guide/finance.md` without local definitions (masked by
   `suppress_warnings=['ref.citation']`). Centralising citations in one
   included references page would fix the class of problem.
3. **Stale roadmap line** — `tutorials/finance/05_mean_risk_optimization.md:325`
   says ENB "remains on the roadmap"; `theory/enb.md` and
   `theory/generalized_enb.md` exist.
4. **Copy-paste hazards** — `user_guide/finance.md:92-98` uses `jnp`
   without importing it; `user_guide/em_fitting.md:23-27` uses
   `NormalInverseGaussian` without an import; `CVaR(0.05)` commented as
   "95% level" (confidence-vs-tail wording).
5. **Navigation polish** — the Reference toctree uses `maxdepth: 1`, so
   theory/design/API subpages are invisible in the sidebar until
   clicked; the landing-page "User guide" card deep-links to
   `exponential_family` rather than a hub page; `docs/changelog.md`
   renders a duplicated "Changelog" heading (wrapper + included file).
6. **Convention drift** — `docs/design/*.md` use raw relative markdown
   links (`../api/index`) instead of `{doc}` roles, bypassing Sphinx
   validation (the project's own cross-link rule).
7. Consider enabling `nitpicky` (or a `{py:*}`-role check) in docs CI so
   class-path typos like (1) fail the build.

## 7. Test suite

1018 fast tests pass; univariate distributions, Bessel, solvers,
conversions, posterior-GIG, varentropy, and finance KKT/QP blocks are
validated against scipy or closed forms with tight tolerances. Gaps
worth closing:

- **No external reference for multivariate marginal densities.** The
  quad-integration check performed for this review (marginal `log_prob`
  vs `∫ f(x,y) dy` on a log-grid; agreement ≤ 4e-11) is cheap and would
  make a strong `contract` test for all four families.
- **EM workflow tests lean on finiteness** (e.g.
  `test_em_regression.py:173`, `test_sp500_distribution_validation.py:67`)
  and very loose recovery tolerances (`test_factor_mixture.py:277`
  rtol=0.5). `test_mcecm.py`'s docstring claims MCECM ≡ EM at the MLE but
  no test compares them.
- **No `tests/conftest.py`** — 32 files repeat the x64 toggle; SP500
  loaders are duplicated across three files with two CSV paths.
- **Marker drift** — `smoke` defined but never used; a `gpu` marker on a
  CPU-backend test; a 4000-sample × 120-iteration factor-EM test and the
  d=50 high-dimensional tests are unmarked (fast suite pays for them).
- **Untested utilities** — `gammaincinv` (vs `scipy.special.gammaincinv`),
  `rvs_pinv`/`build_pinv_table` directly, `quantile_cmc`, and the
  `renyi`-gradient edge case from §1.2.

## 8. Suggestions and feature ideas

1. **Cross-family divergences via GH lifting** (fixes §1.1 and adds a
   feature): in Tier 2/3, when families differ, lift both operands with
   `to_generalized_hyperbolic()` and compare in GH coordinates; raise if
   no lift exists.
2. **Fused CVaR `value_grad_hess`** sharing the quantile solve (§4.2) —
   directly speeds `TransactionCostProblem.solve` and any SQP-style use.
3. **Frozen quantile-table object** for `Univariate*` (and GIG) so
   `cdf`/`ppf`/`rvs` workloads amortise the PINV build.
4. **Moment API completion**: `skewness()` / `kurtosis()` on marginal
   mixtures (closed forms exist for the GH family; standard finance
   need), and `var()`/`std()` already exist via `cov()` — d>1
   `mean/cov/skew/kurt` would round out a scipy-like surface.
5. **`from_expectation` on concrete joints** is intentionally
   unsupported (2026-04-04 review §3) — the README still doesn't say so;
   one sentence would close that item for good.
6. **Docs CI hardening**: enable nitpicky (or a linkcheck for `{py:*}`
   roles) plus a citations page; both classes of docs bug found here
   would then be build failures.
7. **Consider promoting the review-verification harness** (marginal vs
   quad-integration; gradient vs quadrature moments) into
   `tests/test_marginal_quadrature.py` marked `contract` — it is the
   single highest-value guard this suite currently lacks.
8. Env note: this VM has no configured normix environment; `uv sync`
   provisioning worked out of the box, no extra setup was needed.

## Bottom Line

The mathematical core is verified sound — I could not find a single
wrong density, moment, or update formula, and the numerics hold up at
the boundaries the code explicitly engineers for. The remaining work is
edge-hardening (cross-family divergences, `renyi` gradient, GIG
Fisher-information accuracy note), hygiene (15 unused imports, a few
stale rule tables, README omissions), and a handful of concrete
efficiency wins (E-step einsum contraction, fused CVaR derivatives,
PINV-table reuse). Docs are structurally strong; the fixes are localised
(broken theory-page roles, missing citations, small nav polish).
