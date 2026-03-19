# normix JAX Migration Plan

## Status (2026-03-10)

Phases 0–5 complete. The JAX package (`normix/`) replaces the NumPy
implementation. The NumPy reference is preserved in `normix_numpy/`.

Tests: `python3 -m pytest tests/test_jax_*.py` (51 tests, all passing).



Reference: `docs/design/jax_design.md` for architecture decisions.

## Phase 0: Foundation

**Goal:** JAX `_bessel.py` + `ExponentialFamily` base class.

### 0.1 `_bessel.py`
- Port `log_kv` with `@jax.custom_jvp`:
  - Eval: TFP `log_bessel_kve(|v|, z) - z` + asymptotic fallbacks
  - ∂/∂z: exact recurrence (Bessel ratio)
  - ∂/∂ν: DLMF 10.40.2 for large z; central FD (ε=1e-5) otherwise
- Verify: `jax.grad` and `jax.hessian` on `psi_gig(theta)` match normix reference
- Verify: `jax.vmap` over `(v, z)` arrays

### 0.2 `ExponentialFamily(eqx.Module)`
- Abstract: `_log_partition_from_theta`, `natural_params`, `sufficient_statistics`, `log_base_measure`
- Derived: `log_prob`, `expectation_params` (via `jax.grad`), `fisher_information` (via `jax.hessian`), `log_partition`
- Constructors: `from_natural(theta)`, `from_expectation(eta)` (via `jaxopt.LBFGSB`), `fit_mle(X)`
- All methods unbatched; public API vmaps automatically

### 0.3 Validation
- Cross-validate `ExponentialFamily.log_prob` vs direct formula
- Cross-validate `expectation_params()` vs NumPy normix `_natural_to_expectation`
- Cross-validate `fisher_information()` vs NumPy normix numerical Hessian

## Phase 1: Simple Distributions

**Goal:** Gamma, InverseGamma, InverseGaussian.

### 1.1 Gamma
- Stored: `alpha`, `beta` (shape, rate)
- `_log_partition_from_theta`: `gammaln(θ₁+1) - (θ₁+1)log(-θ₂)`
- Analytical `expectation_params` override via `custom_jvp` using `digamma`
- `from_expectation`: closed-form Newton (digamma inversion)

### 1.2 InverseGamma
- Same structure; `_log_partition_from_theta` uses `gammaln`
- Analytical gradient via `custom_jvp`

### 1.3 InverseGaussian
- Stored: `mu`, `lam`
- `_log_partition_from_theta` involves `sqrt`; `jax.grad` works directly

### 1.4 Validation
- For each: `from_classical` → `natural_params` → `from_natural` roundtrip
- `fit_mle(samples)` vs scipy MLE
- `expectation_params()` vs sample mean of sufficient statistics

## Phase 2: GIG

**Goal:** Full GIG with η-rescaling, Bessel gradients, and multi-start L-BFGS-B.

### 2.1 GIG Distribution
- Stored: `p`, `a`, `b`
- `_log_partition_from_theta`: `log(2) + log_kv(p, √(ab)) + (p/2)log(b/a)`
- `jax.grad` flows through `custom_jvp` on `log_kv` — no separate analytical override needed
- Degenerate limits (√(ab) < ε): delegate to Gamma/InverseGamma `_log_partition_from_theta`

### 2.2 η-Rescaling in `from_expectation`
- Compute s = √(η₂/η₃)
- Transform η → η̃ = (η₁ + ½ log(η₂/η₃), √(η₂η₃), √(η₂η₃))
- Solve η̃ → θ̃ via `jaxopt.LBFGSB` with bounds θ̃₂ ≤ 0, θ̃₃ ≤ 0
- Recover θ = (θ̃₁, θ̃₂/s, s·θ̃₃)
- Multi-start: initial guesses from Gamma, InverseGamma, InverseGaussian special cases

### 2.3 Validation
- η→θ→η roundtrip for all test cases from `gig_optimization_test.ipynb`
- Stress cases: large a, small b, large √(ab), near-degenerate limits
- `fit_mle` vs scipy `geninvgauss` on generated samples
- Fisher information vs numerical Hessian

## Phase 3: MultivariateNormal + JointNormalMixture

**Goal:** Multivariate Normal (Cholesky), JointNormalMixture base, conditional expectations.

### 3.1 MultivariateNormal
- Stored: `mu`, `L` (Cholesky of Σ)
- `log_prob` via `solve_triangular` (no explicit Σ⁻¹)
- `from_classical(mu, sigma)`: compute Cholesky at construction

### 3.2 JointNormalMixture(ExponentialFamily)
- Stored: `mu`, `gamma`, `L` + subordinator attributes (subclass-specific)
- `log_prob_joint(x, y) = log f(x|y) + log f_Y(y)` using Cholesky solve
- Abstract: `subordinator()` → returns fitted `ExponentialFamily`
- Abstract: `conditional_expectations(x)` → dict of E[g(Y)|X]
- All linear algebra through `L` (never form Σ explicitly)

### 3.3 NormalMixture
- Stores `_joint: JointNormalMixture`
- `log_prob(x)`: closed-form marginal log-density (Bessel for GH)
- `e_step(X) = jax.vmap(self._joint.conditional_expectations)(X)`
- `m_step(X, expectations) → NormalMixture` (returns new immutable model)

## Phase 4: Mixture Distributions

**Goal:** VG, NInvG, NIG, GH — each as a (Joint, Marginal) pair.

### 4.1 VarianceGamma (Gamma subordinator)
### 4.2 NormalInverseGamma (InverseGamma subordinator)
### 4.3 NormalInverseGaussian (InverseGaussian subordinator)
### 4.4 GeneralizedHyperbolic (GIG subordinator)

Each follows the same pattern:
- Joint: implement `subordinator()`, `conditional_expectations(x)`, subordinator-specific theta construction
- Marginal: implement `log_prob(x)` (closed-form Bessel expression for GH/NIG; numerical integration for VG/NInvG if needed)
- Validate marginal `log_prob` vs numerical integration of joint

## Phase 5: EM Fitting

**Goal:** Separate fitters with `jax.lax.while_loop` / `jax.lax.scan`.

### 5.1 BatchEMFitter
- `jax.lax.while_loop(cond, body, init_state)`
- Convergence: relative log-likelihood change < tol
- Regularization: `det(Σ) = 1` (normalize Cholesky diagonal)

### 5.2 OnlineEMFitter
- `jax.lax.scan` per epoch, Robbins-Monro step sizes τₜ = τ₀ + t
- Update running sufficient statistics η

### 5.3 MiniBatchEMFitter
- Random mini-batches, Robbins-Monro averaging on η

### 5.4 Multi-start
- `jax.vmap(fit_one)(keys)` for parallel initializations

### 5.5 Validation
- Fit GH to S&P 500 data, compare with NumPy normix results
- EM convergence curves (log-likelihood per iteration)
- Multi-start: verify best model selection

## Phase 6: Integration

**Goal:** Clean public API, documentation, packaging.

### 6.1 Public API
- `normix.GIG`, `normix.GeneralizedHyperbolic`, etc.
- `normix.fit(X, distribution='gh', method='em')` convenience function
- Type stubs for IDE support

### 6.2 Documentation
- Sphinx docs with mathematical formulas
- Notebooks: one per distribution family
- README with quick-start examples

### 6.3 Testing
- Property-based tests: roundtrip consistency for all constructors
- Numerical tests: log_prob vs NumPy reference at matching parameters
- EM regression tests: fitted parameters within tolerance of known results
- Benchmark: JAX vs NumPy wall-clock time on S&P 500 data

## Cross-cutting Concerns

### Float64
Set `jax.config.update("jax_enable_x64", True)` at package init. All arrays default to float64.

### NumPy Reference
Keep the current NumPy normix as a subpackage or separate reference for cross-validation. Never delete it until JAX version passes all equivalent tests.

### Immutability
All distributions are immutable `eqx.Module`. No `_fitted` flag, no `_invalidate_cache`. Parameters are set at construction. M-step returns a new model via `eqx.tree_at`.


## Phase 7: CPU Bessel Backend (Performance)

**Goal:** 15× EM speedup via hybrid JAX/CPU execution for Bessel-heavy hot paths.
**Branch:** `feat/jax-native-bessel-v2` → `cursor/cpu-bessel-design-9a75`
**Status:** Implemented (see `docs/design/cpu_bessel_design.md`)

### 7.1 `log_kv(v, z, backend='jax'|'cpu')` — Phase 1 ✓
- `backend='jax'` (default): unchanged pure-JAX path with `@custom_jvp`, JIT-able
- `backend='cpu'`: `scipy.special.kve`, vectorized numpy, fast for EM hot path
- Internal: `_log_kv_jax` (carries `@custom_jvp`) and `_log_kv_cpu` (scipy)

### 7.2 GIG CPU methods — Phase 2 ✓
- `GIG.expectation_params(backend='cpu')`: analytical Bessel ratios via scipy
- `GIG.expectation_params_batch(p, a, b, backend='cpu')`: vectorized (N,3) output
- `GIG.from_expectation(..., solver='cpu')`: self-contained scipy L-BFGS-B solver
  self-contained; no external legacy dependencies

### 7.3 E-step CPU path — Phase 3 ✓
- `NormalMixture.e_step(X, backend='cpu')`: hybrid path
  - Quad forms (`L⁻¹(x-μ)`, `‖z‖²`, `‖w‖²`) stay in JAX vmap (GPU-friendly)
  - Bessel calls go to CPU via `GIG.expectation_params_batch(backend='cpu')`
- `_posterior_gig_params(z2, w2)` added to all four `JointNormalMixture` subclasses

### 7.4 BatchEMFitter integration — Phase 4 ✓
- `BatchEMFitter(e_step_backend='cpu', m_step_solver='cpu')` for full CPU hot path
- Default values `'jax'`/`'newton'` preserve backward compatibility
