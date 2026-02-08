# Refactoring Plan: `cached_property` + Frozen Dataclass Parameters

## Motivation

The current `ExponentialFamily` base class stores natural parameters as a flat `tuple` and
uses `lru_cache` for parameter conversions. This creates three problems:

1. **Flat tuple storage is wasteful for multivariate distributions.** For `MultivariateNormal`
   with $d=50$, the tuple has $50 + 2500 = 2550$ entries. Every access requires reshaping
   and matrix inversions.

2. **Repeated matrix inversions.** `logpdf()`, `mean()`, `var()`, `rvs()` all call
   `get_classical_params()` which triggers `_natural_to_classical()` → `np.linalg.inv()`.

3. **Ad-hoc Cholesky caching.** `JointNormalMixture` has `_L_Sigma`, `set_L_Sigma()`,
   `clear_L_Sigma_cache()` bolted on outside the base class framework.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Caching mechanism | `functools.cached_property` (stdlib) | Same pattern as PyTorch `lazy_property`; no deps |
| Cache invalidation | `__dict__.pop()` via `_invalidate_cache()` | Standard Python pattern (Django, SQLAlchemy) |
| Parameter containers | `@dataclass(frozen=True, slots=True)` | stdlib, IDE autocompletion, immutable |
| Cache infra location | `Distribution` base class | Shared by both `ExponentialFamily` and `NormalMixture` |
| Internal storage | Subclass-defined attributes | Each dist stores its optimal representation |

## Architecture: Before → After

**Before:**
```
_natural_params (flat tuple) → get_classical_params() → dict → method code
                                ↑ lru_cache on tuple key
```

**After:**
```
Internal attributes (e.g. _rate, _mu, _L)
    → cached_property for each parametrization (returns frozen dataclass)
    → _invalidate_cache() clears __dict__ entries on any state change
```

## Phase Dependency Graph

```
Phase 0: Parameter dataclasses + test infrastructure
    │
Phase 1a: Distribution base class (cache infra)
    │
Phase 1b: ExponentialFamily (cached_property, new abstract methods)
    │
    ├──────────────────┬─────────────────────┐
    │                  │                     │
Phase 2:           Phase 3:             Phase 4:
Univariate dists   MultivariateNormal   JointNormalMixture base
                                             │
                                        Phase 5:
                                        Joint mixture dists
                                             │
                                        Phase 6:
                                        NormalMixture base
                                             │
                                        Phase 7:
                                        Marginal mixture dists
                                             │
                                        Phase 8:
                                        Cleanup + docs
```

Phases 2, 3, 4 are independent after Phase 1b.

---

## Phase 0: Parameter Dataclasses + Test Infrastructure

**New file:** `normix/params.py`

Frozen dataclasses for every distribution's classical parameters:

```python
@dataclass(frozen=True, slots=True)
class ExponentialParams:
    rate: float

@dataclass(frozen=True, slots=True)
class GammaParams:
    shape: float
    rate: float

@dataclass(frozen=True, slots=True)
class InverseGammaParams:
    shape: float
    scale: float

@dataclass(frozen=True, slots=True)
class InverseGaussianParams:
    mean: float
    shape: float

@dataclass(frozen=True, slots=True)
class GIGParams:
    p: float
    a: float
    b: float

@dataclass(frozen=True, slots=True)
class MultivariateNormalParams:
    mu: np.ndarray
    sigma: np.ndarray

@dataclass(frozen=True, slots=True)
class VarianceGammaParams:
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    shape: float
    rate: float

@dataclass(frozen=True, slots=True)
class NormalInverseGammaParams:
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    shape: float
    rate: float

@dataclass(frozen=True, slots=True)
class NormalInverseGaussianParams:
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    delta: float
    eta: float

@dataclass(frozen=True, slots=True)
class GHParams:
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    p: float
    a: float
    b: float
```

**New file:** `tests/test_params.py`
- Dataclass construction with valid values
- Frozen immutability (assigning raises `FrozenInstanceError`)
- `dataclasses.asdict()` roundtrip
- Backward compat: dict-like access via `asdict()`

**Commit:** `feat(params): add frozen dataclass parameter containers`

---

## Phase 1a: Distribution Base Class — Cache Infrastructure

**File:** `normix/base/distribution.py`

Add to `Distribution.__init__()`:
```python
self._fitted = False
```

Add methods:
```python
_cached_attrs: tuple[str, ...] = ()

def _check_fitted(self):
    if not self._fitted:
        raise ValueError(
            f"{self.__class__.__name__} parameters not set. "
            "Use from_*_params() or fit()."
        )

def _invalidate_cache(self):
    for attr in self._cached_attrs:
        self.__dict__.pop(attr, None)
```

**Tests:** `tests/test_cached_property_invalidation.py`
- Minimal subclass that uses `cached_property`
- Verify `_invalidate_cache()` clears the cached value
- Verify `_invalidate_cache()` is idempotent (safe when no cache)
- Verify `_check_fitted()` raises before fitting, passes after

**Commit:** `refactor(base): add cache infrastructure to Distribution`

---

## Phase 1b: ExponentialFamily — `cached_property` + New Abstract Methods

**File:** `normix/base/exponential_family.py`

### Backward Compatibility Strategy

Both old and new interfaces coexist. The base class detects which a subclass
implements and dispatches accordingly. This allows migrating one distribution
at a time.

**Old interface** (kept during transition):
- `_classical_to_natural(**kwargs) → NDArray`
- `_natural_to_classical(theta) → dict`

**New interface** (subclasses implement these):
- `_set_from_classical(**kwargs) → None` (sets internal attrs, calls `_invalidate_cache()`)
- `_set_from_natural(theta: NDArray) → None` (sets internal attrs, calls `_invalidate_cache()`)
- `_compute_natural_params() → NDArray` (derives from internal state)
- `_compute_classical_params() → dataclass` (derives from internal state)

### Changes

1. Remove `_natural_params = None` (tuple storage), replace with `_fitted` flag from `Distribution`
2. Remove `lru_cache` imports and `_get_expectation_params_cached` / `_get_classical_params_cached`
3. Remove `_clear_param_cache()` (replaced by `_invalidate_cache()`)
4. Add `_cached_attrs = ('natural_params', 'classical_params', 'expectation_params')`
5. Add `cached_property` for `natural_params`, `classical_params`, `expectation_params`
6. Add backward-compat dispatch in `set_classical_params()` / `set_natural_params()`
7. Update `logpdf()`, `fit()`, `__repr__` to use `_check_fitted()` + properties

### Dispatch Logic

```python
def set_classical_params(self, **kwargs):
    if self._has_new_interface():
        self._set_from_classical(**kwargs)
    else:
        # Old path for un-migrated subclasses
        theta = self._classical_to_natural(**kwargs)
        self._store_natural_params_legacy(theta)
    return self

def _has_new_interface(self):
    return type(self)._set_from_classical is not ExponentialFamily._set_from_classical
```

**Tests:** Update `tests/test_exponential_family.py`
- All 25 existing tests must pass (old interface still works)
- New: `_check_fitted()` behavior
- New: `cached_property` invalidation cycle
- New: Both old and new interface paths work

**Commit:** `refactor(base): ExponentialFamily with cached_property and backward compat`

---

## Phase 2: Univariate Distributions (5 files)

Migrate in order: Exponential → Gamma → InverseGamma → InverseGaussian → GIG

For each distribution:

1. Add internal attributes (e.g., `self._rate` for Exponential)
2. Implement `_set_from_classical()`, `_set_from_natural()`
3. Implement `_compute_natural_params()`, `_compute_classical_params()`
4. Update `mean()`, `var()`, `rvs()`, `cdf()` to use internal attributes directly
5. Keep old `_classical_to_natural()` / `_natural_to_classical()` as they are
   (they still work via the backward compat layer, and we remove them in Phase 8)

### Internal State Per Distribution

| Distribution | Internal Attributes |
|---|---|
| `Exponential` | `_rate: float` |
| `Gamma` | `_shape: float`, `_rate: float` |
| `InverseGamma` | `_shape: float`, `_scale: float` |
| `InverseGaussian` | `_mean_param: float`, `_shape: float` |
| `GIG` | `_p: float`, `_a: float`, `_b: float` |

### Tests

- All existing tests in `test_exponential_family.py` and `test_distributions_vs_scipy.py` must pass
- New tests per distribution:
  - `classical_params` returns correct dataclass type (e.g., `ExponentialParams`)
  - Attribute access works: `params.rate`, `params.shape`, etc.
  - `get_classical_params()` still works (backward compat)
  - Cache invalidation after `set_classical_params()`

**Commit:** `refactor(univariate): migrate to internal attributes + dataclass params`

---

## Phase 3: MultivariateNormal — Cholesky-First Storage

**File:** `normix/distributions/multivariate/normal.py`

### Internal State

```python
self._mu: NDArray     # (d,) mean vector
self._L: NDArray      # (d,d) lower Cholesky of Sigma: Sigma = L @ L.T
```

### Cached Properties

```python
_cached_attrs = ExponentialFamily._cached_attrs + ('log_det_Sigma', 'L_inv')

@cached_property
def log_det_Sigma(self) -> float:
    return 2.0 * np.sum(np.log(np.diag(self._L)))

@cached_property
def L_inv(self) -> NDArray:
    """Lower triangular inverse: L_inv @ L = I. Used to get Lambda = L_inv.T @ L_inv."""
    return solve_triangular(self._L, np.eye(self._d), lower=True)
```

### Key Changes

- `_set_from_classical(*, mu, sigma)` → `cholesky(sigma)` stored as `self._L`
- `_set_from_natural(theta)` → Cholesky-solve to get `mu`, compute `L` from `L_Lambda`
- `logpdf()` uses `solve_triangular(self._L, ...)` — no `np.linalg.inv`
- `rvs()` uses `self._L` directly for sampling
- `_compute_classical_params()` → `MultivariateNormalParams(mu=..., sigma=L@L.T)`
- `_compute_natural_params()` → builds flat theta from `L_inv`

### Tests

**Existing (must pass):**
- `test_distributions_vs_scipy.py::TestMultivariateNormalVsScipy`
- `test_multivariate_rvs.py` (all)
- `test_high_dimensional.py` (all)

**New:** `tests/test_multivariate_normal_refactor.py`
- `L @ L.T == Sigma` (Cholesky correctness)
- `log_det_Sigma == np.linalg.slogdet(Sigma)[1]`
- `logpdf` matches `scipy.stats.multivariate_normal.logpdf`
- `classical_params` returns `MultivariateNormalParams`
- Cache invalidation works

**Commit:** `refactor(mvn): Cholesky-first storage, eliminate np.linalg.inv from logpdf`

---

## Phase 4: JointNormalMixture Base — Cached Cholesky Properties

**File:** `normix/base/mixture.py` (`JointNormalMixture` class)

### Internal State

```python
self._mu: NDArray          # (d,) location
self._gamma: NDArray       # (d,) skewness
self._L_Sigma: NDArray     # (d,d) lower Cholesky of Sigma
```

### Cached Properties

```python
_cached_attrs = ExponentialFamily._cached_attrs + (
    'log_det_Sigma', 'L_Sigma_inv', 'gamma_mahal_sq'
)

@cached_property
def log_det_Sigma(self) -> float:
    return 2.0 * np.sum(np.log(np.diag(self._L_Sigma)))

@cached_property
def L_Sigma_inv(self) -> NDArray:
    return solve_triangular(self._L_Sigma, np.eye(self._d), lower=True)

@cached_property
def gamma_mahal_sq(self) -> float:
    """gamma^T Sigma^{-1} gamma — used in marginal PDFs."""
    z = solve_triangular(self._L_Sigma, self._gamma, lower=True)
    return float(z @ z)
```

### Key Changes

1. Remove `get_L_Sigma()`, `set_L_Sigma()`, `clear_L_Sigma_cache()` ad-hoc methods
2. Add `_set_normal_params(mu, gamma, L_Sigma)` helper for M-step direct updates
3. Update `_extract_normal_params_from_theta()` and `_extract_normal_params_with_cholesky()`
   to work from internal attributes when available
4. Keep theta-based fallback for un-migrated subclasses

### Tests

- `test_multivariate_rvs.py` must pass (tests `get_L_Sigma`, Cholesky correctness)
- `test_high_dimensional.py` must pass
- New: `L_Sigma_inv`, `gamma_mahal_sq` correctness tests

**Commit:** `refactor(mixture-base): JointNormalMixture with cached Cholesky properties`

---

## Phase 5: Joint Mixture Distributions (4 files)

Migrate: JointVG → JointNInvG → JointNIG → JointGH

For each joint distribution:

1. Internal state: `_mu`, `_gamma`, `_L_Sigma` (from base) + mixing params
2. `_set_from_classical()` → `cholesky(sigma)`, store mixing params
3. `_set_from_natural(theta)` → Cholesky-solve, store attrs
4. `_compute_classical_params()` → returns appropriate dataclass
5. `_log_partition()` uses internal attributes directly

### Mixing Parameter Storage

| Joint Distribution | Additional Internal Attrs |
|---|---|
| `JointVarianceGamma` | `_mixing_shape: float`, `_mixing_rate: float` |
| `JointNormalInverseGamma` | `_mixing_shape: float`, `_mixing_rate: float` |
| `JointNormalInverseGaussian` | `_mixing_delta: float`, `_mixing_eta: float` |
| `JointGeneralizedHyperbolic` | `_mixing_p: float`, `_mixing_a: float`, `_mixing_b: float` |

### Tests

- All existing tests for each distribution must pass
- New: `classical_params` returns correct dataclass type
- New: `_log_partition()` regression test (same values as before)

**Commit:** `refactor(joint-mixtures): migrate Joint{VG,NInvG,NIG,GH} to internal attrs`

---

## Phase 6: NormalMixture Base Class

**File:** `normix/base/mixture.py` (`NormalMixture` class)

### Changes

1. Use `_fitted` and `_invalidate_cache()` from `Distribution`
2. `classical_params` as `cached_property` delegating to `self.joint.classical_params`
3. `set_classical_params()` calls joint setter + `self._invalidate_cache()`
4. `mean()`, `var()`, `cov()` access `self.joint._mu`, `self.joint._gamma` directly
5. Update `__repr__` to use `_fitted`

**Tests:** All existing NormalMixture tests must pass.

**Commit:** `refactor(mixture-base): NormalMixture using cached_property from Distribution`

---

## Phase 7: Marginal Mixture Distributions (4 files)

Migrate: VG → NInvG → NIG → GH

For each marginal distribution:

1. `_marginal_logpdf()` accesses `self.joint._mu`, `self.joint._gamma`,
   `self.joint.L_Sigma_inv`, `self.joint.log_det_Sigma` directly
2. `_conditional_expectation_y_given_x()` same direct access
3. `fit()` / `_m_step()` updates params via `self.joint._set_normal_params()`
   and mixing param setters — no more converting to theta and back
4. Remove `get_classical_params()` calls inside hot loops

### Critical: GH EM Algorithm

The M-step in `generalized_hyperbolic.py` currently does:
```python
self._joint.set_classical_params(mu=..., gamma=..., sigma=..., p=..., a=..., b=...)
self._joint.set_L_Sigma(L_Sigma)
```
After refactor:
```python
self._joint._set_normal_params(mu_new, gamma_new, L_Sigma_new)
self._joint._set_mixing_params(p_new, a_new, b_new)
self._joint._invalidate_cache()
```

### Tests

- ALL existing tests must pass (parameter recovery, EM convergence, edge cases)
- New: EM regression test with fixed seed, compare final params to 4 digits

**Commit:** `refactor(marginal-mixtures): migrate VG, NInvG, NIG, GH to direct attr access`

---

## Phase 8: Cleanup + Documentation

1. Remove backward compat dispatch in `ExponentialFamily` (old `_classical_to_natural` /
   `_natural_to_classical` fallback path)
2. Remove old `_classical_to_natural()` / `_natural_to_classical()` from migrated subclasses
3. Update `normix/__init__.py` exports to include `normix.params`
4. Update `normix/base/__init__.py` exports
5. Update ROADMAP.md
6. Update `.cursor/rules/project-overview.mdc`
7. Run full test suite + manual notebook validation

**Commit:** `chore: remove backward compat shims, update exports and docs`

---

## Test Strategy

### Invariant After Every Phase

```bash
uv run pytest tests/ → 206+ passed, 0 failed
```

### New Test Files

| File | Phase | Purpose |
|------|-------|---------|
| `tests/test_params.py` | 0 | Dataclass construction, immutability |
| `tests/test_cached_property_invalidation.py` | 1a | Cache infra on Distribution |
| `tests/test_multivariate_normal_refactor.py` | 3 | Cholesky internals, no-inv |

### Existing Test Files (must pass throughout)

| File | Tests | Key Assertions |
|------|-------|----------------|
| `test_exponential_family.py` | 25 | Parameter roundtrips, logpdf |
| `test_distributions_vs_scipy.py` | ~35 | PDF/CDF vs scipy |
| `test_variance_gamma.py` | ~30 | VG joint+marginal, EM |
| `test_normal_inverse_gamma.py` | ~25 | NInvG joint+marginal, EM |
| `test_normal_inverse_gaussian.py` | ~25 | NIG joint+marginal, EM |
| `test_generalized_hyperbolic.py` | ~30 | GH joint+marginal, EM, regularization |
| `test_multivariate_rvs.py` | ~15 | Cholesky, rvs covariance |
| `test_high_dimensional.py` | ~15 | d=100 roundtrips, performance |

---

## Risk Mitigation

1. **EM convergence regression:** EM algorithms are numerically sensitive. After Phase 7,
   run each EM test with a fixed seed and compare final parameters to 4-digit precision.

2. **`frozen=True` dataclass with numpy:** `frozen=True` prevents attribute reassignment,
   but numpy arrays are internally mutable (`params.mu[0] = 999` still works). Document
   this. Optionally set `arr.flags.writeable = False` in `__post_init__`.

3. **`cached_property` + `__slots__`:** Distribution classes must NOT use `__slots__`
   (cached_property stores in `__dict__`). Only the dataclass param containers use
   `slots=True`.

4. **Inheritance of `_cached_attrs`:** Each subclass extends via tuple concatenation:
   ```python
   class MultivariateNormal(ExponentialFamily):
       _cached_attrs = ExponentialFamily._cached_attrs + ('log_det_Sigma', 'L_inv')
   ```

---

## Commit Sequence

| # | Phase | Commit Message |
|---|-------|---------------|
| 1 | 0 | `feat(params): add frozen dataclass parameter containers` |
| 2 | 1a | `refactor(base): add cache infrastructure to Distribution` |
| 3 | 1b | `refactor(base): ExponentialFamily with cached_property and backward compat` |
| 4 | 2 | `refactor(univariate): migrate to internal attributes + dataclass params` |
| 5 | 3 | `refactor(mvn): Cholesky-first storage, eliminate np.linalg.inv from logpdf` |
| 6 | 4 | `refactor(mixture-base): JointNormalMixture with cached Cholesky properties` |
| 7 | 5 | `refactor(joint-mixtures): migrate Joint{VG,NInvG,NIG,GH} to internal attrs` |
| 8 | 6+7 | `refactor(marginal-mixtures): migrate VG, NInvG, NIG, GH marginals + EM` |
| 9 | 8 | `chore: remove backward compat shims, update exports and docs` |
