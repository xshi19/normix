# normix Implementation Roadmap

## Overview

Implementation follows a bottom-up approach, starting from the simplest distributions and building up to complex mixture models. Each step includes implementation, testing, and documentation. A step is complete only when all tests pass.

---

## Step 1: Base Classes

### 1.1 Exponential Family Base Class
**File:** `normix/base/exponential_family.py`

**Implementation:**
- Define `ParameterType` enum (CLASSICAL, NATURAL, EXPECTATION)
- Implement abstract `ExponentialFamily` base class:
  - Abstract methods:
    - `_sufficient_statistics(x)` → T(x)
    - `_log_partition(natural_params)` → A(η)
    - `_log_base_measure(x)` → log h(x)
    - `_classical_to_natural(params)` - parameter conversion
    - `_natural_to_classical(params)` - parameter conversion
    - `_natural_to_expectation(params)` - ∇A(η)
    - `_expectation_to_natural(params)` - inverse
    - `fit(X, y=None, **kwargs)` - sklearn-style fitting
  - Concrete methods:
    - `pdf(x)`, `logpdf(x)` - using exponential family form
    - `set_params(param_type, params)` - set in any parametrization
    - `get_params(param_type)` - get in any parametrization
    - `score(X, y=None)` - mean log-likelihood

**Utilities needed:**
- `normix/utils/validation.py`: `check_positive()`, `check_array_like()`, `check_shape()`

**Tests:** `tests/test_exponential_family.py`
- Test with simple exponential distribution
- Test parameter conversions
- Test abstract method requirements

**Documentation:**
- Docstrings with LaTeX formulas
- Example of creating a subclass

---

### 1.2 Mixture Distribution Base Class
**File:** `normix/base/mixture.py`

**Implementation:**
- Implement `NormalMixtureDistribution` abstract base class:
  - Properties: `mixing_distribution`, `conditional_params`
  - Abstract methods:
    - `mixing_distribution_class` - class property
    - `_initialize_parameters(X)` - initialization
    - `_e_step(X)` → expectations
    - `_m_step(X, expectations)` - parameter updates
    - `_compute_marginal_logpdf(x)` - marginal density
    - `_sample_conditional(y)` - sample X|Y
  - Concrete methods:
    - `joint_logpdf(x, y)` - f(x,y) = f(x|y)f(y)
    - `conditional_logpdf(x, y)` - f(x|y)
    - `marginal_logpdf(x)` - f(x)
    - `fit(X, method='em', ...)` - sklearn-style
    - `_fit_em(X, ...)` - EM framework
    - `rvs(size, return_mixing=False)` - sampling

**Tests:** `tests/test_mixture_base.py`
- Test with simple concrete example
- Test EM convergence
- Test joint/marginal consistency

**Documentation:**
- Docstrings explaining normal mixture formulation
- EM algorithm details

---

## Step 2: Simple Univariate Distributions

### 2.1 Multivariate Normal
**File:** `normix/distributions/multivariate/normal.py`

**Implementation:**
- `MultivariateNormal(ExponentialFamily)`:
  - Classical params: (μ, Σ)
  - Natural params: (Σ⁻¹μ, -½Σ⁻¹)
  - Sufficient statistics: (x, xx^T)
  - Wrap scipy.stats.multivariate_normal

**Utilities needed:**
- `normix/utils/validation.py`: `check_square_matrix()`, `check_positive_definite()`

**Tests:** `tests/test_multivariate_normal.py`
- Test parameter conversions
- Compare with scipy.stats.multivariate_normal
- Test fitting with simulated data

**Documentation:**
- Exponential family formulation of multivariate normal

---

### 2.2 Gamma Distribution
**File:** `normix/distributions/univariate/gamma.py`

**Implementation:**
- `Gamma(ExponentialFamily)`:
  - Classical params: (α, β) or (shape, rate)
  - Natural params: (α-1, -β)
  - Sufficient statistics: (log x, x)
  - sklearn-style API

**Tests:** `tests/test_gamma.py`
- Parameter conversions (all three types)
- Compare with scipy.stats.gamma
- Test fitting and parameter recovery

**Documentation:**
- Gamma as exponential family
- Parametrization conventions

---

### 2.3 Inverse Gamma Distribution
**File:** `normix/distributions/univariate/inverse_gamma.py`

**Implementation:**
- `InverseGamma(ExponentialFamily)`:
  - Classical params: (α, β)
  - Natural params and sufficient statistics
  - sklearn-style API

**Tests:** `tests/test_inverse_gamma.py`
- Compare with scipy.stats.invgamma
- Test exponential family structure

**Documentation:**
- Relationship to Gamma distribution

---

## Step 3: Generalized Inverse Gaussian

### 3.1 GIG Distribution
**File:** `normix/distributions/univariate/generalized_inverse_gaussian.py`

**Implementation:**
- `GIG(ExponentialFamily)`:
  - Classical params: (λ, χ, ψ)
  - Natural params: (λ-1, -χ/2, -ψ/2)
  - Sufficient statistics: (log x, 1/x, x)
  - Log partition: A(η) = log K_λ(√(χψ)) + (λ/2)log(ψ/χ)
  - Fitting methods: MLE, moments, natural gradient
  - Sampling: implement efficient algorithm
  - Properties: mean, var, moment(n), entropy
- Migrate useful code from `normix/legacy/gig.py`

**Utilities needed:**
- `normix/utils/bessel.py`: Migrate `logkv`, `logkvp`, `kvratio` from `normix/legacy/func.py`
- `normix/utils/numerical.py`: `log_sum_exp()`, `safe_log()`, `safe_div()`

**Tests:** `tests/test_gig.py`
- All three parameter conversions work correctly
- PDF integrates to 1 (numerical)
- Moments match theoretical values
- Fit simulated data, recover parameters
- Test edge cases (λ < 0, λ = 0, λ > 0)
- Compare with legacy implementation
- Numerical stability with extreme parameters

**Documentation:**
- GIG as exponential family
- Mathematical properties
- Relationship to other distributions
- Usage examples

---

### 3.2 Inverse Gaussian
**File:** `normix/distributions/univariate/inverse_gaussian.py`

**Implementation:**
- `InverseGaussian(GIG)`:
  - Special case with λ = -1/2
  - Standard params: (μ, λ)
  - Override methods for efficiency
  - Conversion to/from GIG params

**Tests:** `tests/test_inverse_gaussian.py`
- Equivalence with GIG(λ=-0.5)
- Compare with scipy.stats.invgauss
- Fitting and parameter recovery

**Documentation:**
- IG as special case of GIG
- When to use IG vs GIG

---

## Step 4: Simple Mixture Distributions

### 4.1 Normal-Gamma (Variance Gamma)
**Files:**
- `normix/distributions/mixtures/joint_variance_gamma.py` (joint f(x,y))
- `normix/distributions/mixtures/variance_gamma.py` (marginal f(x))

**Implementation:**
- `JointVG`: Joint distribution with X|Y ~ N(μ + γY, σY), Y ~ Gamma
- `VG(NormalMixtureDistribution)`: Marginal distribution
  - sklearn-style API: initialize empty, fit returns self
  - EM algorithm for fitting
  - Closed-form marginal density
  - Properties: mean, cov

**Tests:**
- `tests/test_joint_vg.py`: Joint distribution tests
- `tests/test_vg.py`: 
  - sklearn API (fit returns self, chaining)
  - EM convergence
  - Parameter recovery from simulated data
  - Univariate and multivariate cases

**Documentation:**
- VG as normal-gamma mixture
- EM algorithm details
- Usage examples

---

### 4.2 Normal-Inverse Gamma
**Files:**
- `normix/distributions/mixtures/joint_normal_inverse_gamma.py`
- `normix/distributions/mixtures/normal_inverse_gamma.py`

**Implementation:**
- `JointNInvG` and `NInvG` (similar structure to VG)
- X|Y ~ N(μ + γY, σY), Y ~ InvGamma

**Tests:**
- `tests/test_joint_ninvg.py`, `tests/test_ninvg.py`
- Similar to VG tests

**Documentation:**
- Normal-inverse gamma mixture
- Comparison with VG

---

## Step 5: Normal-Inverse Gaussian

### 5.1 NIG Distribution
**Files:**
- `normix/distributions/mixtures/joint_normal_inverse_gaussian.py`
- `normix/distributions/mixtures/normal_inverse_gaussian.py`

**Implementation:**
- `JointNIG`: X|Y ~ N(μ + γY, σY), Y ~ IG
- `NIG(NormalMixtureDistribution)`:
  - Marginal distribution (closed form)
  - EM algorithm for fitting
  - sklearn-style API

**Tests:**
- `tests/test_joint_nig.py`, `tests/test_nig.py`
- Parameter recovery
- Compare with literature values
- Univariate and multivariate

**Documentation:**
- NIG distribution properties
- Applications in finance
- Usage examples

---

## Step 6: Generalized Hyperbolic

### 6.1 GH Distribution (Final Step)
**Files:**
- `normix/distributions/mixtures/joint_generalized_hyperbolic.py`
- `normix/distributions/mixtures/generalized_hyperbolic.py`

**Implementation:**
- `JointGH`: X|Y ~ N(μ + ΓY, ΣY), Y ~ GIG(λ, χ, ψ)
  - Joint exponential family structure
  - Joint sufficient statistics
- `GH(NormalMixtureDistribution)`:
  - Params: (μ, Σ, Γ, λ, χ, ψ)
  - Marginal density using Bessel K functions
  - EM algorithm:
    - E-step: E[Y|X], E[log Y|X], E[1/Y|X] using Bessel ratios
    - M-step: update all parameters
  - Parameter regularization methods
  - sklearn-style API
- Migrate code from `normix/legacy/gh.py`

**Utilities needed:**
- Bessel function ratios for E-step
- Numerical stability for extreme parameters

**Tests:** `tests/test_joint_gh.py`, `tests/test_gh.py`
- sklearn API fully functional
- EM convergence with various initializations
- Parameter recovery from simulated data
- Univariate and multivariate cases
- Special cases: symmetric GH, boundary cases
- Compare with legacy implementation
- Integration tests with real data
- Verify all special cases (NIG, VG, etc.) as limits

**Documentation:**
- GH as the general case encompassing all previous distributions
- Mathematical derivation
- EM algorithm details
- Comprehensive examples
- Comparison with other packages (R's GeneralizedHyperbolic)

---

## Step 7: Package Integration

### 7.1 Main Package Interface
**File:** `normix/__init__.py`

**Implementation:**
```python
from normix.distributions.univariate import GIG, InverseGaussian, Gamma, InverseGamma
from normix.distributions.multivariate import MultivariateNormal
from normix.distributions.mixtures import GH, NIG, VG, NInvG
from normix.distributions.mixtures import JointGH, JointNIG, JointVG, JointNInvG

__version__ = "1.0.0"
__all__ = [
    "GIG", "InverseGaussian", "Gamma", "InverseGamma",
    "MultivariateNormal",
    "GH", "NIG", "VG", "NInvG",
    "JointGH", "JointNIG", "JointVG", "JointNInvG",
]
```

**Tests:** `tests/test_package.py`
- All imports work correctly
- API consistency across distributions
- Backward compatibility where possible

**Documentation:**
- Update main README.md
- API overview
- Quick start guide

---

### 7.2 Examples and Tutorials

**Examples to create:**
- `examples/01_basic_gig.py` - GIG basics
- `examples/02_parameter_conversions.py` - Exponential family parametrizations
- `examples/03_gh_fitting.py` - Fitting GH to data
- `examples/04_mixture_interpretation.py` - Understanding normal mixtures
- `examples/05_special_cases.py` - NIG, VG as special cases

**Tutorials:**
- `docs/tutorials/getting_started.rst`
- `docs/tutorials/exponential_families.rst`
- `docs/tutorials/generalized_hyperbolic.rst`

---

## Testing Strategy

### Per-Step Testing
Each step includes its own tests that must pass before moving to the next step.

### Test Categories
1. **Unit tests**: Test individual methods
2. **Integration tests**: Test workflows across components
3. **Numerical tests**: Compare with reference implementations (scipy, R)
4. **Edge case tests**: Extreme parameters, boundary conditions
5. **API tests**: Consistent sklearn-style interface

### Test Coverage
- Aim for >90% coverage
- All public methods tested
- Edge cases covered

---

## Implementation Order Summary

1. **Base classes** → Foundation for everything
2. **Normal, Gamma, InvGamma** → Simple distributions to test framework
3. **GIG** → Key mixing distribution
4. **InverseGaussian** → Special case of GIG
5. **VG, NInvG** → Simple mixtures
6. **NIG** → More complex mixture
7. **GH** → Most general case, encompasses all others
8. **Package integration** → Clean public API

Each step builds on previous ones. Complete one step fully (code + tests + docs) before moving to the next.

---

## Migration from Legacy Code

### Legacy Files (for reference only):
- `normix/legacy/gig.py` - Original GIG implementation
- `normix/legacy/gh.py` - Original GH implementation
- `normix/legacy/func.py` - Original utility functions

These files are kept for reference during refactoring. Do not import from legacy in new code.

### Key improvements in new code:
- sklearn-style API (fit returns self)
- Exponential family structure
- Three parametrizations (classical, natural, expectation)
- Better numerical stability
- Comprehensive testing
- Type hints and documentation

---

## Current Status

- [x] Package structure created
- [x] Legacy code moved to `normix/legacy/`
- [x] Step 1: Base classes
  - [x] 1.1 ExponentialFamily base class with three parametrizations
  - [x] 1.2 Mixture base class with JointNormalMixture and NormalMixture
- [x] Step 2: Simple univariate distributions
  - [x] 2.1 Multivariate Normal (exponential family form)
  - [x] 2.2 Gamma distribution
  - [x] 2.3 Inverse Gamma distribution
  - [x] Exponential distribution (bonus)
- [x] Step 3: GIG and IG
  - [x] 3.1 Generalized Inverse Gaussian (GIG)
  - [x] 3.2 Inverse Gaussian (IG)
- [x] Step 4: Simple mixtures (VG, NInvG)
  - [x] 4.1 Variance Gamma (JointVarianceGamma, VarianceGamma)
  - [x] 4.2 Normal-Inverse Gamma (JointNormalInverseGamma, NormalInverseGamma)
- [x] Step 5: NIG
  - [x] 5.1 Joint Normal Inverse Gaussian (JointNormalInverseGaussian)
  - [x] 5.2 Normal Inverse Gaussian marginal (NormalInverseGaussian)
- [ ] Step 6: GH
- [ ] Step 7: Package integration

### Tests Completed
- [x] `tests/test_exponential_family.py` - Base class tests
- [x] `tests/test_distributions_vs_scipy.py` - Generic scipy comparison framework
- [x] `tests/test_variance_gamma.py` - Variance Gamma tests
- [x] `tests/test_normal_inverse_gamma.py` - Normal Inverse Gamma tests
- [x] `tests/test_normal_inverse_gaussian.py` - Normal Inverse Gaussian tests

### Notebooks Completed
- [x] `notebooks/exponential_distribution.ipynb`
- [x] `notebooks/gamma_distribution.ipynb`
- [x] `notebooks/inverse_gamma_distribution.ipynb`
- [x] `notebooks/inverse_gaussian_distribution.ipynb`
- [x] `notebooks/generalized_inverse_gaussian_distribution.ipynb`
- [x] `notebooks/multivariate_normal_distribution.ipynb`
- [x] `notebooks/variance_gamma_distribution.ipynb`

**Next action:** Implement Step 6 - Generalized Hyperbolic distribution (the most general case)
