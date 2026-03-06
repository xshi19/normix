# Survey: JAX Distribution Ecosystem

**Date:** 2026-03-06  
**Purpose:** Comprehensive survey of JAX-based probability distribution packages, evaluated for potential use as a foundation for normix's JAX migration.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [NumPyro](#2-numpyro)
3. [BlackJAX](#3-blackjax)
4. [jax.scipy.stats](#4-jaxscipystats)
5. [Dynamax](#5-dynamax)
6. [efax](#6-efax)
7. [Distrax](#7-distrax)
8. [Distreqx](#8-distreqx)
9. [FlowJAX](#9-flowjax)
10. [GMMX](#10-gmmx)
11. [TensorFlow Probability (JAX substrate)](#11-tensorflow-probability-jax-substrate)
12. [Minor / Niche Libraries](#12-minor--niche-libraries)
13. [Specific Distribution Availability](#13-specific-distribution-availability)
14. [The Bessel Function Gap](#14-the-bessel-function-gap)
15. [Comparison Matrix](#15-comparison-matrix)
16. [Recommendations for normix](#16-recommendations-for-normix)

---

## 1. Executive Summary

The JAX distribution ecosystem is fragmented across many libraries, each with a distinct focus:

| Library | Focus | Exp. Family? | Fitting? | Mixtures? | Maintenance |
|---------|-------|:---:|:---:|:---:|-------------|
| **NumPyro** | Probabilistic programming | No | Via SVI/MCMC | Yes | Active (2.6k stars) |
| **TFP-JAX** | General distributions | Partial (GLM) | Via optimizers | Yes | Active (4k+ stars) |
| **efax** | Exponential families | **Yes** | MLE via exp params | No | Active (75 stars) |
| **Distrax** | Lightweight distributions | No | No | No | Stale (~600 stars) |
| **Distreqx** | Distrax on Equinox | No | No | No | Low activity (30 stars) |
| **FlowJAX** | Normalizing flows | No | Via flows | No | Active (68 stars) |
| **GMMX** | Gaussian mixtures | No | EM | GMM only | Active (24 stars) |
| **Dynamax** | State space models | No | EM/SGD | HMM emissions | Active (810 stars) |
| **BlackJAX** | MCMC/SMC samplers | No | No | No | Active (900+ stars) |
| **jax.scipy.stats** | Stateless functions | No | No | No | Part of JAX |

**Key finding:** No existing library provides what normix needs — exponential family distributions with natural/expectation/classical parametrizations, mixture distributions (VG, NIG, GH), EM fitting, *and* the GIG/GH family. The closest is **efax** (exponential family focus) but it lacks GIG, GH, and mixture distributions. A custom implementation building on JAX primitives remains the best path.

---

## 2. NumPyro

**Repository:** [pyro-ppl/numpyro](https://github.com/pyro-ppl/numpyro)  
**Stars:** ~2,600 | **Last release:** v0.20.0 (Jan 2025) | **License:** Apache-2.0  
**Status:** Actively maintained, OpenSSF certified

### Base Class Design

```python
class Distribution:
    pytree_data_fields = ()      # Parameters (participate in JAX transforms)
    pytree_aux_fields = ('_batch_shape', '_event_shape')  # Metadata
    arg_constraints = {}
    support = ...
    has_enumerate_support = False
    reparametrized_params = []
```

Follows PyTorch's `torch.distributions` API. Distributions are registered as **JAX PyTrees** via `tree_flatten()` / `tree_unflatten()`.

### Key Methods

| Method | Description |
|--------|-------------|
| `sample(key, sample_shape=())` | Generate samples (requires PRNG key) |
| `log_prob(value)` | Log probability density/mass |
| `cdf(value)` | Cumulative distribution function (v0.20.0+) |
| `mean` | Property: distribution mean |
| `variance` | Property: distribution variance |
| `enumerate_support()` | For discrete distributions |

### PyTree / JIT / vmap / grad

**Yes** — fully pytree-compatible. Distributions can be passed through `jit`, `vmap`, `grad`.

### Fitting / MLE

No direct `fit()` method. Fitting is done through NumPyro's inference machinery:
- **SVI** (Stochastic Variational Inference) with `AutoGuide`s
- **MCMC** (NUTS, HMC) via BlackJAX-compatible kernels
- Not designed for direct MLE of distribution parameters

### Mixture Distributions

**Yes** — comprehensive support:
- `MixtureSameFamily`: All components same distribution type (vectorized)
- `MixtureGeneral`: Different component distribution types
- `Mixture()` factory function auto-selects the right class
- Methods: `component_mean`, `component_variance`, `component_log_probs()`, `component_sample()`, `component_cdf()`

### Exponential Family Structure

**No explicit support.** No natural parameters, sufficient statistics, or log partition function in the base class. Distributions are parameterized in classical form only.

### Distributions Available (relevant to normix)

- Normal, MultivariateNormal, LowRankMultivariateNormal
- Gamma, Beta, Exponential, Dirichlet
- InverseGamma: **Yes** (in `numpyro.distributions.continuous`)
- InverseGaussian: **Not confirmed** — not in standard distribution list
- **GIG: No**
- **GH: No**
- StudentT, Cauchy, Laplace, LogNormal, Pareto, Weibull, etc.

### Suitability for normix

**Partial.** NumPyro's `Distribution` base class is well-designed and pytree-compatible but lacks exponential family structure. Using NumPyro as a dependency would bring in the full probabilistic programming stack (models, inference, etc.) which is heavyweight for normix's needs. The mixture infrastructure is excellent but normix's mixtures are variance-mean mixtures (not finite component mixtures).

---

## 3. BlackJAX

**Repository:** [blackjax-devs/blackjax](https://github.com/blackjax-devs/blackjax)  
**Stars:** ~900+ | **License:** Apache-2.0  
**Status:** Actively maintained

### Distribution Abstraction

**BlackJAX has NO distribution abstraction.** It operates purely on log-density functions:

```python
logprior_fn(states: pytree) -> Array        # scalar
loglikelihood_fn(states: pytree, data) -> Array  # scalar
logposterior_fn(states: pytree, data) -> Array   # scalar
```

BlackJAX is a *sampling library* (MCMC, SMC, VI) that consumes log-densities. It doesn't define, wrap, or provide distributions — it expects the user (or another library like NumPyro/TFP) to supply `logdensity_fn`.

### Key Design: Functional Kernels

```python
def kernel(rng_key, state, **parameters) -> (state, info)
```

### Suitability for normix

**Not applicable.** BlackJAX is a sampler, not a distribution library. It could be used *downstream* of normix (e.g., for Bayesian parameter inference with normix distributions as likelihoods), but provides nothing for building distribution classes.

---

## 4. jax.scipy.stats

**Part of JAX itself** — no separate install needed.

### Available Distributions

Based on the JAX documentation, `jax.scipy.stats` provides **stateless function modules** (not classes) for:

| Distribution | pdf | logpdf | cdf | logcdf | ppf | sf | logsf | isf |
|-------------|:---:|:------:|:---:|:------:|:---:|:--:|:-----:|:---:|
| `norm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `t` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `expon` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gamma` | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| `chi2` | ✓ | ✓ | ✓ | — | — | ✓ | — | — |
| `beta` | ✓ | ✓ | — | — | — | — | — | — |
| `multivariate_normal` | ✓ | ✓ | — | — | — | — | — | — |
| `uniform` | ✓ | ✓ | ✓ | — | ✓ | — | — | — |
| `laplace` | ✓ | ✓ | ✓ | — | — | — | — | — |
| `cauchy` | ✓ | ✓ | ✓ | — | — | — | — | — |
| `logistic` | ✓ | ✓ | ✓ | — | — | ✓ | — | — |
| `pareto` | ✓ | ✓ | — | — | — | — | — | — |
| `poisson` | ✓ | ✓ | — | — | — | — | — | — |
| `bernoulli` | — | ✓ | — | — | — | — | — | — |
| `truncnorm` | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — |
| `nbinom` | ✓ | ✓ | — | — | — | — | — | — |
| `geom` | ✓ | ✓ | — | — | — | — | — | — |
| `lognorm` | ✓ | ✓ | ✓ | — | — | — | — | — |
| `vonmises` | ✓ | ✓ | — | — | — | — | — | — |

### API Style

These are **pure functions**, not classes. They follow SciPy's `loc`/`scale` convention:

```python
jax.scipy.stats.norm.logpdf(x, loc=0, scale=1)
jax.scipy.stats.gamma.pdf(x, a, loc=0, scale=1)
```

### PyTree / JIT / vmap / grad

**Yes** — these are plain JAX functions, fully compatible with all JAX transformations.

### Fitting / MLE / Mixtures / Exponential Family

**None.** These are raw density/CDF evaluators only. No parameter estimation, no distribution objects, no exponential family structure.

### Missing Distributions (relevant to normix)

- **No** InverseGamma, InverseGaussian, GIG, GH
- **No** Bessel functions ($K_\nu$) in `jax.scipy.special`

### Suitability for normix

**Useful as building blocks** (e.g., `jax.scipy.stats.norm.logpdf` for computing Normal log-densities inside a mixture), but provides nothing for distribution architecture. normix would call these functions internally but not inherit from or wrap them.

---

## 5. Dynamax

**Repository:** [probml/dynamax](https://github.com/probml/dynamax)  
**Stars:** ~810 | **Last release:** v1.0.1 | **License:** MIT  
**Status:** Actively maintained, JOSS-published (April 2025)

### Distribution Design

Dynamax does **not** provide general-purpose distribution classes. It has model-specific emission/transition distributions tightly coupled to state space models:

- **HMM emissions:** `GaussianHMM`, `DiagonalGaussianHMM`, `SphericalGaussianHMM`, `SharedCovarianceGaussianHMM`
- **Linear Gaussian SSM:** $y_t | z_t \sim \mathcal{N}(Hz_t, R)$
- **Conjugate priors:** Normal-Inverse Wishart for emission parameters

Fitting via EM and SGD (using optax). Bayesian inference via BlackJAX integration.

### Suitability for normix

**Not applicable.** Dynamax's distributions are SSM-specific. However, its EM implementation patterns (functional style, JAX-native) could serve as reference for normix's JAX EM.

---

## 6. efax

**Repository:** [NeilGirdhar/efax](https://github.com/NeilGirdhar/efax)  
**Stars:** ~75 | **Last release:** v1.22.3 (Aug 2025) | **License:** Apache-2.0  
**Status:** Actively maintained, 572+ commits

### Base Class Design

efax is the **most architecturally relevant** library for normix. It centers on exponential families with two parametrization types:

```python
class NaturalParametrization:
    def to_exp(self) -> ExpectationParametrization  # Convert to expectation params
    def log_normalizer(self) -> Array                # Log partition function ψ(θ)
    def sufficient_statistic(self, x) -> ExpectationParametrization
    def carrier_measure(self, x) -> Array            # h(x)
    def log_prob(self, x) -> Array
    def sample(self, key, shape) -> Array
    def flattened() / unflattened()                   # For ML model integration

class ExpectationParametrization:
    def to_nat(self) -> NaturalParametrization  # Convert to natural params
```

All parametrization objects are `tjax.dataclass` (JAX PyTree-registered dataclasses).

### Key Methods

| Method | Description |
|--------|-------------|
| `log_prob(x)` | Log density evaluation |
| `sample(key, shape)` | Random sampling |
| `to_exp()` / `to_nat()` | Parametrization conversion |
| `log_normalizer()` | Log partition function $\psi(\theta)$ |
| `sufficient_statistic(x)` | Sufficient statistic $t(x)$ |
| `carrier_measure(x)` | Base measure $h(x)$ |
| `cross_entropy()` | Efficient cross-entropy (nat × exp) |
| `flattened()` / `unflattened()` | ML integration |

### PyTree / JIT / vmap / grad

**Yes** — all objects are JAX pytrees via `tjax.dataclass`. Vectorized operations are native (a single parametrization object can hold batches of distributions).

### Fitting / MLE

**Implicit via expectation parametrization.** The library's design philosophy: combining independent observations into an MLE is natural in expectation parametrization (just average the sufficient statistics). No explicit `fit()` method, but the mathematical machinery is there.

### Mixture Distributions

**No.** efax focuses on exponential families. No mixture classes.

### Exponential Family Structure

**This is efax's raison d'être.** Full support for:
- Natural parameters $\theta$
- Expectation parameters $\eta = \nabla\psi(\theta) = E[t(X)]$
- Sufficient statistics $t(x)$
- Log partition (log normalizer) $\psi(\theta)$
- Carrier measure $h(x)$
- Conversions between parametrizations

### Distributions Available

Normal, Gamma, Beta, Exponential, Poisson, Binomial, Bernoulli, Dirichlet, Generalized Dirichlet, Multivariate Normal, and others. **No GIG, no GH, no InverseGaussian, no InverseGamma.**

### Suitability for normix

**The closest match architecturally.** efax's dual-parametrization design mirrors normix's three-parametrization approach (classical/natural/expectation). However:

**Pros:**
- Exponential family native — natural params, sufficient stats, log partition
- PyTree-compatible, JIT/vmap/grad ready
- Clean base class design
- Active maintenance

**Cons:**
- Missing distributions: no GIG, GH, IG, InvGamma
- No mixture distributions
- No EM fitting infrastructure
- Depends on `tjax` (author's own JAX utility library)
- Small community (75 stars)
- No classical parametrization concept (only natural and expectation)
- Design philosophy differs: efax uses type-level encoding (distribution family = type), normix uses instance-level

**Verdict:** efax is worth studying for design ideas but is **not suitable as a base dependency** for normix. normix needs custom mixture distributions (VG, NIG, GH) and EM fitting that efax doesn't provide.

---

## 7. Distrax

**Repository:** [google-deepmind/distrax](https://github.com/google-deepmind/distrax)  
**Stars:** ~600 | **Last release:** v0.1.7 | **License:** Apache-2.0  
**Status:** Likely stale (development status: Beta, limited recent activity)

### Base Class Design

```python
class Distribution:
    def sample(seed, sample_shape=()) -> Array
    def sample_and_log_prob(seed, sample_shape=()) -> (Array, Array)
    def log_prob(value) -> Array
    def prob(value) -> Array
    def cdf(value) -> Array
    def log_cdf(value) -> Array
    def survival_function(value) -> Array
    def entropy() -> Array
    def mean() -> Array
    def variance() -> Array
    def mode() -> Array
    def kl_divergence(other) -> Array
    def event_shape -> tuple
    def batch_shape -> tuple
```

Cross-compatible with TFP API. Reimplements a subset of TFP with focus on readability and JAX-nativeness.

### PyTree / JIT / vmap / grad

**Yes** — distributions are registered as JAX pytrees. Full compatibility with JAX transforms.

### Distributions Available

Normal, MultivariateNormalDiag, MultivariateNormalFullCovariance, MultivariateNormalTri, Beta, Gamma, Laplace, Categorical, Bernoulli, Multinomial, Uniform, Logistic, and bijectors (for normalizing flows).

### Fitting / Mixtures / Exponential Family

**No fitting. No mixtures. No exponential family structure.**

### Suitability for normix

**Not recommended.** Maintenance appears stale. Provides less than TFP-JAX while adding a dependency. No exponential family structure.

---

## 8. Distreqx

**Repository:** [lockwo/distreqx](https://github.com/lockwo/distreqx)  
**Stars:** ~30 | **License:** Apache-2.0  
**Status:** Low activity

"Distrax, but in Equinox." Reimplements Distrax using `eqx.Module` instead of custom pytree registration. Distributions are Equinox modules, giving automatic PyTree compatibility.

### Suitability for normix

**Same limitations as Distrax** (no exp family, no fitting, no mixtures), but with a more modern PyTree approach (Equinox). The Equinox pattern (`eqx.Module`) is worth noting as a design choice for normix's JAX migration.

---

## 9. FlowJAX

**Repository:** [danielward27/flowjax](https://github.com/danielward27/flowjax)  
**Stars:** ~68 | **Last release:** v19.1.0 | **License:** MIT  
**Status:** Actively maintained

### Base Class Design

```python
class AbstractDistribution(eqx.Module):
    shape: tuple       # Shape of a single sample
    cond_shape: tuple   # Shape of conditioning variable (None if unconditional)

    @abstractmethod
    def _sample(self, key, condition=None) -> Array:  # Single unbatched sample
        ...
    
    @abstractmethod
    def _log_prob(self, x, condition=None) -> Scalar:  # Scalar log prob
        ...
    
    def sample(self, key, shape=()) -> Array:  # Vectorized via vmap
        ...
    
    def log_prob(self, x, condition=None) -> Array:  # Vectorized via vmap
        ...
```

Built on Equinox. First-class support for **conditional distributions** (important for normalizing flows / amortized inference).

### Suitability for normix

**Not applicable.** Focused on normalizing flows and bijective transforms. No exponential family structure, no EM, no classical distribution fitting.

---

## 10. GMMX

**Repository:** [adonath/gmmx](https://github.com/adonath/gmmx)  
**Stars:** ~24 | **Last release:** v0.7 (Jul 2025) | **License:** BSD-3  
**Status:** Actively maintained

### Design

```python
class GaussianMixtureModelJax:
    @classmethod
    def create(cls, n_components, n_features) -> Self
    
    def sample(self, n_samples, key) -> Array
    def log_prob(self, x) -> Array
    def predict_proba(self, x) -> Array  # Responsibilities
    
class EMFitter:
    def fit(self, x, gmm) -> GaussianMixtureModelJax
```

Uses `register_dataclass_jax` for PyTree compatibility. Internal 4D array representation for broadcasting. Multiple covariance types (full, tied, diag, spherical).

### Suitability for normix

**Interesting for EM design patterns.** GMMX demonstrates how to write a JAX-native EM fitter with functional style. However, normix's mixtures are variance-mean mixtures (continuous mixing), not finite Gaussian mixtures (discrete mixing). The EM pattern is still relevant — particularly the functional `EMFitter` returning a new model rather than mutating state.

---

## 11. TensorFlow Probability (JAX substrate)

**Repository:** [tensorflow/probability](https://github.com/tensorflow/probability)  
**Stars:** ~4,000+ | **License:** Apache-2.0  
**Status:** Actively maintained

### Usage

```python
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
```

TFP on JAX does **not** depend on TensorFlow. It provides the full TFP distribution API with JAX as backend.

### Base Class

```python
class Distribution:
    def sample(sample_shape, seed) -> Array
    def log_prob(value) -> Array
    def prob(value) -> Array
    def cdf(value) -> Array
    def entropy() -> Array
    def mean() -> Array
    def variance() -> Array
    def mode() -> Array
    def kl_divergence(other) -> Array
    # ... many more methods
```

### Distributions Available (relevant to normix)

| Distribution | Available? |
|-------------|:---:|
| Normal | ✓ |
| MultivariateNormalTriL | ✓ |
| MultivariateNormalFullCovariance | ✓ |
| Gamma | ✓ |
| InverseGamma | ✓ |
| InverseGaussian | ✓ |
| **GeneralizedInverseGaussian** | **No** |
| **GeneralizedHyperbolic** | **No** |
| NormalInverseGaussian | ✓ |
| Beta | ✓ |
| Exponential | ✓ |
| MixtureSameFamily | ✓ |
| Categorical | ✓ |
| GeneralizedNormal | ✓ (not GH) |

### Exponential Family

TFP has `tfp.glm.ExponentialFamily` for GLM fitting but it's a narrow interface (link function, log-likelihood, mean), **not** a general exponential family framework with natural/expectation params.

### Bessel Functions

TFP provides `tfp.math.bessel_kve(v, z)` — the exponentially-scaled modified Bessel function $K_\nu(z) \cdot e^{|z|}$. **Gradients w.r.t. order $\nu$ are NOT supported.** This is a critical limitation for GIG/GH where $\nu$ (the $p$ parameter) needs to be optimized.

### Suitability for normix

**Heavyweight dependency, partial coverage.** TFP-JAX provides `InverseGaussian` and `NormalInverseGaussian` but lacks GIG and GH. The MixtureSameFamily is for finite mixtures, not variance-mean mixtures. Using TFP would add a large dependency for limited benefit. However, `tfp.math.bessel_kve` could be useful as a building block.

---

## 12. Minor / Niche Libraries

### PJAX
- **Repository:** lawortsmann/pjax | **Stars:** minimal | **Last release:** v0.0.2 (Dec 2022)
- Minimal distribution library. `sample()` and `log_pdf()` only. Gamma, Normal, etc.
- **Dead project.** 12 downloads/month.

### Lévy-stable-jax
- **Repository:** [tjhunter/levy-stable-jax](https://github.com/tjhunter/levy-stable-jax) | **Last release:** v0.2.1 (May 2024)
- Implements Lévy alpha-stable distributions in JAX
- Niche: heavy-tailed distributions, limited alpha range (1.01–2)
- NumPyro/PyMC compatible
- **Not relevant to normix's distributions but interesting for heavy-tail modeling.**

### GenJAX
- **Repository:** probcomp/genjax | **Stars:** moderate | **Last release:** v0.10.3 (Mar 2025)
- Probabilistic programming language on JAX (Gen ecosystem)
- Distributions are generative functions with `sample` and `logpdf`
- Focus: programmable inference, not distribution mathematics
- **Not relevant for normix.**

### JAXNS
- **Repository:** [Joshuaalbert/jaxns](https://github.com/Joshuaalbert/jaxns) | **Last release:** v2.6.9
- Nested sampling in JAX. Uses TFP distributions for priors.
- **Not relevant for normix.**

### GPJax
- **Repository:** jaxgaussianprocesses/gpjax
- `GaussianDistribution` class for GP posterior predictions
- `sample()`, `log_prob()`, `entropy()`, `kl_divergence()`
- Cholesky-based sampling (reparameterization trick)
- **GP-specific, not general purpose.**

### GibbsGMM
- **Repository:** yongquan-qu/GibbsGMM
- Gibbs sampling for Gaussian mixture models in JAX
- **Niche alternative to EM for GMMs.**

---

## 13. Specific Distribution Availability

### JAX Generalized Inverse Gaussian (GIG)

**Not available in any JAX library.** Neither NumPyro, TFP-JAX, efax, Distrax, nor any other surveyed library implements GIG. This is a critical gap since GIG is the mixing distribution for the Generalized Hyperbolic family.

SciPy has `scipy.stats.geninvgauss` (NumPy-based, no JAX). normix would need to implement GIG from scratch in JAX.

### JAX Generalized Hyperbolic (GH)

**Not available in any JAX library.** SciPy has `scipy.stats.genhyperbolic` (NumPy-based). The GH distribution requires modified Bessel functions $K_\nu$ which JAX lacks natively (see Section 14).

### JAX Normal-Inverse Gaussian (NIG)

**Available in TFP-JAX only:** `tfp.substrates.jax.distributions.NormalInverseGaussian` with params `(loc, scale, tailweight, skewness)`. Uses the variance-mean mixture representation with `InverseGaussian` mixing.

### JAX Variance-Gamma (VG)

**Not directly available.** Would need custom implementation.

### JAX Normal Mixture Models (finite)

**Available in multiple libraries:**
- NumPyro: `MixtureSameFamily`, `MixtureGeneral`
- TFP-JAX: `MixtureSameFamily`
- GMMX: `GaussianMixtureModelJax` with EM fitting

Note: These are all *finite component* mixtures (discrete mixing). normix's mixtures are *continuous* variance-mean mixtures — a fundamentally different concept.

### JAX EM Algorithm

**Available in:**
- GMMX: `EMFitter` for Gaussian mixtures
- Dynamax: EM for HMMs and linear Gaussian SSMs

No general-purpose exponential family EM framework exists in JAX.

---

## 14. The Bessel Function Gap

The modified Bessel function of the second kind $K_\nu(z)$ is essential for GIG and GH distributions. The JAX ecosystem has significant gaps here:

| Source | Function | Grad w.r.t. $\nu$? | Grad w.r.t. $z$? | Notes |
|--------|----------|:---:|:---:|-------|
| `jax.scipy.special` | **No $K_\nu$** | — | — | Only $I_0, I_1$ available |
| `tfp.math.bessel_kve` | $K_\nu(z) \cdot e^{|z|}$ | **No** | Yes | Exponentially scaled |
| `tfp.math.bessel_ive` | $I_\nu(z) \cdot e^{-|z|}$ | No | Yes | First kind |
| normix custom | `log_kv` | Via finite diff | Via AD | Current NumPy impl |

**This is the single biggest technical blocker** for a JAX-native GIG/GH implementation. Options:

1. **Use `tfp.math.bessel_kve`** and work in log-space: $\log K_\nu(z) = \log(\text{kve}(\nu, z)) - |z|$. Limitation: no gradient w.r.t. $\nu$ (the GIG $p$ parameter).
2. **Custom JAX implementation** of $\log K_\nu$ with `jax.custom_jvp` for both $\nu$ and $z$ gradients. This is what normix would likely need.
3. **Series/asymptotic expansions** implemented directly in JAX (differentiable).
4. **Callback to SciPy** via `jax.pure_callback` — gives exact values but breaks JIT/grad.

---

## 15. Comparison Matrix

### Architecture Comparison

| Feature | normix (current) | efax | NumPyro | TFP-JAX | Distrax |
|---------|:-:|:-:|:-:|:-:|:-:|
| Backend | NumPy/SciPy | JAX | JAX | JAX | JAX |
| Natural params | ✓ | ✓ | ✗ | ✗ | ✗ |
| Expectation params | ✓ | ✓ | ✗ | ✗ | ✗ |
| Classical params | ✓ | ✗ | ✓ | ✓ | ✓ |
| Log partition $\psi(\theta)$ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Sufficient stats $t(x)$ | ✓ | ✓ | ✗ | ✗ | ✗ |
| EM fitting | ✓ | ✗ | ✗ | ✗ | ✗ |
| Variance-mean mixtures | ✓ | ✗ | ✗ | ✗ | ✗ |
| GIG / GH | ✓ | ✗ | ✗ | ✗ | ✗ |
| `jit` / `vmap` / `grad` | ✗ | ✓ | ✓ | ✓ | ✓ |
| PyTree distributions | ✗ | ✓ | ✓ | ✓ | ✓ |

### Distribution Coverage

| Distribution | normix | efax | NumPyro | TFP-JAX | Distrax | jax.scipy |
|-------------|:-:|:-:|:-:|:-:|:-:|:-:|
| Exponential | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Gamma | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| InverseGamma | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| InverseGaussian | ✓ | ✗ | ? | ✓ | ✗ | ✗ |
| GIG | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| MultivariateNormal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (pdf only) |
| NIG mixture | ✓ | ✗ | ✗ | ✓* | ✗ | ✗ |
| VG mixture | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| GH mixture | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |

*TFP's NIG uses different parametrization from normix's exponential family form.

---

## 16. Recommendations for normix

### 1. Build Custom — Don't Inherit

No existing JAX library provides normix's unique combination of:
- Three parametrizations (classical/natural/expectation)
- Variance-mean mixture distributions
- GIG/GH family
- EM fitting with exponential family structure

**normix should implement its own JAX distribution base classes** rather than inheriting from any existing library.

### 2. Study efax for Design Patterns

efax's approach to encoding exponential family structure is the most relevant:
- `NaturalParametrization` / `ExpectationParametrization` duality
- `log_normalizer()`, `sufficient_statistic()`, `carrier_measure()`
- PyTree-registered dataclasses

normix can adopt similar patterns while adding:
- Classical parametrization as a third path
- `fit()` / EM infrastructure
- Mixture distribution hierarchy

### 3. Use Equinox for PyTree Infrastructure

The Equinox pattern (`eqx.Module`) is emerging as the standard for JAX pytree classes:
- Used by FlowJAX, Distreqx, and growing ecosystem
- Cleaner than manual `tree_flatten`/`tree_unflatten` (NumPyro/TFP style)
- Frozen by default (fits normix's immutable parameter philosophy)
- Patrick Kidger's library — well-maintained (2.3k+ stars)

### 4. Consider `tfp.math.bessel_kve` as a Starting Point

For the Bessel function challenge, `tfp.math.bessel_kve` provides a differentiable (w.r.t. $z$) implementation. normix would need to:
- Wrap it with `jax.custom_jvp` to add $\nu$-gradient support
- Work in log-space: $\log K_\nu(z) = \log(\text{kve}(\nu, z)) - |z|$

### 5. Study GMMX for EM Pattern

GMMX's functional EM design is a good reference:
- `EMFitter.fit(x, model) -> model` (returns new model, doesn't mutate)
- Configurable tolerance and max iterations
- JAX-native with JIT compilation

### 6. Don't Take a Heavy Dependency

Avoid depending on NumPyro or TFP-JAX for distribution base classes. They bring in large dependency trees and their distribution APIs don't align with normix's exponential family focus. Use JAX + Equinox + (optionally) `tfp.math` for Bessel functions.

### Summary: Recommended Stack

```
JAX (core)
├── jax.numpy / jax.scipy.special          — numerical primitives
├── jax.scipy.stats.{norm,gamma,...}        — density functions as building blocks
├── Equinox (eqx.Module)                   — pytree base class infrastructure
├── tfp.math.bessel_kve (optional)         — Bessel function starting point
└── normix (custom)
    ├── Distribution base (exp family)     — 3 parametrizations, cached properties
    ├── JointNormalMixture base            — variance-mean mixture infrastructure
    ├── Individual distributions           — GIG, GH, VG, NIG, etc.
    └── EM fitting                         — functional style (EMFitter)
```
