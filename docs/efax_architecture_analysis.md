# EFAX Architecture Analysis

## Overview

[efax](https://github.com/NeilGirdhar/efax) (v1.23.0) is a production-stable Python library for working with **exponential family distributions** in JAX. Created by Neil Girdhar, it provides the two most important parametrizations—natural and expectation—with a uniform API for machine learning operations. Licensed under Apache 2.0, it requires Python 3.12+ and depends on JAX, tjax, numpy, scipy, optimistix, and tensorflow-probability (for Bessel functions).

**Key insight:** efax represents each parametrization of a distribution as a **separate class** (a JAX-compatible dataclass), not as a mode switch on a single class. A Gamma distribution has `GammaNP` (natural params), `GammaEP` (expectation params), and optionally `GammaVP` (variance params)—each is its own pytree-registered dataclass.

---

## 1. Overall Design Pattern: Parametrizations as Class Objects

Each exponential family distribution is implemented as **two or more dataclass types**, one per parametrization:

| Distribution | Natural Params (`NP`) | Expectation Params (`EP`) | Other Params |
|---|---|---|---|
| Normal | `NormalNP` | `NormalEP` | `NormalVP`, `NormalDP` |
| Gamma | `GammaNP` | `GammaEP` | `GammaVP` |
| Inverse Gamma | `InverseGammaNP` | `InverseGammaEP` | — |
| Inverse Gaussian | `InverseGaussianNP` | `InverseGaussianEP` | — |
| Multivariate Normal | `MultivariateNormalNP` | `MultivariateNormalEP` | `MultivariateNormalVP` |

Each class is a `@dataclass` (from `tjax.dataclasses`) whose fields are the parameters of that parametrization. The dataclass decorator registers the type as a JAX pytree, making instances compatible with `jit`, `grad`, `vmap`, etc.

```python
@dataclass
class GammaNP(HasEntropyNP['GammaEP'],
              Samplable,
              NaturalParametrization['GammaEP', JaxRealArray],
              SimpleDistribution):
    negative_rate: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))
    shape_minus_one: JaxRealArray = distribution_parameter(ScalarSupport(ring=RealField(minimum=-1.0)))
```

**Key differences from normix:**
- normix uses a single class per distribution with internal `_set_from_classical` / `_set_from_natural` methods and `cached_property` for lazy conversion
- efax uses separate class per parametrization with explicit `to_exp()` / `to_nat()` conversion methods that return new objects
- efax objects are immutable (frozen dataclasses); normix objects are mutable with cache invalidation

---

## 2. Class Hierarchy and Key Base Classes

### Core hierarchy (simplified):

```
Distribution (abstract base)
├── SimpleDistribution (no sub-distributions, has domain_support)
│   ├── NaturalParametrization[EP, Domain] (log_normalizer, to_exp, carrier_measure, sufficient_statistics, pdf, log_pdf, fisher_information, kl_divergence)
│   │   ├── HasEntropyNP[EP] (entropy via to_exp().cross_entropy(self))
│   │   ├── TransformedNaturalParametrization[NP, EP, TEP, Domain] (derived from base distribution)
│   │   └── concrete NP classes (GammaNP, NormalNP, etc.)
│   │
│   ├── ExpectationParametrization[NP] (to_nat, kl_divergence)
│   │   ├── HasEntropyEP[NP] (cross_entropy, entropy, expected_carrier_measure)
│   │   │   ├── HasConjugatePrior (conjugate_prior_distribution)
│   │   │   └── concrete EP classes (GammaEP, NormalEP, etc.)
│   │   ├── ExpToNat[NP] (numerical to_nat via Newton's method)
│   │   ├── TransformedExpectationParametrization[EP, NP, TNP]
│   │   └── concrete EP classes
│   │
│   ├── Samplable (sample method)
│   ├── Multidimensional (dimensions method)
│   └── "VP"/"DP" parametrization classes (NormalVP, GammaVP, etc.)
│
└── JointDistribution (container for sub-distributions)
    ├── JointDistributionN (natural params joint)
    └── JointDistributionE (expectation params joint)
```

### Key base classes:

1. **`Distribution`** — Abstract base. Has `shape`, `sub_distributions()`, `__getitem__` for slicing, and `__array_namespace__` for array API compatibility.

2. **`SimpleDistribution`** — A distribution with no sub-distributions. Has `domain_support()` classmethod.

3. **`NaturalParametrization[EP, Domain]`** — Generic over its corresponding EP type and the domain type. Defines the core exponential family interface.

4. **`ExpectationParametrization[NP]`** — Generic over its corresponding NP type. Implements `kl_divergence`.

5. **`JointDistribution`** — Wraps a dict of sub-distributions. `JointDistributionN` and `JointDistributionE` are typed versions.

### Mixin classes:

- **`HasEntropyNP`** / **`HasEntropyEP`** — Entropy and cross-entropy computations
- **`Samplable`** — `sample(key, shape)` method
- **`Multidimensional`** — `dimensions()` method for vector-valued distributions
- **`HasConjugatePrior`** — Conjugate prior support
- **`ExpToNat`** — Numerical expectation-to-natural conversion (Newton's method via optimistix)
- **`TransformedNaturalParametrization`** / **`TransformedExpectationParametrization`** — Derive distributions via transformations of existing ones (e.g., InverseGamma from Gamma)

---

## 3. How Parameters Are Handled

### Parameter declaration

Fields are declared using `distribution_parameter(support, fixed=False, static=False)`, which creates a `tjax.dataclasses.field` with metadata:

```python
negative_rate: JaxRealArray = distribution_parameter(ScalarSupport(ring=negative_support))
```

The metadata records:
- **`support`**: a `Support` object (ScalarSupport, VectorSupport, SymmetricMatrixSupport, etc.)
- **`fixed`**: whether the parameter is fixed w.r.t. the exponential family (e.g., failure count in negative binomial)
- **`parameter`**: always `True` (marker flag)

### Support system

The `Support` hierarchy describes the shape and domain of each parameter:

```
Support
├── ScalarSupport        — shape ()
├── VectorSupport        — shape (n,)
├── SimplexSupport       — shape (n-1,) for probability simplex
├── SymmetricMatrixSupport — shape (n, n) symmetric/Hermitian
├── SquareMatrixSupport  — shape (n, n) general
└── CircularBoundedSupport — shape (n,) bounded by radius
```

Each `Support` has a `Ring` that specifies the algebraic structure and domain constraints:

```
Ring
├── RealField(minimum=None, maximum=None)   — constrained reals
├── ComplexField(minimum_modulus, maximum_modulus)
├── BooleanRing
└── IntegralRing(minimum, maximum)
```

Pre-defined instances: `real_field`, `positive_support = RealField(minimum=0.0)`, `negative_support = RealField(maximum=0.0)`.

The Ring handles **mapping to/from the full plane** (unconstrained reals) via softplus, logit, etc. This is critical for optimization and neural network outputs.

### Natural parameters

Each `NaturalParametrization` stores the natural parameters directly as its dataclass fields. The field names describe the mathematical meaning:

| Distribution | Natural Parameter Fields |
|---|---|
| Normal | `mean_times_precision`, `negative_half_precision` |
| Gamma | `negative_rate`, `shape_minus_one` |
| Exponential | `negative_rate` |
| Inverse Gaussian | `negative_lambda_over_two_mu_squared`, `negative_lambda_over_two` |
| MultivariateNormal | `mean_times_precision` (vector), `negative_half_precision` (symmetric matrix) |

### Expectation parameters

Each `ExpectationParametrization` stores the expectation parameters (= $E[t(X)]$, the expected sufficient statistics):

| Distribution | Expectation Parameter Fields |
|---|---|
| Normal | `mean`, `second_moment` |
| Gamma | `mean`, `mean_log` |
| Exponential | `mean` |
| Inverse Gaussian | `mean`, `mean_reciprocal` |
| MultivariateNormal | `mean` (vector), `second_moment` (symmetric matrix) |

### Additional parametrizations

Some distributions have convenience parametrizations (not part of the exponential family formalism):

- `NormalVP` (mean, variance), `NormalDP` (mean, deviation)
- `GammaVP` (mean, variance)
- `MultivariateNormalVP` (mean, variance matrix)

These are plain `SimpleDistribution` subclasses with `to_nat()` / `to_exp()` methods but **not** subclasses of `NaturalParametrization` or `ExpectationParametrization`.

---

## 4. Conversions Between Parametrizations

### Natural → Expectation: `NP.to_exp() -> EP`

Each NP class implements `to_exp()` analytically. This is the **gradient of the log-normalizer** $\nabla\psi(\theta)$:

```python
# GammaNP.to_exp()
def to_exp(self) -> GammaEP:
    shape = self.shape_minus_one + 1.0
    return GammaEP(-shape / self.negative_rate,              # mean = E[X]
                   jss.digamma(shape) - xp.log(-self.negative_rate))  # mean_log = E[log X]
```

### Expectation → Natural: `EP.to_nat() -> NP`

This is the **inverse** of the gradient of the log-normalizer. Two strategies:

1. **Analytical** (when a closed-form inverse exists):
```python
# NormalEP.to_nat()
def to_nat(self) -> NormalNP:
    return NormalNP(self.mean / self.variance(), -0.5 / self.variance())
```

2. **Numerical via `ExpToNat` mixin** (when no closed form exists, e.g., Gamma, Dirichlet):

The `ExpToNat` mixin uses **Newton's method** (via the `optimistix` library) to invert $\nabla\psi(\theta) = \eta$ by finding $\theta$ such that $\nabla\psi(\theta) = \eta_{\text{observed}}$. It does this by:
- Flattening the EP into a real vector
- Running a root-finding algorithm in "search parameter" space
- The gradient is: `search_flat(nabla_psi(theta)) - self_flat` (difference between predicted and observed expectations)
- Converting the solution back to an NP object

```python
# GammaEP inherits from ExpToNat[GammaNP]
class GammaEP(HasEntropyEP[GammaNP], Samplable, ExpToNat[GammaNP], SimpleDistribution):
    def search_to_natural(self, search_parameters):
        shape = softplus(search_parameters[..., 0])
        rate = shape / self.mean
        return GammaNP(-rate, shape - 1.0)
    
    def search_gradient(self, search_parameters):
        shape = softplus(search_parameters[..., 0])
        return (log_mean_minus_mean_log - xp.log(shape) + jss.digamma(shape))[..., xp.newaxis]
```

### Transformed distributions

The `TransformedNaturalParametrization` / `TransformedExpectationParametrization` mixins allow defining a distribution by relating it to an existing one via a transformation:

```python
# InverseGamma is defined as a transformation of Gamma (x → 1/x)
class InverseGammaNP(TransformedNaturalParametrization[GammaNP, GammaEP, 'InverseGammaEP', JaxRealArray]):
    def base_distribution(self) -> GammaNP:
        return GammaNP(self.negative_scale, self.shape_minus_one)
    
    @classmethod
    def sample_to_base_sample(cls, x, **fixed_parameters):
        return xp.reciprocal(x)
```

The base class automatically derives `log_normalizer`, `to_exp`, `carrier_measure`, and `sufficient_statistics` from the base distribution and the sample transformation. The carrier measure includes the Jacobian correction $\log|\det J|$ computed via `jax.jacobian`.

### Convenience parametrization conversions

VP/DP classes have `to_nat()` / `to_exp()` methods that chain through one another:

```
NormalDP → NormalVP → NormalNP → NormalEP
```

---

## 5. Gradients and Hessians of Log Partition Functions

### Log-normalizer (log partition function)

Each NP class implements `log_normalizer()`:

```python
# GammaNP
def log_normalizer(self) -> JaxRealArray:
    shape = self.shape_minus_one + 1.0
    return jss.gammaln(shape) - shape * xp.log(-self.negative_rate)
```

### Custom JVP for numerical stability

The `log_normalizer` method has a **custom JVP rule** `_log_normalizer_jvp` registered via `@abstract_custom_jvp`:

```python
def _log_normalizer_jvp(primals, tangents):
    q, = primals
    q_dot, = tangents
    y = q.log_normalizer()
    p = q.to_exp()  # gradient of log_normalizer = expectation parameters
    y_dot = parameter_dot_product(q_dot, p)
    return y, y_dot
```

This exploits the fundamental identity: **the gradient of the log-normalizer equals the expectation parameters**:

$$\nabla_\theta \psi(\theta) = E[t(X)] = \eta$$

By using the analytically-computed `to_exp()` for the JVP instead of automatic differentiation, this provides vastly improved numerical stability.

### Fisher information matrix

The Fisher information is the **Hessian of the log-normalizer**:

$$\mathcal{I}(\theta) = \nabla^2 \psi(\theta) = \text{Cov}[t(X)]$$

efax computes this using `jax.jacfwd(jax.grad(...))` on the flattened log-normalizer:

```python
def _fisher_information_matrix(self) -> JaxRealArray:
    flattener, flattened = Flattener.flatten(self, map_to_plane=False)
    fisher_info_f = jacfwd(grad(self._flat_log_normalizer))
    for _ in range(len(self.shape)):
        fisher_info_f = vmap(fisher_info_f)
    return fisher_info_f(flattened, flattener)
```

This leverages JAX's AD to compute the full Hessian without implementing it manually for each distribution.

### Apply Fisher information efficiently

The `apply_fisher_information` method uses `jax.vjp` to efficiently multiply the Fisher information matrix by a vector without materializing the full matrix:

```python
def apply_fisher_information(self, vector: EP) -> tuple[EP, Self]:
    expectation_parameters, f_vjp = vjp(type(self).to_exp, self)
    return expectation_parameters, f_vjp(vector)
```

---

## 6. Distributions Implemented

### Positive reals (relevant to normix):

| Distribution | NP Class | EP Class | Notes |
|---|---|---|---|
| **Gamma** | `GammaNP` | `GammaEP` | Direct NP; EP uses `ExpToNat` (Newton's method) |
| **Inverse Gamma** | `InverseGammaNP` | `InverseGammaEP` | Via `TransformedNaturalParametrization` from Gamma |
| **Inverse Gaussian** | `InverseGaussianNP` | `InverseGaussianEP` | Direct NP; EP has analytical `to_nat()` |
| **Exponential** | `ExponentialNP` | `ExponentialEP` | Both analytical; has conjugate prior (Gamma) |

### Normal family:

| Distribution | NP Class | EP Class | Notes |
|---|---|---|---|
| **Normal** | `NormalNP` | `NormalEP` | Also `NormalVP`, `NormalDP` |
| **Unit Variance Normal** | `UnitVarianceNormalNP` | `UnitVarianceNormalEP` | — |
| **Multivariate Normal** | `MultivariateNormalNP` | `MultivariateNormalEP` | Also `MultivariateNormalVP` |
| **Diagonal MV Normal** | `MultivariateDiagonalNormalNP` | `MultivariateDiagonalNormalEP` | Also `MultivariateDiagonalNormalVP` |
| **Isotropic MV Normal** | `IsotropicNormalNP` | `IsotropicNormalEP` | — |
| **Fixed Variance MV Normal** | `MultivariateFixedVarianceNormalNP` | `MultivariateFixedVarianceNormalEP` | — |
| **Complex Normal** | `ComplexNormalNP` | `ComplexNormalEP` | — |
| **Log-Normal** | `LogNormalNP` | `LogNormalEP` | Via transformation of Normal |
| **Softplus Normal** | `SoftplusNormalNP` | `SoftplusNormalEP` | Via transformation of Normal |

### Other distributions:

Bernoulli, Beta, Chi, Chi-Square, Dirichlet (standard + generalized), Geometric, Logarithmic, Multinomial, Negative Binomial, Poisson, Rayleigh, Von Mises-Fisher, Weibull.

### Not implemented (relevant to normix):

- **Generalized Inverse Gaussian (GIG)** — not present in efax
- **Variance Gamma, Normal-Inverse Gaussian, Normal-Inverse Gamma, Generalized Hyperbolic** — not present (these are mixture distributions, not pure exponential families in the usual sense)

---

## 7. How efax Leverages JAX Features

### PyTree registration via `tjax.dataclasses.dataclass`

All distribution classes use `@dataclass` from `tjax.dataclasses`, which registers them as JAX pytrees. This means instances can be:
- Passed through `jit`-compiled functions
- Differentiated with `grad` / `jacfwd` / `jacrev`
- Vectorized with `vmap`
- Used in `lax.while_loop`, `lax.scan`, etc.

Fields marked `static=True` are treated as static (not traced through JAX transforms).

### `jit` and `abstract_jit`

Methods are decorated with `@jit` (from tjax) for JIT compilation. Abstract methods use `@abstract_jit` to ensure JIT applies to concrete implementations:

```python
@abstract_jit
@abstractmethod
def log_normalizer(self) -> JaxRealArray:
    raise NotImplementedError
```

### `grad` for automatic differentiation

Used throughout:
- Fisher information: `jacfwd(grad(log_normalizer))`
- Optimization: `grad(cross_entropy_loss)` in examples
- Custom JVP: `_log_normalizer_jvp` uses the analytical gradient for stability

### `vmap` for vectorization

Applied programmatically for batched operations:

```python
for _ in range(len(self.shape)):
    fisher_info_f = vmap(fisher_info_f)
```

This handles arbitrary batch dimensions automatically.

### `vjp` for efficient Fisher-vector products

```python
expectation_parameters, f_vjp = vjp(type(self).to_exp, self)
return expectation_parameters, f_vjp(vector)
```

### Custom JVP rules (`abstract_custom_jvp`)

The log-normalizer has a custom JVP that uses the analytical gradient ($\nabla\psi = \eta$) instead of AD for better numerical stability:

```python
@abstract_custom_jvp(_log_normalizer_jvp)
@abstract_jit
@abstractmethod
def log_normalizer(self) -> JaxRealArray:
    ...
```

### `jacobian` for transformed distributions

The carrier measure of transformed distributions uses `jax.jacobian` to compute the Jacobian correction:

```python
jac_y = jacobian(bound_fy)(x)
log_abs_det_jac = xp.log(xp.abs(xp.linalg.det(jac_y)))
```

---

## 8. Overall API Design Pattern

### Creation: construct with arrays

```python
q = GammaNP(negative_rate=jnp.array([-2.0]), shape_minus_one=jnp.array([1.5]))
p = BernoulliEP(jnp.array([0.4, 0.5, 0.6]))
```

### Conversion: explicit methods returning new objects

```python
ep = q.to_exp()       # NP → EP
np = ep.to_nat()      # EP → NP
vp = np.to_variance_parametrization()  # convenience
```

### Information-theoretic operations

```python
ep.cross_entropy(np)   # H(ep, np) — efficient with ep in EP, np in NP
ep.entropy()           # H(ep) = cross_entropy(ep, stop_gradient(ep.to_nat()))
np.kl_divergence(np2)  # KL(np || np2)
```

### Density evaluation

```python
np.log_pdf(x)  # log p(x|θ) = θ·t(x) - ψ(θ) + h(x)
np.pdf(x)      # exp(log_pdf(x))
```

### Maximum likelihood estimation

```python
estimator = MaximumLikelihoodEstimator.create_simple_estimator(GammaEP)
ss = estimator.sufficient_statistics(samples)    # returns EP
ss_mean = parameter_mean(ss, axis=0)             # average sufficient statistics
estimated_np = ss_mean.to_nat()                  # convert to natural params
```

### Bayesian evidence combination (natural parameter addition)

```python
posterior_np = parameter_map(add, prior_np, likelihood_np)
```

### Fisher information

```python
np.fisher_information_diagonal()   # diagonal of Fisher matrix
np.fisher_information_trace()      # trace
np.apply_fisher_information(ep)    # Fisher matrix × vector (efficient via vjp)
np.jeffreys_prior()                # sqrt(det(Fisher))
```

### Flattening/unflattening (for neural network integration)

```python
flattener, flat_array = Flattener.flatten(distribution, map_to_plane=True)
distribution_back = flattener.unflatten(flat_array)
```

The `map_to_plane` option maps constrained parameters to unconstrained reals (via softplus, logit, etc.) for use as neural network outputs.

### Sampling

```python
samples = distribution.sample(jr.key(42), shape=(1000,))
```

---

## 9. Key Design Comparisons: efax vs normix

| Aspect | efax | normix |
|---|---|---|
| **Parametrization** | Separate class per parametrization | Single class with mode switching |
| **Framework** | JAX (functional, immutable) | NumPy/SciPy (imperative, mutable) |
| **Mutability** | Frozen dataclasses (immutable pytrees) | Mutable with cache invalidation |
| **Storage** | Natural params in NP class, expectation params in EP class | Named attributes as single source of truth |
| **Conversion** | `to_exp()` / `to_nat()` return new objects | `_set_from_classical` / `_set_from_natural` mutate in place |
| **Caching** | No caching (immutable; recompute or store externally) | `functools.cached_property` with invalidation |
| **Fitting** | MLE via `sufficient_statistics` + `parameter_mean` | `fit()` returns self (sklearn-style) |
| **Differentiation** | JAX `grad` / `jacfwd` (automatic) | Analytical implementations |
| **Vectorization** | JAX `vmap` (automatic) | NumPy broadcasting |
| **Hessian** | `jacfwd(grad(...))` automatic | Not implemented |
| **Fisher info** | Automatic via AD | Not implemented |
| **Custom JVP** | `abstract_custom_jvp` on log-normalizer | N/A |
| **GIG/GH** | Not implemented | Fully implemented |
| **EM algorithm** | Not implemented | Core feature |
| **Mixture distributions** | `JointDistribution` for product distributions | Full mixture hierarchy (VG, NIG, NInvG, GH) |

---

## 10. Ideas Potentially Applicable to normix

1. **Custom JVP for log partition functions** — The identity $\nabla_\theta\psi(\theta) = E[t(X)]$ could be used in JAX-based implementations for numerical stability.

2. **Separate parametrization types** — While normix uses a single-class approach (which is better for mutable EM), the type-safety of separate classes is worth noting.

3. **Support/Ring system for parameter constraints** — The mapping-to-plane approach (softplus for positive, logit for bounded) could be useful for constrained optimization.

4. **Flattener for neural network integration** — The concept of flattening/unflattening distribution parameters (with optional plane mapping) is a clean pattern for integrating distributions with neural networks.

5. **Transformed distributions** — The `TransformedNaturalParametrization` pattern (InverseGamma from Gamma) with automatic Jacobian correction is elegant and could reduce code duplication.

6. **Fisher information via AD** — Using `jacfwd(grad(log_normalizer))` to compute Fisher information automatically for any exponential family is powerful.

7. **Cross-entropy formula** — The efficient cross-entropy `H(p, q) = -⟨θ_q, η_p⟩ + ψ(θ_q) - E_p[h(X)]` requires p in expectation form and q in natural form.

8. **Numerical ExpToNat** — The Newton's method approach for inverting $\nabla\psi$ when no closed form exists (using optimistix) is a well-engineered fallback.
