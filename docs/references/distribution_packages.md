# JAX Distribution Packages: Survey and Design Recommendations

**Date:** 2026-03-06

---

## 1. Package Survey

### 1.1 TFP JAX (`tensorflow_probability.substrates.jax`)

**Status:** Active, v0.25.0, backed by Google  
**Stars:** ~4,100 (main TFP repo)

The most comprehensive JAX distribution library. Distributions are pytree-compatible classes with `log_prob`, `sample`, `mean`, `variance`, `entropy`, `kl_divergence`, `cdf`, etc.

**Distributions relevant to normix:** `Normal`, `MultivariateNormalTriL`, `Gamma`, `InverseGamma`, `InverseGaussian`, `GeneralizedInverseGaussian` (exists but limited), `NormalInverseGaussian`.

**Architecture:**
- Base class `tfd.Distribution` — heavyweight, ~200 public methods
- Parameters stored as tensor attributes, lazy validation
- Pytree-compatible with JAX substrate
- `JointDistributionCoroutine` for hierarchical models
- `MixtureSameFamily` for mixtures

**Pros:**
- Most distributions already implemented
- Has `bessel_kve` / `log_bessel_kve` in `tfp.math`
- KL divergence registry
- Heavily tested

**Cons:**
- Very large dependency (~50MB)
- No exponential family abstraction (no natural params, sufficient stats, log partition)
- No EM algorithm
- No gradient w.r.t. Bessel order $\nu$
- API is TensorFlow-flavored, not JAX-idiomatic
- `vmap` support is inconsistent

---

### 1.2 Distrax (`google-deepmind/distrax`)

**Status:** Maintenance mode, v0.1.7, last commit 2026-01-23  
**Stars:** ~626

Lightweight DeepMind distribution library. Clean, focused API.

**Architecture:**
- `Jittable` base — auto-registers every subclass as a JAX pytree via `__new__`
- `Distribution(Jittable)` — abstract, requires `_sample_n`, `log_prob`, `event_shape`
- `Bijector(Jittable)` — for transformed distributions
- MVN implemented as `Transformed(std_normal, Affine)` — elegant but indirect

**Distributions:** `Normal`, `Gamma`, `Beta`, `Laplace`, `MultivariateNormalTri/Diag/Full`, `Categorical`, `Bernoulli`, `Dirichlet`, `MixtureSameFamily`, `MixtureOfTwo`, `Transformed`.

**Missing:** No `InverseGamma`, `InverseGaussian`, `GIG`, or any GH-family. No exponential family structure.

**Pros:**
- Clean, minimal API (3 abstract methods)
- Pytree registration is automatic and robust
- Bidirectional TFP bridge (`distribution_from_tfp`, `to_tfp`)
- `MixtureSameFamily` uses efficient sample-all-then-mask strategy

**Cons:**
- Hard dependency on TFP (for KL registry)
- `vmap` support is "experimental and incomplete" (official warning)
- No fitting methods
- Maintenance mode — no new features being added

---

### 1.3 FlowJAX (`danielward27/flowjax`)

**Status:** Actively maintained, v19.1.0, last commit 2026-02-21  
**Stars:** ~219

Equinox-based normalizing flow library with a clean distribution abstraction.

**Architecture:**
- `AbstractDistribution(eqx.Module)` — base class
- Abstract methods: `_log_prob(x)` and `_sample(key)` (unbatched, single-sample)
- Public methods `log_prob` and `sample` auto-vectorize via `jnp.vectorize`
- Most distributions are `Transformed(StandardX, Affine)` — same pattern as distrax
- `VmapMixture` for mixture distributions

**Key features:**
- **`paramax.Parameterize`** — wraps constrained parameters (e.g., `softplus` for scale). Stores unconstrained value; applies transform on access. Enables gradient-based optimization over unconstrained space.
- **`paramax.NonTrainable`** — marks parameters as frozen during training
- **`fit_to_data(key, dist, data)`** — gradient-based fitting with Adam, early stopping, train/val split
- Full normalizing flow support (MAF, coupling, spline, planar, BNAF)

**Distributions:** `Normal`, `MultivariateNormal` (Cholesky), `Gamma`, `Beta`, `StudentT`, `Laplace`, `Logistic`, `Cauchy`, `Exponential`, `Uniform`, `LogNormal`, `Gumbel`, `VmapMixture`.

**Missing:** No `InverseGamma`, `InverseGaussian`, `GIG`, GH-family. No exponential family. No EM.

**Pros:**
- Built on Equinox — cleanest JAX-native design
- Actively maintained with frequent releases
- `fit_to_data` provides a training loop out of the box
- `Parameterize` pattern for constrained parameters is very useful
- `_log_prob` and `_sample` are unbatched — vectorization is automatic
- MIT license

**Cons:**
- Focused on normalizing flows, not classical statistics
- No exponential family structure

---

### 1.4 NumPyro (`pyro-ppl/numpyro`)

**Status:** Active, ~2,600 stars  

Probabilistic programming language on JAX (Pyro port).

**Architecture:**
- `numpyro.distributions.Distribution` — follows PyTorch's distribution API
- `arg_constraints` dict declares parameter constraints
- `support` property for distribution support
- `log_prob`, `sample`, `mean`, `variance`, `cdf`, `icdf`, `entropy`
- Pytree-compatible
- `MixtureSameFamily`, `MixtureGeneral` for mixtures

**Distributions:** ~80+ distributions including `GaussianHMM`, `InverseGamma`, but no `GIG` or `GeneralizedHyperbolic`.

**Pros:**
- Huge distribution library
- Mature, well-tested
- MCMC/SVI fitting (Bayesian)

**Cons:**
- No exponential family abstraction
- Fitting is Bayesian (MCMC/SVI), not MLE/EM
- Heavy dependency (entire PPL stack)

---

### 1.5 efax (`NeilGirdhar/efax`)

**Status:** Niche, ~75 stars  

The only JAX library with an explicit exponential family abstraction.

**Architecture:**
- Separate class per parametrization: `GammaNP`, `GammaEP`
- `NaturalParametrization` base: `log_normalizer()`, `sufficient_statistic()`, `carrier_measure()`
- `ExpectationParametrization` base: `to_nat()` conversion
- Custom JVP on `log_normalizer` using `to_exp()` as gradient
- Fisher information via `jacfwd(grad(log_normalizer))`

**Distributions:** `Gamma`, `InverseGamma` (via transform), `Normal`, `MultivariateNormal`, `InverseGaussian`, `Exponential`, `Beta`, `Poisson`, `Bernoulli`.

**Missing:** No `GIG`, no `GH`, no mixtures, no EM, no fitting.

**Pros:**
- Only library with explicit exponential family form
- Custom JVP for log partition gradient is exactly what normix needs
- Automatic Fisher information

**Cons:**
- Class-per-parametrization triples the number of classes
- Missing the distributions normix cares about (GIG, GH)
- Small, single-maintainer project
- No fitting/EM

---

### 1.6 Other Packages

| Package | Focus | Relevant? |
|---|---|---|
| **GMMX** (`adonath/gmmx`, 24 stars) | JAX Gaussian Mixture EM | Good EM design reference |
| **BlackJAX** (~900 stars) | MCMC sampling kernels | No distribution abstraction |
| **Dynamax** (~810 stars) | State-space models | Has EM for HMMs, not general |
| **jax.scipy.stats** | Stateless function modules | `logpdf`/`pdf`/`cdf`/`ppf` — no objects |

---

## 2. Comparison Matrix

| Feature | TFP | Distrax | FlowJAX | NumPyro | efax | normix needs |
|---|---|---|---|---|---|---|
| **Pytree** | Yes | Yes (auto) | Yes (eqx) | Yes | Yes | **Yes** |
| **Immutable** | Mostly | Yes | Yes | Yes | Yes | **Yes** (for JAX) |
| **`log_prob`** | Yes | Yes | Yes | Yes | Yes | **Yes** |
| **`sample`** | Yes | Yes | Yes | Yes | Partial | **Yes** |
| **`fit()`** | No | No | Yes (grad) | SVI/MCMC | No | **Yes (EM)** |
| **Exp. family** | No | No | No | No | **Yes** | **Yes** |
| **Natural params** | No | No | No | No | **Yes** | **Yes** |
| **Log partition** | No | No | No | No | **Yes** | **Yes** |
| **Fisher info** | No | No | No | No | **Yes** | **Yes** |
| **Mixtures** | Yes | Yes | Yes | Yes | No | **Yes** |
| **GIG** | Partial | No | No | No | No | **Yes** |
| **GH family** | No | No | No | No | No | **Yes** |
| **EM algorithm** | No | No | No | No | No | **Yes** |
| **Active** | Yes | Maint. | **Yes** | Yes | Niche | — |
| **Lightweight** | No | Yes | Yes | No | Yes | Preferred |

**Key observation:** No single package covers normix's needs. The GIG/GH family with EM fitting is unique to normix.

---

## 3. Design Analysis

### 3.1 Option A: Build on Distrax

Inherit from `distrax.Distribution` and extend with exponential family methods.

```python
class ExponentialFamily(distrax.Distribution):
    def natural_params(self) -> jax.Array: ...
    def log_partition(self) -> jax.Array: ...
    def sufficient_statistics(self, x) -> jax.Array: ...
    def expectation_params(self) -> jax.Array: ...
    def fisher_information(self) -> jax.Array: ...
```

**Pros:**
- Get `log_prob`, `sample`, `mean`, `variance`, pytree registration for free
- TFP bridge lets you wrap TFP distributions as distrax objects
- `MixtureSameFamily` available

**Cons:**
- **Hard TFP dependency** — distrax requires TFP even if you only use basic distributions
- **`vmap` is broken** — distrax officially warns against it. This blocks `vmap`-based multi-start EM.
- **Maintenance mode** — no new features, only JAX compat fixes
- The `Jittable` pytree registration is fragile (metadata classification issues)
- No `fit()` infrastructure

**Verdict:** **Not recommended.** The `vmap` limitation and TFP dependency are dealbreakers.

---

### 3.2 Option B: Build on FlowJAX's `AbstractDistribution`

Inherit from or follow `flowjax.distributions.AbstractDistribution`.

```python
class ExponentialFamily(flowjax.distributions.AbstractDistribution):
    # flowjax requires: shape, _log_prob, _sample
    # normix adds: natural_params, log_partition, etc.
```

**Pros:**
- Built on Equinox — the most JAX-native design
- `_log_prob` and `_sample` are unbatched → auto-vectorized
- `fit_to_data` provides a training loop
- `Parameterize` for constrained parameters (e.g., ensuring $a > 0$, $b > 0$ in GIG)
- Actively maintained, MIT license
- `VmapMixture` for mixture components

**Cons:**
- FlowJAX is focused on normalizing flows — the base distribution abstraction is simple but the ecosystem around it is flow-oriented
- Adding exponential family methods to `AbstractDistribution` requires understanding its internal vectorization (`jnp.vectorize` with ufunc signatures)
- The `Parameterize`/`unwrap` pattern adds complexity to the parameter storage model
- FlowJAX takes `paramax` as a dependency (parameter management library)

**Verdict:** **Possible but not ideal.** The base class is clean, but inheriting brings in flow-focused assumptions and dependencies.

---

### 3.3 Option C: Build on Equinox directly (recommended)

Use `eqx.Module` as the base class. Study FlowJAX's patterns but don't depend on FlowJAX.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from abc import abstractmethod

class Distribution(eqx.Module):
    """Base distribution class. Immutable JAX pytree."""

    @abstractmethod
    def log_prob(self, x: jax.Array) -> jax.Array:
        ...

    @abstractmethod
    def sample(self, key: jax.Array, sample_shape: tuple = ()) -> jax.Array:
        ...

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        ...


class ExponentialFamily(Distribution):
    """Exponential family distribution with three parametrizations."""

    @abstractmethod
    def natural_params(self) -> jax.Array:
        ...

    @abstractmethod
    def log_partition(self) -> jax.Array:
        """Log partition function ψ(θ) at current natural parameters."""
        ...

    @abstractmethod
    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        ...

    @abstractmethod
    def log_base_measure(self, x: jax.Array) -> jax.Array:
        ...

    def expectation_params(self) -> jax.Array:
        """η = ∇ψ(θ) via jax.grad — automatic."""
        theta = self.natural_params()
        return jax.grad(self._log_partition_from_theta)(theta)

    def fisher_information(self) -> jax.Array:
        """I(θ) = ∇²ψ(θ) via jax.hessian — automatic."""
        theta = self.natural_params()
        return jax.hessian(self._log_partition_from_theta)(theta)

    def log_prob(self, x):
        theta = self.natural_params()
        return (self.log_base_measure(x)
                + jnp.dot(self.sufficient_statistics(x), theta)
                - self.log_partition())

    @classmethod
    def from_natural(cls, theta: jax.Array) -> 'ExponentialFamily':
        ...

    @classmethod
    def from_expectation(cls, eta: jax.Array) -> 'ExponentialFamily':
        ...

    @classmethod
    def fit(cls, X: jax.Array) -> 'ExponentialFamily':
        """MLE: η̂ = mean(t(X)), then convert to natural."""
        ...
```

**Pros:**
- **Minimal dependencies**: only `jax` + `equinox` (both are standard in the JAX ecosystem)
- **Full control**: no constraints from upstream library design
- **Pytree-native**: `eqx.Module` gives automatic pytree registration
- **Immutable**: Equinox modules are frozen by default — no cache invalidation needed
- **`vmap`-safe**: Equinox modules work correctly with `vmap` (unlike distrax)
- **Adopt best patterns** from FlowJAX (unbatched `_log_prob`/`_sample`, `Parameterize`), distrax (auto-pytree), and efax (`custom_jvp` on log partition) without depending on any of them
- **EM fits naturally**: the EM algorithm is a pure function `(model, data) → new_model` over immutable pytrees

**Cons:**
- More upfront work — must implement `log_prob`, `sample`, etc. from scratch
- No built-in KL divergence registry (can implement manually for needed pairs)
- No bijector/transformed distribution system (not needed for normix's scope)

**Verdict:** **Recommended.** This gives the most flexibility with the least baggage.

---

### 3.4 Option D: Build on NumPyro's `Distribution`

Inherit from `numpyro.distributions.Distribution`.

**Pros:**
- Large existing distribution library to draw from
- Good PyTorch-like API with `arg_constraints`, `support`
- Many distributions already implemented

**Cons:**
- Heavy dependency (entire NumPyro PPL stack)
- No exponential family abstraction
- API designed for Bayesian PPL, not MLE/EM
- Would fight the framework's assumptions about how distributions are used

**Verdict:** **Not recommended.** Too heavy, wrong paradigm.

---

## 4. Recommended Design

### 4.1 Architecture: Equinox-based with patterns borrowed from FlowJAX and efax

```
eqx.Module
├── Distribution (ABC)
│   ├── log_prob, sample, event_shape
│   ├── mean, variance (optional overrides)
│   │
│   └── ExponentialFamily (ABC)
│       ├── natural_params, log_partition, sufficient_statistics, log_base_measure
│       ├── expectation_params (auto via jax.grad)
│       ├── fisher_information (auto via jax.hessian)
│       ├── from_natural, from_classical, from_expectation, fit
│       │
│       ├── Exponential, Gamma, InverseGamma
│       ├── InverseGaussian, GIG
│       ├── MultivariateNormal
│       │
│       └── JointNormalMixture (ABC)
│           ├── JointVarianceGamma
│           ├── JointNormalInverseGamma
│           ├── JointNormalInverseGaussian
│           └── JointGeneralizedHyperbolic
│
└── NormalMixtureModel (ABC)
    ├── log_prob (marginal), sample, fit (EM)
    ├── joint → JointNormalMixture
    ├── VarianceGamma
    ├── NormalInverseGamma
    ├── NormalInverseGaussian
    └── GeneralizedHyperbolic
```

### 4.2 What to borrow from each package

| Pattern | Source | How to use |
|---|---|---|
| **`eqx.Module` as base** | FlowJAX | All distribution objects are immutable Equinox modules |
| **Unbatched `_log_prob` / `_sample`** | FlowJAX | Implement single-sample logic; auto-vectorize with `vmap` |
| **`custom_jvp` on log partition** | efax | $\nabla\psi(\theta)$ returns `expectation_params()` — makes Fisher info automatic |
| **Pytree auto-registration** | Distrax/FlowJAX/Equinox | Handled by `eqx.Module` |
| **`Parameterize` for constraints** | FlowJAX (`paramax`) | Store unconstrained values, apply `softplus`/`exp` on access. Optional — can also use `jnp.maximum` clamping. |
| **`fit_to_data` pattern** | FlowJAX | Separate fitting from distribution class. `EMFitter` as a standalone module. |
| **`bessel_kve` for evaluation** | TFP | Use `tfp.math.log_bessel_kve` + `custom_jvp` as established in the Bessel notebook |
| **Sample-all-then-mask mixtures** | Distrax | For `MixtureSameFamily` if needed |

### 4.3 What NOT to borrow

| Anti-pattern | Source | Why avoid |
|---|---|---|
| Class-per-parametrization | efax | Triples number of classes; awkward for EM |
| TFP as hard dependency | Distrax | Too heavy; only need `tfp.math.bessel_kve` |
| `Jittable` pytree registration | Distrax | Fragile; `eqx.Module` is more robust |
| Bijector/Transformed system | Distrax/FlowJAX | Not needed — normix distributions have closed-form PDFs |
| `jnp.vectorize` for batching | FlowJAX | `vmap` is more explicit and JAX-idiomatic |
| PyTorch-style `arg_constraints` | NumPyro | Over-engineered for normix's needs |

### 4.4 Dependency strategy

**Core (required):**
```
jax >= 0.4.20
equinox >= 0.11
```

**For Bessel functions (required for GIG/GH):**
```
tensorflow-probability >= 0.25  # only tfp.math.bessel_kve used
```

Alternatively, vendor `bessel_kve` from TFP to avoid the full dependency. The relevant code is ~500 lines in `tfp.math.bessel` and is Apache-2.0 licensed.

**For optimization (required for $\eta \to \theta$):**
```
jaxopt >= 0.8  # L-BFGS-B for GIG
# OR use optax for simpler distributions
```

**Optional:**
```
optax  # gradient-based parameter optimization alternative
```

### 4.5 Concrete example: GIG distribution

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from normix_jax.bessel import log_kv  # TFP + custom_jvp wrapper


class GIG(ExponentialFamily):
    """Generalized Inverse Gaussian: GIG(p, a, b)."""
    p: jax.Array
    a: jax.Array
    b: jax.Array

    @property
    def event_shape(self):
        return ()

    def natural_params(self):
        return jnp.array([self.p - 1, -self.b / 2, -self.a / 2])

    def log_partition(self):
        sqrt_ab = jnp.sqrt(self.a * self.b)
        return jnp.log(2.0) + log_kv(self.p, sqrt_ab) + (self.p / 2) * (jnp.log(self.b) - jnp.log(self.a))

    def sufficient_statistics(self, x):
        return jnp.array([jnp.log(x), 1.0 / x, x])

    def log_base_measure(self, x):
        return jnp.where(x > 0, 0.0, -jnp.inf)

    def sample(self, key, sample_shape=()):
        # Use TFP's GIG sampler or rejection sampling
        ...

    @classmethod
    def from_classical(cls, *, p, a, b):
        return cls(p=jnp.asarray(p, dtype=jnp.float64),
                   a=jnp.asarray(a, dtype=jnp.float64),
                   b=jnp.asarray(b, dtype=jnp.float64))

    @classmethod
    def from_natural(cls, theta):
        p = theta[0] + 1
        b = jnp.maximum(-2 * theta[1], 1e-30)
        a = jnp.maximum(-2 * theta[2], 1e-30)
        return cls(p=p, a=a, b=b)

    def mean(self):
        return jnp.sqrt(self.b / self.a) * jnp.exp(
            log_kv(self.p + 1, jnp.sqrt(self.a * self.b))
            - log_kv(self.p, jnp.sqrt(self.a * self.b)))
```

Usage:
```python
gig = GIG.from_classical(p=1.5, a=2.0, b=1.0)

# Automatic expectation params via jax.grad of log partition
eta = gig.expectation_params()    # [E[log X], E[1/X], E[X]]

# Automatic Fisher information via jax.hessian
fisher = gig.fisher_information()  # 3×3 matrix

# Vectorized evaluation
log_probs = jax.vmap(gig.log_prob)(data)

# JIT everything
fast_logprob = jax.jit(jax.vmap(gig.log_prob))
```

### 4.6 EM algorithm design

Follow GMMX's pattern: separate the algorithm from the model.

```python
class EMFitter(eqx.Module):
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)
    regularization: str = eqx.field(static=True, default='det_sigma_one')

    def fit(self, model_cls, X, *, key):
        """Fit a NormalMixtureModel via EM. Returns new fitted model."""
        model = self._initialize(model_cls, X, key)

        for i in range(self.max_iter):
            new_model = self._em_step(model, X)
            if self._converged(model, new_model):
                break
            model = new_model

        return model

    @eqx.filter_jit
    def _em_step(self, model, X):
        expectations = model.e_step(X)
        return model.m_step(X, expectations)
```

---

## 5. Summary

### Use Equinox directly, borrow patterns from FlowJAX and efax

The JAX distribution ecosystem is fragmented. No package provides what normix needs (exponential family + GIG/GH + EM). The recommended path:

1. **Base class**: `eqx.Module` — gives immutable pytrees, `jit`/`vmap`/`grad` compatibility
2. **Distribution API**: inspired by FlowJAX's unbatched `_log_prob`/`_sample` design
3. **Exponential family**: inspired by efax's `custom_jvp` on log partition for automatic Fisher information
4. **Bessel functions**: TFP's `log_bessel_kve` + `custom_jvp` (already validated in the Bessel notebook)
5. **EM algorithm**: separated from distribution class, JIT-compiled steps, inspired by GMMX
6. **Dependencies**: only `jax` + `equinox` + `tensorflow-probability` (for Bessel only)

This keeps normix focused on its unique value (GH family + EM) while using the best infrastructure from the JAX ecosystem.
