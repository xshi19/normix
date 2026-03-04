# Feasibility Report: Migrating normix from NumPy/SciPy to JAX

**Date:** 2026-03-04  
**Status:** Design exploration  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current normix Architecture](#2-current-normix-architecture)
3. [The efax Package: Design Analysis](#3-the-efax-package-design-analysis)
4. [JAX Ecosystem Assessment](#4-jax-ecosystem-assessment)
5. [The Bessel Function Challenge](#5-the-bessel-function-challenge)
6. [Proposed JAX-Based Architecture](#6-proposed-jax-based-architecture)
7. [EM Algorithm in JAX](#7-em-algorithm-in-jax)
8. [Migration Strategy](#8-migration-strategy)
9. [Risk Assessment](#9-risk-assessment)
10. [Recommendations](#10-recommendations)

---

## 1. Executive Summary

Migrating normix to JAX is **feasible**, but requires a significant architectural redesign rather than a mechanical translation. The key benefits would be:

- **Automatic differentiation** of the log partition function $\psi(\theta)$, eliminating hand-coded gradients and numerical differentiation for $\eta = \nabla\psi(\theta)$ and $I(\theta) = \nabla^2\psi(\theta)$
- **JIT compilation** for 10-100x speedups in EM iterations
- **`vmap`** for natural batching over samples, parameter initializations, and multiple restarts
- **GPU/TPU support** for large-scale fitting
- **Composability** with the broader JAX ML ecosystem (Optax, Equinox, Flax)

The main challenges are:

1. **Bessel functions**: The modified Bessel function of the second kind $K_\nu(z)$ and its derivative with respect to the order $\nu$ are central to normix. JAX has no built-in $K_\nu$, but the `logbesselk` package provides a JAX-compatible $\log K_\nu(z)$ with full autodiff support for both $\nu$ and $z$.
2. **Constrained optimization**: The $\eta \to \theta$ conversion for the GIG distribution currently relies on SciPy's L-BFGS-B with bounds. JAXopt provides `LBFGSB` as a drop-in replacement.
3. **Paradigm shift**: JAX is purely functional and immutable. The current normix design uses mutable objects with cache invalidation. A new design based on Equinox modules or JAX pytrees is needed.

**Verdict:** The migration is viable. The `logbesselk` package is the critical enabler for the Bessel function problem. A hybrid design combining efax's parametrization-as-type pattern with a more ML-oriented "trainable model" API is recommended.

---

## 2. Current normix Architecture

### 2.1 Class Hierarchy

```
Distribution (ABC)
├── ExponentialFamily (ABC)
│   ├── Exponential
│   ├── Gamma
│   ├── InverseGamma
│   ├── InverseGaussian
│   ├── GeneralizedInverseGaussian (GIG)
│   ├── MultivariateNormal
│   └── JointNormalMixture (ABC)   ← joint f(x,y) IS exponential family
│       ├── JointVarianceGamma
│       ├── JointNormalInverseGamma
│       ├── JointNormalInverseGaussian
│       └── JointGeneralizedHyperbolic
│
└── NormalMixture (ABC)             ← marginal f(x) is NOT exponential family
    ├── VarianceGamma
    ├── NormalInverseGamma
    ├── NormalInverseGaussian
    └── GeneralizedHyperbolic
```

### 2.2 Key Design Patterns

| Aspect | Current normix |
|---|---|
| **Mutability** | Mutable objects; `_invalidate_cache()` clears `functools.cached_property` entries |
| **Parameter storage** | Named internal attributes (`_alpha`, `_beta`, `_mu`, `_L_Sigma`) as single source of truth |
| **Parametrizations** | Three views (classical, natural, expectation) computed lazily via `cached_property` |
| **Gradient of $\psi$** | `scipy.differentiate.jacobian` (numerical) or hand-coded analytical overrides |
| **Hessian of $\psi$** | `scipy.differentiate.hessian` (numerical) or hand-coded analytical overrides |
| **$\eta \to \theta$** | `scipy.optimize.minimize` (L-BFGS-B) with multi-start |
| **Bessel functions** | `scipy.special.kve` with hand-coded log-space wrappers and asymptotic fallbacks |
| **EM algorithm** | Python `for` loop; E-step/M-step as methods on the `NormalMixture` subclass |
| **Covariance** | Cholesky-first everywhere (`scipy.linalg.cho_solve`, `solve_triangular`) |

### 2.3 NumPy/SciPy Dependencies

The following SciPy/NumPy functions are used throughout the codebase:

| Category | Functions |
|---|---|
| **Special functions** | `scipy.special.kve`, `gammaln`, `digamma`, `polygamma` |
| **Linear algebra** | `scipy.linalg.cholesky`, `cho_solve`, `solve_triangular` |
| **Optimization** | `scipy.optimize.minimize` (L-BFGS-B), `Bounds` |
| **Differentiation** | `scipy.differentiate.jacobian`, `hessian` |
| **Statistics** | `scipy.stats.geninvgauss`, `norm`, `multivariate_normal` |
| **NumPy core** | `np.einsum`, `np.dot`, `np.linalg.slogdet`, `np.linalg.eigvalsh` |

---

## 3. The efax Package: Design Analysis

### 3.1 Core Design Pattern

efax uses a radically different design from normix: **each parametrization of a distribution is a separate class**. For example, a Gamma distribution has:

- `GammaNP` — natural parametrization (a frozen JAX pytree)
- `GammaEP` — expectation parametrization (a frozen JAX pytree)
- `GammaVP` — "value parametrization" (classical parameters)

Each class is an immutable JAX-registered dataclass (pytree node). Converting between parametrizations creates a **new object** of the target type.

### 3.2 Class Hierarchy

```
Distribution
└── SimpleDistribution
    ├── NaturalParametrization[EP, Domain]   ← base for NP classes
    │   ├── GammaNP
    │   ├── NormalNP
    │   ├── InverseGaussianNP
    │   └── ...
    └── ExpectationParametrization[NP]       ← base for EP classes
        ├── GammaEP
        ├── NormalEP
        └── ...

Mixins: HasEntropyNP, HasEntropyEP, Samplable, ExpToNat, ...
```

### 3.3 Key Mechanisms

**Log partition gradient via custom JVP:**
efax defines a custom JVP rule on `log_normalizer` that computes $\nabla\psi(\theta) = E[t(X)]$ using the analytical `to_exp()` method. This avoids double numerical differentiation — the gradient of the log normalizer is the conversion from NP to EP, which is defined analytically per distribution.

**Fisher information via JAX autodiff:**
$$I(\theta) = \nabla^2\psi(\theta) = \texttt{jacfwd}(\texttt{grad}(\psi))(\theta)$$
This is fully automatic for any distribution once `log_normalizer` and `to_exp` are defined.

**$\eta \to \theta$ via Newton's method:**
efax uses the `ExpToNat` mixin with the `optimistix` library (Newton's method). The Jacobian of the map $\theta \mapsto \eta(\theta)$ is the Fisher information, which efax computes automatically.

### 3.4 What efax Does and Does NOT Have

| Feature | efax | normix |
|---|---|---|
| Gamma | Yes | Yes |
| Inverse Gamma | Yes (via transform) | Yes |
| Inverse Gaussian | Yes | Yes |
| Normal (univariate) | Yes | Yes |
| Normal (multivariate) | Yes | Yes |
| **GIG** | **No** | Yes |
| **Normal mixtures** | **No** | Yes |
| **EM algorithm** | **No** | Yes |
| **Bessel $K_\nu$** | **No** | Yes |
| Fitting / `fit()` | No | Yes |
| Sampling | Partial | Yes |
| JIT/grad/vmap | Yes (native) | No |

**Key insight:** efax covers only the simpler distributions. The GIG, all normal mixture distributions, the EM algorithm, and the Bessel function infrastructure are **unique to normix**. These are precisely the parts that are hardest to port to JAX.

### 3.5 What to Adopt from efax

1. **Custom JVP on log partition**: The idea that $\nabla\psi(\theta)$ is the NP→EP conversion, used as a custom JVP rule, is elegant and should be adopted. It makes Fisher information computation fully automatic.

2. **Immutable pytree objects**: All distribution objects are JAX pytrees, making them compatible with `jit`, `vmap`, `grad`, and `scan` out of the box.

3. **Type-safe parametrizations**: Having separate types for NP and EP prevents accidental mixing.

### 3.6 What NOT to Adopt from efax

1. **Separate class per parametrization**: This triples the number of classes. For normix's complex mixture hierarchy, this would lead to an explosion of types (e.g., `JointGeneralizedHyperbolicNP`, `JointGeneralizedHyperbolicEP`, `JointGeneralizedHyperbolicCP`, `GeneralizedHyperbolicModel`, ...). A single class with methods for each parametrization is more practical.

2. **No model/fitting concept**: efax is a parametrization library, not a model library. normix needs `fit()`, EM, sampling, and PDF evaluation — a "model" abstraction is essential.

3. **Newton-only $\eta \to \theta$**: efax's Newton method does not work well for GIG (see Section 5). normix needs constrained optimization.

---

## 4. JAX Ecosystem Assessment

### 4.1 Direct Replacements for SciPy/NumPy

| normix uses | JAX replacement | Notes |
|---|---|---|
| `numpy.*` | `jax.numpy.*` | Near drop-in; enable `jax_enable_x64` for float64 |
| `scipy.special.gammaln` | `jax.scipy.special.gammaln` | Available |
| `scipy.special.digamma` | `jax.scipy.special.digamma` | Available |
| `scipy.special.kve` | **Not available** | Use `logbesselk` (see Section 5) |
| `scipy.linalg.cholesky` | `jax.scipy.linalg.cholesky` | Available |
| `scipy.linalg.cho_solve` | `jax.scipy.linalg.cho_solve` | Available |
| `scipy.linalg.solve_triangular` | `jax.scipy.linalg.solve_triangular` | Available |
| `scipy.optimize.minimize` (L-BFGS-B) | `jaxopt.LBFGSB` | Available in JAXopt |
| `scipy.differentiate.jacobian` | `jax.jacfwd` / `jax.jacrev` | Exact, not numerical |
| `scipy.differentiate.hessian` | `jax.hessian` | Exact, not numerical |
| `numpy.random.*` | `jax.random.*` | Explicit PRNG keys |

### 4.2 Key JAX Libraries

| Library | Purpose | Relevance |
|---|---|---|
| **Equinox** | Pytree-based neural network library | Best model abstraction (`eqx.Module`) |
| **JAXopt** | Optimization in JAX | L-BFGS-B for $\eta \to \theta$, projected gradient |
| **Optax** | Gradient-based optimizers | Alternative for parameter optimization |
| **Distrax** | Probability distributions (DeepMind) | Reference for distribution API design |
| **TFP JAX** | TensorFlow Probability on JAX | Has `GeneralizedInverseGaussian`, `MultivariateNormalTriL` |
| **logbesselk** | $\log K_\nu(z)$ with full autodiff | **Critical** for GIG/GH (see Section 5) |

### 4.3 TFP JAX Substrate

TensorFlow Probability's JAX substrate (`tfp.substrates.jax`) is worth noting:

- Has `tfd.GeneralizedInverseGaussian` — but uses a different parametrization and may not support all the operations normix needs
- Has `tfp.math.bessel_kve` and `log_bessel_kve` — but gradients with respect to the **order** $\nu$ are not defined
- Has `tfd.MultivariateNormalTriL` — Cholesky-parametrized multivariate normal

TFP is useful as a reference but cannot replace normix's Bessel infrastructure because it lacks $\partial/\partial\nu$ support.

---

## 5. The Bessel Function Challenge

This is the single most critical technical challenge in the migration.

### 5.1 Why Bessel Functions Matter

The GIG log partition function is:

$$\psi(\theta) = \log 2 + \log K_p(\sqrt{ab}) + \frac{p}{2}(\log b - \log a)$$

where $\theta = (p-1, -b/2, -a/2)$. Computing expectation parameters $\eta = \nabla\psi(\theta)$ requires:

$$\frac{\partial}{\partial p} \log K_p(\sqrt{ab}), \quad \frac{\partial}{\partial z} \log K_p(z)\bigg|_{z=\sqrt{ab}}$$

The derivative with respect to **z** can use the recurrence $K'_\nu(z) = -(K_{\nu-1}(z) + K_{\nu+1}(z))/2$.

The derivative with respect to the **order** $\nu$ has **no closed-form recurrence**. normix currently uses central finite differences:

$$\frac{\partial}{\partial\nu} \log K_\nu(z) \approx \frac{\log K_{\nu+\varepsilon}(z) - \log K_{\nu-\varepsilon}(z)}{2\varepsilon}$$

### 5.2 The `logbesselk` Package

The **`logbesselk`** package (v3.4.0+, [GitHub](https://github.com/tk2lab/logbesselk)) is the critical enabler:

```python
from logbesselk.jax import log_bessel_k

# Computes log K_v(z) with full JAX autodiff support
val = log_bessel_k(v, z)

# Gradient w.r.t. BOTH v and z — this is the key capability
grad_v = jax.grad(log_bessel_k, argnums=0)(v, z)
grad_z = jax.grad(log_bessel_k, argnums=1)(v, z)

# Hessian is also available
hess = jax.hessian(log_bessel_k, argnums=(0, 1))(v, z)
```

**Key properties:**
- Based on numerical integration of the integral representation of $K_\nu(z)$
- Supports `jax.jit`, `jax.vmap`, `jax.grad` for both arguments
- Requires JAX >= 0.4, Python >= 3.10
- Numerically stable in log space

### 5.3 Impact on the Architecture

With `logbesselk`, the GIG log partition function becomes fully differentiable via JAX autodiff:

```python
def log_partition_gig(theta):
    p = theta[0] + 1
    b = -2 * theta[1]
    a = -2 * theta[2]
    sqrt_ab = jnp.sqrt(a * b)
    return jnp.log(2.0) + log_bessel_k(p, sqrt_ab) + (p / 2) * (jnp.log(b) - jnp.log(a))
```

Then:
- $\eta = \nabla\psi(\theta)$ is just `jax.grad(log_partition_gig)(theta)` — no hand-coded derivatives, no finite differences
- $I(\theta) = \nabla^2\psi(\theta)$ is just `jax.hessian(log_partition_gig)(theta)`
- The custom JVP trick from efax can still be used for efficiency

This **eliminates** the need for:
- `log_kv_derivative_v` (central differences)
- `log_kv_derivative_z` (recurrence)
- `kv_ratio` (manual log-space tricks)
- The entire `normix/utils/bessel.py` module (replaced by `logbesselk`)

### 5.4 Degenerate Cases ($\sqrt{ab} \to 0$)

normix currently handles the Gamma limit ($b \to 0$) and Inverse Gamma limit ($a \to 0$) with special-case branching. In JAX, Python-level `if` statements are not compatible with `jit`. Two options:

1. **`jnp.where` branching**: Replace `if sqrt_ab < threshold` with `jnp.where(sqrt_ab < threshold, gamma_limit, bessel_formula)`. Both branches are evaluated, but the correct one is selected. This works with `jit` and `grad`.

2. **Regularization**: Clamp $a$ and $b$ to a minimum value (e.g., $10^{-10}$) so $\sqrt{ab}$ is always above the degenerate threshold. Then rely on `logbesselk`'s own numerical handling of small arguments.

Option 2 is simpler and recommended for the JAX port. The Gamma and InverseGamma distributions can be implemented as separate classes for the exact limiting case if needed.

### 5.5 Bessel Function Stability: `logbesselk` vs normix

| Aspect | normix (current) | JAX + `logbesselk` |
|---|---|---|
| Core function | `scipy.special.kve` + manual log wrapper | `log_bessel_k` (native log space) |
| $\partial/\partial z$ | Recurrence relation (analytical) | `jax.grad` (automatic) |
| $\partial/\partial\nu$ | Central differences ($\varepsilon = 10^{-6}$) | `jax.grad` (automatic, exact) |
| Small $z$ asymptotics | Hand-coded fallbacks | Handled internally by `logbesselk` |
| Hessian | `scipy.differentiate.hessian` (numerical) | `jax.hessian` (automatic, exact) |
| JIT-compatible | No | Yes |
| Vectorizable | Manual `np.vectorize` | `jax.vmap` |

**The JAX version would be strictly superior** in terms of derivative accuracy and code simplicity.

### 5.6 Remaining Risk

The `logbesselk` package is maintained by a single researcher (Tomoaki Takao). Its numerical accuracy for extreme parameter ranges ($|p| > 100$, $z < 10^{-15}$, or $z > 10^6$) should be validated against SciPy's `kve` before relying on it in production. A comprehensive test suite comparing `logbesselk` against `scipy.special.kve` across the parameter ranges relevant to normix is recommended as an early validation step.

---

## 6. Proposed JAX-Based Architecture

### 6.1 Design Philosophy

The new design should combine:

- **From efax**: Immutable pytree objects, custom JVP for log partition gradient, automatic Fisher information
- **From ML frameworks (Equinox/Flax)**: The "model" abstraction with `fit()`, `log_prob()`, `sample()`
- **From normix**: The exponential family structure, three parametrizations, normal mixture hierarchy, EM algorithm

The core idea: **a distribution is an immutable Equinox module** whose array-valued fields are its parameters. Creating a distribution from different parametrizations uses `@classmethod` constructors. Fitting returns a **new** distribution object (no mutation). EM iterations produce a sequence of immutable state objects.

### 6.2 Proposed Class Hierarchy

```python
import equinox as eqx
import jax.numpy as jnp

class ExponentialFamily(eqx.Module):
    """Base class for exponential family distributions.
    
    Subclasses define their parameters as eqx.Module fields.
    The log partition function is the core primitive — all else follows from autodiff.
    """
    
    # --- Core abstract methods (subclass must implement) ---
    
    @abc.abstractmethod
    def log_partition(self) -> jax.Array:
        """Log partition function ψ(θ) evaluated at current natural parameters."""
        ...
    
    @abc.abstractmethod
    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        """Sufficient statistics t(x)."""
        ...
    
    @abc.abstractmethod
    def log_base_measure(self, x: jax.Array) -> jax.Array:
        """Log base measure log h(x)."""
        ...
    
    @abc.abstractmethod
    def natural_params(self) -> jax.Array:
        """Return natural parameters θ as a flat array."""
        ...
    
    # --- Derived methods (automatic via JAX) ---
    
    def expectation_params(self) -> jax.Array:
        """η = ∇ψ(θ) via jax.grad."""
        return jax.grad(self._log_partition_from_theta)(self.natural_params())
    
    def fisher_information(self) -> jax.Array:
        """I(θ) = ∇²ψ(θ) via jax.hessian."""
        return jax.hessian(self._log_partition_from_theta)(self.natural_params())
    
    def log_prob(self, x: jax.Array) -> jax.Array:
        """log p(x|θ) = log h(x) + θᵀt(x) − ψ(θ)."""
        theta = self.natural_params()
        return self.log_base_measure(x) + jnp.dot(theta, self.sufficient_statistics(x)) - self.log_partition()
    
    # --- Constructors ---
    
    @classmethod
    def from_natural(cls, theta: jax.Array) -> 'ExponentialFamily':
        """Create from natural parameters."""
        ...
    
    @classmethod
    def from_classical(cls, **kwargs) -> 'ExponentialFamily':
        """Create from classical parameters."""
        ...
    
    @classmethod
    def from_expectation(cls, eta: jax.Array) -> 'ExponentialFamily':
        """Create from expectation parameters (requires solving ∇ψ(θ) = η)."""
        ...
    
    @classmethod
    def fit(cls, X: jax.Array) -> 'ExponentialFamily':
        """MLE: η̂ = mean(t(X)), then convert to natural params."""
        ...
```

### 6.3 Concrete Distribution Example: GIG

```python
class GIG(ExponentialFamily):
    """Generalized Inverse Gaussian distribution.
    
    Stores classical parameters (p, a, b) as the internal representation.
    Natural and expectation parameters are computed on demand via methods.
    """
    p: jax.Array   # shape parameter
    a: jax.Array   # rate parameter (coefficient of x)
    b: jax.Array   # rate parameter (coefficient of 1/x)
    
    def log_partition(self) -> jax.Array:
        sqrt_ab = jnp.sqrt(self.a * self.b)
        return (jnp.log(2.0) 
                + log_bessel_k(self.p, sqrt_ab) 
                + (self.p / 2) * (jnp.log(self.b) - jnp.log(self.a)))
    
    def natural_params(self) -> jax.Array:
        return jnp.array([self.p - 1, -self.b / 2, -self.a / 2])
    
    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        return jnp.array([jnp.log(x), 1.0 / x, x])
    
    def log_base_measure(self, x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, 0.0, -jnp.inf)
    
    @classmethod
    def from_classical(cls, *, p, a, b) -> 'GIG':
        return cls(p=jnp.asarray(p), a=jnp.asarray(a), b=jnp.asarray(b))
    
    @classmethod
    def from_natural(cls, theta: jax.Array) -> 'GIG':
        p = theta[0] + 1
        b = jnp.maximum(-2 * theta[1], 0.0)
        a = jnp.maximum(-2 * theta[2], 0.0)
        return cls(p=p, a=a, b=b)
    
    def mean(self) -> jax.Array:
        return jnp.sqrt(self.b / self.a) * jnp.exp(
            log_bessel_k(self.p + 1, jnp.sqrt(self.a * self.b)) 
            - log_bessel_k(self.p, jnp.sqrt(self.a * self.b))
        )
```

### 6.4 Normal Mixture Model

The marginal normal mixture distribution (e.g., GH) is not an exponential family, so it gets a different base class focused on the "model" pattern:

```python
class NormalMixtureModel(eqx.Module):
    """Base class for normal mixture models X = μ + γY + √Y·Z, Z ~ N(0,Σ).
    
    This is a trainable model: you create it, fit it to data, then use it
    for inference (log_prob, sample, moments).
    """
    mu: jax.Array           # location (d,)
    gamma: jax.Array        # skewness (d,)
    L_Sigma: jax.Array      # Cholesky of Σ (d, d)
    
    # Subordinator parameters defined by subclass
    
    @abc.abstractmethod
    def subordinator(self) -> ExponentialFamily:
        """Return the mixing distribution as an ExponentialFamily object."""
        ...
    
    def log_prob(self, x: jax.Array) -> jax.Array:
        """Marginal log PDF: log f(x) = log ∫ f(x|y) f(y) dy."""
        ...
    
    def sample(self, key: jax.Array, shape: tuple) -> jax.Array:
        """Sample from the marginal distribution."""
        key1, key2 = jax.random.split(key)
        y = self.subordinator().sample(key1, shape)
        z = jax.random.normal(key2, shape=(*shape, self.mu.shape[0]))
        return self.mu + self.gamma * y[..., None] + jnp.sqrt(y[..., None]) * (z @ self.L_Sigma.T)
    
    @classmethod
    def fit(cls, X: jax.Array, *, key: jax.Array, max_iter=200, tol=1e-6,
            regularization='det_sigma_one') -> 'NormalMixtureModel':
        """Fit via EM algorithm. Returns a new fitted model."""
        ...
    
    # Joint distribution access
    def joint(self) -> 'JointNormalMixture':
        """Return the joint distribution f(x,y) as an ExponentialFamily."""
        ...
```

### 6.5 Parametrization Design: Single Class vs Multiple Classes

#### Option A: Single class with methods (recommended)

```python
gig = GIG.from_classical(p=1.0, a=2.0, b=3.0)
theta = gig.natural_params()          # jax.Array
eta = gig.expectation_params()        # jax.Array via jax.grad
fisher = gig.fisher_information()     # jax.Array via jax.hessian
params = gig.classical_params()       # GIGParams dataclass

# Create from different parametrizations
gig2 = GIG.from_natural(theta)
gig3 = GIG.from_expectation(eta)
```

**Pros:** Simpler API, fewer classes, natural for the "model" pattern.  
**Cons:** Less type-safe (natural and expectation params are both `jax.Array`).

#### Option B: Separate types per parametrization (efax style)

```python
gig_np = GIGNaturalParams(theta_1=0.0, theta_2=-1.5, theta_3=-1.0)
gig_ep = gig_np.to_expectation()   # → GIGExpectationParams
gig_cp = gig_np.to_classical()     # → GIGClassicalParams
gig_np2 = gig_ep.to_natural()      # → GIGNaturalParams
```

**Pros:** Type-safe, prevents mixing parametrizations accidentally.  
**Cons:** Triple the classes; awkward for EM (which needs to update parameters across steps).

#### Recommendation

**Option A** for the main API, with Option B available as an internal mechanism if needed. The key reason: the EM algorithm needs to repeatedly create new distribution objects from updated parameters. Having a single class with `from_natural()` / `from_classical()` constructors is more ergonomic than juggling multiple types.

However, **adopt efax's custom JVP trick** within Option A: define a `_log_partition_from_theta` function with a custom JVP that returns `expectation_params()` as the gradient. This gives the best of both worlds.

### 6.6 Immutability and the EM Fast Path

In the current normix, `_set_internal()` provides a "zero-overhead" fast path for EM by directly setting attributes without Cholesky recomputation. In JAX, since objects are immutable, the pattern becomes:

```python
@jax.jit
def em_step(model, X):
    # E-step: compute conditional expectations
    e_y, e_inv_y, e_log_y = model.conditional_expectations(X)
    
    # M-step: compute new parameters (pure functions)
    new_mu, new_gamma, new_L_Sigma = m_step_normal(X, e_y, e_inv_y)
    new_sub_params = m_step_subordinator(e_y, e_inv_y, e_log_y)
    
    # Create new model (immutable — no cache invalidation needed)
    return eqx.tree_at(
        lambda m: (m.mu, m.gamma, m.L_Sigma, m.sub_p, m.sub_a, m.sub_b),
        model,
        (new_mu, new_gamma, new_L_Sigma, *new_sub_params)
    )
```

With JIT compilation, this is likely **faster** than the current normix approach, because:
- No Python-level cache invalidation overhead
- The entire EM step is compiled to XLA
- Memory allocation is optimized by the XLA compiler

---

## 7. EM Algorithm in JAX

### 7.1 Functional EM Pattern

The EM algorithm becomes a pure function that transforms state:

```python
@dataclass
class EMState:
    model: NormalMixtureModel
    log_likelihood: jax.Array
    iteration: int
    converged: bool

def em_step(state: EMState, X: jax.Array) -> EMState:
    """Single EM iteration (pure function)."""
    model = state.model
    
    # E-step
    expectations = e_step(model, X)
    
    # M-step
    new_model = m_step(model, X, expectations)
    
    # Compute log-likelihood
    new_ll = jnp.mean(jax.vmap(new_model.log_prob)(X))
    
    # Check convergence
    converged = check_convergence(model, new_model, state.log_likelihood, new_ll)
    
    return EMState(
        model=new_model,
        log_likelihood=new_ll,
        iteration=state.iteration + 1,
        converged=converged
    )
```

### 7.2 Loop Strategies

**Option 1: Python loop with JIT step (recommended for development)**
```python
state = init_em_state(X, key)
for i in range(max_iter):
    state = jit_em_step(state, X)
    if state.converged:
        break
```
Easy to debug, supports early stopping, compatible with logging.

**Option 2: `jax.lax.while_loop` (maximum performance)**
```python
def cond_fn(state):
    return ~state.converged & (state.iteration < max_iter)

def body_fn(state):
    return em_step(state, X)

final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
```
Fully compiled, no Python overhead, but harder to debug and not reverse-mode differentiable.

**Option 3: `jax.lax.scan` with convergence flag (hybrid)**
```python
def scan_fn(state, _):
    new_state = jax.lax.cond(
        state.converged,
        lambda s: s,      # no-op if converged
        lambda s: em_step(s, X),
        state
    )
    return new_state, new_state.log_likelihood

final_state, ll_trace = jax.lax.scan(scan_fn, init_state, jnp.arange(max_iter))
```
Fixed iteration count but freezes parameters after convergence. Returns full trace for diagnostics.

**Recommendation:** Start with Option 1 for development and testing. Move to Option 2 or 3 for production after validating correctness.

### 7.3 Vectorized Multi-Start Initialization

One major advantage of JAX: the GH distribution's multi-start initialization (trying NIG, VG, NInvG as starting points) can be **`vmap`-ed**:

```python
@jax.vmap
def fit_single_start(key, X, init_params):
    """Fit from a single starting point."""
    model = GH.from_params(init_params)
    return run_em(model, X, key)

# Run all starting points in parallel
keys = jax.random.split(key, n_starts)
init_params_batch = jnp.stack([nig_init, vg_init, ninvg_init, ...])
results = fit_single_start(keys, X, init_params_batch)

# Select best by log-likelihood
best_idx = jnp.argmax(results.log_likelihood)
best_model = jax.tree.map(lambda x: x[best_idx], results)
```

This replaces the sequential for-loop over starting points in the current normix, potentially giving a significant speedup.

### 7.4 Convergence Checking

Convergence in JAX cannot use Python-level comparisons inside `jit`. Use:

```python
def check_convergence(old_model, new_model, old_ll, new_ll, tol=1e-6):
    """Check convergence using relative parameter change."""
    delta_mu = jnp.max(jnp.abs(new_model.mu - old_model.mu)) / (jnp.max(jnp.abs(old_model.mu)) + 1e-10)
    delta_gamma = jnp.max(jnp.abs(new_model.gamma - old_model.gamma)) / (jnp.max(jnp.abs(old_model.gamma)) + 1e-10)
    delta_L = jnp.linalg.norm(new_model.L_Sigma - old_model.L_Sigma) / (jnp.linalg.norm(old_model.L_Sigma) + 1e-10)
    
    return (delta_mu < tol) & (delta_gamma < tol) & (delta_L < tol)
```

### 7.5 Data Normalization

The current normix normalizes data (median/MAD) before EM and denormalizes afterwards. This is a pure transformation and works the same way in JAX:

```python
def normalize(X):
    median = jnp.median(X, axis=0)
    mad = 1.4826 * jnp.median(jnp.abs(X - median), axis=0)
    mad = jnp.where(mad < 1e-10, 1.0, mad)
    return (X - median) / mad, median, mad

def denormalize_params(mu, gamma, L_Sigma, median, mad):
    """Map fitted parameters back to original scale."""
    D = jnp.diag(mad)
    return median + D @ mu, D @ gamma, D @ L_Sigma
```

---

## 8. Migration Strategy

### 8.1 Phased Approach

#### Phase 0: Validation Infrastructure
- Set up a test harness that compares JAX implementations against existing normix (NumPy/SciPy) for numerical correctness
- Validate `logbesselk` against `scipy.special.kve` across the full parameter range used by normix
- Establish float64 precision baselines

#### Phase 1: Core Primitives
- Port `log_kv` → `logbesselk.jax.log_bessel_k`
- Implement `ExponentialFamily` base as `eqx.Module`
- Port simple distributions: Exponential, Gamma, InverseGamma
- Verify all three parametrization conversions match normix

#### Phase 2: GIG Distribution
- Port `GeneralizedInverseGaussian` with `logbesselk`
- Verify that `jax.grad(log_partition)` matches normix's hand-coded `_natural_to_expectation`
- Implement $\eta \to \theta$ using `jaxopt.LBFGSB` (or Newton + projection)
- Handle degenerate cases ($\sqrt{ab} \to 0$) with `jnp.where`

#### Phase 3: Normal Mixture Infrastructure
- Port `MultivariateNormal` (Cholesky-first)
- Implement `JointNormalMixture` base
- Port `NormalMixture` base with EM framework

#### Phase 4: Mixture Distributions
- Port in order: NIG → VG → NInvG → GH
- For each: verify marginal logpdf, conditional expectations, EM fitting
- GH last (most complex, depends on all others)

#### Phase 5: Performance and API Polish
- JIT-compile full EM pipeline
- `vmap` multi-start initialization
- Clean public API with `fit()`, `log_prob()`, `sample()`
- Documentation and examples

### 8.2 Estimated Effort

| Phase | Effort | Risk |
|---|---|---|
| Phase 0 | 1-2 weeks | Low |
| Phase 1 | 2-3 weeks | Low |
| Phase 2 | 2-3 weeks | **Medium** (Bessel validation) |
| Phase 3 | 2-3 weeks | Low |
| Phase 4 | 3-4 weeks | **Medium** (EM convergence) |
| Phase 5 | 2-3 weeks | Low |
| **Total** | **12-18 weeks** | |

### 8.3 Coexistence Strategy

During migration, both versions can coexist:

```
normix/             ← current NumPy/SciPy version (stable, tested)
normix_jax/         ← new JAX version (under development)
tests/
├── test_*.py              ← existing tests
└── test_jax_vs_numpy/     ← cross-validation tests
```

The cross-validation tests ensure the JAX version produces the same results as the NumPy version, within floating-point tolerance.

---

## 9. Risk Assessment

### 9.1 High Risk

| Risk | Impact | Mitigation |
|---|---|---|
| `logbesselk` numerical instability for extreme parameters | GIG/GH fitting fails | Extensive validation; fallback to TFP's `bessel_kve` for evaluation-only; implement custom asymptotic handlers |
| `logbesselk` maintenance risk (single maintainer) | Package could become abandoned | Vendor the source code; it's small (~500 lines); consider contributing to JAX core |
| EM convergence differs between NumPy and JAX versions | Hard to debug, subtle numerical differences | Cross-validation test suite; float64 everywhere; tolerance-based comparison |

### 9.2 Medium Risk

| Risk | Impact | Mitigation |
|---|---|---|
| JAX float64 precision on GPU | Results differ from CPU | Always enable `jax_enable_x64`; test on both CPU and GPU |
| `jaxopt.LBFGSB` behavior differs from SciPy's L-BFGS-B | $\eta \to \theta$ conversion quality degrades | Compare results; tune tolerances; fall back to SciPy for this step if needed |
| Cholesky decomposition failure in JAX (no `robust_cholesky`) | EM iteration crashes on ill-conditioned $\Sigma$ | Implement JAX-compatible `safe_cholesky` with jitter; use regularization |
| JIT compilation overhead for first call | Slow interactive use | Cache compiled functions; provide non-JIT debug mode |

### 9.3 Low Risk

| Risk | Impact | Mitigation |
|---|---|---|
| `jnp.where` evaluates both branches | Wasted computation for degenerate cases | Minimal performance impact in practice |
| Random number generation differs (JAX explicit keys vs NumPy global state) | Different samples for same seed | By design; document the change |
| No `scipy.stats.geninvgauss` for sampling | Need alternative GIG sampler | Use TFP's GIG distribution for sampling, or implement the Hörmann-Leydold algorithm in JAX |

---

## 10. Recommendations

### 10.1 Architecture Recommendations

1. **Use Equinox `eqx.Module` as the base class** for all distributions. This gives JAX pytree registration, immutability, and a natural "model" abstraction without the boilerplate of manual pytree registration.

2. **Single class per distribution** with `from_classical()`, `from_natural()`, `from_expectation()` constructors. Do not adopt efax's separate-class-per-parametrization pattern — it would lead to an unmanageable number of classes for the mixture hierarchy.

3. **Adopt efax's custom JVP trick** for the log partition function. Define `expectation_params()` analytically where possible, and register it as the JVP of `log_partition()`. This makes Fisher information computation fully automatic via `jax.hessian`.

4. **Separate the EM algorithm from the distribution class.** Create an `EMFitter` class (or module of pure functions) that takes a model and data and returns a fitted model. This follows the GMMX design pattern and keeps the distribution class focused on mathematical operations.

5. **Store Cholesky factors** as the primary representation for covariance, as normix already does. JAX's `jax.scipy.linalg.cho_solve` and `solve_triangular` are direct replacements.

### 10.2 Bessel Function Recommendations

6. **Use `logbesselk` as the primary Bessel function implementation.** It provides exactly what normix needs: $\log K_\nu(z)$ with full autodiff for both $\nu$ and $z$.

7. **Validate `logbesselk` extensively** before committing to it. Build a test suite that compares `logbesselk.jax.log_bessel_k(v, z)` against `scipy.special.kve`-based `log_kv(v, z)` for the parameter ranges encountered in normix:
   - $p \in [-10, 10]$, $z \in [10^{-8}, 10^4]$ for typical GIG parameters
   - Degenerate cases: $z \to 0$, $|p| \to \infty$
   - Gradient accuracy: compare `jax.grad(log_bessel_k)` against normix's `log_kv_derivative_v` and `log_kv_derivative_z`

8. **Consider vendoring `logbesselk`** (copying its source into the project) to avoid a fragile external dependency. The package is small and MIT-licensed.

### 10.3 Optimization Recommendations

9. **For $\eta \to \theta$ in simple distributions** (Gamma, InverseGamma, InverseGaussian): use Newton's method (as efax does). These are well-conditioned and Newton converges quickly.

10. **For $\eta \to \theta$ in GIG**: use `jaxopt.LBFGSB` with box constraints, mirroring the current SciPy approach. Newton's method (as used by efax) does not handle the GIG well because:
    - The parameter space has boundaries ($\theta_2 \le 0$, $\theta_3 \le 0$)
    - The Fisher information can be ill-conditioned near degenerate limits
    - Multi-start is important for robustness

    However, since `logbesselk` gives exact gradients (unlike normix's finite differences), the L-BFGS-B convergence should actually improve.

11. **Alternative for GIG $\eta \to \theta$**: Use a custom Newton method with projection onto the constraint set and Levenberg-Marquardt damping. Since JAX provides exact Hessians via `jax.hessian(log_partition)`, this could be more efficient than L-BFGS-B. Worth experimenting.

### 10.4 Development Recommendations

12. **Start with Phase 0** (validation infrastructure) to derisk the Bessel function dependency before investing in the full port.

13. **Keep the NumPy/SciPy version** as the reference implementation. Do not delete it. The JAX version should be a separate package or subpackage.

14. **Float64 everywhere**: Always call `jax.config.update("jax_enable_x64", True)` at package import. Statistical computing requires float64 precision, especially for Bessel function evaluations and log-likelihood computations.

15. **Test-driven migration**: For each distribution, write the cross-validation test first (comparing JAX output to NumPy output), then implement the JAX version until the test passes.

### 10.5 Dependencies for JAX Version

```
jax >= 0.4.20
jaxlib >= 0.4.20
equinox >= 0.11
jaxopt >= 0.8
logbesselk >= 3.4.0
```

Optional (for extended functionality):
```
tensorflow-probability[jax]   # GIG sampling, additional distributions
distrax                       # Reference distribution API
optax                         # Alternative optimizers for MLE
```

---

## Appendix A: API Comparison

### Current normix (NumPy/SciPy)

```python
from normix import GH

# Fit
model = GH(d=2)
model.fit(X, regularization='det_sigma_one', max_iter=200)

# Use (mutable object, cached properties)
ll = model.logpdf(X_test)
params = model.classical_params        # cached_property
theta = model.natural_params            # cached_property
eta = model.expectation_params          # cached_property
X_samples = model.rvs(size=1000)
fisher = model.joint.fisher_information()

# Joint distribution
joint_ll = model.joint.logpdf(X, Y)
```

### Proposed JAX normix

```python
from normix_jax import GH
import jax

# Fit (returns new immutable object)
model = GH.fit(X, key=jax.random.key(0), regularization='det_sigma_one', max_iter=200)

# Use (immutable object, methods compute on demand)
ll = jax.vmap(model.log_prob)(X_test)           # vectorized
params = model.classical_params()                 # method call, not property
theta = model.natural_params()                    # method call
eta = model.expectation_params()                  # via jax.grad
X_samples = model.sample(jax.random.key(1), shape=(1000,))
fisher = model.joint().fisher_information()       # via jax.hessian

# Joint distribution
joint = model.joint()                             # returns ExponentialFamily
joint_ll = jax.vmap(joint.log_prob)(X, Y)

# JIT everything
fast_log_prob = jax.jit(jax.vmap(model.log_prob))
ll = fast_log_prob(X_test)

# Gradient of log-likelihood w.r.t. parameters
def nll(model, X):
    return -jnp.mean(jax.vmap(model.log_prob)(X))

grad_nll = jax.grad(nll)(model, X)  # gradient w.r.t. all model parameters
```

---

## Appendix B: GIG Log Partition — normix vs JAX

### Current normix (handcoded derivatives)

```python
# normix/utils/bessel.py — 264 lines of hand-coded Bessel infrastructure
def log_kv(v, z):
    """Uses scipy.special.kve + manual log wrapper + asymptotic fallbacks."""
    ...

def log_kv_derivative_v(v, z, eps=1e-6):
    """Central finite differences — O(ε²) accuracy."""
    return (log_kv(v + eps, z) - log_kv(v - eps, z)) / (2 * eps)

def log_kv_derivative_z(v, z):
    """Recurrence relation — exact but manually derived."""
    ...

# normix/distributions/univariate/generalized_inverse_gaussian.py
def _natural_to_expectation(self, theta):
    """50+ lines of hand-coded Bessel ratio formulas + degenerate case handling."""
    ...
```

### Proposed JAX version

```python
from logbesselk.jax import log_bessel_k

def log_partition_gig(theta):
    """GIG log partition — 5 lines, fully differentiable."""
    p = theta[0] + 1
    b = jax.nn.softplus(-2 * theta[1])  # ensures b > 0, smooth
    a = jax.nn.softplus(-2 * theta[2])  # ensures a > 0, smooth
    sqrt_ab = jnp.sqrt(a * b)
    return jnp.log(2.0) + log_bessel_k(p, sqrt_ab) + (p / 2) * (jnp.log(b) - jnp.log(a))

# Expectation parameters: AUTOMATIC
eta = jax.grad(log_partition_gig)(theta)

# Fisher information: AUTOMATIC
fisher = jax.hessian(log_partition_gig)(theta)

# All Bessel derivatives: handled by logbesselk + JAX autodiff
```

The JAX version is **dramatically simpler** — the entire 264-line `bessel.py` and 50+ lines of hand-coded derivatives are replaced by 5 lines plus `jax.grad`.

---

## Appendix C: Summary Decision Matrix

| Design Decision | Recommendation | Rationale |
|---|---|---|
| Base class | `eqx.Module` | Pytree-native, immutable, supports methods |
| Parametrization style | Single class, multiple constructors | Simpler than efax's multi-class approach; better for EM |
| Bessel functions | `logbesselk` | Only library with $\partial/\partial\nu$ support in JAX |
| $\eta \to \theta$ (simple dists) | Newton's method | Well-conditioned, fast |
| $\eta \to \theta$ (GIG) | `jaxopt.LBFGSB` | Handles constraints, multi-start |
| EM loop | Python loop + JIT step initially; `lax.while_loop` later | Pragmatic: easy debugging first, performance later |
| EM state | Immutable `eqx.Module` or `NamedTuple` | Functional pattern, compatible with `lax.while_loop` |
| Covariance | Cholesky-first (`L_Sigma`) | Same as current normix; direct JAX support |
| Float precision | float64 always | Required for statistical computing |
| Sampling (GIG) | TFP JAX substrate or custom | No `scipy.stats.geninvgauss` in JAX |
| Multi-start init | `jax.vmap` over starting points | Natural parallelism |
| Package structure | Separate `normix_jax` package | Keep NumPy version as reference |
