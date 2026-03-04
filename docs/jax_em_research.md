# JAX Ecosystem for EM Algorithms: Research Summary

> Research date: 2026-03-04
> Context: Exploring JAX-based design for `normix` — a Python package implementing
> Generalized Hyperbolic distributions and related distributions as exponential families.

---

## Table of Contents

1. [EM Algorithm Implementations in JAX](#1-em-algorithm-implementations-in-jax)
2. [Key JAX Features for EM](#2-key-jax-features-for-em)
3. [Iterative EM with `lax.while_loop` and `lax.scan`](#3-iterative-em-with-laxwhile_loop-and-laxscan)
4. [Pytrees for Distribution Parameters](#4-pytrees-for-distribution-parameters)
5. [Existing JAX Mixture Model Libraries](#5-existing-jax-mixture-model-libraries)
6. [Convergence Checking in JAX's Functional Paradigm](#6-convergence-checking-in-jaxs-functional-paradigm)
7. [JAX-Compatible Random Number Generation](#7-jax-compatible-random-number-generation)
8. [Design Patterns in JAX ML Libraries](#8-design-patterns-in-jax-ml-libraries)
9. [Equinox Model-with-Parameters Pattern](#9-equinox-model-with-parameters-pattern)
10. [Distrax API Design](#10-distrax-api-design)
11. [TensorFlow Probability JAX Substrate](#11-tensorflow-probability-jax-substrate)
12. [Design Recommendations for normix](#12-design-recommendations-for-normix)

---

## 1. EM Algorithm Implementations in JAX

### General Pattern

EM in JAX follows a **pure-functional** style: the entire algorithm state (parameters,
log-likelihood, iteration counter) is packed into a single pytree that flows through
each iteration as an immutable value.

```python
import jax
import jax.numpy as jnp

def em_step(state, X):
    """One E-step + M-step.  Pure function: no side effects."""
    params, ll_prev = state

    # --- E-step: compute responsibilities ---
    log_resp = e_step(params, X)                    # (N, K)
    resp = jax.nn.softmax(log_resp, axis=-1)

    # --- M-step: update parameters ---
    params_new = m_step(resp, X)

    # --- Log-likelihood for convergence ---
    ll_new = compute_log_likelihood(params_new, X)

    return (params_new, ll_new)
```

### Key Observations

- **No mutation**: each step returns a *new* parameter pytree.
- **JIT-friendly**: pure functions compile to XLA without Python-side effects.
- **vmap-compatible**: the same `em_step` can be vmapped over multiple initializations.

---

## 2. Key JAX Features for EM

### 2.1 `jax.jit` — JIT Compilation

Compiles Python functions to optimized XLA code. The entire EM body (E-step + M-step)
can be compiled as a single fused computation:

```python
@jax.jit
def em_step(params, X):
    ...
```

**Caveats for EM:**
- All array shapes must be statically known at compile time.
- Python control flow (if/else on traced values) must use `jax.lax.cond`.
- Recompilation occurs whenever static arguments change (e.g., `n_components`).

### 2.2 `jax.vmap` — Vectorized Map

Automatically batches a function over an extra leading axis. Critical EM use cases:

| Use Case | Code Pattern |
|---|---|
| Batch log-prob over N samples | `vmap(log_prob_single, in_axes=(0, None))` |
| Multiple random restarts | `vmap(run_em, in_axes=(0, None))(init_params_batch, X)` |
| Component-wise sufficient statistics | `vmap(compute_stats, in_axes=(None, 0))` |

```python
# Evaluate log_prob for N samples against K components
# single_log_prob: (x: (D,), params_k: Params) -> scalar
batch_log_prob = jax.vmap(
    jax.vmap(single_log_prob, in_axes=(None, 0)),  # over K components
    in_axes=(0, None)                               # over N samples
)   # result: (N, K)
```

### 2.3 `jax.lax.scan` — Sequential Iteration with Fixed Count

Replaces Python for-loops while carrying state. Ideal when the number of EM iterations
is predetermined:

```python
def scan_body(carry, _):
    params, ll = carry
    params_new, ll_new = em_step((params, ll), X)
    return (params_new, ll_new), ll_new   # (carry, output)

(final_params, final_ll), ll_trace = jax.lax.scan(
    scan_body,
    init=(init_params, -jnp.inf),
    xs=jnp.arange(max_iter)     # dummy xs; length determines iteration count
)
# ll_trace: (max_iter,) — log-likelihood at each step
```

**Advantages over Python loop:**
- Compiles to a single XLA `While` op (no per-iteration recompilation).
- Supports reverse-mode differentiation (backprop through iterations).
- Returns stacked outputs for diagnostics.

### 2.4 `jax.lax.while_loop` — Dynamic Convergence

Loops until a condition becomes False. Essential for "run until converged":

```python
def cond_fn(state):
    i, params, ll, ll_prev = state
    not_converged = jnp.abs(ll - ll_prev) > tol
    not_exceeded = i < max_iter
    return not_converged & not_exceeded   # must use & not `and`

def body_fn(state):
    i, params, ll, ll_prev = state
    params_new, ll_new = em_step((params, ll), X)
    return (i + 1, params_new, ll_new, ll)

init_state = (0, init_params, -jnp.inf, -jnp.inf)
final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
```

**Limitations:**
- **Not reverse-mode differentiable** (XLA requires static memory bounds).
- All carry values must maintain fixed shapes/dtypes across iterations.
- Cannot use Python `and`/`or` on traced values — use `&` / `|`.

---

## 3. Iterative EM with `lax.while_loop` and `lax.scan`

### Choosing Between Them

| Feature | `lax.scan` | `lax.while_loop` |
|---|---|---|
| Iteration count | Fixed (known at compile) | Dynamic (data-dependent) |
| Reverse-mode diff | Yes | No |
| Convergence trace | Returns stacked outputs | Only final state |
| Typical EM use | Fixed iterations + diagnostics | Run-until-converged |

### Hybrid Pattern (Recommended for EM)

Use `lax.scan` with an internal convergence flag to get the best of both worlds:

```python
def scan_em_body(carry, _):
    params, ll_prev, converged = carry

    # Only compute if not converged (still runs but result is discarded)
    params_new, ll_new = em_step(params, X)

    # Check convergence
    improvement = jnp.abs(ll_new - ll_prev)
    newly_converged = converged | (improvement < tol)

    # If converged, keep old params (effectively a no-op)
    params_out = jax.tree.map(
        lambda new, old: jnp.where(newly_converged, old, new),
        params_new, params
    )
    ll_out = jnp.where(newly_converged, ll_prev, ll_new)

    return (params_out, ll_out, newly_converged), ll_out

(final_params, final_ll, _), ll_trace = jax.lax.scan(
    scan_em_body,
    init=(init_params, -jnp.inf, False),
    xs=None, length=max_iter
)
```

This always runs `max_iter` iterations (required for `scan`), but freezes parameters
once convergence is reached. The full log-likelihood trace is available for diagnostics.

### `lax.while_loop` for True Early Stopping

When wasted computation matters (large data, expensive M-step):

```python
EMState = namedtuple('EMState', ['iteration', 'params', 'll', 'll_prev'])

def cond_fn(state: EMState) -> bool:
    not_converged = jnp.abs(state.ll - state.ll_prev) > tol
    not_exceeded = state.iteration < max_iter
    return not_converged & not_exceeded

def body_fn(state: EMState) -> EMState:
    new_params, new_ll = em_step(state.params, X)
    return EMState(state.iteration + 1, new_params, new_ll, state.ll)

result = jax.lax.while_loop(cond_fn, body_fn, EMState(0, init_params, -jnp.inf, -jnp.inf))
```

---

## 4. Pytrees for Distribution Parameters

### What is a Pytree?

A pytree is any nested structure of Python containers (dicts, lists, tuples, dataclasses)
whose leaves are JAX arrays. JAX can traverse, map, and transform pytrees automatically.

### Representing Distribution Parameters

**Option A: NamedTuple (lightweight, immutable)**
```python
from typing import NamedTuple

class GMMParams(NamedTuple):
    weights: jnp.ndarray      # (K,)
    means: jnp.ndarray        # (K, D)
    chol_covs: jnp.ndarray    # (K, D, D)  — lower Cholesky factors
```

**Option B: Registered Dataclass (more flexible)**
```python
from dataclasses import dataclass
from functools import partial
import jax

@partial(jax.tree_util.register_dataclass,
         data_fields=['weights', 'means', 'chol_covs'],
         meta_fields=['n_components'])
@dataclass(frozen=True)
class GMMParams:
    weights: jnp.ndarray
    means: jnp.ndarray
    chol_covs: jnp.ndarray
    n_components: int         # static metadata, not a JAX array
```

**Option C: Equinox Module (richest, with methods)**
```python
import equinox as eqx

class GMMParams(eqx.Module):
    weights: jax.Array        # (K,)
    means: jax.Array          # (K, D)
    chol_covs: jax.Array      # (K, D, D)

    @property
    def n_components(self):
        return self.weights.shape[0]

    def log_prob(self, x):
        ...
```

### Pytree Operations on Parameters

```python
# Functional update: create new params with modified means
new_params = jax.tree.map(lambda p: p * 0.99, params)  # scale all arrays

# Selective update with Equinox
new_params = eqx.tree_at(lambda p: p.means, params, new_means)

# Gradient through parameters
grads = jax.grad(loss_fn)(params, X)   # grads has same pytree structure
```

### Mapping to normix's Current Design

normix currently uses named attributes (`_mu`, `_gamma`, `_L_Sigma`, etc.) as the
single source of truth. In JAX, these would become pytree leaves:

| normix (NumPy) | JAX Pytree Equivalent |
|---|---|
| `self._mu = np.array(...)` | `mu: jax.Array` field in a Module/dataclass |
| `self._L_Sigma = np.array(...)` | `L_Sigma: jax.Array` field |
| `self._alpha, self._beta` (subordinator) | Nested sub-pytree or flat fields |
| `_invalidate_cache()` | Not needed — immutable pytrees have no cache |

---

## 5. Existing JAX Mixture Model Libraries

### 5.1 GMMX (adonath/gmmx)

**Most relevant library.** Minimal JAX Gaussian Mixture Models with EM.

- **Repository:** https://github.com/adonath/gmmx
- **Latest:** v0.7 (July 2025), BSD-3 license
- **Key design decisions:**
  - `GaussianMixtureModelJax` is a registered JAX dataclass (pytree).
  - Internal arrays use a consistent 4D shape `(batch, components, features, features_covar)`.
  - `Axis` enum maps semantic names to integer dimensions.
  - Separate covariance-type classes (full, tied, diag, spherical) avoid branching.
  - `EMFitter` is a separate object from the model — separation of algorithm and state.
  - Avoids Python loops; JAX compiler handles iteration.

```python
from gmmx import GaussianMixtureModelJax, EMFitter

gmm = GaussianMixtureModelJax.create(n_components=16, n_features=32)
x = gmm.sample(n_samples=10_000)
em_fitter = EMFitter(tol=1e-3, max_iter=100)
gmm_fitted = em_fitter.fit(x=x, gmm=gmm)
```

### 5.2 OTT-JAX (ott-jax)

Optimal Transport Tools for JAX, includes GMM fitting.

- **Repository:** https://github.com/ott-jax/ott
- `fit_model_em(gmm, points, point_weights, steps, jit=True)` — runs a fixed number of EM steps.
- Integrates with Wasserstein distance for coupled GMM fitting.
- `GaussianMixture` is a pytree-registered class.

### 5.3 DEM (k-cybulski/dem)

Dynamic Expectation Maximization using JAX autodiff.

### 5.4 GenJAX

Probabilistic programming framework with mixture model support via a `mix` combinator.

---

## 6. Convergence Checking in JAX's Functional Paradigm

### The Challenge

JAX's functional/compiled paradigm forbids:
- Python `print()` inside JIT-compiled code (use `jax.debug.print` instead).
- Python `if` on traced values (use `jax.lax.cond`).
- Early `return` from loops (use convergence flags in carry state).
- Mutable state for tracking convergence history.

### Pattern 1: `while_loop` with Convergence State

```python
@jax.jit
def fit_em(init_params, X, tol=1e-6, max_iter=200):
    def cond_fn(state):
        i, _, ll, ll_prev = state
        return (i < max_iter) & (jnp.abs(ll - ll_prev) > tol)

    def body_fn(state):
        i, params, ll, ll_prev = state
        params_new = m_step(e_step(params, X), X)
        ll_new = log_likelihood(params_new, X)
        return (i + 1, params_new, ll_new, ll)

    init = (0, init_params, -jnp.inf, -jnp.inf)
    _, final_params, final_ll, _ = jax.lax.while_loop(cond_fn, body_fn, init)
    return final_params, final_ll
```

### Pattern 2: `scan` with Convergence Flag (for diagnostics)

```python
@jax.jit
def fit_em_with_trace(init_params, X, max_iter=200, tol=1e-6):
    def step(carry, _):
        params, ll_prev, done = carry
        params_new = m_step(e_step(params, X), X)
        ll_new = log_likelihood(params_new, X)

        newly_done = done | (jnp.abs(ll_new - ll_prev) < tol)
        # Freeze parameters once converged
        params_out = jax.tree.map(
            lambda a, b: jnp.where(newly_done, b, a), params_new, params
        )
        ll_out = jnp.where(newly_done, ll_prev, ll_new)
        return (params_out, ll_out, newly_done), ll_out

    (final_params, final_ll, _), ll_history = jax.lax.scan(
        step, (init_params, -jnp.inf, jnp.bool_(False)), None, length=max_iter
    )
    return final_params, final_ll, ll_history
```

### Pattern 3: Equinox Bounded While Loop (Differentiable)

Equinox provides `eqx.internal.while_loop` with bounded iteration counts, enabling
reverse-mode differentiation through the loop:

```python
import equinox as eqx

final_state = eqx.internal.while_loop(
    cond_fn, body_fn, init_state, max_steps=max_iter, kind="checkpointed"
)
```

### Convergence Monitoring (Outside JIT)

For interactive use, run EM in a Python loop and JIT only the step:

```python
@jax.jit
def em_step_jit(params, X):
    resp = e_step(params, X)
    new_params = m_step(resp, X)
    ll = log_likelihood(new_params, X)
    return new_params, ll

params = init_params
for i in range(max_iter):
    params, ll = em_step_jit(params, X)
    print(f"Iter {i}: ll={ll:.6f}")
    if abs(ll - ll_prev) < tol:
        break
    ll_prev = ll
```

This sacrifices some performance (Python loop overhead, dispatch per iteration) but
enables `print`, early stopping, and arbitrary Python logic.

---

## 7. JAX-Compatible Random Number Generation

### PRNG Design

JAX uses **explicit, functional PRNG keys** — no global random state.

```python
key = jax.random.key(42)           # Create initial key
key, subkey = jax.random.split(key) # Split for consumption

# Sample from distributions
x = jax.random.normal(subkey, shape=(N, D))
```

### Key Splitting for EM Initialization

```python
def initialize_gmm(key, K, D, X):
    k1, k2, k3 = jax.random.split(key, 3)

    # Random means from data points
    indices = jax.random.choice(k1, X.shape[0], shape=(K,), replace=False)
    means = X[indices]

    # Random covariances (identity + noise)
    chol_covs = jnp.tile(jnp.eye(D), (K, 1, 1))

    # Uniform weights
    weights = jnp.ones(K) / K

    return GMMParams(weights=weights, means=means, chol_covs=chol_covs)
```

### Sampling from Mixture Models

```python
def sample_gmm(key, params, n_samples):
    k1, k2 = jax.random.split(key)

    # Sample component assignments
    components = jax.random.categorical(k1, jnp.log(params.weights), shape=(n_samples,))

    # Sample from each component
    means = params.means[components]           # (N, D)
    chols = params.chol_covs[components]       # (N, D, D)
    z = jax.random.normal(k2, shape=(n_samples, params.means.shape[-1]))
    x = means + jnp.einsum('nij,nj->ni', chols, z)

    return x
```

### Key Management Patterns

| Pattern | Code |
|---|---|
| Sequential splitting | `key, subkey = jax.random.split(key)` |
| Multiple keys at once | `keys = jax.random.split(key, n)` |
| Inside `scan` | Pass key as carry, split at each step |
| Inside `vmap` | Split into batch of keys, vmap over them |
| Reproducible restarts | `keys = jax.random.split(key, n_restarts)` then `vmap` |

---

## 8. Design Patterns in JAX ML Libraries

### 8.1 Flax: Explicit State Management

Flax separates model definition from parameters/state:

```python
import flax.linen as nn

class MLP(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        return nn.relu(x)

model = MLP(features=64)
params = model.init(key, x_dummy)          # returns a frozen dict
y = model.apply(params, x)                 # apply is a pure function
```

**Key pattern:** `model.apply(params, x)` — the model object carries the *structure*
(architecture), while `params` carries the *values* (weights). This is analogous to
separating the distribution class from its parameter values.

### 8.2 Equinox: Models ARE Pytrees

Equinox unifies structure and parameters into a single pytree:

```python
import equinox as eqx

class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(2, 64, key=k1),
                       eqx.nn.Linear(64, 1, key=k2)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

model = MLP(key)                           # model IS the pytree of params
grads = jax.grad(loss)(model, x, y)        # grad w.r.t. the model itself
new_model = eqx.apply_updates(model, grads)
```

**Key insight:** `eqx.Module` is a frozen dataclass registered as a JAX pytree. Fields
that are JAX arrays become pytree leaves (differentiable); other fields are auxiliary
data (static).

### 8.3 JAXopt: Solver Objects

JAXopt uses solver objects that encapsulate algorithm hyperparameters and provide
`init_state` / `update` / `run` methods:

```python
import jaxopt

solver = jaxopt.LBFGS(fun=loss_fn, maxiter=100)
params, state = solver.run(init_params, X=X)
```

**Relevant pattern for EM:** The solver (algorithm) is separate from the model
(parameters). The solver carries hyperparameters (tol, max_iter); the run method
takes initial params and returns final params.

### 8.4 Summary of Approaches

| Library | Model | Parameters | Update Pattern |
|---|---|---|---|
| Flax | `nn.Module` (template) | Separate `params` dict | `model.apply(params, x)` |
| Equinox | `eqx.Module` (pytree) | Inside the module | `eqx.apply_updates(model, grads)` |
| JAXopt | `Solver` object | Separate pytree | `solver.run(init_params)` |
| GMMX | `GMMJax` (dataclass/pytree) | Inside the dataclass | `EMFitter.fit(x, gmm)` returns new gmm |

---

## 9. Equinox Model-with-Parameters Pattern

### Core Idea

An `eqx.Module` is simultaneously:
- A **Python class** with methods (`.log_prob()`, `.sample()`, etc.)
- A **JAX pytree** whose array fields are differentiable/traceable leaves.
- **Immutable** — updates create new instances via `eqx.tree_at`.

### Distribution as Equinox Module

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class MultivariateNormal(eqx.Module):
    mu: jax.Array              # (D,)
    L_Sigma: jax.Array         # (D, D) lower Cholesky of covariance

    def log_prob(self, x):
        D = self.mu.shape[0]
        diff = x - self.mu
        solve = jax.scipy.linalg.solve_triangular(self.L_Sigma, diff, lower=True)
        log_det = jnp.sum(jnp.log(jnp.diag(self.L_Sigma)))
        return -0.5 * (D * jnp.log(2 * jnp.pi) + 2 * log_det + jnp.dot(solve, solve))

    def sample(self, key, n=1):
        z = jax.random.normal(key, shape=(n, self.mu.shape[0]))
        return self.mu + z @ self.L_Sigma.T

    @classmethod
    def from_covariance(cls, mu, Sigma):
        L = jnp.linalg.cholesky(Sigma)
        return cls(mu=mu, L_Sigma=L)
```

### Immutable Parameter Updates

```python
# Update mean only
new_dist = eqx.tree_at(lambda d: d.mu, dist, new_mu)

# Update from gradient
grads = jax.grad(lambda d: d.log_prob(x_obs))(dist)
new_dist = eqx.apply_updates(dist, grads)
```

### Filtering Static vs Dynamic Fields

```python
class GHDistribution(eqx.Module):
    # Dynamic (JAX arrays, differentiable)
    mu: jax.Array
    gamma: jax.Array
    L_Sigma: jax.Array
    p: jax.Array
    a: jax.Array
    b: jax.Array

    # Static (not traced by JAX)
    dim: int = eqx.field(static=True)

    def log_prob(self, x, y):
        ...
```

### Why Equinox Fits normix Well

| normix concept | Equinox equivalent |
|---|---|
| Frozen `@dataclass` params | `eqx.Module` with array fields |
| `_invalidate_cache()` | Not needed — immutable, no cache |
| `cached_property` | `jax.jit` handles caching at compilation level |
| `_set_from_classical` / `_set_from_natural` | Class methods / `@staticmethod` constructors |
| Named attrs (`_mu`, `_L_Sigma`) | Public fields (`mu`, `L_Sigma`) |
| `self._joint` delegation | Nested module field |

---

## 10. Distrax API Design

### Architecture

Distrax (DeepMind) is a lightweight, JAX-native probability distribution library.
It reimplements a subset of TensorFlow Probability with an emphasis on:

1. **Readability** — self-contained implementations close to the math.
2. **Extensibility** — easy to define custom distributions.
3. **TFP Compatibility** — cross-compatible API with TFP.

### Distribution Base Class

```python
class Distribution(abc.ABC):
    """Abstract base for all distributions."""

    @abc.abstractmethod
    def _sample_n(self, key, n):
        """Draw n samples."""

    @abc.abstractmethod
    def _log_prob(self, value):
        """Compute log probability."""

    def sample(self, *, seed, sample_shape=()):
        """Public sampling API."""

    def log_prob(self, value):
        """Public log_prob API."""

    def sample_and_log_prob(self, *, seed, sample_shape=()):
        """Efficient joint sample + log_prob (can be overridden)."""

    @property
    def event_shape(self):
        """Shape of a single sample."""

    @property
    def batch_shape(self):
        """Shape of the batch of distributions."""
```

### Key Design Decisions

1. **Distributions are pytrees** — they work with `jax.jit`, `jax.vmap`, `jax.grad`.
2. **`sample_and_log_prob`** — single method for joint sampling and density evaluation,
   more efficient for some distributions (avoids redundant computation).
3. **Bijectors** — invertible transformations with log-det-Jacobian, composable.
4. **`Transformed` distribution** — wraps a base distribution with a bijector chain.
5. **`MixtureSameFamily`** — parameterized by a categorical + components distribution.

### MixtureSameFamily Pattern

```python
import distrax

# Define mixture of 3 Normals
mixture = distrax.MixtureSameFamily(
    mixture_distribution=distrax.Categorical(logits=jnp.zeros(3)),
    components_distribution=distrax.Normal(
        loc=jnp.array([-1.0, 0.0, 1.0]),
        scale=jnp.array([0.5, 1.0, 0.5])
    )
)

samples = mixture.sample(seed=jax.random.key(0), sample_shape=(1000,))
log_probs = mixture.log_prob(samples)
```

### Relevance to normix

- Distrax shows that distribution objects **can** be JAX pytrees with methods.
- The `_log_prob` / `_sample_n` abstract interface is clean and minimal.
- `sample_and_log_prob` is relevant for MCMC/variational inference extensions.
- Bijectors could represent the classical ↔ natural parameter transformations.

---

## 11. TensorFlow Probability JAX Substrate

### Overview

TFP provides a full-featured JAX backend via `tfp.substrates.jax`:

```python
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
```

### Distribution Base Class

The TFP `Distribution` base class is heavyweight but comprehensive:

- **`_parameter_properties`** — declares all parameters, their shapes, constraints, and
  bijectors for unconstrained optimization.
- **Batch/event shape system** — rigorous broadcasting semantics.
- **`_log_prob`, `_sample_n`** — abstract methods subclasses implement.
- **Automatic parameter validation** with `validate_args`.
- **Slicing support** — `dist[2:5]` returns a distribution over a batch slice.

### Key Features Relevant to normix

1. **`MultivariateNormalTriL`** — accepts Cholesky factor directly:
   ```python
   dist = tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)
   ```

2. **`MixtureSameFamily`** — mixture model:
   ```python
   mix = tfd.MixtureSameFamily(
       mixture_distribution=tfd.Categorical(probs=weights),
       components_distribution=tfd.MultivariateNormalTriL(
           loc=means, scale_tril=chol_covs
       )
   )
   ```

3. **`JointDistributionCoroutine`** — for joint distributions like $f(x, y)$:
   ```python
   @tfd.JointDistributionCoroutine
   def model():
       y = yield tfd.InverseGaussian(loc=delta, concentration=eta)
       x = yield tfd.MultivariateNormalTriL(
           loc=mu + gamma * y,
           scale_tril=jnp.sqrt(y) * L_Sigma
       )
   ```

4. **`GeneralizedInverseGaussian`** — available in TFP:
   ```python
   gig = tfd.GeneralizedInverseGaussian(loc=p, concentration=a, scale=b)
   ```

### Design Lessons

- TFP's `_parameter_properties` system is powerful but complex. For normix, a simpler
  approach (Equinox Module or registered dataclass) is likely sufficient.
- The `MultivariateNormalTriL` convention (Cholesky-first) matches normix's existing design.
- `JointDistributionCoroutine` is elegant for expressing conditional structure.

---

## 12. Design Recommendations for normix

### Architecture Decision: Equinox Module Approach

Based on the research, the **Equinox Module pattern** is the best fit for normix because:

1. **Models are pytrees** — seamless `jit`/`vmap`/`grad` compatibility.
2. **Immutable by default** — matches normix's frozen dataclass pattern.
3. **Methods on pytrees** — `.log_prob()`, `.sample()`, `.fit()` live on the object.
4. **Nested modules** — `JointNormalMixture` can contain a subordinator module.
5. **No cache invalidation needed** — immutable objects, JIT handles performance.
6. **Static fields** — `eqx.field(static=True)` for non-array config (dimension, etc.).

### Proposed Class Hierarchy

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class ExponentialFamily(eqx.Module):
    """Abstract base. Subclasses store parameters as Module fields."""

    @abc.abstractmethod
    def sufficient_statistics(self, x):
        ...

    @abc.abstractmethod
    def log_partition(self):
        ...

    @abc.abstractmethod
    def log_base_measure(self, x):
        ...

    def log_prob(self, x):
        theta = self.natural_params()
        return (jnp.dot(theta, self.sufficient_statistics(x))
                - self.log_partition()
                + self.log_base_measure(x))


class MultivariateNormal(ExponentialFamily):
    mu: jax.Array           # (D,)
    L_Sigma: jax.Array      # (D, D) lower Cholesky

    dim: int = eqx.field(static=True)

    def natural_params(self):
        ...

    @classmethod
    def from_natural(cls, theta):
        ...

    @classmethod
    def from_classical(cls, *, mu, Sigma):
        L = jnp.linalg.cholesky(Sigma)
        return cls(mu=mu, L_Sigma=L, dim=mu.shape[0])


class JointNormalInverseGaussian(ExponentialFamily):
    mu: jax.Array
    gamma: jax.Array
    L_Sigma: jax.Array
    delta: jax.Array       # subordinator param
    eta: jax.Array         # subordinator param

    dim: int = eqx.field(static=True)

    def log_prob(self, x, y):
        ...

    def e_step(self, X):
        """Compute E[Y|X] and E[log Y|X] etc."""
        ...

    def m_step(self, X, expected_stats):
        """Return a new JointNormalInverseGaussian with updated params."""
        ...
```

### EM Algorithm Design

Separate the algorithm from the model (JAXopt / GMMX pattern):

```python
class EMFitter(eqx.Module):
    max_iter: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)

    def fit(self, model, X):
        """Run EM and return a new model with fitted parameters."""

        @jax.jit
        def em_step(model):
            stats = model.e_step(X)
            return model.m_step(X, stats)

        @jax.jit
        def log_likelihood(model):
            return jnp.sum(model.log_prob(X))

        # Python loop for flexibility (print, early stop)
        ll_prev = -jnp.inf
        for i in range(self.max_iter):
            model = em_step(model)
            ll = log_likelihood(model)
            if jnp.abs(ll - ll_prev) < self.tol:
                break
            ll_prev = ll

        return model

    def fit_jit(self, model, X):
        """Fully JIT-compiled EM using lax.while_loop."""

        def cond_fn(state):
            i, _, ll, ll_prev = state
            return (i < self.max_iter) & (jnp.abs(ll - ll_prev) > self.tol)

        def body_fn(state):
            i, model, ll, ll_prev = state
            stats = model.e_step(X)
            new_model = model.m_step(X, stats)
            new_ll = jnp.sum(new_model.log_prob(X))
            return (i + 1, new_model, new_ll, ll)

        init = (0, model, -jnp.inf, -jnp.inf)
        _, final_model, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)
        return final_model
```

### Parameter Constructors (Replacing Setters)

In JAX, immutable objects replace the setter pattern. Use class methods:

```python
class InverseGaussian(ExponentialFamily):
    mu: jax.Array
    lam: jax.Array   # _lambda in normix (reserved keyword)

    @classmethod
    def from_classical(cls, *, mu, lam):
        return cls(mu=jnp.asarray(mu), lam=jnp.asarray(lam))

    @classmethod
    def from_natural(cls, theta):
        theta1, theta2 = theta[0], theta[1]
        lam = -2 * theta2
        mu = jnp.sqrt(lam / (-2 * theta1))
        return cls(mu=mu, lam=lam)

    @classmethod
    def from_expectation(cls, eta):
        # Newton solve for natural params, then construct
        theta = _expectation_to_natural(eta)
        return cls.from_natural(theta)
```

### Cholesky Everywhere (Already normix's Pattern)

JAX's `jnp.linalg.cholesky` + `jax.scipy.linalg.cho_solve` / `solve_triangular`
map directly to normix's existing Cholesky-first design:

```python
def log_prob_normal(x, mu, L_Sigma):
    D = mu.shape[0]
    diff = x - mu
    v = jax.scipy.linalg.solve_triangular(L_Sigma, diff, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(L_Sigma)))
    return -0.5 * (D * jnp.log(2 * jnp.pi) + 2 * log_det + jnp.dot(v, v))
```

**Caveat:** `jnp.linalg.cholesky` returns NaN (not an error) for non-positive-definite
input. Add jitter for numerical safety:

```python
def safe_cholesky(A, jitter=1e-6):
    return jnp.linalg.cholesky(A + jitter * jnp.eye(A.shape[-1]))
```

### Multiple Random Restarts with `vmap`

```python
def fit_with_restarts(X, n_restarts=10, key=jax.random.key(0)):
    keys = jax.random.split(key, n_restarts)

    @jax.vmap
    def init_and_fit(key):
        model = initialize_model(key, X)
        fitter = EMFitter(max_iter=200, tol=1e-6)
        fitted = fitter.fit_jit(model, X)
        ll = jnp.sum(fitted.log_prob(X))
        return fitted, ll

    models, lls = init_and_fit(keys)
    best_idx = jnp.argmax(lls)
    return jax.tree.map(lambda x: x[best_idx], models)
```

### Summary of Recommendations

| Aspect | Recommendation |
|---|---|
| **Base class** | `eqx.Module` (or registered dataclass as fallback) |
| **Parameters** | Public array fields on Module, static config via `eqx.field(static=True)` |
| **Constructors** | `@classmethod` for `from_classical`, `from_natural`, `from_expectation` |
| **EM loop** | Python loop with JIT-compiled step (interactive) or `lax.while_loop` (full JIT) |
| **Convergence** | `while_loop` for early stop; `scan` for diagnostics/differentiability |
| **Random state** | Explicit PRNG keys, split at each level |
| **Covariance** | Cholesky-first (`L_Sigma`), `safe_cholesky` with jitter |
| **Batch ops** | `vmap` for multiple samples, components, restarts |
| **Algorithm separation** | `EMFitter` class separate from distribution model |
| **Dependencies** | Core: `jax`, `jaxlib`. Optional: `equinox`, `distrax`/`tfp` for comparison |

---

## Appendix: Library Versions and Links

| Library | Version | URL |
|---|---|---|
| JAX | ≥0.4.30 | https://github.com/jax-ml/jax |
| Equinox | ≥0.11 | https://github.com/patrick-kidger/equinox |
| Distrax | 0.1.7 | https://github.com/google-deepmind/distrax |
| TFP (JAX) | 0.23+ | https://github.com/tensorflow/probability |
| GMMX | 0.7 | https://github.com/adonath/gmmx |
| OTT-JAX | latest | https://github.com/ott-jax/ott |
| JAXopt | 0.8+ | https://github.com/google/jaxopt |
