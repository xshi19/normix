# JAX normix: Detailed Design Based on FlowJAX, Equinox, and GMMX

**Date:** 2026-03-06

---

## 1. Equinox Fundamentals

### 1.1 What is `eqx.Module`?

Equinox's `Module` is a **frozen dataclass that is automatically a JAX pytree**. When you define a subclass, the metaclass (`_ModuleMeta.__new__`) does two things:

1. Converts the class into a frozen `dataclass` (fields become `__init__` parameters)
2. Registers it as a JAX pytree via `jax.tree_util.register_pytree_with_keys()`

After `__init__` completes, all attribute assignment is blocked (`FrozenInstanceError`). This makes modules inherently immutable — exactly what JAX's functional paradigm needs.

```python
import equinox as eqx
import jax.numpy as jnp

class MyDist(eqx.Module):
    mu: jax.Array
    sigma: jax.Array
    name: str = eqx.field(static=True)  # not a JAX leaf

dist = MyDist(mu=jnp.array(0.0), sigma=jnp.array(1.0), name="normal")
# dist.mu = 1.0  ← raises FrozenInstanceError
```

### 1.2 Leaves vs Static Fields

Every field is either a **pytree leaf** (traced by JAX) or **static metadata** (part of the treedef, must be hashable):

- **Default:** fields are leaves. JAX arrays, nested modules, etc.
- **`eqx.field(static=True)`:** metadata. Strings, ints, bools, tuples of ints. Not traced — changing a static field triggers recompilation.

```python
class GIG(eqx.Module):
    p: jax.Array                          # leaf — traced by JAX
    a: jax.Array                          # leaf
    b: jax.Array                          # leaf
    regularization: str = eqx.field(static=True, default='none')  # static
```

### 1.3 Functional Updates

Since modules are immutable, "updating" creates a new object:

```python
# Change one field
new_dist = eqx.tree_at(lambda d: d.mu, dist, jnp.array(1.0))

# Change multiple fields
new_dist = eqx.tree_at(
    lambda d: (d.mu, d.sigma),
    dist,
    (new_mu, new_sigma)
)
```

This is the mechanism for EM parameter updates: each M-step creates a new model.

### 1.4 Filter Transforms

Equinox provides `filter_jit`, `filter_vmap`, `filter_grad` that automatically handle the leaf/static split:

```python
@eqx.filter_jit
def log_prob(dist, x):
    return dist.log_prob(x)

# Equivalent to: jax.jit(log_prob, static_argnums=(...))
# but Equinox figures out which parts are static automatically
```

`eqx.filter_grad` only differentiates w.r.t. array leaves, not static fields:

```python
# Gradient of loss w.r.t. all array parameters in the model
grads = eqx.filter_grad(loss_fn)(model, data)
```

### 1.5 `eqx.Module` vs `flax.nnx.Module`

| Aspect | `eqx.Module` | `flax.nnx.Module` |
|---|---|---|
| **Mutability** | Immutable (frozen dataclass) | **Mutable** (can set attributes) |
| **Pytree** | Automatic via metaclass | Automatic via `ObjectState` |
| **State updates** | Functional (`eqx.tree_at`) | In-place (`self.param = new_val`) |
| **RNG handling** | Explicit key passing | Built-in `Rngs` object |
| **Use case** | Functional math, distributions | Neural networks with state |
| **Overhead** | Minimal (just dataclass + pytree) | Heavier (graph, state management) |

**For normix:** `eqx.Module` is the clear choice. Distributions are mathematical objects — immutability matches their semantics. There's no mutable state (no batch norm, no running averages). FlowJAX chose Equinox for the same reasons.

---

## 2. FlowJAX `AbstractDistribution` Design

### 2.1 The Unbatched-First Pattern

FlowJAX's key insight: subclasses implement **unbatched** methods that handle a single sample. The base class vectorizes them automatically.

```python
class AbstractDistribution(eqx.Module):
    shape: tuple[int, ...]           # abstract — shape of one sample
    cond_shape: tuple[int, ...] | None  # abstract — None if unconditional

    @abstractmethod
    def _log_prob(self, x, condition=None):
        """Log prob of a SINGLE x. Returns a scalar."""
        ...

    @abstractmethod
    def _sample(self, key, condition=None):
        """Draw a SINGLE sample. Returns array of shape self.shape."""
        ...

    def log_prob(self, x, condition=None):
        """Vectorized log_prob — handles arbitrary batch dimensions."""
        fn = self._vectorize(self._log_prob, ufunc_sig="out")
        return fn(unwrap(self), x, condition)

    def sample(self, key, sample_shape=(), condition=None):
        """Batched sampling."""
        ...
```

The vectorization uses `jnp.vectorize` with ufunc signatures (e.g., `"(d),(c)->()"` for a d-dimensional distribution conditioned on c-dimensional input). This gives numpy-style broadcasting automatically.

### 2.2 Parameter Storage: The Transformed Pattern

Most FlowJAX distributions are `AbstractTransformed = base_dist + bijection`:

```
Normal(loc=0, scale=1)
  └── AbstractLocScaleDistribution
       ├── base_dist: StandardNormal(shape=(,))
       └── bijection: Affine(loc=0.0, scale=Parameterize(softplus, inv_softplus(1.0)))

MultivariateNormal(loc, covariance)
  └── AbstractTransformed
       ├── base_dist: StandardNormal(shape=(d,))
       └── bijection: TriangularAffine(loc, lower_tri=cholesky(covariance))
```

The `Parameterize` wrapper from `paramax` stores the **unconstrained** value and applies a transform on access:

```python
from paramax import Parameterize

# Stores inv_softplus(scale) as the leaf; applies softplus when read
scale_param = Parameterize(jax.nn.softplus, jax.nn.log_expm1(scale))

# unwrap(model) applies all Parameterize transforms in the tree
model = unwrap(model)  # now model.bijection.scale is the actual scale value
```

This enables gradient-based optimization over unconstrained parameters.

### 2.3 Conditional Distributions

FlowJAX handles conditional distributions via `cond_shape`:

```python
class ConditionalNormal(AbstractDistribution):
    shape = (1,)
    cond_shape = (3,)  # conditioned on a 3D variable

    def _log_prob(self, x, condition):
        mu = some_function_of(condition)
        return -0.5 * ((x - mu) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)

    def _sample(self, key, condition):
        mu = some_function_of(condition)
        return mu + sigma * jax.random.normal(key, shape=self.shape)
```

For normalizing flows, the condition modifies the bijection (e.g., `MaskedAutoregressive` with `cond_dim` passes the condition through the neural network that produces transform parameters).

### 2.4 Applicability to normix's $X|Y$

For normix's conditional $X|Y \sim N(\mu + \gamma Y, \Sigma Y)$, FlowJAX's conditional mechanism could work but is **overkill**. The joint distribution has a closed-form exponential family structure — we don't need a bijection framework.

Better approach: implement the joint $f(x, y)$ directly as an `ExponentialFamily`, and compute the conditional expectations $E[g(Y)|X]$ as a method. The conditional distribution $X|Y$ is only used internally during EM (E-step), not as a standalone distribution object.

---

## 3. Design for Classical Statistics in JAX

### 3.1 The Three Parametrizations as JIT-able Methods

The key normix feature missing from all JAX distribution packages is the exponential family structure. Here's how to make it JIT-compatible:

```python
class ExponentialFamily(eqx.Module):

    # --- Core: subclass implements these ---

    @abstractmethod
    def natural_params(self) -> jax.Array:
        """θ from internal state. Pure function, JIT-able."""
        ...

    @abstractmethod
    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        """ψ(θ) as a function of θ. Must be differentiable."""
        ...

    @abstractmethod
    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        """t(x). Pure function."""
        ...

    @abstractmethod
    def log_base_measure(self, x: jax.Array) -> jax.Array:
        """log h(x). Pure function."""
        ...

    # --- Derived: automatic via JAX autodiff ---

    def log_partition(self) -> jax.Array:
        """ψ(θ) at current parameters."""
        return self._log_partition_from_theta(self.natural_params())

    def expectation_params(self) -> jax.Array:
        """η = ∇ψ(θ) — automatic via jax.grad."""
        return jax.grad(self._log_partition_from_theta)(self.natural_params())

    def fisher_information(self) -> jax.Array:
        """I(θ) = ∇²ψ(θ) — automatic via jax.hessian."""
        return jax.hessian(self._log_partition_from_theta)(self.natural_params())

    def log_prob(self, x: jax.Array) -> jax.Array:
        """log p(x|θ) = log h(x) + θᵀt(x) − ψ(θ)."""
        theta = self.natural_params()
        return (self.log_base_measure(x)
                + jnp.dot(self.sufficient_statistics(x), theta)
                - self._log_partition_from_theta(theta))

    # --- Constructors ---

    @classmethod
    def from_natural(cls, theta: jax.Array) -> 'ExponentialFamily':
        ...

    @classmethod
    def from_expectation(cls, eta: jax.Array) -> 'ExponentialFamily':
        """Solve ∇ψ(θ) = η for θ, then construct."""
        ...

    @classmethod
    def fit_mle(cls, X: jax.Array) -> 'ExponentialFamily':
        """MLE: η̂ = mean(t(X)), then from_expectation(η̂)."""
        ...
```

### 3.2 Analytical Overrides with `custom_jvp`

For distributions where $\nabla\psi(\theta)$ has a known analytical form (Gamma, InverseGamma, Normal, etc.), override `expectation_params()` and register a `custom_jvp` on `_log_partition_from_theta` so that the gradient uses the analytical formula:

```python
class Gamma(ExponentialFamily):
    alpha: jax.Array  # shape
    beta: jax.Array   # rate

    def _log_partition_from_theta(self, theta):
        alpha = theta[0] + 1
        beta = -theta[1]
        return jax.scipy.special.gammaln(alpha) - alpha * jnp.log(beta)

    def expectation_params(self):
        """Analytical: η = [ψ(α) - log(β), α/β]."""
        return jnp.array([
            jax.scipy.special.digamma(self.alpha) - jnp.log(self.beta),
            self.alpha / self.beta
        ])
```

For the GIG, the log partition involves the Bessel function and uses the TFP + `custom_jvp` wrapper from the Bessel notebook. `jax.grad` flows through the `custom_jvp` automatically.

### 3.3 $\eta \to \theta$ Conversion

This is an optimization problem: find $\theta^* = \arg\max_\theta [\theta \cdot \eta - \psi(\theta)]$.

```python
import jaxopt

@classmethod
def from_expectation(cls, eta, theta0=None):
    """Convert expectation params to natural params via optimization."""

    def objective(theta):
        return cls._log_partition_from_theta_static(theta) - jnp.dot(theta, eta)

    if theta0 is None:
        theta0 = cls._initial_theta_from_eta(eta)

    solver = jaxopt.LBFGSB(fun=objective, maxiter=500, tol=1e-10)
    result = solver.run(theta0, bounds=cls._theta_bounds())
    return cls.from_natural(result.params)
```

For simple distributions (Gamma, Normal), this has a closed-form inverse. For GIG, L-BFGS-B with bounds is needed.

---

## 4. EM Framework Design

### 4.1 GMMX's Architecture

GMMX separates the model from the fitting algorithm:

- **`GaussianMixtureModelJax`** — an immutable pytree with `weights`, `means`, `covariances`. Has `log_prob`, `predict_proba` (E-step), and `from_responsibilities` (M-step as a classmethod).
- **`EMFitter`** — a dataclass with `max_iter`, `tol`, `reg_covar`. Its `fit(x, gmm_init)` method runs the EM loop via `jax.lax.while_loop`, creating a new model at each iteration.

The EM loop state is a tuple `(data, model, n_iter, log_likelihood, ll_diff)`. The entire fit is one JIT'd call.

### 4.2 Why Separate Fitter from Model?

| Benefit | Explanation |
|---|---|
| **Single responsibility** | Model knows math (log_prob, E-step, M-step). Fitter knows iteration (convergence, restarts, scheduling). |
| **Composability** | Same model can be fit with different strategies (batch EM, online EM, mini-batch, SGD). |
| **JIT-friendliness** | Fitter config (`max_iter`, `tol`) is static. Model params are dynamic. Clean separation for `jax.lax.while_loop`. |
| **Testing** | Test E-step/M-step independently of convergence logic. |

### 4.3 Proposed EM Design for normix

```python
class NormalMixtureModel(eqx.Module):
    """The model — knows math, not iteration."""
    mu: jax.Array
    gamma: jax.Array
    L_Sigma: jax.Array
    # subordinator params defined by subclass

    def e_step(self, X: jax.Array) -> dict:
        """Compute E[Y|X], E[1/Y|X], E[log Y|X] for each observation."""
        ...

    def m_step(self, X: jax.Array, expectations: dict) -> 'NormalMixtureModel':
        """Return a NEW model with updated parameters."""
        ...

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Marginal log f(x)."""
        ...

    def marginal_log_likelihood(self, X: jax.Array) -> jax.Array:
        """Mean log-likelihood over dataset."""
        return jnp.mean(jax.vmap(self.log_prob)(X))
```

```python
class EMFitter(eqx.Module):
    """The algorithm — knows iteration, not math."""
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)
    regularization: str = eqx.field(static=True, default='det_sigma_one')

    def fit(self, model: NormalMixtureModel, X: jax.Array) -> NormalMixtureModel:
        """Run batch EM. Returns fitted model."""
        def cond(state):
            model, i, ll, ll_diff = state
            return (i < self.max_iter) & (ll_diff > self.tol)

        def body(state):
            model, i, ll, ll_diff = state
            expectations = model.e_step(X)
            new_model = model.m_step(X, expectations)
            new_model = self._regularize(new_model)
            new_ll = new_model.marginal_log_likelihood(X)
            return new_model, i + 1, new_ll, jnp.abs(new_ll - ll)

        init_state = (model, 0, jnp.array(-jnp.inf), jnp.array(jnp.inf))
        final_model, n_iter, final_ll, _ = jax.lax.while_loop(cond, body, init_state)
        return final_model
```

The `fit` method on the distribution class delegates to a fitter:

```python
class GeneralizedHyperbolic(NormalMixtureModel):

    @classmethod
    def fit(cls, X, *, key, max_iter=200, tol=1e-6,
            regularization='det_sigma_one', method='batch') -> 'GeneralizedHyperbolic':
        """Public API — delegates to the appropriate fitter."""
        if method == 'batch':
            fitter = EMFitter(max_iter=max_iter, tol=tol, regularization=regularization)
        elif method == 'online':
            fitter = OnlineEMFitter(tau0=10.0, max_epochs=max_iter)
        elif method == 'minibatch':
            fitter = MiniBatchEMFitter(batch_size=256, max_iter=max_iter, tol=tol)

        model = cls._initialize(X, key)
        return fitter.fit(model, X)
```

---

## 5. Batch, Mini-Batch, and Online EM

### 5.1 Three EM Strategies

| Strategy | Data per step | Convergence | Use case |
|---|---|---|---|
| **Batch EM** | All $n$ samples | Monotone increase in $\ell(\theta)$ | Small-medium data |
| **Mini-batch EM** | $B$ samples | Approximate; step size needed | Large data |
| **Online EM** | 1 sample | Robbins-Monro convergence | Streaming data |

### 5.2 Batch EM (Standard)

As above. The E-step processes all data, the M-step updates parameters. Each iteration monotonically increases the log-likelihood.

```python
class EMFitter(eqx.Module):
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)

    def fit(self, model, X):
        """Standard batch EM via lax.while_loop."""
        ...
```

### 5.3 Online EM (from `docs/theory/online_em.rst`)

From the normix theory document, the online EM for exponential families updates running sufficient statistics:

$$\eta_t = \eta_{t-1} + \tau_t^{-1}(\bar{t}(x_t | \theta_{t-1}) - \eta_{t-1})$$

where $\bar{t}(x_t | \theta_{t-1}) = E[t(X,Y) | X=x_t, \theta_{t-1}]$ and $\tau_t = \tau_0 + t$.

This is a **Robbins-Monro stochastic approximation** on the expectation parameters. The step size $\tau_t^{-1}$ decreases as $O(1/t)$.

```python
class OnlineEMFitter(eqx.Module):
    tau0: float = eqx.field(static=True, default=10.0)
    max_epochs: int = eqx.field(static=True, default=5)

    def fit(self, model, X, *, key):
        """Online EM: process one sample at a time."""
        n = X.shape[0]
        eta = model.joint().expectation_params()  # running sufficient stats

        def step(carry, x_t):
            eta, theta, t = carry
            tau_t = self.tau0 + t
            # E-step: conditional expectation of sufficient stats
            t_bar = model._conditional_sufficient_stats(x_t, theta)
            # Online update
            eta_new = eta + (t_bar - eta) / tau_t
            # M-step: convert expectation params to model params
            theta_new = model._theta_from_eta(eta_new)
            return (eta_new, theta_new, t + 1), None

        for epoch in range(self.max_epochs):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n)
            X_shuffled = X[perm]
            (eta, theta, _), _ = jax.lax.scan(
                step,
                (eta, model.natural_params(), jnp.float64(epoch * n)),
                X_shuffled
            )

        return model.__class__.from_natural(theta)
```

### 5.4 Mini-Batch EM (Generalization)

Mini-batch EM generalizes online EM: instead of one sample, process a batch of $B$ samples per step. Each batch produces an approximate E-step, and the M-step uses a weighted average of the new and old sufficient statistics.

```python
class MiniBatchEMFitter(eqx.Module):
    batch_size: int = eqx.field(static=True, default=256)
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)
    n_inner: int = eqx.field(static=True, default=1)

    def fit(self, model, X, *, key):
        """Mini-batch EM with Robbins-Monro averaging."""
        n = X.shape[0]
        eta = model.joint().expectation_params()

        for t in range(self.max_iter):
            key, subkey = jax.random.split(key)
            # Sample a mini-batch
            indices = jax.random.choice(subkey, n, shape=(self.batch_size,), replace=False)
            X_batch = X[indices]

            # E-step on mini-batch
            for _ in range(self.n_inner):
                expectations = model.e_step(X_batch)
                # Compute batch sufficient statistics
                eta_batch = model._eta_from_expectations(X_batch, expectations)

            # Robbins-Monro update
            step_size = 1.0 / (10.0 + t)
            eta = eta + step_size * (eta_batch - eta)

            # M-step: reconstruct model from updated eta
            model = model.__class__._from_eta(eta)

        return model
```

**Design principle:** The mini-batch and online fitters call the same `e_step` and `m_step` methods on the model — only the iteration logic differs. This is why separating the fitter from the model is valuable.

### 5.5 The `n_inner` Parameter

The `n_inner` parameter controls how many E-M iterations are run per batch without requiring full convergence. This is a practical middle ground:

- `n_inner=1`: standard mini-batch EM (one E+M per batch)
- `n_inner>1`: run multiple E-M iterations on the same batch before moving to the next
- `n_inner=∞` (until convergence): equivalent to fitting the model fully on each batch (overfits to the batch)

For normix, `n_inner=1` is recommended for mini-batch EM. The batch size controls the trade-off between noise and computation.

### 5.6 Unified `fit` API

```python
class GeneralizedHyperbolic(NormalMixtureModel):

    @classmethod
    def fit(cls, X, *, key,
            method='batch',
            max_iter=200,
            tol=1e-6,
            batch_size=None,
            regularization='det_sigma_one',
            n_init=3) -> 'GeneralizedHyperbolic':
        """
        Fit GH distribution to data.

        Parameters
        ----------
        X : array, shape (n, d)
            Training data.
        key : PRNGKey
            Random key for initialization and mini-batching.
        method : str
            'batch' — standard EM (all data per step)
            'online' — online EM (one sample at a time)
            'minibatch' — mini-batch EM
        max_iter : int
            Maximum iterations (or epochs for online/minibatch).
        tol : float
            Convergence tolerance (batch EM only).
        batch_size : int, optional
            Batch size for mini-batch EM. Default: min(256, n).
        regularization : str
            'det_sigma_one', 'sigma_diagonal_one', 'fix_p', 'none'.
        n_init : int
            Number of random initializations. Best model (by log-likelihood) is returned.

        Returns
        -------
        model : GeneralizedHyperbolic
            Fitted model (immutable).
        """
        if batch_size is None:
            batch_size = min(256, X.shape[0])

        if method == 'batch':
            fitter = BatchEMFitter(max_iter=max_iter, tol=tol, regularization=regularization)
        elif method == 'online':
            fitter = OnlineEMFitter(tau0=10.0, max_epochs=max_iter)
        elif method == 'minibatch':
            fitter = MiniBatchEMFitter(
                batch_size=batch_size, max_iter=max_iter, tol=tol)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Multi-start: try n_init random initializations, pick best
        keys = jax.random.split(key, n_init)

        def fit_one(k):
            init_model = cls._initialize(X, k)
            fitted = fitter.fit(init_model, X, key=k)
            ll = fitted.marginal_log_likelihood(X)
            return fitted, ll

        # vmap over initializations for batch EM
        # (for online/minibatch, use a Python loop since they need key splitting)
        if method == 'batch':
            models, lls = jax.vmap(fit_one)(keys)
            best_idx = jnp.argmax(lls)
            return jax.tree.map(lambda x: x[best_idx], models)
        else:
            best_model, best_ll = None, -jnp.inf
            for k in keys:
                model = fit_one(k)
                ll = model[1]
                if ll > best_ll:
                    best_model, best_ll = model[0], ll
            return best_model
```

---

## 6. Conditional Distributions and normix Mixtures

### 6.1 How FlowJAX Handles Conditions

FlowJAX's `cond_shape` is a tuple declaring the shape of the conditioning variable. Methods `_log_prob(x, condition)` and `_sample(key, condition)` take it as an argument. The vectorization handles broadcasting over batch dimensions of both `x` and `condition`.

For conditional flows, the condition is passed to the bijection's neural network, which produces location/scale parameters.

### 6.2 normix's Mixture Structure

normix's mixtures are **not** standard conditional distributions. The structure is:

- **Joint** $f(x, y)$ — an exponential family
- **Marginal** $f(x) = \int f(x,y) dy$ — not an exponential family
- **Conditional** $f(y|x)$ — used in the E-step to compute $E[g(Y)|X]$

The FlowJAX conditional mechanism is designed for $f(x|\text{condition})$ where the condition is observed. In normix, $Y$ is **latent** — it's never observed during fitting. The conditional $Y|X$ is only computed internally during the E-step.

### 6.3 Recommended Approach

Don't use FlowJAX's conditional framework. Instead, implement the joint distribution directly:

```python
class JointNormalMixture(ExponentialFamily):
    """Joint f(x, y) — exponential family."""
    mu: jax.Array         # (d,)
    gamma: jax.Array      # (d,)
    L_Sigma: jax.Array    # (d, d) Cholesky

    @abstractmethod
    def subordinator(self) -> ExponentialFamily:
        """The mixing distribution for Y."""
        ...

    def log_prob_joint(self, x, y):
        """log f(x, y) = log f(x|y) + log f(y)."""
        ...

    def conditional_expectations(self, x):
        """E[Y|X], E[1/Y|X], E[log Y|X] — for EM E-step."""
        ...
```

```python
class NormalMixtureModel(eqx.Module):
    """Marginal f(x) — not exponential family."""
    _joint: JointNormalMixture

    def log_prob(self, x):
        """Marginal log f(x) = log ∫ f(x,y) dy — closed form with Bessel."""
        ...

    def e_step(self, X):
        return jax.vmap(self._joint.conditional_expectations)(X)

    def m_step(self, X, expectations):
        """Return new model with updated params."""
        ...
```

This preserves normix's existing mathematical structure while making it JAX-compatible.

---

## 7. Summary of Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Base class** | `eqx.Module` | Immutable pytree, `jit`/`vmap`/`grad` safe, FlowJAX precedent |
| **vs Flax NNX** | Equinox | Distributions don't need mutability |
| **Parametrizations** | Methods on single class | `natural_params()`, `expectation_params()`, `classical_params()` |
| **Exp family gradients** | `jax.grad(_log_partition_from_theta)` | Automatic Fisher info; analytical overrides via `custom_jvp` |
| **$\eta \to \theta$** | `jaxopt.LBFGSB` for GIG; analytical for simple dists | Same as current normix but with exact gradients |
| **Conditional $X\|Y$** | Direct implementation in `JointNormalMixture` | Not FlowJAX's conditional framework (wrong abstraction) |
| **EM separation** | `EMFitter` separate from model (GMMX pattern) | Enables batch/online/minibatch with same model |
| **Batch EM** | `jax.lax.while_loop` (GMMX pattern) | Fully JIT'd, early stopping |
| **Online EM** | `jax.lax.scan` per epoch (from `docs/theory/online_em.rst`) | Processes one sample at a time, Robbins-Monro step sizes |
| **Mini-batch EM** | Robbins-Monro on sufficient statistics | Processes $B$ samples, $n_\text{inner}$ E-M steps per batch |
| **Multi-start** | `jax.vmap(fit_one)(keys)` for batch EM | Parallel initialization |
| **Parameter constraints** | `jnp.maximum(x, eps)` clamping | Simpler than `paramax.Parameterize` for normix's needs |
| **Functional updates** | `eqx.tree_at` in M-step | Creates new immutable model each iteration |
