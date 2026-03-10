# normix JAX Design

## Design Philosophy

**Minimalist and elegant.** Inspired by FlowJAX: each concept has one canonical representation; no redundant methods, wrappers, or indirection. If a computation appears in multiple places, factor it into a single function. Immutable pytrees everywhere — no mutation, no caching, no `self._fitted` flags.

## Core Dependencies

| Package | Role |
|---|---|
| `jax` | Array computation, autodiff, JIT, vmap |
| `equinox` | Pytree-based modules (immutable, filterable) |
| `jaxopt` | L-BFGS-B for GIG η→θ constrained optimization |
| `tensorflow_probability.substrates.jax` | `log_bessel_kve` for Bessel evaluation |
| `optax` | Optimizers for gradient-based fitting (optional) |

## Architecture

### Module Hierarchy

```
normix/
├── _bessel.py              # log_kv with custom_jvp (TFP + asymptotic)
├── _types.py               # type aliases
├── exponential_family.py   # ExponentialFamily(eqx.Module)
├── distributions/
│   ├── gamma.py
│   ├── inverse_gamma.py
│   ├── inverse_gaussian.py
│   ├── gig.py              # GeneralizedInverseGaussian
│   ├── normal.py           # MultivariateNormal
│   ├── variance_gamma.py
│   ├── normal_inverse_gamma.py
│   ├── normal_inverse_gaussian.py
│   └── generalized_hyperbolic.py
├── mixtures/
│   ├── joint.py            # JointNormalMixture(ExponentialFamily)
│   └── marginal.py         # NormalMixture (owns a JointNormalMixture)
└── fitting/
    ├── em.py               # BatchEMFitter, OnlineEMFitter, MiniBatchEMFitter
    └── mle.py              # fit_mle (convenience)
```

### Base Class: `ExponentialFamily`

One class. Three parametrizations as methods, not separate classes.

```python
class ExponentialFamily(eqx.Module):
    # --- Subclass implements these (all pure functions, unbatched) ---
    @abstractmethod
    def _log_partition_from_theta(self, theta: Array) -> Array: ...
    @abstractmethod
    def natural_params(self) -> Array: ...
    @abstractmethod
    def sufficient_statistics(self, x: Array) -> Array: ...
    @abstractmethod
    def log_base_measure(self, x: Array) -> Array: ...

    # --- Derived automatically via JAX autodiff ---
    def log_partition(self) -> Array:
        return self._log_partition_from_theta(self.natural_params())

    def expectation_params(self) -> Array:
        return jax.grad(self._log_partition_from_theta)(self.natural_params())

    def fisher_information(self) -> Array:
        return jax.hessian(self._log_partition_from_theta)(self.natural_params())

    def log_prob(self, x: Array) -> Array:
        theta = self.natural_params()
        return (self.log_base_measure(x)
                + jnp.dot(self.sufficient_statistics(x), theta)
                - self._log_partition_from_theta(theta))

    # --- Constructors ---
    @classmethod
    def from_natural(cls, theta: Array) -> 'ExponentialFamily': ...
    @classmethod
    def from_expectation(cls, eta: Array) -> 'ExponentialFamily': ...
    @classmethod
    def fit_mle(cls, X: Array) -> 'ExponentialFamily': ...
```

Every derived quantity — `expectation_params`, `fisher_information`, higher-order cumulants — comes from differentiating `_log_partition_from_theta`. No separate `_natural_to_expectation` override unless the analytical form is both faster and registered via `custom_jvp` so that `jax.grad` remains correct at higher orders.

### Distribution Storage

Each distribution stores its **canonical parameters as named attributes** (the minimal set from which everything else is computable). No redundant storage.

| Distribution | Stored attributes | Notes |
|---|---|---|
| Gamma | `alpha`, `beta` | shape, rate |
| InverseGamma | `alpha`, `beta` | shape, rate |
| InverseGaussian | `mu`, `lam` | mean, shape |
| GIG | `p`, `a`, `b` | shape, rate, rate |
| MultivariateNormal | `mu`, `L` | mean, Cholesky of Σ |
| JointNormalMixture | `mu`, `gamma`, `L` + subordinator attrs | location, skewness, Cholesky + mixing |

### Mixture Structure

```python
class JointNormalMixture(ExponentialFamily):
    """f(x, y) — exponential family. X|Y ~ N(μ + γy, Σy), Y ~ subordinator."""
    mu: Array
    gamma: Array
    L: Array  # Cholesky of Σ

    @abstractmethod
    def subordinator(self) -> ExponentialFamily: ...

    def log_prob_joint(self, x, y):
        """log f(x,y) = log f(x|y) + log f_Y(y)."""
        ...

    def conditional_expectations(self, x):
        """E[Y|X], E[1/Y|X], E[log Y|X] — for EM E-step."""
        ...

class NormalMixture(eqx.Module):
    """f(x) = ∫ f(x,y) dy — not exponential family. Owns a JointNormalMixture."""
    _joint: JointNormalMixture

    def log_prob(self, x): ...
    def e_step(self, X): return jax.vmap(self._joint.conditional_expectations)(X)
    def m_step(self, X, expectations) -> 'NormalMixture': ...
```

### EM Fitting: Separate Fitter from Model

Following GMMX: the model knows math, the fitter knows iteration.

```python
class BatchEMFitter(eqx.Module):
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)

    def fit(self, model, X):
        """Batch EM via jax.lax.while_loop. Returns new model."""
        ...
```

The E-step and M-step are methods on the model. The fitter only controls the loop, convergence, and regularization. Same model can be used with `BatchEMFitter`, `OnlineEMFitter`, or `MiniBatchEMFitter`.

## Bessel Functions: `_bessel.py`

### `log_kv(v, z)` — `@jax.custom_jvp`

- **Evaluation:** TFP `log_bessel_kve(|v|, z) - z`, with asymptotic fallbacks for overflow/NaN
- **∂/∂z (exact):** recurrence $K'_\nu = -(K_{\nu-1} + K_{\nu+1})/2$
- **∂/∂ν (large z):** analytical DLMF 10.40.2: $S'(\nu,z)/S(\nu,z)$
- **∂/∂ν (small z):** central finite differences, ε = 10⁻⁵

### Gradient accuracy

Verified across $(v, z)$ grid with $v \in [-100, 500]$, $z \in [10^{-6}, 10^3]$: relative errors $< 10^{-6}$ in all cases. See `notebooks/bessel_function_comparison.ipynb`.

## GIG η→θ Optimization

### Problem

Given $\eta = (\eta_1, \eta_2, \eta_3) = (E[\log Y], E[1/Y], E[Y])$, find $\theta$ such that $\nabla\psi(\theta) = \eta$.

### η-Rescaling

Before optimization, rescale to a symmetric GIG:

$$s = \sqrt{\eta_2/\eta_3}, \quad \tilde\eta = \bigl(\eta_1 + \tfrac{1}{2}\log s^2,\; \sqrt{\eta_2\eta_3},\; \sqrt{\eta_2\eta_3}\bigr)$$

The scaled GIG has $a' = b' = \sqrt{ab}$, giving symmetric Fisher information. After solving $\tilde\theta$:

$$\theta = (\tilde\theta_1,\; \tilde\theta_2/s,\; s\tilde\theta_3)$$

This reduces Fisher condition number by up to $10^{30}$ for extreme $a/b$.

### Solver

`jaxopt.LBFGSB` with bounds $\theta_2 \leq 0$, $\theta_3 \leq 0$. The gradient $\nabla\psi(\theta) - \eta$ is computed via `jax.grad(_log_partition_from_theta)`, which flows through the `custom_jvp` on `log_kv`. Multi-start with initial guesses from Gamma, InverseGamma, and InverseGaussian special cases.

**Alternative:** `optimistix.Newton` on $\nabla\psi(\theta) = \eta$ using `jax.hessian(psi)` as Jacobian. Potentially faster (~5 Newton steps vs ~50 L-BFGS-B), but requires careful handling of bounds.

## Parameter Constraints

Use `jnp.maximum(x, eps)` clamping, not `paramax.Parameterize`. Simpler, no extra dependency, and normix uses EM (not gradient descent on parameters) so gradient flow through constraints is not critical.

## Vectorization

Following FlowJAX: implement `_log_prob` and `_sample` as **unbatched** (single observation). The public API uses `jnp.vectorize` or `jax.vmap` for batching automatically.

## Float Precision

Float64 everywhere. Set `jax.config.update("jax_enable_x64", True)` at package init. The GIG Bessel functions and EM convergence require double precision.
