# Fit API Design for JAX normix

**Date:** 2026-03-07

---

## 1. The Core Insight: MLE = $\eta \to \theta$

For an exponential family with density $f(x|\theta) = h(x)\exp(\theta^\top t(x) - \psi(\theta))$, the MLE has a beautiful structure:

$$\hat{\eta} = \frac{1}{n}\sum_{i=1}^n t(x_i) \qquad \Longrightarrow \qquad \hat{\theta} = \nabla\phi(\hat{\eta})$$

where $\phi$ is the Legendre dual of $\psi$. Equivalently, $\hat{\theta}$ solves $\nabla\psi(\theta) = \hat{\eta}$, or minimizes the Bregman divergence $D_\psi(\theta \| \theta_0)$ subject to the moment constraint.

This means **fitting an exponential family is the same operation as converting expectation parameters to natural parameters**. There is no gradient descent involved — it's either a closed-form formula or a convex optimization problem.

For the normal mixture distributions, the EM algorithm reduces each M-step to exactly this operation: compute the expected sufficient statistics (E-step), then solve $\eta \to \theta$ (M-step).

---

## 2. How FlowJAX and efax Handle Fitting

### 2.1 FlowJAX: Generic SGD Training Loop

FlowJAX's `fit_to_data(key, dist, x, ...)` is a general-purpose gradient descent loop:

1. Partition model into `(params, static)` via `eqx.partition(dist, eqx.is_inexact_array)`
2. Define loss: `MaximumLikelihoodLoss` = `-dist.log_prob(x).mean()`
3. Run epochs: shuffle data → mini-batches → `optax.adam` step → early stopping

This is appropriate for normalizing flows where the log-likelihood has no closed-form MLE — you must differentiate through a chain of bijections and optimize with gradient descent.

**For exponential families, this is wasteful.** You already know $\hat{\eta} = \overline{t(x)}$ — there's nothing to optimize over the data.

### 2.2 efax: Algebraic MLE as Composition

efax has **no `fit()` method**. Instead, MLE is expressed as a three-step pipeline:

```
sufficient_statistics(x)  →  mean over samples  →  to_nat()
```

Concretely:
1. `ep = dist_np.sufficient_statistic(x)` — returns an `ExpectationParametrization` object per sample
2. `eta_hat = jnp.mean(ep, axis=0)` (via `parameter_mean`)
3. `theta_hat = eta_hat.to_nat()` — solves $\nabla\psi(\theta) = \hat{\eta}$ via Newton (using `optimistix`)

The $\eta \to \theta$ step uses the Fisher information $I(\theta) = \nabla^2\psi(\theta)$ as the Hessian in Newton's method. efax computes this automatically via `jax.hessian(log_normalizer)`.

---

## 3. Recommended Design

### 3.1 Two-Level Architecture

The key insight is that there are **two distinct fitting problems**:

| Level | Problem | Method | Where |
|---|---|---|---|
| **ExponentialFamily** | Given $\hat{\eta}$, find $\hat{\theta}$ | Convex optimization (Newton or L-BFGS-B) | `eta_to_theta()` function |
| **NormalMixtureModel** | Given data $X$ with latent $Y$, find parameters | EM algorithm (calls `eta_to_theta` in M-step) | `EMFitter` class |

These should be **separate concerns**. The `eta_to_theta` operation is a reusable building block used both in standalone MLE fitting and inside the EM M-step.

### 3.2 `eta_to_theta` as a Standalone Function

Following efax's philosophy, the $\eta \to \theta$ conversion is a **pure function**, not a method:

```python
def eta_to_theta(dist_cls, eta, *, theta0=None):
    """
    Convert expectation parameters to natural parameters.

    Solves: θ* = argmin_θ [ψ(θ) - θ·η]  (Bregman divergence minimization)

    Parameters
    ----------
    dist_cls : type
        The ExponentialFamily subclass (needed for _log_partition_from_theta, bounds).
    eta : Array
        Expectation parameter vector.
    theta0 : Array, optional
        Initial guess for Newton/L-BFGS-B.

    Returns
    -------
    theta : Array
        Natural parameter vector.
    """
    def objective(theta):
        return dist_cls._log_partition_static(theta) - jnp.dot(theta, eta)

    # Use Newton for simple distributions, L-BFGS-B for GIG
    ...
```

This function is:
- **JIT-able**: pure function of arrays
- **Differentiable**: can compute $\partial\hat{\theta}/\partial\hat{\eta}$ via `jax.jacfwd` (useful for sensitivity analysis)
- **Reusable**: called by both `ExponentialFamily.fit()` and the EM M-step
- **Testable**: test it independently of any distribution object

### 3.3 `fit` as a Classmethod on `ExponentialFamily`

For user-facing convenience, provide a classmethod that composes the pipeline:

```python
class ExponentialFamily(eqx.Module):

    @classmethod
    def fit(cls, X: jax.Array) -> 'ExponentialFamily':
        """
        Maximum likelihood estimation.

        For exponential families: η̂ = mean(t(X)), then solve ∇ψ(θ) = η̂.

        Parameters
        ----------
        X : Array, shape (n, ...) or (n,)
            Observed data.

        Returns
        -------
        model : ExponentialFamily
            New fitted model (immutable).
        """
        eta_hat = jnp.mean(jax.vmap(cls._sufficient_statistics_static)(X), axis=0)
        theta_hat = eta_to_theta(cls, eta_hat)
        return cls.from_natural(theta_hat)
```

This is a **classmethod** (not an instance method) because:
1. It creates a new model — it doesn't modify an existing one (immutability)
2. It doesn't need an existing instance's parameters
3. It mirrors the mathematical structure: MLE is a map from data to parameters

### 3.4 Why Not FlowJAX-style External Training Loop?

For exponential families, a gradient descent loop is mathematically unnecessary and computationally wasteful:

| Approach | Steps | Operations | When to use |
|---|---|---|---|
| **Closed-form MLE** | 1 | mean(t(X)) → eta_to_theta | Gamma, Normal, InverseGamma |
| **Convex optimization** | ~10-50 Newton steps | Same, but eta_to_theta uses L-BFGS-B | GIG (constrained) |
| **Gradient descent** | ~1000 epochs × batches | Differentiate log_prob, optimizer step | Normalizing flows |

For normix's distributions, the first two cover everything. Gradient descent would be like using SGD to compute a sample mean.

However, a FlowJAX-style training loop could be useful as an **alternative fitting method** for cases where the standard approach fails (e.g., very high-dimensional problems where Newton's Hessian is too expensive). It's worth supporting as an option, but not the default.

---

## 4. EM Algorithm Design

### 4.1 The M-Step IS `eta_to_theta`

For the joint normal mixture $f(x, y | \theta)$, the EM M-step is:

$$\hat{\eta}_{\text{M-step}} = \frac{1}{n}\sum_{i=1}^n E[t(x_i, Y) | x_i, \theta_{\text{old}}]$$

This is exactly the same form as the MLE — the expected sufficient statistics play the role of $\hat{\eta}$. The M-step then converts $\hat{\eta} \to \hat{\theta}$ using the same `eta_to_theta` function.

For normix's normal mixtures, the M-step has two parts:
1. **Normal parameters** ($\mu$, $\gamma$, $\Sigma$): closed-form from the expected sufficient statistics
2. **Subordinator parameters** (e.g., $p$, $a$, $b$ for GIG): `eta_to_theta` on the subordinator's exponential family

This means the EM M-step reuses `eta_to_theta` internally — confirming the two-level architecture.

### 4.2 Fitter as Separate Class

```python
class BatchEMFitter(eqx.Module):
    """Batch EM: process all data per iteration."""
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)

    def fit(self, model, X):
        def cond(state):
            _, i, _, ll_diff = state
            return (i < self.max_iter) & (ll_diff > self.tol)

        def body(state):
            model, i, ll, _ = state
            # E-step: expected sufficient statistics
            exp_suff_stats = model.expected_sufficient_statistics(X)
            # M-step: eta_to_theta on appropriate components
            new_model = model.from_expected_sufficient_statistics(exp_suff_stats)
            new_ll = new_model.marginal_log_likelihood(X)
            return new_model, i + 1, new_ll, new_ll - ll

        init = (model, 0, jnp.array(-jnp.inf), jnp.array(jnp.inf))
        return jax.lax.while_loop(cond, body, init)[0]


class OnlineEMFitter(eqx.Module):
    """Online EM: Robbins-Monro updates on sufficient statistics."""
    tau0: float = eqx.field(static=True, default=10.0)
    n_epochs: int = eqx.field(static=True, default=5)

    def fit(self, model, X, *, key):
        eta = model.joint().expectation_params()
        n = X.shape[0]

        def step(carry, x_t):
            eta, t = carry
            tau_t = self.tau0 + t
            t_bar = model.conditional_expected_sufficient_statistics(x_t)
            eta_new = eta + (t_bar - eta) / tau_t
            return (eta_new, t + 1), None

        for epoch in range(self.n_epochs):
            key, sk = jax.random.split(key)
            X_shuffled = X[jax.random.permutation(sk, n)]
            (eta, _), _ = jax.lax.scan(step, (eta, epoch * n), X_shuffled)

        return model.__class__.from_expectation(eta)


class MiniBatchEMFitter(eqx.Module):
    """Mini-batch EM: process B samples per step."""
    batch_size: int = eqx.field(static=True, default=256)
    max_iter: int = eqx.field(static=True, default=200)

    def fit(self, model, X, *, key):
        n = X.shape[0]
        eta = model.joint().expectation_params()

        for t in range(self.max_iter):
            key, sk = jax.random.split(key)
            idx = jax.random.choice(sk, n, shape=(self.batch_size,), replace=False)
            X_batch = X[idx]
            t_bar = jnp.mean(
                jax.vmap(model.conditional_expected_sufficient_statistics)(X_batch),
                axis=0)
            step_size = 1.0 / (10.0 + t)
            eta = eta + step_size * (t_bar - eta)

        return model.__class__.from_expectation(eta)
```

### 4.3 Public `fit` API

The distribution class provides a user-friendly `fit` classmethod that composes initialization + fitter:

```python
class GeneralizedHyperbolic(NormalMixtureModel):

    @classmethod
    def fit(cls, X, *, key,
            method='batch', max_iter=200, tol=1e-6, batch_size=None,
            regularization='det_sigma_one', n_init=3):
        """
        Fit GH distribution to data via EM.

        Parameters
        ----------
        X : Array, shape (n, d)
        key : PRNGKey
        method : 'batch', 'online', 'minibatch'
        max_iter, tol, batch_size : algorithm parameters
        regularization : str
        n_init : int, number of random restarts

        Returns
        -------
        model : GeneralizedHyperbolic (immutable)
        """
        fitter = _make_fitter(method, max_iter=max_iter, tol=tol, batch_size=batch_size)

        # Multi-start
        keys = jax.random.split(key, n_init)
        best_model, best_ll = None, -jnp.inf
        for k in keys:
            init_model = cls._initialize(X, k)
            fitted = fitter.fit(init_model, X, key=k)
            ll = fitted.marginal_log_likelihood(X)
            if ll > best_ll:
                best_model, best_ll = fitted, ll

        return best_model
```

---

## 5. Putting It All Together

### 5.1 The Full Stack

```
User: GH.fit(X, key=key)
  │
  ├── Initialization: GH._initialize(X, key)
  │     └── Try NIG, VG, NInvG initializations → pick best
  │
  ├── EMFitter.fit(model, X)
  │     └── Loop:
  │           ├── E-step: model.expected_sufficient_statistics(X)
  │           │     └── vmap over X: compute E[t(x,Y)|x, θ] per sample
  │           │           └── Conditional GIG expectations (Bessel ratios)
  │           │
  │           └── M-step: model.from_expected_sufficient_statistics(eta)
  │                 ├── Normal params (μ, γ, Σ): closed-form from eta
  │                 └── GIG params (p, a, b): eta_to_theta(GIG, eta_sub)
  │                       └── jaxopt.LBFGSB (GIG is constrained)
  │                             └── jax.grad(GIG._log_partition_static)
  │                                   └── custom_jvp through log_kv (TFP + Bessel)
  │
  └── Return: fitted GH model (immutable eqx.Module)
```

### 5.2 What Lives Where

| Component | Location | Responsibility |
|---|---|---|
| `eta_to_theta(cls, eta)` | Free function in `normix.fitting` | Convex optimization: $\nabla\psi(\theta) = \eta$ |
| `ExponentialFamily.fit(X)` | Classmethod | MLE: $\hat{\eta} = \overline{t(X)} \to \hat{\theta}$ |
| `ExponentialFamily.from_expectation(eta)` | Classmethod | Convenience: calls `eta_to_theta` then `from_natural` |
| `BatchEMFitter` | Separate class in `normix.fitting` | EM iteration logic |
| `OnlineEMFitter` | Separate class in `normix.fitting` | Online EM (Cappé-Moulines) |
| `MiniBatchEMFitter` | Separate class in `normix.fitting` | Mini-batch EM |
| `NormalMixtureModel.fit(X)` | Classmethod | Initialization + delegates to fitter |
| `NormalMixtureModel.e_step(X)` | Method | Compute $E[t(X,Y)|X,\theta]$ |
| `NormalMixtureModel.m_step(eta)` | Method | Reconstruct model from expected sufficient stats |

### 5.3 Design Principles

1. **`eta_to_theta` is the atomic operation.** Everything else composes it.

2. **`fit` is a classmethod** — it constructs a new model, not modifying an existing one. This matches immutability and mathematical semantics (MLE is a map from data to parameters, not a mutation of existing parameters).

3. **The fitter is separate from the model** — enables batch/online/minibatch with the same model. The model knows math (E-step, M-step formulas), the fitter knows iteration (convergence, step sizes, batching).

4. **All parametrization conversions are JIT-able pure functions.** `natural_params()`, `expectation_params()`, `_log_partition_from_theta()` are methods that return arrays. `jax.grad` and `jax.hessian` flow through them.

5. **Analytical gradients are preferred** but not required. If a distribution overrides `expectation_params()` with a closed-form formula, register a `custom_jvp` on `_log_partition_from_theta` so the Hessian (Fisher info) is also analytical. If not overridden, `jax.grad` provides a numerical fallback.

6. **The gradient descent fallback exists** for edge cases. A `GradientFitter` (FlowJAX-style) can optimize `log_prob` directly using optax, useful when the standard pipeline fails or for non-exponential-family extensions.
