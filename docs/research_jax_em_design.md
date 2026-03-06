# Research: JAX EM Algorithm Design Patterns

## Table of Contents
1. [GMMX: Deep Design Analysis](#1-gmmx-deep-design-analysis)
2. [OTT-JAX: GMM EM Implementation](#2-ott-jax-gmm-em-implementation)
3. [Dynamax: EM for State Space Models](#3-dynamax-em-for-state-space-models)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Online and Mini-Batch EM](#5-online-and-mini-batch-em)
6. [Key Papers](#6-key-papers)
7. [Design Recommendations for normix](#7-design-recommendations-for-normix)

---

## 1. GMMX: Deep Design Analysis

**Repository:** https://github.com/adonath/gmmx  
**Structure:** Only 4 source files: `gmm.py`, `fit.py`, `utils.py`, `__init__.py`

### 1.1 Architecture Overview

GMMX has a clean two-class separation:

| Class | Role | File |
|---|---|---|
| `GaussianMixtureModelJax` | Model (parameters + log-prob + sampling) | `gmm.py` |
| `EMFitter` | Fitting algorithm (E-step + M-step + loop) | `fit.py` |
| `EMFitterResult` | Immutable result container | `fit.py` |

The model knows nothing about fitting. The fitter knows nothing about internal parameter storage.

### 1.2 The Model Class: `GaussianMixtureModelJax`

**Parameter storage:** JAX dataclass with 3 fields registered as JAX pytree data:

```python
@register_dataclass_jax(data_fields=["weights", "means", "covariances"])
@dataclass
class GaussianMixtureModelJax:
    weights: jax.Array       # (1, n_components, 1, 1)
    means: jax.Array         # (1, n_components, n_features, 1)
    covariances: FullCovariances  # also a JAX-registered dataclass
```

**Key design: 4D arrays everywhere.** All arrays use 4 dimensions: `(batch, components, features, features_covar)`. This eliminates ad-hoc reshaping and enables clean broadcasting. The `Axis` enum provides "poor man's named axes":

```python
class Axis(int, Enum):
    batch = 0
    components = 1
    features = 2
    features_covar = 3
```

**Immutable model pattern:** The model class has no mutating methods. The M-step creates a *new* model instance via the class method `from_responsibilities()`:

```python
@classmethod
def from_responsibilities(cls, x, resp, reg_covar, covariance_type):
    nk = jnp.sum(resp, axis=Axis.batch, keepdims=True) + 10 * jnp.finfo(resp.dtype).eps
    means = jnp.matmul(resp.T, x.T.mT).T / nk
    covariances = COVARIANCE[covariance_type].from_responsibilities(
        x=x, means=means, resp=resp, nk=nk, reg_covar=reg_covar
    )
    return cls(weights=nk / nk.sum(), means=means, covariances=covariances)
```

**Covariance type hierarchy:** Covariance types (`FullCovariances`, `DiagCovariances`) are separate dataclasses with a registry. Each implements its own `log_prob()`, `from_responsibilities()`, and `precisions_cholesky`. This avoids if-else chains entirely.

### 1.3 The Fitter: `EMFitter`

```python
@register_dataclass_jax(meta_fields=["max_iter", "tol", "reg_covar"])
@dataclass
class EMFitter:
    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-6
```

Note: `max_iter`, `tol`, `reg_covar` are `meta_fields` (not traced by JAX), so they become compile-time constants baked into the XLA program.

### 1.4 E-step and M-step (Separated as Methods)

```python
def e_step(self, x, gmm):
    log_prob = gmm.log_prob(x)
    log_prob_norm = jax.scipy.special.logsumexp(
        log_prob, axis=Axis.components, keepdims=True
    )
    log_resp = log_prob - log_prob_norm
    return jnp.mean(log_prob_norm), log_resp

def m_step(self, x, gmm, log_resp):
    x = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))
    return gmm.from_responsibilities(
        x, jnp.exp(log_resp),
        reg_covar=self.reg_covar,
        covariance_type=gmm.covariances.type,
    )
```

The E-step returns `(log_likelihood, log_responsibilities)`. The M-step creates a *new* model from responsibilities.

### 1.5 The EM Loop: `jax.lax.while_loop`

This is the most critical design decision. The entire `fit()` method is `@jax.jit` and uses `lax.while_loop` for the iteration:

```python
@jax.jit
def fit(self, x, gmm):
    def em_step(args):
        x, gmm, n_iter, log_likelihood_prev, _ = args
        log_likelihood, log_resp = self.e_step(x, gmm)
        gmm = self.m_step(x, gmm, log_resp)
        return (
            x, gmm, n_iter + 1,
            log_likelihood,
            jnp.abs(log_likelihood - log_likelihood_prev),
        )

    def em_cond(args):
        _, _, n_iter, _, log_likelihood_diff = args
        return (n_iter < self.max_iter) & (log_likelihood_diff >= self.tol)

    result = jax.lax.while_loop(
        cond_fun=em_cond,
        body_fun=em_step,
        init_val=(x, gmm, 0, jnp.asarray(jnp.inf), jnp.array(jnp.inf)),
    )
    return EMFitterResult(*result, converged=result[2] < self.max_iter)
```

**Why `lax.while_loop`?**
- The entire EM (including all iterations) compiles into a single XLA program
- Supports dynamic stopping (convergence check), unlike `lax.scan` which requires fixed iteration count
- The loop body is traced once, so all iterations execute the same compiled code
- No Python overhead per iteration — everything runs on-device
- The model, data, and loop state are all carried as a flat pytree (thanks to `register_dataclass_jax`)

**Key constraint:** `lax.while_loop` requires fixed shapes/dtypes across iterations, which is naturally satisfied since GMM parameters have fixed sizes.

### 1.6 Convergence Checking

```python
def em_cond(args):
    _, _, n_iter, _, log_likelihood_diff = args
    return (n_iter < self.max_iter) & (log_likelihood_diff >= self.tol)
```

- Checks absolute change in mean log-likelihood: `|LL_new - LL_prev|`
- AND checks iteration count against max_iter
- Both conditions must hold to continue (uses `&` for JAX-compatible boolean)
- `converged` flag: `result[2] < self.max_iter` (stopped before hitting max)

### 1.7 Parameter Update Pattern

Parameters are never mutated in place. Each iteration:
1. E-step computes responsibilities from current model
2. M-step creates an entirely *new* `GaussianMixtureModelJax` from responsibilities
3. The new model replaces the old one in the loop carry state

This is the **functional/immutable pattern** required by JAX's tracing model.

### 1.8 Initialization

GMMX supports k-means initialization via `from_k_means()`:

```python
@classmethod
def from_k_means(cls, x, n_components, reg_covar=1e-6, covariance_type="full", **kwargs):
    from sklearn.cluster import KMeans
    
    resp = jnp.zeros((n_samples, n_components), device=jax.devices("cpu")[0])
    label = KMeans(n_clusters=n_components, **kwargs).fit(x).labels_
    
    idx = jnp.arange(n_samples)
    resp = resp.at[idx, label].set(1.0)
    
    # ... expand dims ...
    return cls.from_responsibilities(xp, resp, reg_covar=reg_covar, ...)
```

Note: k-means runs on CPU via sklearn (not JAX). The JAX model is created from the k-means assignments.

### 1.9 Covariance Regularization

Regularization is added inside `from_responsibilities()` for each covariance type:

```python
# FullCovariances.from_responsibilities
values = jnp.matmul(resp * diff, diff.mT) / nk
idx = jnp.arange(x.shape[Axis.features])
values = values.at[:, :, idx, idx].add(reg_covar)  # add to diagonal

# DiagCovariances.from_responsibilities
values = x_squared_mean - means**2 + reg_covar  # add scalar
```

### 1.10 The `register_dataclass_jax` Utility

```python
class register_dataclass_jax:
    def __init__(self, data_fields=None, meta_fields=None):
        self.data_fields = data_fields or []
        self.meta_fields = meta_fields or []

    def __call__(self, cls):
        jax.tree_util.register_dataclass(
            cls, data_fields=self.data_fields, meta_fields=self.meta_fields
        )
        return cls
```

This makes dataclasses JAX-compatible pytrees:
- `data_fields`: traced by JAX (arrays, nested pytrees)
- `meta_fields`: compile-time constants (int, float, str)

### 1.11 Advantages of Separating Fitter from Model

1. **Single Responsibility:** Model handles representation + evaluation; Fitter handles optimization
2. **Composability:** Different fitters (EM, SGD, Bayesian) can be swapped for the same model
3. **Testability:** Model's `log_prob` can be tested independently of fitting
4. **JAX compatibility:** The model can be a pure pytree (no mutable state), and the fitter can be JIT-compiled
5. **Reusability:** The same model class serves for inference (predict, score) after fitting
6. **Multiple restarts:** Easy to run multiple EMFitters and pick the best result

---

## 2. OTT-JAX: GMM EM Implementation

**Repository:** https://github.com/ott-jax/ott  
**Module:** `ott.tools.gaussian_mixture`

### 2.1 Architecture

OTT-JAX uses **free functions** rather than fitter classes:

| Component | Type | Role |
|---|---|---|
| `GaussianMixture` | Class (pytree) | Model with params |
| `fit_model_em()` | Free function | EM loop |
| `get_assignment_probs()` | Free function | E-step |
| `GaussianMixture.from_points_and_assignment_probs()` | Classmethod | M-step |
| `initialize()` / `from_kmeans_plusplus()` | Free functions | Initialization |

### 2.2 The EM Loop: Python `for` Loop (NOT `lax.while_loop`)

```python
def fit_model_em(gmm, points, point_weights, steps, jit=True, verbose=False):
    if point_weights is None:
        point_weights = jnp.ones(points.shape[:-1])
    
    loss_fn = log_prob_loss
    get_q_fn = get_q
    e_step_fn = get_assignment_probs
    m_step_fn = GaussianMixture.from_points_and_assignment_probs
    
    if jit:
        loss_fn = jax.jit(loss_fn)
        get_q_fn = jax.jit(get_q_fn)
        e_step_fn = jax.jit(e_step_fn)
        m_step_fn = jax.jit(m_step_fn)

    for i in range(steps):
        assignment_probs = e_step_fn(gmm, points)
        gmm_new = m_step_fn(points, point_weights, assignment_probs)
        if gmm_new.has_nans():
            raise ValueError("NaNs in fit.")
        if verbose:
            loss = loss_fn(gmm_new, points, point_weights)
            q = get_q_fn(gmm=gmm_new, assignment_probs=assignment_probs, ...)
            print(f"{i}  q={q}  -log prob={loss}")
        gmm = gmm_new
    return gmm
```

**Key differences from GMMX:**
- Uses Python `for` loop (not `lax.while_loop`)
- Fixed number of steps (no convergence check!)
- Each step individually JIT-compiled (E-step and M-step are separate JIT'd functions)
- Supports NaN checking between iterations (not possible inside `lax.while_loop`)
- Supports verbose printing per iteration
- No convergence result returned — just the final model

### 2.3 E-step

```python
def get_assignment_probs(gmm, points):
    return jnp.exp(gmm.get_log_component_posterior(points))
```

### 2.4 M-step (via classmethod)

```python
@classmethod
def from_points_and_assignment_probs(cls, points, point_weights, assignment_probs):
    mean, cov, wts = get_summary_stats_from_points_and_assignment_probs(
        points=points, point_weights=point_weights, assignment_probs=assignment_probs
    )
    return cls.from_mean_cov_component_weights(mean=mean, cov=cov, component_weights=wts)
```

The summary stats computation uses `jax.vmap`:

```python
def get_summary_stats_from_points_and_assignment_probs(points, point_weights, assignment_probs):
    def component_from_points(points, point_weights, assignment_probs):
        component_weight = jnp.sum(point_weights * assignment_probs) / jnp.sum(point_weights)
        component_mean, component_cov = linalg.get_mean_and_cov(
            points=points, weights=point_weights * assignment_probs
        )
        return component_mean, component_cov, component_weight

    components_from_points_fn = jax.vmap(
        component_from_points, in_axes=(None, None, 1), out_axes=0
    )
    return components_from_points_fn(points, point_weights, assignment_probs)
```

### 2.5 Initialization: K-means++

```python
def initialize(rng, points, point_weights, n_components, n_attempts=50, verbose=False):
    for attempt in range(n_attempts):
        rng, subrng = jax.random.split(rng)
        try:
            return from_kmeans_plusplus(rng=subrng, points=points, ...)
        except ValueError:
            if verbose:
                print(f"Failed to initialize, attempt {attempt}.")
    raise ValueError("Failed to initialize.")
```

Implements K-means++ from scratch in JAX (not using sklearn).

### 2.6 Penalized EM for GMM Pairs

OTT-JAX also has a penalized EM for fitting GMM pairs (with Wasserstein distance penalty). The M-step there uses **Adam optimizer** via `optax` to maximize the penalized Q function — a gradient-based M-step rather than closed-form:

```python
def get_m_step_fn(learning_rate, objective_fn, jit):
    def _m_step_fn(pair, obs0, obs1, steps):
        state = opt_init((pair,))
        for _ in range(steps):
            grad_objective = grad_objective_fn(pair, obs0, obs1)
            updates, state = opt_update(grad_objective, state, (pair,))
            (pair,) = optax.apply_updates((pair,), updates)
        return pair

    grad_objective_fn = jax.grad(objective_fn, argnums=(0,))
    opt_init, opt_update = optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale(learning_rate)
    )
    return _m_step_fn
```

This shows that the M-step doesn't have to be closed-form — it can be gradient-based optimization.

### 2.7 Parameter Storage

OTT-JAX stores parameters in an unconstrained form:
- Locations (`loc`): raw arrays
- Scale parameters (`scale_params`): parameterized lower-triangular Cholesky via `ScaleTriL`
- Component weights: `Probabilities` object (logits internally, softmax for probabilities)

The model is registered as a pytree via `@jax.tree_util.register_pytree_node_class`.

---

## 3. Dynamax: EM for State Space Models

**Repository:** https://github.com/probml/dynamax  
**Focus:** HMMs and Linear Gaussian SSMs

### 3.1 Architecture

Dynamax uses a **deep class hierarchy** with the SSM base class providing the EM loop:

```
SSM (base)
├── fit_em()         # Generic EM loop (Python for loop)
├── fit_sgd()        # SGD alternative
├── e_step()         # Abstract
└── m_step()         # Abstract
    │
    └── HMM (subclass)
        ├── e_step()     # Calls smoother → collect_suff_stats
        ├── m_step()     # Delegates to initial/transition/emission components
        ├── initial_component: HMMInitialState
        ├── transition_component: HMMTransitions
        └── emission_component: HMMEmissions
```

### 3.2 The EM Loop: Python `for` Loop with JIT'd Step

```python
def fit_em(self, params, props, emissions, inputs=None, num_iters=50, verbose=True):
    batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
    batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

    @jit
    def em_step(params, m_step_state):
        batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_inputs)
        lp = self.log_prior(params) + lls.sum()
        params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
        return params, m_step_state, lp

    log_probs = []
    m_step_state = self.initialize_m_step_state(params, props)
    pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
    for _ in pbar:
        params, m_step_state, marginal_logprob = em_step(params, m_step_state)
        log_probs.append(marginal_logprob)
    return params, jnp.array(log_probs)
```

**Key design choices:**
- Python `for` loop (not `lax.while_loop`)
- Single JIT'd `em_step` function (E + M together)
- `vmap` over batch of sequences in the E-step
- Fixed number of iterations (no convergence check)
- Parameters are passed in/out (not stored on the model)
- `m_step_state` carries optimizer state for gradient-based M-steps
- Progress bar support

### 3.3 E-step: Sufficient Statistics

```python
def e_step(self, params, emissions, inputs=None):
    args = self._inference_args(params, emissions, inputs)
    posterior = hmm_two_filter_smoother(*args)
    
    initial_stats = self.initial_component.collect_suff_stats(params.initial, posterior, inputs)
    transition_stats = self.transition_component.collect_suff_stats(params.transitions, posterior, inputs)
    emission_stats = self.emission_component.collect_suff_stats(params.emissions, posterior, emissions, inputs)
    return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik
```

Note: The E-step returns **sufficient statistics** (not responsibilities), which is the proper exponential family approach.

### 3.4 M-step: Component-wise

```python
def m_step(self, params, props, batch_stats, m_step_state):
    batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
    initial_m_step_state, transitions_m_step_state, emissions_m_step_state = m_step_state
    
    initial_params, initial_m_step_state = self.initial_component.m_step(
        params.initial, props.initial, batch_initial_stats, initial_m_step_state)
    transition_params, transitions_m_step_state = self.transition_component.m_step(
        params.transitions, props.transitions, batch_transition_stats, transitions_m_step_state)
    emission_params, emissions_m_step_state = self.emission_component.m_step(
        params.emissions, props.emissions, batch_emission_stats, emissions_m_step_state)
    
    params = params._replace(initial=initial_params, transitions=transition_params, ...)
    return params, m_step_state
```

Each component can have its own M-step strategy (closed-form or gradient-based via optax).

### 3.5 SGD Alternative

```python
def fit_sgd(self, params, props, emissions, inputs=None,
            optimizer=optax.adam(1e-3), batch_size=1, num_epochs=50, ...):
    unc_params = to_unconstrained(params, props)
    
    def _loss_fn(unc_params, minibatch):
        params = from_unconstrained(unc_params, props)
        minibatch_emissions, minibatch_inputs = minibatch
        scale = len(batch_emissions) / len(minibatch_emissions)
        minibatch_lls = vmap(partial(self.marginal_log_prob, params))(
            minibatch_emissions, minibatch_inputs)
        lp = self.log_prior(params) + minibatch_lls.sum() * scale
        return -lp / batch_emissions.size
    
    unc_params, losses = run_sgd(_loss_fn, unc_params, dataset, ...)
    params = from_unconstrained(unc_params, props)
    return params, losses
```

This is interesting: Dynamax provides both EM and SGD fitting, with the same model class supporting both. The SGD path uses unconstrained parameter transformations.

---

## 4. Comparative Analysis

### 4.1 EM Loop Strategy

| Package | Loop Type | Convergence Check | JIT Strategy |
|---|---|---|---|
| **GMMX** | `lax.while_loop` | Yes (LL diff < tol) | Entire fit is one JIT call |
| **OTT-JAX** | Python `for` | No (fixed steps) | Each step individually JIT'd |
| **Dynamax** | Python `for` | No (fixed steps) | `em_step` function JIT'd |

**Trade-offs:**

| | `lax.while_loop` | Python `for` + JIT'd step |
|---|---|---|
| **Compilation** | One trace, one compile | One trace for step, Python loop overhead |
| **Convergence** | Dynamic stopping ✓ | Fixed iterations only |
| **Debugging** | Hard (no print inside) | Easy (print, NaN check between iters) |
| **Performance** | Best (zero Python overhead) | Good (small Python overhead per iter) |
| **Differentiation** | Cannot reverse-mode diff through | Can accumulate gradients |
| **Memory** | Fixed (no unrolling) | Fixed (JIT'd step doesn't unroll) |

### 4.2 Model/Fitter Separation

| Package | Approach | Fitter Type |
|---|---|---|
| **GMMX** | Separate `EMFitter` dataclass | Class with methods |
| **OTT-JAX** | Free functions | `fit_model_em()` function |
| **Dynamax** | Methods on SSM base class | `fit_em()` is a method |

### 4.3 Parameter Update Pattern

| Package | Pattern |
|---|---|
| **GMMX** | Create new model each iteration (functional) |
| **OTT-JAX** | Create new model each iteration (functional) |
| **Dynamax** | Return updated params NamedTuple (params separate from model) |

### 4.4 M-step Implementation

| Package | M-step Type |
|---|---|
| **GMMX** | Closed-form (from responsibilities) |
| **OTT-JAX (basic)** | Closed-form (from assignment probs) |
| **OTT-JAX (pair)** | Gradient-based (Adam optimizer) |
| **Dynamax** | Component-specific (closed-form or gradient-based) |

---

## 5. Online and Mini-Batch EM

### 5.1 Standard (Batch) EM Recap

For an exponential family model with complete-data sufficient statistics $t(x, z)$:

$$\hat{s}_n = \frac{1}{n} \sum_{i=1}^{n} E[t(x_i, z_i) | x_i, \theta^{(t)}]$$

$$\theta^{(t+1)} = \arg\max_\theta \langle \hat{s}_n, \theta \rangle - A(\theta)$$

### 5.2 Online EM (Cappé & Moulines, 2009)

**Core idea:** Maintain a running average of sufficient statistics, update with each new observation.

**Algorithm:**

At step $n$, given new observation $x_n$ and current parameters $\theta_{n-1}$:

1. **E-step (stochastic):** Compute expected sufficient statistics for the new observation:

$$\hat{t}_n = E[t(x_n, z_n) | x_n, \theta_{n-1}]$$

2. **Stochastic approximation:** Update running sufficient statistics with step size $\gamma_n$:

$$\bar{s}_n = (1 - \gamma_n) \bar{s}_{n-1} + \gamma_n \hat{t}_n$$

3. **M-step:** Update parameters from the smoothed sufficient statistics:

$$\theta_n = \arg\max_\theta \langle \bar{s}_n, \theta \rangle - A(\theta)$$

**Step size requirements** (Robbins-Monro conditions):

$$\sum_{n=1}^{\infty} \gamma_n = \infty, \quad \sum_{n=1}^{\infty} \gamma_n^2 < \infty$$

Common choices:
- $\gamma_n = n^{-\alpha}$ where $\alpha \in (0.5, 1]$
- $\gamma_n = (n + \tau)^{-\kappa}$ with delay $\tau$ and rate $\kappa$

**Convergence:** Converges to a stationary point of the KL divergence at the optimal rate (same as batch MLE).

### 5.3 Mini-Batch EM

**Core idea:** Process a batch of $B$ observations at each step instead of one.

At step $t$, given mini-batch $\mathcal{B}_t$ of size $B$:

1. **E-step:** Compute batch sufficient statistics:

$$\hat{t}_t = \frac{1}{B} \sum_{x_i \in \mathcal{B}_t} E[t(x_i, z_i) | x_i, \theta_{t-1}]$$

2. **Update:** 

$$\bar{s}_t = (1 - \gamma_t) \bar{s}_{t-1} + \gamma_t \hat{t}_t$$

3. **M-step:** Same as online EM.

**Advantages over online EM:**
- Better gradient estimates (lower variance)
- Can leverage GPU parallelism over batch dimension
- More stable updates

### 5.4 Neal & Hinton (1998) Incremental EM

**Key insight:** View EM as maximizing a **free energy** function $F(q, \theta)$ where $q$ is the distribution over latent variables.

$$F(q, \theta) = E_q[\log p(x, z | \theta)] + H(q)$$

Standard EM alternates:
- E-step: maximize $F$ w.r.t. $q$ → set $q(z) = p(z|x, \theta)$
- M-step: maximize $F$ w.r.t. $\theta$

**Incremental EM:** Instead of updating all $q_i$ simultaneously, update one at a time:
- Pick data point $i$
- Update $q_i(z_i) = p(z_i | x_i, \theta)$
- Update $\theta$ based on all $q$'s

**Guarantee:** Each partial update is guaranteed to not decrease the objective.

**Variants justified:**
- Partial E-steps (update subset of $q_i$'s)
- Partial M-steps (don't fully maximize)
- Sparse updates (only update components with significant responsibility)

### 5.5 Practical Implementation for Exponential Families

For exponential families, online EM has a particularly elegant form because the M-step reduces to computing sufficient statistics:

```python
# Online EM for exponential family mixture
def online_em_step(suff_stats, theta, x_new, step_size):
    # E-step: compute responsibilities for new observation
    resp = e_step(x_new, theta)  # (n_components,)
    
    # Compute expected sufficient statistics for this observation
    t_new = compute_expected_suff_stats(x_new, resp)
    
    # Stochastic approximation update
    suff_stats = (1 - step_size) * suff_stats + step_size * t_new
    
    # M-step: update parameters from smoothed sufficient statistics
    theta = m_step_from_suff_stats(suff_stats)
    
    return suff_stats, theta
```

For a Gaussian mixture specifically:
- Sufficient statistics: $\{n_k, \sum_k x, \sum_k x x^T\}$ for each component $k$
- The M-step from sufficient statistics is always closed-form

### 5.6 Step Size Schedules

| Schedule | Formula | Properties |
|---|---|---|
| Polynomial decay | $\gamma_n = (n + \tau)^{-\kappa}$ | $\kappa = 0.6$ often works well |
| Averaging | $\gamma_n = 1/n$ | Corresponds to running mean |
| Constant + decay | $\gamma_n = \gamma_0 / (1 + n/n_0)$ | Warm start with $\gamma_0$ |
| Mini-batch scaled | $\gamma_n = B / (n \cdot B + B_0)$ | Scales with batch size |

### 5.7 Convergence Guarantees Summary

| Method | Convergence Rate | Conditions |
|---|---|---|
| Batch EM | Linear to local max | Standard regularity |
| Online EM | $O(1/n)$ to stationary point | Robbins-Monro step sizes |
| Mini-batch EM | $O(1/(nB))$ per sample | Robbins-Monro + IID batches |
| Incremental EM | Monotone increase in F | Exact partial E-steps |

---

## 6. Key Papers

1. **Cappé & Moulines (2009)** — "On-line expectation-maximization algorithm for latent data models" JRSSB 71(3):593-613
   - Foundational paper for online EM
   - Works in the space of sufficient statistics
   - Convergence at optimal (MLE) rate
   - MATLAB code: https://www.di.ens.fr/~cappe/Code/OnlineEM/

2. **Neal & Hinton (1998)** — "A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants"
   - Free energy interpretation of EM
   - Justifies partial E-steps and M-steps
   - Foundation for incremental/online variants

3. **Karimi, Lavielle, Moulines (2019)** — "Non-asymptotic analysis of the EM algorithm and mini-batch EM"
   - Finite-time convergence rates for mini-batch EM
   - Analysis of stochastic approximation EM variants
   - Batch size affects limit distribution of estimators

4. **Liang & Klein (2009)** — "Online EM for Unsupervised Models" (NAACL)
   - Practical stepwise EM for NLP models
   - Comparison of step size schedules
   - Shows mini-batch can outperform batch EM

---

## 7. Design Recommendations for normix

### 7.1 Recommended Architecture

Based on the analysis, the recommended pattern for normix combines the best of all three approaches:

```
EMFitter (separate class)
├── fit(x, model) → EMResult          # Main entry point
├── _em_step(x, model) → (model, ll)  # Single step
├── e_step(x, model) → (ll, resp)     # E-step (delegated)  
└── m_step(x, model, resp) → model    # M-step (delegated)

Distribution (model class)
├── log_prob(x) → array               # Likelihood computation
├── from_responsibilities(x, resp) → cls  # M-step factory method
├── sufficient_statistics(x, resp) → stats  # For online EM
└── from_sufficient_statistics(stats) → cls  # For online EM
```

### 7.2 Loop Strategy Recommendation

Use **`lax.while_loop`** for the standard EM (following GMMX), because:
- Convergence checking is essential for production use
- Zero Python overhead per iteration matters for large datasets
- normix distributions are exponential families with fixed-size parameters (satisfies shape constraint)

Keep a Python-loop fallback for debugging/verbose mode.

### 7.3 Online EM Extension

For exponential families like normix, online EM is particularly natural because:
- Sufficient statistics are always well-defined
- The M-step from sufficient statistics is always closed-form
- The stochastic approximation update is a simple weighted average of statistics

The architecture should support both batch and online EM with the same E-step/M-step interface.

### 7.4 Key Implementation Decisions

| Decision | Recommendation | Rationale |
|---|---|---|
| Model mutation | Immutable (create new) | Required for JAX tracing |
| Parameter storage | Named attributes (current normix) | Already implemented |
| Fitter coupling | Separate class | Composability, testability |
| Loop type | `lax.while_loop` | Dynamic convergence |
| Convergence | LL diff threshold | Standard, works well |
| Regularization | Via M-step (add to diagonal) | Per GMMX pattern |
| Initialization | K-means (sklearn) | Proven approach |
| JAX registration | `register_dataclass_jax` or pytree | Required for lax loops |
