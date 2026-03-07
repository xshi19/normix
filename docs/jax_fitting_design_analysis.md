# JAX Fitting/Training Module Design Analysis

**Date:** 2026-03-07  
**Purpose:** Detailed source-level analysis of FlowJAX and efax fitting modules, plus design patterns for normix JAX port.

---

## 1. FlowJAX Train Module

**Repository:** https://github.com/danielward27/flowjax  
**Module:** `flowjax/train/`

### 1.1 Module Structure

```
flowjax/train/
├── __init__.py          # Exports: fit_to_data, fit_to_key_based_loss, step
├── loops.py             # Main training loops
├── losses.py            # Loss functions (MaximumLikelihoodLoss, ElboLoss, ContrastiveLoss)
└── train_utils.py       # step(), train_val_split(), get_batches(), count_fruitless()
```

### 1.2 `fit_to_data()` — Full Source & Analysis

This is the primary data-fitting function. It trains any PyTree (typically a distribution) against samples.

```python
def fit_to_data(
    key: PRNGKeyArray,
    dist: PyTree,
    data: ArrayLike | Iterable[ArrayLike] = (),
    *,
    loss_fn: Callable | None = None,          # Default: MaximumLikelihoodLoss()
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,  # Default: adam
    max_epochs: int = 100,
    max_patience: int = 5,                    # Early stopping patience
    batch_size: int = 100,
    val_prop: float = 0.1,                    # Validation split proportion
    return_best: bool = True,                 # Return params at best val loss
    show_progress: bool = True,
):
```

**Key design decisions:**
- `data` can be a single array (unconditional) or tuple of arrays (conditional, e.g., `(target, condition)`)
- All keyword arguments after `data` are keyword-only (enforced by `*`)
- Returns `(trained_dist, losses_dict)` where losses is `{"train": [...], "val": [...]}`

**Training loop internals:**

```python
# 1. Partition trainable vs non-trainable
params, static = eqx.partition(
    dist,
    eqx.is_inexact_array,
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
)
best_params = params
opt_state = optimizer.init(params)

# 2. Train/val split
train_data, val_data = train_val_split(subkey, data, val_prop=val_prop)

# 3. Epoch loop with early stopping
for _ in loop:
    # Shuffle each epoch
    train_data = [jr.permutation(subkeys[0], a) for a in train_data]
    
    # Train batches
    for batch in zip(*get_batches(train_data, batch_size)):
        params, opt_state, loss_i = step(
            params, static, *batch,
            optimizer=optimizer, opt_state=opt_state, loss_fn=loss_fn, key=subkey,
        )
    
    # Validation batches (no gradient updates)
    for batch in zip(*get_batches(val_data, batch_size)):
        loss_i = eqx.filter_jit(loss_fn)(params, static, *batch, key=subkey)
    
    # Early stopping
    if losses["val"][-1] == min(losses["val"]):
        best_params = params
    elif count_fruitless(losses["val"]) > max_patience:
        break

# 4. Recombine and return
dist = eqx.combine(params, static)
return dist, losses
```

### 1.3 Trainable vs Non-trainable Parameter Partitioning

FlowJAX uses **equinox + paramax** for parameter partitioning:

```python
params, static = eqx.partition(
    tree,
    eqx.is_inexact_array,                                      # Filter: only float arrays are params
    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable), # Treat NonTrainable as leaf
)
```

**How it works:**
- `eqx.partition(tree, filter_fn)` splits any PyTree into two: one with leaves matching the filter, the other with `None` in those positions
- `eqx.is_inexact_array` matches `float32`, `float64` arrays — integers, booleans, and non-array attributes become static
- `paramax.NonTrainable` is a wrapper that marks specific parameters as non-trainable even if they're float arrays
- Later, `eqx.combine(params, static)` merges them back

**This is the Equinox convention** — no explicit "trainable" flags. Instead:
- All float arrays are trainable by default
- Wrap with `NonTrainable(x)` to exclude specific parameters
- The tree structure itself determines what's a parameter

### 1.4 The `step()` Function

```python
@eqx.filter_jit
def step(params, *args, optimizer, opt_state, loss_fn, **kwargs):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, *args, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val
```

**Key details:**
- `eqx.filter_value_and_grad` computes gradients only w.r.t. the first argument (`params`)
- `optimizer.update` is the standard optax interface
- `eqx.apply_updates` handles the PyTree structure (adds updates to params leaf-by-leaf)
- The whole function is JIT-compiled via `@eqx.filter_jit`

### 1.5 Loss Functions

#### `MaximumLikelihoodLoss`

```python
class MaximumLikelihoodLoss:
    @eqx.filter_jit
    def __call__(self, params, static, x, condition=None, key=None):
        dist = paramax.unwrap(eqx.combine(params, static))
        return -dist.log_prob(x, condition).mean()
```

**Design:**
- Callable class (not a function) — allows stateful losses
- `paramax.unwrap` resolves any `AbstractUnwrappable` wrappers (like constrained parameters)
- Returns scalar: mean negative log-likelihood over the batch
- `key` is accepted but ignored (API consistency with stochastic losses)

#### `ElboLoss` (Variational Inference)

```python
class ElboLoss:
    target: Callable[[ArrayLike], Array]   # Log target density
    num_samples: int
    stick_the_landing: bool                # Lower-variance gradient estimator
    
    def __call__(self, params, static, key):
        dist = eqx.combine(params, static)
        samples, log_probs = dist.sample_and_log_prob(key, (self.num_samples,))
        target_density = vmap(self.target)(samples)
        return (log_probs - target_density).mean()
```

#### `ContrastiveLoss` (Simulation-Based Inference)

Uses contrastive samples from the batch itself for sequential neural posterior estimation.

### 1.6 `fit_to_key_based_loss()` — Simpler Training Loop

For losses that don't use data batches (e.g., ELBO where samples are generated):

```python
def fit_to_key_based_loss(key, tree, *, loss_fn, steps, learning_rate=5e-4, optimizer=None, show_progress=True):
    params, static = eqx.partition(tree, eqx.is_inexact_array,
                                    is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable))
    opt_state = optimizer.init(params)
    losses = []
    for key in jr.split(key, steps):
        params, opt_state, loss = step(params, static, key=key,
                                        optimizer=optimizer, opt_state=opt_state, loss_fn=loss_fn)
        losses.append(loss.item())
    return eqx.combine(params, static), losses
```

**Differences from `fit_to_data`:**
- No data batching or train/val split
- No early stopping — runs for fixed `steps`
- Loss signature is `(params, static, key)` instead of `(params, static, *arrays, key)`

### 1.7 Could FlowJAX's Training Loop Be Reused for Exponential Family MLE?

**Yes, with caveats:**

**What works directly:**
- The `step()` function is completely generic — any differentiable loss on any PyTree
- `MaximumLikelihoodLoss` is exactly negative mean log-likelihood
- Early stopping + train/val split is standard best practice
- Parameter partitioning via `eqx.partition` works for any Equinox module

**What would need adaptation for normix:**
1. **Closed-form MLE**: For exponential families, MLE has a closed-form solution via sufficient statistics: $\hat{\eta} = \frac{1}{n}\sum_i t(x_i)$, then convert $\eta \to \theta$. No gradient descent needed.
2. **EM algorithm**: FlowJAX's loop is pure SGD. EM requires alternating E-step (compute expectations) and M-step (maximize), which has different structure.
3. **Natural gradient**: For exponential families, natural gradient descent (using Fisher information) converges in one step. FlowJAX uses standard Adam.

**Recommendation:** Don't reuse FlowJAX's loop directly. Instead:
- Use closed-form MLE where possible (sufficient statistics → expectation params → natural params)
- For EM, implement custom E/M steps
- Consider natural gradient via Fisher information (as efax does)

---

## 2. efax Fitting/Optimization Module

**Repository:** https://github.com/NeilGirdhar/efax  
**Key insight:** efax doesn't have a `fit()` function. Instead, MLE is done purely through the exponential family's algebraic structure.

### 2.1 The MLE Pattern in efax

efax's MLE works in three steps:

```python
# Step 1: Compute sufficient statistics (= expectation parameters)
estimator = MaximumLikelihoodEstimator.create_estimator(exp_parameters)
sufficient_stats = estimator.sufficient_statistics(x)   # EP for each observation

# Step 2: Average to get MLE expectation parameters
mle_exp_params = parameter_mean(sufficient_stats, axis=sample_axes)

# Step 3: Convert expectation → natural parameters (the hard part)
mle_nat_params = mle_exp_params.to_nat()
```

**This is the exponential family closed-form MLE:**

$$ \hat{\eta} = \frac{1}{n} \sum_{i=1}^{n} t(x_i), \quad \hat{\theta} = (\nabla \psi)^{-1}(\hat{\eta}) $$

### 2.2 `ExpToNat` Mixin — Converting Expectation → Natural Parameters

This is the mathematically non-trivial part. For most exponential families, inverting $\nabla\psi(\theta) = \eta$ requires solving a nonlinear system.

```python
@dataclass
class ExpToNat(ExpectationParametrization[NP], SimpleDistribution, Generic[NP]):
    """Implements conversion from expectation to natural parameters.
    Uses Newton's method to invert the gradient of the log-normalizer.
    """
    minimizer: ExpToNatMinimizer | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.minimizer is None:
            initial_search_parameters = self.initial_search_parameters()
            # Newton for multivariate, bisection for univariate
            self.minimizer = (default_minimizer
                              if initial_search_parameters.shape[-1] > 1
                              else default_bisection_minimizer)

    @jit
    def to_nat(self) -> NP:
        flattener, flattened = Flattener.flatten(self)
        def solve(flattener, flattened):
            x = flattener.unflatten(flattened)
            return self.minimizer.solve(x)
        # vmap over batch dimensions
        for _ in range(self.ndim):
            solve = vmap(solve, in_axes=(None, 0))
        return self.search_to_natural(solve(flattener, flattened))
```

**Key methods that subclasses override:**

```python
def initial_search_parameters(self) -> JaxRealArray:
    """Starting point for the root-finding algorithm."""
    # Default: zeros. Subclasses can provide better initializations.

def search_to_natural(self, search_parameters) -> NP:
    """Convert search-space parameters to NaturalParametrization."""
    # Default: unflatten via Flattener with mapped_to_plane=True

def search_gradient(self, search_parameters) -> JaxRealArray:
    """The gradient = (current_exp - target_exp) in flattened space."""
    search_np = self.search_to_natural(search_parameters)
    search_ep = search_np.to_exp()           # nat → exp via grad(log_normalizer)
    _, self_flat = Flattener.flatten(self, map_to_plane=False)
    _, search_flat = Flattener.flatten(search_ep, map_to_plane=False)
    return search_flat - self_flat            # Residual to drive to zero
```

### 2.3 The Optimization: optimistix Root-Finding

```python
@dataclass
class OptimistixRootFinder(ExpToNatMinimizer):
    solver: RootFinder     # optx.Newton or optx.Bisection
    max_steps: int
    send_lower_and_upper: bool

    def solve(self, exp_to_nat: ExpToNat) -> JaxRealArray:
        def f(x, args):
            return exp_to_nat.search_gradient(x)  # Residual function
        
        initial = exp_to_nat.initial_search_parameters()
        results = optx.root_find(f, self.solver, initial, max_steps=self.max_steps, throw=False)
        return results.value

# Default solvers:
default_minimizer = OptimistixRootFinder(
    solver=optx.Newton(rtol=0.0, atol=1e-7),
    max_steps=1000)

default_bisection_minimizer = OptimistixRootFinder(
    solver=optx.Bisection(rtol=0.0, atol=1e-7, flip='detect', expand_if_necessary=True),
    max_steps=1000,
    send_lower_and_upper=True)
```

**How Newton's method works here:**
1. We want to find $\theta$ such that $\nabla\psi(\theta) = \hat{\eta}$ (the target expectation params)
2. Define residual $r(\theta) = \nabla\psi(\theta) - \hat{\eta}$
3. Newton step: $\theta_{k+1} = \theta_k - [\nabla^2\psi(\theta_k)]^{-1} r(\theta_k)$
4. But $\nabla^2\psi(\theta) = \text{Fisher Information Matrix}$!
5. So this is actually **natural gradient descent** on the KL divergence

### 2.4 Concrete Example: GammaEP

The Gamma distribution shows how subclasses customize the search:

```python
@dataclass
class GammaEP(HasEntropyEP[GammaNP], ExpToNat[GammaNP], SimpleDistribution):
    mean: JaxRealArray       # E[X]
    mean_log: JaxRealArray   # E[log X]

    def search_to_natural(self, search_parameters):
        shape = softplus(search_parameters[..., 0])  # Ensure shape > 0
        rate = shape / self.mean
        return GammaNP(-rate, shape - 1.0)

    def search_gradient(self, search_parameters):
        shape = softplus(search_parameters[..., 0])
        log_mean_minus_mean_log = log(self.mean) - self.mean_log
        return (log_mean_minus_mean_log - log(shape) + digamma(shape))[..., newaxis]

    def initial_search_parameters(self):
        # Analytic approximation for initial shape parameter
        s = log(self.mean) - self.mean_log
        initial_shape = (3 - s + sqrt((s-3)**2 + 24*s)) / (12*s)
        return inverse_softplus(initial_shape)[..., newaxis]
```

**Note:** The search is over a **reparametrized** 1D space (just the shape parameter), not the full 2D natural parameter space. The rate is determined analytically from the shape and the known mean. This is a key optimization — reduce the dimension of the root-finding problem.

### 2.5 Fisher Information Matrix

The Fisher information is computed via automatic differentiation of the log-normalizer:

```python
class NaturalParametrization:
    def _fisher_information_matrix(self):
        flattener, flattened = Flattener.flatten(self, map_to_plane=False)
        fisher_info_f = jacfwd(grad(self._flat_log_normalizer))
        for _ in range(len(self.shape)):
            fisher_info_f = vmap(fisher_info_f)
        return fisher_info_f(flattened, flattener)
```

This computes $\nabla^2\psi(\theta)$, the Hessian of the log-normalizer, which equals the Fisher information for exponential families. It uses `jacfwd(grad(...))` — forward-mode Jacobian of the gradient.

There's also an efficient Fisher-vector product via VJP:

```python
def apply_fisher_information(self, vector):
    """Efficiently apply Fisher information to a vector without materializing the matrix."""
    expectation_parameters, f_vjp = vjp(type(self).to_exp, self)
    return expectation_parameters, f_vjp(vector)
```

### 2.6 The `log_normalizer` Custom JVP

This is a critical numerical stability trick:

```python
def _log_normalizer_jvp(primals, tangents):
    """The log-normalizer's special JVP vastly improves numerical stability."""
    q, = primals
    q_dot, = tangents
    y = q.log_normalizer()        # Forward pass: compute log A(θ)
    p = q.to_exp()                # Compute expectation params η = ∇ψ(θ)
    y_dot = parameter_dot_product(q_dot, p)  # JVP: dy = <dθ, η>
    return y, y_dot

@abstract_custom_jvp(_log_normalizer_jvp)
def log_normalizer(self) -> JaxRealArray:
    ...
```

**Why this matters:**
- The gradient of `log_normalizer` w.r.t. natural parameters $\theta$ is the expectation parameters $\eta$
- By defining a custom JVP, the gradient is computed via `to_exp()` (which each distribution implements analytically) rather than through AD of the log-normalizer expression
- This avoids numerical issues from differentiating through `gammaln`, `log`, etc.
- The JVP rule says: $d\psi = \langle d\theta, \eta \rangle$ where $\eta = \nabla\psi(\theta)$

**Impact on optimization:**
- When computing `grad(log_pdf)` w.r.t. natural parameters, the custom JVP ensures the gradient is computed using the stable `to_exp()` path
- This propagates through to the Newton solver in `ExpToNat`, the Fisher information computation, and the KL divergence

### 2.7 `MaximumLikelihoodEstimator` Class

```python
@dataclass
class MaximumLikelihoodEstimator(Structure[P]):
    """Does maximum likelihood estimation. Stores structure and fixed parameters."""
    fixed_parameters: dict[Path, JaxComplexArray]

    @classmethod
    def create_estimator(cls, p):
        """Create from an ExpectationParametrization instance."""
        infos = cls.create(p).infos
        fixed_parameters = parameters(p, fixed=True)
        return cls(infos, fixed_parameters)

    def sufficient_statistics(self, x):
        """Compute sufficient statistics for observations."""
        # Delegates to NP.sufficient_statistics() for each sub-distribution
        # Returns ExpectationParametrization objects
```

**The MLE workflow in full:**

```python
# Given: observations x, a known distribution type
# 1. Create estimator
estimator = MaximumLikelihoodEstimator.create_estimator(some_exp_params)

# 2. Compute sufficient statistics T(x) for each observation
ss = estimator.sufficient_statistics(x)  # Returns EP with batch dim

# 3. Average to get η̂ = (1/n) Σ T(xᵢ)
eta_hat = parameter_mean(ss, axis=0)

# 4. Convert to natural parameters: θ̂ = (∇ψ)⁻¹(η̂)
theta_hat = eta_hat.to_nat()  # Uses ExpToNat Newton solver
```

### 2.8 No `fit()` Method — By Design

efax deliberately has **no** `dist.fit(X)` method. The reasoning:
- For exponential families, MLE is a mathematical operation (mean of sufficient statistics + parameter conversion), not an iterative optimization
- The user composes the pieces: `sufficient_statistics` → `parameter_mean` → `to_nat()`
- This is more flexible (e.g., weighted MLE, online updates, conjugate priors)

---

## 3. Design Patterns for Fitting

### 3.1 sklearn: `model.fit(X)` Returns Self

```python
class GaussianMixture:
    def fit(self, X, y=None):
        # ... EM algorithm ...
        self.means_ = ...
        self.covariances_ = ...
        self.converged_ = ...
        return self  # Method chaining
    
    def score(self, X, y=None):
        return self.score_samples(X).mean()
```

**Key conventions:**
- `fit()` mutates `self` and returns `self`
- Fitted attributes have trailing underscore (`means_`)
- `y=None` for unsupervised models
- Separate `fit()`, `predict()`, `score()`, `sample()`

### 3.2 PyTorch Distributions: No Fit (External Optimization)

```python
# PyTorch distributions are parameter containers, not fitters
dist = torch.distributions.Normal(loc=mu, scale=sigma)
log_prob = dist.log_prob(x)

# Fitting is done externally via autograd:
mu = torch.tensor(0.0, requires_grad=True)
sigma = torch.tensor(1.0, requires_grad=True)
optimizer = torch.optim.Adam([mu, sigma])

for batch in data_loader:
    dist = Normal(mu, softplus(sigma))
    loss = -dist.log_prob(batch).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Key pattern:** Distribution objects are ephemeral — recreated each iteration with current parameters.

### 3.3 JAX/Equinox Convention

The emerging JAX convention (from Equinox, FlowJAX, Diffrax, etc.) is:

```python
# 1. Model is an eqx.Module (immutable PyTree)
model = MyDistribution(params=initial_params)

# 2. Training is a pure function that returns new model
trained_model, losses = fit_to_data(key, model, data)

# 3. Or explicit loop:
params, static = eqx.partition(model, eqx.is_inexact_array)
opt_state = optimizer.init(params)

@eqx.filter_jit
def step(params, static, batch, opt_state):
    def loss_fn(p):
        model = eqx.combine(p, static)
        return -model.log_prob(batch).mean()
    loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss

for batch in data:
    params, opt_state, loss = step(params, static, batch, opt_state)

model = eqx.combine(params, static)
```

**Key principles:**
- **Immutable models** — no mutation, functions return new models
- **Partition/combine** — split trainable from static for gradient computation
- **filter_jit** — JIT-compile while handling non-arraylike leaves
- **External training loop** — the model doesn't know how to train itself

### 3.4 Comparison Table

| Aspect | sklearn | PyTorch | JAX/Equinox | efax |
|---|---|---|---|---|
| `fit()` method? | Yes, mutates self | No | No | No |
| Return type | `self` | N/A | `(new_model, losses)` | Composable ops |
| Parameters | Attributes | `nn.Parameter` | Float arrays | Dataclass fields |
| Trainable marking | N/A | `requires_grad` | `eqx.partition` filter | `distribution_parameter()` |
| Non-trainable | N/A | `.detach()` | `NonTrainable` wrapper | `fixed=True` metadata |
| Optimization | Internal (EM, etc.) | External (autograd) | External (optax) | Root-finding (Newton) |
| MLE | Built-in | Manual | Manual | `sufficient_statistics` + `to_nat()` |
| Immutable? | No (mutates) | No (mutates) | Yes | Yes (frozen dataclass) |

---

## 4. Implications for normix JAX Port

### 4.1 What to Borrow from FlowJAX

1. **`eqx.partition` / `eqx.combine`** for splitting trainable params — universal pattern
2. **Callable loss classes** (`MaximumLikelihoodLoss`) — clean, composable
3. **`step()` as a reusable JIT-compiled primitive** — generic optimization step
4. **Early stopping with patience** — practical for EM convergence monitoring
5. **Train/val split** — for EM with held-out log-likelihood monitoring

### 4.2 What to Borrow from efax

1. **`ExpToNat` mixin** — the mathematical core for exponential family MLE
2. **`log_normalizer` custom JVP** — critical numerical stability trick
3. **`sufficient_statistics` → `parameter_mean` → `to_nat()`** — the MLE pipeline
4. **Fisher information via `jacfwd(grad(log_normalizer))`** — for natural gradient
5. **Subclass-specific search spaces** (like Gamma's 1D search) — reduces root-finding dimension
6. **`MaximumLikelihoodEstimator`** — structural MLE that handles joint distributions

### 4.3 What normix Should Do Differently

1. **Keep sklearn-style `fit()` API** — normix's existing pattern (`fit()` returns `self`) is user-friendly and should be preserved. Internally it can use JAX-style immutable operations.

2. **Hybrid approach for MLE:**
   - Simple distributions (Exponential, Normal): closed-form via sufficient statistics
   - Complex distributions (GIG, GH): Newton solver via `optimistix` (like efax)
   - Mixture distributions: EM algorithm with Equinox-style parameter updates

3. **EM as an external loop** (not baked into the distribution):
   ```python
   def em_fit(key, joint_dist, data, *, max_iter=100, tol=1e-8):
       for i in range(max_iter):
           # E-step: compute expectations (uses dist.log_prob, etc.)
           # M-step: update params (uses sufficient stats + to_nat)
           joint_dist = joint_dist.with_params(new_params)  # Return new immutable object
       return joint_dist
   ```

4. **Custom JVP for log_partition** — adopt efax's trick for numerical stability when differentiating through Bessel functions etc.

### 4.4 Recommended Architecture

```
normix_jax/
├── distributions/           # Equinox modules (immutable)
│   ├── base.py              # ExponentialFamily(eqx.Module)
│   ├── gamma.py
│   ├── gig.py
│   └── ...
├── fitting/
│   ├── mle.py               # sufficient_stats() + exp_to_nat() closed-form MLE
│   ├── em.py                # EM algorithm for mixtures
│   ├── losses.py            # NegativeLogLikelihood, etc.
│   └── train_utils.py       # step(), batching (borrowed from FlowJAX)
├── parametrizations/
│   ├── natural.py           # Natural parameter containers
│   ├── expectation.py       # Expectation parameter containers
│   └── conversion.py        # exp_to_nat via Newton/optimistix
└── utils/
    ├── bessel.py            # Bessel functions with custom JVPs
    └── fisher.py            # Fisher information computation
```
