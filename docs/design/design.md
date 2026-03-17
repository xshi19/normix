# normix Design

## Philosophy

**Minimalist, immutable, autodiff-first.** Each concept has one canonical representation; no redundant methods, wrappers, or indirection. Immutable pytrees everywhere — no mutation, no caching, no `_fitted` flags.

When evaluating changes, weigh the complexity cost against the improvement magnitude. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a simplification win.

---

## Core Dependencies

| Package | Role |
|---|---|
| `jax` | Array computation, autodiff, JIT, vmap |
| `equinox` | Pytree-based modules (immutable, filterable) |
| `jaxopt` | L-BFGS-B for GIG η→θ constrained optimization |
| `scipy` | CPU Bessel evaluation via `kve` (EM hot path only) |

---

## Architecture

```
normix/
├── exponential_family.py         # ExponentialFamily(eqx.Module) base class
├── distributions/
│   ├── gamma.py                  # Gamma(α, β)
│   ├── inverse_gamma.py          # InverseGamma(α, β)
│   ├── inverse_gaussian.py       # InverseGaussian(μ, λ)
│   ├── generalized_inverse_gaussian.py  # GIG(p, a, b)
│   ├── normal.py                 # MultivariateNormal(μ, L_Sigma)
│   ├── variance_gamma.py         # VarianceGamma / JointVarianceGamma
│   ├── normal_inverse_gamma.py   # NormalInverseGamma / JointNormalInverseGamma
│   ├── normal_inverse_gaussian.py # NormalInverseGaussian / JointNormalInverseGaussian
│   └── generalized_hyperbolic.py # GeneralizedHyperbolic / JointGeneralizedHyperbolic
├── mixtures/
│   ├── joint.py                  # JointNormalMixture(ExponentialFamily)
│   └── marginal.py               # NormalMixture (owns a JointNormalMixture)
├── fitting/
│   └── em.py                     # BatchEMFitter, OnlineEMFitter, MiniBatchEMFitter
└── utils/
    ├── bessel.py                 # log_kv with custom_jvp
    ├── constants.py              # LOG_EPS, GIG_EPS_V_HESS, GIG_EPS_NP
    ├── plotting.py               # Notebook helpers
    └── validation.py             # EM validation helpers
```

---

## Equinox: Why Immutable Pytrees

`eqx.Module` is a frozen dataclass that is automatically a JAX pytree. After `__init__`, all attribute assignment is blocked. This makes modules inherently immutable — matching JAX's functional paradigm.

**Why not mutable (Flax NNX)?** Distributions are mathematical objects. Immutability matches their semantics. There's no mutable state (no batch norm, no running averages). FlowJAX chose Equinox for the same reason.

**Parameter updates** create a new instance via `eqx.tree_at`:
```python
new_dist = eqx.tree_at(lambda d: d.mu, dist, new_mu)
```

**Static fields** (config, strings) use `eqx.field(static=True)` — not JAX leaves, changing them triggers recompilation.

---

## Exponential Family Base Class

Three parametrizations, all derived from one function:

```python
class ExponentialFamily(eqx.Module):
    # Subclass implements (all pure, unbatched):
    @abstractmethod
    def _log_partition_from_theta(self, theta: Array) -> Array: ...
    @abstractmethod
    def natural_params(self) -> Array: ...
    @abstractmethod
    def sufficient_statistics(self, x: Array) -> Array: ...
    @abstractmethod
    def log_base_measure(self, x: Array) -> Array: ...

    # Derived automatically via JAX autodiff:
    def expectation_params(self) -> Array:
        return jax.grad(self._log_partition_from_theta)(self.natural_params())
    def fisher_information(self) -> Array:
        return jax.hessian(self._log_partition_from_theta)(self.natural_params())
    def log_prob(self, x: Array) -> Array:
        theta = self.natural_params()
        return (self.log_base_measure(x)
                + jnp.dot(self.sufficient_statistics(x), theta)
                - self._log_partition_from_theta(theta))
```

Every derived quantity comes from differentiating `_log_partition_from_theta`. No separate `_natural_to_expectation` unless the analytical form is both faster **and** registered via `custom_jvp`.

**Three constructors:**
- `from_classical(...)` — from shape/rate/mean/etc. (readable English names)
- `from_natural(theta)` — from natural parameters θ
- `from_expectation(eta, *, theta0, maxiter, tol)` — solves ∇ψ(θ)=η via jaxopt.LBFGS-B
- `fit_mle(X, *, theta0, maxiter, tol)` — MLE via exponential family identity: η̂ = mean_i t(xᵢ)

---

## Mixture Architecture

The Generalized Hyperbolic family is a normal variance-mean mixture:

$$X \mid Y \sim \mathcal{N}(\mu + \gamma y,\, \Sigma y), \quad Y \sim \text{subordinator}$$

**Two-class design, not one:**

```
JointNormalMixture(ExponentialFamily)   — f(x, y), IS an exponential family
    ↑ inherits
JointVarianceGamma, JointNIG, etc.      — concrete joints

NormalMixture(eqx.Module)               — f(x) = ∫f(x,y)dy, NOT an exp. family
    owns a JointNormalMixture
    ↑ inherits
VarianceGamma, NIG, NInvG, GH           — concrete marginals
```

**Why two classes?** The joint is an exponential family (its natural/sufficient parameters have closed-form expressions, EM is exact). The marginal is not — it requires numerical integration (Bessel functions). Separating them keeps each class focused.

The `JointNormalMixture` provides:
- `conditional_expectations(x)` → E[log Y|X=x], E[1/Y|X=x], E[Y|X=x] (EM E-step)
- `_mstep_normal_params(E_X, E_X/Y, E_XX'/Y, E_1/Y, E_Y)` → μ, γ, L_Sigma (closed form)
- `_quad_forms(x)` → z=L_Sigma⁻¹(x-μ), w=L_Sigma⁻¹γ and their norms (hot path)

Each concrete joint implements `_compute_posterior_expectations(x)` which computes posterior GIG parameters and returns expectations.

---

## EM Framework: Model/Fitter Separation

Following GMMX: the model knows math, the fitter knows iteration.

```python
class BatchEMFitter(eqx.Module):
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)

    def fit(self, model: NormalMixture, X: Array) -> NormalMixture:
        """Batch EM via jax.lax.while_loop. Returns new model."""
        ...
```

The same `NormalMixture` model works with `BatchEMFitter`, `OnlineEMFitter`, or `MiniBatchEMFitter`. The E-step and M-step are methods on the model; the fitter only controls the loop.

**EM steps on `NormalMixture`:**
- `e_step(X, backend='jax'|'cpu')` — computes conditional expectations via vmap
- `m_step(X, expectations)` → new `NormalMixture` with updated parameters

---

## Bessel Functions

`log_kv(v, z, backend='jax'|'cpu')` in `normix/utils/bessel.py`.

### Pure-JAX backend (default)

Regime selection via `lax.cond` (only the selected branch executes at runtime):
1. `z > max(25, v²/4)` → Hankel asymptotic (DLMF 10.40.2)
2. `|v| > 25` (not Hankel) → Olver uniform expansion (DLMF 10.41.3-4)
3. `z < 1e-6, |v| > 0.5` → Small-z leading asymptotic
4. Otherwise → 64-point Gauss-Legendre quadrature (Takekawa 2022)

Custom JVP via `@jax.custom_jvp`:
- ∂/∂z: exact recurrence $K'_\nu = -(K_{\nu-1} + K_{\nu+1})/2$
- ∂/∂ν: central FD with ε = 10⁻⁵

### CPU backend (EM hot path)

`scipy.special.kve`, fully vectorized NumPy. Not JIT-able. For the EM hot path with large N, a single `kve` C-call per element is faster than vmapping JAX's `lax.cond` (which causes separate GPU kernel launches per condition check).

**Why `backend` is Python-level:** The string is resolved before JAX tracing begins. When `backend='jax'`, all code is traceable. When `backend='cpu'`, the code runs eagerly — appropriate since EM loops are already Python `for` loops.

See `docs/tech_notes/bessel_implementations_survey.md` and `docs/tech_notes/em_gpu_profiling.md` for benchmarks.

---

## GIG η→θ Optimization

Given $\eta = (E[\log Y], E[1/Y], E[Y])$, find θ such that $\nabla\psi(\theta) = \eta$.

### Why this is hard

The GIG Fisher information can be extremely ill-conditioned when $a \ll b$ or $a \gg b$ (condition numbers up to $10^{30}$). Standard LBFGS-B fails without rescaling.

### η-Rescaling (reduces condition number)

Before optimization, rescale to a symmetric GIG:
$$s = \sqrt{\eta_2/\eta_3}, \quad \tilde\eta = \bigl(\eta_1 + \tfrac{1}{2}\log s^2,\; \sqrt{\eta_2\eta_3},\; \sqrt{\eta_2\eta_3}\bigr)$$

The scaled GIG has $a' = b' = \sqrt{ab}$, symmetric Fisher matrix. After solving $\tilde\theta$:
$$\theta = (\tilde\theta_1,\; \tilde\theta_2/s,\; s\tilde\theta_3)$$

### Solver

`jaxopt.LBFGSB` minimizes $\psi(\theta) - \theta\cdot\eta$ with bounds $\theta_2 \leq 0$, $\theta_3 \leq 0$.

CPU fallback uses `scipy.optimize.minimize` with scipy.kve (much faster per call).

See `docs/tech_notes/gig_eta_to_theta.md` for derivations and benchmark comparisons.

---

## CPU/GPU Hybrid Backend

EM timing on 468 stocks, 2552 observations (GH distribution):

| Phase | JAX (GPU) | CPU hybrid | Speedup |
|---|---|---|---|
| E-step | ~1.1s | ~0.07s | ~15× |
| M-step (GIG solve) | ~5–7s | ~0.01s | ~500× |

**Hybrid strategy:**
- Quad forms ($L_\Sigma^{-1}(x-\mu)$ etc.) stay in JAX (d-dimensional, GPU-friendly)
- `log_kv` calls and GIG optimization move to CPU (`backend='cpu'`)

See `docs/tech_notes/em_gpu_profiling.md`.

---

## Design Decision Table

| Decision | Choice | Why |
|---|---|---|
| Base module | `eqx.Module` (not Flax NNX) | Immutable matches math semantics; no mutable state needed |
| Parametrizations | One class, three constructors | Avoids class explosion; common operations share code |
| Autodiff | `jax.grad` on `_log_partition_from_theta` | Single source of truth; no sync bugs |
| Analytical overrides | Only with `custom_jvp` | Keeps autodiff correct at all orders |
| Unbatched core | `log_prob(x)` for single obs | Clean; batch via `jax.vmap` at call site |
| Mixture design | Joint + Marginal classes | Joint IS an exponential family; marginal is not |
| EM separation | Model + Fitter (GMMX-style) | Swap fitter without changing distribution |
| Bessel | Pure-JAX + CPU backend | JAX for JIT/autodiff; CPU for EM performance |
| η→θ solver | η-rescaled LBFGS-B | Ill-conditioning requires rescaling for convergence |
| Constraints | `jnp.maximum(x, LOG_EPS)` | Simpler than paramax; EM doesn't need grad through constraints |
| Precision | Float64 throughout | Bessel functions and EM convergence require double precision |
