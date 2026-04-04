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
│   ├── generalized_inverse_gaussian.py  # GeneralizedInverseGaussian / GIG(p, a, b)
│   ├── normal.py                 # MultivariateNormal(μ, L_Sigma)
│   ├── variance_gamma.py         # VarianceGamma / JointVarianceGamma
│   ├── normal_inverse_gamma.py   # NormalInverseGamma / JointNormalInverseGamma
│   ├── normal_inverse_gaussian.py # NormalInverseGaussian / JointNormalInverseGaussian
│   └── generalized_hyperbolic.py # GeneralizedHyperbolic / JointGeneralizedHyperbolic
├── mixtures/
│   ├── joint.py                  # JointNormalMixture(ExponentialFamily)
│   └── marginal.py               # NormalMixture (owns a JointNormalMixture)
├── fitting/
│   ├── em.py                     # EMResult; BatchEMFitter, IncrementalEMFitter
│   ├── eta.py                    # NormalMixtureEta, affine_combine
│   ├── eta_rules.py              # EtaUpdateRule (eqx.Module) and concrete rules
│   └── solvers.py                # solve_bregman*, BregmanResult (η→θ Bregman minimisation)
└── utils/
    ├── bessel.py                 # log_kv with custom_jvp
    ├── constants.py              # LOG_EPS, TINY, BESSEL_EPS_V, GIG_DEGEN_THRESHOLD, ...
    ├── plotting.py               # Notebook helpers
    └── validation.py             # Moment checks, pretty-print (notebooks)
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
    # Tier 1 — Subclass implements (pure, unbatched):
    @abstractmethod
    def _log_partition_from_theta(theta: Array) -> Array: ...
    @abstractmethod
    def natural_params(self) -> Array: ...
    @abstractmethod
    def sufficient_statistics(x: Array) -> Array: ...
    @abstractmethod
    def log_base_measure(x: Array) -> Array: ...

    # Tier 2 — JAX grad/Hessian (override for analytical formulas):
    @classmethod
    def _grad_log_partition(cls, theta: Array) -> Array:
        return jax.grad(cls._log_partition_from_theta)(theta)
    @classmethod
    def _hessian_log_partition(cls, theta: Array) -> Array:
        return jax.hessian(cls._log_partition_from_theta)(theta)

    # Tier 3 — CPU versions (override for scipy/Bessel-heavy distributions):
    @classmethod
    def _log_partition_cpu(cls, theta) -> float: ...       # wraps JAX
    @classmethod
    def _grad_log_partition_cpu(cls, theta) -> ndarray: ...  # wraps JAX
    @classmethod
    def _hessian_log_partition_cpu(cls, theta) -> ndarray: ...  # wraps JAX

    # Derived automatically via triad:
    def expectation_params(self, backend='jax') -> Array: ...
    def fisher_information(self, backend='jax') -> Array: ...
    def log_prob(self, x: Array) -> Array: ...
```

Every derived quantity comes from differentiating `_log_partition_from_theta`. The triad (`_grad_log_partition`, `_hessian_log_partition` plus CPU variants) separates the mathematical formula from the evaluation backend, enabling the solver to work in both JAX and CPU modes without distribution-specific knowledge.

**Three constructors plus fitting helpers:**
- `from_classical(...)` — from shape/rate/mean/etc. (readable English names)
- `from_natural(theta)` — from natural parameters θ
- `from_expectation(eta, *, backend, method, theta0, maxiter, tol, verbose)` — solves ∇ψ(θ)=η via `solve_bregman`; passes `grad_fn`/`hess_fn` from triad. If `theta0` is omitted, uses **`jnp.zeros_like(eta)`** (no separate `_init_theta_from_eta`).
- `fit_mle(X, *, theta0, maxiter, tol, verbose)` — MLE: η̂ = mean_i t(xᵢ), then `from_expectation`
- `default_init(X)` — same η̂ path as MLE; for closed-form `from_expectation` subclasses this is the MLE; otherwise a reasonable cold start
- `fit(self, X, *, ...)` — instance method: η̂ from data, **`theta0=self.natural_params()`** for warm-started η→θ (same kwargs as `from_expectation` aside from `theta0`)

**Bregman divergence:**  `bregman_divergence(theta, eta)` = ψ(θ) − θ·η. Minimising over θ gives ∇ψ(θ*) = η. Available as a class method on all `ExponentialFamily` subclasses and as the universal `fitting.solvers.bregman_objective` for use with any log-partition function.

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

### Joint classes as public exponential families (D2)

**Decision:** `JointVarianceGamma`, `JointNormalInverseGamma`, `JointNormalInverseGaussian`, `JointGeneralizedHyperbolic`, and the abstract `JointNormalMixture` are **first-class public** `ExponentialFamily` objects: exported from `normix`, documented alongside marginals, and intended for direct use where the joint law :math:`f(x,y)` matters (simulation, complete-data MLE, divergences, custom EM variants).

**Observation vector for Tier-1 methods:** `sufficient_statistics`, `log_base_measure`, and inherited `log_prob` / `pdf` take a **single** flat array `xy = jnp.concatenate([x, y])` with `x` of shape `(d,)` and scalar `y > 0` last. This matches the sufficient-statistic block :math:`[\log y,\,1/y,\,y,\,\ldots]` in `mixtures/joint.py`. For readability, `log_prob_joint(x, y)` and `rvs(n, seed) -> (X, Y)` remain the preferred entry points when you already have :math:`x` and :math:`y` separate.

**Constructors:** Use `from_classical(...)` (and, for analysis, `natural_params()` / `expectation_params()` / the log-partition triad). `from_natural` is **not** implemented on concrete joints (it would require a joint constrained inverse map); `from_expectation` on the full joint is therefore unsupported until such a constructor exists. Marginal `NormalMixture` subclasses continue to drive EM via `e_step` / `m_step` and classical parameters.

**Correctness:** For GIG-based joints, natural parameters :math:`\theta_2,\theta_3` must align with the **GIG** convention :math:`\theta_{\mathrm{GIG}} = [p-1,\,-b/2,\,-a/2]` on :math:`[\log y,\,1/y,\,y]`, i.e. scalar coefficients :math:`-(b/2 + \cdots)` and :math:`-(a/2 + \cdots)` on :math:`1/y` and :math:`y`, not :math:`-b` and :math:`-a`. Gamma and inverse-gamma joints already matched this limit; generalized hyperbolic and NIG joints follow the same pattern.

---

## EM Framework: Model/Fitter Separation

Following GMMX: the model knows math, the fitter knows iteration.

**`EMResult`** (frozen dataclass): `model`, optional `log_likelihoods`, `param_changes`, `n_iter`, `converged`, `elapsed_time`.

Fitters are plain Python classes (configuration + `fit`), **not** `eqx.Module`s:

- **`BatchEMFitter`** — full-dataset EM with convergence monitoring. Supports an optional `eta_update` rule for penalised / shrinkage EM.
- **`IncrementalEMFitter`** — mini-batch / online / fine-tuning EM with configurable η-update rules and fixed iteration budget. Replaces the former `OnlineEMFitter` and `MiniBatchEMFitter`.

Both return `EMResult`.

**EM steps on `NormalMixture`:**
- `e_step(X, backend='jax'|'cpu') -> NormalMixtureEta` — subordinator conditionals + batch aggregation into a 6-field pytree
- `m_step(eta, **kwargs) -> NormalMixture` — full M-step (normal params + subordinator) from aggregated η
- `m_step_normal(eta) -> NormalMixture` — MCECM cycle 1: update μ, γ, Σ only
- `m_step_subordinator(eta, **kwargs) -> NormalMixture` — MCECM cycle 2: update subordinator only
- `compute_eta_from_model() -> NormalMixtureEta` — reconstruct η from current parameters (initialises incremental EM)
- `fit(X, **kwargs) -> EMResult` — convenience: `BatchEMFitter(**kwargs).fit(self, X)` using **`self` as initialization**
- `default_init(X)` — moment-based starting model; pair with `model.fit(X)` for cold start

**`NormalMixtureEta`** — the six batch-averaged sufficient statistics as an `eqx.Module` pytree: `E_log_Y` (scalar), `E_inv_Y` (scalar), `E_Y` (scalar), `E_X` (d,), `E_X_inv_Y` (d,), `E_XXT_inv_Y` (d,d). This is the expectation parametrization of the joint distribution — $\hat\eta = E[t(X,Y)|X]$ averaged over observations. The pytree structure (vs. a flat vector) keeps heterogeneous shapes readable and makes `jax.tree.map` apply affine weights naturally.

**`affine_combine(eta_prev, eta_new, b, c, a=None)`** — the single operation behind all η-update rules: $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta$. JIT-able (pure function on pytrees).

**`EtaUpdateRule`** (`eqx.Module` ABC) — each concrete rule computes $(a, b, c)$ from step index, batch size, and state. Hyperparameters are JAX array leaves — JIT-compatible and differentiable for meta-learning step-size schedules. Concrete rules: `IdentityUpdate`, `RobbinsMonroUpdate`, `SampleWeightedUpdate`, `EWMAUpdate`, `ShrinkageUpdate`, `AffineUpdate`.

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

### CPU versions for Bessel-dependent functions

**Design rule:** any distribution that calls `log_kv` must override the Tier 3 CPU triad classmethods so that the CPU solver path (`solve_bregman(backend='cpu')`) avoids JAX dispatch entirely. The three classmethods to override are `_log_partition_cpu`, `_grad_log_partition_cpu`, and `_hessian_log_partition_cpu` — all accepting and returning numpy arrays.

Distributions that do not call `log_kv` (Gamma, InverseGamma, InverseGaussian) inherit default Tier 3 implementations that simply wrap the JAX versions. This is sufficient since these distributions do not involve expensive JAX-to-numpy conversions in the solver hot path.

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

### Solvers (general + GIG-specific)

`fitting/solvers.py` provides universal Bregman divergence solvers for any exponential family:
- `solve_bregman(f, eta, theta0, *, backend, method, verbose=0, ...)` — single starting point
- `solve_bregman_multistart(f, eta, theta0_batch, *, backend, method, verbose=0, ...)` — best of K starts

Both return **`BregmanResult`**: `theta`, `fun`, `grad_norm`, `num_steps`, `converged`, **`elapsed_time`** (wall clock). Some scalar fields are typed loosely (`Any`) so a `BregmanResult` can be nested inside **`lax.scan`** (e.g. JAX Newton path) without `ConcretizationTypeError` from forcing Python `float`/`bool`.

Solver axes: `backend='jax'|'cpu'` × `method='newton'|'lbfgs'|'bfgs'`.

For GIG the preferred warm-start solver is `backend='cpu', method='lbfgs'` (scipy L-BFGS-B + `scipy.kve`), which avoids JAX GPU kernel dispatch overhead on the 3-dimensional scalar problem. **`verbose`** is threaded from `GIG.from_expectation` / multistart into `solve_bregman` for solver-side logging.

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
| Analytical overrides | Classmethods `_grad_log_partition`, `_hessian_log_partition` | Separates formula from evaluation; override without affecting autodiff |
| Unbatched core | `log_prob(x)` for single obs | Clean; batch via `jax.vmap` at call site |
| Mixture design | Joint + Marginal classes | Joint IS an exponential family; marginal is not |
| Joint public API (D2) | Public `ExponentialFamily` exports | Joints are first-class; `log_prob` uses flat `concat(x,[y])`; `from_natural` on joints unimplemented by design until a robust inverse exists |
| EM separation | Model + Fitter (GMMX-style) | Swap fitter without changing distribution |
| EM return value | `EMResult` (not bare model) | Diagnostics, timing, optional LL trace; `result.model` is the fitted pytree |
| Batch EM convergence | Relative change in μ, γ, L (not GIG) | Stable criterion; GIG η→θ has its own solver tolerance |
| Fitter classes | `BatchEMFitter` + `IncrementalEMFitter` | Batch has convergence monitoring; incremental has fixed budget |
| η-update rules | Affine combination $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta$ | Unifies online, EWMA, shrinkage, sample-weighted in one abstraction |
| `EtaUpdateRule` | `eqx.Module` (not plain ABC) | Hyperparameters are JAX array leaves — JIT-compatible, differentiable for future meta-learning |
| `NormalMixtureEta` | Pytree of 6 heterogeneous-shape components | Readable; `jax.tree.map` applies scalar weights naturally |
| `lax.scan` EM | When both EM backends JAX and low verbosity | JIT-friendly full batch EM; else Python loop for CPU / verbose tables |
| Cold vs warm η→θ | `default_init` / `theta0=None` vs `fit(self,X)` | Zeros-like default in `from_expectation`; instance `fit` uses `natural_params()` |
| Bessel | Pure-JAX + CPU backend | JAX for JIT/autodiff; CPU for EM performance |
| η→θ solver | η-rescaled + CPU L-BFGS-B | Ill-conditioning requires rescaling; CPU avoids GPU overhead |
| Constraints | `jnp.maximum(x, LOG_EPS)` | Simpler than paramax; EM doesn't need grad through constraints |
| Precision | Float64 throughout | Bessel functions and EM convergence require double precision |
| Bregman divergence | `fitting/solvers.py` universal solvers | Decouple optimization from distribution math |
| Solver interface | `grad_fn` + `hess_fn` (θ-space) not phi-space | Distributions provide math primitives; solver handles reparametrization |
| `jnp.where` in log-partition | Not `lax.cond` | `jnp.where` is vmap-compatible; clamping prevents NaN gradients |
| Numerical constants | Centralized in `utils/constants.py` | Single source of truth; no scattered magic numbers |
| Class naming | `GeneralizedInverseGaussian` primary, `GIG` alias | Full name is canonical; short alias for backward compatibility |
| Module-level functions | Never; use classmethods or staticmethods | Keeps the interface on the class; avoids scattered module globals |

---

## MultivariateNormal as ExponentialFamily (D3)

**Decision (Phase 7):** `MultivariateNormal` is promoted from a plain `eqx.Module` to a full `ExponentialFamily` subclass.

**EF structure:**

| Component | Expression |
|-----------|-----------|
| $t(x)$ | $[x,\; \operatorname{vec}(xx^\top)]$ — shape $(d + d^2,)$ |
| $\theta$ | $[\Sigma^{-1}\mu,\; -\tfrac{1}{2}\operatorname{vec}(\Sigma^{-1})]$ — same shape |
| $\log h(x)$ | $0$ |
| $\psi(\theta)$ | $\tfrac{1}{2}\theta_1^\top\Lambda^{-1}\theta_1 - \tfrac{1}{2}\log\|\Lambda\| + \tfrac{d}{2}\log(2\pi)$ where $\Lambda = -2\,\mathrm{reshape}(\theta_2, d, d)$ |

`vec` uses row-major (`ravel()`) throughout. All parametrization conversions are analytical (closed-form) — no Bregman solver is ever invoked:

| Conversion | Method | Formula |
|---|---|---|
| classical → natural | `natural_params()` | $\theta_1 = \Sigma^{-1}\mu$, $\theta_2 = -\tfrac{1}{2}\operatorname{vec}(\Sigma^{-1})$ |
| natural → classical | `from_natural(θ)` | $\Lambda = -2\,\mathrm{reshape}(\theta_2)$, $\mu = \Lambda^{-1}\theta_1$, $\Sigma = \Lambda^{-1}$ |
| natural → expectation | `_grad_log_partition(θ)` | $\eta = [\mu,\; \operatorname{vec}(\Sigma + \mu\mu^\top)]$ (analytical Tier 2 override) |
| expectation → classical | `from_expectation(η)` | $\mu = \eta_1$, $\Sigma = \mathrm{reshape}(\eta_2) - \mu\mu^\top$ |

`_log_partition_from_theta` uses the Cholesky of $\Lambda$ for numerical stability. `log_prob` overrides the inherited EF formula with the direct Cholesky path (more efficient). `fit_mle` computes $\hat\eta = n^{-1}\sum_i t(x_i)$ then calls the closed-form `from_expectation`.

Added: `mean()`, `cov()`, `rvs(n, seed)`. Legacy `sample(key, shape)` kept for notebook backward compatibility.

---

## jaxopt migration (D4)

**Evaluation (Phase 7):** jaxopt is unmaintained upstream (last release 0.8.3, 2024) and emits a `DeprecationWarning` on import. In `normix`, jaxopt is used exclusively in `fitting/solvers.py` for `jaxopt.LBFGS` and `jaxopt.BFGS` in the `backend='jax', method='lbfgs'|'bfgs'` solver paths. The `LBFGSB` variant is deliberately avoided (int32/int64 dtype mismatch with `jax_enable_x64`).

**Why not migrate now:**
- `optax` absorbed L-BFGS from jaxopt but provides only an update rule (`optax.scale_by_lbfgs`), not a ready-made convergence-monitored solver loop. Writing that loop correctly (gradient norm stopping, step size scheduling) adds ~100–150 lines of non-trivial code.
- `optimistix` (Kidger) has `LBFGS` but lacks box constraints; it would require the same reparameterization we already do, and its convergence API differs from jaxopt's. Not yet production-mature for this use case.
- jaxopt still functions correctly despite the deprecation warning.

**Migration path (when jaxopt breaks or is removed from PyPI):**
1. Implement a thin `_lbfgs_loop(f, phi0, max_steps, tol)` using `optax.scale_by_lbfgs` + `jax.lax.while_loop` (~100 lines) — see `fitting/solvers.py` docstring for the reparameterization convention.
2. Replace the `jaxopt.LBFGS(...)` / `jaxopt.BFGS(...)` calls in `_jax_quasi_newton` with this loop.
3. Drop `jaxopt` from `pyproject.toml` dependencies.

**Current mitigation:** The `DeprecationWarning` is suppressed in `normix/__init__.py` so library users are not affected.
