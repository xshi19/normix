# Architecture

> Quick reference: `AGENTS.md`. Full design rationale: `docs/design/jax_design.md`.

## Module Hierarchy

```
normix/                     # JAX implementation (current)
├── _bessel.py              # log_kv(v, z, backend='jax'|'cpu') — pure-JAX + scipy paths
├── _types.py               # type aliases
├── exponential_family.py   # ExponentialFamily(eqx.Module)
├── distributions/
│   ├── gamma.py
│   ├── inverse_gamma.py
│   ├── inverse_gaussian.py
│   ├── gig.py              # GIG / GeneralizedInverseGaussian
│   ├── normal.py           # MultivariateNormal
│   ├── variance_gamma.py
│   ├── normal_inverse_gamma.py
│   ├── normal_inverse_gaussian.py
│   └── generalized_hyperbolic.py
├── mixtures/
│   ├── joint.py            # JointNormalMixture(ExponentialFamily)
│   └── marginal.py         # NormalMixture (owns a JointNormalMixture)
├── fitting/
│   ├── em.py               # BatchEMFitter, OnlineEMFitter, MiniBatchEMFitter
│   └── __init__.py
└── utils/
    ├── __init__.py          # re-exports from plotting and validation
    ├── plotting.py          # notebook plotting helpers (golden-ratio figures)
    └── validation.py        # moment validation, EM runner, parameter printing

normix_numpy/               # NumPy/SciPy reference implementation (preserved)
```

## Exponential Family Core

Every distribution subclasses `ExponentialFamily(eqx.Module)` and implements four abstract methods:

| Method | Purpose |
|---|---|
| `_log_partition_from_theta(theta)` | Log partition function $\psi(\theta)$ — **single source of truth** |
| `natural_params()` | Compute $\theta$ from stored classical parameters |
| `sufficient_statistics(x)` | Compute $t(x)$ for a single observation |
| `log_base_measure(x)` | Compute $\log h(x)$ |

Everything else is derived automatically via JAX autodiff:

- `log_partition()` = `_log_partition_from_theta(natural_params())`
- `expectation_params()` = `jax.grad(_log_partition_from_theta)(θ)` → $\eta = \nabla\psi(\theta)$
- `fisher_information()` = `jax.hessian(_log_partition_from_theta)(θ)` → $I(\theta) = \nabla^2\psi(\theta)$
- `log_prob(x)` = `log_base_measure(x) + t(x)·θ − ψ(θ)`
- `pdf(x)` = `exp(log_prob(x))`
- `cdf(x)` — analytical where available (Gamma, InverseGamma, InverseGaussian); otherwise `NotImplementedError`
- `mean()`, `var()`, `std()` — analytical formulas per distribution
- `rvs(n, seed)` — numpy/scipy-based sampling (not JIT-able)

No separate analytical overrides unless both (a) faster and (b) registered via `custom_jvp` so higher-order autodiff remains correct.

### Marginal Distribution Methods

`NormalMixture` provides:
- `mean()` = $\mu + \gamma E[Y]$
- `cov()` = $E[Y]\Sigma + \text{Var}[Y]\gamma\gamma^\top$
- `rvs(n, seed)` — samples X from the marginal
- `pdf(x)` = `exp(log_prob(x))`

`JointNormalMixture` provides:
- `rvs(n, seed)` → `(X, Y)` — samples both X and Y jointly

### Three Parametrizations

```
classical (α, β, μ, L, ...)
    ↕  from_classical / natural_params
natural θ
    ↕  from_natural / expectation_params (jax.grad)
expectation η = E[t(X)]
    ↕  from_expectation (jaxopt.LBFGSB solve)
```

All conversions are JIT-compatible and support `jax.grad` and `jax.vmap`.

### Distribution Storage

Each distribution stores canonical parameters as named `eqx.Module` fields — the minimal set from which everything else is computable. No redundant storage.

| Distribution | Stored Attributes | Notes |
|---|---|---|
| Gamma | `alpha`, `beta` | shape, rate |
| InverseGamma | `alpha`, `beta` | shape, rate |
| InverseGaussian | `mu`, `lam` | mean, shape |
| GIG | `p`, `a`, `b` | shape, rate, rate |
| MultivariateNormal | `mu`, `L` | mean, Cholesky of Σ |
| VarianceGamma | `mu`, `gamma`, `L`, `alpha`, `beta` | Gamma subordinator |
| NormalInverseGamma | `mu`, `gamma`, `L`, `alpha`, `beta` | InverseGamma subordinator |
| NormalInverseGaussian | `mu`, `gamma`, `L`, `delta`, `eta` | InverseGaussian subordinator |
| GeneralizedHyperbolic | `mu`, `gamma`, `L`, `p`, `a`, `b` | GIG subordinator |

## Mixture Structure

```
JointNormalMixture(ExponentialFamily)     f(x,y) — X|Y ~ N(μ + γy, Σy)
    ├── JointVarianceGamma                Y ~ Gamma
    ├── JointNormalInverseGamma           Y ~ InverseGamma
    ├── JointNormalInverseGaussian        Y ~ InverseGaussian
    └── JointGeneralizedHyperbolic        Y ~ GIG

NormalMixture(eqx.Module)                f(x) = ∫ f(x,y) dy
    ├── VarianceGamma
    ├── NormalInverseGamma
    ├── NormalInverseGaussian
    └── GeneralizedHyperbolic
```

`NormalMixture` owns a `JointNormalMixture`. The joint is an exponential family; the marginal is not.

## EM Algorithm

The model knows math; the fitter knows iteration (following GMMX).

- **E-step**: `model.e_step(X, backend='jax'|'cpu')` computes $E[Y|X]$, $E[1/Y|X]$, $E[\log Y|X]$
  - `backend='jax'` (default): `jax.vmap(joint.conditional_expectations)(X)` — JIT-able
  - `backend='cpu'`: quad forms stay in JAX (vmapped), Bessel goes to CPU via `scipy.kve` — ~15× faster for EM
  - Requires `joint._posterior_gig_params(z2, w2)` (implemented by all four joint subclasses)
- **M-step**: `model.m_step(X, expectations)` → converts expectation parameters to classical, returns new immutable model
- **Fitter**: `BatchEMFitter(e_step_backend='cpu', m_step_solver='cpu')` — controls execution strategy for the EM hot path
  - `e_step_backend` forwarded to `model.e_step`; `m_step_solver` forwarded to `model.m_step` (for GH/GIG)

## Bessel Functions (`_bessel.py`)

`log_kv(v, z, backend='jax')` — unified entry point with backend selection:

- **`backend='jax'` (default)**: pure-JAX, `@jax.custom_jvp`, JIT-able, differentiable.
  - **Evaluation**: 4-regime `lax.cond` dispatch (Hankel/Olver/small-z/quadrature)
  - **∂/∂z (exact)**: recurrence $K'_\nu = -(K_{\nu-1} + K_{\nu+1})/2$
  - **∂/∂ν**: central finite differences, ε = 10⁻⁵
  - Internal name: `_log_kv_jax` (private, carries the `@custom_jvp`)

- **`backend='cpu'`**: `scipy.special.kve`, fully vectorized numpy. Not JIT-able.
  Fast for EM hot path (6 C-level array calls). Handles overflow/NaN via
  `inf_mask` fix using asymptotic Γ-function formula.
  Internal name: `_log_kv_cpu` (private).

Verified across $(v, z)$ grid, $v \in [-100, 500]$, $z \in [10^{-6}, 10^3]$: relative errors < 10⁻⁶. See `notebooks/bessel_function_comparison.ipynb`.

**Why `backend` does not break JIT**: the parameter is a Python-level string resolved before JAX tracing. When `backend='jax'` (the default), JAX traces `_log_kv_jax` — identical to the previous behaviour. See `docs/design/cpu_bessel_design.md` §4.

## GIG η→θ Optimization

Given $\eta = (\eta_1, \eta_2, \eta_3) = (E[\log Y], E[1/Y], E[Y])$, find $\theta$ such that $\nabla\psi(\theta) = \eta$.

**η-rescaling** before optimization:

$$s = \sqrt{\eta_2/\eta_3}, \quad \tilde\eta = \bigl(\eta_1 + \tfrac{1}{2}\log s^2,\; \sqrt{\eta_2\eta_3},\; \sqrt{\eta_2\eta_3}\bigr)$$

This reduces Fisher condition number by up to $10^{30}$ for extreme $a/b$ ratios.

Solver: `jaxopt.LBFGSB` with bounds $\theta_2 \leq 0$, $\theta_3 \leq 0$. Multi-start with initial guesses from Gamma, InverseGamma, and InverseGaussian special cases. See `notebooks/gig_optimization_test.ipynb`.

## Design Docs

| Document | Content |
|---|---|
| `docs/design/jax_design.md` | Full architecture rationale with code examples |
| `docs/design/detailed_design.md` | Equinox fundamentals, FlowJAX patterns, GMMX patterns |
| `docs/plans/migration_plan.md` | Phased migration: foundation → distributions → mixtures → EM |
| `docs/references/distribution_packages.md` | Survey of TFP, FlowJAX, efax, GMMX |
