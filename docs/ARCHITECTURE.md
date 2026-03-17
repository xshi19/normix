# Architecture

> Quick reference: `AGENTS.md`. Full design rationale: `docs/design/design.md`.

## Module Hierarchy

```
normix/                     # JAX implementation
├── exponential_family.py   # ExponentialFamily(eqx.Module)
├── distributions/
│   ├── gamma.py                          # Gamma(α, β)
│   ├── inverse_gamma.py                  # InverseGamma(α, β)
│   ├── inverse_gaussian.py               # InverseGaussian(μ, λ)
│   ├── generalized_inverse_gaussian.py   # GIG / GeneralizedInverseGaussian(p, a, b)
│   ├── normal.py                         # MultivariateNormal(μ, L_Sigma)
│   ├── variance_gamma.py                 # VarianceGamma / JointVarianceGamma
│   ├── normal_inverse_gamma.py           # NormalInverseGamma / JointNormalInverseGamma
│   ├── normal_inverse_gaussian.py        # NormalInverseGaussian / JointNormalInverseGaussian
│   └── generalized_hyperbolic.py         # GeneralizedHyperbolic / JointGeneralizedHyperbolic
├── mixtures/
│   ├── joint.py            # JointNormalMixture(ExponentialFamily)
│   └── marginal.py         # NormalMixture (owns a JointNormalMixture)
├── fitting/
│   ├── em.py               # BatchEMFitter, OnlineEMFitter, MiniBatchEMFitter
│   └── __init__.py
└── utils/
    ├── bessel.py            # log_kv(v, z, backend='jax'|'cpu')
    ├── constants.py         # LOG_EPS, GIG_EPS_V_HESS, GIG_EPS_NP
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
- `cdf(x)` — analytical where available (Gamma, InverseGamma, InverseGaussian)
- `mean()`, `var()`, `std()` — analytical formulas per distribution
- `rvs(n, seed)` — numpy/scipy-based sampling (not JIT-able)

### Three Parametrizations

```
classical (α, β, μ, L_Sigma, ...)
    ↕  from_classical / natural_params
natural θ
    ↕  from_natural / expectation_params (jax.grad)
expectation η = E[t(X)]
    ↕  from_expectation(eta, *, theta0, maxiter, tol)
         fit_mle(X, *, theta0, maxiter, tol)
```

All conversions are JIT-compatible and support `jax.grad` and `jax.vmap`.

### Distribution Storage

Each distribution stores canonical parameters as named `eqx.Module` fields. Cholesky factors of covariance matrices are always named `L_Sigma`.

| Distribution | Stored Attributes | Notes |
|---|---|---|
| Gamma | `alpha`, `beta` | shape, rate |
| InverseGamma | `alpha`, `beta` | shape, rate |
| InverseGaussian | `mu`, `lam` | mean, shape |
| GIG | `p`, `a`, `b` | shape, rate, rate |
| MultivariateNormal | `mu`, `L_Sigma` | mean, Cholesky of Σ |
| VarianceGamma | `mu`, `gamma`, `L_Sigma`, `alpha`, `beta` | Gamma subordinator |
| NormalInverseGamma | `mu`, `gamma`, `L_Sigma`, `alpha`, `beta` | InverseGamma subordinator |
| NormalInverseGaussian | `mu`, `gamma`, `L_Sigma`, `mu_ig`, `lam` | InverseGaussian subordinator |
| GeneralizedHyperbolic | `mu`, `gamma`, `L_Sigma`, `p`, `a`, `b` | GIG subordinator |

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

Key methods on `JointNormalMixture`:
- `conditional_expectations(x)` → E[log Y|x], E[1/Y|x], E[Y|x] (EM E-step)
- `_compute_posterior_expectations(x)` — implemented by each concrete joint
- `_mstep_normal_params(...)` → μ, γ, L_Sigma (closed-form M-step for normal parameters)
- `_quad_forms(x)` → z=L_Sigma⁻¹(x-μ), w=L_Sigma⁻¹γ (EM hot path)

## EM Algorithm

The model knows math; the fitter knows iteration (following GMMX).

- **E-step**: `model.e_step(X, backend='jax'|'cpu')`
  - `backend='jax'` (default): `jax.vmap(joint.conditional_expectations)(X)` — JIT-able
  - `backend='cpu'`: quad forms stay in JAX (vmapped), Bessel on CPU via `scipy.kve` — ~15× faster for large N
- **M-step**: `model.m_step(X, expectations)` → new immutable model
- **Fitter**: `BatchEMFitter(e_step_backend='cpu', m_step_solver='cpu')`

## Bessel Functions (`utils/bessel.py`)

`log_kv(v, z, backend='jax')` — unified entry point:

- **`backend='jax'` (default)**: pure-JAX, `@jax.custom_jvp`, JIT-able, differentiable.
  4-regime `lax.cond` dispatch: Hankel / Olver / small-z / Gauss-Legendre quadrature.
  Derivatives: exact recurrence for ∂/∂z; central FD (ε=10⁻⁵) for ∂/∂ν.

- **`backend='cpu'`**: `scipy.special.kve`, fully vectorized NumPy. Not JIT-able.
  Fast for EM hot path. Overflow handled via asymptotic Γ-function formula.

## GIG η→θ Optimization

Given $\eta = (E[\log Y], E[1/Y], E[Y])$, find $\theta$ such that $\nabla\psi(\theta) = \eta$.

**η-rescaling** reduces Fisher condition number by up to $10^{30}$:
$$s = \sqrt{\eta_2/\eta_3}, \quad \tilde\eta = \bigl(\eta_1 + \tfrac{1}{2}\log s^2,\; \sqrt{\eta_2\eta_3},\; \sqrt{\eta_2\eta_3}\bigr)$$

Solver: `jaxopt.LBFGSB` with bounds $\theta_2 \leq 0$, $\theta_3 \leq 0$.
CPU alternative: `scipy.optimize.minimize` with `scipy.kve` (~500× faster per call).

See `docs/tech_notes/gig_eta_to_theta.md` for derivations and benchmarks.

## Documentation Map

| Document | Content |
|---|---|
| `docs/design/design.md` | Design rationale, architecture decisions |
| `docs/tech_notes/` | Bessel survey, EM profiling, GIG optimization details |
| `docs/theory/` | Mathematical derivations (rst) |
| `docs/references/distribution_packages.md` | Survey of TFP, FlowJAX, efax, GMMX |
