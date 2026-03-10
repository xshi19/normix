# Architecture

> Quick reference: `AGENTS.md`. Full design rationale: `docs/design/jax_design.md`.

## Module Hierarchy

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
    └── mle.py              # fit_mle convenience
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

No separate analytical overrides unless both (a) faster and (b) registered via `custom_jvp` so higher-order autodiff remains correct.

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

- **E-step**: `model.e_step(X)` → `jax.vmap(joint.conditional_expectations)(X)` computes $E[Y|X]$, $E[1/Y|X]$, $E[\log Y|X]$
- **M-step**: `model.m_step(X, expectations)` → converts expectation parameters to classical, returns new immutable model via `eqx.tree_at`
- **Fitter**: `BatchEMFitter.fit(model, X)` → `jax.lax.while_loop` until convergence; `OnlineEMFitter` uses `jax.lax.scan` with Robbins-Monro step sizes

## Bessel Functions (`_bessel.py`)

`log_kv(v, z)` with `@jax.custom_jvp`:

- **Evaluation**: TFP `log_bessel_kve(|v|, z) - z` + asymptotic fallbacks for overflow/NaN
- **∂/∂z (exact)**: recurrence $K'_\nu = -(K_{\nu-1} + K_{\nu+1})/2$
- **∂/∂ν (large z)**: analytical DLMF 10.40.2: $S'(\nu,z)/S(\nu,z)$
- **∂/∂ν (small z)**: central finite differences, ε = 10⁻⁵

Verified across $(v, z)$ grid, $v \in [-100, 500]$, $z \in [10^{-6}, 10^3]$: relative errors < 10⁻⁶. See `notebooks/bessel_function_comparison.ipynb`.

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
