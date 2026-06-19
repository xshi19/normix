# Architecture

> Quick reference: `AGENTS.md`. Full design rationale: `design/design.md`.

## Module Hierarchy

```
normix/                     # JAX implementation
├── exponential_family.py   # ExponentialFamily(eqx.Module)
├── divergences.py          # squared_hellinger, kl_divergence (Tier 1 + Tier 3)
├── distributions/
│   ├── gamma.py                          # Gamma(α, β)
│   ├── inverse_gamma.py                  # InverseGamma(α, β)
│   ├── inverse_gaussian.py               # InverseGaussian(μ, λ)
│   ├── generalized_inverse_gaussian.py   # GeneralizedInverseGaussian (primary), GIG (alias)
│   │                                     # + private Devroye TDR / PINV samplers
│   ├── normal.py                         # MultivariateNormal(μ, L_Sigma)
│   ├── variance_gamma.py                 # Joint + Marginal + Factor VarianceGamma
│   ├── normal_inverse_gamma.py           # Joint + Marginal + Factor NormalInverseGamma
│   ├── normal_inverse_gaussian.py        # Joint + Marginal + Factor NormalInverseGaussian
│   └── generalized_hyperbolic.py         # Joint + Marginal + Factor GeneralizedHyperbolic
├── mixtures/
│   ├── joint.py            # JointNormalMixture(ExponentialFamily) abstract
│   ├── marginal.py         # MarginalMixture ABC + NormalMixture (owns a JointNormalMixture)
│   └── factor.py           # FactorNormalMixture(MarginalMixture) abstract; Σ = F Fᵀ + diag(D)
├── fitting/
│   ├── em.py               # EMResult; BatchEMFitter, IncrementalEMFitter
│   ├── eta.py              # NormalMixtureEta, FactorMixtureStats, affine_combine
│   ├── eta_rules.py        # EtaUpdateRule + AffineRule; Identity, RobbinsMonro, EWMA, Shrinkage, ...
│   ├── shrinkage_targets.py # eta0_from_model, eta0_isotropic, eta0_diagonal, eta0_with_sigma
│   ├── solvers.py          # solve_bregman*, BregmanResult, make_jit_newton_solver; Newton, L-BFGS, scipy multi-start
│   └── __init__.py
├── finance/
│   ├── projection.py       # project_portfolio (wᵀX → Univariate*)
│   ├── risk.py             # RiskMeasure ABC, CVaR(α) with value, gradient, Hessian in (μ̃,γ̃,σ̃) and w
│   └── _mc.py              # Rao-Blackwellised CDF / VaR bisection over the subordinator Y
└── utils/
    ├── bessel.py            # log_kv(v, z, backend='jax'|'cpu')
    ├── constants.py         # LOG_EPS, TINY, BESSEL_EPS_V, GIG_DEGEN_THRESHOLD, ...
    ├── gammainc.py          # gammaincinv (JAX equivalent of scipy.special.gammaincinv)
    ├── rvs.py               # Generic RVS: build_pinv_table (pure JAX), rvs_pinv
    ├── plotting.py          # notebook plotting helpers (golden-ratio figures)
    └── validation.py        # moment validation, parameter printing (notebooks)
```

## Exponential Family Core

Every distribution subclasses `ExponentialFamily(eqx.Module)` and implements four abstract methods:

| Method | Purpose |
|---|---|
| `_log_partition_from_theta(theta)` | Log partition function $\psi(\theta)$ — **single source of truth** |
| `natural_params()` | Compute $\theta$ from stored classical parameters |
| `sufficient_statistics(x)` | Compute $t(x)$ for a single observation |
| `log_base_measure(x)` | Compute $\log h(x)$ |

### The Log-Partition Triad

Every distribution gets six derived classmethods, organised as three pairs:

```
                     JAX (JIT-able)                  CPU (numpy/scipy)
                     ─────────────                   ─────────────────
log-partition        _log_partition_from_theta        _log_partition_cpu
gradient             _grad_log_partition              _grad_log_partition_cpu
Hessian              _hessian_log_partition           _hessian_log_partition_cpu
```

**Tier 1** (`_log_partition_from_theta`) is abstract — subclasses must implement it.

**Tier 2** (JAX grad/Hessian) defaults to `jax.grad` / `jax.hessian`; subclasses override with analytical formulas when available.

**Tier 3** (CPU) defaults to wrapping the JAX versions; Bessel-dependent distributions (GIG) override with native numpy/scipy implementations.

| Method | Gamma | InverseGamma | InverseGaussian | GIG |
|---|---|---|---|---|
| `_log_partition_from_theta` | ✓ | ✓ | ✓ | ✓ |
| `_grad_log_partition` | analytical | analytical | analytical | inherits (`jax.grad`) |
| `_hessian_log_partition` | analytical | analytical | analytical | analytical (7-Bessel) |
| `_log_partition_cpu` | inherits | inherits | inherits | scipy Bessel |
| `_grad_log_partition_cpu` | inherits | inherits | inherits | scipy Bessel |
| `_hessian_log_partition_cpu` | inherits | inherits | inherits | central FD on CPU ψ |

Everything else is derived automatically:

- `log_partition()` = `_log_partition_from_theta(natural_params())`
- `expectation_params(backend='jax'|'cpu')` = `_grad_log_partition(θ)` → $\eta = \nabla\psi(\theta)$
- `fisher_information(backend='jax'|'cpu')` = `_hessian_log_partition(θ)` → $I(\theta) = \nabla^2\psi(\theta)$
- `log_prob(x)` = `log_base_measure(x) + t(x)·θ − ψ(θ)`
- `pdf(x)` = `exp(log_prob(x))`
- `cdf(x)` — analytical where available (Gamma, InverseGamma, InverseGaussian)
- `mean()`, `var()`, `std()` — analytical formulas per distribution
- `rvs(n, seed)` — JAX-based sampling (Devroye TDR / PINV for GIG; JAX primitives for others)

### Three Parametrizations

```
classical (α, β, μ, L_Sigma, ...)
    ↕  from_classical / natural_params
natural θ
    ↕  from_natural / expectation_params (via _grad_log_partition)
expectation η = E[t(X)]
    ↕  from_expectation(eta, *, backend, method, theta0, maxiter, tol, verbose)
         fit_mle(X, *, theta0, maxiter, tol, verbose)
         fit(self, X, ...) — warm-start η→θ from self.natural_params()
         default_init(X) — η̂ then from_expectation(η̂) for cold start
```

All conversions are JIT-compatible and support `jax.grad` and `jax.vmap`.

### Divergences

Three-tier design for statistical divergences between exponential family distributions:

**Tier 1** (`divergences.py`): Functional core — `squared_hellinger_from_psi(psi, θ_p, θ_q)`, `kl_divergence_from_psi(psi, grad_psi, θ_p, θ_q)`. Pure functions of callables + arrays. Maximally composable with `jax.jit`, `jax.vmap`, `jax.grad`.

**Tier 2** (`ExponentialFamily`): Instance methods — `model.squared_hellinger(other)`, `model.kl_divergence(other)`. Default calls Tier 1 with the class's `_log_partition_from_theta`. Subclasses may override for numerically improved variants. `NormalMixture` delegates to its joint distribution (upper bound).

**Tier 3** (`divergences.py`): Module convenience — `squared_hellinger(p, q)`, `kl_divergence(p, q)`. Accepts `ExponentialFamily` or `NormalMixture`, delegates to Tier 2.

### Distribution Storage

Each distribution stores canonical parameters as named `eqx.Module` fields. Cholesky factors of covariance matrices are always named `L_Sigma`.

| Distribution | Stored Attributes | Notes |
|---|---|---|
| Gamma | `alpha`, `beta` | shape, rate |
| InverseGamma | `alpha`, `beta` | shape, rate |
| InverseGaussian | `mu`, `lam` | mean, shape |
| GeneralizedInverseGaussian (GIG) | `p`, `a`, `b` | shape, rate, rate |
| MultivariateNormal | `mu`, `L_Sigma` | mean, Cholesky of covariance |
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
    └── JointGeneralizedHyperbolic        Y ~ GeneralizedInverseGaussian

MarginalMixture(eqx.Module)              abstract; fitter contract
├── NormalMixture                         owns a JointNormalMixture (full Σ)
│   ├── VarianceGamma
│   ├── NormalInverseGamma
│   ├── NormalInverseGaussian
│   └── GeneralizedHyperbolic
└── FactorNormalMixture                   Σ = F Fᵀ + diag(D); Woodbury solves
    ├── FactorVarianceGamma
    ├── FactorNormalInverseGamma
    ├── FactorNormalInverseGaussian
    └── FactorGeneralizedHyperbolic
```

`NormalMixture` owns a `JointNormalMixture`. The joint is an exponential family; the marginal is not. Per `docs/theory/gh.rst`, both layers are parameterised by the same classical tuple `(μ, γ, Σ, subordinator)`, so the marginal exposes those parameters as forwarders on top of its joint storage.

`FactorNormalMixture` is a sibling of `NormalMixture`: same `MarginalMixture` contract, but stores `(μ, γ, F, D, subordinator)` directly without a joint exponential-family layer (the FA complete-data structure is over `(X, Y, Z)` with ten sufficient statistics — `FactorMixtureStats` in `fitting/eta.py` — rather than the six of `NormalMixtureEta`). All Σ-related linear algebra (`_solve`, `_quad_form`, `_log_det_sigma`, `_beta`) goes through Woodbury at `O(d r² + r³)`, never forming a dense `d × d` solve. Convergence is measured on `Σ = F Fᵀ + diag(D)` to sidestep `F`'s rotational gauge. See `docs/theory/factor_analysis.rst` and `design/mixtures.md` § 6.

Both `NormalMixtureEta` and `FactorMixtureStats` lay out their fields in **theory order** (`s_1 = E[Y⁻¹]`, `s_2 = E[Y]`, `s_3 = E[log Y]`, `s_4 = E[X]`, `s_5 = E[X/Y]`, `s_6 = E[X X^T/Y]`; factor stats add `s_7..s_10` for the latent `Z`). Sharing the first six fields means shrinkage targets and η-update rule weights port across the two families.

Key methods on `JointNormalMixture`:
- `conditional_expectations(x)` → E[log Y|x], E[1/Y|x], E[Y|x] (EM E-step)
- `_compute_posterior_expectations(x)` — implemented by each concrete joint
- `_mstep_normal_params(eta: NormalMixtureEta)` → μ, γ, L_Sigma (closed-form M-step for normal parameters)
- `_quad_forms(x)` → z=L_Sigma⁻¹(x-μ), w=L_Sigma⁻¹γ (EM hot path)
- `from_expectation(eta, **kw)` — η→joint constructor; dispatches on `isinstance(eta, NormalMixtureEta)` (closed-form M-step) vs flat `jax.Array` (inherited Bregman solver). Per-subclass `_subordinator_from_eta` and `_from_normal_and_subordinator` hooks bridge the differing subordinator field names.

Key methods on `NormalMixture`:
- `from_expectation(eta, **kw)` → marginal wrapper around the joint constructor; the canonical η→model map.
- `mu`, `gamma`, `L_Sigma`, `sigma()`, `log_det_sigma()` — read-only forwarders to the joint.
- Per-subclass subordinator forwarders: `alpha`/`beta` (VG, NInvG), `mu_ig`/`lam` (NIG), `p`/`a`/`b` (GH).
- `replace(**updates)` — immutable update via `eqx.tree_at`. Accepts normal keys, subordinator keys (per `_subordinator_keys()`), and `sigma` as a Cholesky-converted alias for `L_Sigma`.
- `m_step(eta) ≡ type(self).from_expectation(eta, **kw)` for VG / NInvG / NIG; `GeneralizedHyperbolic` overrides to inject a warm-start `theta0` for the GIG solver and a sanity-check fallback.

## EM Algorithm

The model knows math; the fitter knows iteration (following GMMX).

- **E-step**: `model.e_step(X, backend='jax'|'cpu') -> NormalMixtureEta`
  - `backend='jax'` (default): `jax.vmap(joint.conditional_expectations)(X)` — JIT-able
  - `backend='cpu'`: quad forms stay in JAX (vmapped), Bessel on CPU via `scipy.kve` — ~15× faster for large N
  - Returns a `NormalMixtureEta` pytree (6 aggregated expectation fields), not raw per-observation dicts.
- **M-step**: `model.m_step(eta: NormalMixtureEta, **kwargs) -> NormalMixture` — full update from aggregated η. MCECM uses `m_step_normal(eta)` and `m_step_subordinator(eta, **kwargs)` separately. The closed-form `μ, γ, Σ` update uses `E[1/Y|x]`, which diverges for near-mode observations when the subordinator has `b=0` (VG only); the E-step floors `b_post` at `B_POST_FLOOR` to bound it. See `tech_notes/vg_em_inverse_moment_singularity.md`.
- **`compute_eta_from_model() -> NormalMixtureEta`**: reconstruct η from model parameters (initialises incremental EM running average).
- **Batch fitter**: `BatchEMFitter` is a plain Python class (not an `eqx.Module`). Its `fit(model, X)` returns **`EMResult`**: fitted `model`, optional per-iteration log-likelihoods, `param_changes` (max relative change in normal parameters μ, γ, `L_Sigma` — GIG/subordinator excluded), `n_iter`, `converged`, `elapsed_time`. Optional `eta_update` rule enables penalised / shrinkage batch EM.
- **Incremental fitter**: `IncrementalEMFitter` processes data in random mini-batches with a pluggable `EtaUpdateRule` (Robbins-Monro, EWMA, sample-weighted, shrinkage, etc.). With both E- and M-step backends set to `'jax'` and `verbose==0`, the outer minibatch iterations run inside `jax.lax.scan` (nested `jax.lax.fori_loop` when `inner_iter>1`) so the algorithm body traces once; verbose output or CPU backends keep a Python loop using the same pre-stacked RNG keys per step as before.
- **Defaults**: `tol=1e-3`, `regularization='none'`, `e_step_backend='jax'`, `m_step_backend='cpu'`, `m_step_method='newton'`.
- **Regularizations** (post-M-step rescales; the joint density is unchanged in every case — they pick a canonical representative of the orbit $Y \to s\,Y$):

  | Mode | Effect | Marginal method |
  |---|---|---|
  | `'none'` | identity | — |
  | `'det_sigma_one'` | `det(Σ) = 1` (classical GH convention) | `regularize_det_sigma(0.0)` |
  | `'det_sigma_x'` | `log det(Σ)` matches the **initial** model passed to `fit` (target captured once) | `regularize_det_sigma(target)` |
  | `'a_eq_b'` | rescale GIG so `a = b = sqrt(a·b)` (orbit invariant); NIG: `μ_IG = 1`; no-op for VG / NInvG / MVN | `regularize_a_eq_b()` |

  Each marginal owns the Σ / γ side via `_rescale(s)`; per-subclass `_build_rescaled` handles the subordinator (e.g. for GIG: `a → a/s`, `b → b·s`). `FactorNormalMixture` follows the same pattern with `F → F/√s`, `D → D/s`. See `design/em_framework.md` § 5.
- **Verbosity**: `verbose=0` silent, `1` summary, `2` per-iteration diagnostics.
- **Convenience**: `NormalMixture.fit(X, **fitter_kwargs) → EMResult` delegates to `BatchEMFitter` using **`self` as initialization**. Cold start: `SomeMixture.default_init(X)` then `result = model.fit(X)`.

## Bessel Functions (`utils/bessel.py`)

`log_kv(v, z, backend='jax')` — unified entry point:

- **`backend='jax'` (default)**: pure-JAX, `@jax.custom_jvp`, JIT-able, differentiable.
  4-regime `lax.cond` dispatch: Hankel / Olver / small-z / Gauss-Legendre quadrature.
  Derivatives: exact recurrence for ∂/∂z; central FD (ε=10⁻⁵) for ∂/∂ν.

- **`backend='cpu'`**: `scipy.special.kve`, fully vectorized NumPy. Not JIT-able.
  Fast for EM hot path. Overflow handled via asymptotic Γ-function formula.

### CPU Versions for Bessel-Dependent Functions

Any distribution that calls `log_kv` must override the Tier 3 CPU classmethods (`_log_partition_cpu`, `_grad_log_partition_cpu`, `_hessian_log_partition_cpu`) so that the CPU solver path (`solve_bregman(backend='cpu')`) avoids JAX dispatch entirely. Distributions that do not call `log_kv` inherit the default CPU wrappers at no additional cost. See `design/exponential_family.md` § 2.2 for rationale.

## Random Variate Generation (`utils/rvs.py`, `distributions/generalized_inverse_gaussian.py`)

Generic PINV (Polynomial-Interpolation-based Numerical Inversion) in `utils/rvs.py`:

- `build_pinv_table(log_kernel, mode, *, x_of_w, n_grid, tail_eps)` — pure-JAX quantile table from any univariate log-kernel. Tail bisection via `lax.fori_loop`, trapezoidal CDF via `jnp.cumsum`. No normalising constant needed.
- `rvs_pinv(key, u_grid, x_grid, n)` — samples via `jnp.interp`. Fully vectorised, GPU-friendly.

Distributions on $(0,\infty)$ supply `log_kernel(w) = log_prob(exp(w)) + w` and seed the table at `jnp.log(self.mode())`. Closed-form `mode()` methods live on `Gamma`, `InverseGamma`, `InverseGaussian`, and `GIG`. `InverseGaussian.ppf` and both `GIG.cdf`/`GIG.ppf` inline a single `build_pinv_table` call — no per-distribution wrapper.

GIG-specific sampling lives inline in `distributions/generalized_inverse_gaussian.py`:

- `_gig_rvs_devroye(key, p, a, b, n)` — TDR on $w = \log x$. Batch-parallel (no `while_loop`).
- `_gig_rvs_pinv(key, u_grid, x_grid, n)` — thin alias of `rvs_pinv` for `GIG.rvs(method='pinv')`.

Neither method evaluates the Bessel normalising constant. See `tech_notes/gig_rvs.md`.

## Numerical Constants (`utils/constants.py`)

All shared numerical constants are defined in `utils/constants.py` and imported
from there. Never define magic numbers locally in distribution files.

| Constant | Value | Purpose |
|---|---|---|
| `LOG_EPS` | `1e-30` | Floor for JAX log-space clamping |
| `TINY` | `1e-300` | Floor for numpy-side log |
| `BESSEL_EPS_V` | `1e-5` | FD step for ∂log K_v/∂v |
| `GIG_DEGEN_THRESHOLD` | `1e-10` | √(ab) threshold for GIG degenerate limits |
| `HESSIAN_DAMPING` | `1e-6` | Tikhonov damping in Newton Hessian |
| `THETA_FLOOR` | `-1e-8` | Floor for GIG θ₂, θ₃ warm-start |
| `B_POST_FLOOR` | `1e-6` | Floor for posterior GIG scale `b_post` in E-step (bounds the conditional inverse moment near the mode; only binds for VG) |
| `ALPHA_MOMENT_MARGIN` | `0.1` | Floor on the `(α−1)` denominator of the VG/NInvG prior moment `β/(α−1)` in `compute_eta_from_model` (keeps it finite and positive when `α ≤ 1`) |
| `SIGMA_REG` | `1e-8` | Covariance regularisation in M-step |
| `SAFE_DENOMINATOR` | `1e-10` | Floor for D = 1 − E[1/Y]·E[Y] |
| `D_FLOOR` | `1e-8` | Positivity floor for diagonal `D` in factor M-step |
| `PARAM_CHANGE_EPS` | `1e-10` | Floor on ‖old‖ in the EM relative parameter-change convergence measure |
| `FD_EPS_FISHER` | `1e-4` | FD step for Fisher information |

## GIG η→θ Optimization

Given $\eta = (E[\log Y], E[1/Y], E[Y])$, find $\theta$ such that $\nabla\psi(\theta) = \eta$.

This is equivalent to minimising the **Bregman divergence** $\psi(\theta) - \theta\cdot\eta$.

**η-rescaling** reduces Fisher condition number by up to $10^{30}$:
$$s = \sqrt{\eta_2/\eta_3}, \quad \tilde\eta = \bigl(\eta_1 + \tfrac{1}{2}\log s^2,\; \sqrt{\eta_2\eta_3},\; \sqrt{\eta_2\eta_3}\bigr)$$

**Solvers** (via `GeneralizedInverseGaussian.from_expectation(backend, method)`):
- `backend='cpu', method='lbfgs'` (typical for EM M-step): `scipy.optimize.minimize` + `scipy.kve` — avoids GPU kernel dispatch overhead on this 3D scalar problem.
- `backend='jax', method='newton'`: JAX Newton via `lax.scan`. Uses `GIG._hessian_log_partition` (7-Bessel analytical Hessian). The warm-started hot path is routed through a module-level `_gig_jax_newton_jit` produced by `make_jit_newton_solver` so all warm-started GIG solves share one cached XLA executable.
- `backend='jax', method='lbfgs'`: JAXopt L-BFGS.
- Omitting `theta0` in **`ExponentialFamily.from_expectation`**: defaults to **`jnp.zeros_like(eta)`**. **GIG** overrides: `theta0=None` runs **`solve_bregman_multistart`** on the η-rescaled problem (CPU L-BFGS-B, seeds from Gamma / InverseGamma / InverseGaussian special cases).

The solver passes `grad_fn` and `hess_fn` (both in θ-space) from the triad. The solver applies the φ↔θ chain rule internally via `jax.jacobian(to_theta)`, so distributions never need to know about reparametrization.

The general solver infrastructure lives in `fitting/solvers.py`: **`solve_bregman`**, **`solve_bregman_multistart`**, returning **`BregmanResult`** (`theta`, objective value, `grad_norm`, `num_steps`, `converged`, **`elapsed_time`**). Optional **`verbose`** prints solver progress. Scalar result fields use loose typing so results can be carried through **`lax.scan`** without forcing concrete Python `float`/`bool` on traced values. **`make_jit_newton_solver(f, grad_fn, hess_fn, bounds)`** builds a `@jax.jit`-decorated specialised Newton solve whose XLA cache survives across calls — used by the GIG warm-start hot path to avoid the per-call retrace that previously dominated GH JAX/JAX EM time (see `tech_notes/jax_overhead_diagnosis.md` § Resolution).

See `tech_notes/gig_eta_to_theta.md` for derivations and benchmarks.

## Documentation Map

| Document | Content |
|---|---|
| `design/index.md` | TOC of all living design docs (internal); public subset at `docs/design/` |
| `design/design.md` | Philosophy + canonical decision table |
| `design/exponential_family.md` | EF base class, log-partition triad, Bregman solver interface |
| `design/mixtures.md` | Joint vs Marginal, parameter facade, factor-analysis sibling family |
| `design/em_framework.md` | Model/Fitter separation, η-rules, `Shrinkage`, covariance regularisations |
| `design/solvers_and_bessel.md` | Bregman solver, GIG η→θ, Bessel regimes, CPU/GPU hybrid, RVS |
| `design/agent_instructions_design.md` | How AGENTS.md, rules, skills, and design docs work together |
| `tech_notes/` | Deep dives: Bessel survey, EM profiling, GIG optimization, GIG RVS benchmarks, distribution conversions, VG inverse-moment singularity |
| `docs/theory/` | Mathematical derivations (`.rst`) |
| `references/distribution_packages.md` | Survey of TFP, FlowJAX, efax, GMMX |
| `plans/finance_architecture.md` | `normix.finance` roadmap; Phase D implemented (projection + CVaR), Phases E and F still proposed |
| `archive/design/` | Implemented proposals retained for context (em_covariance_extensions, penalised_em, log_partition_triad, solver_redesign) |
