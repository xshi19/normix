# Exponential Family Core

> **Scope.** Why every distribution lives behind one log-partition function,
> how the triad pattern keeps gradient/Hessian and JAX/CPU evaluation
> consistent, and how these primitives plug into the η→θ Bregman solver.
>
> **Where things live.** Module hierarchy and the full triad table are in
> `docs/ARCHITECTURE.md` § *Exponential Family Core*. This file records
> the rationale.

---

## 1. One Function, Three Parametrizations

Each distribution is described by a single convex log-partition
$\psi(\theta)$. Everything else — densities, expectation parameters,
Fisher information, Bregman inversion — is derived from it. Three
parametrizations form a triangle:

```
classical (μ, σ², α, β, …)   ←→   natural θ   ←→   expectation η = ∇ψ(θ)
                  via from_classical / natural_params                ↑
                                                                from_expectation
```

`from_natural`, `natural_params`, and `from_expectation` are the only three
constructors a user ever needs. `from_expectation(η)` runs the universal
Bregman solver `min_θ [ψ(θ) − θ·η]` from `fitting/solvers.py`.

Why it matters: distributions never need to know about parameter
conversions other than their own classical mapping. The conversion graph
collapses into derivatives and Bregman minimisation of a single function.

---

## 2. The Log-Partition Triad

Three functions × two backends:

```
                     JAX (JIT-able)                  CPU (numpy/scipy)
log-partition        _log_partition_from_theta        _log_partition_cpu
gradient             _grad_log_partition              _grad_log_partition_cpu
Hessian              _hessian_log_partition           _hessian_log_partition_cpu
```

| Tier | Default | Override when |
|------|---------|---------------|
| 1: `_log_partition_from_theta` | abstract | always (it *is* the distribution) |
| 2: `_grad_*`, `_hessian_*` (JAX) | `jax.grad` / `jax.hessian` | analytical formula avoids recompilation or saves Bessel calls |
| 3: `_log_partition_cpu` etc.    | numpy wrappers around the JAX version | distribution calls `log_kv` (Bessel) and EM hot path needs CPU evaluation |

### 2.1 Why `@classmethod` for the triad

Tier 2 needs to differentiate the **subclass**'s `_log_partition_from_theta`,
not the parent's. `@classmethod` gives `cls`, which routes to the right
implementation. `@staticmethod` cannot — it would always call the parent.
`cls` is not traced by JAX, so this is JIT-safe.

### 2.2 Why the CPU tier exists

`scipy.special.kve` is **fast** for vectorised Bessel evaluation in the
EM hot path; a `vmap` over JAX's `lax.cond`-dispatched `log_kv` triggers
separate kernel launches per regime check and is much slower. Distributions
that don't call `log_kv` (Gamma, InverseGamma, InverseGaussian) inherit
the default `np.asarray(jax_version(...))` wrappers — they pay nothing for
the CPU tier.

### 2.3 Why we don't share Bessel calls between gradient and Hessian

A combined `_grad_hess` could share Bessel evaluations (≈ 7 calls vs
12 for separate). We keep them separate anyway: the saving is small,
and combining them re-introduces a phi-space chain rule the *distribution*
must understand. Keeping `_grad_log_partition` and `_hessian_log_partition`
as θ-space-only lets the solver (§3) own the chain rule generically.

---

## 3. Bregman Solver Interface

`fitting/solvers.py` exposes `solve_bregman(f, η, θ₀, ...)` for any convex
$f$ — not just log-partitions. The interface uses **θ-space only** for
gradient/Hessian; bounds are handled by the solver via
`_setup_reparam` (φ ↔ θ). Distributions never see the reparametrization.

```python
solve_bregman(f, eta, theta0, *, backend, method, bounds,
              grad_fn, hess_fn, max_steps, tol, verbose)
```

| Decision | Choice | Why |
|---|---|---|
| `grad_fn` + `hess_fn` (vs combined `grad_hess_fn`) | separate | Distributions don't know about φ-space; solver applies chain rule via `jax.jacobian(to_theta)` |
| Newton implementation | hand-rolled `lax.while_loop` | No JAX library provides a Newton **minimizer** that accepts a user-supplied Hessian (Optimistix Newton is root-finding only, JAXopt has no Newton) |
| Multi-start | `jax.vmap` for JAX, Python `for` for CPU | Orthogonal wrapper, not baked into solver name |
| Result type | `BregmanResult` with `Any`-typed scalars | Survives `lax.scan` without `ConcretizationTypeError` |
| Cached JIT | module-level `_gig_jax_newton_jit` via `make_jit_newton_solver` | The GIG warm-start hot path otherwise retraces per call (see `docs/tech_notes/jax_overhead_diagnosis.md`) |

**`backend × method` matrix:**

| backend | method | bounds | What runs |
|---|---|---|---|
| `jax` | `newton` | reparam | hand-rolled `while_loop`, autodiff or supplied `grad_fn`/`hess_fn` |
| `jax` | `lbfgs` | native (LBFGSB) | `jaxopt.LBFGSB` |
| `jax` | `bfgs`  | reparam | `jaxopt.BFGS` |
| `cpu` | `newton`| reparam | `scipy.optimize.minimize(method='trust-exact')` |
| `cpu` | `lbfgs` | native | `scipy.optimize.minimize(method='L-BFGS-B')` |
| `cpu` | `bfgs`  | none   | `scipy.optimize.minimize(method='BFGS')` |

GIG warm-start hot path: `backend='cpu', method='lbfgs'` (scipy's L-BFGS-B
+ `scipy.kve`) avoids GPU dispatch on this 3-D scalar problem. See
`docs/tech_notes/gig_eta_to_theta.md` for the η-rescaling derivation and
benchmarks.

---

## 4. Pre-1.0 Decisions Recorded Here

### 4.1 D3 — `MultivariateNormal` as `ExponentialFamily`

Promoted from a plain `eqx.Module` to a full `ExponentialFamily`. EF
structure:

| Component | Expression |
|---|---|
| $t(x)$ | $[x,\;\operatorname{vec}(xx^\top)]$, shape $(d + d^2,)$ |
| $\theta$ | $[\Sigma^{-1}\mu,\;-\tfrac12\operatorname{vec}(\Sigma^{-1})]$ |
| $\log h(x)$ | $0$ |
| $\psi(\theta)$ | $\tfrac12\theta_1^\top\Lambda^{-1}\theta_1 - \tfrac12\log\|\Lambda\| + \tfrac d2\log(2\pi)$, $\Lambda = -2\,\mathrm{reshape}(\theta_2)$ |

`vec` uses row-major (`ravel()`) throughout. All conversions are
analytical (Tier 2 override) — no Bregman solver is ever invoked for
MVN. `_log_partition_from_theta` uses Cholesky of $\Lambda$ for numerical
stability; `log_prob` overrides the inherited EF formula with a direct
Cholesky path (more efficient).

### 4.2 D4 — Keep `jaxopt` for now

JAXopt is unmaintained (last release 0.8.3) and emits a `DeprecationWarning`
on import. We keep it: it is the only pure-JAX library with a native box-
constrained quasi-Newton (`LBFGSB`). Migrating to `optax.scale_by_lbfgs`
loses the convergence loop; `optimistix.LBFGS` lacks box constraints.

The deprecation warning is suppressed in `normix/__init__.py`. Migration
recipe (when JAXopt breaks): wrap `optax.scale_by_lbfgs` in
`jax.lax.while_loop` (~100 lines), then drop `jaxopt`.

### 4.3 Constraints handling

`jnp.maximum(x, LOG_EPS)` (clamp), not `paramax`. Reasons:

- The reparametrization we need is 8 lines, fully understandable.
- EM does not need gradients through the constraints (it parameterises θ
  in the constrained space, not log-space).
- No extra dependency.

`jnp.where` is preferred over `lax.cond` whenever possible: it is
`vmap`-compatible without changing the trace, and the clamping prevents
NaN gradients without branch divergence.

### 4.4 Module-level functions are forbidden

Distribution behaviour lives on the class as `@classmethod` or
`@staticmethod`. No `_helper(self.alpha, ...)` module-level functions
that close over attributes — they leak the class API into module globals
and are hard to override.

---

## 5. Cross-References

- Architecture surface: `docs/ARCHITECTURE.md` § *Exponential Family Core*.
- η→θ optimization for GIG: `docs/design/solvers_and_bessel.md`.
- Tech notes: `docs/tech_notes/gig_eta_to_theta.md`,
  `docs/tech_notes/bessel_implementations_survey.md`,
  `docs/tech_notes/jax_overhead_diagnosis.md`.
- Historical / archived rationale: `docs/archive/design/log_partition_triad.md`,
  `docs/archive/design/solver_redesign.md`.
