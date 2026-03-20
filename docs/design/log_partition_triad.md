# Log-Partition Triad: Redesigning the ExponentialFamily–GIG Interface

**Date:** 2026-03-20
**Status:** Implemented (branch: `cursor/log-partition-triad-architecture-a531`)
**Scope:** `normix/exponential_family.py`, `normix/distributions/generalized_inverse_gaussian.py`,
`normix/fitting/solvers.py`, and all `ExponentialFamily` subclasses

---

## 1. Problem Statement

### 1.1 GIG has 6 messy module-level functions

`generalized_inverse_gaussian.py` defines 6 functions outside the class:

| Function | Lines | Purpose |
|---|---|---|
| `_log_partition_gig_cpu` | 389–412 | CPU log-partition (numpy + scipy Bessel) |
| `_gig_bessel_quantities` | 419–454 | 7-call Bessel derivative helper |
| `_analytical_grad_hess_phi` | 457–504 | Analytical grad+Hessian in solver's phi-space |
| `_gig_cpu_grad` | 511–521 | CPU gradient (scipy.kve) |
| `_cpu_grad_hess_theta` | 524–543 | CPU grad+Hessian for trust-exact Newton |
| `_initial_guesses` | 546–603 | Multi-start initialization from Gamma/IG/IGauss limits |

These names are inconsistent (`_gig_cpu_grad` vs `_cpu_grad_hess_theta`), their
relationship to the class is unclear, and they duplicate structure that should be
shared across all exponential family subclasses.

### 1.2 Missing parent-class structure for gradient and Hessian

The optimization in `from_expectation` (minimizing the Bregman divergence
ψ(θ) − θ·η) needs exactly three building blocks:

1. **Log-partition** ψ(θ)
2. **Gradient** ∇ψ(θ) = η (expectation parameters)
3. **Hessian** ∇²ψ(θ) = I(θ) (Fisher information)

The parent class provides the log-partition as an abstract method and derives
gradient/Hessian via `jax.grad`/`jax.hessian`. But there is no overridable
method for these — subclasses that want analytical formulas must override the
high-level `expectation_params()` or `fisher_information()` instead, which
conflates "the formula for ∇ψ" with "evaluate ∇ψ at the current instance's θ".

Similarly, the CPU backend pattern (numpy + scipy versions for solver use) is
entirely ad-hoc: GIG hand-wires CPU functions as module-level callbacks. Other
distributions inherit no CPU support at all.

### 1.3 Redundant phi-space chain rule in GIG

`_analytical_grad_hess_phi` manually converts theta-space gradient and Hessian to
the solver's reparametrized phi-space. The solver (`solvers.py`) already has the
phi↔theta conversion infrastructure (`_setup_reparam`, `to_theta`). The chain
rule for these element-wise reparametrizations is trivial for JAX to autodiff.
There is no need for the distribution to know about phi-space at all.

### 1.4 Duplicated `fit_mle` overrides

All four subordinator distributions (Gamma, InverseGamma, InverseGaussian, GIG)
override `fit_mle` to manually compute `[mean(log X), mean(1/X), mean(X)]` etc.
The parent's `fit_mle` already does `vmap(sufficient_statistics)(X)` then
`mean(axis=0)` — producing identical results. All four overrides are redundant.

---

## 2. Design: The Log-Partition Triad

### 2.1 Three functions × two backends

Every exponential family distribution needs the same three functions. Each exists
in a JAX version (JIT-able, autodiff-compatible) and a CPU version (numpy/scipy,
for the CPU solver path):

```
                     JAX (JIT-able)                  CPU (numpy/scipy)
                     ─────────────                   ─────────────────
log-partition        _log_partition_from_theta        _log_partition_cpu
gradient             _grad_log_partition              _grad_log_partition_cpu
Hessian              _hessian_log_partition           _hessian_log_partition_cpu
```

### 2.2 Parent class provides defaults

```python
class ExponentialFamily(eqx.Module):

    # ── Tier 1: Abstract (subclass MUST implement) ───────────────

    @staticmethod
    @abstractmethod
    def _log_partition_from_theta(theta): ...     # existing

    @abstractmethod
    def natural_params(self): ...                 # existing

    @staticmethod
    @abstractmethod
    def sufficient_statistics(x): ...             # existing

    @staticmethod
    @abstractmethod
    def log_base_measure(x): ...                  # existing

    # ── Tier 2: JAX grad & Hessian (override for analytical) ─────

    @classmethod
    def _grad_log_partition(cls, theta):
        """∇ψ(θ). Default: jax.grad."""
        return jax.grad(cls._log_partition_from_theta)(theta)

    @classmethod
    def _hessian_log_partition(cls, theta):
        """∇²ψ(θ) = I(θ). Default: jax.hessian."""
        return jax.hessian(cls._log_partition_from_theta)(theta)

    # ── Tier 3: CPU versions (override for native numpy/scipy) ───

    @classmethod
    def _log_partition_cpu(cls, theta):
        """Default: wraps JAX version."""
        return float(cls._log_partition_from_theta(jnp.asarray(theta, dtype=jnp.float64)))

    @classmethod
    def _grad_log_partition_cpu(cls, theta):
        """Default: wraps JAX grad."""
        return np.asarray(cls._grad_log_partition(jnp.asarray(theta, dtype=jnp.float64)))

    @classmethod
    def _hessian_log_partition_cpu(cls, theta):
        """Default: wraps JAX hessian."""
        return np.asarray(cls._hessian_log_partition(jnp.asarray(theta, dtype=jnp.float64)))
```

**Why `@classmethod`:** The gradient and Hessian need to reference the correct
subclass's `_log_partition_from_theta` via `cls`. The `cls` parameter is not a
JAX array and does not interfere with tracing — only `theta` is traced.

**Why not `@staticmethod`:** A static method cannot reference `cls`, so the parent's
default implementation would have no way to call the subclass's `_log_partition_from_theta`.

### 2.3 Derived quantities become trivial

```python
    def log_partition(self):
        return type(self)._log_partition_from_theta(self.natural_params())

    def expectation_params(self, backend='jax'):
        theta = self.natural_params()
        if backend == 'cpu':
            return jnp.asarray(type(self)._grad_log_partition_cpu(np.asarray(theta)))
        return type(self)._grad_log_partition(theta)

    def fisher_information(self, backend='jax'):
        theta = self.natural_params()
        if backend == 'cpu':
            return jnp.asarray(type(self)._hessian_log_partition_cpu(np.asarray(theta)))
        return type(self)._hessian_log_partition(theta)
```

The `backend` kwarg moves **up** from GIG into the parent. Every distribution
gets CPU support for free (the default CPU methods wrap the JAX versions).

---

## 3. Solver Interface Change

### 3.1 Replace `grad_hess_fn` with separate `grad_fn` + `hess_fn`

Current solver interface:

```python
solve_bregman(f, eta, theta0, ...,
              grad_fn=None,          # theta → ∇ψ(θ) as ndarray
              grad_hess_fn=None)     # (phi, eta) → (g_phi, H_phi) in phi-space
```

Problems:
- `grad_hess_fn` operates in **phi-space** (solver's reparametrized coordinates),
  forcing the distribution to know about the solver's internals.
- `grad_fn` and `grad_hess_fn` have different coordinate conventions (theta-space
  vs phi-space), which is confusing.

Proposed interface:

```python
solve_bregman(f, eta, theta0, ...,
              grad_fn=None,          # theta → ∇ψ(θ)
              hess_fn=None)          # theta → ∇²ψ(θ)
```

Both operate in **theta-space** only. The solver handles all reparametrization
internally.

### 3.2 Solver applies the chain rule generically

When `grad_fn` and `hess_fn` are provided, the JAX Newton solver computes
phi-space quantities using JAX autodiff through `to_theta`:

```python
def _jax_newton_raw(f, eta, theta0, bounds, max_steps, tol, grad_fn, hess_fn):
    phi0, to_theta, _ = _setup_reparam(theta0, bounds)

    def obj(phi):
        theta = to_theta(phi)
        return f(theta) - jnp.dot(theta, eta)

    if grad_fn is not None and hess_fn is not None:
        def get_g_H(phi):
            theta = to_theta(phi)
            g_theta = grad_fn(theta) - eta
            H_theta = hess_fn(theta)
            J = jax.jacobian(to_theta)(phi)
            g_phi = J.T @ g_theta
            def theta_dot_g(p):
                return jnp.dot(to_theta(p), g_theta)
            H_phi = J.T @ H_theta @ J + jax.hessian(theta_dot_g)(phi)
            return g_phi, H_phi
    else:
        _grad = jax.grad(obj)
        _hess = jax.hessian(obj)
        def get_g_H(phi):
            return _grad(phi), _hess(phi)
    ...
```

`jax.jacobian(to_theta)` traces through element-wise `exp`/identity — negligible
cost. The expensive computation (Bessel calls for GIG) lives entirely in
`grad_fn` and `hess_fn`, which are the class's theta-space methods.

For the CPU Newton solver, the integration is straightforward:

```python
def _cpu_solve(f, eta, theta0, ..., grad_fn, hess_fn):
    ...
    if method == 'newton':
        def jac_np(theta_np):
            return np.asarray(grad_fn(theta_np)) - eta_np
        def hess_np(theta_np):
            return np.asarray(hess_fn(theta_np))
        # Pass jac and hess to scipy.optimize.minimize(method='trust-exact')
```

### 3.3 Why `_analytical_grad_hess_phi` is eliminated

`_analytical_grad_hess_phi` did two things:

1. Computed analytical ∇ψ(θ) and ∇²ψ(θ) in theta-space using 7 Bessel calls.
2. Applied the chain rule θ→φ manually:
   - `g_phi = g_theta * j` where `j = diag(1, θ₂, θ₃)`
   - `H_phi = H_theta ⊙ outer(j,j) + diag(0, g[1]θ₂, g[2]θ₃)`

Step 1 is now `GIG._grad_log_partition` + `GIG._hessian_log_partition`.
Step 2 is now handled generically by the solver (§3.2). The distribution never
needs to know about phi-space.

### 3.4 Cost comparison for JAX Newton (GIG)

| Approach | Bessel calls per step | Notes |
|---|---|---|
| Full autodiff (`jax.hessian` through `log_kv`) | ~25 | Current default path |
| `_analytical_grad_hess_phi` (combined, shared Bessel) | 7 | Current analytical path |
| Separate `_grad` + `_hessian` (analytical, no sharing) | 5 + 7 = 12 | Proposed |

12 vs 7 is a minor overhead from not sharing Bessel evaluations between gradient
and Hessian. Per the simplicity criterion, eliminating `_analytical_grad_hess_phi`,
`_gig_bessel_quantities`, and the phi-space chain rule (saving ~90 lines of
GIG-specific code) outweighs 5 extra Bessel calls per Newton step.

---

## 4. `from_expectation`: Parent vs GIG

### 4.1 Generic parent implementation

```python
@classmethod
def from_expectation(cls, eta, *, backend='jax', method='lbfgs',
                     theta0=None, maxiter=500, tol=1e-10):
    if theta0 is None:
        theta0 = cls._init_theta_from_eta(eta)
    bounds = cls._theta_bounds()

    solver_kwargs = dict(bounds=bounds, max_steps=maxiter, tol=tol)

    if backend == 'cpu':
        solver_kwargs.update(
            grad_fn=cls._grad_log_partition_cpu,
            hess_fn=cls._hessian_log_partition_cpu,
        )
        f = cls._log_partition_cpu
    else:
        solver_kwargs.update(
            grad_fn=cls._grad_log_partition,
            hess_fn=cls._hessian_log_partition,
        )
        f = cls._log_partition_from_theta

    result = solve_bregman(f, eta, theta0,
                           backend=backend, method=method, **solver_kwargs)
    return cls.from_natural(result.theta)
```

This is fully generic. Any distribution that provides the triad (with defaults or
overrides) gets `from_expectation` with both JAX and CPU backends, all solver
methods, and Newton with the correct Hessian — for free.

### 4.2 GIG still overrides `from_expectation`

GIG has two genuinely unique requirements that the generic parent cannot handle:

1. **η-rescaling.** The GIG Fisher condition number reaches 10³⁰ for extreme a/b
   ratios. Before optimization, η is rescaled to a symmetric GIG (ã = b̃):
   ```
   s = √(η₂/η₃)
   η̃ = (η₁ + ½ log(η₂/η₃),  √(η₂η₃),  √(η₂η₃))
   ```
   After solving, θ is unscaled: `θ = (θ̃₁, θ̃₂/s, s·θ̃₃)`.

2. **Multi-start initialization.** When no `theta0` is provided, GIG generates
   starting points from Gamma, InverseGamma, and InverseGaussian special cases
   (`_initial_guesses`), plus perturbed copies.

But inside the override, GIG uses the standard triad:

```python
@classmethod
def from_expectation(cls, eta, *, backend='jax', method='lbfgs', ...):
    # ... η-rescaling (GIG-specific) ...

    if backend == 'cpu':
        f = cls._log_partition_cpu
        grad_fn = cls._grad_log_partition_cpu
    else:
        f = cls._log_partition_from_theta
        grad_fn = cls._grad_log_partition

    # ... multi-start or single-start via solve_bregman ...
    # ... θ unscaling (GIG-specific) ...
```

---

## 5. Per-Distribution Override Map

### 5.1 Triad overrides

```
                        _log_partition    _grad_log_     _hessian_log_
                        _from_theta       partition       partition
                        ───────────       ─────────       ──────────
Gamma                   override ✓        override ✓ᵃ     override ✓ᵇ
InverseGamma            override ✓        override ✓ᵃ     override ✓ᵇ
InverseGaussian         override ✓        override ✓ᵃ     override ✓ᵇ
GIG                     override ✓        inherit ᶜ       override ✓ᵈ

                        _log_partition    _grad_log_      _hessian_log_
                        _cpu              partition_cpu    partition_cpu
                        ────────────      ───────────      ─────────────
Gamma                   inherit ᵉ         inherit ᵉ        inherit ᵉ
InverseGamma            inherit ᵉ         inherit ᵉ        inherit ᵉ
InverseGaussian         inherit ᵉ         inherit ᵉ        inherit ᵉ
GIG                     override ✓ᶠ       override ✓ᶠ      override ✓ᶠ

ᵃ Analytical: digamma / closed-form (moved from current instance-method
  expectation_params to classmethod taking theta)
ᵇ Analytical: trigamma / closed-form Hessian
ᶜ Inherits default jax.grad — works because log_kv has @jax.custom_jvp
ᵈ Analytical 7-Bessel Hessian (absorbs _gig_bessel_quantities logic)
ᵉ Default wrappers (call JAX version) — sufficient since these distributions
  don't involve Bessel functions
ᶠ Native numpy + scipy.special.kve — avoids JAX dispatch for the CPU solver path
```

### 5.2 What each distribution overrides (full picture)

| Method | Parent default | Gamma | InvGamma | InvGauss | GIG |
|---|---|---|---|---|---|
| `_log_partition_from_theta` | abstract | ✓ | ✓ | ✓ | ✓ |
| `natural_params` | abstract | ✓ | ✓ | ✓ | ✓ |
| `sufficient_statistics` | abstract | ✓ | ✓ | ✓ | ✓ |
| `log_base_measure` | abstract | ✓ | ✓ | ✓ | ✓ |
| `_grad_log_partition` | `jax.grad` | analytical | analytical | analytical | inherit |
| `_hessian_log_partition` | `jax.hessian` | analytical | analytical | analytical | analytical (Bessel) |
| `_log_partition_cpu` | wraps JAX | inherit | inherit | inherit | scipy Bessel |
| `_grad_log_partition_cpu` | wraps JAX grad | inherit | inherit | inherit | scipy Bessel |
| `_hessian_log_partition_cpu` | wraps JAX hess | inherit | inherit | inherit | FD on CPU log-partition |
| `from_natural` | NotImplemented | ✓ | ✓ | ✓ | ✓ |
| `from_expectation` | generic solver | closed-form | closed-form | closed-form | override (rescaling + multi-start) |
| `fit_mle` | `vmap` + `from_expectation` | **delete** | **delete** | **delete** | **delete** |
| `mean` | NotImplemented | analytical | analytical | analytical | `eta[2]` |
| `var` | NotImplemented | analytical | analytical | analytical | `Fisher[2,2]` |
| `_theta_bounds` | None | — | — | — | ✓ |

---

## 6. What Gets Eliminated

### 6.1 Module-level functions → class methods or gone

| Current function | Disposition | Rationale |
|---|---|---|
| `_log_partition_gig_cpu` | → `GIG._log_partition_cpu` | Standard triad override |
| `_gig_cpu_grad` | → `GIG._grad_log_partition_cpu` | Standard triad override |
| `_cpu_grad_hess_theta` | **eliminated** | Solver constructs this generically from `grad_fn` + `hess_fn` |
| `_analytical_grad_hess_phi` | **eliminated** | Solver applies chain rule generically (§3.2); theta-space grad/hessian from triad |
| `_gig_bessel_quantities` | **absorbed** | Logic moves into `GIG._hessian_log_partition` |
| `_initial_guesses` | → `GIG._initial_guesses` | Staticmethod (GIG-specific multi-start) |

**Net result:** 6 module-level functions → 0. Three become standard triad
overrides with clean names inherited from the parent interface. One becomes a
properly scoped staticmethod. Two are eliminated entirely.

### 6.2 Redundant method overrides deleted

- `Gamma.fit_mle` — parent's `vmap(sufficient_statistics)` + `from_expectation` is identical
- `InverseGamma.fit_mle` — same
- `InverseGaussian.fit_mle` — same
- `GIG.fit_mle` — same
- `Gamma.expectation_params` → derived from `_grad_log_partition` in parent
- `InverseGamma.expectation_params` → same
- `InverseGaussian.expectation_params` → same
- `GIG.expectation_params` → same (backend kwarg now in parent)
- `GIG.fisher_information` → derived from `_hessian_log_partition` in parent

### 6.3 Solver parameter removed

`grad_hess_fn` (combined phi-space gradient+Hessian) is replaced by separate
`grad_fn` + `hess_fn` (both theta-space). Distributions no longer need to know
about the solver's reparametrization.

---

## 7. Implementation Plan

### Phase 1: Add triad to parent class

**Files:** `normix/exponential_family.py`

- Add `_grad_log_partition`, `_hessian_log_partition` (JAX, default via `jax.grad`/`jax.hessian`)
- Add `_log_partition_cpu`, `_grad_log_partition_cpu`, `_hessian_log_partition_cpu` (CPU, default wrapping JAX)
- Add `backend` kwarg to `expectation_params()` and `fisher_information()`
- Update `from_expectation` to construct solver calls from the triad
- Keep existing interface working (backward-compatible)

### Phase 2: Update solver interface

**Files:** `normix/fitting/solvers.py`

- Replace `grad_hess_fn` parameter with separate `grad_fn` + `hess_fn` (both theta-space)
- In `_jax_newton_raw`: when `grad_fn`/`hess_fn` provided, apply chain rule via `jax.jacobian(to_theta)` 
- In `_cpu_solve`: when `hess_fn` provided, pass it directly to scipy `minimize(hess=...)`
- Keep `_cpu_solve` hybrid path (jax.grad fallback) working for the case when neither is provided

### Phase 3: Migrate Gamma, InverseGamma, InverseGaussian

**Files:** `normix/distributions/gamma.py`, `inverse_gamma.py`, `inverse_gaussian.py`

For each:
- Add `_grad_log_partition` classmethod with existing analytical formula (from current `expectation_params`)
- Add `_hessian_log_partition` classmethod with analytical Hessian
- Delete `expectation_params` override (now derived from `_grad_log_partition` in parent)
- Delete `fit_mle` override (parent's generic version is identical)

### Phase 4: Migrate GIG

**Files:** `normix/distributions/generalized_inverse_gaussian.py`

- Add `_hessian_log_partition` classmethod (analytical 7-Bessel Hessian in theta-space, absorbing `_gig_bessel_quantities`)
- Move `_log_partition_gig_cpu` → `_log_partition_cpu` classmethod
- Move `_gig_cpu_grad` → `_grad_log_partition_cpu` classmethod
- Move `_initial_guesses` → `_initial_guesses` staticmethod
- Update `from_expectation` to use triad methods instead of module-level functions
- Delete `_analytical_grad_hess_phi` (solver handles chain rule)
- Delete `_cpu_grad_hess_theta` (solver constructs from `grad_fn` + `hess_fn`)
- Delete `_gig_bessel_quantities` (absorbed into `_hessian_log_partition`)
- Delete `fisher_information` override (now derived from `_hessian_log_partition` in parent)
- Delete `expectation_params` override (now derived from `_grad_log_partition` in parent)
- Delete `fit_mle` override

### Phase 5: Update documentation

- Update `docs/ARCHITECTURE.md` § "Exponential Family Core" and § "CPU Versions" to reflect the triad
- Update `docs/design/solver_redesign.md` § 3.1 to reflect `grad_fn` + `hess_fn` replacing `grad_hess_fn`
- Update `docs/ARCHITECTURE.md` § "GIG η→θ Optimization" to remove references to `_analytical_grad_hess_phi`

### Phase 6: Verify

- Run `uv run pytest tests/` — all existing tests must pass
- Verify GIG `from_expectation` with all backend × method combinations
- Verify that `fisher_information(backend='cpu')` matches `fisher_information(backend='jax')` for GIG
- Verify that Gamma/IG/IGauss `fit_mle` still works after deleting overrides

---

## 8. Design Decisions

| Question | Decision | Rationale |
|---|---|---|
| Separate `grad_fn` + `hess_fn` or combined `grad_hess_fn`? | Separate | Distributions provide theta-space primitives; solver handles phi-space chain rule. Eliminates `_analytical_grad_hess_phi`. |
| `@classmethod` or `@staticmethod` for triad? | `@classmethod` | Needs `cls` to resolve the correct subclass's `_log_partition_from_theta`. `cls` is not traced by JAX. |
| Share Bessel calls between grad and Hessian (7 total) or compute separately (12 total)? | Separate (12) | Saves ~90 lines of GIG-specific code (`_gig_bessel_quantities`, `_analytical_grad_hess_phi`, phi-space chain rule). 12 vs 7 Bessel calls per Newton step is a minor cost. Simplicity criterion applies. |
| CPU defaults: wrap JAX or raise NotImplemented? | Wrap JAX | Every distribution gets `backend='cpu'` for free. Only Bessel-dependent distributions (GIG) need native overrides. |
| Where does `backend` kwarg live? | Parent class | `expectation_params(backend=...)` and `fisher_information(backend=...)` on all distributions. GIG is no longer special in this regard. |
| Delete `fit_mle` overrides? | Yes | Parent's `vmap(sufficient_statistics)` + `mean` + `from_expectation` produces identical results for all four subordinator distributions. |
