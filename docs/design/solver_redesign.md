# Solver Module Redesign

**Date:** 2026-03-18  
**Status:** Proposal (v2)  
**Scope:** `normix/fitting/solvers.py`, interactions with `ExponentialFamily.from_expectation`

---

## 1. Motivation

The current `fitting/solvers.py` has grown organically and has several issues:

1. **Naming confusion.** `solve_scipy_multistart` conflates the scipy backend with the
   multi-start strategy. Multi-start is orthogonal to solver choice — any solver can
   be run from multiple starting points. The real distinction between the current four
   solvers is *backend* (JAX vs CPU) and *method* (Newton vs L-BFGS).

2. **Duplicate optimization in the base class.** `ExponentialFamily.from_expectation`
   has its own inline `jaxopt.LBFGS`/`LBFGSB` call, while `solvers.py` provides the
   "real" solvers used by GIG. Simpler distributions (Gamma, InverseGaussian) override
   `from_expectation` with closed-form solutions and never touch either. The base-class
   optimizer is dead code for distributions that override, and under-featured for GIG.

3. **Unnecessary coupling to exponential families.** The solver takes
   `log_partition_fn` by name, but the Bregman divergence min_θ [f(θ) − θ·η] works
   for *any* convex function f. The solver should be agnostic to whether f is a
   log-partition function.

4. **No `vmap`-based multi-start.** Pure-JAX solvers (Newton, L-BFGS) are called on a
   single `theta0`. Multi-start is only available via the scipy backend's Python
   for-loop. For pure-JAX solvers, `vmap` over initial points would give parallel
   multi-start with no additional code complexity.

5. **Fixed iteration count.** The Newton solver uses `lax.scan` with a fixed
   `scan_length` and simulates early stopping via a `converged` flag that freezes
   updates. This wastes computation when convergence is fast.

---

## 2. Library Choices

### 2.1 JAXopt vs Optax vs Optimistix

| Library | Newton | LBFGS | LBFGSB (box) | JIT | Status |
|---|---|---|---|---|---|
| **JAXopt** | No | Yes | **Yes** | Yes | Migrating to Optax |
| **Optax** | No | Yes (from JAXopt) | **No** | Yes | Active (ML focus) |
| **Optimistix** | Root-find only | Yes | No | Yes | Active (Kidger) |

**Decision: keep JAXopt for now.**

- JAXopt works, we already depend on it, and its `LBFGSB` is the only pure-JAX solver
  with native box constraints.
- Optax absorbed L-BFGS from JAXopt but *not* LBFGSB, and has no Newton. Switching
  loses functionality.
- Optimistix is elegant and Equinox-native, but has no box constraints and its Newton
  is only for root-finding (not minimization). It could be used in the future — see
  §2.2.
- When JAXopt is eventually deprecated, we can migrate to whichever library absorbs
  LBFGSB, or use reparameterization + Optimistix unconstrained solvers.

### 2.2 Newton: custom vs library

**Why we write Newton ourselves:**

No existing JAX library provides a Newton *minimizer* that accepts a user-supplied
Hessian. Optimistix has `Newton` for root-finding, and its quasi-Newton minimizers
(BFGS, L-BFGS) always autodiff. JAXopt has no Newton minimizer.

For the GIG analytical Hessian (7 `log_kv` calls vs ~25 for `jax.hessian`), a custom
Newton is essential. More generally, when the Hessian of the convex function f is
available cheaply (it's the Fisher information for exponential families), supplying it
directly is a significant performance win.

**Optimistix Newton root-finding as an alternative path:**

The η→θ inversion *is* a root-finding problem: find θ such that ∇f(θ) = η. This
could be solved via Optimistix:

```python
import optimistix as optx
result = optx.root_find(
    lambda theta, eta: jax.grad(f)(theta) - eta,
    optx.Newton(rtol=tol, atol=tol), theta0, args=eta,
)
```

This is elegant but doesn't support a user-supplied Jacobian (= Hessian of f), so it
won't help for the GIG analytical Hessian case. Worth considering for distributions
where autodiff Hessians are cheap (dim ≤ 3 without Bessel functions).

### 2.3 paramax vs hand-rolled reparameterization

**Decision: keep hand-rolled.** Our reparameterization is 8 lines, trivially
understandable. Adding `paramax` as a dependency for this alone is not justified.

---

## 3. Proposed API

### 3.1 `solve_bregman` — single entry point

The Bregman divergence minimization min_θ [f(θ) − θ·η] works for any convex f.
The API should reflect this generality:

```python
def solve_bregman(
    f: Callable[[Array], Array],
    eta: Array,
    theta0: Array,
    *,
    backend: str = "jax",
    method: str = "lbfgs",
    bounds: list[tuple[float, float]] | None = None,
    max_steps: int = 500,
    tol: float = 1e-10,
    grad_fn: Callable | None = None,
    grad_hess_fn: Callable | None = None,
) -> BregmanResult:
    """Minimise f(θ) − θ·η over θ.

    Parameters
    ----------
    f : convex function θ → scalar (e.g. log-partition ψ)
    eta : target (e.g. expectation parameters)
    theta0 : initial guess
    backend : 'jax' or 'cpu'
    method : 'newton', 'lbfgs', 'bfgs'
    bounds : per-dimension (lower, upper) bounds on θ
    max_steps : iteration budget
    tol : convergence tolerance (gradient norm)
    grad_fn : ∇f(θ) → Array.  If None, derived from f.
    grad_hess_fn : (∇f(θ), ∇²f(θ)) → (Array, Array).  If None, derived from f.
    """
```

**How backend × method × bounds interact:**

| backend | method | bounds | What happens |
|---|---|---|---|
| `jax` | `newton` | None | JAX Newton via `lax.while_loop`, autodiff or supplied `grad_hess_fn` |
| `jax` | `newton` | given | Reparameterize → optimize in unconstrained φ-space → map back |
| `jax` | `lbfgs` | None | `jaxopt.LBFGS` (or Optimistix in future) |
| `jax` | `lbfgs` | given | `jaxopt.LBFGSB` with bounds passed directly |
| `jax` | `bfgs` | None | `jaxopt.BFGS` (or Optimistix in future) |
| `jax` | `bfgs` | given | Reparameterize → unconstrained BFGS |
| `cpu` | `lbfgs` | any | `scipy.optimize.minimize(method='L-BFGS-B')` |
| `cpu` | `bfgs` | any | `scipy.optimize.minimize(method='BFGS')` |
| `cpu` | `newton` | any | `scipy.optimize.minimize(method='trust-exact')` with Hessian |

**Gradient logic:**

| `grad_fn` | `grad_hess_fn` | backend | Behavior |
|---|---|---|---|
| None | None | `jax` | `jax.grad(f)` and `jax.hessian(f)` (for Newton) |
| None | None | `cpu` | **Hybrid:** `jax.grad(f)` compiled to NumPy callback + scipy optimizer |
| provided | None | `cpu` | Pure CPU: user-supplied ∇f (e.g. scipy Bessel) + scipy optimizer |
| None | provided | `jax` | User-supplied ∇f + ∇²f (e.g. GIG analytical 7-Bessel Hessian) |
| provided | provided | either | User-supplied gradient and Hessian used directly |

The key insight: `objective_and_grad_fn` is unnecessary. The objective is always
`f(θ) − θ·η`, so the Bregman gradient is `grad_fn(θ) − η`. We only need `grad_fn`
(the gradient of the convex function f itself).

### 3.2 Result object

```python
@dataclass(frozen=True)
class BregmanResult:
    theta: jax.Array       # optimal θ*
    fun: float             # f(θ*) − θ*·η at solution
    grad_norm: float       # ‖∇f(θ*) − η‖∞
    num_steps: int         # iterations used
    converged: bool        # whether tolerance was met
```

### 3.3 Multi-start wrapper

```python
def solve_bregman_multistart(
    f: Callable,
    eta: Array,
    theta0_batch: Array,      # (K, dim) for JAX; list of arrays for CPU
    *,
    backend: str = "jax",
    method: str = "lbfgs",
    **kwargs,
) -> BregmanResult:
    """Run solve_bregman from K starting points, return the best."""
```

| backend | Multi-start mechanism |
|---|---|
| `jax` | `jax.vmap` over `theta0_batch` — parallel solves |
| `cpu` | Python for-loop over `theta0_batch` |

For `backend='jax'`: `vmap` over Optimistix/JAXopt solvers works because they use
`lax.while_loop` internally. Under `vmap`, all K starts run for `max_steps`
(converged ones no-op), then we pick the best. For d=3 and K=10, this is fast.

### 3.4 Integration with `ExponentialFamily`

```python
# In ExponentialFamily
@classmethod
def from_expectation(cls, eta, *, theta0=None, backend="jax",
                     method="lbfgs", **solver_kwargs):
    eta = jnp.asarray(eta, dtype=jnp.float64)
    if theta0 is None:
        theta0 = cls._init_theta_from_eta(eta)
    result = solve_bregman(
        f=cls._static_log_partition(),
        eta=eta, theta0=theta0,
        backend=backend, method=method,
        bounds=cls._theta_bounds(),
        **solver_kwargs,
    )
    return cls.from_natural(result.theta)
```

Each subclass provides:
- `_static_log_partition()` → stateless `Callable[[Array], Array]`
- `_theta_bounds()` → `[(lo, hi), ...]` or `None`
- Subclasses with closed-form η→θ (Gamma, InverseGaussian, InverseGamma)
  override `from_expectation` entirely — no solver needed.

---

## 4. Detailed Design

### 4.1 Bounds handling: reparameterization for JAX, native for scipy

When `bounds` is provided:

**`backend='jax'`:** Reparameterize to unconstrained space, optimize there, map back.
The reparameterization for each dimension is determined by the bound type:

| Bound | Transform θ → φ | Inverse φ → θ |
|---|---|---|
| `(-∞, 0)` | `φ = log(-θ)` | `θ = -exp(φ)` |
| `(0, +∞)` | `φ = log(θ)` | `θ = exp(φ)` |
| `(lo, hi)` | `φ = logit((θ-lo)/(hi-lo))` | `θ = lo + (hi-lo)·σ(φ)` |
| `(-∞, +∞)` | `φ = θ` | `θ = φ` (identity) |

This is the same approach as the current `_phi_to_theta` / `_theta_to_phi`, but
generalized from hard-coded "negative indices" to arbitrary bounds.

**`backend='cpu'`:** Pass bounds directly to `scipy.optimize.minimize` (native L-BFGS-B
box constraints). No reparameterization needed.

Exception: `jaxopt.LBFGSB` (method `lbfgs` with `backend='jax'`) supports bounds
natively, so no reparameterization is needed for that combination either.

### 4.2 Newton solver with `lax.while_loop`

Replace `lax.scan` (fixed count) with `lax.while_loop` (true early stopping):

```python
def _jax_newton(f, eta, theta0, bounds, max_steps, tol, grad_hess_fn):
    phi0, to_theta, to_phi = _setup_reparam(theta0, bounds)
    dim = phi0.shape[0]

    if grad_hess_fn is not None:
        get_g_H = lambda phi: grad_hess_fn(to_theta(phi))
    else:
        obj = lambda phi: f(to_theta(phi)) - jnp.dot(to_theta(phi), eta)
        get_g_H = lambda phi: (jax.grad(obj)(phi), jax.hessian(obj)(phi))

    def cond(state):
        return (~state.converged) & (state.step < max_steps)

    def body(state):
        g, H = get_g_H(state.phi)
        H_safe = H + HESSIAN_DAMPING * jnp.eye(dim)
        delta = jnp.linalg.solve(H_safe, g)
        alpha = _backtrack(...)
        phi_new = state.phi - alpha * delta
        converged = jnp.max(jnp.abs(g)) < tol
        return State(phi_new, state.step + 1, converged)

    final = jax.lax.while_loop(cond, body, State(phi0, 0, False))
    return to_theta(final.phi)
```

**`lax.scan` variant for vmap multi-start:** `lax.while_loop` under `vmap` runs all
branches for `max_steps` anyway (converged ones just no-op with identical state). So
the efficiency difference between `scan` and `while_loop` under `vmap` is minimal.
We can use `while_loop` uniformly and simplify.

### 4.3 JAX L-BFGS / BFGS via JAXopt

For `backend='jax'` with `method='lbfgs'` or `method='bfgs'`:

```python
def _jax_lbfgs(f, eta, theta0, bounds, max_steps, tol):
    import jaxopt

    if bounds is not None:
        # LBFGSB: pass bounds directly, no reparameterization
        def obj(theta):
            return f(theta) - jnp.dot(theta, eta)
        solver = jaxopt.LBFGSB(fun=obj, maxiter=max_steps, tol=tol)
        result = solver.run(theta0, bounds=bounds)
    else:
        def obj(theta):
            return f(theta) - jnp.dot(theta, eta)
        solver = jaxopt.LBFGS(fun=obj, maxiter=max_steps, tol=tol)
        result = solver.run(theta0)
    return result.params
```

**Future migration path:** When JAXopt is fully deprecated, replace with
Optimistix BFGS/LBFGS (unconstrained) + reparameterization for bounds. Or
adopt whatever absorbs LBFGSB from JAXopt.

### 4.4 CPU backend via scipy

For `backend='cpu'`, all methods delegate to `scipy.optimize.minimize`:

```python
def _cpu_solve(f, eta, theta0, bounds, max_steps, tol, method, grad_fn, grad_hess_fn):
    from scipy.optimize import minimize

    # Build objective: always f(θ) − θ·η
    if grad_fn is not None:
        # Pure CPU path: user-supplied ∇f (e.g. scipy Bessel for GIG)
        def objective_np(theta_np):
            return float(f_cpu(theta_np)) - np.dot(theta_np, eta_np)
        def gradient_np(theta_np):
            return np.asarray(grad_fn(theta_np)) - eta_np
    else:
        # Hybrid path: jax.grad compiled + scipy optimizer
        obj_jit, grad_jit = _compile_jax_obj_and_grad(f)
        def objective_np(theta_np):
            return float(obj_jit(jnp.array(theta_np), eta_jnp))
        def gradient_np(theta_np):
            return np.array(grad_jit(jnp.array(theta_np), eta_jnp))

    scipy_method = {'lbfgs': 'L-BFGS-B', 'bfgs': 'BFGS',
                    'newton': 'trust-exact'}[method]
    kwargs = {'jac': gradient_np, 'bounds': bounds,
              'options': {'maxiter': max_steps, 'gtol': tol}}

    if method == 'newton' and grad_hess_fn is not None:
        kwargs['hess'] = lambda t: np.asarray(grad_hess_fn(t)[1])

    result = minimize(objective_np, np.asarray(theta0), method=scipy_method, **kwargs)
    return jnp.asarray(result.x, dtype=jnp.float64)
```

Scipy `method` mapping:

| Our `method` | Scipy `method` | Needs Hessian | Supports bounds |
|---|---|---|---|
| `lbfgs` | `L-BFGS-B` | No | Yes (box) |
| `bfgs` | `BFGS` | No | No |
| `newton` | `trust-exact` | Yes | No (use reparameterization) |

`trust-constr` is another option for Newton + box constraints; can be added later.

### 4.5 JIT cache for hybrid (JAX grad + scipy) path

```python
import functools

@functools.lru_cache(maxsize=32)
def _compile_jax_obj_and_grad(f):
    """JIT-compile Bregman objective and gradient for a given convex function."""
    def _obj(theta, eta):
        return f(theta) - jnp.dot(theta, eta)
    return jax.jit(_obj), jax.jit(jax.grad(_obj, argnums=0))
```

Replaces the current `_jit_cache` dict keyed by `id(log_partition_fn)`.

---

## 5. Multi-start Design

### 5.1 JAX multi-start via vmap

```python
def _multistart_jax(f, eta, theta0_batch, method, bounds, max_steps, tol, **kw):
    """Parallel multi-start. theta0_batch: (K, dim)."""
    def solve_one(t0):
        result = solve_bregman(f, eta, t0, backend='jax', method=method,
                               bounds=bounds, max_steps=max_steps, tol=tol, **kw)
        return result.theta, result.fun

    all_theta, all_obj = jax.vmap(solve_one)(theta0_batch)
    best = jnp.argmin(all_obj)
    return all_theta[best]
```

Under `vmap`, `lax.while_loop` runs all K starts for `max_steps` (converged ones
no-op). For d=3 and K=10–20, this overhead is negligible.

### 5.2 CPU multi-start via for-loop

```python
def _multistart_cpu(f, eta, theta0_list, method, bounds, max_steps, tol, **kw):
    best_theta, best_val = None, np.inf
    for t0 in theta0_list:
        result = solve_bregman(f, eta, t0, backend='cpu', method=method,
                               bounds=bounds, max_steps=max_steps, tol=tol, **kw)
        if result.fun < best_val:
            best_val, best_theta = result.fun, result.theta
    return best_theta
```

---

## 6. Other Algorithms to Consider

### 6.1 Trust-region Newton (scipy `trust-constr`)

For dim=3 problems like GIG, `trust-constr` combines trust-region robustness with
second-order convergence and native box constraints. Since we already have the
analytical Hessian, the per-iteration cost is low. Could expose as
`method='newton'` + `backend='cpu'` with bounds.

### 6.2 Natural gradient = Newton on Bregman

For exponential families, the Hessian of ψ(θ) is the Fisher information I(θ).
Newton's method on the Bregman objective is therefore equivalent to natural gradient
descent. No separate implementation needed — the Newton solver already does this.

### 6.3 Nelder-Mead (derivative-free fallback)

Useful near GIG degenerate limits where Bessel derivatives have numerical issues.
Both Optimistix and scipy provide Nelder-Mead. Could expose as
`method='nelder_mead'` in the future.

### 6.4 Optimistix root-finding for η→θ

Since η→θ is ∇f(θ) = η, it's a root-finding problem. Optimistix's
`root_find(residual, Newton(), theta0, args=eta)` is a clean alternative when no
custom Hessian is needed (autodiff is acceptable). Worth exploring for distributions
simpler than GIG.

---

## 7. Module Structure

```python
# normix/fitting/solvers.py

# --- Result ---
@dataclass(frozen=True)
class BregmanResult: ...

# --- Public API ---
def solve_bregman(f, eta, theta0, *, backend, method, bounds, ...) -> BregmanResult
def solve_bregman_multistart(f, eta, theta0_batch, *, backend, method, ...) -> BregmanResult
def bregman_objective(theta, eta, f) -> scalar

# --- JAX backends ---
def _jax_newton(f, eta, theta0, bounds, max_steps, tol, grad_hess_fn) -> Array
def _jax_lbfgs(f, eta, theta0, bounds, max_steps, tol) -> Array
def _jax_bfgs(f, eta, theta0, bounds, max_steps, tol) -> Array

# --- CPU backends ---
def _cpu_solve(f, eta, theta0, bounds, max_steps, tol, method, grad_fn, grad_hess_fn) -> Array

# --- Reparameterization ---
def _setup_reparam(theta0, bounds) -> (phi0, to_theta_fn, to_phi_fn)
def _backtrack(obj, phi, delta, f0, slope) -> alpha

# --- JIT cache ---
@lru_cache
def _compile_jax_obj_and_grad(f) -> (obj_jit, grad_jit)
```

Estimated: ~250 lines (vs 320 current), cleaner separation.

---

## 8. Migration Plan

### Phase 1: Unified API (non-breaking)

- Add `solve_bregman`, `solve_bregman_multistart`, `BregmanResult`.
- Internally dispatch to existing `solve_*` functions.
- Update `GIG.from_expectation` to use `solve_bregman`.
- Old functions remain as deprecated aliases.

### Phase 2: Base class unification

- `ExponentialFamily.from_expectation` delegates to `solve_bregman`.
- Remove inline `jaxopt.LBFGS`/`LBFGSB` from `exponential_family.py`.
- Each subclass provides `_static_log_partition()` and `_theta_bounds()`.

### Phase 3: Cleanup

- Remove deprecated `solve_newton_scan`, `solve_lbfgs`, `solve_scipy_multistart`,
  `solve_cpu_lbfgs`.
- Add vmap multi-start for JAX backend.
- Update `ARCHITECTURE.md` and tech notes.

### Phase 4 (future): Evaluate Optimistix

- If JAXopt becomes unusable, evaluate Optimistix BFGS/LBFGS + reparameterization
  as replacement. Also evaluate `optx.root_find(Newton)` for simpler distributions.

---

## 9. Dependency Impact

| Dependency | Current | Proposed | Notes |
|---|---|---|---|
| `jaxopt` | Required | **Keep** | LBFGSB is uniquely useful; revisit when deprecated |
| `scipy` | Required | Keep | CPU backend |
| `optimistix` | — | Not adding (yet) | Future option when JAXopt dies |
| `paramax` | — | Not adding | Hand-rolled reparam is sufficient |

---

## 10. Summary of Design Decisions

| Question | Decision | Rationale |
|---|---|---|
| JAXopt vs Optax? | Keep JAXopt | Optax lacks LBFGSB and Newton |
| Add Optimistix? | Not yet | JAXopt works; switch later if needed |
| Add paramax? | No | 8 lines of code vs a dependency |
| `log_partition_fn` or `f`? | General `f` | Bregman works for any convex function |
| API axes? | `backend` × `method` | Consistent with distribution API (`backend='jax'`/`'cpu'`) |
| Constraints vs bounds? | `bounds` (reparameterize for JAX) | Simpler; bounds are sufficient for all current distributions |
| `objective_and_grad_fn`? | Replace with `grad_fn` | Objective is always f(θ)−θ·η; only ∇f needs to be custom |
| Multi-start? | `vmap` for JAX, for-loop for CPU | Orthogonal wrapper, not baked into solver name |
| Newton implementation? | Custom `lax.while_loop` | No library provides Newton minimizer with custom Hessian |
| Early stopping? | `lax.while_loop` | True early stop; under vmap, equivalent to scan anyway |
