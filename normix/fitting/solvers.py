"""
Bregman divergence solvers.

Minimises  f(θ) − θ·η  over θ, where f is any convex function
(e.g. the log-partition ψ for an exponential family).
At the minimum ∇f(θ*) = η.

Public API
----------
solve_bregman             single starting point
solve_bregman_multistart  multiple starting points (vmap for JAX Newton;
                          for-loop for quasi-Newton and CPU)
bregman_objective         utility: f(θ) − θ·η
make_jit_newton_solver    build a stable ``@jax.jit`` Newton solve specialised
                          to a fixed ``(f, grad_fn, hess_fn, bounds)`` —
                          repeated calls with matching shapes/dtypes hit the
                          XLA cache, avoiding the per-call re-tracing that
                          ``solve_bregman`` incurs from fresh closures.

Backends × methods
------------------
backend='jax', method='newton'  custom lax.scan Newton, autodiff or analytical Hessian
backend='jax', method='lbfgs'   jaxopt LBFGSB (bounds native) or LBFGS (reparam)
backend='jax', method='bfgs'    jaxopt BFGS with reparameterization for bounds
backend='cpu', method='lbfgs'   scipy L-BFGS-B
backend='cpu', method='bfgs'    scipy BFGS
backend='cpu', method='newton'  scipy trust-exact with Hessian

Gradient / Hessian sources
--------------------------
grad_fn  : θ → ∇f(θ).  For backend='cpu': pure CPU (numpy) gradient.
           For backend='jax', method='newton': JAX-traceable ∇ψ(θ).
           If None with backend='cpu': hybrid — jax.grad compiled → NumPy callbacks.
hess_fn  : θ → ∇²f(θ).  Required for method='newton'.
           For backend='cpu': pure CPU (numpy) Hessian.
           For backend='jax', method='newton': JAX-traceable ∇²ψ(θ).
           If None with method='newton': jax.hessian of the full objective.

Both grad_fn and hess_fn operate in theta-space only.
The solver handles all reparameterization internally via the chain rule.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.constants import LOG_EPS, HESSIAN_DAMPING


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BregmanResult:
    """Result of a Bregman divergence minimization.

    Scalar fields accept both Python and JAX types so the result
    can live inside a lax.scan carry without concretization errors.
    """
    theta: jax.Array    # optimal θ*
    fun: Any            # f(θ*) − θ*·η at solution
    grad_norm: Any      # ‖∇f(θ*) − η‖∞
    num_steps: int      # iterations performed
    converged: Any      # whether tolerance was met
    elapsed_time: float = 0.0  # wall-clock seconds


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------

def bregman_objective(
    theta: jax.Array,
    eta: jax.Array,
    f: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """f(θ) − θ·η — convex dual whose minimum gives ∇f(θ*) = η."""
    return f(theta) - jnp.dot(theta, eta)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_bregman(
    f: Callable[[jax.Array], jax.Array],
    eta: jax.Array,
    theta0: jax.Array,
    *,
    backend: str = "jax",
    method: str = "lbfgs",
    bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
    max_steps: int = 500,
    tol: float = 1e-10,
    grad_fn: Optional[Callable] = None,
    hess_fn: Optional[Callable] = None,
    verbose: int = 0,
) -> BregmanResult:
    """Minimise f(θ) − θ·η over θ.

    Parameters
    ----------
    f : convex function θ → scalar  (e.g. log-partition ψ)
    eta : target vector  (e.g. expectation parameters η)
    theta0 : initial guess
    backend : 'jax' (JIT-able) or 'cpu' (scipy, not JIT-able)
    method : 'lbfgs', 'bfgs', or 'newton'
    bounds : ``(lower, upper)`` pair of JAX arrays, each shape ``(d,)``;
        ``None`` → unconstrained.
        For backend='jax': enforced via reparameterization.
        For backend='cpu': converted to scipy format internally.
    max_steps : iteration budget
    tol : convergence tolerance on ‖∇f(θ) − η‖∞
    grad_fn : θ → ∇f(θ).
        For backend='cpu': must accept and return numpy arrays.
        For backend='jax', method='newton': must be JAX-traceable.
        If None with backend='cpu': falls back to jax.grad (hybrid mode).
    hess_fn : θ → ∇²f(θ).
        Required for method='newton'.
        For backend='cpu': must accept numpy arrays and return numpy array.
        For backend='jax', method='newton': must be JAX-traceable.
        If None with method='newton': jax.hessian of the full objective is used.
    verbose : int
        0 = silent, >= 1 = print summary after solve.

    Returns
    -------
    BregmanResult
    """
    eta = jnp.asarray(eta, dtype=jnp.float64)
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)
    t0 = time.perf_counter()

    if backend == "jax":
        if method == "newton":
            theta, fun, gn, conv = _jax_newton_raw(
                f, eta, theta0, bounds, max_steps, tol, grad_fn, hess_fn,
            )
            # No float()/bool() here — values may be JAX tracers inside lax.scan
            result = BregmanResult(
                theta=theta,
                fun=fun,
                grad_norm=gn,
                num_steps=max_steps,
                converged=conv,
                elapsed_time=time.perf_counter() - t0,
            )
        elif method in ("lbfgs", "bfgs"):
            result = _jax_quasi_newton(f, eta, theta0, bounds, max_steps, tol, method)
            result = BregmanResult(
                theta=result.theta, fun=result.fun, grad_norm=result.grad_norm,
                num_steps=result.num_steps, converged=result.converged,
                elapsed_time=time.perf_counter() - t0,
            )
        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'newton', 'lbfgs', 'bfgs'.")
    elif backend == "cpu":
        result = _cpu_solve(f, eta, theta0, bounds, max_steps, tol, method, grad_fn, hess_fn)
        result = BregmanResult(
            theta=result.theta, fun=result.fun, grad_norm=result.grad_norm,
            num_steps=result.num_steps, converged=result.converged,
            elapsed_time=time.perf_counter() - t0,
        )
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'jax' or 'cpu'.")

    if verbose >= 1:
        status = "converged" if bool(result.converged) else "NOT converged"
        print(
            f"Bregman [{backend}/{method}]: {status} in "
            f"{result.num_steps} iters ({result.elapsed_time:.3f}s), "
            f"|grad|={float(result.grad_norm):.2e}"
        )

    return result


def solve_bregman_multistart(
    f: Callable[[jax.Array], jax.Array],
    eta: jax.Array,
    theta0_batch,
    *,
    backend: str = "jax",
    method: str = "lbfgs",
    bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
    max_steps: int = 500,
    tol: float = 1e-10,
    grad_fn: Optional[Callable] = None,
    hess_fn: Optional[Callable] = None,
    verbose: int = 0,
) -> BregmanResult:
    """Run solve_bregman from multiple starting points; return the best result.

    Parameters
    ----------
    theta0_batch : (K, dim) jax.Array for backend='jax', method='newton'
                   (parallel via vmap); list of arrays otherwise
                   (sequential for-loop).
    verbose : int
        0 = silent, >= 1 = print summary.
    """
    t0 = time.perf_counter()
    if backend == "jax" and method == "newton":
        result = _multistart_jax_newton(
            f, eta, jnp.asarray(theta0_batch, dtype=jnp.float64),
            bounds, max_steps, tol, grad_fn, hess_fn,
        )
    else:
        theta0_list = list(theta0_batch) if not isinstance(theta0_batch, list) else theta0_batch
        result = _multistart_loop(
            f, eta, theta0_list,
            backend=backend, method=method, bounds=bounds,
            max_steps=max_steps, tol=tol, grad_fn=grad_fn, hess_fn=hess_fn,
        )
    elapsed = time.perf_counter() - t0
    result = BregmanResult(
        theta=result.theta, fun=result.fun, grad_norm=result.grad_norm,
        num_steps=result.num_steps, converged=result.converged,
        elapsed_time=elapsed,
    )
    if verbose >= 1:
        k = len(theta0_batch) if hasattr(theta0_batch, '__len__') else '?'
        status = "converged" if result.converged else "NOT converged"
        print(
            f"Bregman multistart [{backend}/{method}, {k} starts]: "
            f"{status} ({elapsed:.3f}s), |grad|={result.grad_norm:.2e}"
        )
    return result


# ---------------------------------------------------------------------------
# Reparameterization  (bounded → unconstrained)
# ---------------------------------------------------------------------------

_NEG_EXP = 0
_POS_EXP = 1
_IDENTITY = 2
_BOUNDED = 3


def _setup_reparam(
    theta0: jax.Array,
    bounds: Optional[Tuple[jax.Array, jax.Array]],
):
    """Return (phi0, to_theta, to_phi) for bounded ↔ unconstrained transforms.

    Per-dimension transforms based on bound type:
      (-∞, 0)  : θ = −exp(φ),  φ = log(−θ)
      (0, +∞)  : θ = exp(φ),   φ = log(θ)
      (-∞, +∞) : θ = φ         (identity)
      (lo, hi) : θ = lo+(hi−lo)·σ(φ),  φ = logit((θ−lo)/(hi−lo))

    Parameters
    ----------
    bounds : (lower, upper) pair of JAX arrays, each shape ``(d,)``, or None.
    """
    if bounds is None:
        return theta0, lambda phi: phi, lambda theta: theta

    lower, upper = bounds
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    is_neg_inf_lo = jnp.isinf(lower) & (lower < 0)
    is_pos_inf_hi = jnp.isinf(upper) & (upper > 0)

    btype = jnp.where(
        is_neg_inf_lo & (upper == 0.0), _NEG_EXP,
        jnp.where(
            (lower == 0.0) & is_pos_inf_hi, _POS_EXP,
            jnp.where(
                is_neg_inf_lo & is_pos_inf_hi, _IDENTITY,
                _BOUNDED)))

    lo_safe = jnp.where(jnp.isinf(lower), 0.0, lower)
    hi_safe = jnp.where(jnp.isinf(upper), 1.0, upper)
    span = jnp.maximum(hi_safe - lo_safe, LOG_EPS)

    def to_theta(phi: jax.Array) -> jax.Array:
        return jnp.where(btype == _NEG_EXP, -jnp.exp(phi),
               jnp.where(btype == _POS_EXP, jnp.exp(phi),
               jnp.where(btype == _BOUNDED,
                          lo_safe + span * jax.nn.sigmoid(phi),
                          phi)))

    def to_phi(theta: jax.Array) -> jax.Array:
        norm = jnp.clip((theta - lo_safe) / span, LOG_EPS, 1.0 - LOG_EPS)
        return jnp.where(btype == _NEG_EXP, jnp.log(jnp.maximum(-theta, LOG_EPS)),
               jnp.where(btype == _POS_EXP, jnp.log(jnp.maximum(theta, LOG_EPS)),
               jnp.where(btype == _BOUNDED, jnp.log(norm / (1.0 - norm)),
                          theta)))

    phi0 = to_phi(theta0)
    return phi0, to_theta, to_phi


# ---------------------------------------------------------------------------
# JAX Newton  (lax.scan, vmap-compatible)
# ---------------------------------------------------------------------------

def _jax_newton_raw(
    f: Callable,
    eta: jax.Array,
    theta0: jax.Array,
    bounds,
    max_steps: int,
    tol: float,
    grad_fn,   # theta → ∇ψ(θ), or None
    hess_fn,   # theta → ∇²ψ(θ), or None
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Newton in reparametrised φ-space via lax.scan.

    When grad_fn and hess_fn are both provided, the chain rule is applied
    generically via jax.jacobian(to_theta):
        g_phi = J^T @ (grad_fn(theta) − eta)
        H_phi = J^T @ hess_fn(theta) @ J + ∇²[to_theta(phi)·g_theta]

    Returns (theta_opt, fun, grad_norm, converged) as JAX arrays —
    all are vmappable.
    """
    phi0, to_theta, _ = _setup_reparam(theta0, bounds)
    dim = phi0.shape[0]

    def obj(phi: jax.Array) -> jax.Array:
        theta = to_theta(phi)
        return f(theta) - jnp.dot(theta, eta)

    if grad_fn is not None and hess_fn is not None:
        def get_g_H(phi):
            theta = to_theta(phi)
            g_theta = grad_fn(theta) - eta
            H_theta = hess_fn(theta)
            J = jax.jacobian(to_theta)(phi)
            g_phi = J.T @ g_theta
            # Second-order correction: ∇²[to_theta(phi)·g_theta]
            def theta_dot_g(p):
                return jnp.dot(to_theta(p), g_theta)
            H_phi = J.T @ H_theta @ J + jax.hessian(theta_dot_g)(phi)
            return g_phi, H_phi
    else:
        _grad = jax.grad(obj)
        _hess = jax.hessian(obj)
        def get_g_H(phi):
            return _grad(phi), _hess(phi)

    def newton_body(carry, _):
        phi, converged = carry
        g, H = get_g_H(phi)
        H_safe = H + HESSIAN_DAMPING * jnp.eye(dim)
        delta = jnp.linalg.solve(H_safe, g)
        f0 = obj(phi)
        slope = jnp.dot(g, delta)
        alpha = _backtrack(obj, phi, delta, f0, slope)
        phi_new = phi - alpha * delta
        grad_norm = jnp.max(jnp.abs(g))
        converged_new = converged | (grad_norm < tol)
        phi_out = jnp.where(converged_new, phi, phi_new)
        return (phi_out, converged_new), grad_norm

    (phi_opt, converged), grad_norms = jax.lax.scan(
        newton_body, (phi0, jnp.bool_(False)), None, length=max_steps
    )
    theta_opt = to_theta(phi_opt)
    final_obj = f(theta_opt) - jnp.dot(theta_opt, eta)
    return theta_opt, final_obj, grad_norms[-1], converged


# ---------------------------------------------------------------------------
# Stable jitted Newton — bake (f, grad, hess, bounds) into one JIT cache key
# ---------------------------------------------------------------------------

def make_jit_newton_solver(
    f: Callable[[jax.Array], jax.Array],
    grad_fn: Callable[[jax.Array], jax.Array],
    hess_fn: Callable[[jax.Array], jax.Array],
    bounds: Optional[Tuple[jax.Array, jax.Array]] = None,
) -> Callable:
    r"""Build a ``@jax.jit``\ -decorated Newton solver specialised to one problem.

    The returned callable has signature
    ``solve(eta, theta0, max_steps=20, tol=1e-10) -> (theta_opt, fun, grad_norm, converged)``
    where ``max_steps`` is a static argument (required by ``lax.scan``).

    All distribution-level inputs (``f``, ``grad_fn``, ``hess_fn``, ``bounds``)
    are baked into the closure at construction time. Repeated calls with the
    same array shapes and dtypes therefore reuse the compiled XLA executable.

    Use this in EM hot paths where ``solve_bregman`` would otherwise build a
    fresh Python closure on every call and force JAX to re-trace the same
    Newton kernel on each iteration.

    Parameters
    ----------
    f : convex objective ψ(θ) → scalar (must be JAX-traceable).
    grad_fn : ∇ψ(θ) → (d,).
    hess_fn : ∇²ψ(θ) → (d, d).
    bounds : ``(lower, upper)`` of shape ``(d,)`` each, or ``None``.

    Returns
    -------
    Callable
        Jit-compiled Newton solver. Returns a 4-tuple of JAX arrays
        ``(theta, fun, grad_norm, converged)``; wrap in ``BregmanResult``
        externally if needed.
    """
    @partial(jax.jit, static_argnames=("max_steps", "tol"))
    def solve(eta: jax.Array, theta0: jax.Array,
              max_steps: int = 20, tol: float = 1e-10):
        eta = jnp.asarray(eta, dtype=jnp.float64)
        theta0 = jnp.asarray(theta0, dtype=jnp.float64)
        return _jax_newton_raw(
            f, eta, theta0, bounds, max_steps, tol, grad_fn, hess_fn,
        )

    return solve


def _multistart_jax_newton(
    f, eta, theta0_batch, bounds, max_steps, tol, grad_fn, hess_fn,
) -> BregmanResult:
    """Parallel multi-start Newton via vmap over (K, dim) starting points."""
    def solve_one(t0):
        return _jax_newton_raw(f, eta, t0, bounds, max_steps, tol, grad_fn, hess_fn)

    all_theta, all_fun, all_gn, all_conv = jax.vmap(solve_one)(theta0_batch)
    best = jnp.argmin(all_fun)
    return BregmanResult(
        theta=all_theta[best],
        fun=float(all_fun[best]),
        grad_norm=float(all_gn[best]),
        num_steps=max_steps,
        converged=bool(all_conv[best]),
    )


# ---------------------------------------------------------------------------
# JAX quasi-Newton  (jaxopt LBFGS / BFGS / LBFGSB)
# ---------------------------------------------------------------------------

def _jax_quasi_newton(f, eta, theta0, bounds, max_steps, tol, method) -> BregmanResult:
    """jaxopt L-BFGS / BFGS with reparameterization for bounds.

    Note: jaxopt.LBFGSB is avoided because it has dtype incompatibilities with
    jax_enable_x64 (int32/int64 mismatch in its argsort active-set projection).
    Reparameterization is used instead for all bounded problems.
    """
    import jaxopt

    # Always reparameterize (works for both bounded and unbounded cases).
    phi0, to_theta, _ = _setup_reparam(theta0, bounds)

    def obj_phi(phi):
        theta = to_theta(phi)
        return f(theta) - jnp.dot(theta, eta)

    if method == "lbfgs":
        solver = jaxopt.LBFGS(fun=obj_phi, maxiter=max_steps, tol=tol,
                               implicit_diff=True, jit=True)
    else:  # bfgs
        solver = jaxopt.BFGS(fun=obj_phi, maxiter=max_steps, tol=tol,
                              implicit_diff=True, jit=True)
    result = solver.run(phi0)
    theta_opt = to_theta(result.params)

    g = jax.grad(lambda t: f(t) - jnp.dot(t, eta))(theta_opt)
    final_obj = float(f(theta_opt) - jnp.dot(theta_opt, eta))

    # jaxopt state attributes vary between LBFGS / LBFGSB
    state = result.state
    n_iter = int(state.iter_num) if hasattr(state, "iter_num") else max_steps
    err = float(state.error) if hasattr(state, "error") else float("nan")

    return BregmanResult(
        theta=theta_opt,
        fun=final_obj,
        grad_norm=float(jnp.max(jnp.abs(g))),
        num_steps=n_iter,
        converged=bool(err < tol) if not np.isnan(err) else False,
    )


# ---------------------------------------------------------------------------
# CPU backend  (scipy.optimize.minimize)
# ---------------------------------------------------------------------------

def _cpu_solve(
    f, eta, theta0, bounds, max_steps, tol, method, grad_fn, hess_fn,
) -> BregmanResult:
    """scipy.optimize.minimize for the Bregman divergence.

    Two modes depending on whether grad_fn is provided:

    Pure CPU (grad_fn given): f and grad_fn accept numpy arrays.
        No JAX dispatch — suitable when f itself uses scipy (e.g. CPU Bessel).
    Hybrid  (grad_fn=None): f is a JAX function; gradient via jax.grad.
        Inputs are converted to jnp arrays for tracing.
    """
    from scipy.optimize import minimize

    eta_np = np.asarray(eta, dtype=np.float64)
    theta0_np = np.asarray(theta0, dtype=np.float64)

    if grad_fn is not None:
        def fun_np(theta_np):
            return float(f(theta_np)) - float(np.dot(theta_np, eta_np))

        def jac_np(theta_np):
            return np.asarray(grad_fn(theta_np), dtype=np.float64) - eta_np
    else:
        eta_jnp = jnp.asarray(eta, dtype=jnp.float64)
        _grad_f = jax.grad(f)

        def fun_np(theta_np):
            t = jnp.asarray(theta_np, dtype=jnp.float64)
            return float(f(t) - jnp.dot(t, eta_jnp))

        def jac_np(theta_np):
            t = jnp.asarray(theta_np, dtype=jnp.float64)
            return np.asarray(_grad_f(t), dtype=np.float64) - eta_np

    scipy_method = {"lbfgs": "L-BFGS-B", "bfgs": "BFGS", "newton": "trust-exact"}[method]

    scipy_bounds = None
    if bounds is not None:
        lo_np = np.asarray(bounds[0], dtype=np.float64)
        hi_np = np.asarray(bounds[1], dtype=np.float64)
        scipy_bounds = list(zip(lo_np, hi_np))

    kwargs: dict = {
        "jac": jac_np,
        "options": {"maxiter": max_steps, "gtol": tol},
    }
    if method == "lbfgs":
        kwargs["bounds"] = scipy_bounds
        kwargs["options"]["ftol"] = tol ** 2
    if method == "newton":
        if hess_fn is None:
            raise ValueError(
                "method='newton' with backend='cpu' requires hess_fn. "
                "Provide _hessian_log_partition_cpu or equivalent."
            )
        def hess_np(theta_np):
            return np.asarray(
                hess_fn(np.asarray(theta_np, dtype=np.float64)), dtype=np.float64)
        kwargs["hess"] = hess_np

    result = minimize(fun_np, theta0_np, method=scipy_method, **kwargs)
    theta_opt = jnp.asarray(result.x, dtype=jnp.float64)
    jac_val = result.jac if result.jac is not None else jac_np(result.x)
    return BregmanResult(
        theta=theta_opt,
        fun=float(result.fun),
        grad_norm=float(np.max(np.abs(jac_val))),
        num_steps=int(result.nit),
        converged=bool(result.success),
    )


# ---------------------------------------------------------------------------
# Multi-start: Python for-loop  (CPU, and JAX quasi-Newton)
# ---------------------------------------------------------------------------

def _multistart_loop(
    f, eta, theta0_list, *, backend, method, bounds, max_steps, tol, grad_fn, hess_fn,
) -> BregmanResult:
    """Sequential multi-start: try each θ₀ and return the best result."""
    best: Optional[BregmanResult] = None
    for t0 in theta0_list:
        try:
            res = solve_bregman(
                f, eta, t0,
                backend=backend, method=method, bounds=bounds,
                max_steps=max_steps, tol=tol,
                grad_fn=grad_fn, hess_fn=hess_fn,
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass

    if best is None:
        # All starts failed — attempt once without catching
        best = solve_bregman(
            f, eta, theta0_list[0],
            backend=backend, method=method, bounds=bounds,
            max_steps=max_steps, tol=tol,
        )
    return best


# ---------------------------------------------------------------------------
# Shared backtracking line search
# ---------------------------------------------------------------------------

def _backtrack(obj, phi, delta, f0, slope, beta: float = 0.5, c: float = 1e-4):
    """Armijo backtracking via lax.while_loop."""
    def cond(state):
        alpha, _ = state
        return (obj(phi - alpha * delta) > f0 - c * alpha * slope) & (alpha > 1e-10)

    def body(state):
        alpha, i = state
        return (alpha * beta, i + 1)

    alpha, _ = jax.lax.while_loop(cond, body, (1.0, 0))
    return alpha
