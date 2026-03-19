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
If grad_fn=None and backend='cpu':  hybrid — jax.grad compiled → NumPy callbacks
If grad_fn provided  and backend='cpu':  pure CPU ∇f (e.g. scipy Bessel for GIG)
If grad_hess_fn provided and backend='jax', method='newton':
    user-supplied (grad, Hessian) in reparametrised φ-space (skips autodiff)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.constants import LOG_EPS, HESSIAN_DAMPING


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BregmanResult:
    """Result of a Bregman divergence minimization."""
    theta: jax.Array    # optimal θ*
    fun: float          # f(θ*) − θ*·η at solution
    grad_norm: float    # ‖∇f(θ*) − η‖∞
    num_steps: int      # iterations performed
    converged: bool     # whether tolerance was met


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
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_steps: int = 500,
    tol: float = 1e-10,
    grad_fn: Optional[Callable] = None,
    grad_hess_fn: Optional[Callable] = None,
) -> BregmanResult:
    """Minimise f(θ) − θ·η over θ.

    Parameters
    ----------
    f : convex function θ → scalar  (e.g. log-partition ψ)
    eta : target vector  (e.g. expectation parameters η)
    theta0 : initial guess
    backend : 'jax' (JIT-able) or 'cpu' (scipy, not JIT-able)
    method : 'lbfgs', 'bfgs', or 'newton'
    bounds : list of (lo, hi) per dimension; None → unconstrained.
        For backend='jax': enforced via reparameterization (except
        method='lbfgs' which uses jaxopt.LBFGSB natively).
        For backend='cpu': passed directly as scipy bounds.
    max_steps : iteration budget
    tol : convergence tolerance on ‖∇f(θ) − η‖∞
    grad_fn : θ_np → ∇f(θ) as ndarray, for backend='cpu' only.
        If None with backend='cpu', falls back to jax.grad compiled
        gradient (hybrid mode).
    grad_hess_fn : (phi, eta) → (g_phi, H_phi) in reparametrised φ-space,
        for backend='jax', method='newton' only.
        If None, jax.grad and jax.hessian are used.

    Returns
    -------
    BregmanResult
    """
    eta = jnp.asarray(eta, dtype=jnp.float64)
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)

    if backend == "jax":
        if method == "newton":
            return _wrap_jax_newton(f, eta, theta0, bounds, max_steps, tol, grad_hess_fn)
        elif method in ("lbfgs", "bfgs"):
            return _jax_quasi_newton(f, eta, theta0, bounds, max_steps, tol, method)
        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'newton', 'lbfgs', 'bfgs'.")
    elif backend == "cpu":
        return _cpu_solve(f, eta, theta0, bounds, max_steps, tol, method, grad_fn, grad_hess_fn)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'jax' or 'cpu'.")


def solve_bregman_multistart(
    f: Callable[[jax.Array], jax.Array],
    eta: jax.Array,
    theta0_batch,
    *,
    backend: str = "jax",
    method: str = "lbfgs",
    bounds: Optional[List[Tuple[float, float]]] = None,
    max_steps: int = 500,
    tol: float = 1e-10,
    grad_fn: Optional[Callable] = None,
    grad_hess_fn: Optional[Callable] = None,
) -> BregmanResult:
    """Run solve_bregman from multiple starting points; return the best result.

    Parameters
    ----------
    theta0_batch : (K, dim) jax.Array for backend='jax', method='newton'
                   (parallel via vmap); list of arrays otherwise
                   (sequential for-loop).
    """
    if backend == "jax" and method == "newton":
        return _multistart_jax_newton(
            f, eta, jnp.asarray(theta0_batch, dtype=jnp.float64),
            bounds, max_steps, tol, grad_hess_fn,
        )
    # For quasi-Newton (jaxopt vmappability uncertain) and CPU: for-loop
    theta0_list = list(theta0_batch) if not isinstance(theta0_batch, list) else theta0_batch
    return _multistart_loop(
        f, eta, theta0_list,
        backend=backend, method=method, bounds=bounds,
        max_steps=max_steps, tol=tol, grad_fn=grad_fn, grad_hess_fn=grad_hess_fn,
    )


# ---------------------------------------------------------------------------
# Reparameterization  (bounded → unconstrained)
# ---------------------------------------------------------------------------

def _setup_reparam(
    theta0: jax.Array,
    bounds: Optional[List[Tuple[float, float]]],
):
    """Return (phi0, to_theta, to_phi) for bounded ↔ unconstrained transforms.

    Per-dimension transforms based on bound type:
      (-∞, 0)  : θ = −exp(φ),  φ = log(−θ)
      (0, +∞)  : θ = exp(φ),   φ = log(θ)
      (-∞, +∞) : θ = φ         (identity)
      (lo, hi) : θ = lo+(hi−lo)·σ(φ),  φ = logit((θ−lo)/(hi−lo))
    """
    if bounds is None:
        return theta0, lambda phi: phi, lambda theta: theta

    transforms: list = []
    for lo, hi in bounds:
        lo, hi = float(lo), float(hi)
        if lo == -np.inf and hi == 0.0:
            transforms.append("neg_exp")
        elif lo == 0.0 and hi == np.inf:
            transforms.append("pos_exp")
        elif lo == -np.inf and hi == np.inf:
            transforms.append("identity")
        else:
            transforms.append(("bounded", lo, hi))

    def to_theta(phi: jax.Array) -> jax.Array:
        result = phi
        for i, t in enumerate(transforms):
            if t == "neg_exp":
                result = result.at[i].set(-jnp.exp(phi[i]))
            elif t == "pos_exp":
                result = result.at[i].set(jnp.exp(phi[i]))
            elif isinstance(t, tuple):
                lo, hi = t[1], t[2]
                result = result.at[i].set(lo + (hi - lo) * jax.nn.sigmoid(phi[i]))
        return result

    def to_phi(theta: jax.Array) -> jax.Array:
        result = theta
        for i, t in enumerate(transforms):
            if t == "neg_exp":
                result = result.at[i].set(jnp.log(jnp.maximum(-theta[i], LOG_EPS)))
            elif t == "pos_exp":
                result = result.at[i].set(jnp.log(jnp.maximum(theta[i], LOG_EPS)))
            elif isinstance(t, tuple):
                lo, hi = t[1], t[2]
                normalized = jnp.clip((theta[i] - lo) / (hi - lo), LOG_EPS, 1.0 - LOG_EPS)
                result = result.at[i].set(jnp.log(normalized / (1.0 - normalized)))
        return result

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
    grad_hess_fn,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Newton in reparametrised φ-space via lax.scan.

    Returns (theta_opt, fun, grad_norm, converged) as JAX arrays —
    all are vmappable.
    """
    phi0, to_theta, _ = _setup_reparam(theta0, bounds)
    dim = phi0.shape[0]

    def obj(phi: jax.Array) -> jax.Array:
        theta = to_theta(phi)
        return f(theta) - jnp.dot(theta, eta)

    if grad_hess_fn is not None:
        def get_g_H(phi):
            return grad_hess_fn(phi, eta)
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


def _wrap_jax_newton(f, eta, theta0, bounds, max_steps, tol, grad_hess_fn) -> BregmanResult:
    theta, fun, gn, conv = _jax_newton_raw(f, eta, theta0, bounds, max_steps, tol, grad_hess_fn)
    return BregmanResult(
        theta=theta,
        fun=float(fun),
        grad_norm=float(gn),
        num_steps=max_steps,
        converged=bool(conv),
    )


def _multistart_jax_newton(f, eta, theta0_batch, bounds, max_steps, tol, grad_hess_fn) -> BregmanResult:
    """Parallel multi-start Newton via vmap over (K, dim) starting points."""
    def solve_one(t0):
        return _jax_newton_raw(f, eta, t0, bounds, max_steps, tol, grad_hess_fn)

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

def _cpu_solve(f, eta, theta0, bounds, max_steps, tol, method, grad_fn, grad_hess_fn) -> BregmanResult:
    """scipy.optimize.minimize for the Bregman divergence."""
    from scipy.optimize import minimize

    eta_np = np.asarray(eta, dtype=np.float64)
    theta0_np = np.asarray(theta0, dtype=np.float64)
    eta_jnp = jnp.asarray(eta, dtype=jnp.float64)

    def fun_np(theta_np):
        t = jnp.asarray(theta_np, dtype=jnp.float64)
        return float(f(t) - jnp.dot(t, eta_jnp))

    if grad_fn is not None:
        def jac_np(theta_np):
            return np.asarray(grad_fn(theta_np), dtype=np.float64) - eta_np
    else:
        _grad_f = jax.grad(f)
        def jac_np(theta_np):
            t = jnp.asarray(theta_np, dtype=jnp.float64)
            return np.asarray(_grad_f(t), dtype=np.float64) - eta_np

    scipy_method = {"lbfgs": "L-BFGS-B", "bfgs": "BFGS", "newton": "trust-exact"}[method]

    kwargs: dict = {
        "jac": jac_np,
        "options": {"maxiter": max_steps, "gtol": tol},
    }
    if method == "lbfgs":
        kwargs["bounds"] = bounds
        kwargs["options"]["ftol"] = tol ** 2
    if method == "newton":
        if grad_hess_fn is None:
            raise ValueError("method='newton' with backend='cpu' requires grad_hess_fn.")
        def hess_np(theta_np):
            _, H = grad_hess_fn(np.asarray(theta_np), eta_np)
            return np.asarray(H, dtype=np.float64)
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
    f, eta, theta0_list, *, backend, method, bounds, max_steps, tol, grad_fn, grad_hess_fn,
) -> BregmanResult:
    """Sequential multi-start: try each θ₀ and return the best result."""
    best: Optional[BregmanResult] = None
    for t0 in theta0_list:
        try:
            res = solve_bregman(
                f, eta, t0,
                backend=backend, method=method, bounds=bounds,
                max_steps=max_steps, tol=tol,
                grad_fn=grad_fn, grad_hess_fn=grad_hess_fn,
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


