"""
General-purpose η → θ solvers for exponential families.

The η → θ inversion minimises the Bregman divergence (conjugate dual):

    f(θ) = ψ(θ) − θ·η

where ψ is the log-partition function and η are the target expectation
parameters. At the minimum ∇ψ(θ*) = η.

Solvers
-------
solve_newton_scan : pure-JAX Newton with exp-reparametrisation, lax.scan
solve_lbfgs       : JAXopt L-BFGS, gradient-only, JIT-able
solve_scipy       : multi-start scipy L-BFGS-B, cold-start
solve_cpu_lbfgs   : scipy L-BFGS-B with CPU-side gradient callback

All solvers work with any ExponentialFamily subclass via its
``_log_partition_from_theta`` method.

JIT caching
-----------
``solve_scipy_multistart`` needs compiled JAX objective/gradient functions.
A module-level cache (keyed by ``id(log_partition_fn)``) ensures the JIT
compilation happens once per unique log-partition function, regardless of
how many times the solver is called with different ``eta`` values.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.constants import LOG_EPS, HESSIAN_DAMPING

# Cache of JIT'd (objective, gradient) pairs, keyed by id(log_partition_fn).
# This ensures that for a given distribution's ψ(θ), we compile once and
# reuse across all `from_expectation` calls with different η values.
_jit_cache: Dict[int, tuple] = {}


# ---------------------------------------------------------------------------
# Bregman divergence (universal objective)
# ---------------------------------------------------------------------------

def bregman_objective(
    theta: jax.Array,
    eta: jax.Array,
    log_partition_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """ψ(θ) − θ·η — the conjugate dual whose minimum gives ∇ψ(θ) = η."""
    return log_partition_fn(theta) - jnp.dot(theta, eta)


# ---------------------------------------------------------------------------
# Reparametrised objective for constrained θ
# ---------------------------------------------------------------------------

def _reparam_objective(
    phi: jax.Array,
    eta: jax.Array,
    log_partition_fn: Callable[[jax.Array], jax.Array],
    constrained_indices: Tuple[int, ...] = (),
) -> jax.Array:
    """Bregman objective in exp-reparametrised space.

    For indices in ``constrained_indices``, θᵢ = −exp(φᵢ) (enforces θᵢ < 0).
    Other indices: θᵢ = φᵢ (unconstrained).
    """
    theta = _phi_to_theta(phi, constrained_indices)
    return bregman_objective(theta, eta, log_partition_fn)


def _phi_to_theta(
    phi: jax.Array,
    constrained_indices: Tuple[int, ...],
) -> jax.Array:
    theta = phi.copy()
    for i in constrained_indices:
        theta = theta.at[i].set(-jnp.exp(phi[i]))
    return theta


def _theta_to_phi(
    theta: jax.Array,
    constrained_indices: Tuple[int, ...],
) -> jax.Array:
    phi = theta.copy()
    for i in constrained_indices:
        phi = phi.at[i].set(jnp.log(jnp.maximum(-theta[i], LOG_EPS)))
    return phi


# ---------------------------------------------------------------------------
# Newton solver via lax.scan
# ---------------------------------------------------------------------------

def solve_newton_scan(
    eta: jax.Array,
    theta0: jax.Array,
    log_partition_fn: Callable[[jax.Array], jax.Array],
    *,
    constrained_indices: Tuple[int, ...] = (),
    tol: float = 1e-10,
    scan_length: int = 20,
    grad_hess_fn: Optional[Callable] = None,
) -> jax.Array:
    """Pure-JAX Newton solver with exp-reparametrisation via lax.scan.

    Parameters
    ----------
    grad_hess_fn : optional callable(phi, eta) -> (g_phi, H_phi)
        Analytical gradient and Hessian in φ-space. If None, uses
        jax.grad and jax.hessian of the reparametrised objective.
    """
    phi0 = _theta_to_phi(theta0, constrained_indices)
    dim = phi0.shape[0]

    def obj(phi):
        return _reparam_objective(phi, eta, log_partition_fn, constrained_indices)

    if grad_hess_fn is not None:
        def _get_grad_hess(phi):
            return grad_hess_fn(phi, eta)
    else:
        _grad_fn = jax.grad(obj)
        _hess_fn = jax.hessian(obj)
        def _get_grad_hess(phi):
            return _grad_fn(phi), _hess_fn(phi)

    def newton_body(carry, _):
        phi, converged = carry
        g, H = _get_grad_hess(phi)
        H_damped = H + HESSIAN_DAMPING * jnp.eye(dim)
        delta = jnp.linalg.solve(H_damped, g)

        f0 = obj(phi)
        slope = jnp.dot(g, delta)
        alpha = _backtrack(obj, phi, delta, f0, slope)
        phi_new = phi - alpha * delta

        grad_norm = jnp.max(jnp.abs(g))
        converged = converged | (grad_norm < tol)
        phi_out = jnp.where(converged, phi, phi_new)
        return (phi_out, converged), None

    (phi_opt, _), _ = jax.lax.scan(
        newton_body, (phi0, jnp.bool_(False)), None, length=scan_length
    )
    return _phi_to_theta(phi_opt, constrained_indices)


# ---------------------------------------------------------------------------
# L-BFGS solver (JAXopt)
# ---------------------------------------------------------------------------

def solve_lbfgs(
    eta: jax.Array,
    theta0: jax.Array,
    log_partition_fn: Callable[[jax.Array], jax.Array],
    *,
    constrained_indices: Tuple[int, ...] = (),
    maxiter: int = 500,
    tol: float = 1e-10,
) -> jax.Array:
    """JAXopt L-BFGS with exp-reparametrisation."""
    import jaxopt

    phi0 = _theta_to_phi(theta0, constrained_indices)

    def obj_fn(phi):
        return _reparam_objective(phi, eta, log_partition_fn, constrained_indices)

    solver = jaxopt.LBFGS(fun=obj_fn, maxiter=maxiter, tol=tol,
                           implicit_diff=True, jit=True)
    result = solver.run(phi0)
    return _phi_to_theta(result.params, constrained_indices)


# ---------------------------------------------------------------------------
# Multi-start scipy solver (cold-start)
# ---------------------------------------------------------------------------

def solve_scipy_multistart(
    eta: jax.Array,
    theta0_list: list,
    log_partition_fn: Callable[[jax.Array], jax.Array],
    *,
    bounds: Optional[list] = None,
    theta_floor: Optional[dict] = None,
    maxiter: int = 500,
    tol: float = 1e-10,
) -> jax.Array:
    """Multi-start scipy L-BFGS-B for cold-start robustness.

    Parameters
    ----------
    bounds : list of (lower, upper) tuples per dimension
    theta_floor : dict mapping index -> floor value for clamping initial guesses

    Notes
    -----
    JIT compilation is cached by ``id(log_partition_fn)`` so that repeated
    calls with different ``eta`` values do not trigger retracing.  The
    compiled functions take ``(theta, eta)`` as JAX-traced arguments.
    """
    from scipy.optimize import minimize

    # Look up or create JIT'd objective/gradient for this log-partition function.
    fn_id = id(log_partition_fn)
    if fn_id not in _jit_cache:
        def _obj(theta, eta_):
            return bregman_objective(theta, eta_, log_partition_fn)
        _jit_cache[fn_id] = (
            jax.jit(_obj),
            jax.jit(jax.grad(_obj, argnums=0)),
        )
    obj_jit, grad_jit = _jit_cache[fn_id]

    eta_jnp = jnp.asarray(eta, dtype=jnp.float64)

    def objective_np(theta_np):
        return float(obj_jit(jnp.asarray(theta_np, dtype=jnp.float64), eta_jnp))

    def gradient_np(theta_np):
        return np.array(grad_jit(jnp.asarray(theta_np, dtype=jnp.float64), eta_jnp))

    if bounds is None:
        bounds_scipy = None
    else:
        bounds_scipy = bounds

    best_theta = None
    best_val = np.inf

    for t0 in theta0_list:
        t0_np = np.array(t0)
        if theta_floor:
            for idx, floor in theta_floor.items():
                t0_np[idx] = min(t0_np[idx], floor)
        try:
            res = minimize(
                fun=objective_np,
                x0=t0_np,
                jac=gradient_np,
                method='L-BFGS-B',
                bounds=bounds_scipy,
                options={'maxiter': maxiter, 'ftol': tol**2, 'gtol': tol},
            )
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x
        except Exception:
            pass

    if best_theta is None:
        best_theta = np.array(theta0_list[0])
        if theta_floor:
            for idx, floor in theta_floor.items():
                best_theta[idx] = min(best_theta[idx], floor)

    return jnp.asarray(best_theta, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# CPU L-BFGS-B solver (for distributions with CPU-side gradient)
# ---------------------------------------------------------------------------

def solve_cpu_lbfgs(
    eta: jax.Array,
    theta0: jax.Array,
    objective_and_grad_fn: Callable,
    *,
    bounds: Optional[list] = None,
    maxiter: int = 500,
    tol: float = 1e-10,
) -> jax.Array:
    """scipy L-BFGS-B with a user-supplied objective+gradient function.

    ``objective_and_grad_fn(theta_np, eta_np) -> (obj, grad)``
    runs on CPU (e.g. scipy Bessel). This avoids JAX dispatch overhead
    for low-dimensional problems.
    """
    from scipy.optimize import minimize

    eta_np = np.asarray(eta, dtype=np.float64)
    theta0_np = np.asarray(theta0, dtype=np.float64)

    result = minimize(
        lambda theta_np: objective_and_grad_fn(theta_np, eta_np),
        theta0_np,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds or [(-np.inf, np.inf)] * len(theta0_np),
        options={'maxiter': maxiter, 'ftol': tol ** 2, 'gtol': tol},
    )
    return jnp.asarray(result.x, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Shared backtracking line search
# ---------------------------------------------------------------------------

def _backtrack(obj, phi, delta, f0, slope, beta=0.5, c=1e-4):
    """Armijo backtracking via lax.while_loop."""
    def cond(state):
        alpha, _ = state
        return (obj(phi - alpha * delta) > f0 - c * alpha * slope) & (alpha > 1e-10)

    def body(state):
        alpha, i = state
        return (alpha * beta, i + 1)

    alpha, _ = jax.lax.while_loop(cond, body, (1.0, 0))
    return alpha
