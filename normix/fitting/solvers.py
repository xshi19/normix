"""
Bregman divergence solvers.

Minimises  f(θ) − θ·η  over θ, where f is any convex function
(e.g. the log-partition ψ for an exponential family).
At the minimum ∇f(θ*) = η.

Public API
----------
solve_bregman
    Single starting point.
solve_bregman_multistart
    Multiple starting points (``vmap`` for JAX Newton; a Python loop
    for quasi-Newton and CPU).
bregman_objective
    Utility :math:`f(\\theta) - \\theta\\cdot\\eta`.
make_jit_newton_solver
    Build a stable ``@jax.jit`` Newton solve specialised to a fixed
    ``(f, grad_fn, hess_fn, bounds)``. Repeated calls with matching
    shapes/dtypes hit the XLA cache, avoiding the per-call re-tracing
    that ``solve_bregman`` incurs from fresh closures.

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
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.constants import (
    LOG_EPS, HESSIAN_DAMPING, THETA_FLOOR, KKT_NEAR_GAP, BREGMAN_INVERT_ATOL,
)


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
    bounds : tuple of jax.Array, or None
        ``(lower, upper)`` pair, each shape ``(d,)``; ``None`` → unconstrained.
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
    bounds : tuple of jax.Array, or None
        ``(lower, upper)`` pair, each shape ``(d,)``.
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

def _damped_hessian(H: jax.Array, damping: float = HESSIAN_DAMPING) -> jax.Array:
    r"""Relative Tikhonov: :math:`H + \lambda (\mathrm{tr}\,H / n)\,I`.

    ``damping`` is a dimensionless coefficient. Apply this to the
    **θ-space** Hessian (the Fisher), then sandwich with :math:`J`, not
    to :math:`H_\phi`. On concentrated GIG, :math:`H_\theta` entries are
    :math:`O(1/z)` while :math:`\mathrm{tr}(H_\phi)` is :math:`O(z)` from
    the bound Jacobian, so damping :math:`H_\phi` swamps the
    :math:`p`-direction. A non-positive trace falls back to
    :math:`\mathrm{mean}|H_{ii}|` rather than a zero ridge.
    """
    dim = H.shape[-1]
    tr_scale = jnp.trace(H, axis1=-2, axis2=-1) / dim
    mag_scale = jnp.mean(jnp.abs(jnp.diagonal(H, axis1=-2, axis2=-1)), axis=-1)
    scale = jnp.where(tr_scale > 0.0, tr_scale, mag_scale)
    eye = jnp.eye(dim, dtype=H.dtype)
    return H + damping * scale[..., None, None] * eye


def _natural_grad(
    f: Callable,
    eta: jax.Array,
    theta: jax.Array,
    grad_fn,
) -> jax.Array:
    """:math:`g_\\theta = \\nabla f(\\theta) - \\eta`."""
    if grad_fn is not None:
        return grad_fn(theta) - eta
    return jax.grad(lambda t: f(t) - jnp.dot(t, eta))(theta)


def _kkt_residual(
    g: jax.Array, theta: jax.Array, bounds, gap: float = -THETA_FLOOR,
) -> jax.Array:
    r"""Infinity norm of :math:`g_\theta` after dropping bound-active components.

    A coordinate within ``gap`` of a finite bound is active when the
    gradient points out of the feasible set: :math:`g_i \le 0` on an
    upper bound, :math:`g_i \ge 0` on a lower bound. That entry is the
    multiplier. The free residual is what can still be driven to ``tol``.

    The iteration uses ``gap = -THETA_FLOOR``. ``from_expectation`` also
    consults :data:`~normix.utils.constants.KKT_NEAR_GAP`, because a
    20-step budget reaches the multiplier regime before :math:`\theta`
    is within ``10^{-8}`` of the bound.

    The reported ``grad_norm`` stays :math:`\lVert g_\theta\rVert_\infty`,
    multiplier included. Stopping on :math:`\lVert g_\phi\rVert_\infty`
    is a different test: near a bound it can be tiny while a free
    coordinate of :math:`g_\theta` is still :math:`O(1)`.
    """
    if bounds is None:
        return jnp.max(jnp.abs(g))
    lower, upper = bounds
    gap_a = jnp.asarray(gap, dtype=theta.dtype)
    lower = jnp.asarray(lower, dtype=theta.dtype)
    upper = jnp.asarray(upper, dtype=theta.dtype)
    at_upper = jnp.isfinite(upper) & (theta >= upper - gap_a)
    at_lower = jnp.isfinite(lower) & (theta <= lower + gap_a)
    blocked = (at_upper & (g <= 0.0)) | (at_lower & (g >= 0.0))
    return jnp.max(jnp.abs(jnp.where(blocked, 0.0, g)))


def _inversion_converged(g, theta, bounds, tol) -> jax.Array:
    """True when the free residual met ``tol`` or :data:`BREGMAN_INVERT_ATOL`.

    The second clause uses :data:`KKT_NEAR_GAP`. It is what lets a
    trust-exact stop at :math:`\\sim 10^{-9}` and a short Newton budget
    on a bound-active GIG return a model. A residual of :math:`O(1)`
    does not pass.
    """
    kkt = _kkt_residual(g, theta, bounds)
    near = _kkt_residual(g, theta, bounds, gap=KKT_NEAR_GAP)
    tol_b = jnp.asarray(tol, dtype=jnp.result_type(g, theta))
    atol = jnp.asarray(BREGMAN_INVERT_ATOL, dtype=tol_b.dtype)
    finite = jnp.isfinite(kkt) & jnp.isfinite(near)
    return finite & ((kkt < tol_b) | (near < atol))


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
    r"""Gauss–Newton in reparametrised :math:`\phi`-space via ``lax.scan``.

    When ``grad_fn`` and ``hess_fn`` are both provided the chain rule is
    applied via ``jax.jacobian(to_theta)``:

    .. math::

        g_\phi = J^\top g_\theta, \qquad
        H_\phi = J^\top H_\theta^{\mathrm{damped}} J,

    where :math:`g_\theta = \nabla f(\theta) - \eta`, :math:`J = \partial\theta/\partial\phi`,
    and :math:`H_\theta^{\mathrm{damped}}` is the relative Tikhonov ridge
    :math:`\lambda\,\mathrm{tr}(H_\theta)/n` on the Fisher. The
    second-fundamental-form term
    :math:`\sum_i (g_\theta)_i \nabla^2\theta_i(\phi)` is omitted: it is
    indefinite away from a root, so the resulting step is ascent, and it
    is :math:`O(\lVert g_\theta\rVert)` so the local rate stays quadratic.
    For square invertible :math:`J` the step is Newton on the convex
    :math:`\theta`-problem pulled back by :math:`J^{-1}`.

    The returned ``grad_norm`` is the natural residual
    :math:`\lVert g_\theta\rVert_\infty`. Convergence uses the same
    residual after dropping components a finite bound blocks
    (:func:`_kkt_residual`). :math:`\lVert g_\phi\rVert_\infty` is not the
    stop: it can be tiny near a bound while a free coordinate of
    :math:`g_\theta` is still :math:`O(1)`. At a bound optimum the
    multiplier stays in ``grad_norm`` and does not by itself fail the solve.

    A full Gauss–Newton step is taken when it meets the Armijo test at
    step length 1. Otherwise an adaptive Levenberg–Marquardt shift
    :math:`\mu = \lVert g_\phi\rVert_\infty` (multiplied by ten until the
    full step is accepted) is added to :math:`H_\phi`. A permanent ridge
    would swamp the :math:`p`-direction of a concentrated GIG, whose
    :math:`\phi`-gradient is :math:`O(|\theta|)`. :math:`\mu` is unused
    once the undamped step is valid, so the local rate stays quadratic.
    The fallback exists because :math:`J=\mathrm{diag}(\theta)\to 0` on
    an exp bound makes :math:`H_\phi=O(\theta^2)` and the undamped step
    :math:`O(1/\theta)`, which overflows :math:`\exp`; the same long
    step, accepted at a tiny Armijo length, walks the GIG warm start
    into that bound.

    Returns ``(theta_opt, fun, grad_norm, converged)`` as JAX arrays —
    all are vmappable.
    """
    phi0, to_theta, _ = _setup_reparam(theta0, bounds)

    def obj(phi: jax.Array) -> jax.Array:
        theta = to_theta(phi)
        return f(theta) - jnp.dot(theta, eta)

    if grad_fn is not None and hess_fn is not None:
        def get_g_H(phi):
            theta = to_theta(phi)
            g_theta = grad_fn(theta) - eta
            H_theta = _damped_hessian(hess_fn(theta))
            J = jax.jacobian(to_theta)(phi)
            g_phi = J.T @ g_theta
            # Drop Σ_i (g_θ)_i ∇²θ_i(φ): indefinite away from the root.
            H_phi = J.T @ H_theta @ J
            return g_phi, H_phi, g_theta
        damp_phi = False
    else:
        _grad = jax.grad(obj)
        _hess = jax.hessian(obj)
        _grad_theta = jax.grad(lambda t: f(t) - jnp.dot(t, eta))

        def get_g_H(phi):
            return _grad(phi), _hess(phi), _grad_theta(to_theta(phi))
        damp_phi = True

    eye = jnp.eye(phi0.shape[-1], dtype=phi0.dtype)

    def _lm_step(phi, g, H, delta, f0):
        """Grow :math:`\\mu I` until a full :math:`\\phi`-step passes Armijo."""
        mu0 = jnp.maximum(jnp.max(jnp.abs(g)), LOG_EPS)

        def cond(state):
            i, _mu, _alpha, _delta, done = state
            return (~done) & (i < 8)

        def body(state):
            i, mu, alpha_lm, delta_lm, done = state
            delta_lm = jnp.linalg.solve(H + mu * eye, g)
            slope_lm = jnp.dot(g, delta_lm)
            alpha_lm = _backtrack(obj, phi, delta_lm, f0, slope_lm)
            done = (alpha_lm >= 1.0) & (slope_lm > 0.0)
            return i + 1, mu * 10.0, alpha_lm, delta_lm, done

        _i, _mu, alpha_lm, delta_lm, _done = jax.lax.while_loop(
            cond, body,
            (
                jnp.int32(0), mu0, jnp.array(0.0, dtype=phi.dtype),
                delta, jnp.bool_(False),
            ),
        )
        return phi - alpha_lm * delta_lm

    def newton_body(carry, _):
        phi, converged = carry
        g, H, g_theta = get_g_H(phi)
        H_safe = _damped_hessian(H) if damp_phi else H
        delta = jnp.linalg.solve(H_safe, g)
        f0 = obj(phi)
        slope = jnp.dot(g, delta)
        alpha = _backtrack(obj, phi, delta, f0, slope)
        if damp_phi:
            phi_new = phi - alpha * delta
        else:
            # Full step only. Backtracking this direction still follows
            # the huge flat component and walks into the bound.
            full = (alpha >= 1.0) & (slope > 0.0)
            phi_new = jax.lax.cond(
                full,
                lambda _: phi - delta,
                lambda _: _lm_step(phi, g, H_safe, delta, f0),
                operand=None,
            )
        theta = to_theta(phi)
        grad_norm = jnp.max(jnp.abs(g_theta))
        converged_new = converged | (_kkt_residual(g_theta, theta, bounds) < tol)
        phi_out = jnp.where(converged_new, phi, phi_new)
        return (phi_out, converged_new), grad_norm

    (phi_opt, converged), _ = jax.lax.scan(
        newton_body, (phi0, jnp.bool_(False)), None, length=max_steps
    )
    theta_opt = to_theta(phi_opt)
    g_final = _natural_grad(f, eta, theta_opt, grad_fn)
    grad_norm = jnp.max(jnp.abs(g_final))
    # The scan freezes once the pre-step KKT residual is under tol. The
    # last accepted step can land inside tol without a further frozen check.
    # The returned flag is wider: a free residual under BREGMAN_INVERT_ATOL
    # (multiplier dropped inside KKT_NEAR_GAP) is an inverted η.
    strict = _kkt_residual(g_final, theta_opt, bounds) < tol
    converged = converged | _inversion_converged(g_final, theta_opt, bounds, tol)
    converged = converged | strict
    final_obj = f(theta_opt) - jnp.dot(theta_opt, eta)
    return theta_opt, final_obj, grad_norm, converged


def _require_solved_theta(
    theta: jax.Array,
    converged: Any,
    *,
    grad_norm: Any = None,
) -> jax.Array:
    """Return ``theta`` only when the Bregman solve converged.

    Eager calls raise :class:`RuntimeError`. Under tracing (``jit``,
    ``lax.scan``, EM) a failed solve is NaN, so the caller cannot treat
    an uninverted :math:`\\eta` as a model. GH M-step sanity checks
    already reject non-finite subordinator parameters and keep the
    previous ones.
    """
    converged_b = jnp.asarray(converged, dtype=jnp.bool_)
    masked = jnp.where(converged_b, theta, jnp.full_like(theta, jnp.nan))
    if isinstance(theta, jax.core.Tracer) or isinstance(converged, jax.core.Tracer):
        return masked
    if bool(converged_b):
        return theta
    detail = ""
    if grad_norm is not None:
        try:
            detail = f" (‖∇f(θ)−η‖∞={float(grad_norm):.3e})"
        except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
            detail = ""
    raise RuntimeError(
        "Bregman solve did not converge"
        + detail
        + "; refusing to return a model whose expectation parameters were not inverted."
    )


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
    grad_fn : Callable
        :math:`\nabla\psi(\theta) \to \mathbb{R}^d`.
    hess_fn : Callable
        :math:`\nabla^2\psi(\theta) \to \mathbb{R}^{d\times d}`.
    bounds : tuple or None
        ``(lower, upper)``, each shape :math:`(d,)`, or ``None``.

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
    # A start that met tol outranks a lower objective that did not.
    finite = jnp.isfinite(all_fun)
    best_conv = jnp.argmin(jnp.where(all_conv & finite, all_fun, jnp.inf))
    best_any = jnp.argmin(jnp.where(finite, all_fun, jnp.inf))
    best = jnp.where(jnp.any(all_conv & finite), best_conv, best_any)
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

    # jaxopt state.error is ‖∇_φ obj‖, which is tiny on an exp bound
    # while g_θ is still O(1). Acceptance is the natural residual.
    state = result.state
    n_iter = int(state.iter_num) if hasattr(state, "iter_num") else max_steps

    return BregmanResult(
        theta=theta_opt,
        fun=final_obj,
        grad_norm=float(jnp.max(jnp.abs(g))),
        num_steps=n_iter,
        converged=bool(_inversion_converged(g, theta_opt, bounds, tol)),
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
    g = jnp.asarray(jac_val, dtype=jnp.float64)
    # scipy success is not the natural residual: trust-exact status 2
    # stops just above gtol, and L-BFGS-B can report success with a
    # large bound multiplier. Gate on the free residual.
    return BregmanResult(
        theta=theta_opt,
        fun=float(result.fun),
        grad_norm=float(np.max(np.abs(jac_val))),
        num_steps=int(result.nit),
        converged=bool(_inversion_converged(g, theta_opt, bounds, tol)),
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
            # A start that met tol outranks a lower objective that did not.
            # A non-finite objective must not block a later finite start.
            res_fin = bool(np.isfinite(float(res.fun)))
            if best is None:
                best = res
            else:
                best_fin = bool(np.isfinite(float(best.fun)))
                if res_fin and not best_fin:
                    best = res
                elif res_fin and best_fin:
                    if bool(res.converged) and not bool(best.converged):
                        best = res
                    elif (
                        bool(res.converged) == bool(best.converged)
                        and res.fun < best.fun
                    ):
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
        trial = obj(phi - alpha * delta)
        # NaN is not greater than the threshold, so a non-finite trial
        # would otherwise be accepted (exp overflow on a long φ step).
        insufficient = (~jnp.isfinite(trial)) | (trial > f0 - c * alpha * slope)
        return insufficient & (alpha > 1e-10)

    def body(state):
        alpha, i = state
        return (alpha * beta, i + 1)

    alpha, _ = jax.lax.while_loop(cond, body, (1.0, 0))
    return alpha
