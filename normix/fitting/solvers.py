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
    Ordered starting points, stopping at the first finite converged
    JAX solve. CPU solves select the best result from a Python loop.
bregman_objective
    Utility :math:`f(\\theta) - \\theta\\cdot\\eta`.
make_jit_newton_solver
    Build a stable ``@jax.jit`` Newton solve specialised to a fixed
    ``(f, grad_fn, hess_fn, bounds)``. Repeated calls with matching
    shapes/dtypes hit the XLA cache, avoiding the per-call re-tracing
    that ``solve_bregman`` incurs from fresh closures.

Backends × methods
------------------
backend='jax', method='newton'  custom lax.while_loop Newton, autodiff or analytical Hessian
backend='jax', method='lbfgs'   jaxopt LBFGSB (bounds native) or LBFGS (reparam)
backend='jax', method='bfgs'    jaxopt BFGS with reparameterization for bounds
backend='cpu', method='lbfgs'   scipy L-BFGS-B (native bounds)
backend='cpu', method='bfgs'    scipy BFGS; bounds via the same reparametrization
backend='cpu', method='newton'  scipy trust-exact; bounds via the same reparametrization

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
from dataclasses import dataclass, replace
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
    num_steps: Any      # steps taken (SciPy nit, jaxopt iter_num)
    converged: Any      # _inversion_converged; traceable
    elapsed_time: float = 0.0  # wall-clock seconds
    kkt_residual: Any = None  # free residual of g_θ (bound multipliers dropped)
    reason: Any = None  # int code, eager and traced; names in _REASON_NAMES


class SolverDidNotConverge(RuntimeError):
    """Eager Bregman inversion did not meet the residual tolerance."""


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


_REASON_CONVERGED = 0
_REASON_ATOL = 1
_REASON_MAX_STEPS = 2
_REASON_NONFINITE = 3
_REASON_NAMES = ("converged", "atol", "max_steps", "non_finite")


def _stop_parts(kkt: jax.Array, near: jax.Array, tol: float):
    """Finite flag, strict tolerance, and the atol clause. One predicate."""
    tol_b = jnp.asarray(tol, dtype=jnp.result_type(kkt, near))
    atol = jnp.asarray(BREGMAN_INVERT_ATOL, dtype=tol_b.dtype)
    finite = jnp.isfinite(kkt) & jnp.isfinite(near)
    strict = finite & (kkt < tol_b)
    atol_hit = finite & ~strict & (near < atol)
    return finite, strict, atol_hit


def _stop_reason(kkt: jax.Array, near: jax.Array, tol: float) -> jax.Array:
    """Integer stop code. See :data:`_REASON_NAMES`."""
    finite, strict, atol_hit = _stop_parts(kkt, near, tol)
    return jnp.where(
        ~finite,
        jnp.int32(_REASON_NONFINITE),
        jnp.where(
            strict,
            jnp.int32(_REASON_CONVERGED),
            jnp.where(atol_hit, jnp.int32(_REASON_ATOL), jnp.int32(_REASON_MAX_STEPS)),
        ),
    )


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
    f : Callable
        Convex function θ → scalar (e.g. log-partition ψ).
    eta : jax.Array
        Target vector (e.g. expectation parameters η).
    theta0 : jax.Array
        Initial guess.
    backend : 'jax' or 'cpu'
        ``'jax'`` is JIT-able; ``'cpu'`` uses scipy and is not JIT-able.
    method : 'lbfgs', 'bfgs', or 'newton'
    bounds : tuple of jax.Array, or None
        ``(lower, upper)`` pair, each shape ``(d,)``; ``None`` → unconstrained.
        JAX Newton, JAX BFGS, CPU BFGS, and CPU trust-exact enforce bounds
        by reparameterization. CPU L-BFGS-B takes them natively.
    max_steps : int
        Iteration budget.
    tol : float
        Convergence tolerance on ‖∇f(θ) − η‖∞.
        On a reparametrised CPU solve, SciPy ``gtol`` is the φ-gradient.
        ``converged`` is still the free θ residual.
    grad_fn : Callable, optional
        θ → ∇f(θ).
        For backend='cpu': must accept and return numpy arrays.
        For backend='jax', method='newton': must be JAX-traceable.
        If None with backend='cpu': falls back to jax.grad (hybrid mode).
    hess_fn : Callable, optional
        θ → ∇²f(θ).
        Required for method='newton'.
        For backend='cpu': must accept numpy arrays and return numpy array.
        For backend='jax', method='newton': must be JAX-traceable.
        If None with method='newton': jax.hessian of the full objective is used.
    verbose : int
        0 = silent, >= 1 = print summary after solve.
        Summaries are skipped while tracing.

    Returns
    -------
    BregmanResult
    """
    eta = jnp.asarray(eta, dtype=jnp.float64)
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)
    t0 = time.perf_counter()

    if backend == "jax":
        if method == "newton":
            theta, fun, gn, conv, n_iter, kkt, reason = _jax_newton_raw(
                f, eta, theta0, bounds, max_steps, tol, grad_fn, hess_fn,
            )
            result = BregmanResult(
                theta=theta,
                fun=fun,
                grad_norm=gn,
                num_steps=n_iter,
                converged=conv,
                elapsed_time=time.perf_counter() - t0,
                kkt_residual=kkt,
                reason=reason,
            )
        elif method in ("lbfgs", "bfgs"):
            result = _jax_quasi_newton(f, eta, theta0, bounds, max_steps, tol, method)
            result = replace(result, elapsed_time=time.perf_counter() - t0)
        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'newton', 'lbfgs', 'bfgs'.")
    elif backend == "cpu":
        result = _cpu_solve(f, eta, theta0, bounds, max_steps, tol, method, grad_fn, hess_fn)
        result = replace(result, elapsed_time=time.perf_counter() - t0)
    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'jax' or 'cpu'.")

    if verbose >= 1 and not isinstance(result.grad_norm, jax.core.Tracer):
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
    """Solve from ordered starting points, preferring finite convergence.

    Parameters
    ----------
    theta0_batch : jax.Array or list of jax.Array
        Nonempty ``(K, dim)`` array or list of arrays. JAX tries starts in
        order and stops at the first finite converged result. If none
        converges, it returns the lowest finite objective. Put preferred
        starts first. CPU solves use a Python loop over all starts.
    verbose : int
        0 = silent, >= 1 = print summary.
        Summaries are skipped while tracing.
    """
    t0 = time.perf_counter()
    if backend == "jax" and method in ("newton", "lbfgs", "bfgs"):
        theta, fun, gn, conv, steps, kkt, reason = _multistart_jax(
            f, eta, jnp.asarray(theta0_batch, dtype=jnp.float64),
            bounds, max_steps, tol, grad_fn, hess_fn, method,
        )
        result = BregmanResult(
            theta, fun, gn, steps, conv,
            kkt_residual=kkt, reason=reason,
        )
    else:
        theta0_list = list(theta0_batch) if not isinstance(theta0_batch, list) else theta0_batch
        result = _multistart_loop(
            f, eta, theta0_list,
            backend=backend, method=method, bounds=bounds,
            max_steps=max_steps, tol=tol, grad_fn=grad_fn, hess_fn=hess_fn,
        )
    elapsed = time.perf_counter() - t0
    result = replace(result, elapsed_time=elapsed)
    if verbose >= 1 and not isinstance(result.grad_norm, jax.core.Tracer):
        k = len(theta0_batch) if hasattr(theta0_batch, '__len__') else '?'
        status = "converged" if bool(np.asarray(result.converged)) else "NOT converged"
        print(
            f"Bregman multistart [{backend}/{method}, {k} starts]: "
            f"{status} ({elapsed:.3f}s), |grad|={float(result.grad_norm):.2e}"
        )
    return result


# ---------------------------------------------------------------------------
# Reparameterization  (bounded → unconstrained)
# ---------------------------------------------------------------------------

_NEG_EXP = 0
_POS_EXP = 1
_IDENTITY = 2
_BOUNDED = 3


def _reparam_coeffs(bounds: Tuple[jax.Array, jax.Array]):
    """Bound type, safe lower endpoint, and span for each coordinate."""
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
    return btype, lo_safe, span


def _sigmoid_np(phi: np.ndarray) -> np.ndarray:
    return np.where(
        phi >= 0.0,
        1.0 / (1.0 + np.exp(-phi)),
        np.exp(phi) / (1.0 + np.exp(phi)),
    )


def _theta_map_np(phi, btype, lo, span):
    """θ(φ), θ'(φ), θ''(φ). The map is diagonal.

    Exp, either sign: θ' = θ'' = θ.
    Sigmoid on a finite span: θ' = span·s·(1−s),
    θ'' = span·s·(1−s)·(1−2s).
    Identity: θ' = 1, θ'' = 0.
    """
    phi = np.asarray(phi, dtype=np.float64)
    btype = np.asarray(btype)
    lo = np.asarray(lo, dtype=np.float64)
    span = np.asarray(span, dtype=np.float64)
    s = _sigmoid_np(phi)
    exp_phi = np.exp(phi)
    theta = np.where(
        btype == _NEG_EXP, -exp_phi,
        np.where(
            btype == _POS_EXP, exp_phi,
            np.where(btype == _BOUNDED, lo + span * s, phi),
        ),
    )
    s1 = s * (1.0 - s)
    exp_bound = (btype == _NEG_EXP) | (btype == _POS_EXP)
    d1 = np.where(exp_bound, theta, np.where(btype == _BOUNDED, span * s1, 1.0))
    d2 = np.where(
        exp_bound, theta,
        np.where(btype == _BOUNDED, span * s1 * (1.0 - 2.0 * s), 0.0),
    )
    return theta, d1, d2


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

    btype, lo_safe, span = _reparam_coeffs(bounds)

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
# JAX Newton  (lax.while_loop, vmap-compatible)
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
    _finite, strict, atol_hit = _stop_parts(kkt, near, tol)
    return strict | atol_hit


def _jax_newton_raw(
    f: Callable,
    eta: jax.Array,
    theta0: jax.Array,
    bounds,
    max_steps: int,
    tol: float,
    grad_fn,   # theta → ∇ψ(θ), or None
    hess_fn,   # theta → ∇²ψ(θ), or None
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Gauss–Newton in reparametrised :math:`\phi`-space with early stopping.

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
    (``_kkt_residual``). :math:`\lVert g_\phi\rVert_\infty` is not the
    stop: it can be tiny near a bound while a free coordinate of
    :math:`g_\theta` is still :math:`O(1)`. At a bound optimum the
    multiplier stays in ``grad_norm`` and does not by itself fail the solve.

    A full Gauss–Newton step is taken when it meets the rounding-tolerant
    Armijo test in :func:`_backtrack` at step length 1.
    Otherwise an adaptive Levenberg–Marquardt shift
    :math:`\mu = \lVert g_\phi\rVert_\infty` (multiplied by ten until the
    full step is accepted, for at most eight trials) is added to :math:`H_\phi`.
    If all full steps fail, only the final LM direction is backtracked.
    A permanent ridge
    would swamp the :math:`p`-direction of a concentrated GIG, whose
    :math:`\phi`-gradient is :math:`O(|\theta|)`. :math:`\mu` is unused
    once the undamped step is valid, so the local rate stays quadratic.
    The fallback exists because :math:`J=\mathrm{diag}(\theta)\to 0` on
    an exp bound makes :math:`H_\phi=O(\theta^2)` and the undamped step
    :math:`O(1/\theta)`, which overflows :math:`\exp`; the same long
    step, accepted at a tiny Armijo length, walks the GIG warm start
    into that bound.

    Returns ``(theta_opt, fun, grad_norm, converged, num_steps, kkt, reason)``
    as JAX arrays. The loop stops when the pre-step residual is inside
    ``tol`` or the budget is spent. A converged iterate skips the linear
    solve and line search. ``num_steps`` counts steps taken.
    The check that finds an interior start, and the check after the last
    step, do not increment it. A start already inside ``tol`` reports 0.
    ``converged`` is :func:`_inversion_converged` at the returned point.
    ``kkt`` is the free residual there. ``reason`` is an integer code
    (:data:`_REASON_NAMES`), the same array type eager and traced.
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
        """Try eight full LM steps, then backtrack only the last direction."""
        mu0 = jnp.maximum(jnp.max(jnp.abs(g)), LOG_EPS)

        def cond(state):
            i, _mu, _delta, done = state
            return (~done) & (i < 8)

        def body(state):
            i, mu, delta_lm, done = state
            delta_lm = jnp.linalg.solve(H + mu * eye, g)
            slope_lm = jnp.dot(g, delta_lm)
            done = _armijo_accept(obj(phi - delta_lm), f0, slope_lm) & (slope_lm > 0.0)
            return i + 1, mu * 10.0, delta_lm, done

        _i, _mu, delta_lm, done = jax.lax.while_loop(
            cond, body,
            (jnp.int32(0), mu0, delta, jnp.bool_(False)),
        )
        alpha_lm = jax.lax.cond(
            done,
            lambda _: jnp.array(1.0, dtype=phi.dtype),
            lambda _: _backtrack(obj, phi, delta_lm, f0, jnp.dot(g, delta_lm)),
            operand=None,
        )
        return phi - alpha_lm * delta_lm

    def newton_cond(carry):
        step, _phi, converged = carry
        return (step < max_steps) & ~converged

    def newton_body(carry):
        step, phi, _converged = carry
        g, H, g_theta = get_g_H(phi)
        theta = to_theta(phi)
        pre_ok = _kkt_residual(g_theta, theta, bounds) < tol

        def take_step(_):
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
            return step + jnp.int32(1), phi_new, jnp.bool_(False)

        return jax.lax.cond(
            pre_ok,
            lambda _: (step, phi, jnp.bool_(True)),
            take_step,
            operand=None,
        )

    num_steps, phi_opt, _stopped = jax.lax.while_loop(
        newton_cond, newton_body, (jnp.int32(0), phi0, jnp.bool_(False)),
    )
    theta_opt = to_theta(phi_opt)
    g_final = _natural_grad(f, eta, theta_opt, grad_fn)
    grad_norm = jnp.max(jnp.abs(g_final))
    kkt = _kkt_residual(g_final, theta_opt, bounds)
    near = _kkt_residual(g_final, theta_opt, bounds, gap=KKT_NEAR_GAP)
    converged = _inversion_converged(g_final, theta_opt, bounds, tol)
    reason = _stop_reason(kkt, near, tol)
    final_obj = f(theta_opt) - jnp.dot(theta_opt, eta)
    return theta_opt, final_obj, grad_norm, converged, num_steps, kkt, reason


def _require_solved_theta(
    theta: jax.Array,
    converged: Any,
    *,
    grad_norm: Any = None,
    traced: bool = False,
) -> jax.Array:
    """Return ``theta`` when the Bregman solve converged.

    ``traced`` keeps the caller's trace when the solve inputs were
    stopped for implicit differentiation. An eager call raises
    :class:`SolverDidNotConverge`. A traced call returns NaN, so a
    failed solve is not a model.
    """
    converged_b = jnp.asarray(converged, dtype=jnp.bool_)
    masked = jnp.where(converged_b, theta, jnp.full_like(theta, jnp.nan))
    if traced or isinstance(theta, jax.core.Tracer) or isinstance(converged, jax.core.Tracer):
        return masked
    if bool(converged_b):
        return theta
    detail = ""
    if grad_norm is not None:
        try:
            detail = f" (‖∇f(θ)−η‖∞={float(grad_norm):.3e})"
        except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
            detail = ""
    raise SolverDidNotConverge(
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
    where ``max_steps`` and ``tol`` are static arguments.

    All distribution-level inputs (``f``, ``grad_fn``, ``hess_fn``, ``bounds``)
    are baked into the closure at construction time. Repeated calls with the
    same array shapes and dtypes therefore reuse the compiled XLA executable.

    Use this in EM hot paths where ``solve_bregman`` would otherwise build a
    fresh Python closure on every call and force JAX to re-trace the same
    Newton kernel on each iteration.

    Parameters
    ----------
    f : Callable
        Convex objective ψ(θ) → scalar (must be JAX-traceable).
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
        theta, fun, grad_norm, converged, *_rest = _jax_newton_raw(
            f, eta, theta0, bounds, max_steps, tol, grad_fn, hess_fn,
        )
        return theta, fun, grad_norm, converged

    return solve


@partial(jax.jit, static_argnums=(0, 4, 6, 7, 8), inline=True)
def _multistart_jax(
    f, eta, theta0_batch, bounds, max_steps, tol, grad_fn, hess_fn, method,
):
    """Cached sequential solves, stopping at the first finite convergence."""
    def solve_one(t0):
        if method == "newton":
            return _jax_newton_raw(
                f, eta, t0, bounds, max_steps, tol, grad_fn, hess_fn,
            )
        result = _jax_quasi_newton(f, eta, t0, bounds, max_steps, tol, method)
        return (
            result.theta, result.fun, result.grad_norm, result.converged,
            jnp.asarray(result.num_steps, dtype=jnp.int32),
            result.kkt_residual, result.reason,
        )

    def cond(state):
        i, best = state
        return (i < theta0_batch.shape[0]) & ~(best[3] & jnp.isfinite(best[1]))

    def body(state):
        i, best = state
        result = solve_one(theta0_batch[i])
        take = (i == 0) | (jnp.isfinite(result[1]) & (
            ~jnp.isfinite(best[1]) | result[3] | (result[1] < best[1])
        ))
        best = jax.tree.map(lambda new, old: jnp.where(take, new, old), result, best)
        return i + 1, best

    unsolved = (
        theta0_batch[0],
        jnp.array(jnp.inf, dtype=theta0_batch.dtype),
        jnp.array(jnp.inf, dtype=theta0_batch.dtype),
        jnp.bool_(False),
        jnp.int32(0),
        jnp.array(jnp.inf, dtype=theta0_batch.dtype),
        jnp.int32(_REASON_NONFINITE),
    )
    _, best = jax.lax.while_loop(cond, body, (jnp.int32(0), unsolved))
    return best


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
    final_obj = f(theta_opt) - jnp.dot(theta_opt, eta)

    # jaxopt state.error is ‖∇_φ obj‖, which is tiny on an exp bound
    # while g_θ is still O(1). Acceptance is the natural residual.
    state = result.state
    n_iter = state.iter_num if hasattr(state, "iter_num") else max_steps
    kkt = _kkt_residual(g, theta_opt, bounds)
    near = _kkt_residual(g, theta_opt, bounds, gap=KKT_NEAR_GAP)
    return BregmanResult(
        theta=theta_opt,
        fun=final_obj,
        grad_norm=jnp.max(jnp.abs(g)),
        num_steps=n_iter,
        converged=_inversion_converged(g, theta_opt, bounds, tol),
        kkt_residual=kkt,
        reason=_stop_reason(kkt, near, tol),
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

    BFGS and trust-exact have no box constraints. Bounds use the same
    diagonal φ map as the JAX solver. θ' and θ'' are closed form.
    SciPy ``gtol`` stops on the φ-gradient. ``converged`` is the free
    θ residual. Trust-exact keeps
    :math:`\\sum_i (g_\\theta)_i \\theta_i''(\\phi)` in the Hessian.
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

    use_reparam = bounds is not None and method != "lbfgs"
    map_coeffs = None
    if use_reparam:
        phi0, _, _ = _setup_reparam(jnp.asarray(theta0_np), bounds)
        btype, lo_safe, span = _reparam_coeffs(bounds)
        map_coeffs = (
            np.asarray(btype),
            np.asarray(lo_safe, dtype=np.float64),
            np.asarray(span, dtype=np.float64),
        )

        def _at(phi_np):
            return _theta_map_np(phi_np, *map_coeffs)

        def fun_phi(phi_np):
            theta, _d1, _d2 = _at(phi_np)
            return fun_np(theta)

        def jac_phi(phi_np):
            theta, d1, _d2 = _at(phi_np)
            return d1 * jac_np(theta)

        x0 = np.asarray(phi0, dtype=np.float64)
        objective = fun_phi
        jac_used = jac_phi
    else:
        x0 = theta0_np
        objective = fun_np
        jac_used = jac_np

    kwargs: dict = {
        "jac": jac_used,
        "options": {"maxiter": max_steps, "gtol": tol},
    }
    if method == "lbfgs":
        scipy_bounds = None
        if bounds is not None:
            lo_np = np.asarray(bounds[0], dtype=np.float64)
            hi_np = np.asarray(bounds[1], dtype=np.float64)
            scipy_bounds = list(zip(lo_np, hi_np))
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

        if use_reparam:
            def hess_phi(phi_np):
                theta, d1, d2 = _at(phi_np)
                g_theta = jac_np(theta)
                hess = hess_np(theta)
                sandwiched = d1[:, None] * hess * d1[None, :]
                return sandwiched + np.diag(g_theta * d2)

            kwargs["hess"] = hess_phi
        else:
            kwargs["hess"] = hess_np

    result = minimize(objective, x0, method=scipy_method, **kwargs)
    if use_reparam:
        theta_np, _d1, _d2 = _at(result.x)
        theta_opt = jnp.asarray(theta_np, dtype=jnp.float64)
        jac_val = jac_np(theta_np)
    else:
        theta_opt = jnp.asarray(result.x, dtype=jnp.float64)
        jac_val = result.jac if result.jac is not None else jac_np(result.x)
    g = jnp.asarray(jac_val, dtype=jnp.float64)
    # scipy success is not the natural residual: trust-exact status 2
    # stops just above gtol, and L-BFGS-B can report success with a
    # large bound multiplier. Gate on the free residual.
    kkt = _kkt_residual(g, theta_opt, bounds)
    near = _kkt_residual(g, theta_opt, bounds, gap=KKT_NEAR_GAP)
    return BregmanResult(
        theta=theta_opt,
        fun=float(result.fun),
        grad_norm=float(np.max(np.abs(jac_val))),
        num_steps=int(result.nit),
        converged=_inversion_converged(g, theta_opt, bounds, tol),
        kkt_residual=kkt,
        reason=_stop_reason(kkt, near, tol),
    )


# ---------------------------------------------------------------------------
# Multi-start: Python for-loop (CPU)
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

def _armijo_accept(trial, f0, decrease, c: float = 1e-4):
    r"""Accept finite trials within :math:`8\varepsilon\max(|f_0|,1)` of Armijo.

    Several ulps allow for rounding in objective evaluation. The unit
    scale floor also covers cancellation when the objective is near zero.
    """
    tol_f = 8.0 * jnp.finfo(jnp.asarray(f0).dtype).eps * jnp.maximum(jnp.abs(f0), 1.0)
    return jnp.isfinite(trial) & (trial <= f0 - c * decrease + tol_f)


def _backtrack(obj, phi, delta, f0, slope, beta: float = 0.5, c: float = 1e-4):
    """Rounding-tolerant Armijo backtracking via ``lax.while_loop``.

    Accept finite trials according to :func:`_armijo_accept`.
    """
    def cond(state):
        alpha, _ = state
        trial = obj(phi - alpha * delta)
        insufficient = ~_armijo_accept(trial, f0, alpha * slope, c)
        return insufficient & (alpha > 1e-10)

    def body(state):
        alpha, i = state
        return (alpha * beta, i + 1)

    alpha, _ = jax.lax.while_loop(cond, body, (1.0, 0))
    return alpha
