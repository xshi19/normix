"""
Log modified Bessel function of the second kind.

One centered whole-line Gauss–Legendre kernel (S10) implements both
``log_kv`` and :func:`log_kv_moments`. Geometry is frozen under
``stop_gradient``; autodiff of ``log_kv`` yields cumulants of that
discrete measure. No regimes, no ``lax.cond``, no ``custom_jvp``, no
finite differences.

backend='jax'  (default): JAX arrays, JIT-able, differentiable.
backend='cpu'           : the same sums in NumPy.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from normix.utils.constants import (
    BESSEL_QUAD_NODES, BESSEL_QUAD_LOG_DROP, BESSEL_WINDOW_TILT,
    BESSEL_WINDOW_ITERS, BESSEL_WINDOW_HI_MAX, BESSEL_PANEL_FLOOR,
)

# ---------------------------------------------------------------------------
# Moment-quadrature kernel (S10) — one kernel, two array backends
# ---------------------------------------------------------------------------
#
# 2 K_ν(z) = ∫_R exp(ν u − z cosh u) du.  Mode u0 = asinh(ν/z),
# κ = √(ν²+z²), x = u − u0.  Nodes/weights/window are frozen under
# stop_gradient; the log-sum-exp identity is then exact for that geometry,
# so jax.grad / jax.hessian of log_kv are cumulants of the discrete measure.
#
# The centered exponent is evaluated cancellation-free:
#   g_c(x) = ν x − [((κ+ν)/2) expm1(x) + ((κ−ν)/2) expm1(−x)],
#   κ−ν = z²/(κ+ν)  (ν ≥ 0; mirror for ν < 0).
# C's grouping κ·2sinh²(x/2)+ν sinh x cancels in the left tail when κ≈ν
# and is not used.

if BESSEL_QUAD_NODES % 2 != 0:
    raise ValueError("BESSEL_QUAD_NODES must be even (two equal GL panels)")

_N_HALF = BESSEL_QUAD_NODES // 2
_GL_T_NP, _GL_W_NP = np.polynomial.legendre.leggauss(_N_HALF)
_GL_T_NP = np.asarray(_GL_T_NP, dtype=np.float64)
_GL_LOGW_NP = np.log(np.asarray(_GL_W_NP, dtype=np.float64))
_GL_T_J = jnp.asarray(_GL_T_NP)
_GL_LOGW_J = jnp.asarray(_GL_LOGW_NP)
_N_DOUBLE = 8
_LOG_TWO = np.log(2.0)


class BesselMoments(eqx.Module):
    r"""Frozen quadrature moments of :math:`(x, e^{-x}-1, e^{x}-1)`.

    The discrete exponential family lives on the centered coordinate
    :math:`x = u - u_0` with :math:`u_0 = \operatorname{asinh}(\nu/z)`.
    Geometry is constant under AD (see :func:`log_kv_moments`); ``mean``
    and ``cov`` are the mean and centered Gram of
    :math:`s = (x, \operatorname{expm1}(-x), \operatorname{expm1}(x))`.
    ``cov`` is PSD to rounding (relative-ε entries). Argument-jet views
    are projections of
    :math:`w = 2\sinh(u_0 + x/2)\sinh(x/2)`, not reconstructions from
    ``cov``.

    Parameters
    ----------
    log_k : jax.Array
        :math:`\log K_\nu(z)`, broadcast shape of ``(v, z)``.
    u0 : jax.Array
        Mode :math:`\operatorname{asinh}(\nu/z)`.
    mean : jax.Array
        :math:`E[s]`, shape ``(..., 3)``.
    cov : jax.Array
        :math:`\mathrm{Cov}[s]`, shape ``(..., 3, 3)``.
    d_arg : jax.Array
        :math:`\partial_z \log K_\nu = -(\cosh u_0 + E[w])`.
    d2_arg : jax.Array
        :math:`\partial_{zz} \log K_\nu = \mathrm{Var}(w)`.
    d2_order_arg : jax.Array
        :math:`\partial_{\nu z} \log K_\nu = -\mathrm{Cov}(x, w)`.
    """

    log_k: Array
    u0: Array
    mean: Array
    cov: Array
    d_arg: Array
    d2_arg: Array
    d2_order_arg: Array

    @property
    def d_order(self) -> Array:
        r""":math:`\partial_\nu \log K_\nu = E[u] = u_0 + E[x]`."""
        return self.u0 + self.mean[..., 0]

    @property
    def d2_order(self) -> Array:
        r""":math:`\partial_{\nu\nu} \log K_\nu = \mathrm{Var}(u) > 0`."""
        return self.cov[..., 0, 0]


def _is_jax(xp) -> bool:
    return xp is jnp


def _stop_grad(x, xp):
    if _is_jax(xp):
        return jax.lax.stop_gradient(x)
    return x


def _gl_nodes(xp, n_nodes: int):
    if n_nodes == BESSEL_QUAD_NODES:
        if _is_jax(xp):
            return _GL_T_J, _GL_LOGW_J
        return _GL_T_NP, _GL_LOGW_NP
    n_half = n_nodes // 2
    t, w = np.polynomial.legendre.leggauss(n_half)
    t = np.asarray(t, dtype=np.float64)
    logw = np.log(np.asarray(w, dtype=np.float64))
    if _is_jax(xp):
        return jnp.asarray(t), jnp.asarray(logw)
    return t, logw


def _logsumexp(a, xp, axis: int = -1, keepdims: bool = False):
    if _is_jax(xp):
        return jax.scipy.special.logsumexp(a, axis=axis, keepdims=keepdims)
    from scipy.special import logsumexp as np_logsumexp
    return np_logsumexp(a, axis=axis, keepdims=keepdims)


def _fori(n: int, body, init, xp):
    """``body(carry) -> carry``, unrolled in numpy and ``fori_loop`` in JAX."""
    if _is_jax(xp):
        return jax.lax.fori_loop(0, n, lambda _i, c: body(c), init)
    carry = init
    for _ in range(n):
        carry = body(carry)
    return carry


def _as_arrays(v, z, xp):
    v = xp.asarray(v, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)
    if _is_jax(xp):
        shape = jnp.broadcast_shapes(v.shape, z.shape)
    else:
        shape = np.broadcast_shapes(np.shape(v), np.shape(z))
    v = xp.broadcast_to(v, shape)
    z = xp.broadcast_to(z, shape)
    tiny = xp.asarray(np.finfo(np.float64).tiny, dtype=xp.float64)
    z = xp.maximum(z, tiny)
    return v, z


def _ab_coeffs(v, z, xp):
    r"""Stable :math:`(\kappa, (\kappa+\nu)/2, (\kappa-\nu)/2)` with
    :math:`\kappa-\nu = z(z/(\kappa+\lvert\nu\rvert))` (mirrored for
    :math:`\nu<0`). Grouping avoids overflow of :math:`z^2` for
    :math:`z\gtrsim 10^{155}`."""
    kappa = xp.hypot(v, z)
    va = xp.abs(v)
    tiny = xp.asarray(np.finfo(np.float64).tiny, dtype=xp.float64)
    kp = kappa + va
    km = z * (z / xp.maximum(kp, tiny))
    pos = v >= 0
    a = xp.where(pos, 0.5 * kp, 0.5 * km)
    b = xp.where(pos, 0.5 * km, 0.5 * kp)
    return kappa, a, b


def _g_k(x, v, a, b, k, xp):
    r""":math:`g_c(x) + k x` for a Python integer tilt ``k``."""
    # Clip so window-search doubling cannot overflow expm1; at |x|=700 the
    # exponent is already far below −P.
    x = xp.clip(x, -700.0, 700.0)
    combo = a * xp.expm1(x) + b * xp.expm1(-x)
    return (v + k) * x - combo


def _edge(v, a, b, side: float, k: int, P: float, xp, n_iters: int):
    r"""Outer :math:`s>0` such that :math:`g_k(\mathrm{side}\cdot s) \le -P`."""
    hi_max = xp.asarray(BESSEL_WINDOW_HI_MAX, dtype=xp.float64)
    drop = xp.asarray(-P, dtype=xp.float64)
    tiny = xp.asarray(np.finfo(np.float64).tiny, dtype=xp.float64)
    # Right tail ~ a e^{x}, left ~ b e^{-x}.
    coeff = a if side > 0.0 else b
    hi = xp.minimum(
        xp.maximum(xp.log1p(P / xp.maximum(coeff, tiny)), 1.0),
        hi_max,
    )

    def double_step(hi):
        gk = _g_k(side * hi, v, a, b, k, xp)
        return xp.minimum(xp.where(gk > drop, hi * 2.0, hi), hi_max)

    hi = _fori(_N_DOUBLE, double_step, hi, xp)
    lo = xp.zeros_like(hi)

    def bisect_step(carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        gk = _g_k(side * mid, v, a, b, k, xp)
        wider = gk > drop
        return xp.where(wider, mid, lo), xp.where(wider, hi, mid)

    _, hi = _fori(n_iters, bisect_step, (lo, hi), xp)
    return hi


def _geometry(v, z, xp, *, log_drop: float, tilt: int, n_iters: int):
    r"""Mode, :math:`\kappa`, and the tilt-covering window in :math:`x`."""
    kappa, a, b = _ab_coeffs(v, z, xp)
    u0 = xp.arcsinh(v / z)
    P = float(log_drop)
    x_hi = xp.zeros_like(v)
    x_lo = xp.zeros_like(v)
    for k in range(-int(tilt), int(tilt) + 1):
        x_hi = xp.maximum(x_hi, _edge(v, a, b, 1.0, k, P, xp, n_iters))
        x_lo = xp.minimum(x_lo, -_edge(v, a, b, -1.0, k, P, xp, n_iters))
    floor = xp.asarray(BESSEL_PANEL_FLOOR, dtype=xp.float64)
    x_lo = xp.minimum(x_lo, -floor)
    x_hi = xp.maximum(x_hi, floor)
    return u0, kappa, a, b, x_lo, x_hi


def _nodes_and_logw(x_lo, x_hi, xp, n_nodes: int):
    r"""Two GL panels sharing the mode: :math:`[x_\mathrm{lo},0]\cup[0,x_\mathrm{hi}]`."""
    t, logw = _gl_nodes(xp, n_nodes)
    jac_l = 0.5 * (-x_lo)
    jac_r = 0.5 * x_hi
    x_l = jac_l[..., None] * t + 0.5 * x_lo[..., None]
    x_r = jac_r[..., None] * t + 0.5 * x_hi[..., None]
    logw_l = logw + xp.log(jac_l)[..., None]
    logw_r = logw + xp.log(jac_r)[..., None]
    x = xp.concatenate([x_l, x_r], axis=-1)
    log_w = xp.concatenate([logw_l, logw_r], axis=-1)
    return x, log_w


def _moments_from_weights(x, log_w, u0, xp):
    r"""Cumulants of :math:`s=(x,\mathrm{expm1}(-x),\mathrm{expm1}(x))`.

    ``cov`` is the centered Gram :math:`R^\top R` with
    :math:`R_i=\sqrt{p_i}\,(s_i-\bar s)`. Argument projections use

    .. math::

        w = 2\sinh(u_0 + x/2)\sinh(x/2) = \cosh(u_0+x)-\cosh u_0,

    so :math:`\mathrm{Var}(w)` retains the :math:`O(1/z^2)` even component
    that reconstructing ``cov`` from raw moments cancels. The grouping
    :math:`\cosh u_0\cdot 2\sinh^2(x/2)+\sinh u_0\sinh x` is not used:
    it cancels in the left tail when :math:`|u_0|` is large.
    """
    log_n = _logsumexp(log_w, xp, axis=-1, keepdims=True)
    p = xp.exp(log_w - log_n)
    s = xp.stack([x, xp.expm1(-x), xp.expm1(x)], axis=-1)
    mean = xp.sum(p[..., None] * s, axis=-2)
    r = xp.sqrt(p)[..., None] * (s - mean[..., None, :])
    cov = xp.matmul(xp.swapaxes(r, -1, -2), r)
    cov = 0.5 * (cov + xp.swapaxes(cov, -1, -2))

    half = 0.5 * x
    w = 2.0 * xp.sinh(u0[..., None] + half) * xp.sinh(half)
    ew = xp.sum(p * w, axis=-1)
    dw = w - ew[..., None]
    dx = x - mean[..., 0][..., None]
    d_arg = -(xp.cosh(u0) + ew)
    d2_arg = xp.sum(p * dw * dw, axis=-1)
    d2_order_arg = -xp.sum(p * dx * dw, axis=-1)
    return mean, cov, d_arg, d2_arg, d2_order_arg


def _eval_quad(
    v, z, xp, *,
    n_nodes: int = BESSEL_QUAD_NODES,
    log_drop: float = BESSEL_QUAD_LOG_DROP,
    tilt: int = BESSEL_WINDOW_TILT,
    n_iters: int = BESSEL_WINDOW_ITERS,
):
    """Live log-weights on frozen geometry. Returns ``(log_k, u0, x, log_f)``."""
    v, z = _as_arrays(v, z, xp)
    v0 = _stop_grad(v, xp)
    z0 = _stop_grad(z, xp)
    u0, kappa0, a0, b0, x_lo, x_hi = _geometry(
        v0, z0, xp, log_drop=log_drop, tilt=tilt, n_iters=n_iters,
    )
    x, log_w = _nodes_and_logw(x_lo, x_hi, xp, n_nodes)
    u0 = _stop_grad(u0, xp)
    kappa0 = _stop_grad(kappa0, xp)
    a0 = _stop_grad(a0, xp)
    b0 = _stop_grad(b0, xp)
    x = _stop_grad(x, xp)
    log_w = _stop_grad(log_w, xp)
    combo = a0[..., None] * xp.expm1(x) + b0[..., None] * xp.expm1(-x)
    scale = z / z0
    log_f = log_w + v[..., None] * x - scale[..., None] * combo
    log_n = _logsumexp(log_f, xp, axis=-1)
    log_two = xp.asarray(_LOG_TWO, dtype=xp.float64)
    log_k = v * u0 - scale * kappa0 + log_n - log_two
    return log_k, u0, x, log_f


def _moments_xp(v, z, xp, **cfg) -> BesselMoments:
    log_k, u0, x, log_f = _eval_quad(v, z, xp, **cfg)
    mean, cov, d_arg, d2_arg, d2_order_arg = _moments_from_weights(
        x, log_f, u0, xp,
    )
    return BesselMoments(
        log_k=log_k, u0=u0, mean=mean, cov=cov,
        d_arg=d_arg, d2_arg=d2_arg, d2_order_arg=d2_order_arg,
    )


def _log_kv_quad_jax(v, z) -> Array:
    return _eval_quad(v, z, jnp)[0]


def _log_kv_quad_cpu(v, z):
    return _eval_quad(v, z, np)[0]


def _moments_jax(v, z) -> BesselMoments:
    return _moments_xp(v, z, jnp)


def _moments_cpu(v, z) -> BesselMoments:
    return _moments_xp(v, z, np)


_BACKENDS = {
    'jax': (_log_kv_quad_jax, _moments_jax),
    'cpu': (_log_kv_quad_cpu, _moments_cpu),
}


def log_kv_moments(v, z, backend: str = 'jax') -> BesselMoments:
    r"""Log :math:`K_\nu(z)` and the moment bundle of the quadrature measure.

    One centered whole-line Gauss–Legendre kernel; ``backend`` selects the
    array library (JAX or NumPy), not a different approximation. Geometry
    is frozen under :func:`jax.lax.stop_gradient`, so the log-sum-exp
    identity is exact for any fixed nodes and

    .. math::

        \partial_\nu \log K = E[u], \qquad
        \partial_{\nu\nu} \log K = \mathrm{Var}(u) > 0.

    Stated numerical domain: :math:`z \ge 10^{-10}` (GIG never asks below
    ``GIG_DEGEN_THRESHOLD``), :math:`|\nu| \le 300` tested. The point
    :math:`(\nu,z)=(0, 10^{-300})` is unresolved at 192 nodes.

    Parameters
    ----------
    v : scalar or array
        Order (any real; :math:`\nu=0` is an ordinary point).
    z : scalar or array
        Argument (must be :math:`> 0`).
    backend : {'jax', 'cpu'}, optional
        Array backend. ``'jax'`` is JIT-able and differentiable; ``'cpu'``
        is the same sums in NumPy.

    Returns
    -------
    BesselMoments
        ``log_k``, ``u0``, ``mean`` (shape ``(..., 3)``), ``cov``
        (shape ``(..., 3, 3)``). Jet views: ``d_order``, ``d_arg``,
        ``d2_order``, ``d2_order_arg``, ``d2_arg``. Argument jets
        (``d_arg``, ``d2_arg``, ``d2_order_arg``) are stored projections
        of :math:`w=2\sinh(u_0+x/2)\sinh(x/2)`, not affine images of
        ``cov``.

    Notes
    -----
    ``jax.grad(lambda z: log_kv(v, z))(z)`` equals ``d_arg``;
    ``jax.grad`` with respect to ``v`` equals ``d_order``. Both are
    derivatives of the frozen quadrature, not finite differences. GIG
    uses this bundle so :math:`\eta` and the Fisher come from one pass.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"backend must be one of {list(_BACKENDS)}, got {backend!r}"
        )
    return _BACKENDS[backend][1](v, z)


def log_kv(v, z, backend: str = 'jax') -> jax.Array:
    r"""
    :math:`\log K_v(z)` — log modified Bessel function of the second kind.

    Parameters
    ----------
    v : scalar or array
        Order (any real; :math:`K_v = K_{-v}`; :math:`\nu=0` is ordinary).
    z : scalar or array
        Argument (must be :math:`> 0`). Stated domain :math:`z \ge 10^{-10}`.
    backend : str, optional
        ``'jax'`` (default) or ``'cpu'``. Both use the same quadrature;
        ``'jax'`` is JIT-able and differentiable.

    Returns
    -------
    jax.Array
        Same broadcast shape as ``(v, z)``.

    Notes
    -----
    A 192-point Gauss–Legendre sum of the whole-line integrand for
    :math:`K_\nu`. ``jax.grad`` with respect to ``z`` equals
    :func:`log_kv_moments` ``d_arg``; with respect to ``v`` it equals
    ``d_order``. Neither is a finite difference.

    Examples
    --------
    Evaluate at a single point (JAX backend, JIT-able):

    >>> import jax.numpy as jnp
    >>> from normix import log_kv
    >>> float(log_kv(v=0.5, z=1.0))        # doctest: +ELLIPSIS
    -0.774...

    CPU backend (same kernel in NumPy):

    >>> float(log_kv(v=0.5, z=1.0, backend='cpu'))  # doctest: +ELLIPSIS
    -0.774...

    Symmetry :math:`K_v(z) = K_{-v}(z)`:

    >>> abs(float(log_kv(0.5, 2.0)) - float(log_kv(-0.5, 2.0))) < 1e-10
    True

    Differentiable via JAX:

    >>> import jax
    >>> dlogkv_dz = jax.grad(lambda z: log_kv(0.5, z))(jnp.array(1.0))
    >>> float(dlogkv_dz) < 0   # K_v decreases with z
    True
    """
    if backend == 'cpu':
        return _BACKENDS['cpu'][0](v, z)
    return _BACKENDS['jax'][0](v, z)
