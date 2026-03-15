"""
JAX-compatible log modified Bessel function of the second kind.

Composite pure-JAX implementation of log_kv(v, z) = log K_v(z):
  Phase 1: Hankel asymptotic (DLMF 10.40.2) for large z
  Phase 2: Numerical quadrature (Takekawa 2022) for moderate z, v
  Phase 3: Olver uniform expansion for large v; small-z series
  Fallback: scipy kve via jax.pure_callback (to be removed after Phase 3)

Custom JVP for exact gradients:
  ∂/∂z: exact recurrence K'_v = -(K_{v-1}+K_{v+1})/2
  ∂/∂v: central FD on log_kv itself (eps=1e-5)
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Hankel asymptotic for large z (DLMF 10.40.2)
# ──────────────────────────────────────────────────────────────────────

_HANKEL_K = 9

def _hankel_large_z(v: jax.Array, z: jax.Array) -> jax.Array:
    """Pure JAX Hankel asymptotic: log K_v(z) for large z.

    K_v(z) ~ sqrt(π/(2z)) e^{-z} Σ_{k=0}^K a_k(v)/z^k
    where a_0 = 1 and a_k = a_{k-1} · (4v²-(2k-1)²) / (8k).

    In log space:
        log K_v(z) = ½ log(π/(2z)) - z + log(Σ_{k=0}^K a_k/z^k)

    Accurate when z > max(20, v²/8).
    """
    v_sq = v * v
    inv_z = 1.0 / z

    a_k = jnp.ones_like(v)
    inv_z_pow = jnp.ones_like(z)
    S = jnp.ones_like(z)

    for k in range(1, _HANKEL_K + 1):
        a_k = a_k * (4.0 * v_sq - (2.0 * k - 1.0) ** 2) / (8.0 * k)
        inv_z_pow = inv_z_pow * inv_z
        S = S + a_k * inv_z_pow

    return 0.5 * jnp.log(jnp.pi / (2.0 * z)) - z + jnp.log(S)


def _hankel_threshold(v: jax.Array) -> jax.Array:
    """Threshold z above which Hankel expansion is accurate."""
    v_abs = jnp.abs(v)
    return jnp.maximum(20.0, v_abs * v_abs / (_HANKEL_K - 1))


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Numerical quadrature (Takekawa 2022)
# K_v(z) = ∫₀^∞ e^{-z cosh t} cosh(vt) dt
# ──────────────────────────────────────────────────────────────────────

_QUAD_N = 128
_gl_nodes_np, _gl_weights_np = np.polynomial.legendre.leggauss(_QUAD_N)
_GL_NODES = jnp.asarray(_gl_nodes_np, dtype=jnp.float64)
_GL_WEIGHTS = jnp.asarray(_gl_weights_np, dtype=jnp.float64)


def _log_cosh(x: jax.Array) -> jax.Array:
    """Numerically stable log(cosh(x))."""
    ax = jnp.abs(x)
    return jnp.where(ax > 20.0, ax - jnp.log(2.0), jnp.log(jnp.cosh(x)))


def _quadrature_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """Pure JAX quadrature: log K_v(z) via integral representation.

    K_v(z) = ∫₀^∞ e^{-z cosh t} cosh(vt) dt

    Uses Gauss-Legendre quadrature on [0, b] with log-sum-exp for stability.
    Fully differentiable in both v and z.
    """
    v_abs = jnp.abs(v)

    b = jnp.acosh(jnp.maximum(700.0 / jnp.maximum(z, 1e-300), 1.0 + 1e-10))
    b = jnp.clip(b, 1.0, 200.0)

    b_e = b[..., jnp.newaxis]
    z_e = z[..., jnp.newaxis]
    v_e = v_abs[..., jnp.newaxis]

    t = 0.5 * b_e * (_GL_NODES + 1.0)
    w = 0.5 * b_e * _GL_WEIGHTS

    g = -z_e * jnp.cosh(t) + _log_cosh(v_e * t)

    g_max = jnp.max(g, axis=-1, keepdims=True)
    log_integral = g_max[..., 0] + jnp.log(
        jnp.sum(w * jnp.exp(g - g_max), axis=-1)
    )

    return log_integral


# ──────────────────────────────────────────────────────────────────────
# Scipy callback fallback (to be removed after Phase 3)
# ──────────────────────────────────────────────────────────────────────

def _log_kv_numpy_vec(v_arr: np.ndarray, z_arr: np.ndarray) -> np.ndarray:
    """Vectorized log K_v(z) via scipy, with small-z asymptotic fallback."""
    from scipy.special import kve as _kve, gammaln as _gammaln

    v_arr = np.asarray(v_arr, dtype=np.float64)
    z_arr = np.asarray(z_arr, dtype=np.float64)
    shape = np.broadcast_shapes(v_arr.shape, z_arr.shape)
    flat_v = np.broadcast_to(v_arr, shape).ravel()
    flat_z = np.maximum(np.broadcast_to(z_arr, shape).ravel(), np.finfo(np.float64).tiny)
    v_abs = np.abs(flat_v)
    vals = np.log(_kve(v_abs, flat_z)) - flat_z

    inf_mask = ~np.isfinite(vals)
    if inf_mask.any():
        vv = v_abs[inf_mask]
        zz = flat_z[inf_mask]
        large = vv > 1e-10
        out = np.empty_like(vv)
        if large.any():
            out[large] = (
                _gammaln(vv[large])
                - np.log(2.0)
                + vv[large] * (np.log(2.0) - np.log(zz[large]))
            )
        if (~large).any():
            inner = np.maximum(
                -np.log(zz[~large] / 2.0) - np.euler_gamma, np.finfo(np.float64).tiny
            )
            out[~large] = np.log(inner)
        vals[inf_mask] = out
    return vals.reshape(shape)


# ──────────────────────────────────────────────────────────────────────
# Public API: log_kv with custom JVP
# ──────────────────────────────────────────────────────────────────────

@functools.partial(jax.custom_jvp, nondiff_argnums=())
def log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """
    log K_v(z), JAX-compatible with custom JVP for full autodiff.

    Uses a composite strategy:
      - Hankel asymptotic (pure JAX) when z > max(20, v²/8)
      - Numerical quadrature (pure JAX) otherwise

    Parameters
    ----------
    v : scalar or array — order (any real number)
    z : scalar or array — argument (must be > 0), same shape as v

    Returns
    -------
    log_kv : same shape as v and z
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(v.shape, z.shape)
    v = jnp.broadcast_to(v, shape)
    z = jnp.broadcast_to(z, shape)
    z = jnp.maximum(z, jnp.finfo(jnp.float64).tiny)

    use_hankel = z > _hankel_threshold(v)

    z_safe_hankel = jnp.maximum(z, 20.0)
    result_hankel = _hankel_large_z(v, z_safe_hankel)

    z_safe_quad = jnp.clip(z, 1e-300, 1e4)
    result_quad = _quadrature_log_kv(v, z_safe_quad)

    return jnp.where(use_hankel, result_hankel, result_quad)


@log_kv.defjvp
def _log_kv_jvp(primals, tangents):
    v, z = primals
    dv, dz = tangents
    primal_out = log_kv(v, z)

    log_kvm1 = log_kv(v - 1.0, z)
    log_kvp1 = log_kv(v + 1.0, z)
    dlogkv_dz = -0.5 * (jnp.exp(log_kvm1 - primal_out) + jnp.exp(log_kvp1 - primal_out))

    _EPS_V = jnp.asarray(1e-5, dtype=jnp.float64)
    log_kv_vp = log_kv(v + _EPS_V, z)
    log_kv_vm = log_kv(v - _EPS_V, z)
    dlogkv_dv = (log_kv_vp - log_kv_vm) / (2.0 * _EPS_V)

    tangent_out = dlogkv_dz * dz + dlogkv_dv * dv
    return primal_out, tangent_out
