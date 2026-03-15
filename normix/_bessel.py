"""
JAX-compatible log modified Bessel function of the second kind.

Pure-JAX composite implementation of log_kv(v, z) = log K_v(z):
  Phase 1: Hankel asymptotic (DLMF 10.40.2) for large z
  Phase 2: Numerical quadrature (Takekawa 2022) for moderate z, v
  Phase 3: Olver uniform expansion (DLMF 10.41.4) for large v

Zero scipy callbacks — fully JIT-compilable and GPU-native.

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

_HANKEL_K = 20

def _hankel_large_z(v: jax.Array, z: jax.Array) -> jax.Array:
    """Pure JAX Hankel asymptotic: log K_v(z) for large z.

    K_v(z) ~ sqrt(π/(2z)) e^{-z} Σ_{k=0}^K a_k(v)/z^k
    where a_0 = 1 and a_k = a_{k-1} · (4v²-(2k-1)²) / (8k).

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
# Phase 3: Olver uniform expansion for large v (DLMF 10.41.4)
# K_v(vw) ~ sqrt(π/(2v)) · e^{-vη(w)} / (1+w²)^{1/4} · Σ u_k(p)/v^k
# ──────────────────────────────────────────────────────────────────────

_OLVER_V_THRESH = 45.0

def _olver_large_v(v: jax.Array, z: jax.Array) -> jax.Array:
    """Pure JAX Olver uniform asymptotic: log K_v(z) for large |v|.

    DLMF 10.41.4 with 6 terms (u_0 through u_5).
    Accurate when |v| > 50 across all z > 0.
    """
    v_abs = jnp.abs(v)
    w = z / v_abs
    w_sq = w * w
    sqrt_1pw2 = jnp.sqrt(1.0 + w_sq)
    p = 1.0 / sqrt_1pw2

    eta = sqrt_1pw2 + jnp.log(w / (1.0 + sqrt_1pw2))

    p2 = p * p
    p3 = p2 * p
    p4 = p3 * p
    p5 = p4 * p
    p6 = p5 * p
    p7 = p6 * p
    p8 = p7 * p
    p9 = p8 * p
    p10 = p9 * p
    p11 = p10 * p
    p12 = p11 * p
    p13 = p12 * p
    p15 = p13 * p2

    u0 = jnp.ones_like(p)
    u1 = (3.0 * p - 5.0 * p3) / 24.0
    u2 = (81.0 * p2 - 462.0 * p4 + 385.0 * p6) / 1152.0
    u3 = (30375.0 * p3 - 369603.0 * p5 + 765765.0 * p7
           - 425425.0 * p9) / 414720.0
    u4 = (4465125.0 * p4 - 94121676.0 * p6 + 349922430.0 * p8
           - 446185740.0 * p10 + 185910725.0 * p12) / 39813120.0
    u5 = (1519517625.0 * p5 - 49286948607.0 * p7
           + 284499769554.0 * p9 - 614135872350.0 * p11
           + 566098157625.0 * p13
           - 188699385875.0 * p15) / 6688604160.0

    inv_v = 1.0 / v_abs
    S = (u0 - u1 * inv_v + u2 * inv_v**2 - u3 * inv_v**3
         + u4 * inv_v**4 - u5 * inv_v**5)

    return (0.5 * jnp.log(jnp.pi / (2.0 * v_abs))
            - v_abs * eta
            - 0.25 * jnp.log(1.0 + w_sq)
            + jnp.log(S))


# ──────────────────────────────────────────────────────────────────────
# Public API: log_kv with custom JVP
# ──────────────────────────────────────────────────────────────────────

@functools.partial(jax.custom_jvp, nondiff_argnums=())
def log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """
    log K_v(z), pure JAX with custom JVP for full autodiff.

    Uses a composite strategy (zero scipy callbacks):
      1. Hankel asymptotic for z > max(20, v²/8)
      2. Olver uniform expansion for |v| > 50
      3. Numerical quadrature for everything else

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

    v_abs = jnp.abs(v)
    use_hankel = z > _hankel_threshold(v)
    use_olver = (~use_hankel) & (v_abs > _OLVER_V_THRESH)

    z_safe_hankel = jnp.maximum(z, 20.0)
    result_hankel = _hankel_large_z(v, z_safe_hankel)

    v_safe_olver = jnp.where(v_abs > 1.0, v, jnp.ones_like(v) * (_OLVER_V_THRESH + 1.0))
    z_safe_olver = jnp.maximum(z, 1e-300)
    result_olver = _olver_large_v(v_safe_olver, z_safe_olver)

    z_safe_quad = jnp.clip(z, 1e-300, 1e4)
    result_quad = _quadrature_log_kv(v, z_safe_quad)

    result = jnp.where(use_hankel, result_hankel,
             jnp.where(use_olver, result_olver, result_quad))

    return result


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
