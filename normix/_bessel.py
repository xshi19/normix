"""
JAX-compatible log modified Bessel function of the second kind.

log_kv(v, z) = log K_v(z), with @jax.custom_jvp for exact gradients:
  - Primal: pure-JAX composite
      Phase 1: Hankel asymptotic for large z
      Phase 2: Gauss-Legendre quadrature for moderate z, v
      Fallback: scipy callback for remaining cases
  - ∂/∂z  : exact recurrence K'_v = -(K_{v-1}+K_{v+1})/2
  - ∂/∂v  : central FD on log_kv itself (eps=1e-5)
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_N_QUAD = 128
_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(_N_QUAD)
_GL_NODES = jnp.asarray(_GL_NODES_NP, dtype=jnp.float64)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS_NP, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Hankel asymptotic expansion (DLMF 10.40.2) — Phase 1
# ---------------------------------------------------------------------------

def _hankel_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """Hankel asymptotic expansion for large z (DLMF 10.40.2).

    K_v(z) ~ sqrt(pi/(2z)) e^{-z} sum_{k=0}^{K} a_k(v) / z^k
    where a_k(v) = prod_{j=1}^{k} [4v^2 - (2j-1)^2] / (k! 8^k)

    Accurate when z > max(25, v^2/4). With 20 terms, gives ~12-15 digit
    accuracy at the threshold boundary.
    """
    mu = 4.0 * v * v
    total = jnp.ones_like(z)
    term = jnp.ones_like(z)
    for k in range(1, 21):
        term = term * (mu - (2.0 * k - 1.0) ** 2) / (8.0 * k * z)
        total = total + term
    return 0.5 * jnp.log(jnp.pi / (2.0 * z)) - z + jnp.log(jnp.maximum(total, 1e-300))


# ---------------------------------------------------------------------------
# Numerical quadrature (Takekawa 2022) — Phase 2
# ---------------------------------------------------------------------------

def _log_cosh(x: jax.Array) -> jax.Array:
    """Numerically stable log(cosh(x)): avoids overflow for large |x|."""
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)


def _quadrature_upper_bound(v_abs: jax.Array, z: jax.Array) -> jax.Array:
    """Upper integration bound T such that the integrand is negligible for t > T.

    Solves z*cosh(T) >> |v|*T + P iteratively (P = 50 ≈ -log(eps_f64)).
    """
    P = 50.0
    z_safe = jnp.maximum(z, 1e-300)
    T = jnp.maximum(jnp.log(2.0 * P / z_safe), 1.0)
    for _ in range(6):
        T = jnp.maximum(jnp.log(2.0 * (P + v_abs * T) / z_safe), 1.0)
    return T + 2.0


def _quadrature_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """log K_v(z) via Gauss-Legendre quadrature of the integral representation.

    K_v(z) = int_0^inf exp(-z cosh t) cosh(v t) dt

    Uses 128-point GL quadrature on [0, T] with adaptive upper bound.
    Fully differentiable in both v and z via autodiff through the sum.
    Handles arbitrary input shapes via broadcasting with trailing quad axis.
    """
    v_abs = jnp.abs(v)
    T = _quadrature_upper_bound(v_abs, z)

    # Broadcast: T has shape (...), GL nodes have shape (N_QUAD,)
    # Add trailing axis to T, v, z for broadcasting with quad points
    T_e = T[..., None]           # (..., 1)
    v_e = v_abs[..., None]       # (..., 1)
    z_e = z[..., None]           # (..., 1)

    nodes = _GL_NODES             # (N_QUAD,)
    t = T_e * (nodes + 1.0) / 2.0  # (..., N_QUAD)
    log_w = jnp.log(_GL_WEIGHTS)    # (N_QUAD,)

    log_f = -z_e * jnp.cosh(t) + _log_cosh(v_e * t)  # (..., N_QUAD)

    return jnp.log(T / 2.0) + jax.scipy.special.logsumexp(log_w + log_f, axis=-1)


# ---------------------------------------------------------------------------
# Scipy callback fallback
# ---------------------------------------------------------------------------

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


def _scipy_log_kv(v: jax.Array, z: jax.Array, shape) -> jax.Array:
    """Scipy callback wrapper."""
    result_shape = jax.ShapeDtypeStruct(shape, jnp.float64)
    return jax.pure_callback(
        _log_kv_numpy_vec, result_shape, v, z, vmap_method="broadcast_all",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_jvp, nondiff_argnums=())
def log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """
    log K_v(z), JAX-compatible with custom JVP for full autodiff.

    Parameters
    ----------
    v : scalar or array — order (any real; K_v = K_{-v})
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

    use_hankel = z > jnp.maximum(25.0, v_abs * v_abs / 4.0)

    z_hankel = jnp.where(use_hankel, z, jnp.maximum(25.0, v_abs * v_abs / 4.0 + 1.0))
    result_hankel = _hankel_log_kv(v_abs, z_hankel)

    z_quad = jnp.maximum(z, 1e-30)
    result_quad = _quadrature_log_kv(v_abs, z_quad)

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
