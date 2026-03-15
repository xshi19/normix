"""
JAX-compatible log modified Bessel function of the second kind.

log_kv(v, z) = log K_v(z), fully pure-JAX with zero scipy callbacks.

Composite strategy with regime-specific methods:
  1. Hankel asymptotic (DLMF 10.40.2) for large z
  2. Gauss-Legendre quadrature (Takekawa 2022) for moderate z, v
  3. Olver uniform expansion (DLMF 10.41.4) for large v
  4. Small-z leading asymptotic (DLMF 10.30.2) for z → 0

Custom JVP for full autodiff:
  - ∂/∂z : exact recurrence K'_v = -(K_{v-1}+K_{v+1})/2
  - ∂/∂v : central FD on log_kv itself (eps=1e-5)
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
# Phase 1: Hankel asymptotic expansion (DLMF 10.40.2)
# ---------------------------------------------------------------------------

def _hankel_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """Hankel asymptotic expansion for large z.

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
# Phase 2: Numerical quadrature (Takekawa 2022)
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
    """
    v_abs = jnp.abs(v)
    T = _quadrature_upper_bound(v_abs, z)

    T_e = T[..., None]
    v_e = v_abs[..., None]
    z_e = z[..., None]

    t = T_e * (_GL_NODES + 1.0) / 2.0   # (..., N_QUAD)
    log_w = jnp.log(_GL_WEIGHTS)          # (N_QUAD,)

    log_f = -z_e * jnp.cosh(t) + _log_cosh(v_e * t)

    return jnp.log(T / 2.0) + jax.scipy.special.logsumexp(log_w + log_f, axis=-1)


# ---------------------------------------------------------------------------
# Phase 3: Olver uniform asymptotic expansion (DLMF 10.41.3-4)
# ---------------------------------------------------------------------------

def _olver_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """Olver uniform asymptotic expansion for large v.

    K_v(z) = K_v(v * zeta) where zeta = z/v, and:
    K_v(v*zeta) ~ sqrt(pi/(2v)) * exp(-v*eta) / (1+zeta^2)^{1/4} * S

    eta(zeta) = sqrt(1+zeta^2) + ln(zeta / (1+sqrt(1+zeta^2)))
    t = 1/sqrt(1+zeta^2)
    S = sum_{k=0}^{K} (-1)^k u_k(t) / v^k

    Debye polynomials u_k(t) from DLMF 10.41.6-7.
    Accurate for v > 25 with 5 terms.
    """
    zeta = z / v
    zeta2 = zeta * zeta
    sqrt_1pz2 = jnp.sqrt(1.0 + zeta2)
    t = 1.0 / sqrt_1pz2

    # eta = sqrt(1+zeta^2) + ln(zeta / (1 + sqrt(1+zeta^2)))
    eta = sqrt_1pz2 + jnp.log(zeta / (1.0 + sqrt_1pz2))

    # Debye polynomials u_k(t)
    t2 = t * t
    t3 = t2 * t

    u0 = 1.0
    u1 = t * (3.0 - 5.0 * t2) / 24.0
    u2 = t2 * (81.0 - 462.0 * t2 + 385.0 * t2 * t2) / 1152.0
    u3 = t3 * (30375.0 - t2 * (369603.0 - t2 * (765765.0 - 425425.0 * t2))) / 414720.0
    u4 = (t2 * t2 * (4465125.0 - t2 * (94121676.0 - t2 * (
        349922430.0 - t2 * (446185740.0 - 185910725.0 * t2))))
        / 39813120.0)

    inv_v = 1.0 / v
    S = u0 - u1 * inv_v + u2 * inv_v**2 - u3 * inv_v**3 + u4 * inv_v**4

    return (0.5 * jnp.log(jnp.pi / (2.0 * v))
            - v * eta
            - 0.25 * jnp.log(1.0 + zeta2)
            + jnp.log(jnp.maximum(S, 1e-300)))


# ---------------------------------------------------------------------------
# Phase 3: Small-z leading asymptotic (DLMF 10.30.2)
# ---------------------------------------------------------------------------

def _smallz_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """Leading-order asymptotic for z → 0.

    For |v| > 0: K_v(z) ~ Gamma(|v|)/2 * (2/z)^|v|
    log K_v(z) ~ lgamma(|v|) + |v|*ln(2/z) - ln(2)

    For |v| ≈ 0: K_0(z) ~ -ln(z/2) - gamma_euler
    """
    v_abs = jnp.abs(v)
    log_large_v = (jax.lax.lgamma(v_abs) + v_abs * jnp.log(2.0 / z)
                   - jnp.log(2.0))
    log_small_v = jnp.log(jnp.maximum(
        -jnp.log(z / 2.0) - jnp.euler_gamma, 1e-300))
    return jnp.where(v_abs > 0.5, log_large_v, log_small_v)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_HANKEL_THRESHOLD = 25.0
_OLVER_V_THRESHOLD = 25.0

@functools.partial(jax.custom_jvp, nondiff_argnums=())
def log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """
    log K_v(z), pure-JAX with zero scipy callbacks.

    Parameters
    ----------
    v : scalar or array — order (any real; K_v = K_{-v})
    z : scalar or array — argument (must be > 0), same shape as v

    Returns
    -------
    log_kv : same shape as v and z

    Regime selection:
        1. z > max(25, v^2/4)       → Hankel asymptotic
        2. |v| > 25 (not Hankel)    → Olver uniform expansion
        3. z < 1e-6 and |v| > 0.5   → Small-z leading asymptotic
        4. Otherwise                 → Gauss-Legendre quadrature
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(v.shape, z.shape)
    v = jnp.broadcast_to(v, shape)
    z = jnp.broadcast_to(z, shape)
    z = jnp.maximum(z, jnp.finfo(jnp.float64).tiny)
    v_abs = jnp.abs(v)

    # Regime flags (mutually exclusive in priority order)
    use_hankel = z > jnp.maximum(_HANKEL_THRESHOLD, v_abs * v_abs / 4.0)
    use_olver = (~use_hankel) & (v_abs > _OLVER_V_THRESHOLD)
    use_smallz = (~use_hankel) & (~use_olver) & (z < 1e-6) & (v_abs > 0.5)

    # Hankel: clamp z to safe values for non-Hankel branches
    z_hankel = jnp.where(use_hankel, z,
                         jnp.maximum(_HANKEL_THRESHOLD, v_abs * v_abs / 4.0 + 1.0))
    result_hankel = _hankel_log_kv(v_abs, z_hankel)

    # Olver: clamp v to safe values (need v > 0 for zeta = z/v)
    v_olver = jnp.where(use_olver, v_abs, _OLVER_V_THRESHOLD + 1.0)
    z_olver = jnp.maximum(z, 1e-300)
    result_olver = _olver_log_kv(v_olver, z_olver)

    # Small-z: clamp z and v to safe values
    z_smallz = jnp.clip(z, 1e-300, 0.5)
    v_smallz = jnp.maximum(v_abs, 0.5)
    result_smallz = _smallz_log_kv(v_smallz, z_smallz)

    # Quadrature: safe z
    z_quad = jnp.maximum(z, 1e-30)
    result_quad = _quadrature_log_kv(v_abs, z_quad)

    # Compose: priority Hankel > Olver > Small-z > Quadrature
    result = result_quad
    result = jnp.where(use_smallz, result_smallz, result)
    result = jnp.where(use_olver, result_olver, result)
    result = jnp.where(use_hankel, result_hankel, result)

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
