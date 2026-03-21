"""
JAX-compatible log modified Bessel function of the second kind.

log_kv(v, z) = log K_v(z), fully pure-JAX with zero scipy callbacks.

Regime-specific methods, selected via lax.cond (only one branch executes):
  1. Hankel asymptotic (DLMF 10.40.2)   — large z
  2. Olver uniform expansion (DLMF 10.41.4) — large v
  3. Small-z leading asymptotic (DLMF 10.30.2) — z → 0
  4. Gauss-Legendre quadrature (Takekawa 2022) — moderate z, v

Using lax.cond (not jnp.where) means only the selected branch executes
at runtime. lax.cond requires scalar conditions, so the core scalar
function _log_kv_scalar is vmapped over array inputs.

Custom JVP for full autodiff:
  - ∂/∂z : exact recurrence K'_v = −(K_{v−1}+K_{v+1})/2
  - ∂/∂v : central FD on log_kv itself (ε = BESSEL_EPS_V)

backend='jax'  (default): pure-JAX, JIT-able, differentiable.
backend='cpu'           : scipy.special.kve, fully vectorized numpy.
                          Not JIT-able. Fast for EM hot path.
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.constants import TINY, LOG_EPS, BESSEL_SMALLZ_THRESHOLD

jax.config.update("jax_enable_x64", True)

_N_QUAD = 64
_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(_N_QUAD)
_GL_NODES = jnp.asarray(_GL_NODES_NP, dtype=jnp.float64)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS_NP, dtype=jnp.float64)
_LOG_GL_WEIGHTS = jnp.log(_GL_WEIGHTS)

_HANKEL_THRESHOLD = 25.0
_OLVER_V_THRESHOLD = 25.0


# ---------------------------------------------------------------------------
# Branch 1: Hankel asymptotic expansion (DLMF 10.40.2) — large z
# ---------------------------------------------------------------------------

def _hankel_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """K_v(z) ~ √(π/(2z)) e^{−z} Σ_k a_k(v)/z^k, accurate for z > max(25, v²/4)."""
    mu = 4.0 * v * v
    total = jnp.ones_like(z)
    term = jnp.ones_like(z)
    for k in range(1, 21):
        term = term * (mu - (2.0 * k - 1.0) ** 2) / (8.0 * k * z)
        total = total + term
    return 0.5 * jnp.log(jnp.pi / (2.0 * z)) - z + jnp.log(jnp.maximum(total, TINY))


# ---------------------------------------------------------------------------
# Branch 2: Olver uniform expansion (DLMF 10.41.3-4) — large v
# ---------------------------------------------------------------------------

def _olver_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """Uniform asymptotic for large v (v > 25). Accurate to ~12 digits."""
    zeta = z / v
    zeta2 = zeta * zeta
    sqrt_1pz2 = jnp.sqrt(1.0 + zeta2)
    t = 1.0 / sqrt_1pz2
    eta = sqrt_1pz2 + jnp.log(zeta / (1.0 + sqrt_1pz2))

    t2 = t * t
    t3 = t2 * t
    u0 = 1.0
    u1 = t * (3.0 - 5.0 * t2) / 24.0
    u2 = t2 * (81.0 - 462.0 * t2 + 385.0 * t2 * t2) / 1152.0
    u3 = t3 * (30375.0 - t2 * (369603.0 - t2 * (765765.0 - 425425.0 * t2))) / 414720.0
    u4 = (t2 * t2 * (4465125.0 - t2 * (94121676.0 - t2 * (
        349922430.0 - t2 * (446185740.0 - 185910725.0 * t2)))) / 39813120.0)

    inv_v = 1.0 / v
    S = u0 - u1 * inv_v + u2 * inv_v**2 - u3 * inv_v**3 + u4 * inv_v**4

    return (0.5 * jnp.log(jnp.pi / (2.0 * v))
            - v * eta
            - 0.25 * jnp.log(1.0 + zeta2)
            + jnp.log(jnp.maximum(S, TINY)))


# ---------------------------------------------------------------------------
# Branch 3: Small-z leading asymptotic (DLMF 10.30.2) — z → 0
# ---------------------------------------------------------------------------

def _smallz_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """K_v(z) ~ Gamma(|v|)/2 * (2/z)^|v| for z→0, |v|>0."""
    v_abs = jnp.abs(v)
    log_large_v = jax.lax.lgamma(v_abs) + v_abs * jnp.log(2.0 / z) - jnp.log(2.0)
    log_small_v = jnp.log(jnp.maximum(-jnp.log(z / 2.0) - jnp.euler_gamma, TINY))
    # v_abs > 0.5 is a scalar bool here (called from _log_kv_scalar)
    return jax.lax.cond(v_abs > 0.5, lambda: log_large_v, lambda: log_small_v)


# ---------------------------------------------------------------------------
# Branch 4: Gauss-Legendre quadrature (Takekawa 2022) — moderate z, v
# ---------------------------------------------------------------------------

def _log_cosh(x: jax.Array) -> jax.Array:
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - jnp.log(2.0)


def _quadrature_upper_bound(v_abs: jax.Array, z: jax.Array) -> jax.Array:
    P = 50.0
    z_safe = jnp.maximum(z, TINY)
    T = jnp.maximum(jnp.log(2.0 * P / z_safe), 1.0)
    for _ in range(6):
        T = jnp.maximum(jnp.log(2.0 * (P + v_abs * T) / z_safe), 1.0)
    return T + 2.0


def _quadrature_log_kv(v: jax.Array, z: jax.Array) -> jax.Array:
    """64-point GL quadrature of K_v(z) = int_0^inf exp(-z cosh t) cosh(vt) dt."""
    v_abs = jnp.abs(v)
    T = _quadrature_upper_bound(v_abs, z)
    t = T * (_GL_NODES + 1.0) / 2.0     # (N_QUAD,)
    log_f = -z * jnp.cosh(t) + _log_cosh(v_abs * t)
    return jnp.log(T / 2.0) + jax.scipy.special.logsumexp(_LOG_GL_WEIGHTS + log_f)


# ---------------------------------------------------------------------------
# Scalar dispatcher — uses lax.cond so only one branch executes
# ---------------------------------------------------------------------------

def _log_kv_scalar(v: jax.Array, z: jax.Array) -> jax.Array:
    """Compute log K_v(z) for scalar v, z using lax.cond regime selection."""
    v_abs = jnp.abs(v)

    use_hankel = z > jnp.maximum(_HANKEL_THRESHOLD, v_abs * v_abs / 4.0)
    use_olver  = v_abs > _OLVER_V_THRESHOLD
    use_smallz = (z < BESSEL_SMALLZ_THRESHOLD) & (v_abs > 0.5)

    # Innermost: Olver vs Small-z vs Quadrature
    def _not_hankel():
        def _olver():
            return _olver_log_kv(v_abs, jnp.maximum(z, TINY))
        def _not_olver():
            return jax.lax.cond(
                use_smallz,
                lambda: _smallz_log_kv(v_abs, jnp.clip(z, TINY, 0.5)),
                lambda: _quadrature_log_kv(v_abs, jnp.maximum(z, LOG_EPS)),
            )
        return jax.lax.cond(use_olver, _olver, _not_olver)

    def _hankel():
        return _hankel_log_kv(v_abs, z)

    return jax.lax.cond(use_hankel, _hankel, _not_hankel)


# ---------------------------------------------------------------------------
# Pure-JAX implementation (private) — custom JVP, JIT-able, differentiable
# ---------------------------------------------------------------------------

@functools.partial(jax.custom_jvp, nondiff_argnums=())
def _log_kv_jax(v: jax.Array, z: jax.Array) -> jax.Array:
    """
    log K_v(z), pure-JAX with zero scipy callbacks.

    Regime selection via lax.cond (only the selected branch executes):
        1. z > max(25, v²/4)         → Hankel asymptotic
        2. |v| > 25 (not Hankel)     → Olver uniform expansion
        3. z < 1e-6, |v| > 0.5      → Small-z leading asymptotic
        4. Otherwise                  → Gauss-Legendre quadrature

    Parameters
    ----------
    v : scalar or array — order (any real; K_v = K_{-v})
    z : scalar or array — argument (must be > 0)
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(v.shape, z.shape)
    v = jnp.broadcast_to(v, shape)
    z = jnp.broadcast_to(z, shape)
    z = jnp.maximum(z, jnp.finfo(jnp.float64).tiny)

    if shape == ():
        return _log_kv_scalar(v, z)
    return jax.vmap(_log_kv_scalar)(v.ravel(), z.ravel()).reshape(shape)


@_log_kv_jax.defjvp
def _log_kv_jax_jvp(primals, tangents):
    v, z = primals
    dv, dz = tangents
    primal_out = _log_kv_jax(v, z)

    log_kvm1 = _log_kv_jax(v - 1.0, z)
    log_kvp1 = _log_kv_jax(v + 1.0, z)
    dlogkv_dz = -0.5 * (jnp.exp(log_kvm1 - primal_out) + jnp.exp(log_kvp1 - primal_out))

    from normix.utils.constants import BESSEL_EPS_V
    _EPS_V = jnp.asarray(BESSEL_EPS_V, dtype=jnp.float64)
    dlogkv_dv = (_log_kv_jax(v + _EPS_V, z) - _log_kv_jax(v - _EPS_V, z)) / (2.0 * _EPS_V)

    return primal_out, dlogkv_dz * dz + dlogkv_dv * dv


# ---------------------------------------------------------------------------
# CPU implementation (private) — scipy.special.kve, vectorized numpy
# ---------------------------------------------------------------------------

def _log_kv_cpu(v, z):
    """
    log K_v(z) via scipy.special.kve. Fully vectorized numpy.

    Not JIT-able. Fast for the EM hot path (6 C-level array calls).
    Handles scalars and arrays of any shape via broadcasting.
    """
    from scipy.special import kve, gammaln

    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    out_shape = np.broadcast_shapes(v.shape, z.shape)
    # Ravel to 1-d so item assignment always works (0-d arrays don't support it)
    v_flat = np.broadcast_to(v, out_shape).ravel().copy()
    z_flat = np.broadcast_to(z, out_shape).ravel().copy()
    z_flat = np.maximum(z_flat, np.finfo(np.float64).tiny)

    result = np.log(kve(v_flat, z_flat)) - z_flat

    inf_mask = np.isinf(result)
    if np.any(inf_mask):
        v_abs = np.abs(v_flat[inf_mask])
        z_inf = z_flat[inf_mask]
        large_v = v_abs > 0.5
        result[inf_mask] = np.where(
            large_v,
            gammaln(v_abs) - np.log(2.0) + v_abs * (np.log(2.0) - np.log(z_inf)),
            np.log(np.maximum(-np.log(z_inf / 2.0) - np.euler_gamma, TINY)),
        )
    return result.reshape(out_shape) if out_shape else result[0]


# ---------------------------------------------------------------------------
# Public API — unified dispatcher with backend selection
# ---------------------------------------------------------------------------

def log_kv(v, z, backend: str = 'jax'):
    """
    log K_v(z) — log modified Bessel function of the second kind.

    Parameters
    ----------
    v : scalar or array — order (any real; K_v = K_{-v})
    z : scalar or array — argument (must be > 0)
    backend : 'jax' (default) or 'cpu'
        'jax' : pure-JAX, lax.cond regime selection, custom JVP.
                JIT-able, differentiable. Default for log_prob, pdf, etc.
        'cpu' : scipy.special.kve, fully vectorized numpy.
                Not JIT-able. Fast for EM hot path.

    Returns
    -------
    scalar or array of same broadcast shape as (v, z).
    """
    if backend == 'cpu':
        return _log_kv_cpu(v, z)
    return _log_kv_jax(v, z)
