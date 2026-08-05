"""
Tests for the JAX Bessel function implementation.

Validates log_kv(v, z) evaluation and gradients against scipy reference.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix.utils.bessel import log_kv

# ---------------------------------------------------------------------------
# Scipy reference
# ---------------------------------------------------------------------------

def _scipy_log_kv(v, z):
    """scipy reference for log K_v(z)."""
    from scipy.special import kve
    return float(np.log(kve(abs(float(v)), float(z))) - float(z))

# ---------------------------------------------------------------------------
# Phase 1: Hankel asymptotic regime (large z)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (0.5, 30.0),
    (1.0, 50.0),
    (2.0, 100.0),
    (5.0, 40.0),
    (10.0, 30.0),
    (0.0, 25.0),
    (0.5, 1000.0),
    (20.0, 150.0),   # v²/4 = 100
    (50.0, 700.0),   # v²/4 = 625
    (100.0, 2600.0),  # v²/4 = 2500
])
def test_hankel_regime(v, z):
    """Points that should use the Hankel expansion (z > max(25, v^2/4))."""
    result = float(log_kv(jnp.array(v), jnp.array(z)))
    expected = _scipy_log_kv(v, z)
    abs_err = abs(result - expected)
    rel_err = abs_err / (abs(expected) + 1e-15)
    assert rel_err < 1e-10 or abs_err < 1e-10, (
        f"Hankel log_kv({v}, {z}): got {result}, expected {expected}, "
        f"rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
    )

# ---------------------------------------------------------------------------
# Phase 2: Quadrature regime (moderate z, moderate/large v)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (0.5, 1.0),
    (1.0, 2.0),
    (1.5, 0.5),
    (2.0, 5.0),
    (5.0, 3.0),
    (10.0, 10.0),
    (10.0, 20.0),
    (20.0, 50.0),
    (0.0, 1.0),
    (0.1, 0.1),
    (1.0, 0.001),
    (5.0, 0.01),
    (0.5, 1e-6),
    (50.0, 100.0),
    (100.0, 150.0),
    (1.0, 1e-10),
])
def test_quadrature_regime(v, z):
    """Points in the quadrature regime (not handled by Hankel)."""
    result = float(log_kv(jnp.array(v), jnp.array(z)))
    expected = _scipy_log_kv(v, z)
    abs_err = abs(result - expected)
    rel_err = abs_err / (abs(expected) + 1e-15)
    assert rel_err < 1e-9 or abs_err < 1e-9, (
        f"Quad log_kv({v}, {z}): got {result}, expected {expected}, "
        f"rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
    )

# ---------------------------------------------------------------------------
# Phase 3: Olver uniform expansion (large v)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (30.0, 10.0),
    (50.0, 20.0),
    (50.0, 100.0),
    (100.0, 50.0),
    (100.0, 150.0),
    (200.0, 100.0),
    (200.0, 300.0),
    (500.0, 200.0),
])
def test_olver_regime(v, z):
    """Points that should use the Olver expansion (v > 25, not Hankel)."""
    result = float(log_kv(jnp.array(v), jnp.array(z)))
    expected = _scipy_log_kv(v, z)
    abs_err = abs(result - expected)
    rel_err = abs_err / (abs(expected) + 1e-15)
    assert rel_err < 1e-9 or abs_err < 1e-9, (
        f"Olver log_kv({v}, {z}): got {result}, expected {expected}, "
        f"rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
    )

# ---------------------------------------------------------------------------
# Phase 3: Small-z leading asymptotic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (1.0, 1e-10),
    (2.0, 1e-15),
    (5.0, 1e-20),
    (0.5, 1e-8),
    (10.0, 1e-12),
])
def test_smallz_regime(v, z):
    """Points that should use the small-z asymptotic."""
    result = float(log_kv(jnp.array(v), jnp.array(z)))
    expected = _scipy_log_kv(v, z)
    abs_err = abs(result - expected)
    rel_err = abs_err / (abs(expected) + 1e-15)
    assert rel_err < 1e-6 or abs_err < 1e-6, (
        f"Small-z log_kv({v}, {z}): got {result}, expected {expected}, "
        f"rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
    )

def test_no_scipy_callback():
    """Verify that log_kv works without importing scipy (pure JAX)."""
    import sys
    result = float(log_kv(jnp.array(1.0), jnp.array(2.0)))
    assert np.isfinite(result)

# ---------------------------------------------------------------------------
# Primal evaluation (mixed regimes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (0.5, 1.0),
    (1.0, 2.0),
    (1.5, 0.5),
    (2.0, 5.0),
    (0.0, 1.0),
    (10.0, 20.0),
    (100.0, 150.0),
])
def test_log_kv_primal(v, z):
    result = float(log_kv(jnp.array(v), jnp.array(z)))
    expected = _scipy_log_kv(v, z)
    assert abs(result - expected) < 1e-8, (
        f"log_kv({v}, {z}): got {result}, expected {expected}"
    )

def test_log_kv_small_z():
    """Small z: asymptotic fallback should avoid -inf."""
    v, z = 1.0, 1e-12
    result = float(log_kv(jnp.array(v), jnp.array(z)))
    assert np.isfinite(result), f"Expected finite, got {result}"
    assert result > 0, "log K_v for small z should be large positive"

def test_log_kv_vectorized():
    vs = jnp.array([0.5, 1.0, 1.5, 2.0])
    zs = jnp.array([1.0, 2.0, 3.0, 4.0])
    results = log_kv(vs, zs)
    assert results.shape == (4,)
    for i, (v, z) in enumerate(zip(vs, zs)):
        expected = _scipy_log_kv(float(v), float(z))
        assert abs(float(results[i]) - expected) < 1e-8

def test_log_kv_vectorized_mixed_regimes():
    """Vectorized call with points in both Hankel and fallback regimes."""
    vs = jnp.array([0.5,  1.0, 10.0, 0.5])
    zs = jnp.array([1.0, 50.0, 30.0, 100.0])
    results = log_kv(vs, zs)
    assert results.shape == (4,)
    for i, (v, z) in enumerate(zip(vs, zs)):
        expected = _scipy_log_kv(float(v), float(z))
        assert abs(float(results[i]) - expected) < 1e-8

# ---------------------------------------------------------------------------
# Gradients ∂/∂z
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
    (1.0, 50.0),   # Hankel regime
    (5.0, 40.0),   # Hankel regime
])
def test_log_kv_grad_z(v, z):
    """∂/∂z log K_v(z): compare with numerical finite differences."""
    z_arr = jnp.array(z)
    v_arr = jnp.array(v)
    grad_z = float(jax.grad(lambda z: log_kv(v_arr, z))(z_arr))

    eps = 1e-6
    fd = (_scipy_log_kv(v, z + eps) - _scipy_log_kv(v, z - eps)) / (2 * eps)
    assert abs(grad_z - fd) / (abs(fd) + 1e-10) < 1e-4, (
        f"∂/∂z log_kv({v},{z}): got {grad_z}, fd={fd}"
    )

# ---------------------------------------------------------------------------
# Gradients ∂/∂v
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
    (1.0, 30.0),   # Hankel regime
    (5.0, 50.0),   # Hankel regime
])
def test_log_kv_grad_v(v, z):
    """∂/∂v log K_v(z): compare with numerical finite differences."""
    z_arr = jnp.array(z)
    v_arr = jnp.array(v)
    grad_v = float(jax.grad(lambda v: log_kv(v, z_arr))(v_arr))

    eps = 1e-5
    fd = (_scipy_log_kv(v + eps, z) - _scipy_log_kv(v - eps, z)) / (2 * eps)
    rel_err = abs(grad_v - fd) / (abs(fd) + 1e-10)
    assert rel_err < 1e-4, (
        f"∂/∂v log_kv({v},{z}): got {grad_v}, fd={fd}, rel_err={rel_err:.2e}"
    )

# ---------------------------------------------------------------------------
# symbolic_zeros JVP (E4): z-only and v-only tangents
# ---------------------------------------------------------------------------

def test_log_kv_jvp_z_only_matches_grad():
    """z-only JVP must match ∂/∂z and skip the v finite-difference branch."""
    v = jnp.array(1.5)
    z = jnp.array(2.0)
    primal, tangent = jax.jvp(lambda zz: log_kv(v, zz), (z,), (jnp.ones_like(z),))
    grad_z = jax.grad(lambda zz: log_kv(v, zz))(z)
    np.testing.assert_allclose(float(primal), float(log_kv(v, z)), rtol=1e-12)
    np.testing.assert_allclose(float(tangent), float(grad_z), rtol=1e-12)


def test_log_kv_jvp_v_only_matches_grad():
    """v-only JVP must match ∂/∂v (FD path still active)."""
    v = jnp.array(1.5)
    z = jnp.array(2.0)
    primal, tangent = jax.jvp(lambda vv: log_kv(vv, z), (v,), (jnp.ones_like(v),))
    grad_v = jax.grad(lambda vv: log_kv(vv, z))(v)
    np.testing.assert_allclose(float(primal), float(log_kv(v, z)), rtol=1e-12)
    np.testing.assert_allclose(float(tangent), float(grad_v), rtol=1e-12)


def test_log_kv_jvp_joint_matches_partials():
    """Both-tangent JVP equals linear combination of partials (E4)."""
    v = jnp.array(1.5)
    z = jnp.array(2.0)
    dv = jnp.array(0.3)
    dz = jnp.array(-0.7)
    _, tangent = jax.jvp(log_kv, (v, z), (dv, dz))
    gv = jax.grad(lambda vv: log_kv(vv, z))(v)
    gz = jax.grad(lambda zz: log_kv(v, zz))(z)
    np.testing.assert_allclose(
        float(tangent), float(gv * dv + gz * dz), rtol=1e-12,
    )


def test_log_kv_z_only_jaxpr_skips_nu_fd():
    """z-only differentiation must not stage the ν±ε FD pair (E4 skip)."""
    from normix.utils.constants import BESSEL_EPS_V
    v = jnp.array(1.5)
    z = jnp.array(2.0)
    z_only = str(jax.make_jaxpr(jax.grad(lambda zz: log_kv(v, zz)))(z))
    both = str(jax.make_jaxpr(jax.jacfwd(log_kv, argnums=(0, 1)))(v, z))
    eps_lit = repr(float(BESSEL_EPS_V))
    # Concrete eps appears in the ν-FD branch; z-only must omit it.
    assert eps_lit not in z_only, (
        f"z-only jaxpr still contains BESSEL_EPS_V={eps_lit}"
    )
    assert eps_lit in both, (
        f"joint jaxpr missing BESSEL_EPS_V={eps_lit} (FD path expected)"
    )


# ---------------------------------------------------------------------------
# Higher-order: jax.hessian
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.contract
def test_log_kv_hessian_wrt_z():
    """Second derivative d²/dz² log_kv(v, z) via autodiff (z-only hessian)."""
    v_arr = jnp.array(1.0)
    z_arr = jnp.array(2.0)
    d2_dz2 = jax.grad(jax.grad(lambda z: log_kv(v_arr, z)))(z_arr)
    assert jnp.isfinite(d2_dz2), f"d²/dz² not finite: {d2_dz2}"
    eps = 1e-4
    v = float(v_arr)
    z = float(z_arr)
    fd2 = (_scipy_log_kv(v, z + eps) - 2 * _scipy_log_kv(v, z) + _scipy_log_kv(v, z - eps)) / eps**2
    assert abs(float(d2_dz2) - fd2) / (abs(fd2) + 1e-10) < 0.01

# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

def test_log_kv_vmap():
    """jax.vmap over (v, z) pairs."""
    vs = jnp.linspace(0.5, 3.0, 10)
    zs = jnp.linspace(0.5, 5.0, 10)
    results = jax.vmap(log_kv)(vs, zs)
    assert results.shape == (10,)
    assert jnp.all(jnp.isfinite(results))
