"""Contract tests for the Bessel moment-quadrature oracle (S10)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, exp1, polygamma

from normix.utils.bessel import (
    _BACKENDS, _ab_coeffs, _log_kv_quad_jax, log_kv, log_kv_moments,
)
from normix.utils.constants import LOG_EPS

_REF_PATH = Path(__file__).resolve().parent / "data" / "bessel_reference.json"

_BACKEND_NAMES = list(_BACKENDS)

# Former audit points plus the two window-stress cases.
_AUDIT = [
    (0.0, 1.0), (0.5, 1.0), (25.0, 0.1), (25.0, 1.0), (25.0, 1e-6),
    (1.0, 1e4), (5.0, 1e6), (0.0, 1e-4),
]


def _rel_err(got: float, ref: float) -> float:
    """Relative error. Absolute only when the reference is exactly zero."""
    denom = abs(ref)
    if denom == 0.0:
        return abs(got)
    return abs(got - ref) / denom


def _ok(got: float, ref: float, v: float, key: str) -> tuple[bool, float]:
    """Relative vs mpmath; ν=0 mixed / first derivatives are zero by symmetry.

    ``log_k``, ``d_z``, ``d_vv``, ``d_zz`` hold 1e-11. ``d_v`` and ``d_vz``
    at large z are E[u]~ν/z corrections and use 1e-7. Tight small-curvature
    pins are ``test_half_order_arg_curvature`` and the GIG directional check.
    """
    if abs(v) == 0.0 and key == "d_vz":
        err = abs(got - ref)
        return err <= 1e-5, err
    if abs(v) == 0.0 and key == "d_v":
        err = abs(got - ref)
        return err <= 1e-10, err
    err = _rel_err(got, ref)
    tol = 1e-7 if key in ("d_v", "d_vz") else 1e-11
    return err <= tol, err


def _jet_ok(got: float, ref: float, v: float, key: str) -> tuple[bool, float]:
    """AD vs bundle: absolute near symmetry zeros, else 1e-11 relative.

    Mixed Hessians at large z agree to ~11 digits; 1e-12 relative is one
    ulp past the two accumulations. ν=0 first order is O(10^{-17}).
    """
    if abs(v) == 0.0 and key in ("d_v", "d_vz"):
        err = abs(got - ref)
        return err <= 1e-12, err
    err = _rel_err(got, ref)
    return err <= 1e-11, err


def _second_order_jet(fn, v, z):
    """Forward-over-forward 2-jet of a scalar kernel ``fn(v, z)``."""
    v = jnp.asarray(v, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)

    def f(x):
        return fn(x[0], x[1])

    x = jnp.stack([v, z])
    return f(x), jax.grad(f)(x), jax.hessian(f)(x)


def _as_float(x) -> float:
    return float(np.asarray(x))


_jet_fn = jax.jit(lambda v, z: _second_order_jet(_log_kv_quad_jax, v, z))


@pytest.fixture(scope="module")
def ref_table():
    if not _REF_PATH.exists():
        pytest.skip(f"missing {_REF_PATH}; run scripts/gen_bessel_reference.py")
    return json.loads(_REF_PATH.read_text())


# ---------------------------------------------------------------------------
# Reference table
# ---------------------------------------------------------------------------

@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_reference_scaled_error(backend, ref_table):
    """Kernel matches the frozen mpmath table (1e-11 relative; 1e-7 for d_v / d_vz)."""
    rows = [r for r in ref_table["points"] if r["z"] >= 1e-10]
    vs = np.array([r["v"] for r in rows], dtype=np.float64)
    zs = np.array([r["z"] for r in rows], dtype=np.float64)
    if backend == "jax":
        def _pack(v, z):
            m = log_kv_moments(v, z, backend="jax")
            return (m.log_k, m.d_order, m.d_arg, m.d2_order,
                    m.d2_order_arg, m.d2_arg)
        packed = jax.jit(_pack)(jnp.asarray(vs), jnp.asarray(zs))
        got_cols = [np.asarray(a) for a in packed]
    else:
        m = log_kv_moments(vs, zs, backend="cpu")
        got_cols = [
            np.asarray(m.log_k), np.asarray(m.d_order), np.asarray(m.d_arg),
            np.asarray(m.d2_order), np.asarray(m.d2_order_arg),
            np.asarray(m.d2_arg),
        ]
    keys = ("log_k", "d_v", "d_z", "d_vv", "d_vz", "d_zz")
    worst = 0.0
    worst_key = None
    for i, row in enumerate(rows):
        v, z = row["v"], row["z"]
        got = {k: float(got_cols[j][i]) for j, k in enumerate(keys)}
        for key in keys:
            ok, err = _ok(got[key], row[key], v, key)
            if err > worst:
                worst = err
                worst_key = (v, z, key, got[key], row[key])
            assert ok, (
                f"{backend} {key}({v}, {z}): got {got[key]}, ref {row[key]}, "
                f"err={err:.2e}"
            )
    assert worst_key is not None


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
@pytest.mark.parametrize("v,z", _AUDIT + [(10.0, 10.0), (-25.0, 1.0), (300.0, 1.0)])
def test_invariants(backend, v, z):
    m = log_kv_moments(v, z, backend=backend)
    cov = np.asarray(m.cov, dtype=np.float64)
    mean = np.asarray(m.mean, dtype=np.float64)
    w = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert np.all(w >= -1e-12), w
    assert _as_float(m.d2_order) > 0.0
    assert _as_float(m.d2_arg) >= 0.0
    assert _as_float(m.d_arg) < 0.0
    assert np.isfinite(_as_float(m.log_k))
    u0 = _as_float(m.u0)
    em = math.exp(-u0) * (1.0 + mean[1])
    ep = math.exp(u0) * (1.0 + mean[2])
    assert em * ep >= 1.0 - 1e-12


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_order_zero_and_parity(backend):
    z = 2.0
    m0 = log_kv_moments(0.0, z, backend=backend)
    assert abs(_as_float(m0.d_order)) <= 1e-12
    m_pos = log_kv_moments(2.5, z, backend=backend)
    m_neg = log_kv_moments(-2.5, z, backend=backend)
    assert _rel_err(_as_float(m_pos.log_k), _as_float(m_neg.log_k)) <= 1e-12
    assert _rel_err(_as_float(m_pos.d_order), -_as_float(m_neg.d_order)) <= 1e-12


@pytest.mark.contract
def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        log_kv_moments(0.5, 1.0, backend="native")


# ---------------------------------------------------------------------------
# Identities as oracles
# ---------------------------------------------------------------------------

@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_recurrence_lz(backend):
    """L_z = -½ (K_{ν-1} + K_{ν+1}) / K_ν from separately evaluated values."""
    v, z = 1.5, 2.0
    m = log_kv_moments(v, z, backend=backend)
    Lm = log_kv_moments(v - 1.0, z, backend=backend)
    Lp = log_kv_moments(v + 1.0, z, backend=backend)
    Lz_rec = -0.5 * (
        math.exp(_as_float(Lm.log_k) - _as_float(m.log_k))
        + math.exp(_as_float(Lp.log_k) - _as_float(m.log_k))
    )
    assert _rel_err(_as_float(m.d_arg), Lz_rec) <= 1e-12


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_dlmf_10_38_7(backend):
    """∂_ν log K_ν at ν=1/2 equals e^{2z} E_1(2z)."""
    z = 1.0
    got = _as_float(log_kv_moments(0.5, z, backend=backend).d_order)
    ref = math.exp(2.0 * z) * float(exp1(2.0 * z))
    assert _rel_err(got, ref) <= 1e-12


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_smallz_digamma(backend):
    """Small-z leading term: ψ(|ν|) + log(2/z) and ψ₁(|ν|)."""
    v, z = 2.5, 1e-10
    m = log_kv_moments(v, z, backend=backend)
    va = abs(v)
    d1 = float(digamma(va) + np.log(2.0 / z))
    d2 = float(polygamma(1, va))
    assert _rel_err(_as_float(m.d_order), d1) <= 1e-12
    assert _rel_err(_as_float(m.d2_order), d2) <= 1e-12
    log_k = float(math.lgamma(va) + va * math.log(2.0 / z) - math.log(2.0))
    assert _rel_err(_as_float(m.log_k), log_k) <= 1e-12


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_hankel_lvv(backend):
    """Large-z: Var(u) ~ 1/z − 1/(2z²)."""
    v, z = 1.0, 1e4
    got = _as_float(log_kv_moments(v, z, backend=backend).d2_order)
    ref = 1.0 / z - 1.0 / (2.0 * z * z)
    assert _rel_err(got, ref) <= 1e-6


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_log_kv_tiny_z_no_clip(backend):
    """``log_kv(1, 1e-50)`` is log(1/z) ≈ 115.129, not the LOG_EPS clip."""
    got = _as_float(log_kv_moments(1.0, 1e-50, backend=backend).log_k)
    assert abs(got - 115.12925465) < 1e-8
    assert got > 110.0
    assert abs(got - math.log(1.0 / LOG_EPS)) > 1.0


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_gig_degen_psi_via_moments(backend):
    """GIG(-2, 1, 1e-22) log-partition via the Bessel formula, both backends."""
    p, a, b = -2.0, 1.0, 1e-22
    z = math.sqrt(a * b)
    m = log_kv_moments(p, z, backend=backend)
    psi = math.log(2.0) + _as_float(m.log_k) + 0.5 * p * (math.log(b) - math.log(a))
    assert abs(psi - 102.700038453) < 1e-9


# ---------------------------------------------------------------------------
# AD ≡ bundle, jaxpr, vmap
# ---------------------------------------------------------------------------

@pytest.mark.contract
@pytest.mark.parametrize("v,z", _AUDIT[:7])
def test_ad_equals_bundle(v, z):
    m = log_kv_moments(jnp.array(v), jnp.array(z), backend="jax")
    val, g, H = _jet_fn(jnp.array(v), jnp.array(z))
    checks = (
        ("log_k", _as_float(val), _as_float(m.log_k)),
        ("d_v", _as_float(g[0]), _as_float(m.d_order)),
        ("d_z", _as_float(g[1]), _as_float(m.d_arg)),
        ("d_vv", _as_float(H[0, 0]), _as_float(m.d2_order)),
        ("d_vz", _as_float(H[0, 1]), _as_float(m.d2_order_arg)),
        ("d_zz", _as_float(H[1, 1]), _as_float(m.d2_arg)),
    )
    for key, got, ref in checks:
        ok, err = _jet_ok(got, ref, v, key)
        assert ok, f"{key}({v}, {z}): got {got}, ref {ref}, err={err:.2e}"


@pytest.mark.contract
def test_mode_agrees_with_asinh():
    v, z = 2.5, 0.3
    m = log_kv_moments(v, z, backend="cpu")
    assert _rel_err(_as_float(m.u0), math.asinh(v / z)) <= 1e-14


@pytest.mark.contract
def test_third_derivative_finite():
    v = jnp.array(1.5)
    z = jnp.array(2.0)
    d3 = jax.grad(jax.hessian(lambda vv: _log_kv_quad_jax(vv, z)))(v)
    assert jnp.isfinite(d3)


@pytest.mark.contract
def test_jaxpr_has_no_cond():
    v = jnp.array(1.5)
    z = jnp.array(2.0)
    jp = str(jax.make_jaxpr(_log_kv_quad_jax)(v, z))
    assert jp.count("cond") == 0


@pytest.mark.contract
def test_vmap_matches_scalar_loop():
    vs = jnp.array([0.0, 0.5, 2.5, -1.0, 25.0])
    zs = jnp.array([1.0, 1.0, 0.3, 2.0, 1.0])
    batched = jax.jit(jax.vmap(_log_kv_quad_jax))(vs, zs)
    loop = jnp.array([_log_kv_quad_jax(v, z) for v, z in zip(vs, zs)])
    np.testing.assert_allclose(np.asarray(batched), np.asarray(loop), rtol=1e-14)


@pytest.mark.contract
def test_cpu_jax_moments_agree():
    v, z = 3.0, 0.7
    mj = log_kv_moments(v, z, backend="jax")
    mc = log_kv_moments(v, z, backend="cpu")
    for a, b in (
        (mj.log_k, mc.log_k),
        (mj.d_order, mc.d_order),
        (mj.d_arg, mc.d_arg),
        (mj.d2_order, mc.d2_order),
        (mj.d2_arg, mc.d2_arg),
    ):
        assert _rel_err(_as_float(a), _as_float(b)) <= 1e-12


@pytest.mark.contract
def test_cpu_vectorized_batch():
    v = np.full(8, 0.5)
    z = np.linspace(0.5, 5.0, 8)
    m = log_kv_moments(v, z, backend="cpu")
    assert np.asarray(m.log_k).shape == (8,)
    assert np.asarray(m.mean).shape == (8, 3)
    assert np.asarray(m.cov).shape == (8, 3, 3)
    assert np.asarray(m.d_arg).shape == (8,)
    assert np.asarray(m.d2_arg).shape == (8,)


# ---------------------------------------------------------------------------
# Exact identities / extreme argument
# ---------------------------------------------------------------------------

@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
@pytest.mark.parametrize("z", [1e2, 1e4, 1e6, 1e8, 1e10, 1e12])
def test_half_order_arg_curvature(backend, z):
    """2 z² ∂_{zz} log K_{1/2} = 1 (DLMF 10.39.2)."""
    m = log_kv_moments(0.5, z, backend=backend)
    got = 2.0 * z * z * _as_float(m.d2_arg)
    assert abs(got - 1.0) <= 1e-8, (z, got)


@pytest.mark.contract
@pytest.mark.parametrize("backend", _BACKEND_NAMES)
def test_log_kv_large_z_finite(backend):
    """z² in κ−ν must not overflow; log K_{1/2}(10^{155}) is finite."""
    z = 1e155
    kappa, a, b = _ab_coeffs(np.array(0.5), np.array(z), np)
    assert np.isfinite(kappa) and np.isfinite(a) and np.isfinite(b)
    got = _as_float(log_kv(0.5, z, backend=backend))
    assert np.isfinite(got)
    assert got < 0.0


# ---------------------------------------------------------------------------
# Former seams: d_order vs d2_order consistency
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("z", [1e-6, 1.0, 100.0])
def test_former_seam_order_consistency(z):
    """Δ L_ν / h matches L_{νν} at the midpoint; no global Lipschitz bound.

    At z=10^{-6}, L_{νν}(0) ≈ 66 so a 0.01 step in ν moves L_ν by ~0.66.
    A seam would show up as a first-order Taylor residual of that size.
    """
    h = 0.01
    vs = np.arange(-30.0, 30.0 + 1e-12, h)
    m = log_kv_moments(vs, np.full_like(vs, z), backend="cpu")
    lv = np.asarray(m.d_order, dtype=np.float64)
    lvv = np.asarray(m.d2_order, dtype=np.float64)
    assert np.all(lvv > 0.0)
    mid = 0.5 * (lvv[1:] + lvv[:-1])
    resid = np.diff(lv) / h - mid
    scale = np.maximum(np.abs(mid), 1e-12)
    assert np.max(np.abs(resid) / scale) < 0.05


@pytest.mark.slow
def test_third_derivative_richardson():
    """Loose Richardson check on ∂³_ν log K; FD is a test oracle only."""
    v, z = 1.5, 2.0
    d3 = float(jax.grad(jax.hessian(lambda vv: _log_kv_quad_jax(vv, z)))(
        jnp.array(v)
    ))

    def L(order):
        return _as_float(log_kv_moments(order, z, backend="cpu").log_k)

    def fd3(h):
        return (L(v + 2 * h) - 2.0 * L(v + h) + 2.0 * L(v - h) - L(v - 2 * h)) / (
            2.0 * h ** 3
        )

    h1, h2 = 1e-3, 5e-4
    r1, r2 = fd3(h1), fd3(h2)
    # Richardson: (4 r2 − r1)/3 is O(h⁴) for this stencil.
    rich = (4.0 * r2 - r1) / 3.0
    assert _rel_err(d3, rich) < 1e-4
