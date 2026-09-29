"""Ambient E[t] and Cov[t] for joint VG, NInvG, and NIG.

The restricted log-partition drops the frozen GIG coordinate, so its
derivatives are not the moments of the ambient statistic
``[log y, 1/y, y, x, x/y, vec(xxᵀ/y)]``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import (
    JointNormalInverseGamma,
    JointNormalInverseGaussian,
    JointVarianceGamma,
)

_MU = jnp.array([1.0])
_GAMMA = jnp.array([0.3])
_SIGMA = jnp.array([[1.0]])

# docs/theory/gh.md block formulas at the review point (d=1).
_VG_ETA = np.array([0.22963715453852185, 1.0, 1.5, 1.45, 1.3, 2.735])
_NINVG_ETA = np.array([-0.22963715453852185, 1.5, 1.0, 1.3, 1.8, 3.19])
_NIG_ETA = np.array([-0.36132861688822, 2.0, 1.0, 1.3, 2.3, 3.69])


def _vg(**sub):
    return JointVarianceGamma.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, alpha=3.0, beta=2.0, **sub,
    )


@pytest.mark.contract
@pytest.mark.parametrize("backend", ["jax", "cpu"])
def test_vg_review_point_expectation_and_fisher(backend):
    j = _vg()
    eta = np.asarray(j.expectation_params(backend=backend))
    np.testing.assert_allclose(eta, _VG_ETA, rtol=1e-9, atol=1e-12)
    F = np.asarray(j.fisher_information(backend=backend))
    assert F.shape == (6, 6)
    lam = np.linalg.eigvalsh(0.5 * (F + F.T))
    np.testing.assert_allclose(lam[0], 0.015046671244, rtol=0, atol=1e-9)
    assert lam[0] > 0


@pytest.mark.contract
@pytest.mark.parametrize("backend", ["jax", "cpu"])
def test_ninvg_review_point_expectation_and_fisher(backend):
    j = JointNormalInverseGamma.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, alpha=3.0, beta=2.0,
    )
    eta = np.asarray(j.expectation_params(backend=backend))
    np.testing.assert_allclose(eta, _NINVG_ETA, rtol=1e-9, atol=1e-12)
    F = np.asarray(j.fisher_information(backend=backend))
    lam = np.linalg.eigvalsh(0.5 * (F + F.T))
    np.testing.assert_allclose(lam[0], 0.014971013275, rtol=0, atol=1e-9)
    assert lam[0] > 0


@pytest.mark.contract
@pytest.mark.parametrize("backend", ["jax", "cpu"])
def test_nig_review_point_expectation_and_fisher(backend):
    j = JointNormalInverseGaussian.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, mu_ig=1.0, lam=1.0,
    )
    eta = np.asarray(j.expectation_params(backend=backend))
    np.testing.assert_allclose(eta, _NIG_ETA, rtol=1e-9, atol=1e-12)
    F = np.asarray(j.fisher_information(backend=backend))
    lam = np.linalg.eigvalsh(0.5 * (F + F.T))
    np.testing.assert_allclose(lam[0], 0.023395505434, rtol=0, atol=1e-8)
    assert lam[0] > 0


@pytest.mark.contract
def test_vg_expectation_is_not_restricted_gradient():
    """The frozen 1/y slot is E[Y^{-1}], not the zero partial of ψ_ext."""
    j = _vg()
    eta = np.asarray(j.expectation_params())
    grad = np.asarray(jax.grad(type(j)._log_partition_from_theta)(j.natural_params()))
    assert abs(grad[1]) < 1e-8
    np.testing.assert_allclose(eta[1], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(eta[4], 1.3, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grad[4], 0.3, rtol=0, atol=1e-8)


@pytest.mark.contract
def test_vg_fisher_infinite_when_inverse_second_moment_diverges():
    """α ≤ 2: E[Y^{-2}] = +∞, so Cov[t] is not a finite matrix."""
    j = JointVarianceGamma.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, alpha=1.5, beta=2.0,
    )
    F = np.asarray(j.fisher_information())
    assert F.shape == (6, 6)
    assert np.isposinf(F[1, 1])
    assert np.isfinite(F[0, 0])
    assert not np.any(np.isnan(F))
    assert not np.all(np.isfinite(F))


@pytest.mark.contract
def test_ninvg_fisher_infinite_when_second_moment_diverges():
    """α ≤ 2: E[Y²] = +∞, so the y-slot variance is +∞."""
    j = JointNormalInverseGamma.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, alpha=1.5, beta=2.0,
    )
    F = np.asarray(j.fisher_information())
    assert F.shape == (6, 6)
    assert np.isposinf(F[2, 2])
    assert np.isfinite(F[0, 0])
    assert not np.any(np.isnan(F))
    assert not np.all(np.isfinite(F))


@pytest.mark.contract
def test_vg_expectation_infinite_when_inverse_moment_diverges():
    j = JointVarianceGamma.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, alpha=0.5, beta=2.0,
    )
    eta = np.asarray(j.expectation_params())
    assert np.isposinf(eta[1])
    assert np.isfinite(eta[0]) and np.isfinite(eta[2])


@pytest.mark.contract
def test_d2_fisher_kernel_is_skew_duplicate():
    j = JointVarianceGamma.from_classical(
        mu=jnp.array([1.0, -0.5]),
        gamma=jnp.array([0.3, 0.2]),
        sigma=jnp.array([[1.0, 0.3], [0.3, 0.5]]),
        alpha=3.0,
        beta=2.0,
    )
    F = np.asarray(j.fisher_information())
    lam, U = np.linalg.eigh(0.5 * (F + F.T))
    scale = max(abs(lam[-1]), 1.0)
    null = U[:, np.abs(lam) <= 1e-8 * scale]
    assert null.shape[1] == 1
    d = 2
    off = 3 + 2 * d
    skew = np.zeros(F.shape[0])
    skew[off + 1] = 1.0
    skew[off + d] = -1.0
    skew /= np.sqrt(2.0)
    cosine = abs(float(null[:, 0] @ skew))
    assert cosine > 1 - 1e-6
