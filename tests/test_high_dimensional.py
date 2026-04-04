"""
Tests for high-dimensional joint distributions (dim=50).

Verifies that joint distributions work correctly in higher dimensions:
- Construction and parameter access
- rvs shapes and finiteness
- Log-likelihood computation
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions import (
    JointVarianceGamma,
    JointNormalInverseGamma,
    JointNormalInverseGaussian,
    JointGeneralizedHyperbolic,
)


def _simple_cov(d, diag=1.0, off_diag=0.2):
    Sigma = np.full((d, d), off_diag)
    np.fill_diagonal(Sigma, diag)
    return jnp.array(Sigma)


DIM = 50


@pytest.fixture
def vg_high():
    return JointVarianceGamma.from_classical(
        mu=jnp.zeros(DIM), gamma=jnp.full(DIM, 0.1),
        sigma=_simple_cov(DIM), alpha=2.0, beta=1.0,
    )


@pytest.fixture
def ninvg_high():
    return JointNormalInverseGamma.from_classical(
        mu=jnp.zeros(DIM), gamma=jnp.full(DIM, 0.1),
        sigma=_simple_cov(DIM), alpha=3.0, beta=1.0,
    )


@pytest.fixture
def nig_high():
    return JointNormalInverseGaussian.from_classical(
        mu=jnp.zeros(DIM), gamma=jnp.full(DIM, 0.1),
        sigma=_simple_cov(DIM), mu_ig=1.0, lam=1.0,
    )


@pytest.fixture
def gh_high():
    return JointGeneralizedHyperbolic.from_classical(
        mu=jnp.zeros(DIM), gamma=jnp.full(DIM, 0.1),
        sigma=_simple_cov(DIM), p=1.0, a=1.0, b=1.0,
    )


class TestHighDimVG:

    def test_rvs_shapes(self, vg_high):
        X, Y = vg_high.rvs(100, seed=42)
        assert X.shape == (100, DIM)
        assert Y.shape == (100,)

    def test_natural_params_finite(self, vg_high):
        theta = vg_high.natural_params()
        assert jnp.all(jnp.isfinite(theta))

    def test_log_prob_finite(self, vg_high):
        X, Y = vg_high.rvs(20, seed=42)
        for i in range(5):
            lp = float(vg_high.log_prob_joint(X[i], Y[i]))
            assert np.isfinite(lp), f"Non-finite log_prob at sample {i}"


class TestHighDimNInvG:

    def test_rvs_shapes(self, ninvg_high):
        X, Y = ninvg_high.rvs(100, seed=42)
        assert X.shape == (100, DIM)
        assert Y.shape == (100,)

    def test_log_prob_finite(self, ninvg_high):
        X, Y = ninvg_high.rvs(20, seed=42)
        for i in range(5):
            lp = float(ninvg_high.log_prob_joint(X[i], Y[i]))
            assert np.isfinite(lp)


class TestHighDimNIG:

    def test_rvs_shapes(self, nig_high):
        X, Y = nig_high.rvs(100, seed=42)
        assert X.shape == (100, DIM)
        assert Y.shape == (100,)

    def test_log_prob_finite(self, nig_high):
        X, Y = nig_high.rvs(20, seed=42)
        for i in range(5):
            lp = float(nig_high.log_prob_joint(X[i], Y[i]))
            assert np.isfinite(lp)


class TestHighDimGH:

    def test_rvs_shapes(self, gh_high):
        X, Y = gh_high.rvs(100, seed=42)
        assert X.shape == (100, DIM)
        assert Y.shape == (100,)

    def test_log_prob_finite(self, gh_high):
        X, Y = gh_high.rvs(20, seed=42)
        for i in range(5):
            lp = float(gh_high.log_prob_joint(X[i], Y[i]))
            assert np.isfinite(lp)

    def test_log_partition_finite(self, gh_high):
        lp = float(gh_high.log_partition())
        assert np.isfinite(lp)
