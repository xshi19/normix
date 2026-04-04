"""
Tests for Normal Inverse Gamma distributions (current API).

Covers joint and marginal construction, rvs, log_prob, mean/cov,
conditional expectations, and EM fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions import JointNormalInverseGamma, NormalInverseGamma


def _joint_1d(alpha=3.0, beta=1.0):
    return JointNormalInverseGamma.from_classical(
        mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
        sigma=jnp.array([[1.0]]), alpha=alpha, beta=beta,
    )


def _joint_2d():
    return JointNormalInverseGamma.from_classical(
        mu=jnp.array([0.0, 1.0]),
        gamma=jnp.array([0.5, -0.3]),
        sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        alpha=3.0, beta=1.0,
    )


# ============================================================
# Joint Tests
# ============================================================

class TestJointNormalInverseGamma:

    def test_construction(self):
        j = _joint_1d()
        assert j.d == 1
        np.testing.assert_allclose(float(j.alpha), 3.0)
        np.testing.assert_allclose(float(j.beta), 1.0)

    def test_natural_params_finite(self):
        j = _joint_1d()
        theta = j.natural_params()
        assert jnp.all(jnp.isfinite(theta))

    def test_rvs_shapes_1d(self):
        j = _joint_1d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 1)
        assert Y.shape == (100,)

    def test_rvs_shapes_2d(self):
        j = _joint_2d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 2)
        assert Y.shape == (100,)

    def test_rvs_y_mean(self):
        """Y ~ InvGamma(α,β), so E[Y] = β/(α-1)."""
        alpha, beta = 4.0, 2.0
        j = JointNormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), alpha=alpha, beta=beta,
        )
        _, Y = j.rvs(30000, seed=42)
        np.testing.assert_allclose(float(Y.mean()), beta / (alpha - 1), rtol=0.05)

    def test_log_prob_joint_finite(self):
        j = _joint_2d()
        X, Y = j.rvs(50, seed=42)
        lps = jax.vmap(lambda xy: j.log_prob_joint(xy[:2], xy[2]))(
            jnp.column_stack([X, Y[:, None]]))
        assert jnp.all(jnp.isfinite(lps))

    def test_log_prob_vs_log_prob_joint(self):
        j = _joint_2d()
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-10)


# ============================================================
# Marginal Tests
# ============================================================

class TestNormalInverseGamma:

    def test_rvs_shape(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=3.0, beta=1.0,
        )
        X = nig.rvs(100, seed=42)
        assert X.shape == (100, 2)

    def test_log_prob_finite(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=3.0, beta=1.0,
        )
        X = nig.rvs(20, seed=42)
        lps = jax.vmap(nig.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_mean_formula(self):
        """E[X] = μ + γ·E[Y] = μ + γ·β/(α-1)."""
        alpha, beta = 4.0, 2.0
        mu = jnp.array([0.5])
        gamma = jnp.array([0.3])
        nig = NormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=jnp.array([[1.0]]),
            alpha=alpha, beta=beta,
        )
        expected = mu + gamma * (beta / (alpha - 1))
        np.testing.assert_allclose(np.array(nig.mean()), np.array(expected), rtol=1e-10)

    def test_sample_mean_matches(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.3, 0.1]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=4.0, beta=1.0,
        )
        X = nig.rvs(30000, seed=42)
        np.testing.assert_allclose(
            np.array(X.mean(axis=0)), np.array(nig.mean()), rtol=0.1, atol=0.1)

    def test_sample_cov_matches(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=4.0, beta=1.0,
        )
        X = nig.rvs(50000, seed=42)
        np.testing.assert_allclose(
            np.cov(np.array(X), rowvar=False), np.array(nig.cov()), rtol=0.15)


# ============================================================
# Conditional Expectations
# ============================================================

class TestConditionalExpectationsNInvG:

    def test_finite(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=3.0, beta=1.0,
        )
        X = nig.rvs(20, seed=42)
        for i in range(5):
            cond = nig._joint.conditional_expectations(X[i])
            assert np.isfinite(float(cond['E_Y']))
            assert np.isfinite(float(cond['E_inv_Y']))
            assert np.isfinite(float(cond['E_log_Y']))

    def test_positive(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            alpha=3.0, beta=1.0,
        )
        X = nig.rvs(20, seed=42)
        for i in range(5):
            cond = nig._joint.conditional_expectations(X[i])
            assert float(cond['E_Y']) > 0
            assert float(cond['E_inv_Y']) > 0


# ============================================================
# EM Fitting
# ============================================================

class TestNormalInverseGammaFitting:

    def test_fit_em_1d(self):
        true = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), alpha=3.0, beta=1.0,
        )
        X = true.rvs(5000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-5, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), rtol=0.2, atol=0.2)

    def test_fit_em_2d(self):
        true = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=3.5, beta=1.0,
        )
        X = true.rvs(5000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-5, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), rtol=0.25, atol=0.3)


# ============================================================
# Edge Cases
# ============================================================

class TestNormalInverseGammaEdgeCases:

    def test_symmetric(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.0, 0.0]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            alpha=3.0, beta=1.0,
        )
        X = nig.rvs(20, seed=42)
        lps = jax.vmap(nig.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_large_alpha(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=20.0, beta=10.0,
        )
        X = nig.rvs(20, seed=42)
        lps = jax.vmap(nig.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))
