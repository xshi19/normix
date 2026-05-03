"""
Tests for Variance Gamma distributions (current API).

Covers joint and marginal construction, rvs, log_prob, mean/cov,
conditional expectations, and EM fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions import JointVarianceGamma, VarianceGamma


def _joint_1d(mu=0.0, gamma=0.5, sigma=1.0, alpha=2.0, beta=1.0):
    return JointVarianceGamma.from_classical(
        mu=jnp.array([mu]), gamma=jnp.array([gamma]),
        sigma=jnp.array([[sigma]]), alpha=alpha, beta=beta,
    )


def _joint_2d():
    return JointVarianceGamma.from_classical(
        mu=jnp.array([0.0, 1.0]),
        gamma=jnp.array([0.5, -0.3]),
        sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        alpha=2.0, beta=1.0,
    )


# ============================================================
# Joint VG Tests
# ============================================================

class TestJointVarianceGamma:

    def test_construction_attributes(self):
        j = _joint_1d()
        assert j.d == 1
        np.testing.assert_allclose(np.array(j.mu), [0.0])
        np.testing.assert_allclose(np.array(j.gamma), [0.5])
        np.testing.assert_allclose(float(j.alpha), 2.0)
        np.testing.assert_allclose(float(j.beta), 1.0)

    def test_natural_params_finite(self):
        j = _joint_1d()
        theta = j.natural_params()
        assert jnp.all(jnp.isfinite(theta))

    def test_log_prob_joint_finite(self):
        j = _joint_1d()
        x = jnp.array([0.5])
        y = jnp.array(1.0)
        lp = float(j.log_prob_joint(x, y))
        assert np.isfinite(lp)

    def test_rvs_shapes(self):
        j = _joint_1d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 1)
        assert Y.shape == (100,)

    def test_rvs_2d_shapes(self):
        j = _joint_2d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 2)
        assert Y.shape == (100,)

    @pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (3.0, 0.5)])
    def test_rvs_y_mean(self, alpha, beta):
        """Y ~ Gamma(alpha, beta), so E[Y] = alpha/beta."""
        j = JointVarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), alpha=alpha, beta=beta,
        )
        _, Y = j.rvs(20000, seed=42)
        np.testing.assert_allclose(float(Y.mean()), alpha / beta, rtol=0.05)

    def test_log_prob_joint_positive_density(self):
        j = _joint_2d()
        X, Y = j.rvs(50, seed=42)
        lps = jax.vmap(lambda xy: j.log_prob_joint(xy[:2], xy[2]))(
            jnp.column_stack([X, Y[:, None]]))
        assert jnp.all(jnp.isfinite(lps))

    def test_sufficient_statistics_shape(self):
        j = _joint_2d()
        x = jnp.array([0.5, -0.3])
        y = jnp.array(0.8)
        t = j.sufficient_statistics(jnp.concatenate([x, jnp.array([y])]))
        assert t.shape[0] > 0
        assert jnp.all(jnp.isfinite(t))

    def test_log_prob_vs_log_prob_joint(self):
        """EF log_prob(concat(x,[y])) must agree with log_prob_joint(x,y)."""
        j = _joint_2d()
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-10)


# ============================================================
# Marginal VG Tests
# ============================================================

class TestVarianceGamma:

    def test_from_classical(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        assert vg._joint.d == 1

    def test_rvs_shape_1d(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = vg.rvs(100, seed=42)
        assert X.shape == (100, 1)

    def test_rvs_shape_2d(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(100, seed=42)
        assert X.shape == (100, 2)

    def test_log_prob_finite(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_mean_formula(self):
        """E[X] = μ + γ·E[Y] = μ + γ·α/β."""
        alpha, beta = 2.0, 1.0
        mu = jnp.array([0.5])
        gamma = jnp.array([0.3])
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=jnp.array([[1.0]]),
            alpha=alpha, beta=beta,
        )
        expected = mu + gamma * (alpha / beta)
        np.testing.assert_allclose(np.array(vg.mean()), np.array(expected), rtol=1e-10)

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sample_mean_matches_analytical(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.3, 0.1]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(30000, seed=42)
        sample_mean = np.array(X.mean(axis=0))
        analytical_mean = np.array(vg.mean())
        np.testing.assert_allclose(sample_mean, analytical_mean, rtol=0.05, atol=0.05)

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sample_cov_matches_analytical(self):
        """Cov[X] = E[Y]·Σ + Var[Y]·γγ^T."""
        alpha, beta = 2.0, 1.0
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=alpha, beta=beta,
        )
        X = vg.rvs(50000, seed=42)
        sample_cov = np.cov(np.array(X), rowvar=False)
        analytical_cov = np.array(vg.cov())
        np.testing.assert_allclose(sample_cov, analytical_cov, rtol=0.1)


# ============================================================
# Conditional Expectations
# ============================================================

class TestConditionalExpectationsVG:

    def test_conditional_expectations_finite(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = vg.rvs(50, seed=42)
        for i in range(5):
            cond = vg._joint.conditional_expectations(X[i])
            assert np.isfinite(float(cond['E_Y']))
            assert np.isfinite(float(cond['E_inv_Y']))
            assert np.isfinite(float(cond['E_log_Y']))

    def test_conditional_expectations_positive(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(50, seed=42)
        for i in range(5):
            cond = vg._joint.conditional_expectations(X[i])
            assert float(cond['E_Y']) > 0
            assert float(cond['E_inv_Y']) > 0


# ============================================================
# Edge Cases
# ============================================================

class TestVarianceGammaEdgeCases:

    def test_symmetric_case(self):
        """gamma=0 (symmetric) should produce finite log_prob."""
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.0, 0.0]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            alpha=2.5, beta=1.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_large_alpha(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=20.0, beta=10.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))
