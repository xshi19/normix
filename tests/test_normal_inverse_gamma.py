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
from normix import UnivariateNormalInverseGamma


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

    @pytest.mark.slow
    @pytest.mark.stress
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

    @pytest.mark.slow
    @pytest.mark.stress
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

    @pytest.mark.parametrize("alpha,beta", [(3.0, 2.0), (1.5, 0.5), (10.0, 4.0)])
    def test_symmetric_matches_student_t(self, alpha, beta):
        r"""gamma=0: X | Y ~ N(0, sigma^2 Y), Y ~ InvGamma(alpha, beta) is
        exactly a scaled Student-t (df=2*alpha, scale=sigma*sqrt(beta/alpha)).

        Regression test: the marginal log-density's normalising integral
        2(b/a)^{p/2} K_p(sqrt(ab)) has a = gamma^T Lambda gamma = 0
        identically when gamma=0. Feeding the unfloored a into sqrt(ab)
        (fed to ``log_kv``) while the (b/a)^{p/2} ratio used a floored a
        broke the small-z cancellation and inflated the density by ~50
        orders of magnitude (see docs/tutorials/core/02_gh_family_tour.md).
        """
        from scipy import stats
        nig = UnivariateNormalInverseGamma.from_classical(
            mu=0.0, gamma=0.0, sigma=1.0, alpha=alpha, beta=beta)
        xs = jnp.linspace(-6.0, 6.0, 25)
        ours = np.array(jax.vmap(nig.pdf)(xs))
        ref = stats.t(df=2.0 * alpha, scale=np.sqrt(beta / alpha)).pdf(np.array(xs))
        np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)

    def test_symmetric_multivariate_matches_multivariate_t(self):
        r"""Multivariate analogue of ``test_symmetric_matches_student_t``:
        gamma=0 collapses NInvG to a multivariate Student-t with scale
        matrix (beta/alpha)*Sigma and df=2*alpha.
        """
        from scipy.stats import multivariate_t
        alpha, beta = 3.0, 2.0
        Sigma = jnp.array([[1.0, 0.3], [0.3, 1.0]])
        nig = NormalInverseGamma.from_classical(
            mu=jnp.zeros(2), gamma=jnp.zeros(2), sigma=Sigma, alpha=alpha, beta=beta,
        )
        ref = multivariate_t(loc=[0.0, 0.0], shape=(beta / alpha) * np.array(Sigma),
                              df=2.0 * alpha)
        rng = np.random.default_rng(0)
        pts = jnp.asarray(rng.normal(size=(10, 2)) * 2.0)
        ours = np.array(jax.vmap(nig.pdf)(pts))
        expected = ref.pdf(np.array(pts))
        np.testing.assert_allclose(ours, expected, rtol=1e-6, atol=1e-8)
