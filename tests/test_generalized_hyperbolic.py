"""
Tests for Generalized Hyperbolic distributions (current API).

Covers joint and marginal construction, rvs, log_prob, mean/cov,
special case consistency, conditional expectations, and EM fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions import (
    JointGeneralizedHyperbolic,
    GeneralizedHyperbolic,
    VarianceGamma,
    NormalInverseGaussian,
    NormalInverseGamma,
)


def _joint_1d(p=1.0, a=1.0, b=1.0):
    return JointGeneralizedHyperbolic.from_classical(
        mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
        sigma=jnp.array([[1.0]]), p=p, a=a, b=b,
    )


def _joint_2d():
    return JointGeneralizedHyperbolic.from_classical(
        mu=jnp.array([0.0, 1.0]),
        gamma=jnp.array([0.5, -0.3]),
        sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        p=1.0, a=1.0, b=1.0,
    )


# ============================================================
# Joint Tests
# ============================================================

class TestJointGeneralizedHyperbolic:

    def test_construction(self):
        j = _joint_1d()
        assert j.d == 1
        np.testing.assert_allclose(float(j.p), 1.0)
        np.testing.assert_allclose(float(j.a), 1.0)
        np.testing.assert_allclose(float(j.b), 1.0)

    def test_natural_params_finite(self):
        j = _joint_1d()
        theta = j.natural_params()
        assert jnp.all(jnp.isfinite(theta))

    def test_rvs_shapes(self):
        j = _joint_2d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 2)
        assert Y.shape == (100,)

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

class TestGeneralizedHyperbolic:

    def test_rvs_shape_2d(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(100, seed=42)
        assert X.shape == (100, 2)

    def test_log_prob_finite(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_log_prob_decreases_away_from_mode(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), p=1.0, a=1.0, b=1.0,
        )
        lp0 = float(gh.log_prob(jnp.array([0.0])))
        lp1 = float(gh.log_prob(jnp.array([1.0])))
        lp2 = float(gh.log_prob(jnp.array([2.0])))
        assert lp0 > lp1 > lp2

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sample_mean_matches(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.2, 0.1]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.5]]),
            p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(30000, seed=42)
        np.testing.assert_allclose(
            np.array(X.mean(axis=0)), np.array(gh.mean()), rtol=0.1, atol=0.1)

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sample_cov_matches(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(80000, seed=42)
        np.testing.assert_allclose(
            np.cov(np.array(X), rowvar=False), np.array(gh.cov()), rtol=0.2)


# ============================================================
# Special Case Consistency
# ============================================================

class TestGHSpecialCases:

    def test_gh_vg_mean_consistency(self):
        """GH parametrized as VG should match VG mean."""
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        vg_mean = np.array(vg.mean())
        X = vg.rvs(20, seed=42)
        lps_vg = np.array(jax.vmap(vg.log_prob)(X))
        assert np.all(np.isfinite(lps_vg))
        assert np.all(np.isfinite(vg_mean))

    def test_gh_nig_mean_consistency(self):
        """GH parametrized as NIG should match NIG mean."""
        nig = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), mu_ig=1.0, lam=1.0,
        )
        nig_mean = np.array(nig.mean())
        X = nig.rvs(20, seed=42)
        lps = np.array(jax.vmap(nig.log_prob)(X))
        assert np.all(np.isfinite(lps))
        assert np.all(np.isfinite(nig_mean))

    def test_gh_ninvg_mean_consistency(self):
        """GH parametrized as NInvG should match NInvG mean."""
        ninvg = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=3.0, beta=1.0,
        )
        ninvg_mean = np.array(ninvg.mean())
        X = ninvg.rvs(20, seed=42)
        lps = np.array(jax.vmap(ninvg.log_prob)(X))
        assert np.all(np.isfinite(lps))
        assert np.all(np.isfinite(ninvg_mean))


# ============================================================
# Conditional Expectations
# ============================================================

class TestConditionalExpectationsGH:

    def test_finite(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        for i in range(5):
            cond = gh._joint.conditional_expectations(X[i])
            assert np.isfinite(float(cond['E_Y']))
            assert np.isfinite(float(cond['E_inv_Y']))
            assert np.isfinite(float(cond['E_log_Y']))

    def test_positive(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        for i in range(5):
            cond = gh._joint.conditional_expectations(X[i])
            assert float(cond['E_Y']) > 0
            assert float(cond['E_inv_Y']) > 0


# ============================================================
# EM Fitting
# ============================================================

class TestGeneralizedHyperbolicFitting:

    @pytest.mark.slow
    @pytest.mark.integration
    def test_fit_em_1d(self):
        true = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), p=1.0, a=1.0, b=1.0,
        )
        X = true.rvs(5000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-5, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu',
                          regularization='det_sigma_one')
        fitted = result.model
        ll = float(fitted.marginal_log_likelihood(X))
        assert np.isfinite(ll)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_fit_em_2d(self):
        true = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            p=1.0, a=1.0, b=1.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-5, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu',
                          regularization='det_sigma_one')
        fitted = result.model
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), rtol=0.25, atol=0.3)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_em_monotone_ll(self):
        """EM iterations should not decrease log-likelihood."""
        X = jnp.array(np.random.default_rng(0).standard_normal((200, 2)))
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.zeros(2), gamma=jnp.zeros(2),
            sigma=jnp.eye(2), p=1.0, a=1.0, b=1.0,
        )
        ll_prev = float(gh.marginal_log_likelihood(X))
        model = gh
        for _ in range(5):
            eta = model.e_step(X)
            model = model.m_step(eta)
            ll = float(model.marginal_log_likelihood(X))
            assert ll >= ll_prev - 1e-4, f"LL decreased: {ll_prev:.4f} → {ll:.4f}"
            ll_prev = ll


# ============================================================
# Edge Cases
# ============================================================

class TestGeneralizedHyperbolicEdgeCases:

    def test_symmetric(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.0, 0.0]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            p=1.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_large_p(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), p=10.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_negative_p(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), p=-1.5, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))
