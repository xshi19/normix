"""
EM algorithm regression tests (current API).

Verify that EM fitting produces valid, finite results with reasonable
log-likelihoods for all mixture distribution families.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions.variance_gamma import VarianceGamma
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic


class TestVGEMRegression:

    def test_vg_em_1d(self):
        true = VarianceGamma.from_classical(
            mu=jnp.array([0.5]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), atol=0.3)
        assert result.n_iter <= 50

    def test_vg_em_2d(self):
        true = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 1.0]),
            gamma=jnp.array([0.2, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=3.0, beta=2.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0), "Sigma not PD"


class TestNInvGEMRegression:

    def test_ninvg_em_1d(self):
        true = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=3.0, beta=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        assert float(fitted._joint.alpha) > 1.0
        assert result.n_iter <= 50

    def test_ninvg_em_2d(self):
        true = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.5]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
            alpha=4.0, beta=2.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)


class TestNIGEMRegression:

    def test_nig_em_1d(self):
        true = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), mu_ig=1.0, lam=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        assert float(fitted._joint.mu_ig) > 0
        assert float(fitted._joint.lam) > 0
        assert result.n_iter <= 50

    def test_nig_em_2d(self):
        true = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0, 0.5]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
            mu_ig=1.0, lam=2.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)


class TestGHEMRegression:

    def test_gh_em_1d_det_sigma_one(self):
        true = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), p=1.0, a=1.0, b=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=30, tol=1e-4, verbose=0,
                          regularization='det_sigma_one',
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        assert float(fitted._joint.a) > 0
        assert float(fitted._joint.b) > 0
        ll = float(fitted.marginal_log_likelihood(X))
        assert np.isfinite(ll)

    def test_gh_em_2d_det_sigma_one(self):
        true = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.5]),
            gamma=jnp.array([0.2, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            p=-0.5, a=1.0, b=1.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=30, tol=1e-4, verbose=0,
                          regularization='det_sigma_one',
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)
