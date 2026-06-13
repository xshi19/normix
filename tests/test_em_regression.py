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
from normix.fitting.eta import NormalMixtureEta
from normix.mixtures.joint import JointNormalMixture


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
        assert np.all(eigvals > 0)


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


def _consistent_eta(mu_t, gam_t, eta2, D):
    """Build eta so the exact M-step returns (mu_t, gam_t) for this D."""
    eta3 = (1.0 - D) / eta2
    eta5 = gam_t + eta2 * mu_t
    eta4 = mu_t * D + eta3 * eta5
    d = mu_t.shape[0]
    return NormalMixtureEta(
        E_inv_Y=jnp.asarray(eta2),
        E_Y=jnp.asarray(eta3),
        E_log_Y=jnp.asarray(0.0),
        E_X=eta4,
        E_X_inv_Y=eta5,
        E_XXT_inv_Y=jnp.eye(d),
    )


class TestMstepDenominatorSign:
    """B1: sign-preserving M-step denominator D = 1 - eta2*eta3."""

    @pytest.mark.parametrize("D", [-1e-12, -5e-11])
    def test_tiny_D_preserves_mu_gamma_sign(self, D):
        mu_t = jnp.array([0.5, -1.0])
        gam_t = jnp.array([2.0, 0.3])
        eta2 = 1.3
        ref = JointNormalMixture._mstep_normal_params(
            _consistent_eta(mu_t, gam_t, eta2, -1e-8))
        got = JointNormalMixture._mstep_normal_params(
            _consistent_eta(mu_t, gam_t, eta2, D))
        for a, b in zip(ref[:2], got[:2]):
            assert np.all(np.sign(np.array(a)) == np.sign(np.array(b)))

    def test_positive_D_roundoff_uses_negative_floor(self):
        """Real e_step near the Gaussian limit can yield D > 0 from cancellation."""
        key = jax.random.PRNGKey(7)
        d, n = 2, 5000
        mu_true = jnp.array([1.5, -0.7])
        A = jax.random.normal(key, (d, d)) * 0.5
        Sigma = A @ A.T + jnp.eye(d)
        X = jax.random.multivariate_normal(
            jax.random.PRNGKey(8), mu_true, Sigma, (n,))
        model = VarianceGamma.from_classical(
            mu=mu_true, gamma=jnp.array([0.01, -0.01]),
            sigma=Sigma, alpha=1e9, beta=1e9)
        eta = model.e_step(X, backend="jax")
        D = float(1.0 - eta.E_inv_Y * eta.E_Y)
        assert D > 0.0
        mu, _, _ = JointNormalMixture._mstep_normal_params(eta)
        # Pre-B1: +D in the denominator flipped mu[0] to negative (~-3.56).
        assert float(mu[0]) * float(mu_true[0]) > 0

    @pytest.mark.parametrize(
        "cls",
        [VarianceGamma, NormalInverseGamma, NormalInverseGaussian,
         GeneralizedHyperbolic],
    )
    def test_e_step_batch_D_nonpositive(self, cls):
        key = jax.random.PRNGKey(0)
        d, n = 3, 1500
        X = jax.random.multivariate_normal(
            key, jnp.zeros(d), jnp.eye(d), (n,)) * 1.2 + 0.3
        model = cls.default_init(X)
        eta = model.e_step(X, backend="jax")
        D = float(1.0 - eta.E_inv_Y * eta.E_Y)
        assert D <= 1e-9
