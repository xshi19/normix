"""
Tests for multivariate rvs() covariance correctness (current API).

Verifies that:
1. L_Sigma is a proper lower Cholesky factor
2. Multivariate rvs() produces samples with correct covariance structure
3. log_prob is consistent with rvs (samples score well under own distribution)
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
    VarianceGamma,
)


MU_2D = jnp.array([0.1, -0.2])
GAMMA_2D = jnp.array([0.3, -0.15])
SIGMA_2D = jnp.array([[1.0, 0.5], [0.5, 1.5]])

MU_3D = jnp.array([0.0, 0.1, -0.1])
GAMMA_3D = jnp.array([0.2, -0.1, 0.15])
SIGMA_3D = jnp.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.5, -0.2],
    [0.1, -0.2, 0.8],
])


def _make_2d_joints():
    return [
        ("VG", JointVarianceGamma.from_classical(
            mu=MU_2D, gamma=GAMMA_2D, sigma=SIGMA_2D,
            alpha=2.0, beta=1.0)),
        ("NInvG", JointNormalInverseGamma.from_classical(
            mu=MU_2D, gamma=GAMMA_2D, sigma=SIGMA_2D,
            alpha=3.0, beta=1.0)),
        ("NIG", JointNormalInverseGaussian.from_classical(
            mu=MU_2D, gamma=GAMMA_2D, sigma=SIGMA_2D,
            mu_ig=1.0, lam=2.0)),
        ("GH", JointGeneralizedHyperbolic.from_classical(
            mu=MU_2D, gamma=GAMMA_2D, sigma=SIGMA_2D,
            p=-0.5, a=2.0, b=1.0)),
    ]


# ============================================================
# L_Sigma tests
# ============================================================

class TestLSigma:

    @pytest.mark.parametrize("name,dist", _make_2d_joints(),
                             ids=[d[0] for d in _make_2d_joints()])
    def test_L_is_lower_triangular(self, name, dist):
        L = np.array(dist.L_Sigma)
        assert np.allclose(L, np.tril(L)), f"{name}: L not lower triangular"

    @pytest.mark.parametrize("name,dist", _make_2d_joints(),
                             ids=[d[0] for d in _make_2d_joints()])
    def test_L_positive_diagonal(self, name, dist):
        L = np.array(dist.L_Sigma)
        assert np.all(np.diag(L) > 0), f"{name}: L diagonal not positive"

    @pytest.mark.parametrize("name,dist", _make_2d_joints(),
                             ids=[d[0] for d in _make_2d_joints()])
    def test_L_reconstructs_sigma(self, name, dist):
        L = np.array(dist.L_Sigma)
        Sigma_from_L = L @ L.T
        np.testing.assert_allclose(Sigma_from_L, np.array(SIGMA_2D), rtol=1e-10,
                                   err_msg=f"{name}: L@L.T != Sigma")


# ============================================================
# Sample covariance tests
# ============================================================

def _mixing_moments(name, dist):
    """Return (E[Y], Var[Y]) for the mixing distribution."""
    if name == "VG":
        a, b = float(dist.alpha), float(dist.beta)
        return a / b, a / b**2
    elif name == "NInvG":
        a, b = float(dist.alpha), float(dist.beta)
        return b / (a - 1), b**2 / ((a - 1)**2 * (a - 2))
    elif name == "NIG":
        mu_ig, lam = float(dist.mu_ig), float(dist.lam)
        return mu_ig, mu_ig**3 / lam
    elif name == "GH":
        return None, None


class TestSampleCovariance:

    @pytest.mark.parametrize("name,dist", _make_2d_joints(),
                             ids=[d[0] for d in _make_2d_joints()])
    def test_sample_covariance_2d(self, name, dist):
        X, Y = dist.rvs(50000, seed=42)
        X_np = np.array(X)
        Y_np = np.array(Y)
        gamma = np.array(GAMMA_2D)
        sigma = np.array(SIGMA_2D)

        E_Y, Var_Y = _mixing_moments(name, dist)
        if E_Y is None:
            E_Y = float(Y_np.mean())
            Var_Y = float(Y_np.var())

        Cov_theory = E_Y * sigma + Var_Y * np.outer(gamma, gamma)
        Cov_sample = np.cov(X_np, rowvar=False)

        frob_error = np.linalg.norm(Cov_sample - Cov_theory, 'fro')
        frob_norm = np.linalg.norm(Cov_theory, 'fro')
        assert frob_error / frob_norm < 0.06, (
            f"{name}: sample cov error {frob_error/frob_norm:.3f}")

    @pytest.mark.parametrize("name,dist", _make_2d_joints(),
                             ids=[d[0] for d in _make_2d_joints()])
    def test_sample_mean_2d(self, name, dist):
        X, Y = dist.rvs(50000, seed=42)
        X_np = np.array(X)
        Y_np = np.array(Y)
        mu = np.array(MU_2D)
        gamma = np.array(GAMMA_2D)

        E_Y, _ = _mixing_moments(name, dist)
        if E_Y is None:
            E_Y = float(Y_np.mean())

        mean_theory = mu + gamma * E_Y
        np.testing.assert_allclose(
            X_np.mean(axis=0), mean_theory, rtol=0.05, atol=0.05,
            err_msg=f"{name}: sample mean mismatch")


# ============================================================
# 3D covariance test
# ============================================================

class TestSampleCovariance3D:

    def test_vg_3d(self):
        alpha, beta = 2.5, 1.0
        dist = JointVarianceGamma.from_classical(
            mu=MU_3D, gamma=GAMMA_3D, sigma=SIGMA_3D,
            alpha=alpha, beta=beta,
        )
        E_Y = alpha / beta
        Var_Y = alpha / beta**2
        Cov_theory = E_Y * np.array(SIGMA_3D) + Var_Y * np.outer(
            np.array(GAMMA_3D), np.array(GAMMA_3D))

        X, _ = dist.rvs(50000, seed=42)
        Cov_sample = np.cov(np.array(X), rowvar=False)

        frob_error = np.linalg.norm(Cov_sample - Cov_theory, 'fro')
        frob_norm = np.linalg.norm(Cov_theory, 'fro')
        assert frob_error / frob_norm < 0.06

    def test_nig_3d(self):
        mu_ig, lam = 1.0, 2.0
        dist = JointNormalInverseGaussian.from_classical(
            mu=MU_3D, gamma=GAMMA_3D, sigma=SIGMA_3D,
            mu_ig=mu_ig, lam=lam,
        )
        E_Y = mu_ig
        Var_Y = mu_ig**3 / lam
        Cov_theory = E_Y * np.array(SIGMA_3D) + Var_Y * np.outer(
            np.array(GAMMA_3D), np.array(GAMMA_3D))

        X, _ = dist.rvs(50000, seed=42)
        Cov_sample = np.cov(np.array(X), rowvar=False)

        frob_error = np.linalg.norm(Cov_sample - Cov_theory, 'fro')
        frob_norm = np.linalg.norm(Cov_theory, 'fro')
        assert frob_error / frob_norm < 0.06


# ============================================================
# log_prob / rvs consistency
# ============================================================

class TestLogprobRvsConsistency:

    @pytest.mark.parametrize("name,dist", _make_2d_joints(),
                             ids=[d[0] for d in _make_2d_joints()])
    def test_logprob_at_rvs_finite(self, name, dist):
        X, Y = dist.rvs(100, seed=42)
        lps = jax.vmap(lambda xy: dist.log_prob_joint(xy[:2], xy[2]))(
            jnp.column_stack([X, Y[:, None]]))
        assert jnp.all(jnp.isfinite(lps)), f"{name}: non-finite log_prob at rvs"

    def test_own_samples_score_higher(self):
        vg = VarianceGamma.from_classical(
            mu=MU_2D, gamma=GAMMA_2D, sigma=SIGMA_2D,
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(5000, seed=42)
        X_shift = X + jax.random.normal(jax.random.PRNGKey(99), X.shape) * 2.0
        ll_correct = float(jax.vmap(vg.log_prob)(X).mean())
        ll_shifted = float(jax.vmap(vg.log_prob)(X_shift).mean())
        assert ll_correct > ll_shifted
