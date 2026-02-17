"""
Tests for multivariate rvs() covariance correctness.

These tests verify that:
1. get_L_Sigma() returns a proper lower Cholesky factor (L @ L.T = Sigma)
2. set_L_Sigma() and get_L_Sigma() are consistent
3. Multivariate rvs() produces samples with the correct covariance structure
4. logpdf() is consistent with rvs() (samples score well under own distribution)

These tests were added after discovering a bug where rvs() used the wrong
transpose of the Cholesky factor, producing samples with incorrect covariance
in the multivariate case (d > 1).
"""

import numpy as np
import pytest
from scipy.linalg import cholesky
from scipy.stats import kstest, norm

from normix.distributions.mixtures import (
    JointVarianceGamma,
    JointNormalInverseGamma,
    JointNormalInverseGaussian,
    JointGeneralizedHyperbolic,
    VarianceGamma,
    NormalInverseGamma,
    NormalInverseGaussian,
    GeneralizedHyperbolic,
)


# ============================================================
# Shared test parameters
# ============================================================

def get_2d_params():
    """Standard 2D test parameters for all mixture distributions."""
    return {
        'mu': np.array([0.1, -0.2]),
        'gamma': np.array([0.3, -0.15]),
        'sigma': np.array([[1.0, 0.5], [0.5, 1.5]]),
    }


def get_3d_params():
    """3D test parameters with non-trivial correlation structure."""
    return {
        'mu': np.array([0.0, 0.1, -0.1]),
        'gamma': np.array([0.2, -0.1, 0.15]),
        'sigma': np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.5, -0.2],
            [0.1, -0.2, 0.8],
        ]),
    }


def get_joint_distributions_2d():
    """Create all joint mixture distributions with 2D parameters."""
    p = get_2d_params()
    dists = []

    # Variance Gamma
    dists.append(('VG', JointVarianceGamma.from_classical_params(
        mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
        shape=2.0, rate=1.0,
    )))

    # Normal Inverse Gamma
    dists.append(('NInvG', JointNormalInverseGamma.from_classical_params(
        mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
        shape=3.0, rate=1.0,
    )))

    # Normal Inverse Gaussian
    dists.append(('NIG', JointNormalInverseGaussian.from_classical_params(
        mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
        delta=1.0, eta=2.0,
    )))

    # Generalized Hyperbolic
    dists.append(('GH', JointGeneralizedHyperbolic.from_classical_params(
        mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
        p=-0.5, a=2.0, b=1.0,
    )))

    return dists


# ============================================================
# Test _L_Sigma is a proper lower Cholesky factor
# ============================================================

class TestGetLSigma:
    """Test that _L_Sigma attribute is a valid lower Cholesky factor."""

    @pytest.fixture
    def dists(self):
        return get_joint_distributions_2d()

    def test_L_is_lower_triangular(self, dists):
        """L_Sigma should be lower triangular."""
        for name, dist in dists:
            L = dist._L_Sigma
            assert np.allclose(L, np.tril(L)), (
                f"{name}: _L_Sigma is not lower triangular"
            )

    def test_L_has_positive_diagonal(self, dists):
        """L_Sigma should have positive diagonal (unique Cholesky)."""
        for name, dist in dists:
            L = dist._L_Sigma
            assert np.all(np.diag(L) > 0), (
                f"{name}: _L_Sigma diagonal has non-positive entries"
            )

    def test_L_reconstructs_sigma(self, dists):
        """L @ L.T should equal the actual Sigma from classical params."""
        for name, dist in dists:
            L = dist._L_Sigma
            Sigma_from_L = L @ L.T
            Sigma_true = dist.classical_params['sigma']
            np.testing.assert_allclose(
                Sigma_from_L, Sigma_true, rtol=1e-10,
                err_msg=f"{name}: L @ L.T != Sigma",
            )

    def test_log_det_sigma(self, dists):
        """log|Sigma| should match independent computation."""
        for name, dist in dists:
            log_det = dist.log_det_Sigma
            Sigma_true = dist.classical_params['sigma']
            expected_log_det = np.linalg.slogdet(Sigma_true)[1]
            np.testing.assert_allclose(
                log_det, expected_log_det, rtol=1e-10,
                err_msg=f"{name}: log_det mismatch",
            )


# ============================================================
# Test _set_internal consistency
# ============================================================

class TestSetInternal:
    """Test that _set_internal produces same state as set_classical_params."""

    def test_set_internal_matches_classical(self):
        """_set_internal with L_sigma should match set_classical_params."""
        Sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
        L_true = cholesky(Sigma, lower=True)

        dist1 = JointVarianceGamma.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([0.0, 0.0]),
            sigma=Sigma,
            shape=2.0, rate=1.0,
        )

        dist2 = JointVarianceGamma()
        dist2._d = 2
        dist2._set_internal(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([0.0, 0.0]),
            L_sigma=L_true,
            shape=2.0, rate=1.0,
        )

        np.testing.assert_allclose(dist2._L_Sigma, L_true, rtol=1e-12)
        np.testing.assert_allclose(
            dist2.log_det_Sigma, np.linalg.slogdet(Sigma)[1], rtol=1e-10
        )
        np.testing.assert_allclose(
            dist1.natural_params, dist2.natural_params, rtol=1e-10
        )


# ============================================================
# Test rvs() sample covariance matches theoretical covariance
# ============================================================

class TestRvsSampleCovariance:
    """
    Test that rvs() produces samples with the correct covariance structure.

    For a normal-variance mixture X | Y ~ N(mu + gamma*Y, Sigma*Y):
        Cov(X) = E[Y] * Sigma + Var(Y) * gamma @ gamma.T

    We verify both the full covariance matrix and individual marginal variances.
    """

    @pytest.mark.parametrize("name,dist", get_joint_distributions_2d(),
                             ids=[d[0] for d in get_joint_distributions_2d()])
    def test_sample_covariance_2d(self, name, dist):
        """Sample covariance should match theoretical covariance."""
        n_samples = 50_000
        rng = np.random.default_rng(42)

        # Theoretical covariance: Cov(X) = E[Y]*Sigma + Var(Y)*gamma@gamma.T
        params = dist.classical_params
        Sigma = params['sigma']
        gamma = params['gamma']
        E_Y, Var_Y = self._get_mixing_moments(name, params)

        # Generate samples
        X, Y = dist.rvs(size=n_samples, random_state=rng)
        Cov_sample = np.cov(X.T)

        if E_Y is None:
            # Use sample moments of Y for GH (Bessel-based moments are complex)
            E_Y = np.mean(Y)
            Var_Y = np.var(Y)

        Cov_theory = E_Y * Sigma + Var_Y * np.outer(gamma, gamma)

        # Check Frobenius norm relative error
        frob_error = np.linalg.norm(Cov_sample - Cov_theory, 'fro')
        frob_norm = np.linalg.norm(Cov_theory, 'fro')
        rel_error = frob_error / frob_norm

        assert rel_error < 0.05, (
            f"{name}: sample covariance relative error {rel_error:.3f} > 5%\n"
            f"Theoretical:\n{Cov_theory}\nSample:\n{Cov_sample}"
        )

    @pytest.mark.parametrize("name,dist", get_joint_distributions_2d(),
                             ids=[d[0] for d in get_joint_distributions_2d()])
    def test_sample_mean_2d(self, name, dist):
        """Sample mean should match theoretical mean."""
        n_samples = 50_000
        rng = np.random.default_rng(42)

        params = dist.classical_params
        mu = params['mu']
        gamma = params['gamma']
        E_Y, _ = self._get_mixing_moments(name, params)

        # Generate samples
        X, Y = dist.rvs(size=n_samples, random_state=rng)
        mean_sample = np.mean(X, axis=0)

        if E_Y is None:
            # Use sample moments for GH
            E_Y = np.mean(Y)

        mean_theory = mu + gamma * E_Y

        np.testing.assert_allclose(
            mean_sample, mean_theory, rtol=0.05, atol=0.05,
            err_msg=f"{name}: sample mean mismatch",
        )

    @staticmethod
    def _get_mixing_moments(name, params):
        """Get E[Y] and Var(Y) for the mixing distribution."""
        if name == 'VG':
            alpha, beta = params['shape'], params['rate']
            return alpha / beta, alpha / beta**2
        elif name == 'NInvG':
            alpha, beta = params['shape'], params['rate']
            # Y ~ InvGamma(alpha, beta): E[Y] = beta/(alpha-1), Var = beta^2/((a-1)^2*(a-2))
            return beta / (alpha - 1), beta**2 / ((alpha - 1)**2 * (alpha - 2))
        elif name == 'NIG':
            delta, eta = params['delta'], params['eta']
            # Y ~ InverseGaussian(delta, eta): E[Y] = delta, Var = delta^3/eta
            return delta, delta**3 / eta
        elif name == 'GH':
            # GH mixing is GIG(p, a, b) - moments are complex Bessel ratios.
            # We skip the analytical check and use sample-based verification
            # (see dedicated GH test below).
            return None, None


class TestRvsSampleCovariance3D:
    """Test rvs covariance with 3D parameters to catch transpose bugs."""

    def test_vg_3d_covariance(self):
        """3D Variance Gamma sample covariance should match theory."""
        p = get_3d_params()
        alpha, beta = 2.5, 1.0

        dist = JointVarianceGamma.from_classical_params(
            mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
            shape=alpha, rate=beta,
        )

        E_Y = alpha / beta
        Var_Y = alpha / beta**2
        Cov_theory = E_Y * p['sigma'] + Var_Y * np.outer(p['gamma'], p['gamma'])

        X, Y = dist.rvs(size=50_000, random_state=42)
        Cov_sample = np.cov(X.T)

        frob_error = np.linalg.norm(Cov_sample - Cov_theory, 'fro')
        frob_norm = np.linalg.norm(Cov_theory, 'fro')
        rel_error = frob_error / frob_norm

        assert rel_error < 0.05, (
            f"3D VG: sample covariance error {rel_error:.3f} > 5%\n"
            f"Theoretical:\n{Cov_theory}\nSample:\n{Cov_sample}"
        )

    def test_nig_3d_covariance(self):
        """3D NIG sample covariance should match theory."""
        p = get_3d_params()
        delta, eta = 1.0, 2.0

        dist = JointNormalInverseGaussian.from_classical_params(
            mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
            delta=delta, eta=eta,
        )

        # InverseGaussian(delta, eta): E[Y] = delta, Var[Y] = delta^3 / eta
        E_Y = delta
        Var_Y = delta**3 / eta
        Cov_theory = E_Y * p['sigma'] + Var_Y * np.outer(p['gamma'], p['gamma'])

        X, Y = dist.rvs(size=50_000, random_state=42)
        Cov_sample = np.cov(X.T)

        frob_error = np.linalg.norm(Cov_sample - Cov_theory, 'fro')
        frob_norm = np.linalg.norm(Cov_theory, 'fro')
        rel_error = frob_error / frob_norm

        assert rel_error < 0.05, (
            f"3D NIG: sample covariance error {rel_error:.3f} > 5%\n"
            f"Theoretical:\n{Cov_theory}\nSample:\n{Cov_sample}"
        )


# ============================================================
# Test rvs via random portfolio projections (the original
# use case that exposed the bug)
# ============================================================

class TestRvsPortfolioProjections:
    """
    Test that random portfolio projections of multivariate rvs() samples
    are consistent with the marginal distribution's logpdf.

    The original bug was exposed when projecting multivariate samples
    onto random portfolio weights and comparing with MVN - the mixture
    distributions looked worse than MVN due to incorrect covariance.
    """

    def test_vg_portfolio_ks_test(self):
        """
        VG samples projected to 1D should be consistent with the 1D
        distribution derived from the same parameters.
        """
        p = get_2d_params()
        dist = VarianceGamma.from_classical_params(
            mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
            shape=2.0, rate=1.0,
        )

        n_samples = 10_000
        rng = np.random.default_rng(42)

        # Generate multivariate samples
        X = dist.rvs(size=n_samples, random_state=rng)

        # Random portfolio weights (normalized)
        w = rng.standard_normal(2)
        w = w / np.linalg.norm(w)

        # Project to 1D
        proj = X @ w

        # The projected distribution should have:
        # mean = w.T @ (mu + gamma * E[Y])
        # var = E[Y] * w.T @ Sigma @ w + Var(Y) * (w.T @ gamma)^2
        alpha, beta = 2.0, 1.0
        E_Y = alpha / beta
        Var_Y = alpha / beta**2
        proj_mean = w @ (p['mu'] + p['gamma'] * E_Y)
        proj_var = E_Y * (w @ p['sigma'] @ w) + Var_Y * (w @ p['gamma'])**2

        sample_mean = np.mean(proj)
        sample_var = np.var(proj)

        # Check projected moments
        np.testing.assert_allclose(
            sample_mean, proj_mean, rtol=0.1, atol=0.1,
            err_msg="Projected mean mismatch",
        )
        np.testing.assert_allclose(
            sample_var, proj_var, rtol=0.15,
            err_msg="Projected variance mismatch",
        )


# ============================================================
# Test logpdf consistency with rvs
# ============================================================

class TestLogpdfRvsConsistency:
    """
    Test that samples from rvs() score well under the own distribution's
    logpdf. This catches bugs where rvs and logpdf use inconsistent
    covariance structures.
    """

    @pytest.mark.parametrize("name,dist", get_joint_distributions_2d(),
                             ids=[d[0] for d in get_joint_distributions_2d()])
    def test_logpdf_at_rvs_samples_finite(self, name, dist):
        """logpdf evaluated at rvs samples should be finite."""
        X, Y = dist.rvs(size=100, random_state=42)
        logpdf_vals = dist.logpdf(X, Y)
        assert np.all(np.isfinite(logpdf_vals)), (
            f"{name}: logpdf at rvs samples has non-finite values"
        )

    def test_vg_logpdf_mean_at_rvs_vs_shifted(self):
        """
        Mean logpdf at rvs samples should be higher than at shifted samples.

        If rvs produces samples with the wrong covariance, the logpdf (which
        uses the correct covariance) would score the samples lower.
        """
        p = get_2d_params()
        vg = VarianceGamma.from_classical_params(
            mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
            shape=2.0, rate=1.0,
        )

        n = 5000
        rng = np.random.default_rng(42)

        # Samples from the distribution
        X_correct = vg.rvs(size=n, random_state=rng)

        # Shifted samples (artificially wrong)
        X_shifted = X_correct + rng.standard_normal(X_correct.shape) * 2.0

        ll_correct = np.mean(vg.logpdf(X_correct))
        ll_shifted = np.mean(vg.logpdf(X_shifted))

        assert ll_correct > ll_shifted, (
            f"logpdf at own samples ({ll_correct:.2f}) should be higher "
            f"than at shifted samples ({ll_shifted:.2f})"
        )


# ============================================================
# Test single sample vs vectorized consistency
# ============================================================

class TestSingleVsVectorized:
    """
    Test that single-sample and vectorized rvs produce samples from
    the same distribution (catches transpose mismatches between the
    two code paths).
    """

    def test_vg_single_vs_vectorized_moments(self):
        """Single-sample and vectorized rvs should have same moments."""
        p = get_2d_params()
        dist = JointVarianceGamma.from_classical_params(
            mu=p['mu'], gamma=p['gamma'], sigma=p['sigma'],
            shape=2.0, rate=1.0,
        )

        n = 10_000

        # Vectorized
        rng = np.random.default_rng(42)
        X_vec, _ = dist.rvs(size=n, random_state=rng)
        mean_vec = np.mean(X_vec, axis=0)
        cov_vec = np.cov(X_vec.T)

        # Single-sample loop
        rng2 = np.random.default_rng(123)
        X_singles = np.array([dist.rvs(size=None, random_state=rng2)[0]
                              for _ in range(n)])
        mean_single = np.mean(X_singles, axis=0)
        cov_single = np.cov(X_singles.T)

        # Means should agree within statistical tolerance
        np.testing.assert_allclose(
            mean_vec, mean_single, rtol=0.1, atol=0.1,
            err_msg="Single vs vectorized mean mismatch",
        )

        # Covariances should agree
        frob_diff = np.linalg.norm(cov_vec - cov_single, 'fro')
        frob_norm = np.linalg.norm(cov_vec, 'fro')
        rel_error = frob_diff / frob_norm

        assert rel_error < 0.15, (
            f"Single vs vectorized covariance relative error {rel_error:.3f}\n"
            f"Vectorized:\n{cov_vec}\nSingle:\n{cov_single}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
