"""
Tests for high-dimensional joint distributions (dim=100).

These tests verify that the joint distributions work correctly in high dimensions,
particularly testing:
- Parameter roundtrip consistency
- Fitting from generated samples
- Log-likelihood computation
"""

import numpy as np
import pytest

from normix.distributions.mixtures import (
    JointVarianceGamma,
    JointNormalInverseGamma,
    JointNormalInverseGaussian,
    JointGeneralizedHyperbolic,
)


def create_simple_covariance(d: int, diag: float = 1.0, off_diag: float = 0.2) -> np.ndarray:
    """
    Create a simple positive definite covariance matrix.

    Parameters
    ----------
    d : int
        Dimension of the matrix.
    diag : float
        Value on the diagonal (default 1.0).
    off_diag : float
        Value for off-diagonal elements (default 0.2).

    Returns
    -------
    Sigma : ndarray
        Positive definite covariance matrix, shape (d, d).
    """
    Sigma = np.full((d, d), off_diag)
    np.fill_diagonal(Sigma, diag)
    return Sigma


# ============================================================
# High-dimensional JointVarianceGamma Tests
# ============================================================

class TestJointVarianceGammaHighDim:
    """Test JointVarianceGamma with high dimensional X (dim=100)."""

    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def params(self, dim):
        """Test parameters for high-dimensional JointVarianceGamma."""
        return {
            'mu': np.zeros(dim),
            'gamma': np.full(dim, 0.1),
            'sigma': create_simple_covariance(dim),
            'shape': 2.0,
            'rate': 1.0
        }

    def test_parameter_roundtrip(self, params, dim):
        """Test classical -> natural -> classical roundtrip."""
        # Create from classical
        dist = JointVarianceGamma.from_classical_params(**params)

        # Get natural params
        theta = dist.get_natural_params()

        # Create from natural
        dist2 = JointVarianceGamma(d=dim)
        dist2.set_natural_params(theta)

        # Get classical back
        recovered = dist2.get_classical_params()

        # Compare
        np.testing.assert_allclose(
            recovered['mu'], params['mu'], rtol=1e-8,
            err_msg="mu mismatch"
        )
        np.testing.assert_allclose(
            recovered['gamma'], params['gamma'], rtol=1e-8,
            err_msg="gamma mismatch"
        )
        np.testing.assert_allclose(
            recovered['sigma'], params['sigma'], rtol=1e-8,
            err_msg="sigma mismatch"
        )
        np.testing.assert_allclose(
            recovered['shape'], params['shape'], rtol=1e-8,
            err_msg="shape mismatch"
        )
        np.testing.assert_allclose(
            recovered['rate'], params['rate'], rtol=1e-8,
            err_msg="rate mismatch"
        )

    def test_fit_from_samples(self, params, dim):
        """Test fitting from generated samples."""
        n_samples = 2000

        # Create true distribution
        true_dist = JointVarianceGamma.from_classical_params(**params)

        # Generate samples
        X, Y = true_dist.rvs(size=n_samples, random_state=42)

        # Fit new distribution
        fitted = JointVarianceGamma(d=dim).fit(X, Y)

        # Check fitted parameters are reasonable
        fitted_params = fitted.get_classical_params()

        # Check mu (should be close to 0)
        np.testing.assert_allclose(
            fitted_params['mu'], params['mu'], atol=0.2,
            err_msg="Fitted mu not close to true mu"
        )

        # Check shape and rate (with wider tolerance for high dim)
        np.testing.assert_allclose(
            fitted_params['shape'], params['shape'], rtol=0.2,
            err_msg="Fitted shape not close to true shape"
        )
        np.testing.assert_allclose(
            fitted_params['rate'], params['rate'], rtol=0.2,
            err_msg="Fitted rate not close to true rate"
        )

    def test_log_likelihood_finite(self, params, dim):
        """Test that log-likelihood is finite for high-dim data."""
        n_samples = 500

        dist = JointVarianceGamma.from_classical_params(**params)
        X, Y = dist.rvs(size=n_samples, random_state=42)

        # Compute log-likelihood
        logpdf = dist.logpdf(X, Y)

        assert np.all(np.isfinite(logpdf)), "Log-likelihood should be finite"
        assert np.mean(logpdf) < 0, "Log-likelihood should generally be negative"


# ============================================================
# High-dimensional JointNormalInverseGamma Tests
# ============================================================

class TestJointNormalInverseGammaHighDim:
    """Test JointNormalInverseGamma with high dimensional X (dim=100)."""

    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def params(self, dim):
        """Test parameters for high-dimensional JointNormalInverseGamma."""
        return {
            'mu': np.zeros(dim),
            'gamma': np.full(dim, 0.1),
            'sigma': create_simple_covariance(dim),
            'shape': 3.0,  # Need α > 1 for finite mean
            'rate': 1.0
        }

    def test_parameter_roundtrip(self, params, dim):
        """Test classical -> natural -> classical roundtrip."""
        dist = JointNormalInverseGamma.from_classical_params(**params)
        theta = dist.get_natural_params()

        dist2 = JointNormalInverseGamma(d=dim)
        dist2.set_natural_params(theta)
        recovered = dist2.get_classical_params()

        np.testing.assert_allclose(
            recovered['mu'], params['mu'], rtol=1e-8,
            err_msg="mu mismatch"
        )
        np.testing.assert_allclose(
            recovered['gamma'], params['gamma'], rtol=1e-8,
            err_msg="gamma mismatch"
        )
        np.testing.assert_allclose(
            recovered['sigma'], params['sigma'], rtol=1e-8,
            err_msg="sigma mismatch"
        )
        np.testing.assert_allclose(
            recovered['shape'], params['shape'], rtol=1e-8,
            err_msg="shape mismatch"
        )
        np.testing.assert_allclose(
            recovered['rate'], params['rate'], rtol=1e-8,
            err_msg="rate mismatch"
        )

    def test_fit_from_samples(self, params, dim):
        """Test fitting from generated samples."""
        n_samples = 2000

        true_dist = JointNormalInverseGamma.from_classical_params(**params)
        X, Y = true_dist.rvs(size=n_samples, random_state=42)

        fitted = JointNormalInverseGamma(d=dim).fit(X, Y)
        fitted_params = fitted.get_classical_params()

        np.testing.assert_allclose(
            fitted_params['mu'], params['mu'], atol=0.2,
            err_msg="Fitted mu not close to true mu"
        )
        np.testing.assert_allclose(
            fitted_params['shape'], params['shape'], rtol=0.25,
            err_msg="Fitted shape not close to true shape"
        )

    def test_log_likelihood_finite(self, params, dim):
        """Test that log-likelihood is finite for high-dim data."""
        n_samples = 500

        dist = JointNormalInverseGamma.from_classical_params(**params)
        X, Y = dist.rvs(size=n_samples, random_state=42)

        logpdf = dist.logpdf(X, Y)

        assert np.all(np.isfinite(logpdf)), "Log-likelihood should be finite"


# ============================================================
# High-dimensional JointNormalInverseGaussian Tests
# ============================================================

class TestJointNormalInverseGaussianHighDim:
    """Test JointNormalInverseGaussian with high dimensional X (dim=100)."""

    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def params(self, dim):
        """Test parameters for high-dimensional JointNormalInverseGaussian."""
        return {
            'mu': np.zeros(dim),
            'gamma': np.full(dim, 0.1),
            'sigma': create_simple_covariance(dim),
            'delta': 1.0,
            'eta': 1.0
        }

    def test_parameter_roundtrip(self, params, dim):
        """Test classical -> natural -> classical roundtrip."""
        dist = JointNormalInverseGaussian.from_classical_params(**params)
        theta = dist.get_natural_params()

        dist2 = JointNormalInverseGaussian(d=dim)
        dist2.set_natural_params(theta)
        recovered = dist2.get_classical_params()

        np.testing.assert_allclose(
            recovered['mu'], params['mu'], rtol=1e-8,
            err_msg="mu mismatch"
        )
        np.testing.assert_allclose(
            recovered['gamma'], params['gamma'], rtol=1e-8,
            err_msg="gamma mismatch"
        )
        np.testing.assert_allclose(
            recovered['sigma'], params['sigma'], rtol=1e-8,
            err_msg="sigma mismatch"
        )
        np.testing.assert_allclose(
            recovered['delta'], params['delta'], rtol=1e-8,
            err_msg="delta mismatch"
        )
        np.testing.assert_allclose(
            recovered['eta'], params['eta'], rtol=1e-8,
            err_msg="eta mismatch"
        )

    def test_fit_from_samples(self, params, dim):
        """Test fitting from generated samples."""
        n_samples = 2000

        true_dist = JointNormalInverseGaussian.from_classical_params(**params)
        X, Y = true_dist.rvs(size=n_samples, random_state=42)

        fitted = JointNormalInverseGaussian(d=dim).fit(X, Y)
        fitted_params = fitted.get_classical_params()

        np.testing.assert_allclose(
            fitted_params['mu'], params['mu'], atol=0.2,
            err_msg="Fitted mu not close to true mu"
        )
        np.testing.assert_allclose(
            fitted_params['delta'], params['delta'], rtol=0.25,
            err_msg="Fitted delta not close to true delta"
        )

    def test_log_likelihood_finite(self, params, dim):
        """Test that log-likelihood is finite for high-dim data."""
        n_samples = 500

        dist = JointNormalInverseGaussian.from_classical_params(**params)
        X, Y = dist.rvs(size=n_samples, random_state=42)

        logpdf = dist.logpdf(X, Y)

        assert np.all(np.isfinite(logpdf)), "Log-likelihood should be finite"


# ============================================================
# High-dimensional JointGeneralizedHyperbolic Tests
# ============================================================

class TestJointGeneralizedHyperbolicHighDim:
    """Test JointGeneralizedHyperbolic with high dimensional X (dim=100)."""

    @pytest.fixture
    def dim(self):
        return 100

    @pytest.fixture
    def params(self, dim):
        """Test parameters for high-dimensional JointGeneralizedHyperbolic."""
        return {
            'mu': np.zeros(dim),
            'gamma': np.full(dim, 0.1),
            'sigma': create_simple_covariance(dim),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        }

    def test_parameter_roundtrip(self, params, dim):
        """Test classical -> natural -> classical roundtrip."""
        dist = JointGeneralizedHyperbolic.from_classical_params(**params)
        theta = dist.get_natural_params()

        dist2 = JointGeneralizedHyperbolic(d=dim)
        dist2.set_natural_params(theta)
        recovered = dist2.get_classical_params()

        np.testing.assert_allclose(
            recovered['mu'], params['mu'], rtol=1e-8,
            err_msg="mu mismatch"
        )
        np.testing.assert_allclose(
            recovered['gamma'], params['gamma'], rtol=1e-8,
            err_msg="gamma mismatch"
        )
        np.testing.assert_allclose(
            recovered['sigma'], params['sigma'], rtol=1e-8,
            err_msg="sigma mismatch"
        )
        np.testing.assert_allclose(
            recovered['p'], params['p'], rtol=1e-8,
            err_msg="p mismatch"
        )
        np.testing.assert_allclose(
            recovered['a'], params['a'], rtol=1e-8,
            err_msg="a mismatch"
        )
        np.testing.assert_allclose(
            recovered['b'], params['b'], rtol=1e-8,
            err_msg="b mismatch"
        )

    def test_fit_from_samples(self, params, dim):
        """Test fitting from generated samples."""
        n_samples = 2000

        true_dist = JointGeneralizedHyperbolic.from_classical_params(**params)
        X, Y = true_dist.rvs(size=n_samples, random_state=42)

        fitted = JointGeneralizedHyperbolic(d=dim).fit(X, Y)
        fitted_params = fitted.get_classical_params()

        np.testing.assert_allclose(
            fitted_params['mu'], params['mu'], atol=0.2,
            err_msg="Fitted mu not close to true mu"
        )

    def test_log_likelihood_finite(self, params, dim):
        """Test that log-likelihood is finite for high-dim data."""
        n_samples = 500

        dist = JointGeneralizedHyperbolic.from_classical_params(**params)
        X, Y = dist.rvs(size=n_samples, random_state=42)

        logpdf = dist.logpdf(X, Y)

        assert np.all(np.isfinite(logpdf)), "Log-likelihood should be finite"


# ============================================================
# Performance Tests
# ============================================================

class TestHighDimPerformance:
    """Test that high-dimensional operations complete in reasonable time."""

    @pytest.fixture
    def dim(self):
        return 100

    def test_variance_gamma_sampling_performance(self, dim):
        """Test that sampling 1000 samples completes quickly."""
        params = {
            'mu': np.zeros(dim),
            'gamma': np.full(dim, 0.1),
            'sigma': create_simple_covariance(dim),
            'shape': 2.0,
            'rate': 1.0
        }
        dist = JointVarianceGamma.from_classical_params(**params)

        # This should complete in a few seconds
        X, Y = dist.rvs(size=1000, random_state=42)

        assert X.shape == (1000, dim)
        assert Y.shape == (1000,)

    def test_gh_log_partition_computation(self, dim):
        """Test that log partition function computes correctly in high dim."""
        params = {
            'mu': np.zeros(dim),
            'gamma': np.full(dim, 0.1),
            'sigma': create_simple_covariance(dim),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        }
        dist = JointGeneralizedHyperbolic.from_classical_params(**params)

        theta = dist.get_natural_params()
        log_partition = dist._log_partition(theta)

        assert np.isfinite(log_partition), "Log partition should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
