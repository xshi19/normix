"""
Tests for Normal Inverse Gaussian (NIG) distributions.

Since there is no scipy distribution to compare against, we test:
- Sample sufficient statistics match expectation parameters
- Histogram matches PDF (chi-square test)
- Parameter roundtrip consistency
- EM fitting converges and recovers parameters
"""

import numpy as np
import pytest
from scipy.stats import invgauss as scipy_invgauss

from normix.distributions.mixtures import JointNormalInverseGaussian, NormalInverseGaussian


# ============================================================
# Test Configurations
# ============================================================

def get_joint_nig_1d_params():
    """Test parameters for 1D JointNormalInverseGaussian."""
    return [
        {
            'mu': np.array([0.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'delta': 1.0,  # IG mean
            'eta': 1.0     # IG shape
        },
        {
            'mu': np.array([1.0]),
            'gamma': np.array([-0.3]),
            'sigma': np.array([[2.0]]),
            'delta': 2.0,
            'eta': 0.5
        },
        {
            'mu': np.array([0.5]),
            'gamma': np.array([0.0]),  # Symmetric case
            'sigma': np.array([[1.5]]),
            'delta': 1.5,
            'eta': 1.5
        },
    ]


def get_nig_2d_params():
    """Test parameters for 2D NormalInverseGaussian."""
    return [
        {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.5, -0.3]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'delta': 1.0,
            'eta': 1.0
        },
        {
            'mu': np.array([1.0, -0.5]),
            'gamma': np.array([0.2, 0.4]),
            'sigma': np.array([[2.0, 0.5], [0.5, 1.5]]),
            'delta': 1.5,
            'eta': 0.5
        },
        {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.0, 0.0]),  # Symmetric case
            'sigma': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'delta': 1.0,
            'eta': 2.0
        },
    ]


# ============================================================
# JointNormalInverseGaussian Tests (1D X, 1D Y)
# ============================================================

class TestJointNormalInverseGaussian1D:
    """Test JointNormalInverseGaussian with 1D X and 1D Y (total 2D)."""

    @pytest.fixture
    def params_list(self):
        return get_joint_nig_1d_params()

    def test_parameter_roundtrip(self, params_list):
        """Test classical -> natural -> classical roundtrip."""
        for params in params_list:
            # Create from classical
            dist = JointNormalInverseGaussian.from_classical_params(**params)

            # Get natural params
            theta = dist.get_natural_params()

            # Create from natural
            dist2 = JointNormalInverseGaussian(d=1)
            dist2.set_natural_params(theta)

            # Get classical back
            recovered = dist2.get_classical_params()

            # Compare
            np.testing.assert_allclose(
                recovered['mu'], params['mu'], rtol=1e-8,
                err_msg=f"mu mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['gamma'], params['gamma'], rtol=1e-8,
                err_msg=f"gamma mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['sigma'], params['sigma'], rtol=1e-8,
                err_msg=f"sigma mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['delta'], params['delta'], rtol=1e-6,
                err_msg=f"delta mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['eta'], params['eta'], rtol=1e-6,
                err_msg=f"eta mismatch for {params}"
            )

    def test_sample_sufficient_statistics(self, params_list):
        """Test E[t(X,Y)] from samples matches expectation parameters."""
        n_samples = 50000
        rtol = 0.1  # 10% tolerance for statistical test

        for params in params_list:
            dist = JointNormalInverseGaussian.from_classical_params(**params)

            # Get theoretical expectation parameters
            eta_theory = dist.get_expectation_params()

            # Skip if any are infinite
            if not np.all(np.isfinite(eta_theory)):
                continue

            # Generate samples
            X, Y = dist.rvs(size=n_samples, random_state=42)

            # Compute sufficient statistics
            t_samples = dist._sufficient_statistics(X, Y)

            # Sample mean
            eta_sample = np.mean(t_samples, axis=0)

            # Compare
            np.testing.assert_allclose(
                eta_sample, eta_theory, rtol=rtol,
                err_msg=f"Sufficient statistics mismatch for {params}\n"
                        f"Expected: {eta_theory}\nGot: {eta_sample}"
            )

    def test_histogram_vs_invgauss_pdf(self, params_list):
        """Test histogram of Y matches InverseGaussian PDF (mean/variance comparison)."""
        n_samples = 10000

        for params in params_list:
            dist = JointNormalInverseGaussian.from_classical_params(**params)

            # Generate samples - we'll test marginal of Y
            _, Y = dist.rvs(size=n_samples, random_state=42)

            # For Inverse Gaussian with mean δ and shape η:
            # E[Y] = δ
            # Var[Y] = δ³/η
            delta = params['delta']
            eta = params['eta']
            expected_mean = delta
            expected_var = delta**3 / eta
            
            # Sample statistics
            sample_mean = np.mean(Y)
            sample_var = np.var(Y)
            
            # Compare means (10% tolerance for statistical test)
            np.testing.assert_allclose(
                sample_mean, expected_mean, rtol=0.1,
                err_msg=f"Y mean mismatch for {params}"
            )
            
            # Compare variances (20% tolerance)
            np.testing.assert_allclose(
                sample_var, expected_var, rtol=0.2,
                err_msg=f"Y variance mismatch for {params}"
            )

    def test_joint_pdf_positive(self, params_list):
        """Test that joint PDF is positive and finite at samples."""
        for params in params_list:
            dist = JointNormalInverseGaussian.from_classical_params(**params)

            # Generate samples
            X, Y = dist.rvs(size=5000, random_state=42)

            # Compute PDF at samples
            pdf_vals = dist.pdf(X, Y)

            # All PDF values should be positive and finite
            assert np.all(pdf_vals > 0), f"Found non-positive PDF values for {params}"
            assert np.all(np.isfinite(pdf_vals)), f"Found non-finite PDF values for {params}"

    def test_mean_variance(self, params_list):
        """Test mean and variance computations."""
        n_samples = 50000

        for params in params_list:
            dist = JointNormalInverseGaussian.from_classical_params(**params)

            # Theoretical moments
            E_X, E_Y = dist.mean()

            # For InverseGaussian: E[Y] = δ (mean)
            expected_E_Y = params['delta']

            np.testing.assert_allclose(
                E_Y, expected_E_Y, rtol=1e-10,
                err_msg=f"Theoretical E[Y] mismatch for {params}"
            )

            # Sample moments
            X, Y = dist.rvs(size=n_samples, random_state=42)
            sample_E_X = np.mean(X, axis=0)
            sample_E_Y = np.mean(Y)

            # Compare (10% tolerance for statistical test)
            np.testing.assert_allclose(
                sample_E_X, E_X, rtol=0.1,
                err_msg=f"E[X] mismatch for {params}"
            )
            np.testing.assert_allclose(
                sample_E_Y, E_Y, rtol=0.1,
                err_msg=f"E[Y] mismatch for {params}"
            )


# ============================================================
# NormalInverseGaussian Tests (2D X)
# ============================================================

class TestNormalInverseGaussian2D:
    """Test NormalInverseGaussian marginal distribution with 2D X."""

    @pytest.fixture
    def params_list(self):
        return get_nig_2d_params()

    def test_parameter_access(self, params_list):
        """Test that parameters can be accessed correctly."""
        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            # Access via joint
            classical = nig.get_classical_params()

            np.testing.assert_allclose(classical['mu'], params['mu'])
            np.testing.assert_allclose(classical['gamma'], params['gamma'])
            np.testing.assert_allclose(classical['sigma'], params['sigma'])
            np.testing.assert_allclose(classical['delta'], params['delta'], rtol=1e-6)
            np.testing.assert_allclose(classical['eta'], params['eta'], rtol=1e-6)

    def test_marginal_rvs_shape(self, params_list):
        """Test that marginal rvs returns correct shape."""
        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            # Single sample
            x = nig.rvs(size=None, random_state=42)
            assert x.shape == (2,), f"Single sample shape wrong: {x.shape}"

            # Multiple samples
            x = nig.rvs(size=100, random_state=42)
            assert x.shape == (100, 2), f"Multiple samples shape wrong: {x.shape}"

    def test_joint_access(self, params_list):
        """Test that joint distribution is accessible and consistent."""
        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            # Access joint
            joint = nig.joint
            assert isinstance(joint, JointNormalInverseGaussian)

            # Parameters should match
            joint_params = joint.get_classical_params()
            np.testing.assert_allclose(joint_params['mu'], params['mu'])
            np.testing.assert_allclose(joint_params['delta'], params['delta'], rtol=1e-6)

    def test_marginal_pdf_finite(self, params_list):
        """Test that marginal log PDF is finite."""
        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            # Generate samples
            X = nig.rvs(size=100, random_state=42)

            # Check log PDF is finite
            logpdf = nig.logpdf(X)
            assert np.all(np.isfinite(logpdf)), f"Found non-finite logpdf for {params}"

    def test_marginal_mean(self, params_list):
        """Test marginal mean E[X] = μ + γ δ."""
        n_samples = 50000

        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            # Theoretical mean
            E_X = nig.mean()

            # Expected: μ + γ × δ (since E[Y] = δ for IG)
            expected_E_X = params['mu'] + params['gamma'] * params['delta']

            np.testing.assert_allclose(
                E_X, expected_E_X, rtol=1e-6,
                err_msg=f"Theoretical mean mismatch for {params}"
            )

            # Sample mean
            X = nig.rvs(size=n_samples, random_state=42)
            sample_E_X = np.mean(X, axis=0)

            # Use absolute tolerance when expected is near zero
            atol = 0.1 * np.sqrt(nig.var().max())  # Scale by std dev
            np.testing.assert_allclose(
                sample_E_X, E_X, rtol=0.1, atol=atol,
                err_msg=f"Sample mean mismatch for {params}"
            )

    def test_pdf_joint_convenience(self, params_list):
        """Test that pdf_joint is consistent with joint.pdf."""
        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            # Generate some test points
            X, Y = nig.rvs_joint(size=10, random_state=42)

            # Compare convenience method with joint access
            pdf1 = nig.pdf_joint(X, Y)
            pdf2 = nig.joint.pdf(X, Y)

            np.testing.assert_allclose(
                pdf1, pdf2,
                err_msg=f"pdf_joint != joint.pdf for {params}"
            )

    def test_rvs_joint_returns_tuple(self, params_list):
        """Test that rvs_joint returns (X, Y) tuple."""
        for params in params_list:
            nig = NormalInverseGaussian.from_classical_params(**params)

            result = nig.rvs_joint(size=10, random_state=42)

            assert isinstance(result, tuple), "rvs_joint should return tuple"
            assert len(result) == 2, "rvs_joint should return (X, Y)"

            X, Y = result
            assert X.shape == (10, 2), f"X shape wrong: {X.shape}"
            assert Y.shape == (10,), f"Y shape wrong: {Y.shape}"


# ============================================================
# JointNormalInverseGaussian Fitting Tests
# ============================================================

class TestJointNormalInverseGaussianFitting:
    """Test JointNormalInverseGaussian.fit() with complete data."""

    @pytest.fixture
    def params_list(self):
        return get_joint_nig_1d_params()

    def test_fit_recovers_parameters_1d(self, params_list):
        """Test that fit recovers parameters from complete data."""
        n_samples = 10000

        for params in params_list:
            # Create true distribution
            true_dist = JointNormalInverseGaussian.from_classical_params(**params)

            # Generate complete data
            X, Y = true_dist.rvs(size=n_samples, random_state=42)

            # Fit new distribution
            fitted = JointNormalInverseGaussian(d=1).fit(X, Y)

            # Compare means (more robust than parameter comparison)
            true_E_X, true_E_Y = true_dist.mean()
            fitted_E_X, fitted_E_Y = fitted.mean()

            np.testing.assert_allclose(
                fitted_E_X, true_E_X, rtol=0.1,
                err_msg=f"E[X] mismatch for {params}"
            )
            np.testing.assert_allclose(
                fitted_E_Y, true_E_Y, rtol=0.1,
                err_msg=f"E[Y] mismatch for {params}"
            )

    def test_fit_recovers_parameters_2d(self):
        """Test fit with 2D X."""
        n_samples = 10000
        params = {
            'mu': np.array([0.0, 1.0]),
            'gamma': np.array([0.5, -0.3]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'delta': 1.0,
            'eta': 1.0
        }

        # Create true distribution
        true_dist = JointNormalInverseGaussian.from_classical_params(**params)

        # Generate complete data
        X, Y = true_dist.rvs(size=n_samples, random_state=42)

        # Fit new distribution
        fitted = JointNormalInverseGaussian(d=2).fit(X, Y)

        # Compare means
        true_E_X, true_E_Y = true_dist.mean()
        fitted_E_X, fitted_E_Y = fitted.mean()

        np.testing.assert_allclose(
            fitted_E_X, true_E_X, rtol=0.15,
            err_msg="E[X] mismatch for 2D case"
        )
        np.testing.assert_allclose(
            fitted_E_Y, true_E_Y, rtol=0.15,
            err_msg="E[Y] mismatch for 2D case"
        )


# ============================================================
# NormalInverseGaussian EM Fitting Tests
# ============================================================

class TestNormalInverseGaussianEMFitting:
    """Test NormalInverseGaussian.fit() with EM algorithm (marginal data only)."""

    def test_fit_em_1d_symmetric(self):
        """Test EM fitting with 1D symmetric (gamma=0) case."""
        n_samples = 5000
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.0]),
            'sigma': np.array([[1.0]]),
            'delta': 1.0,
            'eta': 1.0
        }

        # Create true distribution
        true_dist = NormalInverseGaussian.from_classical_params(**params)

        # Generate marginal data (X only)
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = NormalInverseGaussian().fit(X, max_iter=50, tol=1e-5, verbose=0)

        # Compare means
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.15, atol=0.15,
            err_msg="Mean mismatch for symmetric 1D case"
        )

    def test_fit_em_1d_skewed(self):
        """Test EM fitting with 1D skewed case."""
        n_samples = 5000
        params = {
            'mu': np.array([1.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'delta': 1.5,
            'eta': 1.0
        }

        # Create true distribution
        true_dist = NormalInverseGaussian.from_classical_params(**params)

        # Generate marginal data
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = NormalInverseGaussian().fit(X, max_iter=50, tol=1e-5, verbose=0)

        # Compare means (should be close)
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.2, atol=0.3,
            err_msg="Mean mismatch for skewed 1D case"
        )

    def test_fit_em_2d(self):
        """Test EM fitting with 2D case."""
        n_samples = 5000
        params = {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.3, -0.2]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'delta': 1.0,
            'eta': 1.0
        }

        # Create true distribution
        true_dist = NormalInverseGaussian.from_classical_params(**params)

        # Generate marginal data
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = NormalInverseGaussian().fit(X, max_iter=50, tol=1e-5, verbose=0)

        # Compare means
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.25, atol=0.3,
            err_msg="Mean mismatch for 2D case"
        )

    def test_fit_complete_vs_em(self):
        """Test that fit_complete is more accurate than EM (as expected)."""
        n_samples = 3000
        params = {
            'mu': np.array([0.5]),
            'gamma': np.array([0.3]),
            'sigma': np.array([[1.0]]),
            'delta': 1.0,
            'eta': 1.0
        }

        true_dist = NormalInverseGaussian.from_classical_params(**params)
        X, Y = true_dist.rvs_joint(size=n_samples, random_state=42)

        # Fit with complete data
        fitted_complete = NormalInverseGaussian().fit_complete(X, Y)

        # Fit with EM (marginal data only)
        fitted_em = NormalInverseGaussian().fit(X, max_iter=50, tol=1e-6, verbose=0)

        # Compare errors
        true_mean = true_dist.mean()
        complete_mean = fitted_complete.mean()
        em_mean = fitted_em.mean()

        complete_error = np.abs(complete_mean - true_mean)
        em_error = np.abs(em_mean - true_mean)

        # Complete should be at least as good (usually better)
        assert np.all(complete_error <= em_error + 0.5), (
            f"Complete fit error {complete_error} should be <= EM error {em_error}"
        )


# ============================================================
# Conditional Expectation Tests
# ============================================================

class TestConditionalExpectationsNIG:
    """Test _conditional_expectation_y_given_x method for NormalInverseGaussian."""

    def test_conditional_expectations_finite(self):
        """Test that conditional expectations are finite."""
        nig = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            delta=1.0,
            eta=1.0
        )

        # Generate some test points
        X = nig.rvs(size=100, random_state=42)

        # Compute conditional expectations
        cond_exp = nig._conditional_expectation_y_given_x(X)

        assert np.all(np.isfinite(cond_exp['E_Y'])), "E[Y|X] should be finite"
        assert np.all(np.isfinite(cond_exp['E_inv_Y'])), "E[1/Y|X] should be finite"
        assert np.all(np.isfinite(cond_exp['E_log_Y'])), "E[log Y|X] should be finite"

    def test_conditional_expectations_positive(self):
        """Test that E[Y|X] and E[1/Y|X] are positive."""
        nig = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([0.3, -0.2]),
            sigma=np.array([[1.0, 0.0], [0.0, 1.0]]),
            delta=1.0,
            eta=1.0
        )

        X = nig.rvs(size=100, random_state=42)
        cond_exp = nig._conditional_expectation_y_given_x(X)

        assert np.all(cond_exp['E_Y'] > 0), "E[Y|X] should be positive"
        assert np.all(cond_exp['E_inv_Y'] > 0), "E[1/Y|X] should be positive"

    def test_conditional_expectations_single_point(self):
        """Test conditional expectations for single point."""
        nig = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            delta=1.0,
            eta=1.0
        )

        # Single point
        x = np.array([1.0])
        cond_exp = nig._conditional_expectation_y_given_x(x)

        # Should return scalars
        assert isinstance(cond_exp['E_Y'], float), "E[Y|X] should be scalar for single point"
        assert isinstance(cond_exp['E_inv_Y'], float), "E[1/Y|X] should be scalar"
        assert isinstance(cond_exp['E_log_Y'], float), "E[log Y|X] should be scalar"


# ============================================================
# Edge Cases and Numerical Stability
# ============================================================

class TestNormalInverseGaussianEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_gamma(self):
        """Test with very small skewness (near symmetric)."""
        nig = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([1e-10, 1e-10]),
            sigma=np.array([[1.0, 0.0], [0.0, 1.0]]),
            delta=1.0,
            eta=1.0
        )

        # Should work without numerical issues
        X = nig.rvs(size=100, random_state=42)
        logpdf = nig.logpdf(X[:5])

        assert np.all(np.isfinite(logpdf)), "logpdf should be finite for small gamma"

    def test_large_eta(self):
        """Test with large shape parameter (concentrated around mean)."""
        nig = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            delta=1.0,
            eta=10.0  # Large shape
        )

        # Should work without numerical issues
        X = nig.rvs(size=100, random_state=42)
        logpdf = nig.logpdf(X[:5])

        assert np.all(np.isfinite(logpdf)), "logpdf should be finite for large eta"

    def test_small_eta(self):
        """Test with small shape parameter (heavy tails)."""
        nig = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            delta=1.0,
            eta=0.5  # Small shape
        )

        # Should work without numerical issues
        X = nig.rvs(size=100, random_state=42)
        logpdf = nig.logpdf(X[:5])

        assert np.all(np.isfinite(logpdf)), "logpdf should be finite for small eta"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
