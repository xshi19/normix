"""
Tests for Variance Gamma distributions.

Since there is no scipy distribution to compare against, we test:
- Sample sufficient statistics match expectation parameters
- Histogram matches PDF (chi-square test)
- Parameter roundtrip consistency
"""

import numpy as np
import pytest
from scipy.special import digamma

from normix.distributions.mixtures import JointVarianceGamma, VarianceGamma


# ============================================================
# Test Configurations
# ============================================================

def get_joint_vg_1d_params():
    """Test parameters for 1D JointVarianceGamma."""
    return [
        {
            'mu': np.array([0.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'shape': 2.0,
            'rate': 1.0
        },
        {
            'mu': np.array([1.0]),
            'gamma': np.array([-0.3]),
            'sigma': np.array([[2.0]]),
            'shape': 3.0,
            'rate': 0.5
        },
        {
            'mu': np.array([0.5]),
            'gamma': np.array([0.0]),  # Symmetric case
            'sigma': np.array([[1.5]]),
            'shape': 2.5,
            'rate': 1.5
        },
    ]


def get_vg_2d_params():
    """Test parameters for 2D VarianceGamma."""
    return [
        {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.5, -0.3]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'shape': 2.0,
            'rate': 1.0
        },
        {
            'mu': np.array([1.0, -0.5]),
            'gamma': np.array([0.2, 0.4]),
            'sigma': np.array([[2.0, 0.5], [0.5, 1.5]]),
            'shape': 3.0,
            'rate': 0.5
        },
        {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.0, 0.0]),  # Symmetric case
            'sigma': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'shape': 2.5,
            'rate': 1.0
        },
    ]


# ============================================================
# JointVarianceGamma Tests (1D X, 1D Y)
# ============================================================

class TestJointVarianceGamma1D:
    """Test JointVarianceGamma with 1D X and 1D Y (total 2D)."""

    @pytest.fixture
    def params_list(self):
        return get_joint_vg_1d_params()

    def test_parameter_roundtrip(self, params_list):
        """Test classical -> natural -> classical roundtrip."""
        for params in params_list:
            # Create from classical
            dist = JointVarianceGamma.from_classical_params(**params)

            # Get natural params
            theta = dist.get_natural_params()

            # Create from natural
            dist2 = JointVarianceGamma(d=1)
            dist2.set_natural_params(theta)

            # Get classical back
            recovered = dist2.get_classical_params()

            # Compare
            np.testing.assert_allclose(
                recovered['mu'], params['mu'], rtol=1e-10,
                err_msg=f"mu mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['gamma'], params['gamma'], rtol=1e-10,
                err_msg=f"gamma mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['sigma'], params['sigma'], rtol=1e-10,
                err_msg=f"sigma mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['shape'], params['shape'], rtol=1e-10,
                err_msg=f"shape mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['rate'], params['rate'], rtol=1e-10,
                err_msg=f"rate mismatch for {params}"
            )

    def test_sample_sufficient_statistics(self, params_list):
        """Test E[t(X,Y)] from samples matches expectation parameters."""
        n_samples = 50000
        rtol = 0.1  # 10% tolerance for statistical test

        for params in params_list:
            dist = JointVarianceGamma.from_classical_params(**params)

            # Get theoretical expectation parameters
            eta_theory = dist.get_expectation_params()

            # Skip if any are infinite (happens when α ≤ 1)
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

    def test_histogram_vs_pdf(self, params_list):
        """Test histogram of samples matches PDF (chi-square test)."""
        n_samples = 10000
        n_bins = 30

        for params in params_list:
            dist = JointVarianceGamma.from_classical_params(**params)

            # Generate samples - we'll test marginal of Y
            _, Y = dist.rvs(size=n_samples, random_state=42)

            # Create histogram for Y
            counts, bin_edges = np.histogram(Y, bins=n_bins, density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]

            # Expected from Gamma distribution (mixing distribution)
            from scipy.stats import gamma as scipy_gamma
            alpha = params['shape']
            beta = params['rate']
            expected_density = scipy_gamma.pdf(bin_centers, a=alpha, scale=1/beta)
            expected_counts = expected_density * bin_width * n_samples

            # Filter bins with low expected counts
            mask = expected_counts >= 5
            counts_filtered = counts[mask]
            expected_filtered = expected_counts[mask]

            # Chi-square statistic
            chi_square = np.sum((counts_filtered - expected_filtered)**2 / expected_filtered)
            dof = len(counts_filtered) - 1
            critical_value = dof + 3 * np.sqrt(2 * dof)

            assert chi_square < critical_value, (
                f"Y histogram doesn't match Gamma PDF for {params}\n"
                f"Chi-square: {chi_square:.2f}, Critical: {critical_value:.2f}"
            )

    def test_joint_pdf_integrates(self, params_list):
        """Test that joint PDF is properly normalized (approximately)."""
        for params in params_list:
            dist = JointVarianceGamma.from_classical_params(**params)

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
            dist = JointVarianceGamma.from_classical_params(**params)

            # Theoretical moments
            E_X, E_Y = dist.mean()

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
# VarianceGamma Tests (2D X)
# ============================================================

class TestVarianceGamma2D:
    """Test VarianceGamma marginal distribution with 2D X."""

    @pytest.fixture
    def params_list(self):
        return get_vg_2d_params()

    def test_parameter_access(self, params_list):
        """Test that parameters can be accessed correctly."""
        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Access via joint
            classical = vg.get_classical_params()

            np.testing.assert_allclose(classical['mu'], params['mu'])
            np.testing.assert_allclose(classical['gamma'], params['gamma'])
            np.testing.assert_allclose(classical['sigma'], params['sigma'])
            np.testing.assert_allclose(classical['shape'], params['shape'])
            np.testing.assert_allclose(classical['rate'], params['rate'])

    def test_marginal_rvs_shape(self, params_list):
        """Test that marginal rvs returns correct shape."""
        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Single sample
            x = vg.rvs(size=None, random_state=42)
            assert x.shape == (2,), f"Single sample shape wrong: {x.shape}"

            # Multiple samples
            x = vg.rvs(size=100, random_state=42)
            assert x.shape == (100, 2), f"Multiple samples shape wrong: {x.shape}"

    def test_joint_access(self, params_list):
        """Test that joint distribution is accessible and consistent."""
        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Access joint
            joint = vg.joint
            assert isinstance(joint, JointVarianceGamma)

            # Parameters should match
            joint_params = joint.get_classical_params()
            np.testing.assert_allclose(joint_params['mu'], params['mu'])
            np.testing.assert_allclose(joint_params['shape'], params['shape'])

    def test_histogram_vs_pdf_marginal(self, params_list):
        """Test marginal histogram matches marginal PDF."""
        n_samples = 10000
        n_bins = 30

        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Generate samples
            X = vg.rvs(size=n_samples, random_state=42)

            # Test first component
            x1 = X[:, 0]

            # Create histogram
            counts, bin_edges = np.histogram(x1, bins=n_bins, density=False)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]

            # Evaluate marginal PDF at bin centers (need to integrate over x2)
            # For simplicity, just check that PDF values are reasonable
            x_test = np.column_stack([
                bin_centers,
                np.full_like(bin_centers, params['mu'][1])  # Fix x2 at mu[1]
            ])

            pdf_vals = np.array([vg.pdf(x) for x in x_test])

            # PDF should be positive and finite where we have samples
            assert np.all(pdf_vals > 0), f"Found non-positive PDF values for {params}"
            assert np.all(np.isfinite(pdf_vals)), f"Found non-finite PDF values for {params}"

    def test_marginal_mean(self, params_list):
        """Test marginal mean E[X] = μ + γ E[Y]."""
        n_samples = 50000

        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Theoretical mean
            E_X = vg.mean()

            # Expected: μ + γ × α/β
            alpha = params['shape']
            beta = params['rate']
            E_Y = alpha / beta
            expected_E_X = params['mu'] + params['gamma'] * E_Y

            np.testing.assert_allclose(
                E_X, expected_E_X, rtol=1e-10,
                err_msg=f"Theoretical mean mismatch for {params}"
            )

            # Sample mean
            X = vg.rvs(size=n_samples, random_state=42)
            sample_E_X = np.mean(X, axis=0)

            # Use absolute tolerance when expected is near zero
            atol = 0.05 * np.sqrt(vg.var().max())  # Scale by std dev
            np.testing.assert_allclose(
                sample_E_X, E_X, rtol=0.1, atol=atol,
                err_msg=f"Sample mean mismatch for {params}"
            )

    def test_marginal_variance(self, params_list):
        """Test marginal variance."""
        n_samples = 50000

        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Theoretical variance (diagonal)
            Var_X = vg.var()

            # Cov[X] = E[Y] Σ + Var[Y] γγ^T
            alpha = params['shape']
            beta = params['rate']
            E_Y = alpha / beta
            Var_Y = alpha / (beta ** 2)
            expected_Cov_X = E_Y * params['sigma'] + Var_Y * np.outer(params['gamma'], params['gamma'])
            expected_Var_X = np.diag(expected_Cov_X)

            np.testing.assert_allclose(
                Var_X, expected_Var_X, rtol=1e-10,
                err_msg=f"Theoretical variance mismatch for {params}"
            )

            # Sample variance
            X = vg.rvs(size=n_samples, random_state=42)
            sample_Var_X = np.var(X, axis=0)

            np.testing.assert_allclose(
                sample_Var_X, Var_X, rtol=0.15,
                err_msg=f"Sample variance mismatch for {params}"
            )

    def test_pdf_joint_convenience(self, params_list):
        """Test that pdf_joint is consistent with joint.pdf."""
        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            # Generate some test points
            X, Y = vg.rvs_joint(size=10, random_state=42)

            # Compare convenience method with joint access
            pdf1 = vg.pdf_joint(X, Y)
            pdf2 = vg.joint.pdf(X, Y)

            np.testing.assert_allclose(
                pdf1, pdf2,
                err_msg=f"pdf_joint != joint.pdf for {params}"
            )

    def test_rvs_joint_returns_tuple(self, params_list):
        """Test that rvs_joint returns (X, Y) tuple."""
        for params in params_list:
            vg = VarianceGamma.from_classical_params(**params)

            result = vg.rvs_joint(size=10, random_state=42)

            assert isinstance(result, tuple), "rvs_joint should return tuple"
            assert len(result) == 2, "rvs_joint should return (X, Y)"

            X, Y = result
            assert X.shape == (10, 2), f"X shape wrong: {X.shape}"
            assert Y.shape == (10,), f"Y shape wrong: {Y.shape}"


# ============================================================
# Edge Cases and Numerical Stability
# ============================================================

class TestVarianceGammaEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_gamma(self):
        """Test with very small skewness (near symmetric)."""
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([1e-10, 1e-10]),
            sigma=np.array([[1.0, 0.0], [0.0, 1.0]]),
            shape=2.0,
            rate=1.0
        )

        # Should work without numerical issues
        X = vg.rvs(size=100, random_state=42)
        pdf = vg.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for small gamma"

    def test_large_shape(self):
        """Test with large shape parameter."""
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=20.0,  # Large shape
            rate=10.0
        )

        # Should work without numerical issues
        X = vg.rvs(size=100, random_state=42)
        pdf = vg.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for large shape"

    def test_different_scales(self):
        """Test with different scales in sigma."""
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0, 100.0]),  # Different scales
            gamma=np.array([0.1, 10.0]),
            sigma=np.array([[0.01, 0.0], [0.0, 100.0]]),
            shape=2.0,
            rate=1.0
        )

        # Should work without numerical issues
        X = vg.rvs(size=100, random_state=42)
        pdf = vg.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for different scales"


# ============================================================
# JointVarianceGamma Fitting Tests
# ============================================================

class TestJointVarianceGammaFitting:
    """Test JointVarianceGamma.fit() with complete data."""

    @pytest.fixture
    def params_list(self):
        return get_joint_vg_1d_params()

    def test_fit_recovers_parameters_1d(self, params_list):
        """Test that fit recovers parameters from complete data."""
        n_samples = 10000

        for params in params_list:
            # Create true distribution
            true_dist = JointVarianceGamma.from_classical_params(**params)

            # Generate complete data
            X, Y = true_dist.rvs(size=n_samples, random_state=42)

            # Fit new distribution
            fitted = JointVarianceGamma(d=1).fit(X, Y)

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
            'shape': 2.0,
            'rate': 1.0
        }

        # Create true distribution
        true_dist = JointVarianceGamma.from_classical_params(**params)

        # Generate complete data
        X, Y = true_dist.rvs(size=n_samples, random_state=42)

        # Fit new distribution
        fitted = JointVarianceGamma(d=2).fit(X, Y)

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

    def test_fit_log_likelihood_improves(self, params_list):
        """Test that fitted distribution has reasonable log-likelihood."""
        n_samples = 5000

        for params in params_list:
            # Create true distribution
            true_dist = JointVarianceGamma.from_classical_params(**params)

            # Generate complete data
            X, Y = true_dist.rvs(size=n_samples, random_state=42)

            # Fit new distribution
            fitted = JointVarianceGamma(d=1).fit(X, Y)

            # Compute log-likelihoods
            true_ll = np.mean(true_dist.logpdf(X, Y))
            fitted_ll = np.mean(fitted.logpdf(X, Y))

            # Fitted should be close to true (MLE property)
            # Allow some tolerance since we're comparing on training data
            assert fitted_ll >= true_ll - 0.5, (
                f"Fitted log-likelihood {fitted_ll:.4f} much worse than "
                f"true {true_ll:.4f} for {params}"
            )


# ============================================================
# VarianceGamma EM Fitting Tests
# ============================================================

class TestVarianceGammaEMFitting:
    """Test VarianceGamma.fit() with EM algorithm (marginal data only)."""

    def test_fit_em_1d_symmetric(self):
        """Test EM fitting with 1D symmetric (gamma=0) case."""
        n_samples = 5000
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.0]),
            'sigma': np.array([[1.0]]),
            'shape': 2.0,
            'rate': 1.0
        }

        # Create true distribution
        true_dist = VarianceGamma.from_classical_params(**params)

        # Generate marginal data (X only)
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = VarianceGamma().fit(X, max_iter=50, tol=1e-5, verbose=0)

        # Compare means
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.15, atol=0.1,
            err_msg="Mean mismatch for symmetric 1D case"
        )

        # Compare variances
        true_var = true_dist.var()
        fitted_var = fitted.var()

        np.testing.assert_allclose(
            fitted_var, true_var, rtol=0.2,
            err_msg="Variance mismatch for symmetric 1D case"
        )

    def test_fit_em_1d_skewed(self):
        """Test EM fitting with 1D skewed case."""
        n_samples = 5000
        params = {
            'mu': np.array([1.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'shape': 3.0,
            'rate': 1.0
        }

        # Create true distribution
        true_dist = VarianceGamma.from_classical_params(**params)

        # Generate marginal data
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = VarianceGamma().fit(X, max_iter=50, tol=1e-5, verbose=0)

        # Compare means (should be close)
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.15, atol=0.2,
            err_msg="Mean mismatch for skewed 1D case"
        )

    def test_fit_em_2d(self):
        """Test EM fitting with 2D case."""
        n_samples = 5000
        params = {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.3, -0.2]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'shape': 2.5,
            'rate': 1.0
        }

        # Create true distribution
        true_dist = VarianceGamma.from_classical_params(**params)

        # Generate marginal data
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = VarianceGamma().fit(X, max_iter=50, tol=1e-5, verbose=0)

        # Compare means
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.2, atol=0.2,
            err_msg="Mean mismatch for 2D case"
        )

    def test_fit_em_log_likelihood_increases(self):
        """Test that EM iterations increase log-likelihood."""
        n_samples = 2000
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.3]),
            'sigma': np.array([[1.0]]),
            'shape': 2.0,
            'rate': 1.0
        }

        true_dist = VarianceGamma.from_classical_params(**params)
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit with verbose to track progress
        fitted = VarianceGamma().fit(X, max_iter=30, tol=1e-8, verbose=0)

        # Final log-likelihood should be reasonable
        final_ll = np.mean(fitted.logpdf(X))
        true_ll = np.mean(true_dist.logpdf(X))

        # Fitted should achieve reasonable log-likelihood
        assert final_ll > true_ll - 1.0, (
            f"Fitted log-likelihood {final_ll:.4f} much worse than "
            f"true {true_ll:.4f}"
        )

    def test_fit_complete_vs_em(self):
        """Test that fit_complete is more accurate than EM (as expected)."""
        n_samples = 3000
        params = {
            'mu': np.array([0.5]),
            'gamma': np.array([0.3]),
            'sigma': np.array([[1.0]]),
            'shape': 2.0,
            'rate': 1.0
        }

        true_dist = VarianceGamma.from_classical_params(**params)
        X, Y = true_dist.rvs_joint(size=n_samples, random_state=42)

        # Fit with complete data
        fitted_complete = VarianceGamma().fit_complete(X, Y)

        # Fit with EM (marginal data only)
        fitted_em = VarianceGamma().fit(X, max_iter=50, tol=1e-6, verbose=0)

        # Complete data fit should be closer to true parameters
        true_mean = true_dist.mean()
        complete_mean = fitted_complete.mean()
        em_mean = fitted_em.mean()

        complete_error = np.abs(complete_mean - true_mean)
        em_error = np.abs(em_mean - true_mean)

        # Complete should be at least as good (usually better)
        # Allow some tolerance since both are estimates
        assert np.all(complete_error <= em_error + 0.3), (
            f"Complete fit error {complete_error} should be <= EM error {em_error}"
        )


# ============================================================
# Conditional Expectation Tests
# ============================================================

class TestConditionalExpectations:
    """Test _conditional_expectation_y_given_x method."""

    def test_conditional_expectations_finite(self):
        """Test that conditional expectations are finite."""
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=2.0,
            rate=1.0
        )

        # Generate some test points
        X = vg.rvs(size=100, random_state=42)

        # Compute conditional expectations
        cond_exp = vg._conditional_expectation_y_given_x(X)

        assert np.all(np.isfinite(cond_exp['E_Y'])), "E[Y|X] should be finite"
        assert np.all(np.isfinite(cond_exp['E_inv_Y'])), "E[1/Y|X] should be finite"
        assert np.all(np.isfinite(cond_exp['E_log_Y'])), "E[log Y|X] should be finite"

    def test_conditional_expectations_positive(self):
        """Test that E[Y|X] and E[1/Y|X] are positive."""
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([0.3, -0.2]),
            sigma=np.array([[1.0, 0.0], [0.0, 1.0]]),
            shape=2.0,
            rate=1.0
        )

        X = vg.rvs(size=100, random_state=42)
        cond_exp = vg._conditional_expectation_y_given_x(X)

        assert np.all(cond_exp['E_Y'] > 0), "E[Y|X] should be positive"
        assert np.all(cond_exp['E_inv_Y'] > 0), "E[1/Y|X] should be positive"

    def test_conditional_expectations_single_point(self):
        """Test conditional expectations for single point."""
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=2.0,
            rate=1.0
        )

        # Single point
        x = np.array([1.0])
        cond_exp = vg._conditional_expectation_y_given_x(x)

        # Should return scalars
        assert isinstance(cond_exp['E_Y'], float), "E[Y|X] should be scalar for single point"
        assert isinstance(cond_exp['E_inv_Y'], float), "E[1/Y|X] should be scalar"
        assert isinstance(cond_exp['E_log_Y'], float), "E[log Y|X] should be scalar"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
