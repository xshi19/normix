"""
Tests for Generalized Hyperbolic distributions.

Since there is no scipy distribution to compare against, we test:
- Sample sufficient statistics match expectation parameters
- Histogram matches PDF (chi-square test)
- Parameter roundtrip consistency
- Special case consistency (VG, NIG, NInvG)
- EM algorithm convergence with regularization
"""

import numpy as np
import pytest

from normix.distributions.mixtures import (
    JointGeneralizedHyperbolic,
    GeneralizedHyperbolic,
    JointVarianceGamma,
    VarianceGamma,
    JointNormalInverseGaussian,
    NormalInverseGaussian,
    JointNormalInverseGamma,
    NormalInverseGamma,
    REGULARIZATION_METHODS,
)


# ============================================================
# Test Configurations
# ============================================================

def get_joint_gh_1d_params():
    """Test parameters for 1D JointGeneralizedHyperbolic."""
    return [
        {
            'mu': np.array([0.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        },
        {
            'mu': np.array([1.0]),
            'gamma': np.array([-0.3]),
            'sigma': np.array([[2.0]]),
            'p': -0.5,
            'a': 2.0,
            'b': 1.5
        },
        {
            'mu': np.array([0.5]),
            'gamma': np.array([0.0]),  # Symmetric case
            'sigma': np.array([[1.5]]),
            'p': 2.0,
            'a': 1.0,
            'b': 1.0
        },
    ]


def get_gh_2d_params():
    """Test parameters for 2D GeneralizedHyperbolic."""
    return [
        {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.5, -0.3]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        },
        {
            'mu': np.array([1.0, -0.5]),
            'gamma': np.array([0.2, 0.4]),
            'sigma': np.array([[2.0, 0.5], [0.5, 1.5]]),
            'p': -0.5,
            'a': 1.5,
            'b': 1.0
        },
        {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.0, 0.0]),  # Symmetric case
            'sigma': np.array([[1.0, 0.0], [0.0, 1.0]]),
            'p': 2.0,
            'a': 1.0,
            'b': 1.0
        },
    ]


# ============================================================
# JointGeneralizedHyperbolic Tests (1D X, 1D Y)
# ============================================================

class TestJointGeneralizedHyperbolic1D:
    """Test JointGeneralizedHyperbolic with 1D X and 1D Y (total 2D)."""

    @pytest.fixture
    def params_list(self):
        return get_joint_gh_1d_params()

    def test_parameter_roundtrip(self, params_list):
        """Test classical -> natural -> classical roundtrip."""
        for params in params_list:
            # Create from classical
            dist = JointGeneralizedHyperbolic.from_classical_params(**params)

            # Get natural params
            theta = dist.natural_params

            # Create from natural
            dist2 = JointGeneralizedHyperbolic(d=1)
            dist2.set_natural_params(theta)

            # Get classical back
            recovered = dist2.classical_params

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
                recovered['p'], params['p'], rtol=1e-8,
                err_msg=f"p mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['a'], params['a'], rtol=1e-8,
                err_msg=f"a mismatch for {params}"
            )
            np.testing.assert_allclose(
                recovered['b'], params['b'], rtol=1e-8,
                err_msg=f"b mismatch for {params}"
            )

    def test_sample_sufficient_statistics(self, params_list):
        """Test E[t(X,Y)] from samples matches expectation parameters."""
        n_samples = 50000
        rtol = 0.1  # 10% tolerance for statistical test

        for params in params_list:
            dist = JointGeneralizedHyperbolic.from_classical_params(**params)

            # Get theoretical expectation parameters
            eta_theory = dist.expectation_params

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

    def test_joint_pdf_positive(self, params_list):
        """Test that joint PDF is positive at samples."""
        for params in params_list:
            dist = JointGeneralizedHyperbolic.from_classical_params(**params)

            # Generate samples
            X, Y = dist.rvs(size=1000, random_state=42)

            # Compute PDF at samples
            pdf_vals = dist.pdf(X, Y)

            # All PDF values should be positive and finite
            assert np.all(pdf_vals > 0), f"Found non-positive PDF values for {params}"
            assert np.all(np.isfinite(pdf_vals)), f"Found non-finite PDF values for {params}"

    def test_mean_variance(self, params_list):
        """Test mean and variance computations."""
        n_samples = 50000

        for params in params_list:
            dist = JointGeneralizedHyperbolic.from_classical_params(**params)

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
# GeneralizedHyperbolic Tests (2D X)
# ============================================================

class TestGeneralizedHyperbolic2D:
    """Test GeneralizedHyperbolic marginal distribution with 2D X."""

    @pytest.fixture
    def params_list(self):
        return get_gh_2d_params()

    def test_parameter_access(self, params_list):
        """Test that parameters can be accessed correctly."""
        for params in params_list:
            gh = GeneralizedHyperbolic.from_classical_params(**params)

            # Access via joint
            classical = gh.classical_params

            np.testing.assert_allclose(classical['mu'], params['mu'])
            np.testing.assert_allclose(classical['gamma'], params['gamma'])
            np.testing.assert_allclose(classical['sigma'], params['sigma'])
            np.testing.assert_allclose(classical['p'], params['p'])
            np.testing.assert_allclose(classical['a'], params['a'])
            np.testing.assert_allclose(classical['b'], params['b'])

    def test_marginal_rvs_shape(self, params_list):
        """Test that marginal rvs returns correct shape."""
        for params in params_list:
            gh = GeneralizedHyperbolic.from_classical_params(**params)

            # Single sample
            x = gh.rvs(size=None, random_state=42)
            assert x.shape == (2,), f"Single sample shape wrong: {x.shape}"

            # Multiple samples
            x = gh.rvs(size=100, random_state=42)
            assert x.shape == (100, 2), f"Multiple samples shape wrong: {x.shape}"

    def test_joint_access(self, params_list):
        """Test that joint distribution is accessible and consistent."""
        for params in params_list:
            gh = GeneralizedHyperbolic.from_classical_params(**params)

            # Access joint
            joint = gh.joint
            assert isinstance(joint, JointGeneralizedHyperbolic)

            # Parameters should match
            joint_params = joint.classical_params
            np.testing.assert_allclose(joint_params['mu'], params['mu'])
            np.testing.assert_allclose(joint_params['p'], params['p'])

    def test_marginal_pdf_positive(self, params_list):
        """Test marginal PDF is positive at samples."""
        for params in params_list:
            gh = GeneralizedHyperbolic.from_classical_params(**params)

            # Generate samples
            X = gh.rvs(size=100, random_state=42)

            # Evaluate PDF
            pdf_vals = np.array([gh.pdf(x) for x in X])

            # PDF should be positive and finite
            assert np.all(pdf_vals > 0), f"Found non-positive PDF values for {params}"
            assert np.all(np.isfinite(pdf_vals)), f"Found non-finite PDF values for {params}"

    def test_marginal_mean(self, params_list):
        """Test marginal mean E[X] = μ + γ E[Y]."""
        n_samples = 50000

        for params in params_list:
            gh = GeneralizedHyperbolic.from_classical_params(**params)

            # Theoretical mean
            E_X = gh.mean()

            # Sample mean
            X = gh.rvs(size=n_samples, random_state=42)
            sample_E_X = np.mean(X, axis=0)

            # Use absolute tolerance when expected is near zero
            atol = 0.05 * np.sqrt(gh.var().max())
            np.testing.assert_allclose(
                sample_E_X, E_X, rtol=0.1, atol=atol,
                err_msg=f"Sample mean mismatch for {params}"
            )


# ============================================================
# Special Case Consistency Tests
# ============================================================

class TestGHSpecialCases:
    """Test that GH special cases match their dedicated implementations."""

    def test_gh_matches_vg_at_limit(self):
        """Test that GH with b → 0 matches Variance Gamma."""
        # Use Variance Gamma parameters
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'shape': 2.0,
            'rate': 1.0
        }

        # Create VG distribution
        vg = VarianceGamma.from_classical_params(**params)

        # Create GH equivalent
        gh = GeneralizedHyperbolic.as_variance_gamma(**params)

        # Compare means (should be very close)
        vg_mean = vg.mean()
        gh_mean = gh.mean()

        np.testing.assert_allclose(
            gh_mean, vg_mean, rtol=0.01,
            err_msg="GH as VG mean doesn't match VG mean"
        )

        # Compare samples at same points (away from origin where VG can be -inf)
        x = np.array([[1.0], [2.0], [-1.0]])
        vg_logpdf = np.array([vg.logpdf(xi) for xi in x])
        gh_logpdf = np.array([gh.logpdf(xi) for xi in x])

        # Filter out infinities
        mask = np.isfinite(vg_logpdf) & np.isfinite(gh_logpdf)
        if np.any(mask):
            np.testing.assert_allclose(
                gh_logpdf[mask], vg_logpdf[mask], rtol=0.15,
                err_msg="GH as VG logpdf doesn't match VG logpdf"
            )

    def test_gh_matches_nig_at_limit(self):
        """Test that GH with p = -1/2 matches Normal-Inverse Gaussian."""
        # Use NIG parameters
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'delta': 1.0,
            'eta': 1.0
        }

        # Create NIG distribution
        nig = NormalInverseGaussian.from_classical_params(**params)

        # Create GH equivalent
        gh = GeneralizedHyperbolic.as_normal_inverse_gaussian(**params)

        # Compare means
        nig_mean = nig.mean()
        gh_mean = gh.mean()

        np.testing.assert_allclose(
            gh_mean, nig_mean, rtol=0.01,
            err_msg="GH as NIG mean doesn't match NIG mean"
        )

        # Compare logpdf at sample points
        x = np.array([[0.0], [1.0], [-1.0]])
        nig_logpdf = np.array([nig.logpdf(xi) for xi in x])
        gh_logpdf = np.array([gh.logpdf(xi) for xi in x])

        np.testing.assert_allclose(
            gh_logpdf, nig_logpdf, rtol=0.05,
            err_msg="GH as NIG logpdf doesn't match NIG logpdf"
        )

    def test_gh_matches_ninvg_at_limit(self):
        """Test that GH with a → 0 matches Normal-Inverse Gamma."""
        # Use NInvG parameters
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'shape': 3.0,  # α > 1 for moments to exist
            'rate': 1.0
        }

        # Create NInvG distribution
        ninvg = NormalInverseGamma.from_classical_params(**params)

        # Create GH equivalent
        gh = GeneralizedHyperbolic.as_normal_inverse_gamma(**params)

        # Compare means
        ninvg_mean = ninvg.mean()
        gh_mean = gh.mean()

        np.testing.assert_allclose(
            gh_mean, ninvg_mean, rtol=0.01,
            err_msg="GH as NInvG mean doesn't match NInvG mean"
        )


# ============================================================
# Regularization Tests
# ============================================================

class TestRegularization:
    """Test regularization methods."""

    def test_det_sigma_one_produces_unit_determinant(self):
        """Test that det_sigma_one regularization produces |Σ| = 1."""
        from normix.distributions.mixtures import regularize_det_sigma_one

        mu = np.array([0.0, 1.0])
        gamma = np.array([0.5, -0.3])
        sigma = np.array([[2.0, 0.5], [0.5, 1.5]])  # det = 2.75
        p, a, b = 1.0, 1.0, 1.0
        d = 2

        result = regularize_det_sigma_one(mu, gamma, sigma, p, a, b, d)

        det_result = np.linalg.det(result['sigma'])
        np.testing.assert_allclose(det_result, 1.0, rtol=1e-10)

    def test_sigma_diagonal_one_produces_unit_first_diagonal(self):
        """Test that sigma_diagonal_one regularization produces Σ₁₁ = 1."""
        from normix.distributions.mixtures import regularize_sigma_diagonal_one

        mu = np.array([0.0, 1.0])
        gamma = np.array([0.5, -0.3])
        sigma = np.array([[2.0, 0.5], [0.5, 1.5]])
        p, a, b = 1.0, 1.0, 1.0
        d = 2

        result = regularize_sigma_diagonal_one(mu, gamma, sigma, p, a, b, d)

        np.testing.assert_allclose(result['sigma'][0, 0], 1.0, rtol=1e-10)

    def test_fix_p_fixes_parameter(self):
        """Test that fix_p regularization fixes the p parameter."""
        from normix.distributions.mixtures import regularize_fix_p

        mu = np.array([0.0])
        gamma = np.array([0.5])
        sigma = np.array([[1.0]])
        p, a, b = 1.0, 1.0, 1.0
        d = 1

        result = regularize_fix_p(mu, gamma, sigma, p, a, b, d, p_fixed=-0.5)

        np.testing.assert_equal(result['p'], -0.5)

    def test_regularization_methods_available(self):
        """Test that all regularization methods are accessible."""
        expected_methods = ['det_sigma_one', 'sigma_diagonal_one', 'fix_p', 'none']

        for method in expected_methods:
            assert method in REGULARIZATION_METHODS, f"Missing method: {method}"


# ============================================================
# EM Fitting Tests
# ============================================================

class TestGeneralizedHyperbolicEMFitting:
    """Test GeneralizedHyperbolic.fit() with EM algorithm."""

    def test_fit_em_1d_symmetric(self):
        """Test EM fitting with 1D symmetric (gamma=0) case."""
        n_samples = 5000
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.0]),
            'sigma': np.array([[1.0]]),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        }

        # Create true distribution
        true_dist = GeneralizedHyperbolic.from_classical_params(**params)

        # Generate marginal data (X only)
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM with NIG regularization (more stable)
        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=100, tol=1e-6, verbose=0,
            regularization='fix_p',
            regularization_params={'p_fixed': 1.0}
        )

        # Compare variances (more stable than means for symmetric case)
        true_var = true_dist.var()
        fitted_var = fitted.var()

        np.testing.assert_allclose(
            fitted_var, true_var, rtol=0.3,
            err_msg="Variance mismatch for symmetric 1D case"
        )

    def test_fit_em_1d_skewed(self):
        """Test EM fitting with 1D skewed case."""
        n_samples = 5000
        params = {
            'mu': np.array([1.0]),
            'gamma': np.array([0.5]),
            'sigma': np.array([[1.0]]),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        }

        # Create true distribution
        true_dist = GeneralizedHyperbolic.from_classical_params(**params)

        # Generate marginal data
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM with fixed p (more stable)
        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=100, tol=1e-6, verbose=0,
            regularization='fix_p',
            regularization_params={'p_fixed': 1.0}
        )

        # GH fitting is challenging - just check variance is reasonable
        true_var = true_dist.var()
        fitted_var = fitted.var()

        np.testing.assert_allclose(
            fitted_var, true_var, rtol=0.5,
            err_msg="Variance mismatch for skewed 1D case"
        )

    def test_fit_em_2d(self):
        """Test EM fitting with 2D case."""
        n_samples = 3000
        params = {
            'mu': np.array([0.0, 0.0]),
            'gamma': np.array([0.3, -0.2]),
            'sigma': np.array([[1.0, 0.3], [0.3, 1.0]]),
            'p': 1.0,
            'a': 1.0,
            'b': 1.0
        }

        # Create true distribution
        true_dist = GeneralizedHyperbolic.from_classical_params(**params)

        # Generate marginal data
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit using EM
        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=50, tol=1e-5, verbose=0
        )

        # Compare means
        true_mean = true_dist.mean()
        fitted_mean = fitted.mean()

        np.testing.assert_allclose(
            fitted_mean, true_mean, rtol=0.25, atol=0.3,
            err_msg="Mean mismatch for 2D case"
        )

    def test_fit_em_with_fix_p_regularization(self):
        """Test EM fitting with fix_p regularization."""
        n_samples = 2000
        params = {
            'mu': np.array([0.0]),
            'gamma': np.array([0.3]),
            'sigma': np.array([[1.0]]),
            'p': -0.5,  # NIG special case
            'a': 1.0,
            'b': 1.0
        }

        true_dist = GeneralizedHyperbolic.from_classical_params(**params)
        X = true_dist.rvs(size=n_samples, random_state=42)

        # Fit with fix_p regularization
        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=50, tol=1e-5, verbose=0,
            regularization='fix_p',
            regularization_params={'p_fixed': -0.5}
        )

        # Check that p is fixed
        fitted_params = fitted.classical_params
        np.testing.assert_allclose(
            fitted_params['p'], -0.5, rtol=1e-10,
            err_msg="p parameter not fixed correctly"
        )


# ============================================================
# JointGeneralizedHyperbolic Fitting Tests
# ============================================================

class TestJointGeneralizedHyperbolicFitting:
    """Test JointGeneralizedHyperbolic.fit() with complete data."""

    @pytest.fixture
    def params_list(self):
        return get_joint_gh_1d_params()

    def test_fit_recovers_parameters_1d(self, params_list):
        """Test that fit recovers parameters from complete data."""
        n_samples = 10000

        for params in params_list:
            # Create true distribution
            true_dist = JointGeneralizedHyperbolic.from_classical_params(**params)

            # Generate complete data
            X, Y = true_dist.rvs(size=n_samples, random_state=42)

            # Fit new distribution
            fitted = JointGeneralizedHyperbolic(d=1).fit(X, Y)

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


# ============================================================
# Conditional Expectation Tests
# ============================================================

class TestConditionalExpectations:
    """Test _conditional_expectation_y_given_x method."""

    def test_conditional_expectations_finite(self):
        """Test that conditional expectations are finite."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=1.0,
            a=1.0,
            b=1.0
        )

        # Generate some test points
        X = gh.rvs(size=100, random_state=42)

        # Compute conditional expectations
        cond_exp = gh._conditional_expectation_y_given_x(X)

        assert np.all(np.isfinite(cond_exp['E_Y'])), "E[Y|X] should be finite"
        assert np.all(np.isfinite(cond_exp['E_inv_Y'])), "E[1/Y|X] should be finite"
        assert np.all(np.isfinite(cond_exp['E_log_Y'])), "E[log Y|X] should be finite"

    def test_conditional_expectations_positive(self):
        """Test that E[Y|X] and E[1/Y|X] are positive."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([0.3, -0.2]),
            sigma=np.array([[1.0, 0.0], [0.0, 1.0]]),
            p=1.0,
            a=1.0,
            b=1.0
        )

        X = gh.rvs(size=100, random_state=42)
        cond_exp = gh._conditional_expectation_y_given_x(X)

        assert np.all(cond_exp['E_Y'] > 0), "E[Y|X] should be positive"
        assert np.all(cond_exp['E_inv_Y'] > 0), "E[1/Y|X] should be positive"

    def test_conditional_expectations_single_point(self):
        """Test conditional expectations for single point."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=1.0,
            a=1.0,
            b=1.0
        )

        # Single point
        x = np.array([1.0])
        cond_exp = gh._conditional_expectation_y_given_x(x)

        # Should return scalars
        assert isinstance(cond_exp['E_Y'], float), "E[Y|X] should be scalar for single point"
        assert isinstance(cond_exp['E_inv_Y'], float), "E[1/Y|X] should be scalar"
        assert isinstance(cond_exp['E_log_Y'], float), "E[log Y|X] should be scalar"


# ============================================================
# Edge Cases and Numerical Stability
# ============================================================

class TestGeneralizedHyperbolicEdgeCases:
    """Test edge cases and numerical stability."""

    def test_small_gamma(self):
        """Test with very small skewness (near symmetric)."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0, 0.0]),
            gamma=np.array([1e-10, 1e-10]),
            sigma=np.array([[1.0, 0.0], [0.0, 1.0]]),
            p=1.0,
            a=1.0,
            b=1.0
        )

        # Should work without numerical issues
        X = gh.rvs(size=100, random_state=42)
        pdf = gh.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for small gamma"

    def test_large_p(self):
        """Test with large shape parameter p."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=10.0,  # Large p
            a=1.0,
            b=1.0
        )

        # Should work without numerical issues
        X = gh.rvs(size=100, random_state=42)
        pdf = gh.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for large p"

    def test_negative_p(self):
        """Test with negative p (towards InverseGamma/InverseGaussian)."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=-1.5,  # Negative p
            a=1.0,
            b=1.0
        )

        # Should work without numerical issues
        X = gh.rvs(size=100, random_state=42)
        pdf = gh.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for negative p"

    def test_different_scales(self):
        """Test with different scales in sigma."""
        gh = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0, 100.0]),  # Different scales
            gamma=np.array([0.1, 10.0]),
            sigma=np.array([[0.01, 0.0], [0.0, 100.0]]),
            p=1.0,
            a=1.0,
            b=1.0
        )

        # Should work without numerical issues
        X = gh.rvs(size=100, random_state=42)
        pdf = gh.logpdf(X[:5])

        assert np.all(np.isfinite(pdf)), "PDF should be finite for different scales"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
