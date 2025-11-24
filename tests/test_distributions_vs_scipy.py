"""
Generic test framework for comparing pygh distributions with scipy.

This module provides a reusable test framework for validating exponential family
distributions against scipy implementations.

Tests include:
- PDF and CDF comparison
- Natural ↔ Expectation parameter conversions
- Random sampling statistics
- Histogram vs PDF comparison
"""

import numpy as np
import pytest
from scipy import stats
from typing import Callable, Dict, Any, Tuple

from pygh.distributions.univariate import Exponential, Gamma


# ============================================================
# Generic Test Framework
# ============================================================

class DistributionTestConfig:
    """
    Configuration for testing a distribution against scipy.
    
    Attributes
    ----------
    pygh_class : class
        The pygh distribution class to test.
    scipy_dist : scipy.stats distribution
        The scipy distribution to compare against.
    param_converter : callable
        Function that converts pygh classical params to scipy params.
        Signature: param_converter(classical_params_dict) -> scipy_kwargs
    test_params : list of dict
        List of parameter sets to test.
    test_points : ndarray
        Points at which to evaluate PDF/CDF.
    """
    
    def __init__(
        self,
        pygh_class,
        scipy_dist,
        param_converter: Callable[[Dict[str, Any]], Dict[str, Any]],
        test_params: list,
        test_points: np.ndarray,
        name: str = None
    ):
        self.pygh_class = pygh_class
        self.scipy_dist = scipy_dist
        self.param_converter = param_converter
        self.test_params = test_params
        self.test_points = test_points
        self.name = name or pygh_class.__name__


def compare_pdf_vs_scipy(config: DistributionTestConfig, rtol=1e-6, atol=1e-8):
    """
    Test that PDF matches scipy implementation.
    
    Parameters
    ----------
    config : DistributionTestConfig
        Test configuration.
    rtol : float
        Relative tolerance for comparison.
    atol : float
        Absolute tolerance for comparison.
    """
    for params in config.test_params:
        # Create pygh distribution
        pygh_dist = config.pygh_class.from_classical_params(**params)
        
        # Create scipy distribution
        scipy_params = config.param_converter(params)
        scipy_dist = config.scipy_dist(**scipy_params)
        
        # Compare PDFs
        pygh_pdf = pygh_dist.pdf(config.test_points)
        scipy_pdf = scipy_dist.pdf(config.test_points)
        
        assert np.allclose(pygh_pdf, scipy_pdf, rtol=rtol, atol=atol), \
            f"PDF mismatch for {config.name} with params {params}"


def compare_cdf_vs_scipy(config: DistributionTestConfig, rtol=1e-6, atol=1e-8):
    """
    Test that CDF matches scipy implementation.
    
    Parameters
    ----------
    config : DistributionTestConfig
        Test configuration.
    rtol : float
        Relative tolerance for comparison.
    atol : float
        Absolute tolerance for comparison.
    """
    for params in config.test_params:
        # Create pygh distribution
        pygh_dist = config.pygh_class.from_classical_params(**params)
        
        # Create scipy distribution
        scipy_params = config.param_converter(params)
        scipy_dist = config.scipy_dist(**scipy_params)
        
        # Compare CDFs
        pygh_cdf = pygh_dist.cdf(config.test_points)
        scipy_cdf = scipy_dist.cdf(config.test_points)
        
        assert np.allclose(pygh_cdf, scipy_cdf, rtol=rtol, atol=atol), \
            f"CDF mismatch for {config.name} with params {params}"


def check_natural_expectation_roundtrip(config: DistributionTestConfig, rtol=2e-2, atol=1e-4):
    """
    Test that Natural → Expectation → Natural conversion is consistent.
    
    Parameters
    ----------
    config : DistributionTestConfig
        Test configuration.
    rtol : float
        Relative tolerance for comparison.
    atol : float
        Absolute tolerance for comparison.
    """
    for params in config.test_params:
        # Create distribution
        dist = config.pygh_class.from_classical_params(**params)
        
        # Get natural parameters
        theta_original = dist.get_natural_params()
        
        # Convert to expectation parameters
        eta = dist.get_expectation_params()
        
        # Convert back to natural parameters
        dist_from_eta = config.pygh_class.from_expectation_params(eta)
        theta_roundtrip = dist_from_eta.get_natural_params()
        
        assert np.allclose(theta_original, theta_roundtrip, rtol=rtol, atol=atol), \
            f"Natural→Expectation→Natural roundtrip failed for {config.name} with params {params}"


def check_sample_sufficient_statistics(
    config: DistributionTestConfig,
    n_samples=50000,
    random_state=42,
    rtol=0.05
):
    """
    Test that sample mean of sufficient statistics matches expectation parameters.
    
    Parameters
    ----------
    config : DistributionTestConfig
        Test configuration.
    n_samples : int
        Number of samples to generate.
    random_state : int
        Random seed.
    rtol : float
        Relative tolerance for comparison.
    """
    for params in config.test_params:
        # Create distribution
        dist = config.pygh_class.from_classical_params(**params)
        
        # Get expectation parameters (theoretical)
        eta_theory = dist.get_expectation_params()
        
        # Generate samples
        samples = dist.rvs(size=n_samples, random_state=random_state)
        
        # Compute sufficient statistics
        t_samples = dist._sufficient_statistics(samples)
        
        # Compute sample mean
        eta_sample = np.mean(t_samples, axis=0)
        
        assert np.allclose(eta_sample, eta_theory, rtol=rtol), \
            f"Sample sufficient statistics do not match expectation parameters for {config.name} with params {params}\n" \
            f"Expected: {eta_theory}, Got: {eta_sample}"


def check_histogram_vs_pdf(
    config: DistributionTestConfig,
    n_samples=10000,
    n_bins=50,
    random_state=42,
    plot=False
):
    """
    Test that histogram of samples matches the PDF.
    
    Uses chi-square goodness-of-fit test.
    
    Parameters
    ----------
    config : DistributionTestConfig
        Test configuration.
    n_samples : int
        Number of samples to generate.
    n_bins : int
        Number of histogram bins.
    random_state : int
        Random seed.
    plot : bool
        Whether to generate plots (for debugging).
    """
    for i, params in enumerate(config.test_params):
        # Create distribution
        dist = config.pygh_class.from_classical_params(**params)
        
        # Generate samples
        samples = dist.rvs(size=n_samples, random_state=random_state)
        
        # Create histogram
        counts, bin_edges = np.histogram(samples, bins=n_bins, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Expected counts from PDF
        expected_density = dist.pdf(bin_centers)
        expected_counts = expected_density * bin_width * n_samples
        
        # Remove bins with very low expected counts (< 5 for chi-square)
        mask = expected_counts >= 5
        counts_filtered = counts[mask]
        expected_filtered = expected_counts[mask]
        
        # Chi-square test: χ² = Σ (observed - expected)² / expected
        chi_square = np.sum((counts_filtered - expected_filtered)**2 / expected_filtered)
        dof = len(counts_filtered) - 1  # degrees of freedom
        
        # Critical value at 95% confidence (approximate)
        # For large dof, χ² ~ N(dof, 2*dof)
        critical_value = dof + 3 * np.sqrt(2 * dof)
        
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.hist(samples, bins=n_bins, density=True, alpha=0.7, 
                        label='Sample histogram')
                x_plot = np.linspace(samples.min(), samples.max(), 200)
                plt.plot(x_plot, dist.pdf(x_plot), 'r-', linewidth=2, 
                        label='Theoretical PDF')
                plt.xlabel('x')
                plt.ylabel('Density')
                plt.title(f'{config.name} - Histogram vs PDF\nParams: {params}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f'histogram_{config.name}_{i}.png', dpi=100, bbox_inches='tight')
                plt.close()
            except ImportError:
                print("Warning: matplotlib not available, skipping plot generation")
        
        assert chi_square < critical_value, \
            f"Histogram does not match PDF for {config.name} with params {params}\n" \
            f"Chi-square: {chi_square:.2f}, Critical value: {critical_value:.2f}"


# ============================================================
# Exponential Distribution Tests
# ============================================================

def get_exponential_config() -> DistributionTestConfig:
    """Get test configuration for Exponential distribution."""
    return DistributionTestConfig(
        pygh_class=Exponential,
        scipy_dist=stats.expon,
        param_converter=lambda p: {'scale': 1.0 / p['rate']},  # scipy uses scale = 1/rate
        test_params=[
            {'rate': 0.5},
            {'rate': 1.0},
            {'rate': 2.0},
            {'rate': 5.0},
        ],
        test_points=np.linspace(0, 5, 100),
        name='Exponential'
    )


class TestExponentialVsScipy:
    """Test Exponential distribution against scipy."""
    
    @pytest.fixture
    def config(self):
        return get_exponential_config()
    
    def test_pdf_comparison(self, config):
        """Test PDF matches scipy."""
        compare_pdf_vs_scipy(config)
    
    def test_cdf_comparison(self, config):
        """Test CDF matches scipy."""
        compare_cdf_vs_scipy(config)
    
    def test_parameter_roundtrip(self, config):
        """Test Natural→Expectation→Natural conversion."""
        check_natural_expectation_roundtrip(config)
    
    def test_sample_statistics(self, config):
        """Test sample sufficient statistics match expectation parameters."""
        check_sample_sufficient_statistics(config)
    
    def test_histogram_pdf(self, config):
        """Test histogram matches PDF."""
        check_histogram_vs_pdf(config)


# ============================================================
# Gamma Distribution Tests
# ============================================================

def get_gamma_config() -> DistributionTestConfig:
    """Get test configuration for Gamma distribution."""
    return DistributionTestConfig(
        pygh_class=Gamma,
        scipy_dist=stats.gamma,
        param_converter=lambda p: {'a': p['shape'], 'scale': 1.0 / p['rate']},  # scipy uses (a, scale)
        test_params=[
            {'shape': 1.0, 'rate': 1.0},
            {'shape': 2.0, 'rate': 1.0},
            {'shape': 2.0, 'rate': 2.0},
            {'shape': 5.0, 'rate': 2.0},
        ],
        test_points=np.linspace(0.01, 10, 100),
        name='Gamma'
    )


class TestGammaVsScipy:
    """Test Gamma distribution against scipy."""
    
    @pytest.fixture
    def config(self):
        return get_gamma_config()
    
    def test_pdf_comparison(self, config):
        """Test PDF matches scipy."""
        compare_pdf_vs_scipy(config)
    
    def test_cdf_comparison(self, config):
        """Test CDF matches scipy."""
        compare_cdf_vs_scipy(config)
    
    def test_parameter_roundtrip(self, config):
        """Test Natural→Expectation→Natural conversion."""
        check_natural_expectation_roundtrip(config)
    
    def test_sample_statistics(self, config):
        """Test sample sufficient statistics match expectation parameters."""
        check_sample_sufficient_statistics(config)
    
    def test_histogram_pdf(self, config):
        """Test histogram matches PDF."""
        check_histogram_vs_pdf(config)


# ============================================================
# Standalone test runner
# ============================================================

def run_all_tests_for_distribution(config: DistributionTestConfig, verbose=True):
    """
    Run all tests for a given distribution configuration.
    
    Parameters
    ----------
    config : DistributionTestConfig
        Test configuration.
    verbose : bool
        Whether to print test results.
    """
    tests = [
        ("PDF vs scipy", compare_pdf_vs_scipy),
        ("CDF vs scipy", compare_cdf_vs_scipy),
        ("Natural↔Expectation roundtrip", check_natural_expectation_roundtrip),
        ("Sample sufficient statistics", check_sample_sufficient_statistics),
        ("Histogram vs PDF", check_histogram_vs_pdf),
    ]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {config.name} Distribution")
        print(f"{'='*60}\n")
    
    results = {}
    for test_name, test_func in tests:
        try:
            test_func(config)
            results[test_name] = "PASSED"
            if verbose:
                print(f"✓ {test_name:40s} PASSED")
        except AssertionError as e:
            results[test_name] = f"FAILED: {str(e)}"
            if verbose:
                print(f"✗ {test_name:40s} FAILED")
                print(f"  {str(e)[:100]}")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            if verbose:
                print(f"✗ {test_name:40s} ERROR")
                print(f"  {str(e)[:100]}")
    
    if verbose:
        print(f"\n{'='*60}")
        passed = sum(1 for r in results.values() if r == "PASSED")
        total = len(results)
        print(f"Results: {passed}/{total} tests passed")
        print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Run all tests for all distributions
    configs = [
        get_exponential_config(),
        get_gamma_config(),
    ]
    
    all_passed = True
    for config in configs:
        results = run_all_tests_for_distribution(config)
        if not all(r == "PASSED" for r in results.values()):
            all_passed = False
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ All tests passed for all distributions!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Some tests failed!")
        print("="*60)
        exit(1)

