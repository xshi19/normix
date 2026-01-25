"""
Tests for the ExponentialFamily base class.

This test suite uses a simple exponential distribution as a concrete example
to test the exponential family framework.
"""

import numpy as np
import pytest

from pygh.base import ExponentialFamily


# ============================================================
# Simple Exponential Distribution as Test Case
# ============================================================

class Exponential(ExponentialFamily):
    """
    Exponential distribution as exponential family.
    
    PDF: p(x|λ) = λ exp(-λx) for x ≥ 0
    
    Exponential family form:
    - h(x) = 1 for x ≥ 0
    - t(x) = x (sufficient statistic)
    - θ = -λ (natural parameter)
    - ψ(θ) = -log(-θ) (log partition function)
    
    Classical: λ (rate), λ > 0
    Natural: θ = -λ, θ < 0
    Expectation: η = E[X] = 1/λ, η > 0
    """
    
    def _get_natural_param_support(self):
        """θ < 0"""
        return [(-np.inf, 0.0)]
    
    def _sufficient_statistics(self, x):
        """t(x) = x"""
        x = np.asarray(x)
        if x.ndim == 0 or x.shape == ():
            # Scalar
            return np.array([x])
        else:
            # Array - return as (n_samples, 1)
            return x.reshape(-1, 1)
    
    def _log_partition(self, theta):
        """ψ(θ) = -log(-θ) for θ < 0"""
        return -np.log(-theta[0])
    
    def _log_base_measure(self, x):
        """log h(x) = 0 for x ≥ 0, -∞ otherwise"""
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        result[x < 0] = -np.inf
        return result
    
    def _classical_to_natural(self, **kwargs):
        """λ → θ = -λ"""
        rate = kwargs['rate']
        return np.array([-rate])
    
    def _natural_to_classical(self, theta):
        """θ → λ = -θ"""
        return {'rate': -theta[0]}
    
    # Override with analytical gradient for efficiency
    def _natural_to_expectation(self, theta):
        """η = ∇ψ(θ) = 1/|θ| = 1/λ"""
        return np.array([-1.0 / theta[0]])
    
    # Override with analytical inverse for efficiency
    def _expectation_to_natural(self, eta):
        """θ = -1/η (inverse of natural_to_expectation)"""
        return np.array([-1.0 / eta[0]])
    
    # Override with analytical Hessian for efficiency
    def fisher_information(self, theta=None):
        """I(θ) = ∇²ψ(θ) = 1/θ²"""
        if theta is None:
            theta = self.get_natural_params()
        return np.array([[1.0 / theta[0]**2]])
    
    # Implement required methods from Distribution
    def rvs(self, size=None, random_state=None):
        """Generate random samples"""
        if self._natural_params is None:
            raise ValueError("Parameters not set")
        
        classical = self.get_classical_params()
        rate = classical['rate']
        
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Exponential via inverse CDF
        u = rng.uniform(size=size)
        return -np.log(u) / rate
    
    def mean(self):
        """Mean = 1/λ"""
        classical = self.get_classical_params()
        return 1.0 / classical['rate']
    
    def var(self):
        """Variance = 1/λ²"""
        classical = self.get_classical_params()
        return 1.0 / classical['rate']**2


# ============================================================
# Tests
# ============================================================

class TestExponentialDistribution:
    """Test Exponential distribution as exponential family."""
    
    def test_from_classical_params(self):
        """Test initialization from classical parameters."""
        dist = Exponential.from_classical_params(rate=2.0)
        
        classical = dist.get_classical_params()
        assert classical['rate'] == 2.0
        
        # Check conversion to natural
        natural = dist.get_natural_params()
        assert np.isclose(natural[0], -2.0)
    
    def test_from_natural_params(self):
        """Test initialization from natural parameters."""
        dist = Exponential.from_natural_params(np.array([-2.0]))
        
        natural = dist.get_natural_params()
        assert natural[0] == -2.0
        
        # Check conversion to classical
        classical = dist.get_classical_params()
        assert np.isclose(classical['rate'], 2.0)
    
    def test_from_expectation_params(self):
        """Test initialization from expectation parameters."""
        # E[X] = 0.5 means rate = 2.0
        dist = Exponential.from_expectation_params(np.array([0.5]))
        
        expectation = dist.get_expectation_params()
        assert np.isclose(expectation[0], 0.5, atol=1e-4)
        
        # Check conversion to classical
        classical = dist.get_classical_params()
        assert np.isclose(classical['rate'], 2.0, atol=1e-3)
    
    def test_parameter_conversions(self):
        """Test conversions between all parametrizations."""
        # Start with classical λ = 2.0
        dist = Exponential.from_classical_params(rate=2.0)
        
        # Classical → Natural: θ = -λ = -2.0
        natural = dist.get_natural_params()
        assert np.isclose(natural[0], -2.0)
        
        # Natural → Expectation: η = 1/λ = 0.5
        expectation = dist.get_expectation_params()
        assert np.isclose(expectation[0], 0.5)
        
        # Back to classical
        classical = dist.get_classical_params()
        assert np.isclose(classical['rate'], 2.0)
    
    def test_set_params(self):
        """Test setting parameters after initialization."""
        dist = Exponential()
        
        # Set classical
        dist.set_classical_params(rate=3.0)
        assert dist.get_classical_params()['rate'] == 3.0
        
        # Set natural
        dist.set_natural_params(np.array([-1.5]))
        assert np.isclose(dist.get_classical_params()['rate'], 1.5)
        
        # Set expectation
        dist.set_expectation_params(np.array([2.0]))
        assert np.isclose(dist.get_classical_params()['rate'], 0.5, atol=1e-4)
    
    def test_pdf_single_value(self):
        """Test PDF at a single point."""
        dist = Exponential.from_classical_params(rate=2.0)
        
        # PDF at x=1: p(1) = 2 * exp(-2) ≈ 0.2707
        pdf_val = dist.pdf(1.0)
        expected = 2.0 * np.exp(-2.0)
        assert np.isclose(pdf_val, expected)
    
    def test_pdf_array(self):
        """Test PDF with array input."""
        dist = Exponential.from_classical_params(rate=1.0)
        
        x = np.array([0.0, 1.0, 2.0])
        pdf_vals = dist.pdf(x)
        expected = np.exp(-x)
        assert np.allclose(pdf_vals, expected)
    
    def test_logpdf(self):
        """Test log PDF."""
        dist = Exponential.from_classical_params(rate=2.0)
        
        # log p(1) = log(2) - 2
        logpdf_val = dist.logpdf(1.0)
        expected = np.log(2.0) - 2.0
        assert np.isclose(logpdf_val, expected)
    
    def test_pdf_negative_values(self):
        """Test that PDF is zero for negative values."""
        dist = Exponential.from_classical_params(rate=1.0)
        
        pdf_val = dist.pdf(-1.0)
        assert np.isclose(pdf_val, 0.0)
    
    def test_mean(self):
        """Test mean calculation."""
        dist = Exponential.from_classical_params(rate=2.0)
        assert np.isclose(dist.mean(), 0.5)
    
    def test_variance(self):
        """Test variance calculation."""
        dist = Exponential.from_classical_params(rate=2.0)
        assert np.isclose(dist.var(), 0.25)
    
    def test_fisher_information(self):
        """Test Fisher information matrix."""
        dist = Exponential.from_classical_params(rate=2.0)
        
        # For exponential: I(θ) = 1/θ² = 1/4 when θ = -2
        fisher = dist.fisher_information()
        expected = np.array([[0.25]])
        assert np.allclose(fisher, expected)
    
    def test_rvs_shape(self):
        """Test random sampling shape."""
        dist = Exponential.from_classical_params(rate=1.0)
        
        # Single sample
        sample = dist.rvs(random_state=42)
        assert isinstance(sample, (float, np.floating))
        
        # Multiple samples
        samples = dist.rvs(size=100, random_state=42)
        assert samples.shape == (100,)
        
        # 2D samples
        samples_2d = dist.rvs(size=(10, 5), random_state=42)
        assert samples_2d.shape == (10, 5)
    
    def test_rvs_statistics(self):
        """Test that random samples have correct statistics."""
        dist = Exponential.from_classical_params(rate=2.0)
        
        # Generate large sample
        samples = dist.rvs(size=10000, random_state=42)
        
        # Check mean (should be close to 0.5)
        assert np.abs(np.mean(samples) - 0.5) < 0.05
        
        # Check variance (should be close to 0.25)
        assert np.abs(np.var(samples) - 0.25) < 0.05
        
        # All samples should be non-negative
        assert np.all(samples >= 0)
    
    def test_fit(self):
        """Test MLE fitting."""
        # Generate data with known parameter
        true_rate = 3.0
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1/true_rate, size=1000)
        
        # Fit distribution
        dist = Exponential().fit(data)
        
        # Check that fitted rate is close to true rate
        fitted = dist.get_classical_params()
        assert np.abs(fitted['rate'] - true_rate) < 0.2
    
    def test_fit_returns_self(self):
        """Test that fit() returns self for chaining."""
        data = np.array([1.0, 2.0, 3.0])
        dist = Exponential()
        result = dist.fit(data)
        assert result is dist
    
    def test_score(self):
        """Test log-likelihood scoring."""
        dist = Exponential.from_classical_params(rate=1.0)
        
        data = np.array([1.0, 2.0, 3.0])
        score = dist.score(data)
        
        # Score should be mean log-likelihood
        expected = np.mean(dist.logpdf(data))
        assert np.isclose(score, expected)
    
    def test_repr_without_params(self):
        """Test string representation without parameters."""
        dist = Exponential()
        repr_str = repr(dist)
        assert "not fitted" in repr_str
    
    def test_repr_with_params(self):
        """Test string representation with parameters."""
        dist = Exponential.from_classical_params(rate=2.0)
        repr_str = repr(dist)
        assert "Exponential" in repr_str
        assert "rate" in repr_str
        assert "2.0" in repr_str
    
    def test_validation_natural_params(self):
        """Test that invalid natural parameters are rejected."""
        # Exponential requires θ < 0
        with pytest.raises(ValueError, match="outside support"):
            Exponential.from_natural_params(np.array([1.0]))  # Invalid: θ > 0
    
    def test_parameter_caching(self):
        """Test that parameter conversions are cached."""
        dist = Exponential.from_classical_params(rate=2.0)
        
        # First call
        exp1 = dist.get_expectation_params()
        # Second call (should be cached)
        exp2 = dist.get_expectation_params()
        
        assert exp1 == exp2
        assert exp1[0] == exp2[0]
    
    def test_method_chaining(self):
        """Test method chaining."""
        data = np.random.exponential(scale=0.5, size=100)
        
        # Chain fit and score
        dist = Exponential().fit(data)
        score = dist.score(data)
        
        assert score is not None
        assert dist.get_classical_params()['rate'] > 0


class TestAbstractMethods:
    """Test that abstract methods are enforced."""
    
    def test_cannot_instantiate_exponential_family_directly(self):
        """Test that ExponentialFamily cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExponentialFamily()


class TestNumericalDifferentiation:
    """Test numerical differentiation fallbacks."""
    
    def test_numerical_gradient(self):
        """Test that numerical gradient works when not overridden."""
        
        class ExponentialNoAnalytical(Exponential):
            """Exponential without analytical gradient."""
            
            # Remove the analytical override to force numerical differentiation
            pass
        
        # Remove the analytical method from the class
        if '_natural_to_expectation' in ExponentialNoAnalytical.__dict__:
            delattr(ExponentialNoAnalytical, '_natural_to_expectation')
        
        dist = ExponentialNoAnalytical.from_classical_params(rate=2.0)
        
        # Should still work with numerical differentiation
        expectation = dist.get_expectation_params()
        assert isinstance(expectation, np.ndarray)
        assert len(expectation) == 1
        assert np.isclose(expectation[0], 0.5, atol=1e-4)
    
    def test_numerical_hessian(self):
        """Test that numerical Hessian works when not overridden."""
        
        class ExponentialNoAnalytical(Exponential):
            """Exponential without analytical Hessian."""
            
            # Don't override fisher_information
            # Force use of numerical differentiation
            def fisher_information(self, *theta):
                # Call parent's numerical implementation
                return ExponentialFamily.fisher_information(self, *theta)
        
        dist = ExponentialNoAnalytical.from_classical_params(rate=2.0)
        
        # Should still work with numerical differentiation
        fisher = dist.fisher_information()
        expected = np.array([[0.25]])
        assert np.allclose(fisher, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
