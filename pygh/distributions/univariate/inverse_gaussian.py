"""
Inverse Gaussian distribution (Wald distribution).

Special case of GIG with p = -1/2.
Also belongs to the exponential family.

The inverse Gaussian distribution has PDF:
    p(x|μ,λ) = √(λ/(2πx³)) * exp(-λ(x-μ)²/(2μ²x)) for x > 0

Exponential family form:
    - h(x) = 1/√(2πx³) for x > 0
    - t(x) = [x, 1/x] (sufficient statistics)
    - θ = [-λ/(2μ²), -λ/2] (natural parameters)
    - ψ(θ) = -2√(θ₁θ₂) - (1/2)log(-2θ₂) (log partition function)

Parametrizations:
    - Classical: μ (mean), λ (shape), μ > 0, λ > 0
    - Natural: θ = [-λ/(2μ²), -λ/2], θ₁ < 0, θ₂ < 0
    - Expectation: η = [μ, 1/μ + 1/λ]

Note: scipy uses (mu, scale) where:
    - scipy_mu = μ/λ (shape parameter, confusingly named)
    - scipy_scale = λ (our shape parameter)
    - Relationship: μ = scipy_mu * scipy_scale, λ = scipy_scale
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional

from pygh.base import ExponentialFamily


class InverseGaussian(ExponentialFamily):
    """
    Inverse Gaussian distribution in exponential family form.
    
    Also known as the Wald distribution.
    
    Parameters
    ----------
    mean : float, optional
        Mean parameter μ > 0. Use from_classical_params(mean=μ, shape=λ).
    shape : float, optional
        Shape parameter λ > 0. Use from_classical_params(mean=μ, shape=λ).
    
    Examples
    --------
    >>> # Create from mean and shape parameters
    >>> dist = InverseGaussian.from_classical_params(mean=1.0, shape=1.0)
    >>> dist.mean()
    1.0
    
    >>> # Create from natural parameters
    >>> dist = InverseGaussian.from_natural_params(np.array([-0.5, -0.5]))
    
    >>> # Fit from data
    >>> import scipy.stats as stats
    >>> data = stats.invgauss.rvs(mu=1.0, scale=1.0, size=1000)
    >>> dist = InverseGaussian().fit(data)
    """
    
    def _get_natural_param_support(self):
        """Natural parameter support: θ₁ < 0, θ₂ < 0."""
        return [(-np.inf, 0.0), (-np.inf, 0.0)]
    
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Sufficient statistics: t(x) = [x, 1/x].
        
        Returns
        -------
        t : ndarray
            Shape (2,) for scalar input, (n, 2) for array input.
        """
        x = np.asarray(x)
        if x.ndim == 0 or x.shape == ():
            # Scalar input
            return np.array([x, 1.0/x])
        else:
            # Array input
            inv_x = 1.0 / x
            return np.column_stack([x, inv_x])
    
    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function: ψ(θ) = -2√(θ₁θ₂) - (1/2)log(-2θ₂).
        
        This matches ψ(θ) = -λ/μ - (1/2)log(λ) in classical parameters.
        """
        sqrt_product = np.sqrt(theta[0] * theta[1])
        lam = -2.0 * theta[1]
        return -2.0 * sqrt_product - 0.5 * np.log(lam)
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Log base measure: log h(x) = -1/2 * log(2πx³) for x > 0.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = x > 0
        result[mask] = -0.5 * np.log(2 * np.pi * x[mask]**3)
        result[~mask] = -np.inf
        return result
    
    def _classical_to_natural(self, **kwargs) -> NDArray:
        """
        Convert mean and shape parameters to natural parameters.
        
        θ = [-λ/(2μ²), -λ/2]
        """
        mean = kwargs['mean']
        shape = kwargs['shape']
        
        if mean <= 0:
            raise ValueError(f"Mean must be positive, got {mean}")
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        
        theta1 = -shape / (2 * mean**2)
        theta2 = -shape / 2
        
        return np.array([theta1, theta2])
    
    def _natural_to_classical(self, theta: NDArray):
        """
        Convert natural parameters to mean and shape: μ, λ.
        
        From θ = [-λ/(2μ²), -λ/2]:
        - λ = -2θ₂
        - μ² = -λ/(2θ₁) = θ₂/θ₁
        """
        lam = -2 * theta[1]
        mu_squared = theta[1] / theta[0]
        mu = np.sqrt(mu_squared)
        
        return {'mean': mu, 'shape': lam}
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Analytical gradient: η = ∇ψ(θ) = [μ, 1/μ + 1/λ].
        """
        params = self._natural_to_classical(theta)
        mu = params['mean']
        lam = params['shape']
        eta1 = mu
        eta2 = 1.0 / mu + 1.0 / lam
        return np.array([eta1, eta2])
    
    def _expectation_to_natural(self, eta: NDArray) -> NDArray:
        """
        Analytical inverse from expectation parameters η = [μ, 1/μ + 1/λ].
        """
        mu = eta[0]
        denom = eta[1] - 1.0 / mu
        if denom <= 0:
            raise ValueError("Invalid expectation parameters for Inverse Gaussian.")
        lam = 1.0 / denom
        theta1 = -lam / (2 * mu**2)
        theta2 = -lam / 2
        return np.array([theta1, theta2])
    
    def _get_initial_natural_params(self, eta: NDArray) -> NDArray:
        """
        Get initial guess for natural parameters from expectation parameters.
        
        Uses the analytical inverse.
        """
        return self._expectation_to_natural(eta)
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Analytical Fisher information: I(θ) = ∇²ψ(θ).
        """
        if theta is None:
            theta = self.get_natural_params()
        
        params = self._natural_to_classical(theta)
        mu = params['mean']
        lam = params['shape']
        
        I_11 = mu**3 / lam
        I_12 = I_21 = -mu / lam
        I_22 = 1.0 / (mu * lam) + 2.0 / (lam**2)
        
        return np.array([[I_11, I_12],
                        [I_21, I_22]])
    
    # Implement required Distribution methods
    
    def rvs(self, size=None, random_state=None):
        """
        Generate random samples from the inverse Gaussian distribution.
        
        Uses numpy's wald distribution which has the same parameterization.
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Shape of samples to generate.
        random_state : int or Generator, optional
            Random number generator seed or instance.
        
        Returns
        -------
        samples : float or ndarray
            Random samples from the distribution.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        classical = self.get_classical_params()
        mu = classical['mean']
        lam = classical['shape']
        
        # Set up random number generator
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # numpy.random.wald uses the same parameterization:
        # mean = μ, scale = λ
        return rng.wald(mean=mu, scale=lam, size=size)
    
    def mean(self):
        """
        Mean of inverse Gaussian distribution: E[X] = μ.
        """
        classical = self.get_classical_params()
        return classical['mean']
    
    def var(self):
        """
        Variance of inverse Gaussian distribution: Var[X] = μ³/λ.
        """
        classical = self.get_classical_params()
        mu = classical['mean']
        lam = classical['shape']
        return (mu**3) / lam
    
    def cdf(self, x: ArrayLike) -> NDArray:
        """
        Cumulative distribution function using scipy's implementation.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate CDF.
        
        Returns
        -------
        cdf : ndarray or float
            CDF values.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        from scipy.stats import invgauss
        
        x = np.asarray(x)
        classical = self.get_classical_params()
        mu = classical['mean']
        lam = classical['shape']
        
        # scipy.stats.invgauss uses (mu, scale) where:
        # scipy_mu = μ/λ, scipy_scale = λ
        scipy_mu = mu / lam
        scipy_scale = lam
        
        result = invgauss.cdf(x, mu=scipy_mu, scale=scipy_scale)
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result
