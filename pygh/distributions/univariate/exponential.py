"""
Exponential distribution as an exponential family.

The exponential distribution has PDF:
    p(x|λ) = λ exp(-λx) for x ≥ 0

Exponential family form:
    - h(x) = 1 for x ≥ 0, 0 otherwise
    - t(x) = x (sufficient statistic)
    - θ = -λ (natural parameter)
    - ψ(θ) = -log(-θ) (log partition function)

Parametrizations:
    - Classical: λ (rate), λ > 0
    - Natural: θ = -λ, θ < 0
    - Expectation: η = E[X] = 1/λ, η > 0
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional

from pygh.base import ExponentialFamily


class Exponential(ExponentialFamily):
    """
    Exponential distribution in exponential family form.
    
    Parameters
    ----------
    rate : float, optional
        Rate parameter λ > 0. Use from_classical_params(rate=λ) to initialize.
    
    Examples
    --------
    >>> # Create from rate parameter
    >>> dist = Exponential.from_classical_params(rate=2.0)
    >>> dist.mean()
    0.5
    
    >>> # Create from natural parameters
    >>> dist = Exponential.from_natural_params(np.array([-2.0]))
    
    >>> # Fit from data
    >>> data = np.random.exponential(scale=0.5, size=1000)
    >>> dist = Exponential().fit(data)
    """
    
    def _get_natural_param_support(self):
        """Natural parameter support: θ < 0."""
        return [(-np.inf, 0.0)]
    
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Sufficient statistic: t(x) = x.
        
        Returns
        -------
        t : ndarray
            Shape (1,) for scalar input, (n, 1) for array input.
        """
        x = np.asarray(x)
        if x.ndim == 0 or x.shape == ():
            # Scalar input
            return np.array([x])
        else:
            # Array input
            return x.reshape(-1, 1)
    
    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function: ψ(θ) = -log(-θ) for θ < 0.
        
        This is the cumulant generating function.
        """
        return -np.log(-theta[0])
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Log base measure: log h(x) = 0 for x ≥ 0, -∞ otherwise.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        result[x < 0] = -np.inf
        return result
    
    def _classical_to_natural(self, **kwargs) -> NDArray:
        """
        Convert rate parameter to natural parameter: θ = -λ.
        """
        rate = kwargs['rate']
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        return np.array([-rate])
    
    def _natural_to_classical(self, theta: NDArray):
        """
        Convert natural parameter to rate: λ = -θ.
        """
        return {'rate': -theta[0]}
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Analytical gradient: η = ∇ψ(θ) = 1/(-θ) = 1/λ.
        
        This is E[X] = 1/λ.
        """
        return np.array([1.0 / (-theta[0])])
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Analytical Fisher information: I(θ) = ∇²ψ(θ) = 1/θ².
        
        For exponential distribution with rate λ:
            I(λ) = 1/λ²
        """
        if theta is None:
            theta = self.get_natural_params()
        return np.array([[1.0 / theta[0]**2]])
    
    # Implement required Distribution methods
    
    def rvs(self, size=None, random_state=None):
        """
        Generate random samples from the exponential distribution.
        
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
        rate = classical['rate']
        
        # Set up random number generator
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate using inverse CDF: X = -log(U) / λ
        u = rng.uniform(size=size)
        return -np.log(u) / rate
    
    def mean(self):
        """
        Mean of exponential distribution: E[X] = 1/λ.
        """
        classical = self.get_classical_params()
        return 1.0 / classical['rate']
    
    def var(self):
        """
        Variance of exponential distribution: Var[X] = 1/λ².
        """
        classical = self.get_classical_params()
        return 1.0 / classical['rate']**2
    
    def cdf(self, x: ArrayLike) -> NDArray:
        """
        Cumulative distribution function: F(x) = 1 - exp(-λx) for x ≥ 0.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate CDF.
        
        Returns
        -------
        cdf : ndarray
            CDF values.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        x = np.asarray(x)
        classical = self.get_classical_params()
        rate = classical['rate']
        
        result = np.zeros_like(x, dtype=float)
        mask = x >= 0
        result[mask] = 1.0 - np.exp(-rate * x[mask])
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result

