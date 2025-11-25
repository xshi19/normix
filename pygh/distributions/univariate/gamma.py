"""
Gamma distribution as an exponential family.

The gamma distribution has PDF:
    p(x|α,β) = (β^α / Γ(α)) x^(α-1) exp(-βx) for x > 0

Exponential family form:
    - h(x) = 1 for x > 0, 0 otherwise
    - t(x) = [log(x), x] (sufficient statistics)
    - θ = [α-1, -β] (natural parameters)
    - ψ(θ) = log Γ(θ₁+1) - (θ₁+1)log(-θ₂) (log partition function)

Parametrizations:
    - Classical: α (shape), β (rate), α > 0, β > 0
    - Natural: θ = [α-1, -β], θ₁ > -1, θ₂ < 0
    - Expectation: η = [ψ(α) - log(β), α/β], where ψ is digamma function
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional
from scipy.special import gammaln, digamma, polygamma

from pygh.base import ExponentialFamily


class Gamma(ExponentialFamily):
    """
    Gamma distribution in exponential family form.
    
    Parameters
    ----------
    shape : float, optional
        Shape parameter α > 0. Use from_classical_params(shape=α, rate=β).
    rate : float, optional
        Rate parameter β > 0. Use from_classical_params(shape=α, rate=β).
    
    Examples
    --------
    >>> # Create from shape and rate parameters
    >>> dist = Gamma.from_classical_params(shape=2.0, rate=1.0)
    >>> dist.mean()
    2.0
    
    >>> # Create from natural parameters
    >>> dist = Gamma.from_natural_params(np.array([1.0, -1.0]))
    
    >>> # Fit from data
    >>> data = np.random.gamma(shape=2.0, scale=1.0, size=1000)
    >>> dist = Gamma().fit(data)
    """
    
    def _get_natural_param_support(self):
        """Natural parameter support: θ₁ > -1, θ₂ < 0."""
        return [(-1.0, np.inf), (-np.inf, 0.0)]
    
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Sufficient statistics: t(x) = [log(x), x].
        
        Returns
        -------
        t : ndarray
            Shape (2,) for scalar input, (n, 2) for array input.
        """
        x = np.asarray(x)
        if x.ndim == 0 or x.shape == ():
            # Scalar input
            return np.array([np.log(x), x])
        else:
            # Array input
            log_x = np.log(x)
            return np.column_stack([log_x, x])
    
    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function: ψ(θ) = log Γ(θ₁+1) - (θ₁+1)log(-θ₂).
        
        This is the cumulant generating function.
        """
        alpha = theta[0] + 1  # α = θ₁ + 1
        beta = -theta[1]       # β = -θ₂
        return gammaln(alpha) - alpha * np.log(beta)
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Log base measure: log h(x) = 0 for x > 0, -∞ otherwise.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        result[x <= 0] = -np.inf
        return result
    
    def _classical_to_natural(self, **kwargs) -> NDArray:
        """
        Convert shape and rate parameters to natural parameters: θ = [α-1, -β].
        """
        shape = kwargs['shape']
        rate = kwargs['rate']
        
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        
        return np.array([shape - 1, -rate])
    
    def _natural_to_classical(self, theta: NDArray):
        """
        Convert natural parameters to shape and rate: α = θ₁+1, β = -θ₂.
        """
        shape = theta[0] + 1
        rate = -theta[1]
        return {'shape': shape, 'rate': rate}
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Analytical gradient: η = ∇ψ(θ) = [ψ(α) - log(β), α/β].
        
        where ψ is the digamma function and α = θ₁+1, β = -θ₂.
        """
        alpha = theta[0] + 1
        beta = -theta[1]
        
        eta1 = digamma(alpha) - np.log(beta)
        eta2 = alpha / beta
        
        return np.array([eta1, eta2])
    
    def _expectation_to_natural(self, eta: NDArray) -> NDArray:
        """
        Analytical inverse using Newton's method.
        
        From η = [ψ(α) - log(β), α/β]:
        - η₁ = α/β → β = α/η₁
        - η₀ = ψ(α) - log(β) → ψ(α) = η₀ + log(β)
        
        We solve for α using Newton's method with β = α/η₁.
        
        Parameters
        ----------
        eta : ndarray
            Expectation parameters [E[log X], E[X]].
        
        Returns
        -------
        theta : ndarray
            Natural parameters [α-1, -β].
        """
        # Use η₁ = α/β to express β in terms of α
        # Then solve: ψ(α) = η₀ + log(α/η₁) = η₀ + log(α) - log(η₁)
        
        # Initial guess for α
        alpha = max(eta[1], 2.0)
        
        # Newton's method to solve: ψ(α) - log(α) = η₀ - log(η₁)
        target = eta[0] - np.log(eta[1])
        
        for _ in range(100):  # Increase iterations for better convergence
            psi_val = digamma(alpha)
            psi_prime = polygamma(1, alpha)
            
            # Current function value: f(α) = ψ(α) - log(α) - target
            f_val = psi_val - np.log(alpha) - target
            
            # Derivative: f'(α) = ψ'(α) - 1/α
            f_prime = psi_prime - 1.0 / alpha
            
            # Newton step
            alpha_new = alpha - f_val / f_prime
            
            # Keep α positive
            alpha_new = max(alpha_new, 0.5)
            
            # Check convergence
            if abs(alpha_new - alpha) / abs(alpha) < 1e-12:
                alpha = alpha_new
                break
                
            alpha = alpha_new
        
        # Compute β from α/β = η₁
        beta = alpha / eta[1]
        
        return np.array([alpha - 1, -beta])
    
    def _get_initial_natural_params(self, eta: NDArray) -> NDArray:
        """
        Get initial guess for natural parameters from expectation parameters.
        
        Uses the analytical inverse (Newton's method).
        """
        return self._expectation_to_natural(eta)
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Analytical Fisher information: I(θ) = ∇²ψ(θ).
        
        The Hessian is:
        I₁₁ = ψ'(α) (trigamma function)
        I₁₂ = I₂₁ = 1/β
        I₂₂ = α/β²
        
        where α = θ₁+1, β = -θ₂.
        """
        if theta is None:
            theta = self.get_natural_params()
        
        alpha = theta[0] + 1
        beta = -theta[1]
        
        # Compute Hessian components
        I_11 = polygamma(1, alpha)  # trigamma(α)
        I_12 = 1.0 / beta
        I_22 = alpha / (beta**2)
        
        return np.array([[I_11, I_12],
                        [I_12, I_22]])
    
    # Implement required Distribution methods
    
    def rvs(self, size=None, random_state=None):
        """
        Generate random samples from the gamma distribution.
        
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
        shape = classical['shape']
        rate = classical['rate']
        
        # Set up random number generator
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate using numpy's gamma (with scale = 1/rate)
        return rng.gamma(shape=shape, scale=1/rate, size=size)
    
    def mean(self):
        """
        Mean of gamma distribution: E[X] = α/β.
        """
        classical = self.get_classical_params()
        return classical['shape'] / classical['rate']
    
    def var(self):
        """
        Variance of gamma distribution: Var[X] = α/β².
        """
        classical = self.get_classical_params()
        return classical['shape'] / (classical['rate']**2)
    
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
        
        from scipy.special import gammainc
        
        x = np.asarray(x)
        classical = self.get_classical_params()
        shape = classical['shape']
        rate = classical['rate']
        
        # CDF: P(X ≤ x) = gammainc(α, βx) (regularized lower incomplete gamma)
        result = np.where(x > 0, gammainc(shape, rate * x), 0.0)
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result

