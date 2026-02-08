"""
Inverse Gamma distribution as an exponential family.

The Inverse Gamma distribution has PDF:

.. math::
    p(x|\\alpha, \\beta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} 
    x^{-\\alpha-1} e^{-\\beta/x}

for :math:`x > 0`, where :math:`\\alpha > 0` is the shape parameter 
and :math:`\\beta > 0` is the rate parameter.

Exponential family form:

- :math:`h(x) = 1` for :math:`x > 0` (base measure)
- :math:`t(x) = [-1/x, \\log x]` (sufficient statistics)
- :math:`\\theta = [\\beta, -(\\alpha+1)]` (natural parameters)
- :math:`\\psi(\\theta) = \\log\\Gamma(-\\theta_2-1) - (-\\theta_2-1)\\log(\\theta_1)` (log partition)

Parametrizations:

- Classical: :math:`\\alpha` (shape), :math:`\\beta` (rate), :math:`\\alpha > 0, \\beta > 0`
- Natural: :math:`\\theta = [\\beta, -(\\alpha+1)]`, :math:`\\theta_1 > 0, \\theta_2 < -1`
- Expectation: :math:`\\eta = [-\\alpha/\\beta, \\log\\beta - \\psi(\\alpha)]`, where 
  :math:`\\psi` is the digamma function

Note: We use rate :math:`\\beta` (like Gamma), while scipy uses scale = :math:`\\beta`.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional
from scipy.special import gammaln, digamma, polygamma

from normix.base import ExponentialFamily


class InverseGamma(ExponentialFamily):
    """
    Inverse Gamma distribution in exponential family form.
    
    The Inverse Gamma distribution has PDF:
    
    .. math::
        p(x|\\alpha, \\beta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)} 
        x^{-\\alpha-1} e^{-\\beta/x}
    
    for :math:`x > 0`, where :math:`\\alpha` is the shape parameter
    and :math:`\\beta` is the rate parameter.
    
    If :math:`X \\sim \\text{Gamma}(\\alpha, \\beta)`, then :math:`1/X \\sim \\text{InvGamma}(\\alpha, \\beta)`.
    
    Parameters
    ----------
    shape : float, optional
        Shape parameter :math:`\\alpha > 0`. Use ``from_classical_params(shape=..., rate=...)``.
    rate : float, optional
        Rate parameter :math:`\\beta > 0`. Use ``from_classical_params(shape=..., rate=...)``.
    
    Attributes
    ----------
    _natural_params : tuple or None
        Internal storage for natural parameters :math:`\\theta = [\\beta, -(\\alpha+1)]`.
    
    Examples
    --------
    >>> # Create from shape and rate parameters
    >>> dist = InverseGamma.from_classical_params(shape=3.0, rate=1.5)
    >>> dist.mean()
    0.75
    
    >>> # Create from natural parameters
    >>> dist = InverseGamma.from_natural_params(np.array([1.5, -4.0]))
    
    >>> # Fit from data
    >>> from scipy.stats import invgamma
    >>> data = invgamma.rvs(a=3.0, scale=1.5, size=1000)
    >>> dist = InverseGamma().fit(data)
    
    See Also
    --------
    Gamma : Inverse of InverseGamma distribution
    GeneralizedInverseGaussian : Generalization including InverseGamma as special case
    
    Notes
    -----
    The Inverse Gamma distribution belongs to the exponential family with:
    
    - Sufficient statistics: :math:`t(x) = [-1/x, \\log x]`
    - Natural parameters: :math:`\\theta = [\\beta, -(\\alpha+1)]`
    - Log partition: :math:`\\psi(\\theta) = \\log\\Gamma(-\\theta_2-1) - (-\\theta_2-1)\\log(\\theta_1)`
    
    The mean exists only for :math:`\\alpha > 1`:
    
    .. math::
        E[X] = \\frac{\\beta}{\\alpha - 1}
    
    The variance exists only for :math:`\\alpha > 2`:
    
    .. math::
        \\text{Var}[X] = \\frac{\\beta^2}{(\\alpha-1)^2(\\alpha-2)}
    
    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.
    """
    
    def __init__(self):
        super().__init__()
        self._shape = None
        self._rate = None

    # ================================================================
    # New interface: internal state management
    # ================================================================

    def _set_from_classical(self, *, shape, rate) -> None:
        """Set internal state from classical parameters."""
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self._shape = float(shape)
        self._rate = float(rate)
        self._natural_params = tuple(np.array([rate, -(shape + 1)]))
        self._fitted = True
        self._invalidate_cache()

    def _set_from_natural(self, theta) -> None:
        """Set internal state from natural parameters."""
        theta = np.asarray(theta)
        self._validate_natural_params(theta)
        self._shape = float(-theta[1] - 1)
        self._rate = float(theta[0])
        self._natural_params = tuple(theta)
        self._fitted = True
        self._invalidate_cache()

    def _compute_natural_params(self):
        """Compute natural parameters from internal state: θ = [β, -(α+1)]."""
        return np.array([self._rate, -(self._shape + 1)])

    def _compute_classical_params(self):
        """Return classical parameters as dict."""
        return {'shape': self._shape, 'rate': self._rate}

    def _get_natural_param_support(self):
        """Natural parameter support: θ₁ > 0, θ₂ < -1."""
        return [(0.0, np.inf), (-np.inf, -1.0)]
    
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Sufficient statistics: t(x) = [-1/x, log(x)].
        
        Returns
        -------
        t : ndarray
            Shape (2,) for scalar input, (n, 2) for array input.
        """
        x = np.asarray(x)
        if x.ndim == 0 or x.shape == ():
            # Scalar input
            return np.array([-1.0/x, np.log(x)])
        else:
            # Array input
            neg_inv_x = -1.0 / x
            log_x = np.log(x)
            return np.column_stack([neg_inv_x, log_x])
    
    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function: psi(theta) = log Gamma(-theta_2-1) - (-theta_2-1)log(theta_1).
        
        .. math::
            \\psi(\\theta) = \\log\\Gamma(-\\theta_2-1) - (-\\theta_2-1)\\log(\\theta_1)
        
        Its gradient gives expectation parameters: :math:`\\nabla\\psi(\\theta) = E[t(X)]`.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector :math:`[\\theta_1, \\theta_2]`.
        
        Returns
        -------
        psi : float
            Log partition function value.
        """
        beta = theta[0]   # β = θ₁
        alpha = -theta[1] - 1  # α = -θ₂ - 1
        return gammaln(alpha) - alpha * np.log(beta)
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Log base measure: log h(x) = 0 for x > 0, -∞ otherwise.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        result[x <= 0] = -np.inf
        return result
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Analytical gradient: η = ∇ψ(θ) = [-α/β, log(β) - ψ(α)].
        
        where ψ is the digamma function and α = -θ₂-1, β = θ₁.
        """
        beta = theta[0]
        alpha = -theta[1] - 1
        
        eta1 = -alpha / beta
        eta2 = np.log(beta) - digamma(alpha)
        
        return np.array([eta1, eta2])
    
    def _get_initial_natural_params(self, eta: NDArray) -> NDArray:
        """
        Get initial guess for natural parameters from expectation parameters.
        
        For Inverse Gamma: η = [-α/β, log(β) - ψ(α)]
        
        We use a more robust iterative approach:
        1. Start with α using the relationship: η₁ ≈ -E[X] where E[X] = β/(α-1)
        2. Iteratively solve the system of equations
        
        Returns θ = [β, -(α+1)]
        """
        # Better initial guess: use approximate relationship
        # Since E[-1/X] = -α/β and E[X] = β/(α-1), we can use:
        # η₁ ≈ -1/E[X] (rough approximation for initial guess)
        # So E[X] ≈ -1/η₁
        
        # Start with: E[X] ≈ -1/η₁, and E[X] = β/(α-1)
        # So β/(α-1) ≈ -1/η₁
        # Use α = 3 initially, solve for β
        alpha = 5.0  # Start with a safe value > 2
        
        # Do more iterations for better convergence
        for outer_iter in range(15):
            # Solve β from -α/β = η₁
            beta = -alpha / eta[0]
            
            # Ensure beta is positive and reasonable
            beta = max(beta, 0.01)
            
            # Update α using Newton's method on: ψ(α) = log(β) - η₂
            target = np.log(beta) - eta[1]
            
            for inner_iter in range(30):
                psi_val = digamma(alpha)
                psi_prime = polygamma(1, alpha)
                
                # Newton step with damping for stability
                delta = (psi_val - target) / psi_prime
                alpha_new = alpha - 0.5 * delta  # Damping factor
                
                # Ensure α stays > 1.5 (we need α > 2 for finite variance)
                alpha_new = max(alpha_new, 2.1)
                
                # Check convergence
                if abs(alpha_new - alpha) < 1e-10:
                    alpha = alpha_new
                    break
                    
                alpha = alpha_new
        
        # Final computation of β
        beta = -alpha / eta[0]
        
        return np.array([beta, -(alpha + 1)])
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Analytical Fisher information: I(θ) = ∇²ψ(θ).
        
        The Hessian is:
        I₁₁ = α/β²
        I₁₂ = I₂₁ = 1/β
        I₂₂ = ψ'(α) (trigamma function)
        
        where α = -θ₂-1, β = θ₁.
        """
        if theta is None:
            theta = self.get_natural_params()
        
        beta = theta[0]
        alpha = -theta[1] - 1
        
        # Compute Hessian components
        I_11 = alpha / (beta**2)
        I_12 = 1.0 / beta
        I_22 = polygamma(1, alpha)  # trigamma(α)
        
        return np.array([[I_11, I_12],
                        [I_12, I_22]])
    
    # Implement required Distribution methods
    
    def rvs(self, size=None, random_state=None):
        """
        Generate random samples from the inverse gamma distribution.
        
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
        self._check_fitted()
        
        # Set up random number generator
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate using scipy parameterization (a, scale)
        # scipy uses scale parameter, we use rate, so scale = rate
        from scipy.stats import invgamma
        return invgamma.rvs(a=self._shape, scale=self._rate, size=size, random_state=rng)
    
    def mean(self) -> float:
        """
        Mean of Inverse Gamma distribution: E[X] = beta/(alpha-1) for alpha > 1.
        
        .. math::
            E[X] = \\frac{\\beta}{\\alpha - 1} \\quad \\text{for } \\alpha > 1
        
        Returns infinity if :math:`\\alpha \\leq 1`.
        
        Returns
        -------
        mean : float
            Mean of the distribution, or infinity if undefined.
        """
        self._check_fitted()
        if self._shape <= 1:
            return np.inf
        return self._rate / (self._shape - 1)
    
    def var(self) -> float:
        """
        Variance of Inverse Gamma distribution for alpha > 2.
        
        .. math::
            \\text{Var}[X] = \\frac{\\beta^2}{(\\alpha-1)^2(\\alpha-2)} \\quad \\text{for } \\alpha > 2
        
        Returns infinity if :math:`\\alpha \\leq 2`.
        
        Returns
        -------
        var : float
            Variance of the distribution, or infinity if undefined.
        """
        self._check_fitted()
        if self._shape <= 2:
            return np.inf
        return (self._rate**2) / ((self._shape - 1)**2 * (self._shape - 2))
    
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
        self._check_fitted()
        
        from scipy.stats import invgamma
        
        x = np.asarray(x)
        
        # CDF: scipy uses scale parameter
        result = invgamma.cdf(x, a=self._shape, scale=self._rate)
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result
