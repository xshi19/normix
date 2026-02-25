"""
Inverse Gaussian distribution (Wald distribution).

Special case of GIG with :math:`p = -1/2`.
Also belongs to the exponential family.

The Inverse Gaussian distribution has PDF:

.. math::
    p(x|\\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}} 
    \\exp\\left(-\\frac{\\lambda(x-\\mu)^2}{2\\mu^2 x}\\right)

for :math:`x > 0`, where :math:`\\mu > 0` is the mean and :math:`\\lambda > 0` 
is the shape parameter.

Exponential family form:

- :math:`h(x) = 1/\\sqrt{2\\pi x^3}` for :math:`x > 0` (base measure)
- :math:`t(x) = [x, 1/x]` (sufficient statistics)
- :math:`\\theta = [-\\lambda/(2\\mu^2), -\\lambda/2]` (natural parameters)
- :math:`\\psi(\\theta) = -2\\sqrt{\\theta_1\\theta_2} - \\frac{1}{2}\\log(-2\\theta_2)` (log partition)

Parametrizations:

- Classical: :math:`\\mu` (mean), :math:`\\lambda` (shape), :math:`\\mu > 0, \\lambda > 0`
- Natural: :math:`\\theta = [-\\lambda/(2\\mu^2), -\\lambda/2]`, :math:`\\theta_1 < 0, \\theta_2 < 0`
- Expectation: :math:`\\eta = [\\mu, 1/\\mu + 1/\\lambda]`

Note: scipy uses (mu, scale) where:

- scipy_mu = :math:`\\mu/\\lambda` (shape parameter, confusingly named)
- scipy_scale = :math:`\\lambda` (our shape parameter)
- Relationship: :math:`\\mu = \\text{scipy\\_mu} \\times \\text{scipy\\_scale}`, 
  :math:`\\lambda = \\text{scipy\\_scale}`
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional

from normix.base import ExponentialFamily
from normix.params import InverseGaussianParams


class InverseGaussian(ExponentialFamily):
    """
    Inverse Gaussian distribution in exponential family form.
    
    Also known as the Wald distribution. Special case of GIG with :math:`p = -1/2`.
    
    The Inverse Gaussian distribution has PDF:
    
    .. math::
        p(x|\\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi x^3}} 
        \\exp\\left(-\\frac{\\lambda(x-\\mu)^2}{2\\mu^2 x}\\right)
    
    for :math:`x > 0`, where :math:`\\mu` is the mean and :math:`\\lambda` 
    is the shape parameter.
    
    Parameters
    ----------
    mean : float, optional
        Mean parameter :math:`\\mu > 0`. Use ``from_classical_params(mean=..., shape=...)``.
    shape : float, optional
        Shape parameter :math:`\\lambda > 0`. Use ``from_classical_params(mean=..., shape=...)``.
    
    Attributes
    ----------
    _natural_params : tuple or None
        Internal storage for natural parameters :math:`\\theta = [-\\lambda/(2\\mu^2), -\\lambda/2]`.
    
    Examples
    --------
    >>> # Create from mean and shape parameters
    >>> dist = InverseGaussian.from_classical_params(mean=1.0, shape=1.0)
    >>> dist.mean()
    1.0
    
    >>> # Create from natural parameters
    >>> dist = InverseGaussian.from_natural_params(np.array([-0.5, -0.5]))
    
    >>> # Fit from data
    >>> from scipy.stats import invgauss
    >>> data = invgauss.rvs(mu=1.0, scale=1.0, size=1000)
    >>> dist = InverseGaussian().fit(data)
    
    See Also
    --------
    GeneralizedInverseGaussian : Generalization with parameter :math:`p`
    
    Notes
    -----
    The Inverse Gaussian distribution belongs to the exponential family with:
    
    - Sufficient statistics: :math:`t(x) = [x, 1/x]`
    - Natural parameters: :math:`\\theta = [-\\lambda/(2\\mu^2), -\\lambda/2]`
    - Log partition: :math:`\\psi(\\theta) = -2\\sqrt{\\theta_1\\theta_2} - \\frac{1}{2}\\log(-2\\theta_2)`
    
    It is a special case of the Generalized Inverse Gaussian (GIG) with :math:`p = -1/2`:
    
    .. math::
        \\text{InvGauss}(\\mu, \\lambda) = \\text{GIG}(p=-1/2, a=\\lambda/\\mu^2, b=\\lambda)
    
    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.

    Chhikara, R. S. & Folks, J. L. (1989). The Inverse Gaussian Distribution.
    """
    
    def __init__(self):
        super().__init__()
        self._mu = None
        self._lambda = None

    # ================================================================
    # New interface: internal state management
    # ================================================================

    def _set_from_classical(self, *, mean, shape) -> None:
        """Set internal state from classical parameters."""
        if mean <= 0:
            raise ValueError(f"Mean must be positive, got {mean}")
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        self._mu = float(mean)
        self._lambda = float(shape)
        theta = np.array([-shape / (2 * mean**2), -shape / 2])
        self._fitted = True
        self._invalidate_cache()

    def _set_from_natural(self, theta) -> None:
        """Set internal state from natural parameters."""
        theta = np.asarray(theta)
        self._validate_natural_params(theta)
        self._lambda = float(-2 * theta[1])
        self._mu = float(np.sqrt(theta[1] / theta[0]))
        self._fitted = True
        self._invalidate_cache()

    def _compute_natural_params(self):
        """Compute natural parameters: θ = [-λ/(2μ²), -λ/2]."""
        return np.array([-self._lambda / (2 * self._mu**2), -self._lambda / 2])

    def _compute_classical_params(self):
        """Return frozen dataclass of classical parameters."""
        return InverseGaussianParams(mean=self._mu, shape=self._lambda)

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
    
    def _compute_expectation_params(self) -> NDArray:
        """Compute expectation parameters directly: η = [μ, 1/μ + 1/λ]."""
        return np.array([self._mu, 1.0 / self._mu + 1.0 / self._lambda])

    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Analytical gradient: η = ∇ψ(θ) = [μ, 1/μ + 1/λ].
        """
        lam = -2 * theta[1]
        mu = np.sqrt(theta[1] / theta[0])
        eta1 = mu
        eta2 = 1.0 / mu + 1.0 / lam
        return np.array([eta1, eta2])
    
    def _expectation_to_natural(self, eta: NDArray, theta0=None) -> NDArray:
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
            theta = self.natural_params
        
        lam = -2 * theta[1]
        mu = np.sqrt(theta[1] / theta[0])
        
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
        self._check_fitted()
        
        # Set up random number generator
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # numpy.random.wald uses the same parameterization:
        # mean = μ, scale = λ
        return rng.wald(mean=self._mu, scale=self._lambda, size=size)
    
    def mean(self) -> float:
        """
        Mean of Inverse Gaussian distribution: E[X] = mu.
        
        .. math::
            E[X] = \\mu
        
        Returns
        -------
        mean : float
            Mean of the distribution.
        """
        self._check_fitted()
        return self._mu
    
    def var(self) -> float:
        """
        Variance of Inverse Gaussian distribution: Var[X] = mu^3/lambda.
        
        .. math::
            \\text{Var}[X] = \\frac{\\mu^3}{\\lambda}
        
        Returns
        -------
        var : float
            Variance of the distribution.
        """
        self._check_fitted()
        return (self._mu**3) / self._lambda
    
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
        
        from scipy.stats import invgauss
        
        x = np.asarray(x)
        
        # scipy.stats.invgauss uses (mu, scale) where:
        # scipy_mu = μ/λ, scipy_scale = λ
        scipy_mu = self._mu / self._lambda
        scipy_scale = self._lambda
        
        result = invgauss.cdf(x, mu=scipy_mu, scale=scipy_scale)
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result
