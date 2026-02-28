"""
Inverse Gaussian distribution (Wald distribution).

Special case of GIG with :math:`p = -1/2`.
Also belongs to the exponential family.

The Inverse Gaussian distribution has PDF:

.. math::
    p(x|\\delta, \\eta) = \\sqrt{\\frac{\\eta}{2\\pi x^3}} 
    \\exp\\left(-\\frac{\\eta(x-\\delta)^2}{2\\delta^2 x}\\right)

for :math:`x > 0`, where :math:`\\delta > 0` is the mean and :math:`\\eta > 0` 
is the shape parameter.

Exponential family form:

- :math:`h(x) = 1/\\sqrt{2\\pi x^3}` for :math:`x > 0` (base measure)
- :math:`t(x) = [x, 1/x]` (sufficient statistics)
- :math:`\\theta = [-\\eta/(2\\delta^2), -\\eta/2]` (natural parameters)
- :math:`\\psi(\\theta) = -2\\sqrt{\\theta_1\\theta_2} - \\frac{1}{2}\\log(-2\\theta_2)` (log partition)

Parametrizations:

- Classical: :math:`\\delta` (mean), :math:`\\eta` (shape), :math:`\\delta > 0, \\eta > 0`
- Natural: :math:`\\theta = [-\\eta/(2\\delta^2), -\\eta/2]`, :math:`\\theta_1 < 0, \\theta_2 < 0`
- Expectation: :math:`\\eta_{exp} = [\\delta, 1/\\delta + 1/\\eta]`

Note: scipy uses (mu, scale) where:

- scipy_mu = :math:`\\delta/\\eta` (shape parameter, confusingly named)
- scipy_scale = :math:`\\eta` (our shape parameter)
- Relationship: :math:`\\delta = \\text{scipy\\_mu} \\times \\text{scipy\\_scale}`, 
  :math:`\\eta = \\text{scipy\\_scale}`
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
        p(x|\\delta, \\eta) = \\sqrt{\\frac{\\eta}{2\\pi x^3}} 
        \\exp\\left(-\\frac{\\eta(x-\\delta)^2}{2\\delta^2 x}\\right)
    
    for :math:`x > 0`, where :math:`\\delta` is the mean and :math:`\\eta` 
    is the shape parameter.
    
    Parameters
    ----------
    delta : float, optional
        Mean parameter :math:`\\delta > 0`. Use ``from_classical_params(delta=..., eta=...)``.
    eta : float, optional
        Shape parameter :math:`\\eta > 0`. Use ``from_classical_params(delta=..., eta=...)``.
    
    Examples
    --------
    >>> # Create from classical parameters
    >>> dist = InverseGaussian.from_classical_params(delta=1.0, eta=1.0)
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
    - Natural parameters: :math:`\\theta = [-\\eta/(2\\delta^2), -\\eta/2]`
    - Log partition: :math:`\\psi(\\theta) = -2\\sqrt{\\theta_1\\theta_2} - \\frac{1}{2}\\log(-2\\theta_2)`
    
    It is a special case of the Generalized Inverse Gaussian (GIG) with :math:`p = -1/2`:
    
    .. math::
        \\text{InvGauss}(\\delta, \\eta) = \\text{GIG}(p=-1/2, a=\\eta/\\delta^2, b=\\eta)
    
    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.

    Chhikara, R. S. & Folks, J. L. (1989). The Inverse Gaussian Distribution.
    """
    
    def __init__(self):
        super().__init__()
        self._delta = None
        self._eta = None

    # ================================================================
    # New interface: internal state management
    # ================================================================

    def _set_from_classical(self, *, delta, eta) -> None:
        """Set internal state from classical parameters."""
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        self._delta = float(delta)
        self._eta = float(eta)
        self._fitted = True
        self._invalidate_cache()

    def _set_from_natural(self, theta) -> None:
        """Set internal state from natural parameters."""
        theta = np.asarray(theta)
        self._validate_natural_params(theta)
        self._eta = float(-2 * theta[1])
        self._delta = float(np.sqrt(theta[1] / theta[0]))
        self._fitted = True
        self._invalidate_cache()

    def _compute_natural_params(self):
        """Compute natural parameters: theta = [-eta/(2*delta^2), -eta/2]."""
        return np.array([-self._eta / (2 * self._delta**2), -self._eta / 2])

    def _compute_classical_params(self):
        """Return frozen dataclass of classical parameters."""
        return InverseGaussianParams(delta=self._delta, eta=self._eta)

    def _get_natural_param_support(self):
        """Natural parameter support: theta_1 < 0, theta_2 < 0."""
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
            return np.array([x, 1.0/x])
        else:
            inv_x = 1.0 / x
            return np.column_stack([x, inv_x])
    
    def _log_partition(self, theta: NDArray) -> float:
        r"""
        Log partition function: :math:`\psi(\theta) = -2\sqrt{\theta_1\theta_2} - \frac{1}{2}\log(-2\theta_2)`.
        
        This matches :math:`\psi(\theta) = -\eta/\delta - \frac{1}{2}\log(\eta)` in classical parameters.
        """
        sqrt_product = np.sqrt(theta[0] * theta[1])
        eta = -2.0 * theta[1]
        return -2.0 * sqrt_product - 0.5 * np.log(eta)
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        r"""
        Log base measure: :math:`\log h(x) = -\frac{1}{2} \log(2\pi x^3)` for :math:`x > 0`.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        mask = x > 0
        result[mask] = -0.5 * np.log(2 * np.pi * x[mask]**3)
        result[~mask] = -np.inf
        return result
    
    def _compute_expectation_params(self) -> NDArray:
        r"""Compute expectation parameters directly: :math:`\eta_{exp} = [\delta, 1/\delta + 1/\eta]`."""
        return np.array([self._delta, 1.0 / self._delta + 1.0 / self._eta])

    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        r"""
        Analytical gradient: :math:`\eta_{exp} = \nabla\psi(\theta) = [\delta, 1/\delta + 1/\eta]`.
        """
        eta = -2 * theta[1]
        delta = np.sqrt(theta[1] / theta[0])
        eta1 = delta
        eta2 = 1.0 / delta + 1.0 / eta
        return np.array([eta1, eta2])
    
    def _expectation_to_natural(self, eta_exp: NDArray, theta0=None) -> NDArray:
        r"""
        Analytical inverse from expectation parameters :math:`\eta_{exp} = [\delta, 1/\delta + 1/\eta]`.
        """
        delta = eta_exp[0]
        denom = eta_exp[1] - 1.0 / delta
        if denom <= 0:
            raise ValueError("Invalid expectation parameters for Inverse Gaussian.")
        eta = 1.0 / denom
        theta1 = -eta / (2 * delta**2)
        theta2 = -eta / 2
        return np.array([theta1, theta2])
    
    def _get_initial_natural_params(self, eta_exp: NDArray) -> NDArray:
        """
        Get initial guess for natural parameters from expectation parameters.
        
        Uses the analytical inverse.
        """
        return self._expectation_to_natural(eta_exp)
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        r"""
        Analytical Fisher information: :math:`I(\theta) = \nabla^2\psi(\theta)`.
        """
        if theta is None:
            theta = self.natural_params
        
        eta = -2 * theta[1]
        delta = np.sqrt(theta[1] / theta[0])
        
        I_11 = delta**3 / eta
        I_12 = I_21 = -delta / eta
        I_22 = 1.0 / (delta * eta) + 2.0 / (eta**2)
        
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
        
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # numpy.random.wald: mean = delta, scale = eta
        return rng.wald(mean=self._delta, scale=self._eta, size=size)
    
    def mean(self) -> float:
        r"""
        Mean of Inverse Gaussian distribution: :math:`E[X] = \delta`.
        
        Returns
        -------
        mean : float
            Mean of the distribution.
        """
        self._check_fitted()
        return self._delta
    
    def var(self) -> float:
        r"""
        Variance of Inverse Gaussian distribution: :math:`\text{Var}[X] = \delta^3/\eta`.
        
        Returns
        -------
        var : float
            Variance of the distribution.
        """
        self._check_fitted()
        return (self._delta**3) / self._eta
    
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
        # scipy_mu = delta/eta, scipy_scale = eta
        scipy_mu = self._delta / self._eta
        scipy_scale = self._eta
        
        result = invgauss.cdf(x, mu=scipy_mu, scale=scipy_scale)
        
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result
