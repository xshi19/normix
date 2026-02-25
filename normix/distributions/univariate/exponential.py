"""
Exponential distribution as an exponential family.

The Exponential distribution has PDF:

.. math::
    p(x|\\lambda) = \\lambda e^{-\\lambda x}

for :math:`x \\geq 0`, where :math:`\\lambda > 0` is the rate parameter.

Exponential family form:

- :math:`h(x) = 1` for :math:`x \\geq 0` (base measure)
- :math:`t(x) = x` (sufficient statistic)
- :math:`\\theta = -\\lambda` (natural parameter)
- :math:`\\psi(\\theta) = -\\log(-\\theta)` (log partition function)

Parametrizations:

- Classical: :math:`\\lambda` (rate), :math:`\\lambda > 0`
- Natural: :math:`\\theta = -\\lambda`, :math:`\\theta < 0`
- Expectation: :math:`\\eta = E[X] = 1/\\lambda`, :math:`\\eta > 0`

Note: scipy uses scale = 1/rate parametrization.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional

from normix.base import ExponentialFamily
from normix.params import ExponentialParams


class Exponential(ExponentialFamily):
    """
    Exponential distribution in exponential family form.
    
    The Exponential distribution has PDF:
    
    .. math::
        p(x|\\lambda) = \\lambda e^{-\\lambda x}
    
    for :math:`x \\geq 0`, where :math:`\\lambda` is the rate parameter.
    
    Parameters
    ----------
    rate : float, optional
        Rate parameter :math:`\\lambda > 0`. Use ``from_classical_params(rate=...)``.
    
    Attributes
    ----------
    _natural_params : tuple or None
        Internal storage for natural parameters :math:`\\theta = -\\lambda`.
    
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
    
    See Also
    --------
    Gamma : Generalization of Exponential (Exponential is Gamma with :math:`\\alpha = 1`)
    
    Notes
    -----
    The Exponential distribution is a special case of the Gamma distribution
    with shape parameter :math:`\\alpha = 1`:
    
    .. math::
        \\text{Exponential}(\\lambda) = \\text{Gamma}(1, \\lambda)
    
    Exponential family form:
    
    - Sufficient statistic: :math:`t(x) = x`
    - Natural parameter: :math:`\\theta = -\\lambda`
    - Log partition: :math:`\\psi(\\theta) = -\\log(-\\theta)`
    
    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.
    """
    
    def __init__(self):
        super().__init__()
        self._lambda = None
    
    # ================================================================
    # New interface: internal state management
    # ================================================================
    
    def _set_from_classical(self, *, rate) -> None:
        """Set internal state from classical parameters."""
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        self._lambda = float(rate)
        self._fitted = True
        self._invalidate_cache()
    
    def _set_from_natural(self, theta) -> None:
        """Set internal state from natural parameters."""
        theta = np.asarray(theta)
        self._validate_natural_params(theta)
        self._lambda = float(-theta[0])
        self._fitted = True
        self._invalidate_cache()
    
    def _compute_natural_params(self):
        """Compute natural parameters from internal state: θ = -λ."""
        return np.array([-self._lambda])
    
    def _compute_classical_params(self):
        """Return frozen dataclass of classical parameters."""
        return ExponentialParams(rate=self._lambda)
    
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
        Log partition function: psi(theta) = -log(-theta) for theta < 0.
        
        .. math::
            \\psi(\\theta) = -\\log(-\\theta)
        
        Its gradient gives the expectation parameter: :math:`\\nabla\\psi(\\theta) = E[X] = 1/\\lambda`.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector :math:`[\\theta]` where :math:`\\theta < 0`.
        
        Returns
        -------
        psi : float
            Log partition function value.
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
    
    def _compute_expectation_params(self) -> NDArray:
        """Compute expectation parameters directly: η = [1/λ]."""
        return np.array([1.0 / self._lambda])

    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Analytical gradient: η = ∇ψ(θ) = 1/(-θ) = 1/λ.
        
        This is E[X] = 1/λ.
        """
        return np.array([1.0 / (-theta[0])])
    
    def _expectation_to_natural(self, eta: NDArray, theta0=None) -> NDArray:
        """
        Analytical inverse: θ = -1/η.
        
        From η = E[X] = 1/λ, we get λ = 1/η, so θ = -λ = -1/η.
        
        Parameters
        ----------
        eta : ndarray
            Expectation parameters [E[X]].
        
        Returns
        -------
        theta : ndarray
            Natural parameters [-λ].
        """
        return np.array([-1.0 / eta[0]])
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Analytical Fisher information: I(theta) = 1/theta^2.
        
        The Fisher information matrix (Hessian of log partition) is:
        
        .. math::
            I(\\theta) = \\frac{1}{\\theta^2} = \\frac{1}{\\lambda^2}
        
        Parameters
        ----------
        theta : ndarray, optional
            Natural parameter vector. If None, uses current parameters.
        
        Returns
        -------
        fisher : ndarray, shape (1, 1)
            Fisher information matrix.
        """
        if theta is None:
            theta = self.natural_params
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
        self._check_fitted()
        rate = self._lambda
        
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
    
    def mean(self) -> float:
        """
        Mean of Exponential distribution: E[X] = 1/lambda.
        
        .. math::
            E[X] = \\frac{1}{\\lambda}
        
        Returns
        -------
        mean : float
            Mean of the distribution.
        """
        self._check_fitted()
        return 1.0 / self._lambda
    
    def var(self) -> float:
        """
        Variance of Exponential distribution: Var[X] = 1/lambda^2.
        
        .. math::
            \\text{Var}[X] = \\frac{1}{\\lambda^2}
        
        Returns
        -------
        var : float
            Variance of the distribution.
        """
        self._check_fitted()
        return 1.0 / self._lambda**2
    
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
        self._check_fitted()
        
        x = np.asarray(x)
        rate = self._lambda
        
        result = np.zeros_like(x, dtype=float)
        mask = x >= 0
        result[mask] = 1.0 - np.exp(-rate * x[mask])
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result

