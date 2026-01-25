"""
Base class for exponential family distributions.

Exponential families have the canonical form:

.. math::
    p(x|\\theta) = h(x) \\exp(\\theta^T t(x) - \\psi(\\theta))

where:

- :math:`\\theta`: natural parameters (d-dimensional vector)
- :math:`t(x)`: sufficient statistics (d-dimensional vector)
- :math:`\\psi(\\theta)`: log partition function (cumulant generating function)
- :math:`h(x)`: base measure

The log partition function satisfies:

.. math::
    \\nabla\\psi(\\theta) = E[t(X)] = \\eta

where :math:`\\eta` are the expectation parameters.

Supports three parametrizations:

- **Classical**: Domain-specific parameters (e.g., :math:`\\mu`, :math:`\\sigma` for Normal)
- **Natural**: :math:`\\theta` in the exponential family form (numpy array)
- **Expectation**: :math:`\\eta = \\nabla\\psi(\\theta) = E[t(X)]` (numpy array)

The Fisher information matrix equals the Hessian of the log partition:

.. math::
    I(\\theta) = \\nabla^2\\psi(\\theta) = \\text{Cov}[t(X)]
"""

from abc import abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.differentiate import jacobian, hessian
from scipy.optimize import minimize, Bounds

from .distribution import Distribution


class ExponentialFamily(Distribution):
    """
    Abstract base class for exponential family distributions.
    
    Exponential family distributions have the probability density:
    
    .. math::
        p(x|\\theta) = h(x) \\exp(\\theta^T t(x) - \\psi(\\theta))
    
    where:
    
    - :math:`\\theta` are the natural parameters (d-dimensional vector)
    - :math:`t(x)` are the sufficient statistics (d-dimensional vector)
    - :math:`\\psi(\\theta)` is the log partition function (scalar)
    - :math:`h(x)` is the base measure (scalar)
    
    Three parametrizations:
    
    - **Classical**: Domain-specific (e.g., :math:`\\mu`, :math:`\\sigma` for Normal; 
      :math:`\\lambda` for Exponential)
    - **Natural**: :math:`\\theta` in exponential family form (numpy array)
    - **Expectation**: :math:`\\eta = \\nabla\\psi(\\theta) = E[t(X)]` (numpy array)
    
    Natural parameters are stored internally as tuples (hashable for caching).
    All methods use numpy arrays, converting to tuples only for caching.
    
    Attributes
    ----------
    _natural_params : tuple or None
        Internal storage for natural parameters (tuple for hashability).
    
    Examples
    --------
    Use factory methods to create instances:
    
    >>> # From classical parameters
    >>> dist = Exponential.from_classical_params(rate=2.0)
    
    >>> # From natural parameters (array)
    >>> dist = Exponential.from_natural_params(np.array([-2.0]))
    
    >>> # From expectation parameters (array)
    >>> dist = Exponential.from_expectation_params(np.array([0.5]))
    
    >>> # Fit from data (returns self for method chaining)
    >>> dist = Exponential().fit(data)
    
    Notes
    -----
    The log partition function :math:`\\psi(\\theta)` is convex, and its gradient
    and Hessian give important quantities:
    
    - Gradient: :math:`\\nabla\\psi(\\theta) = E[t(X)] = \\eta` (expectation parameters)
    - Hessian: :math:`\\nabla^2\\psi(\\theta) = \\text{Cov}[t(X)] = I(\\theta)` (Fisher information)
    
    References
    ----------
    .. [1] Barndorff-Nielsen, O. E. (1978). Information and exponential families
           in statistical theory.
    """
    
    def __init__(self):
        """
        Initialize an unfitted exponential family distribution.
        
        Use factory methods (from_classical_params, from_natural_params, etc.)
        or fit() to set parameters.
        """
        super().__init__()
        self._natural_params = None  # Stored as tuple (hashable for caching)
    
    # ============================================================
    # Factory methods for initialization
    # ============================================================
    
    @classmethod
    def from_classical_params(cls, **kwargs) -> 'ExponentialFamily':
        """
        Create distribution from classical parameters.
        
        Parameters
        ----------
        **kwargs
            Distribution-specific classical parameters.
            E.g., rate=2.0 for Exponential, mu=0.0, sigma=1.0 for Normal.
        
        Returns
        -------
        dist : ExponentialFamily
            Distribution instance with parameters set.
        
        Examples
        --------
        >>> dist = Exponential.from_classical_params(rate=2.0)
        >>> dist = Normal.from_classical_params(mu=0.0, sigma=1.0)
        """
        instance = cls()
        instance.set_classical_params(**kwargs)
        return instance
    
    @classmethod
    def from_natural_params(cls, theta: NDArray) -> 'ExponentialFamily':
        """
        Create distribution from natural parameters.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.
        
        Returns
        -------
        dist : ExponentialFamily
            Distribution instance with parameters set.
        
        Examples
        --------
        >>> dist = Exponential.from_natural_params(np.array([-2.0]))
        >>> dist = Normal.from_natural_params(np.array([0.0, -0.5]))
        """
        instance = cls()
        instance.set_natural_params(theta)
        return instance
    
    @classmethod
    def from_expectation_params(cls, eta: NDArray) -> 'ExponentialFamily':
        """
        Create distribution from expectation parameters.
        
        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector (E[t(X)]).
        
        Returns
        -------
        dist : ExponentialFamily
            Distribution instance with parameters set.
        
        Examples
        --------
        >>> dist = Exponential.from_expectation_params(np.array([0.5]))
        >>> dist = Normal.from_expectation_params(np.array([0.0, 1.0]))
        """
        instance = cls()
        instance.set_expectation_params(eta)
        return instance
    
    # ============================================================
    # Parameter setters
    # ============================================================
    
    def set_classical_params(self, **kwargs) -> 'ExponentialFamily':
        """
        Set parameters from classical parametrization.
        
        Parameters
        ----------
        **kwargs
            Distribution-specific classical parameters.
        
        Returns
        -------
        self : ExponentialFamily
            Returns self for method chaining.
        """
        if not kwargs:
            return self
            
        # Convert to natural parameters (returns numpy array)
        theta = self._classical_to_natural(**kwargs)
        theta = np.asarray(theta)
        
        # Validate and store
        self._validate_natural_params(theta)
        self._natural_params = tuple(theta)
        
        # Clear caches
        self._clear_param_cache()
        
        return self
    
    def set_natural_params(self, theta: NDArray) -> 'ExponentialFamily':
        """
        Set parameters from natural parametrization.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.
        
        Returns
        -------
        self : ExponentialFamily
            Returns self for method chaining.
        """
        theta = np.asarray(theta)
        
        # Validate and store
        self._validate_natural_params(theta)
        self._natural_params = tuple(theta)
        
        # Clear caches
        self._clear_param_cache()
        
        return self
    
    def set_expectation_params(self, eta: NDArray) -> 'ExponentialFamily':
        """
        Set parameters from expectation parametrization.
        
        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector (E[t(X)]).
        
        Returns
        -------
        self : ExponentialFamily
            Returns self for method chaining.
        """
        eta = np.asarray(eta)
        
        # Convert to natural parameters (uses optimization)
        theta = self._expectation_to_natural(eta)
        
        # Validate and store
        self._validate_natural_params(theta)
        self._natural_params = tuple(theta)
        
        # Clear caches
        self._clear_param_cache()
        
        return self
    
    def _clear_param_cache(self):
        """Clear parameter conversion caches."""
        if hasattr(self, '_get_expectation_params_cached'):
            self._get_expectation_params_cached.cache_clear()
        if hasattr(self, '_get_classical_params_cached'):
            self._get_classical_params_cached.cache_clear()
    
    # ============================================================
    # Parameter getters with caching
    # ============================================================
    
    def get_natural_params(self) -> NDArray:
        """
        Get natural parameters θ as numpy array.
        
        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        return np.array(self._natural_params)
    
    @lru_cache(maxsize=1)
    def _get_expectation_params_cached(self, theta_tuple: Tuple[float, ...]) -> Tuple[float, ...]:
        """Cached conversion to expectation parameters (internal)."""
        theta = np.array(theta_tuple)
        eta = self._natural_to_expectation(theta)
        return tuple(eta)
    
    def get_expectation_params(self) -> NDArray:
        """
        Get expectation parameters η = ∇ψ(θ) = E[t(X)].
        
        Uses caching for efficiency.
        
        Returns
        -------
        eta : ndarray
            Expectation parameter vector.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        # Convert to tuple for caching, then back to array
        eta_tuple = self._get_expectation_params_cached(self._natural_params)
        return np.array(eta_tuple)
    
    @lru_cache(maxsize=1)
    def _get_classical_params_cached(self, theta_tuple: Tuple[float, ...]) -> Union[Dict[str, Any], Tuple]:
        """Cached conversion to classical parameters (internal)."""
        theta = np.array(theta_tuple)
        return self._natural_to_classical(theta)
    
    def get_classical_params(self) -> Union[Dict[str, Any], Tuple]:
        """
        Get classical parameters (distribution-specific).
        
        Uses caching for efficiency.
        
        Returns
        -------
        params : dict or tuple
            Classical parameters in distribution-specific format.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        return self._get_classical_params_cached(self._natural_params)
    
    # ============================================================
    # Abstract methods for parameter support/validation
    # ============================================================
    
    @abstractmethod
    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Get support/bounds for each natural parameter component.
        
        Returns
        -------
        bounds : list of tuples
            List of (lower, upper) bounds for each component of θ.
            Use -np.inf or np.inf for unbounded.
            Length must match the dimension of natural parameters.
        
        Examples
        --------
        Exponential: θ < 0
        >>> return [(-np.inf, 0.0)]
        
        Gamma: θ₁ > -1, θ₂ < 0
        >>> return [(-1.0, np.inf), (-np.inf, 0.0)]
        
        Normal: θ₁ unbounded, θ₂ < 0
        >>> return [(-np.inf, np.inf), (-np.inf, 0.0)]
        """
        pass
    
    def _validate_natural_params(self, theta: NDArray) -> None:
        """
        Validate natural parameters against support (vectorized).
        
        Raises ValueError if parameters are outside support.
        """
        bounds = np.array(self._get_natural_param_support())
        
        if len(theta) != len(bounds):
            raise ValueError(
                f"Parameter dimension mismatch: expected {len(bounds)}, got {len(theta)}"
            )
        
        # Vectorized validation
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        # Check all bounds at once
        violations = (theta <= lower_bounds) | (theta >= upper_bounds)
        
        if np.any(violations):
            # Find first violation for error message
            idx = np.where(violations)[0][0]
            raise ValueError(
                f"Natural parameter θ[{idx}] = {theta[idx]:.6f} is outside support "
                f"({lower_bounds[idx]}, {upper_bounds[idx]})"
            )
    
    def _project_to_support(self, theta: NDArray, margin: float = 1e-10) -> NDArray:
        """
        Project parameters to valid support with margin from boundaries (vectorized).
        
        Useful for numerical differentiation near boundaries.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.
        margin : float, optional
            Safety margin from boundaries.
        
        Returns
        -------
        theta_proj : ndarray
            Projected parameter vector.
        """
        bounds = np.array(self._get_natural_param_support())
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        theta_proj = theta.copy()
        
        # Vectorized projection
        # Project to lower bound + margin where finite
        finite_lower = ~np.isinf(lower_bounds)
        theta_proj = np.where(
            finite_lower & (theta_proj < lower_bounds + margin),
            lower_bounds + margin,
            theta_proj
        )
        
        # Project to upper bound - margin where finite
        finite_upper = ~np.isinf(upper_bounds)
        theta_proj = np.where(
            finite_upper & (theta_proj > upper_bounds - margin),
            upper_bounds - margin,
            theta_proj
        )
        
        return theta_proj
    
    # ============================================================
    # Abstract methods for exponential family structure
    # ============================================================
    
    @abstractmethod
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Compute sufficient statistics t(x).
        
        Parameters
        ----------
        x : array_like
            Input data.
        
        Returns
        -------
        t : ndarray
            Sufficient statistics.
            - Shape (d,) for single sample (d = number of parameters)
            - Shape (n, d) for n samples
        
        Examples
        --------
        Exponential: t(x) = x
        >>> x = np.asarray(x)
        >>> return x.reshape(-1, 1) if x.ndim > 0 else np.array([x])
        
        Normal: t(x) = [x, x²]
        >>> x = np.asarray(x)
        >>> if x.ndim == 0:
        >>>     return np.array([x, x**2])
        >>> return np.column_stack([x, x**2])
        """
        pass
    
    @abstractmethod
    def _log_partition(self, theta: NDArray) -> float:
        """
        Compute log partition function ψ(θ).
        
        The log partition function's gradient gives expectation parameters:
            ∇ψ(θ) = E[t(X)] = η
        
        And its Hessian gives Fisher information:
            ∇²ψ(θ) = Cov[t(X)] = I(θ)
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.
        
        Returns
        -------
        psi : float
            Log partition function value.
        
        Examples
        --------
        Exponential: ψ(θ) = -log(-θ) for θ < 0
        >>> return -np.log(-theta[0])
        
        Normal: ψ(θ₁, θ₂) = -θ₁²/(4θ₂) - ½log(-2θ₂) for θ₂ < 0
        >>> return -theta[0]**2 / (4*theta[1]) - 0.5*np.log(-2*theta[1])
        """
        pass
    
    @abstractmethod
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Compute log base measure log h(x).
        
        Parameters
        ----------
        x : array_like
            Input data.
        
        Returns
        -------
        log_h : ndarray or scalar
            Log base measure at each point.
        
        Examples
        --------
        Exponential: h(x) = 1 for x ≥ 0, 0 otherwise
        >>> x = np.asarray(x)
        >>> result = np.zeros_like(x, dtype=float)
        >>> result[x < 0] = -np.inf
        >>> return result
        
        Normal: h(x) = 1/√(2π)
        >>> return -0.5 * np.log(2 * np.pi)
        """
        pass
    
    # ============================================================
    # Abstract methods for parameter conversions
    # ============================================================
    
    @abstractmethod
    def _classical_to_natural(self, **kwargs) -> NDArray:
        """
        Convert classical parameters → natural parameters (vector).
        
        Parameters
        ----------
        **kwargs
            Distribution-specific classical parameters.
        
        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        
        Examples
        --------
        Exponential: λ → θ = -λ
        >>> rate = kwargs['rate']
        >>> return np.array([-rate])
        
        Normal: (μ, σ) → (θ₁, θ₂) = (μ/σ², -1/(2σ²))
        >>> mu = kwargs['mu']
        >>> sigma = kwargs['sigma']
        >>> return np.array([mu / sigma**2, -0.5 / sigma**2])
        """
        pass
    
    @abstractmethod
    def _natural_to_classical(self, theta: NDArray) -> Union[Dict[str, Any], Tuple]:
        """
        Convert natural parameters (vector) → classical parameters.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.
        
        Returns
        -------
        params : dict or tuple
            Classical parameters (distribution-specific format).
        
        Examples
        --------
        Exponential: θ → λ = -θ
        >>> return {'rate': -theta[0]}
        
        Normal: (θ₁, θ₂) → (μ, σ)
        >>> sigma = np.sqrt(-0.5 / theta[1])
        >>> mu = theta[0] * sigma**2
        >>> return {'mu': mu, 'sigma': sigma}
        """
        pass
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Convert natural parameters to expectation parameters: eta = grad psi(theta).
        
        Computes the gradient of the log partition function:
        
        .. math::
            \\eta = \\nabla\\psi(\\theta) = E[t(X)]
        
        Default implementation uses ``scipy.differentiate.jacobian`` for numerical gradient.
        Override for analytical gradient when available (recommended for performance).
        
        Parameters
        ----------
        theta : ndarray
            Natural parameter vector :math:`\\theta`.
        
        Returns
        -------
        eta : ndarray
            Expectation parameter vector :math:`\\eta = E[t(X)]`.
        """
        theta = self._project_to_support(theta)
        
        # Define ψ as function of array for scipy.differentiate
        def psi_func(theta_vec):
            return self._log_partition(theta_vec)
        
        # Compute gradient using scipy.differentiate.jacobian
        # For scalar function → vector, jacobian returns gradient
        result = jacobian(psi_func, theta)
        
        # Extract the gradient from result object
        if hasattr(result, 'ddf'):
            eta = result.ddf
        else:
            eta = result
        
        return np.asarray(eta)
    
    def _expectation_to_natural(self, eta: NDArray) -> NDArray:
        """
        Convert expectation parameters to natural parameters.
        
        Solves the convex optimization problem:
        
        .. math::
            \\theta^* = \\arg\\max_\\theta [\\theta \\cdot \\eta - \\psi(\\theta)]
        
        Equivalently solves the equation :math:`\\nabla\\psi(\\theta^*) = \\eta`.
        
        Uses multi-start optimization with L-BFGS-B. Starting points are
        provided by ``_get_initial_natural_params`` (override in subclasses).
        
        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector :math:`\\eta = E[t(X)]`.
        
        Returns
        -------
        theta : ndarray
            Natural parameter vector :math:`\\theta`.
        """
        eta = np.asarray(eta)
        
        # Get bounds from support
        bounds_list = self._get_natural_param_support()
        lb = np.array([b[0] if not np.isinf(b[0]) else -1e10 for b in bounds_list])
        ub = np.array([b[1] if not np.isinf(b[1]) else 1e10 for b in bounds_list])
        bounds = Bounds(lb=lb, ub=ub)
        
        # Objective: minimize A(θ) - θ·η (negative of dual function)
        def objective(theta_arr):
            theta_arr = self._project_to_support(theta_arr)
            psi = self._log_partition(theta_arr)
            return psi - np.dot(theta_arr, eta)
        
        # Gradient: ∇A(θ) - η
        grad_func = None
        if self._has_analytical_gradient():
            def grad_func(theta_arr):
                theta_arr = self._project_to_support(theta_arr)
                grad_psi = self._natural_to_expectation(theta_arr)
                return grad_psi - eta
        
        # Get starting points from subclass (can be single point or list)
        starting_points = self._get_initial_natural_params(eta)
        if isinstance(starting_points, np.ndarray) and starting_points.ndim == 1:
            starting_points = [starting_points]
        
        # Track best solution
        best_theta = None
        best_obj = np.inf
        best_grad_norm = np.inf
        
        # Try each starting point with L-BFGS-B
        for x0 in starting_points:
            x0 = np.asarray(x0)
            x0 = self._project_to_support(x0)
            
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                jac=grad_func,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Compute gradient norm at solution
            if grad_func is not None:
                grad_norm = np.max(np.abs(grad_func(result.x)))
            else:
                grad_norm = np.inf
            
            # Accept if better objective or much better gradient
            if result.fun < best_obj or grad_norm < best_grad_norm * 0.1:
                best_theta = result.x
                best_obj = result.fun
                best_grad_norm = grad_norm
            
            # Early exit if gradient is small enough
            if grad_norm < 1e-8:
                return result.x
        
        # Warn if solution quality is poor
        if best_grad_norm > 1e-4:
            import warnings
            warnings.warn(
                f"Expectation to natural conversion may not have converged well "
                f"(max gradient norm: {best_grad_norm:.2e})"
            )
        
        return best_theta
    
    def _get_initial_natural_params(self, eta: NDArray) -> Union[NDArray, List[NDArray]]:
        """
        Get initial guess(es) for natural parameters in optimization.
        
        Override in subclasses to provide distribution-specific initialization.
        Can return a single array or a list of arrays for multi-start optimization.
        
        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector.
        
        Returns
        -------
        theta0 : ndarray or list of ndarray
            Initial guess(es) for natural parameters.
        """
        return eta.copy()
    
    def _has_analytical_gradient(self) -> bool:
        """Check if _natural_to_expectation is overridden."""
        # Check if method is defined in subclass
        for cls in type(self).__mro__:
            if '_natural_to_expectation' in cls.__dict__:
                return cls is not ExponentialFamily
        return False
    
    def _has_analytical_hessian(self) -> bool:
        """Check if fisher_information is overridden."""
        # Check if method is defined in subclass
        for cls in type(self).__mro__:
            if 'fisher_information' in cls.__dict__:
                return cls is not ExponentialFamily
        return False
    
    # ============================================================
    # Fisher Information
    # ============================================================
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Compute Fisher information matrix: I(theta) = Hessian of psi(theta).
        
        The Fisher information equals:
        
        .. math::
            I(\\theta) = \\nabla^2\\psi(\\theta) = \\text{Cov}[t(X)]
        
        It is the Hessian of the log partition function (always positive semi-definite
        since :math:`\\psi` is convex), and equals the covariance matrix of the
        sufficient statistics.
        
        Default implementation uses ``scipy.differentiate.hessian``.
        Override for analytical Hessian when available (recommended for performance).
        
        Parameters
        ----------
        theta : ndarray, optional
            Natural parameter vector :math:`\\theta`. If None, uses current parameters.
        
        Returns
        -------
        fisher : ndarray, shape (d, d)
            Fisher information matrix (positive semi-definite).
        
        Examples
        --------
        >>> dist = Exponential.from_classical_params(rate=2.0)
        >>> fisher = dist.fisher_information()
        >>> print(fisher)  # [[0.25]] for rate=2
        """
        if theta is None:
            theta = self.get_natural_params()
        else:
            theta = np.asarray(theta)
        
        theta = self._project_to_support(theta)
        
        # Define ψ for scipy.differentiate
        def psi_func(theta_vec):
            return self._log_partition(theta_vec)
        
        # Compute Hessian using scipy.differentiate.hessian
        result = hessian(psi_func, theta)
        
        # Extract the Hessian from result object
        if hasattr(result, 'ddf'):
            fisher = result.ddf
        else:
            fisher = result
        
        # Ensure it's 2D and symmetric
        fisher = np.asarray(fisher)
        if fisher.ndim == 0:
            fisher = np.array([[fisher]])
        elif fisher.ndim == 1:
            fisher = np.diag(fisher)
        
        # Ensure numerical symmetry
        fisher = (fisher + fisher.T) / 2
        
        return fisher
    
    # ============================================================
    # PDF using exponential family form
    # ============================================================
    
    def logpdf(self, x: ArrayLike) -> Union[float, NDArray[np.floating]]:
        """
        Log probability density using exponential family form.
        
        Computes:
        
        .. math::
            \\log p(x|\\theta) = \\log h(x) + \\theta^T t(x) - \\psi(\\theta)
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate log PDF.
        
        Returns
        -------
        logpdf : float or ndarray
            Log probability density at each point.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        x = np.asarray(x)
        theta = self.get_natural_params()
        
        # Compute components
        log_h = self._log_base_measure(x)
        t_x = self._sufficient_statistics(x)
        psi = self._log_partition(theta)
        
        # log p(x) = log h(x) + θ^T t(x) - ψ(θ)
        if t_x.ndim == 1:
            # Single sample: t(x) is 1D
            theta_t = np.dot(theta, t_x)
        else:
            # Multiple samples: t(x) is 2D (n_samples, d)
            theta_t = np.dot(t_x, theta)
        
        result = log_h + theta_t - psi
        
        # Return scalar if input was scalar
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        
        return result
    
    def pdf(self, x: ArrayLike) -> Union[float, NDArray[np.floating]]:
        """
        Probability density: p(x|θ) = exp(logpdf(x)).
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate PDF.
        
        Returns
        -------
        pdf : float or ndarray
            Probability density at each point.
        """
        return np.exp(self.logpdf(x))
    
    # ============================================================
    # Fitting (MLE)
    # ============================================================
    
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs) -> 'ExponentialFamily':
        """
        Fit distribution parameters using Maximum Likelihood Estimation.
        
        For exponential families, the MLE has a closed form in expectation parameters:
        
        .. math::
            \\hat{\\eta} = \\frac{1}{n} \\sum_{i=1}^n t(x_i)
        
        That is, the MLE of expectation parameters equals the sample mean
        of sufficient statistics. Then converts to natural parameters using
        ``_expectation_to_natural``.
        
        Parameters
        ----------
        X : array_like
            Training data.
        y : array_like, optional
            Ignored (for sklearn API compatibility).
        **kwargs
            Additional options (currently unused).
        
        Returns
        -------
        self : ExponentialFamily
            Returns self for method chaining (sklearn convention).
        
        Examples
        --------
        >>> data = np.random.exponential(scale=0.5, size=1000)
        >>> dist = Exponential().fit(data)
        >>> print(dist.get_classical_params())
        {'rate': 2.01...}
        """
        X = np.asarray(X)
        
        # Compute sufficient statistics for all samples
        t_x = self._sufficient_statistics(X)
        
        # Compute sample mean (MLE in expectation parameters)
        if t_x.ndim == 1:
            # Single sufficient statistic
            eta_hat = np.array([np.mean(t_x)])
        else:
            # Multiple sufficient statistics (n_samples, d)
            eta_hat = np.mean(t_x, axis=0)
        
        # Convert to natural parameters (uses optimization)
        self.set_expectation_params(eta_hat)
        
        return self
    
    def score(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> float:
        """
        Compute mean log-likelihood (sklearn-style scoring).
        
        Higher scores are better (sklearn convention).
        
        Parameters
        ----------
        X : array_like
            Data samples.
        y : array_like, optional
            Ignored (for sklearn API compatibility).
        
        Returns
        -------
        score : float
            Mean log-likelihood.
        """
        X = np.asarray(X)
        logpdf_vals = self.logpdf(X)
        return float(np.mean(logpdf_vals))
    
    # ============================================================
    # String representation
    # ============================================================
    
    def __repr__(self) -> str:
        """String representation of the distribution."""
        if self._natural_params is None:
            return f"{self.__class__.__name__}(not fitted)"
        
        # Show classical parameters if available
        try:
            classical = self.get_classical_params()
            if isinstance(classical, dict):
                param_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float, np.number)) 
                    else f"{k}=..." 
                    for k, v in classical.items()
                )
            else:
                param_str = str(classical)
            return f"{self.__class__.__name__}({param_str})"
        except:
            # Fallback to natural parameters
            theta = self.get_natural_params()
            theta_str = ", ".join(f"{x:.4f}" for x in theta)
            return f"{self.__class__.__name__}(θ=[{theta_str}])"
