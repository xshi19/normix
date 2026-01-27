"""
Multivariate Normal distribution as an exponential family.

The multivariate Normal distribution has PDF:

.. math::
    p(x|\\mu,\\Sigma) = (2\\pi)^{-d/2} |\\Sigma|^{-1/2} 
    \\exp\\left(-\\frac{1}{2} (x-\\mu)^T \\Sigma^{-1} (x-\\mu)\\right)

for :math:`x \\in \\mathbb{R}^d`, where :math:`\\mu` is the mean vector
and :math:`\\Sigma` is the covariance matrix.

Exponential family form:

- :math:`h(x) = 1` (base measure, with :math:`(2\\pi)^{-d/2}` absorbed into :math:`\\psi`)
- :math:`t(x) = [x, \\text{vec}(xx^T)]` (sufficient statistics)
- :math:`\\theta = [\\Lambda\\mu, -\\frac{1}{2}\\text{vec}(\\Lambda)]` where :math:`\\Lambda = \\Sigma^{-1}`
- :math:`\\psi(\\theta) = \\frac{1}{2}\\mu^T\\Lambda\\mu - \\frac{1}{2}\\log|\\Lambda| + \\frac{d}{2}\\log(2\\pi)`

Parametrizations:

- Classical: :math:`\\mu` (mean, d-vector), :math:`\\Sigma` (covariance, d×d positive definite)
- Natural: :math:`\\theta = [\\eta, \\text{vec}(\\Lambda_{half})]` where 
  :math:`\\eta = \\Lambda\\mu`, :math:`\\Lambda_{half} = -\\frac{1}{2}\\Lambda`
- Expectation: :math:`\\eta = [E[X], E[XX^T]] = [\\mu, \\Sigma + \\mu\\mu^T]`

Supports both univariate (d=1) and multivariate (d>1) cases.
For d=1: :math:`\\mu` is scalar, :math:`\\Sigma` is scalar (variance :math:`\\sigma^2`).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats

from normix.base import ExponentialFamily


class MultivariateNormal(ExponentialFamily):
    """
    Multivariate Normal distribution in exponential family form.
    
    The Multivariate Normal distribution has PDF:
    
    .. math::
        p(x|\\mu,\\Sigma) = (2\\pi)^{-d/2} |\\Sigma|^{-1/2} 
        \\exp\\left(-\\frac{1}{2} (x-\\mu)^T \\Sigma^{-1} (x-\\mu)\\right)
    
    Supports both univariate (d=1) and multivariate (d>1) cases.
    
    Parameters
    ----------
    d : int, optional
        Dimension of the distribution. Inferred from parameters if not provided.
    
    Attributes
    ----------
    _natural_params : tuple or None
        Internal storage for natural parameters.
    _d : int or None
        Dimension of the distribution.
    
    Examples
    --------
    >>> # 1D case (univariate normal)
    >>> dist = MultivariateNormal.from_classical_params(mu=0.0, sigma=np.array([[1.0]]))
    >>> dist.mean()
    array([0.])
    
    >>> # 2D case
    >>> mu = np.array([1.0, 2.0])
    >>> sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> dist = MultivariateNormal.from_classical_params(mu=mu, sigma=sigma)
    >>> dist.mean()
    array([1., 2.])
    
    >>> # Fit from data
    >>> data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)
    >>> dist = MultivariateNormal(d=2).fit(data)
    
    Notes
    -----
    The Multivariate Normal distribution belongs to the exponential family with:
    
    - Sufficient statistics: :math:`t(x) = [x, \\text{vec}(xx^T)]`
    - Natural parameters: :math:`\\theta = [\\Lambda\\mu, -\\frac{1}{2}\\text{vec}(\\Lambda)]`
    - Log partition: :math:`\\psi(\\theta) = \\frac{1}{2}\\eta^T\\Sigma\\eta - \\frac{1}{2}\\log|\\Lambda| + \\frac{d}{2}\\log(2\\pi)`
    
    where :math:`\\Lambda = \\Sigma^{-1}` is the precision matrix.
    
    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.
    """
    
    def __init__(self, d: Optional[int] = None):
        """
        Initialize an unfitted multivariate normal distribution.
        
        Parameters
        ----------
        d : int, optional
            Dimension of the distribution.
        """
        super().__init__()
        self._d = d
    
    @property
    def d(self) -> int:
        """Dimension of the distribution."""
        if self._d is None:
            raise ValueError("Dimension not set. Use from_*_params() or fit().")
        return self._d
    
    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Natural parameter support.
        
        For MVN:
        - θ₁ (η = Λμ): unbounded (d components)
        - θ₂ (Λ_half = -1/2 Λ): must be negative definite
          (diagonal elements < 0, stored as d² components)
        
        We return unbounded for η, and (-∞, 0) for diagonal elements of Λ_half.
        The off-diagonal elements are technically unbounded, but we use a loose bound.
        """
        if self._d is None:
            raise ValueError("Dimension not set.")
        
        d = self._d
        bounds = []
        
        # η = Λμ: unbounded (d components)
        for _ in range(d):
            bounds.append((-np.inf, np.inf))
        
        # Λ_half = -1/2 Λ: stored as d² components (flattened)
        # Diagonal elements must be < 0 for Λ to be positive definite
        # Off-diagonal are more loosely bounded
        for i in range(d):
            for j in range(d):
                if i == j:
                    # Diagonal: must be negative (for positive definiteness)
                    bounds.append((-np.inf, 0.0))
                else:
                    # Off-diagonal: loosely bounded
                    bounds.append((-np.inf, np.inf))
        
        return bounds
    
    def _validate_natural_params(self, theta: NDArray) -> None:
        """
        Validate natural parameters.
        
        Checks that Λ = -2 * Λ_half is positive definite.
        """
        d = self._d
        if d is None:
            # Infer dimension from parameter length
            # theta has d + d² elements, so d² + d - len(theta) = 0
            # d = (-1 + sqrt(1 + 4*len(theta))) / 2
            n = len(theta)
            d_inferred = int((-1 + np.sqrt(1 + 4 * n)) / 2)
            if d_inferred * (d_inferred + 1) != n:
                raise ValueError(f"Invalid parameter length {n}")
            self._d = d_inferred
            d = d_inferred
        
        expected_len = d + d * d
        if len(theta) != expected_len:
            raise ValueError(
                f"Expected {expected_len} natural parameters for d={d}, got {len(theta)}"
            )
        
        # Extract Λ_half and check positive definiteness of Λ = -2 * Λ_half
        Lambda_half = theta[d:].reshape(d, d)
        Lambda = -2 * Lambda_half
        
        # Check symmetry (with tolerance)
        if not np.allclose(Lambda, Lambda.T, rtol=1e-10, atol=1e-10):
            raise ValueError("Precision matrix Λ is not symmetric")
        
        # Check positive definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(Lambda)
        if np.any(eigvals <= 0):
            raise ValueError(
                f"Precision matrix Λ is not positive definite. "
                f"Min eigenvalue: {np.min(eigvals):.6f}"
            )
    
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Sufficient statistics: t(x) = [x, vec(xx^T)].
        
        Parameters
        ----------
        x : array_like
            Input data. Shape (d,) for single sample, (n, d) for n samples.
        
        Returns
        -------
        t : ndarray
            Shape (d + d²,) for single sample, (n, d + d²) for n samples.
        """
        x = np.asarray(x)
        d = self.d
        
        # Handle 1D input (single sample of dimension d)
        if x.ndim == 1:
            if len(x) != d:
                raise ValueError(f"Expected {d}-dimensional input, got {len(x)}")
            xx_T = np.outer(x, x).flatten()
            return np.concatenate([x, xx_T])
        
        # Handle 2D input (n samples of dimension d)
        n = x.shape[0]
        if x.shape[1] != d:
            raise ValueError(f"Expected {d}-dimensional input, got {x.shape[1]}")
        
        # Compute x and xx^T for each sample
        t = np.zeros((n, d + d * d))
        t[:, :d] = x
        for i in range(n):
            t[i, d:] = np.outer(x[i], x[i]).flatten()
        
        return t
    
    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function: ψ(θ) = 1/2 μ^T Λ μ - 1/2 log|Λ| + d/2 log(2π).
        
        Given θ = [η, vec(Λ_half)] where η = Λμ and Λ_half = -1/2 Λ:
        - Λ = -2 * Λ_half
        - μ = Λ^{-1} η = Σ η
        """
        d = self.d
        
        # Extract parameters
        eta = theta[:d]
        Lambda_half = theta[d:].reshape(d, d)
        Lambda = -2 * Lambda_half  # Precision matrix
        
        # Compute Σ = Λ^{-1} and μ = Σ η
        Sigma = np.linalg.inv(Lambda)
        mu = Sigma @ eta
        
        # ψ(θ) = 1/2 μ^T Λ μ - 1/2 log|Λ| + d/2 log(2π)
        #      = 1/2 η^T Σ η - 1/2 log|Λ| + d/2 log(2π)
        psi = 0.5 * eta @ Sigma @ eta
        _, logdet_Lambda = np.linalg.slogdet(Lambda)
        psi -= 0.5 * logdet_Lambda
        psi += 0.5 * d * np.log(2 * np.pi)
        
        return float(psi)
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Log base measure: log h(x) = -d/2 log(2π).
        
        Note: We include this in the log partition function, so here we return 0.
        Actually, the standard form has h(x) = 1 and ψ(θ) includes the normalization.
        """
        x = np.asarray(x)
        d = self.d
        
        # Return 0 for all inputs (h(x) = 1)
        # The (2π)^{-d/2} is absorbed into ψ(θ)
        if x.ndim == 1:
            return 0.0
        else:
            return np.zeros(x.shape[0])
    
    def _classical_to_natural(self, **kwargs) -> NDArray:
        """
        Convert classical parameters to natural parameters.
        
        Parameters
        ----------
        mu : array_like
            Mean vector (d,).
        sigma : array_like
            Covariance matrix (d, d).
        
        Returns
        -------
        theta : ndarray
            Natural parameters [η, vec(Λ_half)] where η = Λμ, Λ_half = -1/2 Λ.
        """
        mu = np.asarray(kwargs['mu']).flatten()
        sigma = np.asarray(kwargs['sigma'])
        
        # Handle scalar input for 1D case
        if sigma.ndim == 0:
            sigma = np.array([[sigma]])
        elif sigma.ndim == 1:
            sigma = np.diag(sigma)
        
        d = len(mu)
        if sigma.shape != (d, d):
            raise ValueError(f"sigma shape {sigma.shape} doesn't match mu dimension {d}")
        
        # Set dimension
        self._d = d
        
        # Validate covariance matrix
        if not np.allclose(sigma, sigma.T):
            raise ValueError("Covariance matrix must be symmetric")
        
        eigvals = np.linalg.eigvalsh(sigma)
        if np.any(eigvals <= 0):
            raise ValueError("Covariance matrix must be positive definite")
        
        # Compute natural parameters
        Lambda = np.linalg.inv(sigma)  # Precision matrix
        eta = Lambda @ mu  # η = Λμ
        Lambda_half = -0.5 * Lambda  # Λ_half = -1/2 Λ
        
        # Flatten and concatenate
        theta = np.concatenate([eta, Lambda_half.flatten()])
        
        return theta
    
    def _natural_to_classical(self, theta: NDArray) -> Dict[str, Any]:
        """
        Convert natural parameters to classical parameters.
        
        Parameters
        ----------
        theta : ndarray
            Natural parameters [η, vec(Λ_half)].
        
        Returns
        -------
        params : dict
            {'mu': mean vector, 'sigma': covariance matrix}
        """
        d = self.d
        
        # Extract parameters
        eta = theta[:d]
        Lambda_half = theta[d:].reshape(d, d)
        Lambda = -2 * Lambda_half  # Precision matrix
        
        # Compute classical parameters
        Sigma = np.linalg.inv(Lambda)  # Covariance matrix
        mu = Sigma @ eta  # μ = Σ η = Λ^{-1} η
        
        return {'mu': mu, 'sigma': Sigma}
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Convert natural to expectation parameters.
        
        η = [E[X], E[XX^T]] = [μ, Σ + μμ^T]
        
        Returns
        -------
        eta : ndarray
            Expectation parameters [μ, vec(Σ + μμ^T)].
        """
        d = self.d
        
        # Get classical parameters
        classical = self._natural_to_classical(theta)
        mu = classical['mu']
        Sigma = classical['sigma']
        
        # Compute expectation parameters
        # η₁ = E[X] = μ
        # η₂ = E[XX^T] = Σ + μμ^T
        eta1 = mu
        eta2 = (Sigma + np.outer(mu, mu)).flatten()
        
        return np.concatenate([eta1, eta2])
    
    def _expectation_to_natural(self, eta: NDArray) -> NDArray:
        """
        Convert expectation to natural parameters.
        
        From η = [μ, vec(Σ + μμ^T)]:
        - μ = η₁
        - Σ = η₂.reshape(d,d) - μμ^T
        
        Then compute natural parameters.
        """
        # Infer dimension from eta length: len(eta) = d + d^2
        # Solve: d^2 + d - len(eta) = 0 => d = (-1 + sqrt(1 + 4*len)) / 2
        n = len(eta)
        d_inferred = int((-1 + np.sqrt(1 + 4 * n)) / 2)
        if d_inferred * (d_inferred + 1) != n:
            raise ValueError(f"Invalid expectation parameter length {n}")
        
        if self._d is None:
            self._d = d_inferred
        
        d = self._d
        
        # Extract expectation parameters
        mu = eta[:d]
        second_moment = eta[d:].reshape(d, d)
        
        # Σ = E[XX^T] - μμ^T
        Sigma = second_moment - np.outer(mu, mu)
        
        # Ensure symmetry
        Sigma = (Sigma + Sigma.T) / 2
        
        # Ensure positive definiteness (with small regularization if needed)
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.any(eigvals <= 0):
            # Add small regularization
            min_eig = np.min(eigvals)
            Sigma += (-min_eig + 1e-6) * np.eye(d)
        
        # Convert to natural parameters
        return self._classical_to_natural(mu=mu, sigma=Sigma)
    
    def _get_initial_natural_params(self, eta: NDArray) -> NDArray:
        """Get initial guess for natural parameters."""
        return self._expectation_to_natural(eta)
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Fisher information matrix.
        
        For MVN, the Fisher information has a block structure:
        - I_μμ = Λ (d × d)
        - I_μΣ = 0 (d × d²)
        - I_ΣΣ = 1/2 (Λ ⊗ Λ) (d² × d²)
        
        In natural parameters, this becomes the Hessian of ψ(θ).
        
        For simplicity, we use numerical differentiation from base class.
        """
        # Use base class numerical implementation
        return super().fisher_information(theta)
    
    # ============================================================
    # Override logpdf for better numerical stability
    # ============================================================
    
    def logpdf(self, x: ArrayLike) -> Union[float, NDArray[np.floating]]:
        """
        Log probability density.
        
        log p(x|μ,Σ) = -d/2 log(2π) - 1/2 log|Σ| - 1/2 (x-μ)^T Σ^{-1} (x-μ)
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        x = np.asarray(x)
        classical = self.get_classical_params()
        mu = classical['mu']
        Sigma = classical['sigma']
        d = self.d
        
        # Handle single sample
        if x.ndim == 1:
            if len(x) != d:
                raise ValueError(f"Expected {d}-dimensional input, got {len(x)}")
            
            diff = x - mu
            Lambda = np.linalg.inv(Sigma)
            _, logdet_Sigma = np.linalg.slogdet(Sigma)
            
            logp = -0.5 * d * np.log(2 * np.pi)
            logp -= 0.5 * logdet_Sigma
            logp -= 0.5 * diff @ Lambda @ diff
            
            return float(logp)
        
        # Handle multiple samples
        n = x.shape[0]
        if x.shape[1] != d:
            raise ValueError(f"Expected {d}-dimensional input, got {x.shape[1]}")
        
        Lambda = np.linalg.inv(Sigma)
        _, logdet_Sigma = np.linalg.slogdet(Sigma)
        
        logp = np.zeros(n)
        for i in range(n):
            diff = x[i] - mu
            logp[i] = -0.5 * d * np.log(2 * np.pi)
            logp[i] -= 0.5 * logdet_Sigma
            logp[i] -= 0.5 * diff @ Lambda @ diff
        
        return logp
    
    # ============================================================
    # Distribution methods
    # ============================================================
    
    def rvs(self, size=None, random_state=None) -> NDArray:
        """
        Generate random samples.
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of samples to generate.
        random_state : int or Generator, optional
            Random number generator.
        
        Returns
        -------
        samples : ndarray
            Shape (size, d) for size > 1, (d,) for size = None.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        classical = self.get_classical_params()
        mu = classical['mu']
        Sigma = classical['sigma']
        
        # Set up RNG
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        # Generate samples
        if size is None:
            return rng.multivariate_normal(mu, Sigma)
        else:
            return rng.multivariate_normal(mu, Sigma, size=size)
    
    def mean(self) -> NDArray:
        """Mean of the distribution: E[X] = μ."""
        classical = self.get_classical_params()
        return classical['mu']
    
    def var(self) -> NDArray:
        """Variance (diagonal of covariance matrix)."""
        classical = self.get_classical_params()
        return np.diag(classical['sigma'])
    
    def cov(self) -> NDArray:
        """Covariance matrix Σ."""
        classical = self.get_classical_params()
        return classical['sigma']
    
    def cdf(self, x: ArrayLike) -> Union[float, NDArray[np.floating]]:
        """
        Cumulative distribution function.
        
        For d=1, uses the univariate normal CDF.
        For d>1, uses scipy's multivariate_normal.cdf.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate CDF.
        
        Returns
        -------
        cdf : float or ndarray
            CDF values.
        """
        if self._natural_params is None:
            raise ValueError("Parameters not set. Use from_*_params() or fit().")
        
        x = np.asarray(x)
        classical = self.get_classical_params()
        mu = classical['mu']
        Sigma = classical['sigma']
        d = self.d
        
        # Use scipy's implementation
        if d == 1:
            # Univariate case
            sigma_scalar = np.sqrt(Sigma[0, 0])
            mu_scalar = mu[0]
            
            if x.ndim == 0:
                return float(stats.norm.cdf(x, loc=mu_scalar, scale=sigma_scalar))
            elif x.ndim == 1 and len(x) == 1:
                return float(stats.norm.cdf(x[0], loc=mu_scalar, scale=sigma_scalar))
            elif x.ndim == 1:
                # Multiple scalar values
                return stats.norm.cdf(x, loc=mu_scalar, scale=sigma_scalar)
            else:
                # Shape (n, 1)
                return stats.norm.cdf(x.flatten(), loc=mu_scalar, scale=sigma_scalar)
        else:
            # Multivariate case
            rv = stats.multivariate_normal(mean=mu, cov=Sigma)
            return rv.cdf(x)
    
    def entropy(self) -> float:
        """
        Differential entropy.
        
        H(X) = 1/2 log|2πe Σ| = d/2 (1 + log(2π)) + 1/2 log|Σ|
        """
        classical = self.get_classical_params()
        Sigma = classical['sigma']
        d = self.d
        
        _, logdet = np.linalg.slogdet(Sigma)
        return 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * logdet
    
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs) -> 'MultivariateNormal':
        """
        Fit distribution parameters using Maximum Likelihood Estimation.
        
        For MVN, the MLE is:
        - μ̂ = sample mean
        - Σ̂ = sample covariance
        
        Parameters
        ----------
        X : array_like
            Training data. Shape (n_samples, d) or (n_samples,) for 1D.
        y : array_like, optional
            Ignored.
        
        Returns
        -------
        self : MultivariateNormal
            Fitted distribution.
        """
        X = np.asarray(X)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n, d = X.shape
        
        # Set dimension
        if self._d is None:
            self._d = d
        elif self._d != d:
            raise ValueError(f"Expected {self._d}-dimensional data, got {d}")
        
        # MLE estimates
        mu_hat = np.mean(X, axis=0)
        
        # Sample covariance (with Bessel correction for unbiased estimate)
        # For MLE, we use the biased estimate (divide by n)
        Sigma_hat = np.cov(X, rowvar=False, bias=True)
        
        # Ensure it's 2D for d=1 case
        if d == 1:
            Sigma_hat = np.array([[Sigma_hat]])
        
        # Regularization for numerical stability
        min_eig = np.min(np.linalg.eigvalsh(Sigma_hat))
        if min_eig < 1e-10:
            Sigma_hat += (1e-10 - min_eig) * np.eye(d)
        
        # Set parameters
        self.set_classical_params(mu=mu_hat, sigma=Sigma_hat)
        
        return self
    
    # ============================================================
    # Scipy compatibility
    # ============================================================
    
    def to_scipy(self) -> stats.multivariate_normal:
        """
        Convert to scipy.stats.multivariate_normal.
        
        Returns
        -------
        rv : scipy.stats.multivariate_normal
            Scipy distribution object.
        """
        classical = self.get_classical_params()
        return stats.multivariate_normal(mean=classical['mu'], cov=classical['sigma'])
    
    @classmethod
    def from_scipy(cls, rv: stats.multivariate_normal) -> 'MultivariateNormal':
        """
        Create from scipy.stats.multivariate_normal.
        
        Parameters
        ----------
        rv : scipy.stats.multivariate_normal
            Scipy distribution object.
        
        Returns
        -------
        dist : MultivariateNormal
            normix distribution.
        """
        return cls.from_classical_params(mu=rv.mean, sigma=rv.cov)
    
    # ============================================================
    # String representation
    # ============================================================
    
    def __repr__(self) -> str:
        """String representation."""
        if self._natural_params is None:
            if self._d is not None:
                return f"MultivariateNormal(d={self._d}, not fitted)"
            return "MultivariateNormal(not fitted)"
        
        d = self.d
        classical = self.get_classical_params()
        mu = classical['mu']
        
        if d == 1:
            sigma = classical['sigma'][0, 0]
            return f"MultivariateNormal(μ={mu[0]:.4f}, σ²={sigma:.4f})"
        elif d <= 3:
            mu_str = ", ".join(f"{x:.4f}" for x in mu)
            return f"MultivariateNormal(μ=[{mu_str}], d={d})"
        else:
            return f"MultivariateNormal(d={d})"


# Alias for convenience
MVN = MultivariateNormal
