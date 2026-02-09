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

Internal storage
----------------
After migration, the distribution stores the Cholesky decomposition of the
covariance matrix rather than the full covariance or its inverse:

- ``_mu``: mean vector, shape ``(d,)``
- ``_L``: lower Cholesky factor of :math:`\\Sigma`, shape ``(d, d)``

Derived quantities ``log_det_Sigma`` and ``L_inv`` are cached properties,
computed on demand and invalidated when parameters change.
"""

from functools import cached_property
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.linalg import cholesky, solve_triangular

from normix.base import ExponentialFamily
from normix.params import MultivariateNormalParams


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
    _mu : ndarray or None
        Mean vector, shape ``(d,)``.
    _L : ndarray or None
        Lower Cholesky factor of :math:`\\Sigma`, shape ``(d, d)``.
        :math:`\\Sigma = L L^T`.
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

    The internal storage uses the Cholesky decomposition :math:`L` of the
    covariance :math:`\\Sigma = LL^T`, which:

    - Avoids repeated matrix inversions in ``logpdf`` and ``rvs``
    - Provides numerically stable log-determinant: :math:`\\log|\\Sigma| = 2\\sum_i \\log L_{ii}`
    - Enables efficient Mahalanobis distance via ``solve_triangular``

    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.
    """

    _cached_attrs: Tuple[str, ...] = ExponentialFamily._cached_attrs + (
        'log_det_Sigma', 'L_inv',
    )

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
        self._mu: Optional[NDArray] = None
        self._L: Optional[NDArray] = None  # Lower Cholesky of Sigma

    @property
    def d(self) -> int:
        """Dimension of the distribution."""
        if self._d is None:
            raise ValueError("Dimension not set. Use from_*_params() or fit().")
        return self._d

    # ============================================================
    # Cached derived quantities
    # ============================================================

    @cached_property
    def log_det_Sigma(self) -> float:
        r"""
        Log-determinant of the covariance matrix (cached).

        .. math::
            \log|\Sigma| = 2 \sum_{i=1}^d \log L_{ii}

        Returns
        -------
        log_det : float
        """
        self._check_fitted()
        return 2.0 * np.sum(np.log(np.diag(self._L)))

    @cached_property
    def L_inv(self) -> NDArray:
        r"""
        Inverse of the lower Cholesky factor (cached).

        :math:`L^{-1}` such that :math:`\Sigma^{-1} = L^{-T} L^{-1}`.

        Returns
        -------
        L_inv : ndarray, shape ``(d, d)``
            Lower triangular matrix.
        """
        self._check_fitted()
        return solve_triangular(self._L, np.eye(self._d), lower=True)

    # ============================================================
    # New interface: internal state management
    # ============================================================

    def _set_from_classical(self, *, mu, sigma) -> None:
        """
        Set internal state from classical parameters.

        Computes and stores the Cholesky decomposition of sigma.

        Parameters
        ----------
        mu : array_like
            Mean vector (d,).
        sigma : array_like
            Covariance matrix (d, d), must be symmetric positive definite.
        """
        mu = np.asarray(mu).flatten().astype(float)
        sigma = np.asarray(sigma, dtype=float)

        # Handle scalar input for 1D case
        if sigma.ndim == 0:
            sigma = np.array([[float(sigma)]])
        elif sigma.ndim == 1:
            sigma = np.diag(sigma)

        d = len(mu)
        if sigma.shape != (d, d):
            raise ValueError(f"sigma shape {sigma.shape} doesn't match mu dimension {d}")

        # Set dimension
        self._d = d

        # Validate covariance matrix
        if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric")

        # Compute Cholesky decomposition (also validates positive definiteness)
        try:
            L = cholesky(sigma, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix must be positive definite")

        # Store internal state
        self._mu = mu.copy()
        self._L = L

        # Legacy storage for backward compat
        Lambda = solve_triangular(
            L.T, solve_triangular(L, np.eye(d), lower=True), lower=False
        )
        eta = Lambda @ mu
        Lambda_half = -0.5 * Lambda
        self._natural_params = tuple(np.concatenate([eta, Lambda_half.flatten()]))

        self._fitted = True
        self._invalidate_cache()

    def _set_from_natural(self, theta: NDArray) -> None:
        """
        Set internal state from natural parameters.

        Extracts the precision matrix, computes the covariance via Cholesky solve,
        then stores :math:`\\mu` and the Cholesky factor :math:`L`.

        Parameters
        ----------
        theta : ndarray
            Natural parameters [η, vec(Λ_half)].
        """
        theta = np.asarray(theta, dtype=float)

        # Infer or validate dimension
        n = len(theta)
        if self._d is None:
            d_inferred = int((-1 + np.sqrt(1 + 4 * n)) / 2)
            if d_inferred * (d_inferred + 1) != n:
                raise ValueError(f"Invalid parameter length {n}")
            self._d = d_inferred

        d = self._d
        expected_len = d + d * d
        if len(theta) != expected_len:
            raise ValueError(
                f"Expected {expected_len} natural parameters for d={d}, got {len(theta)}"
            )

        # Extract precision matrix
        eta = theta[:d]
        Lambda_half = theta[d:].reshape(d, d)
        Lambda = -2 * Lambda_half

        # Ensure symmetry
        Lambda = (Lambda + Lambda.T) / 2

        # Cholesky of Lambda (validates positive definiteness)
        try:
            L_Lambda = cholesky(Lambda, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Precision matrix Λ is not positive definite. "
                f"Min eigenvalue: {np.min(np.linalg.eigvalsh(Lambda)):.6f}"
            )

        # Compute Sigma = Lambda^{-1} via Cholesky solve
        # Lambda^{-1} = solve(L_Lambda.T, solve(L_Lambda, I))
        I_d = np.eye(d)
        Lambda_inv = solve_triangular(
            L_Lambda.T, solve_triangular(L_Lambda, I_d, lower=True), lower=False
        )
        Sigma = (Lambda_inv + Lambda_inv.T) / 2  # Ensure symmetry

        # Cholesky of Sigma
        L = cholesky(Sigma, lower=True)

        # Compute mu = Sigma @ eta
        mu = Sigma @ eta

        # Store internal state
        self._mu = mu
        self._L = L
        self._natural_params = tuple(theta)

        self._fitted = True
        self._invalidate_cache()

    def _compute_natural_params(self) -> NDArray:
        """Compute natural parameters from internal state."""
        d = self._d
        L_inv = self.L_inv
        Lambda = L_inv.T @ L_inv  # Sigma^{-1}
        eta = Lambda @ self._mu
        Lambda_half = -0.5 * Lambda
        return np.concatenate([eta, Lambda_half.flatten()])

    def _compute_classical_params(self):
        """Return frozen dataclass of classical parameters."""
        Sigma = self._L @ self._L.T
        return MultivariateNormalParams(mu=self._mu.copy(), sigma=Sigma)

    # ============================================================
    # Natural parameter support / validation
    # ============================================================

    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Natural parameter support.

        For MVN:
        - θ₁ (η = Λμ): unbounded (d components)
        - θ₂ (Λ_half = -1/2 Λ): must be negative definite
          (diagonal elements < 0, stored as d² components)
        """
        if self._d is None:
            raise ValueError("Dimension not set.")

        d = self._d
        bounds = []

        # η = Λμ: unbounded (d components)
        for _ in range(d):
            bounds.append((-np.inf, np.inf))

        # Λ_half = -1/2 Λ: stored as d² components (flattened)
        for i in range(d):
            for j in range(d):
                if i == j:
                    bounds.append((-np.inf, 0.0))
                else:
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

    # ============================================================
    # Exponential family structure
    # ============================================================

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
classical_params
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
        Log base measure: log h(x) = 0.

        The (2π)^{-d/2} normalization is absorbed into ψ(θ).
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return 0.0
        else:
            return np.zeros(x.shape[0])

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

        # Extract classical parameters from theta
        eta = theta[:d]
        Lambda_half = theta[d:].reshape(d, d)
        Lambda = -2 * Lambda_half  # Precision matrix
        Sigma = np.linalg.inv(Lambda)  # Covariance matrix
        mu = Sigma @ eta  # μ = Σ η = Λ^{-1} η

        # Compute expectation parameters
        eta1 = mu
        eta2 = (Sigma + np.outer(mu, mu)).flatten()

        return np.concatenate([eta1, eta2])

    def _expectation_to_natural(self, eta: NDArray, theta0=None) -> NDArray:
        """
        Convert expectation to natural parameters.

        From η = [μ, vec(Σ + μμ^T)]:
        - μ = η₁
        - Σ = η₂.reshape(d,d) - μμ^T

        Then compute natural parameters.
        """
        # Infer dimension from eta length: len(eta) = d + d^2
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
            min_eig = np.min(eigvals)
            Sigma += (-min_eig + 1e-6) * np.eye(d)

        # Convert to natural parameters
        Lambda = np.linalg.inv(Sigma)
        eta_nat = Lambda @ mu
        Lambda_half = -0.5 * Lambda
        return np.concatenate([eta_nat, Lambda_half.flatten()])

    def _get_initial_natural_params(self, eta: NDArray) -> NDArray:
        """Get initial guess for natural parameters."""
        return self._expectation_to_natural(eta)

    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Fisher information matrix.

        Uses numerical differentiation from base class.
        """
        return super().fisher_information(theta)

    # ============================================================
    # Override logpdf for Cholesky-based computation
    # ============================================================

    def logpdf(self, x: ArrayLike) -> Union[float, NDArray[np.floating]]:
        """
        Log probability density using Cholesky-based computation.

        Uses ``solve_triangular`` instead of ``np.linalg.inv`` for the
        Mahalanobis distance:

        .. math::
            \\log p(x|\\mu,\\Sigma) = -\\frac{d}{2}\\log(2\\pi)
            - \\frac{1}{2}\\log|\\Sigma|
            - \\frac{1}{2}(x-\\mu)^T \\Sigma^{-1}(x-\\mu)

        where :math:`\\Sigma^{-1}(x-\\mu) = L^{-T}(L^{-1}(x-\\mu))`.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate log PDF.
            Shape ``(d,)`` for single sample, ``(n, d)`` for n samples.

        Returns
        -------
        logpdf : float or ndarray
            Log probability density at each point.
        """
        self._check_fitted()

        x = np.asarray(x, dtype=float)
        d = self._d
        mu = self._mu
        L = self._L
        log_det = self.log_det_Sigma  # Cached

        const = -0.5 * d * np.log(2 * np.pi) - 0.5 * log_det

        # Handle single sample
        if x.ndim == 1:
            if len(x) != d:
                raise ValueError(f"Expected {d}-dimensional input, got {len(x)}")

            diff = x - mu
            # Solve L @ z = diff => z = L^{-1}(x - μ)
            z = solve_triangular(L, diff, lower=True)
            mahal = z @ z  # ||z||^2 = (x-μ)^T Σ^{-1} (x-μ)

            return float(const - 0.5 * mahal)

        # Handle multiple samples (n, d)
        n = x.shape[0]
        if x.shape[1] != d:
            raise ValueError(f"Expected {d}-dimensional input, got {x.shape[1]}")

        diff = x - mu  # (n, d)
        # Solve L @ Z = diff.T => Z = L^{-1}(X - μ)^T, shape (d, n)
        Z = solve_triangular(L, diff.T, lower=True)
        mahal = np.sum(Z ** 2, axis=0)  # (n,)

        return const - 0.5 * mahal

    def pdf(self, x: ArrayLike) -> Union[float, NDArray[np.floating]]:
        """
        Probability density: p(x|θ) = exp(logpdf(x)).
        """
        return np.exp(self.logpdf(x))

    # ============================================================
    # Distribution methods
    # ============================================================

    def rvs(self, size=None, random_state=None) -> NDArray:
        """
        Generate random samples using :math:`X = \\mu + L Z` where :math:`Z \\sim N(0, I)`.

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
        self._check_fitted()

        mu = self._mu
        L = self._L
        d = self._d

        # Set up RNG
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state

        # X = μ + L @ Z where Z ~ N(0, I)
        if size is None:
            z = rng.standard_normal(d)
            return mu + L @ z
        else:
            if isinstance(size, int):
                z = rng.standard_normal((size, d))
            else:
                z = rng.standard_normal((*size, d))
            return mu + z @ L.T  # Equivalent to (L @ z.T).T

    def mean(self) -> NDArray:
        """Mean of the distribution: E[X] = μ."""
        self._check_fitted()
        return self._mu.copy()

    def var(self) -> NDArray:
        """Variance (diagonal of covariance matrix)."""
        self._check_fitted()
        # diag(Σ) = diag(L L^T) = sum of squares of rows of L
        return np.sum(self._L ** 2, axis=1)

    def cov(self) -> NDArray:
        """Covariance matrix Σ = L L^T."""
        self._check_fitted()
        return self._L @ self._L.T

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
        self._check_fitted()

        x = np.asarray(x)
        mu = self._mu
        Sigma = self._L @ self._L.T
        d = self._d

        # Use scipy's implementation
        if d == 1:
            sigma_scalar = np.sqrt(Sigma[0, 0])
            mu_scalar = mu[0]

            if x.ndim == 0:
                return float(stats.norm.cdf(x, loc=mu_scalar, scale=sigma_scalar))
            elif x.ndim == 1 and len(x) == 1:
                return float(stats.norm.cdf(x[0], loc=mu_scalar, scale=sigma_scalar))
            elif x.ndim == 1:
                return stats.norm.cdf(x, loc=mu_scalar, scale=sigma_scalar)
            else:
                return stats.norm.cdf(x.flatten(), loc=mu_scalar, scale=sigma_scalar)
        else:
            rv = stats.multivariate_normal(mean=mu, cov=Sigma)
            return rv.cdf(x)

    def entropy(self) -> float:
        """
        Differential entropy.

        .. math::
            H(X) = \\frac{d}{2}(1 + \\log(2\\pi)) + \\frac{1}{2}\\log|\\Sigma|
        """
        self._check_fitted()
        d = self._d
        return 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * self.log_det_Sigma

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
        Sigma_hat = np.cov(X, rowvar=False, bias=True)

        # Ensure it's 2D for d=1 case
        if d == 1:
            Sigma_hat = np.array([[Sigma_hat]])

        # Regularization for numerical stability
        min_eig = np.min(np.linalg.eigvalsh(Sigma_hat))
        if min_eig < 1e-10:
            Sigma_hat += (1e-10 - min_eig) * np.eye(d)

        # Set parameters via new interface
        self._set_from_classical(mu=mu_hat, sigma=Sigma_hat)

        return self

    # ============================================================
    # Scipy compatibility
    # ============================================================

    def to_scipy(self) -> stats.multivariate_normal:
        """
        Convert to scipy.stats.multivariate_normal.
        """
        self._check_fitted()
        Sigma = self._L @ self._L.T
        return stats.multivariate_normal(mean=self._mu, cov=Sigma)

    @classmethod
    def from_scipy(cls, rv: stats.multivariate_normal) -> 'MultivariateNormal':
        """
        Create from scipy.stats.multivariate_normal.
        """
        return cls.from_classical_params(mu=rv.mean, sigma=rv.cov)

    # ============================================================
    # String representation
    # ============================================================

    def __repr__(self) -> str:
        """String representation."""
        if not self._fitted:
            if self._d is not None:
                return f"MultivariateNormal(d={self._d}, not fitted)"
            return "MultivariateNormal(not fitted)"

        d = self._d
        mu = self._mu

        if d == 1:
            sigma_sq = self._L[0, 0] ** 2
            return f"MultivariateNormal(μ={mu[0]:.4f}, σ²={sigma_sq:.4f})"
        elif d <= 3:
            mu_str = ", ".join(f"{x:.4f}" for x in mu)
            return f"MultivariateNormal(μ=[{mu_str}], d={d})"
        else:
            return f"MultivariateNormal(d={d})"


# Alias for convenience
MVN = MultivariateNormal
