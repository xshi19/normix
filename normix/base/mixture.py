"""
Base classes for normal mixture distributions.

Normal mixture distributions have the form:

.. math::
    X \\stackrel{d}{=} \\mu + \\gamma Y + \\sqrt{Y} Z

where :math:`Z \\sim N(0, \\Sigma)` is a Gaussian random vector independent of
:math:`Y`, and :math:`Y` follows some positive mixing distribution (GIG, Gamma,
Inverse Gaussian, etc.).

This module provides two abstract base classes:

1. **JointNormalMixture**: The joint distribution :math:`f(x, y)` which IS an
   exponential family. This class extends :class:`ExponentialFamily`.

2. **NormalMixture**: The marginal distribution :math:`f(x) = \\int f(x, y) dy`
   which is NOT an exponential family. This class extends :class:`Distribution`
   and owns a :class:`JointNormalMixture` instance accessible via the ``.joint``
   property.

The key insight is that while the marginal distribution of :math:`X` (with
:math:`Y` integrated out) does not belong to the exponential family, the
joint distribution :math:`(X, Y)` does belong to the exponential family
when both variables are observed.

Method naming convention:

- ``pdf(x)``: Marginal PDF :math:`f(x)` (on NormalMixture)
- ``joint.pdf(x, y)``: Joint PDF :math:`f(x, y)` (via .joint property)
- ``pdf_joint(x, y)``: Convenience alias for ``joint.pdf(x, y)``
- ``rvs(size)``: Sample from marginal (returns X only)
- ``rvs_joint(size)``: Sample from joint (returns X, Y tuple)
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

from scipy.linalg import cholesky, cho_solve, solve_triangular

from .distribution import Distribution
from .exponential_family import ExponentialFamily


# ============================================================================
# JointNormalMixture: Base class for joint distributions f(x, y)
# ============================================================================

class JointNormalMixture(ExponentialFamily, ABC):
    """
    Abstract base class for joint normal mixture distributions :math:`f(x, y)`.

    The joint distribution of :math:`(X, Y)` where:

    .. math::
        X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

        Y \\sim \\text{Mixing distribution}

    This joint distribution belongs to the exponential family with sufficient
    statistics:

    .. math::
        t(x, y) = \\begin{pmatrix} \\log y \\\\ y^{-1} \\\\ y \\\\ x \\\\ x y^{-1} \\\\ x x^T y^{-1} \\end{pmatrix}

    The natural parameters are derived from the classical parameters
    :math:`(\\mu, \\gamma, \\Sigma, \\text{mixing params})`.

    Subclasses must implement:

    - :meth:`_get_mixing_distribution_class`: Return the mixing distribution class
    - :meth:`_mixing_natural_params`: Extract mixing distribution natural params
    - :meth:`_mixing_sufficient_statistics`: Compute mixing dist sufficient stats
    - :meth:`_mixing_log_partition`: Compute mixing distribution log partition
    - :meth:`_classical_to_natural`: Convert classical to natural parameters
    - :meth:`_natural_to_classical`: Convert natural to classical parameters

    Attributes
    ----------
    _d : int or None
        Dimension of the observed variable :math:`X`.
    _natural_params : tuple or None
        Internal storage for natural parameters.

    See Also
    --------
    NormalMixture : Marginal distribution (owns a JointNormalMixture)
    ExponentialFamily : Parent class providing exponential family methods
    """

    _cached_attrs: Tuple[str, ...] = ExponentialFamily._cached_attrs + (
        'log_det_Sigma', 'L_Sigma_inv', 'gamma_mahal_sq',
    )

    def __init__(self, d: Optional[int] = None):
        """
        Initialize a joint normal mixture distribution.

        Parameters
        ----------
        d : int, optional
            Dimension of the observed variable X. Can be inferred from parameters.
        """
        super().__init__()
        self._d = d
        # Internal state: normal distribution parameters
        self._mu: Optional[NDArray] = None
        self._gamma: Optional[NDArray] = None
        self._L_Sigma: Optional[NDArray] = None  # Lower Cholesky of Sigma

    @property
    def d(self) -> int:
        """Dimension of the observed variable X."""
        if self._d is None:
            raise ValueError("Dimension not set. Use from_*_params() first.")
        return self._d

    # ========================================================================
    # Abstract methods for mixing distribution
    # ========================================================================

    @classmethod
    @abstractmethod
    def _get_mixing_distribution_class(cls) -> Type[ExponentialFamily]:
        """
        Return the class of the mixing distribution.

        Returns
        -------
        cls : Type[ExponentialFamily]
            The mixing distribution class (e.g., Gamma, InverseGaussian, GIG).
        """
        pass

    @abstractmethod
    def _get_mixing_natural_params(self, theta: NDArray) -> NDArray:
        """
        Extract mixing distribution natural parameters from joint natural params.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector for the joint distribution.

        Returns
        -------
        theta_y : ndarray
            Natural parameters for the mixing distribution of Y.
        """
        pass

    # ========================================================================
    # Helper methods for parameter extraction (shared across all subclasses)
    # ========================================================================

    def _extract_normal_params_from_theta(
        self, theta: NDArray, *, symmetrize: bool = False
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Extract normal distribution parameters from natural parameter vector.

        This helper method centralizes the common calculation of extracting
        :math:`(\\Lambda, \\Sigma, \\mu, \\gamma)` from the natural parameters,
        avoiding code duplication across multiple methods.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.
        symmetrize : bool, optional
            Whether to symmetrize Lambda for numerical stability.
            Default is False.

        Returns
        -------
        Lambda : ndarray
            Precision matrix :math:`\\Lambda = \\Sigma^{-1}`, shape (d, d).
        Sigma : ndarray
            Covariance scale matrix :math:`\\Sigma`, shape (d, d).
        mu : ndarray
            Location parameter :math:`\\mu`, shape (d,).
        gamma : ndarray
            Skewness parameter :math:`\\gamma`, shape (d,).

        Notes
        -----
        This method delegates to :meth:`_extract_normal_params_with_cholesky`
        and recovers Lambda from its Cholesky factor.

        See Also
        --------
        _extract_normal_params_with_cholesky : Core implementation using Cholesky.
        """
        # Delegate to Cholesky version and recover Lambda from L @ L.T
        L_Lambda, _, mu, gamma, Sigma = self._extract_normal_params_with_cholesky(
            theta, symmetrize=symmetrize, return_sigma=True
        )
        Lambda = L_Lambda @ L_Lambda.T

        return Lambda, Sigma, mu, gamma

    def _extract_normal_params_with_cholesky(
        self, theta: NDArray, *, symmetrize: bool = False, return_sigma: bool = False
    ) -> Tuple[NDArray, float, NDArray, NDArray] | Tuple[NDArray, float, NDArray, NDArray, NDArray]:
        """
        Extract normal parameters using Cholesky decomposition.

        More numerically efficient than :meth:`_extract_normal_params_from_theta`
        for computing log determinants and avoids explicit matrix inversion when
        only :math:`\\mu` and :math:`\\gamma` are needed.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.
        symmetrize : bool, optional
            Whether to symmetrize Lambda for numerical stability. Default False.
        return_sigma : bool, optional
            If True, also compute and return Sigma. Default False.

        Returns
        -------
        L_Lambda : ndarray
            Lower Cholesky factor of precision matrix, shape (d, d).
            Satisfies :math:`\\Lambda = L_{\\Lambda} L_{\\Lambda}^T`.
        log_det_Lambda : float
            Log determinant of Lambda: :math:`\\log|\\Lambda| = 2 \\sum_i \\log L_{ii}`.
        mu : ndarray
            Location parameter :math:`\\mu`, shape (d,).
        gamma : ndarray
            Skewness parameter :math:`\\gamma`, shape (d,).
        Sigma : ndarray, optional
            Covariance scale matrix, shape (d, d). Only returned if ``return_sigma=True``.

        Notes
        -----
        Instead of computing :math:`\\Sigma = \\Lambda^{-1}` explicitly, this method
        solves the linear systems :math:`\\Lambda \\mu = \\theta_5` and
        :math:`\\Lambda \\gamma = \\theta_4` using Cholesky factorization via
        :func:`scipy.linalg.cho_solve`.

        The log determinant is computed as:

        .. math::
            \\log|\\Lambda| = 2 \\sum_{i=1}^{d} \\log L_{ii}

        where :math:`L` is the lower Cholesky factor.

        If ``return_sigma=True``, the covariance matrix is computed by solving
        :math:`L_{\\Lambda} L_{\\Sigma}^T = I`, giving the Cholesky factor of Sigma,
        then :math:`\\Sigma = L_{\\Sigma} L_{\\Sigma}^T`.
        """
        d = self.d

        theta_4 = theta[3:3 + d]  # Λγ
        theta_5 = theta[3 + d:3 + 2 * d]  # Λμ
        theta_6 = theta[3 + 2 * d:].reshape(d, d)  # -1/2 Λ

        # Recover Λ from θ₆ = -1/2 Λ
        Lambda = -2 * theta_6

        if symmetrize:
            Lambda = (Lambda + Lambda.T) / 2

        # Cholesky factorization: Λ = L @ L.T
        L_Lambda = cholesky(Lambda, lower=True)

        # Log determinant: log|Λ| = 2 * Σ log(L_ii)
        log_det_Lambda = 2.0 * np.sum(np.log(np.diag(L_Lambda)))

        # Solve Λ @ μ = θ₅ and Λ @ γ = θ₄ using Cholesky factorization
        # cho_solve takes (L, lower) tuple and solves L @ L.T @ x = b
        mu = cho_solve((L_Lambda, True), theta_5)
        gamma = cho_solve((L_Lambda, True), theta_4)

        if return_sigma:
            # Compute L_Sigma where Sigma = L_Sigma @ L_Sigma.T
            # From L_Lambda @ L_Lambda.T = Lambda, we have Sigma = inv(Lambda)
            # L_Sigma = inv(L_Lambda).T, so solve L_Lambda @ L_Sigma.T = I
            # which gives L_Sigma.T = inv(L_Lambda) (lower triangular)
            # Then L_Sigma = inv(L_Lambda).T (upper triangular -> transpose to lower)
            L_Sigma_T = solve_triangular(L_Lambda, np.eye(d), lower=True)
            Sigma = L_Sigma_T.T @ L_Sigma_T
            return L_Lambda, log_det_Lambda, mu, gamma, Sigma

        return L_Lambda, log_det_Lambda, mu, gamma

    def _get_normal_params(self, theta: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Extract normal distribution parameters from joint natural params.

        Convenience method that returns the most commonly needed parameters
        :math:`(\\mu, \\gamma, \\Sigma)`.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector for the joint distribution.

        Returns
        -------
        mu : ndarray
            Location parameter :math:`\\mu`, shape (d,).
        gamma : ndarray
            Skewness parameter :math:`\\gamma`, shape (d,).
        Sigma : ndarray
            Covariance scale matrix :math:`\\Sigma`, shape (d, d).

        See Also
        --------
        _extract_normal_params_from_theta : Returns (Lambda, Sigma, mu, gamma).
        _extract_normal_params_with_cholesky : Cholesky-based version for efficiency.
        """
        _, Sigma, mu, gamma = self._extract_normal_params_from_theta(theta)
        return mu, gamma, Sigma

    # ========================================================================
    # Internal state management
    # ========================================================================

    def _set_from_natural(self, theta: NDArray) -> None:
        """
        Set internal state from natural parameters.

        Extracts and stores :math:`\\mu`, :math:`\\gamma`, and the Cholesky
        factor of :math:`\\Sigma` alongside the legacy tuple storage.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.
        """
        theta = np.asarray(theta, dtype=float)

        # Infer dimension if needed
        if self._d is None:
            # theta has 3 + 2d + d^2 entries
            # Solve d^2 + 2d + 3 - len(theta) = 0
            n = len(theta)
            # d^2 + 2d + 3 = n => d = (-2 + sqrt(4 + 4*(n-3))) / 2
            disc = 4 + 4 * (n - 3)
            d = int((-2 + np.sqrt(disc)) / 2)
            if 3 + 2 * d + d * d != n:
                raise ValueError(f"Cannot infer d from parameter length {n}")
            self._d = d

        # Validate and store legacy
        self._validate_natural_params(theta)
        self._natural_params = tuple(theta)

        # Extract normal params via Cholesky
        L_Lambda, _, mu, gamma, Sigma = self._extract_normal_params_with_cholesky(
            theta, symmetrize=True, return_sigma=True
        )
        L_Sigma = cholesky(Sigma, lower=True)

        # Store internal state
        self._mu = mu
        self._gamma = gamma
        self._L_Sigma = L_Sigma

        self._fitted = True
        self._invalidate_cache()

    def _compute_classical_params(self):
        """
        Compute classical parameters from internal state.

        Delegates to the legacy ``_natural_to_classical`` for now.
        Subclasses will override when migrated.
        """
        theta = self.natural_params
        return self._natural_to_classical(theta)

    # ========================================================================
    # Cached derived quantities
    # ========================================================================

    @cached_property
    def log_det_Sigma(self) -> float:
        r"""
        Log-determinant of the covariance scale matrix (cached).

        .. math::
            \log|\Sigma| = 2 \sum_{i=1}^d \log (L_\Sigma)_{ii}

        Returns
        -------
        log_det : float
        """
        self._check_fitted()
        return 2.0 * np.sum(np.log(np.diag(self._L_Sigma)))

    @cached_property
    def L_Sigma_inv(self) -> NDArray:
        r"""
        Inverse of the lower Cholesky factor of Sigma (cached).

        :math:`L_\Sigma^{-1}` such that :math:`\Sigma^{-1} = L_\Sigma^{-T} L_\Sigma^{-1}`.

        Returns
        -------
        L_inv : ndarray, shape ``(d, d)``
            Lower triangular matrix.
        """
        self._check_fitted()
        return solve_triangular(self._L_Sigma, np.eye(self._d), lower=True)

    @cached_property
    def gamma_mahal_sq(self) -> float:
        r"""
        Mahalanobis norm of gamma: :math:`\gamma^T \Sigma^{-1} \gamma` (cached).

        Used in marginal PDF computations.

        Returns
        -------
        mahal_sq : float
        """
        self._check_fitted()
        z = solve_triangular(self._L_Sigma, self._gamma, lower=True)
        return float(z @ z)

    # ========================================================================
    # Backward-compatible Cholesky factor access
    # ========================================================================

    def get_L_Sigma(self) -> Tuple[NDArray, float]:
        """
        Get the lower Cholesky factor of Sigma and its log determinant.

        Returns
        -------
        L_Sigma : ndarray
            Lower Cholesky factor of Sigma, shape (d, d).
        log_det_Sigma : float
            Log determinant of Sigma.

        Notes
        -----
        This method now delegates to the internal ``_L_Sigma`` attribute
        and the ``log_det_Sigma`` cached property.
        """
        self._check_fitted()
        return self._L_Sigma, self.log_det_Sigma

    def set_L_Sigma(self, L_Sigma: NDArray, lower: bool = True) -> None:
        """
        Set the Cholesky factor of Sigma.

        Used by the M-step of the EM algorithm.

        Parameters
        ----------
        L_Sigma : ndarray
            Cholesky factor of Sigma, shape (d, d).
        lower : bool, default True
            If True, L_Sigma is lower triangular.
            If False, converts to lower triangular.
        """
        if lower:
            self._L_Sigma = L_Sigma.copy()
        else:
            self._L_Sigma = L_Sigma.T.copy()
        # Invalidate cached properties that depend on L_Sigma
        for attr in ('log_det_Sigma', 'L_Sigma_inv', 'gamma_mahal_sq'):
            self.__dict__.pop(attr, None)

    def clear_L_Sigma_cache(self) -> None:
        """Clear cached Cholesky-derived quantities (backward compat)."""
        for attr in ('log_det_Sigma', 'L_Sigma_inv', 'gamma_mahal_sq'):
            self.__dict__.pop(attr, None)

    # ========================================================================
    # Exponential family components
    # ========================================================================

    def _sufficient_statistics(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Compute sufficient statistics :math:`t(x, y)`.

        The sufficient statistics for the joint normal mixture are:

        .. math::
            t(x, y) = [\\log y, y^{-1}, y, x, x y^{-1}, \\text{vec}(x x^T y^{-1})]

        Parameters
        ----------
        x : array_like
            Observed values of X. Shape (d,) or (n, d).
        y : array_like
            Observed values of Y. Shape () or (n,).

        Returns
        -------
        t : ndarray
            Sufficient statistics. For single observation: shape (3 + 2d + d²,).
            For n observations: shape (n, 3 + 2d + d²).
        """
        x = np.asarray(x)
        y = np.asarray(y)
        d = self.d

        # Handle single observation
        if y.ndim == 0 or (y.ndim == 1 and len(y) == 1):
            y_scalar = float(y) if y.ndim == 0 else float(y[0])
            if x.ndim == 1:
                x = x.reshape(-1)
            elif x.ndim == 2 and x.shape[0] == 1:
                x = x[0]

            if len(x) != d:
                raise ValueError(f"Expected x of dimension {d}, got {len(x)}")

            inv_y = 1.0 / y_scalar
            x_inv_y = x * inv_y
            xxT_inv_y = np.outer(x, x) * inv_y

            # Stack: [log(y), 1/y, y, x, x/y, vec(xxT/y)]
            return np.concatenate([
                [np.log(y_scalar), inv_y, y_scalar],
                x,
                x_inv_y,
                xxT_inv_y.flatten()
            ])

        # Handle multiple observations
        n = len(y)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] != n:
            raise ValueError(f"x has {x.shape[0]} samples but y has {n}")
        if x.shape[1] != d:
            raise ValueError(f"Expected x of dimension {d}, got {x.shape[1]}")

        # Vectorized computation of all statistics
        inv_y = 1.0 / y  # (n,)
        x_inv_y = x / y[:, np.newaxis]  # (n, d)

        # Compute xx^T / y for all samples: (n, d, d) -> (n, d*d)
        # xxT_inv_y[i] = outer(x[i], x[i]) / y[i]
        xxT_inv_y = np.einsum('ni,nj->nij', x, x) / y[:, np.newaxis, np.newaxis]
        xxT_inv_y_flat = xxT_inv_y.reshape(n, d * d)  # (n, d*d)

        # Stack all statistics: [log(y), 1/y, y, x, x/y, vec(xxT/y)]
        t = np.column_stack([
            np.log(y),      # (n,)
            inv_y,          # (n,)
            y,              # (n,)
            x,              # (n, d)
            x_inv_y,        # (n, d)
            xxT_inv_y_flat  # (n, d*d)
        ])

        return t

    def _log_base_measure(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Log base measure: :math:`\\log h(x, y) = -\\frac{d}{2} \\log(2\\pi)` for :math:`y > 0`.

        Parameters
        ----------
        x : array_like
            Observed values of X.
        y : array_like
            Observed values of Y.

        Returns
        -------
        log_h : ndarray
            Log base measure value(s). Returns -inf for y <= 0.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        d = self.d

        log_h_const = -0.5 * d * np.log(2 * np.pi)

        if y.ndim == 0:
            return log_h_const if y > 0 else -np.inf

        result = np.full_like(y, log_h_const, dtype=float)
        result[y <= 0] = -np.inf
        return result

    # ========================================================================
    # PDF and log PDF (override for joint distribution signature)
    # ========================================================================

    def pdf(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Joint probability density function :math:`f(x, y)`.

        Parameters
        ----------
        x : array_like
            Observed values of X. Shape (d,) or (n, d).
        y : array_like
            Observed values of Y. Shape () or (n,).

        Returns
        -------
        pdf : ndarray
            Joint density values.
        """
        return np.exp(self.logpdf(x, y))

    def logpdf(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Log joint probability density function :math:`\\log f(x, y)`.

        .. math::
            \\log f(x, y) = \\log h(x, y) + \\theta^T t(x, y) - \\psi(\\theta)

        Parameters
        ----------
        x : array_like
            Observed values of X. Shape (d,) or (n, d).
        y : array_like
            Observed values of Y. Shape () or (n,).

        Returns
        -------
        logpdf : ndarray
            Log joint density values.
        """
        self._check_fitted()

        theta = self.natural_params
        t_xy = self._sufficient_statistics(x, y)
        log_h = self._log_base_measure(x, y)
        psi = self._log_partition(theta)

        # Handle single vs multiple observations
        if t_xy.ndim == 1:
            log_h_scalar = float(log_h) if np.ndim(log_h) == 0 else float(log_h.flat[0])
            return log_h_scalar + np.dot(theta, t_xy) - psi
        else:
            return log_h + t_xy @ theta - psi

    # ========================================================================
    # Random sampling
    # ========================================================================

    def rvs(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Generate random samples from the joint distribution.

        Samples :math:`(X, Y)` pairs where:

        1. :math:`Y \\sim` mixing distribution
        2. :math:`X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)`

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of samples to generate. If None, returns single sample.
        random_state : int or Generator, optional
            Random number generator or seed.

        Returns
        -------
        X : ndarray
            Sampled X values. Shape (d,) if size is None, (n, d) otherwise.
        Y : ndarray
            Sampled Y values. Shape () if size is None, (n,) otherwise.
        """
        self._check_fitted()

        # Set up RNG
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state

        # Use internal state directly
        mu = self._mu
        gamma = self._gamma
        L_Sigma = self._L_Sigma
        d = self._d
        theta = self.natural_params

        # Create mixing distribution instance
        mixing_class = self._get_mixing_distribution_class()
        mixing_theta = self._get_mixing_natural_params(theta)
        mixing_dist = mixing_class.from_natural_params(mixing_theta)

        # Sample Y from mixing distribution
        Y = mixing_dist.rvs(size=size, random_state=rng)

        # Sample X | Y ~ N(mu + gamma * Y, Sigma * Y)
        # Using the representation: X = mu + gamma * Y + sqrt(Y) * Z, Z ~ N(0, Sigma)
        # Since Σ = L @ L.T, we have: Z = L @ W where W ~ N(0, I)
        if size is None:
            # Single sample
            Y_scalar = float(Y)
            W = rng.standard_normal(d)  # W ~ N(0, I)
            Z = L_Sigma @ W  # Z ~ N(0, Sigma)
            X = mu + gamma * Y_scalar + np.sqrt(Y_scalar) * Z
        else:
            # Multiple samples - vectorized
            n = size if isinstance(size, int) else np.prod(size)
            Y_flat = np.atleast_1d(Y).flatten()

            # Sample W ~ N(0, I) for all n samples, then transform to Z ~ N(0, Sigma)
            # W has shape (n, d), Z = W @ L.T (since Z_i = L @ W_i means Z = W @ L.T in matrix form)
            W = rng.standard_normal((n, d))  # (n, d)
            Z = W @ L_Sigma.T  # (n, d) @ (d, d) = (n, d), each row is L @ W_i

            # Vectorized: X = mu + gamma * Y + sqrt(Y) * Z
            # Shape: (n, d) = (d,) + (d,) * (n, 1) + (n, d) * (n, 1)
            sqrt_Y = np.sqrt(Y_flat)[:, np.newaxis]  # (n, 1)
            X = mu + np.outer(Y_flat, gamma) + Z * sqrt_Y

            # Reshape if needed
            if isinstance(size, tuple):
                X = X.reshape(size + (d,))
                Y = Y_flat.reshape(size)

        return X, Y

    # ========================================================================
    # Conditional distribution methods
    # ========================================================================

    def conditional_mean_x_given_y(self, y: ArrayLike) -> NDArray:
        """
        Conditional mean :math:`E[X | Y = y] = \\mu + \\gamma y`.

        Parameters
        ----------
        y : array_like
            Conditioning value(s) of Y.

        Returns
        -------
        mean : ndarray
            Conditional mean of X given Y.
        """
        self._check_fitted()

        mu = self._mu
        gamma = self._gamma
        y = np.asarray(y)

        if y.ndim == 0:
            return mu + gamma * float(y)
        else:
            return mu[np.newaxis, :] + np.outer(y, gamma)

    def conditional_cov_x_given_y(self, y: ArrayLike) -> NDArray:
        """
        Conditional covariance :math:`\\text{Cov}[X | Y = y] = \\Sigma y`.

        Parameters
        ----------
        y : array_like
            Conditioning value(s) of Y.

        Returns
        -------
        cov : ndarray
            Conditional covariance of X given Y.
        """
        self._check_fitted()

        Sigma = self._L_Sigma @ self._L_Sigma.T
        y = np.asarray(y)

        if y.ndim == 0:
            return Sigma * float(y)
        else:
            # Return array of covariance matrices
            return Sigma[np.newaxis, :, :] * y[:, np.newaxis, np.newaxis]

    # ========================================================================
    # Moments
    # ========================================================================

    def mean(self) -> Tuple[NDArray, float]:
        """
        Mean of the joint distribution.

        Returns
        -------
        E_X : ndarray
            :math:`E[X] = \\mu + \\gamma E[Y]`
        E_Y : float
            :math:`E[Y]` from the mixing distribution
        """
        self._check_fitted()

        mu = self._mu
        gamma = self._gamma
        theta = self.natural_params

        # Get E[Y] from mixing distribution
        mixing_class = self._get_mixing_distribution_class()
        mixing_theta = self._get_mixing_natural_params(theta)
        mixing_dist = mixing_class.from_natural_params(mixing_theta)
        E_Y = float(mixing_dist.mean())

        E_X = mu + gamma * E_Y

        return E_X, E_Y

    def var(self) -> Tuple[NDArray, float]:
        """
        Variance of the joint distribution (diagonal of covariance).

        Returns
        -------
        Var_X : ndarray
            Variance of X (diagonal of covariance matrix)
        Var_Y : float
            Variance of Y from the mixing distribution
        """
        cov_X, var_Y = self.cov()
        return np.diag(cov_X), var_Y

    def cov(self) -> Tuple[NDArray, float]:
        """
        Covariance of the joint distribution.

        .. math::
            \\text{Cov}[X] = E[Y] \\Sigma + \\text{Var}[Y] \\gamma \\gamma^T

        Returns
        -------
        Cov_X : ndarray
            Covariance matrix of X, shape (d, d)
        Var_Y : float
            Variance of Y from the mixing distribution
        """
        self._check_fitted()

        gamma = self._gamma
        Sigma = self._L_Sigma @ self._L_Sigma.T
        theta = self.natural_params

        # Get E[Y] and Var[Y] from mixing distribution
        mixing_class = self._get_mixing_distribution_class()
        mixing_theta = self._get_mixing_natural_params(theta)
        mixing_dist = mixing_class.from_natural_params(mixing_theta)
        E_Y = float(mixing_dist.mean())
        Var_Y = float(mixing_dist.var())

        # Cov[X] = E[Y] * Sigma + Var[Y] * gamma * gamma^T
        Cov_X = E_Y * Sigma + Var_Y * np.outer(gamma, gamma)

        return Cov_X, Var_Y

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation of the joint distribution."""
        name = self.__class__.__name__
        if not self._fitted:
            if self._d is not None:
                return f"{name}(d={self._d}, not fitted)"
            return f"{name}(not fitted)"
        return f"{name}(d={self.d})"


# ============================================================================
# NormalMixture: Base class for marginal distributions f(x)
# ============================================================================

class NormalMixture(Distribution, ABC):
    """
    Abstract base class for marginal normal mixture distributions :math:`f(x)`.

    The marginal distribution of :math:`X` where:

    .. math::
        f(x) = \\int_0^\\infty f(x, y) \\, dy = \\int_0^\\infty f(x | y) f(y) \\, dy

    This marginal distribution is NOT an exponential family. However, the joint
    distribution :math:`f(x, y)` IS an exponential family and can be accessed
    via the :attr:`joint` property.

    Method naming convention:

    - ``pdf(x)``: Marginal PDF :math:`f(x)` (default)
    - ``joint.pdf(x, y)``: Joint PDF :math:`f(x, y)` (via .joint property)
    - ``pdf_joint(x, y)``: Convenience alias for ``joint.pdf(x, y)``
    - ``rvs(size)``: Sample from marginal (returns X only)
    - ``rvs_joint(size)``: Sample from joint (returns X, Y tuple)

    Subclasses must implement:

    - :meth:`_create_joint_distribution`: Factory method to create the joint dist
    - :meth:`_marginal_logpdf`: Compute the marginal log PDF
    - :meth:`_conditional_expectation_y_given_x`: E[Y | X] for EM algorithm

    Attributes
    ----------
    _joint : JointNormalMixture or None
        The underlying joint distribution instance.

    See Also
    --------
    JointNormalMixture : Joint distribution (exponential family)
    Distribution : Parent class defining the scipy-like API

    Examples
    --------
    >>> # Access marginal distribution methods (default)
    >>> gh = GeneralizedHyperbolic.from_classical_params(mu=0, gamma=1, ...)
    >>> gh.pdf(x)  # Marginal PDF f(x)
    >>> gh.rvs(size=100)  # Sample X values only

    >>> # Access joint distribution methods via .joint property
    >>> gh.joint.pdf(x, y)  # Joint PDF f(x, y)
    >>> gh.joint.sufficient_statistics(x, y)  # Exponential family stats
    >>> gh.joint.natural_params  # Natural parameters

    >>> # Convenience aliases for joint methods
    >>> gh.pdf_joint(x, y)  # Same as gh.joint.pdf(x, y)
    >>> gh.rvs_joint(size=100)  # Returns (X, Y) tuple
    """

    def __init__(self):
        """Initialize an unfitted marginal normal mixture distribution."""
        super().__init__()
        self._joint: Optional[JointNormalMixture] = None

    # ========================================================================
    # Abstract methods
    # ========================================================================

    @abstractmethod
    def _create_joint_distribution(self) -> JointNormalMixture:
        """
        Factory method to create the underlying joint distribution.

        Returns
        -------
        joint : JointNormalMixture
            A new instance of the corresponding joint distribution class.
        """
        pass

    @abstractmethod
    def _marginal_logpdf(self, x: ArrayLike) -> NDArray:
        """
        Compute the marginal log PDF :math:`\\log f(x)`.

        This is the distribution-specific implementation that computes
        the marginal density by integrating out Y.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log PDF.

        Returns
        -------
        logpdf : ndarray
            Log probability density values.
        """
        pass

    @abstractmethod
    def _conditional_expectation_y_given_x(
        self, x: ArrayLike
    ) -> Dict[str, NDArray]:
        """
        Compute conditional expectations :math:`E[g(Y) | X = x]` for EM algorithm.

        These conditional expectations are used in the E-step of the EM algorithm.

        Parameters
        ----------
        x : array_like
            Observed X values.

        Returns
        -------
        expectations : dict
            Dictionary containing:
            - 'E_Y': :math:`E[Y | X]`
            - 'E_inv_Y': :math:`E[1/Y | X]`
            - 'E_log_Y': :math:`E[\\log Y | X]`
        """
        pass

    # ========================================================================
    # Joint distribution access
    # ========================================================================

    @property
    def joint(self) -> JointNormalMixture:
        """
        Access the joint distribution :math:`f(x, y)`.

        The joint distribution is an :class:`ExponentialFamily` with all
        associated methods (natural parameters, sufficient statistics,
        Fisher information, etc.).

        Returns
        -------
        joint : JointNormalMixture
            The underlying joint distribution instance.

        Examples
        --------
        >>> gh = GeneralizedHyperbolic.from_classical_params(...)
        >>> gh.joint.pdf(x, y)  # Joint PDF
        >>> gh.joint.natural_params  # Natural parameters
        >>> gh.joint.sufficient_statistics(x, y)  # Sufficient stats
        """
        if self._joint is None:
            raise ValueError("Distribution not initialized. Use from_*_params().")
        return self._joint

    @property
    def d(self) -> int:
        """Dimension of the distribution."""
        return self.joint.d

    # ========================================================================
    # Factory methods (override to sync with joint)
    # ========================================================================

    @classmethod
    def from_classical_params(cls, **kwargs) -> 'NormalMixture':
        """
        Create distribution from classical parameters.

        Parameters
        ----------
        **kwargs
            Distribution-specific classical parameters.
            Common parameters:
            - mu : Location parameter (d,)
            - gamma : Skewness parameter (d,)
            - sigma : Covariance scale matrix (d, d)
            - Plus mixing distribution parameters

        Returns
        -------
        dist : NormalMixture
            Distribution instance with parameters set.

        Examples
        --------
        >>> gh = GeneralizedHyperbolic.from_classical_params(
        ...     mu=0.0, gamma=0.5, sigma=1.0, p=-0.5, a=1.0, b=1.0
        ... )
        """
        instance = cls()
        instance.set_classical_params(**kwargs)
        return instance

    @classmethod
    def from_natural_params(cls, theta: NDArray) -> 'NormalMixture':
        """
        Create distribution from natural parameters.

        The natural parameters are those of the joint distribution.

        Parameters
        ----------
        theta : ndarray
            Natural parameter vector for the joint distribution.

        Returns
        -------
        dist : NormalMixture
            Distribution instance with parameters set.
        """
        instance = cls()
        instance.set_natural_params(theta)
        return instance

    @classmethod
    def from_expectation_params(cls, eta: NDArray) -> 'NormalMixture':
        """
        Create distribution from expectation parameters.

        The expectation parameters are those of the joint distribution.

        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector for the joint distribution.

        Returns
        -------
        dist : NormalMixture
            Distribution instance with parameters set.
        """
        instance = cls()
        instance.set_expectation_params(eta)
        return instance

    # ========================================================================
    # Parameter setters (delegate to joint)
    # ========================================================================

    def set_classical_params(self, **kwargs) -> 'NormalMixture':
        """
        Set parameters from classical parametrization.

        Parameters
        ----------
        **kwargs
            Distribution-specific classical parameters.

        Returns
        -------
        self : NormalMixture
            Returns self for method chaining.
        """
        if self._joint is None:
            self._joint = self._create_joint_distribution()

        self._joint.set_classical_params(**kwargs)
        return self

    def set_natural_params(self, theta: NDArray) -> 'NormalMixture':
        """
        Set parameters from natural parametrization.

        Parameters
        ----------
        theta : ndarray
            Natural parameter vector for the joint distribution.

        Returns
        -------
        self : NormalMixture
            Returns self for method chaining.
        """
        if self._joint is None:
            self._joint = self._create_joint_distribution()

        self._joint.set_natural_params(theta)
        return self

    def set_expectation_params(self, eta: NDArray) -> 'NormalMixture':
        """
        Set parameters from expectation parametrization.

        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector for the joint distribution.

        Returns
        -------
        self : NormalMixture
            Returns self for method chaining.
        """
        if self._joint is None:
            self._joint = self._create_joint_distribution()

        self._joint.set_expectation_params(eta)
        return self

    # ========================================================================
    # Parameter getters (delegate to joint)
    # ========================================================================

    def get_natural_params(self) -> NDArray:
        """
        Get natural parameters of the joint distribution.

        Note: The marginal distribution itself is not an exponential family,
        so these are the natural parameters of the joint distribution.

        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
        return self.joint.get_natural_params()

    def get_expectation_params(self) -> NDArray:
        """
        Get expectation parameters of the joint distribution.

        Returns
        -------
        eta : ndarray
            Expectation parameter vector :math:`E[t(X, Y)]`.
        """
        return self.joint.get_expectation_params()

    def get_classical_params(self) -> Dict[str, Any]:
        """
        Get classical parameters.

        Returns
        -------
        params : dict
            Dictionary of classical parameters.
        """
        return self.joint.get_classical_params()

    # ========================================================================
    # Marginal distribution methods (main interface)
    # ========================================================================

    def pdf(self, x: ArrayLike) -> NDArray:
        """
        Marginal probability density function :math:`f(x)`.

        .. math::
            f(x) = \\int_0^\\infty f(x, y) \\, dy

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF.

        Returns
        -------
        pdf : ndarray
            Probability density values.
        """
        return np.exp(self.logpdf(x))

    def logpdf(self, x: ArrayLike) -> NDArray:
        """
        Log marginal probability density function :math:`\\log f(x)`.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log PDF.

        Returns
        -------
        logpdf : ndarray
            Log probability density values.
        """
        if self._joint is None:
            raise ValueError("Parameters not set. Use from_*_params().")
        return self._marginal_logpdf(x)

    def rvs(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> NDArray:
        """
        Generate random samples from the marginal distribution.

        Samples X values only (Y is integrated out).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of samples to generate.
        random_state : int or Generator, optional
            Random number generator or seed.

        Returns
        -------
        X : ndarray
            Sampled X values. Shape ``(d,)`` if size is None, else ``(n, d)`` or
            ``(*size, d)``.
        """
        X, _ = self.rvs_joint(size=size, random_state=random_state)
        return X

    # ========================================================================
    # Joint distribution convenience methods
    # ========================================================================

    def pdf_joint(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Joint probability density function :math:`f(x, y)`.

        Convenience alias for ``self.joint.pdf(x, y)``.

        Parameters
        ----------
        x : array_like
            Observed X values.
        y : array_like
            Observed Y values.

        Returns
        -------
        pdf : ndarray
            Joint density values.
        """
        return self.joint.pdf(x, y)

    def logpdf_joint(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """
        Log joint probability density function :math:`\\log f(x, y)`.

        Convenience alias for ``self.joint.logpdf(x, y)``.

        Parameters
        ----------
        x : array_like
            Observed X values.
        y : array_like
            Observed Y values.

        Returns
        -------
        logpdf : ndarray
            Log joint density values.
        """
        return self.joint.logpdf(x, y)

    def rvs_joint(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Generate random samples from the joint distribution.

        Samples both X and Y values.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of samples to generate.
        random_state : int or Generator, optional
            Random number generator or seed.

        Returns
        -------
        X : ndarray
            Sampled X values.
        Y : ndarray
            Sampled Y values (latent mixing variable).
        """
        return self.joint.rvs(size=size, random_state=random_state)

    # ========================================================================
    # Moments (marginal)
    # ========================================================================

    def mean(self) -> NDArray:
        """
        Mean of the marginal distribution.

        .. math::
            E[X] = \\mu + \\gamma E[Y]

        Returns
        -------
        mean : ndarray
            Mean vector of X, shape (d,).
        """
        E_X, _ = self.joint.mean()
        return E_X

    def var(self) -> NDArray:
        """
        Variance of the marginal distribution (diagonal of covariance).

        Returns
        -------
        var : ndarray
            Variance of each component, shape (d,).
        """
        return np.diag(self.cov())

    def cov(self) -> NDArray:
        """
        Covariance matrix of the marginal distribution.

        .. math::
            \\text{Cov}[X] = E[Y] \\Sigma + \\text{Var}[Y] \\gamma \\gamma^T

        Returns
        -------
        cov : ndarray
            Covariance matrix, shape (d, d).
        """
        Cov_X, _ = self.joint.cov()
        return Cov_X

    def std(self) -> NDArray:
        """
        Standard deviation of the marginal distribution.

        Returns
        -------
        std : ndarray
            Standard deviation of each component, shape (d,).
        """
        return np.sqrt(self.var())

    # ========================================================================
    # Mixing distribution access
    # ========================================================================

    @property
    def mixing_distribution(self) -> ExponentialFamily:
        """
        Access the mixing distribution of Y.

        Returns
        -------
        mixing : ExponentialFamily
            The mixing distribution (e.g., GIG, Gamma, InverseGaussian).
        """
        theta = self.joint.get_natural_params()
        mixing_class = self.joint._get_mixing_distribution_class()
        mixing_theta = self.joint._get_mixing_natural_params(theta)
        return mixing_class.from_natural_params(mixing_theta)

    # ========================================================================
    # Fitting (placeholder - to be implemented in subclasses)
    # ========================================================================

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None, **kwargs) -> 'NormalMixture':
        """
        Fit distribution to data using EM algorithm.

        When only X is observed (Y is latent), uses the EM algorithm.
        For complete data (both X and Y observed), use :meth:`fit_complete`.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d).
        y : array_like, optional
            Ignored.
        **kwargs
            Additional fitting parameters.

        Returns
        -------
        self : NormalMixture
            Fitted distribution.

        Note
        ----
        Not yet implemented. Placeholder for EM algorithm.
        """
        raise NotImplementedError(
            "EM fitting not yet implemented. "
            "Use fit_complete(X, Y) if you have complete data."
        )

    def fit_complete(self, X: ArrayLike, Y: ArrayLike, **kwargs) -> 'NormalMixture':
        """
        Fit distribution from complete data (both X and Y observed).

        Since the joint distribution is an exponential family, this uses
        closed-form MLE.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d).
        Y : array_like
            Observed Y data, shape (n_samples,).
        **kwargs
            Additional fitting parameters.

        Returns
        -------
        self : NormalMixture
            Fitted distribution.

        Note
        ----
        Not yet implemented.
        """
        raise NotImplementedError("Complete data fitting not yet implemented.")

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation of the marginal distribution."""
        name = self.__class__.__name__
        if self._joint is None:
            return f"{name}(not fitted)"
        try:
            d = self.d
            return f"{name}(d={d})"
        except ValueError:
            return f"{name}(not fitted)"
