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

from scipy.linalg import cho_solve, solve_triangular

from normix.utils import robust_cholesky, column_median_mad

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
    - :meth:`_store_mixing_params`: Store mixing params from kwargs
    - :meth:`_store_mixing_params_from_theta`: Extract mixing params from theta
    - :meth:`_compute_mixing_theta`: Build theta_1, theta_2, theta_3
    - :meth:`_create_mixing_distribution`: Construct mixing dist from named attrs

    Attributes
    ----------
    _d : int or None
        Dimension of the observed variable :math:`X`.

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
        self._mu: Optional[NDArray] = None
        self._gamma: Optional[NDArray] = None
        self._L_Sigma: Optional[NDArray] = None

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
    def _store_mixing_params(self, **kwargs) -> None:
        """
        Store mixing distribution parameters as named attributes.

        Called by ``_set_from_classical`` and ``_set_internal``.

        Parameters
        ----------
        **kwargs
            Mixing-specific parameters (e.g., shape, rate for VG/NInvG;
            delta, eta for NIG; p, a, b for GH).
        """
        pass

    @abstractmethod
    def _store_mixing_params_from_theta(self, theta: NDArray) -> None:
        """
        Extract and store mixing parameters from natural parameter vector.

        Called by ``_set_from_natural`` after normal params have been stored.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.
        """
        pass

    @abstractmethod
    def _compute_mixing_theta(
        self, theta_4: NDArray, theta_5: NDArray
    ) -> Tuple[float, float, float]:
        """
        Compute the first three natural parameters from mixing params.

        Parameters
        ----------
        theta_4 : ndarray
            :math:`\\Lambda \\gamma`, shape (d,).
        theta_5 : ndarray
            :math:`\\Lambda \\mu`, shape (d,).

        Returns
        -------
        theta_1, theta_2, theta_3 : float
            First three scalar natural parameters.
        """
        pass

    @abstractmethod
    def _create_mixing_distribution(self) -> ExponentialFamily:
        """
        Create mixing distribution from named internal attributes.

        Returns
        -------
        mixing : ExponentialFamily
            A fitted mixing distribution instance.
        """
        pass

    # ========================================================================
    # Helper methods for parameter extraction (shared across all subclasses)
    # ========================================================================

    def _extract_normal_params_with_cholesky(
        self, theta: NDArray, *, return_sigma: bool = False
    ) -> Tuple[NDArray, float, NDArray, NDArray] | Tuple[NDArray, float, NDArray, NDArray, NDArray]:
        """
        Extract normal parameters using Cholesky decomposition.

        Numerically efficient method for extracting parameters from theta
        using Cholesky decomposition. Avoids explicit matrix inversion when
        only :math:`\\mu` and :math:`\\gamma` are needed.

        Uses :func:`~normix.utils.robust_cholesky` for the Cholesky factorization,
        which handles near-singular matrices via diagonal regularization. Since
        LAPACK only reads one triangle, explicit symmetrization is unnecessary.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.
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

        # Cholesky factorization: Λ = L @ L.T
        L_Lambda = robust_cholesky(Lambda)

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

    # ========================================================================
    # Internal state management
    # ========================================================================

    def _store_normal_params(self, *, mu, gamma, sigma) -> None:
        """
        Shared helper to store normal distribution parameters.

        Parameters
        ----------
        mu : array_like
            Location vector, shape (d,).
        gamma : array_like
            Skewness vector, shape (d,).
        sigma : array_like
            Covariance scale matrix, shape (d, d).
        """
        mu = np.asarray(mu, dtype=float).flatten()
        gamma = np.asarray(gamma, dtype=float).flatten()
        Sigma = np.asarray(sigma, dtype=float)
        self._d = len(mu)
        self._mu = mu
        self._gamma = gamma
        self._L_Sigma = robust_cholesky(Sigma)

    def _set_from_natural(self, theta: NDArray) -> None:
        """
        Set internal state from natural parameters.

        Extracts and stores :math:`\\mu`, :math:`\\gamma`, and the Cholesky
        factor of :math:`\\Sigma` from the natural parameter vector.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.
        """
        theta = np.asarray(theta, dtype=float)

        # Infer dimension if needed
        if self._d is None:
            n = len(theta)
            disc = 4 + 4 * (n - 3)
            d = int((-2 + np.sqrt(disc)) / 2)
            if 3 + 2 * d + d * d != n:
                raise ValueError(f"Cannot infer d from parameter length {n}")
            self._d = d

        self._validate_natural_params(theta)

        # Extract normal params via Cholesky
        L_Lambda, _, mu, gamma, Sigma = self._extract_normal_params_with_cholesky(
            theta, return_sigma=True
        )
        self._mu = mu
        self._gamma = gamma
        self._L_Sigma = robust_cholesky(Sigma)

        # Store mixing params from theta (subclass-specific)
        self._store_mixing_params_from_theta(theta)

        self._fitted = True
        self._invalidate_cache()

    def _set_internal(self, *, mu, gamma, L_sigma, **mixing_kwargs) -> None:
        """
        EM fast path: set internal state from pre-computed values.

        No Cholesky decomposition is performed -- the caller provides
        the Cholesky factor directly.

        Parameters
        ----------
        mu : array_like
            Location vector, shape (d,).
        gamma : array_like
            Skewness vector, shape (d,).
        L_sigma : ndarray
            Lower Cholesky factor of Sigma, shape (d, d).
        **mixing_kwargs
            Mixing distribution parameters (passed to ``_store_mixing_params``).
        """
        self._mu = np.asarray(mu, dtype=float).flatten()
        self._gamma = np.asarray(gamma, dtype=float).flatten()
        self._L_Sigma = np.asarray(L_sigma, dtype=float)
        self._d = len(self._mu)
        self._store_mixing_params(**mixing_kwargs)
        self._fitted = True
        self._invalidate_cache()

    def _compute_natural_params(self) -> NDArray:
        """
        Build natural parameter vector from named internal attributes.

        Uses ``cho_solve`` and triangular solve on ``_L_Sigma``.

        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
        d = self._d
        theta_5 = cho_solve((self._L_Sigma, True), self._mu)
        theta_4 = cho_solve((self._L_Sigma, True), self._gamma)
        L_inv = solve_triangular(self._L_Sigma, np.eye(d), lower=True)
        Lambda = L_inv.T @ L_inv
        theta_6 = -0.5 * Lambda
        theta_1, theta_2, theta_3 = self._compute_mixing_theta(theta_4, theta_5)
        return np.concatenate([
            [theta_1, theta_2, theta_3],
            theta_4,
            theta_5,
            theta_6.flatten()
        ])

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
        Log joint probability density using Cholesky-based computation.

        Decomposes as :math:`\\log f(x,y) = \\log f(x|y) + \\log f_Y(y)`:

        .. math::
            \\log f(x|y) = -\\frac{d}{2}\\log(2\\pi y)
            - \\frac{1}{2}\\log|\\Sigma|
            - \\frac{1}{2y}(x-\\mu-\\gamma y)^T \\Sigma^{-1}(x-\\mu-\\gamma y)

        Uses ``solve_triangular`` with the stored Cholesky factor
        :math:`L_\\Sigma` instead of constructing the full natural parameter
        vector and sufficient statistics.

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

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = self._d
        mu = self._mu
        gamma = self._gamma
        L = self._L_Sigma
        log_det = self.log_det_Sigma

        # Mixing distribution logpdf
        mixing_dist = self._create_mixing_distribution()

        # Handle single observation
        if y.ndim == 0 or (y.ndim == 1 and len(y) == 1):
            y_scalar = float(y) if y.ndim == 0 else float(y[0])
            if x.ndim == 2 and x.shape[0] == 1:
                x = x[0]

            diff = x - mu - gamma * y_scalar
            z = solve_triangular(L, diff, lower=True)
            mahal = float(z @ z) / y_scalar

            log_normal = (-0.5 * d * np.log(2 * np.pi * y_scalar)
                         - 0.5 * log_det - 0.5 * mahal)
            log_mixing = float(np.atleast_1d(mixing_dist.logpdf(np.atleast_1d(y_scalar)))[0])
            return log_normal + log_mixing

        # Handle multiple observations
        n = len(y)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # diff_i = x_i - mu - gamma * y_i, shape (n, d)
        diff = x - mu - np.outer(y, gamma)

        # z_i = L^{-1} diff_i via solve_triangular, shape (d, n)
        Z = solve_triangular(L, diff.T, lower=True)
        mahal = np.sum(Z ** 2, axis=0) / y  # (n,)

        log_normal = (-0.5 * d * np.log(2 * np.pi * y)
                     - 0.5 * log_det - 0.5 * mahal)
        log_mixing = mixing_dist.logpdf(y)

        return log_normal + log_mixing

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

        mu = self._mu
        gamma = self._gamma
        L_Sigma = self._L_Sigma
        d = self._d

        # Create mixing distribution from named attributes
        mixing_dist = self._create_mixing_distribution()

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

        mixing_dist = self._create_mixing_distribution()
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

        mixing_dist = self._create_mixing_distribution()
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

    _cached_attrs: Tuple[str, ...] = ()

    def __init__(self):
        """Initialize an unfitted marginal normal mixture distribution."""
        super().__init__()
        self._joint: Optional[JointNormalMixture] = None

    def _check_fitted(self) -> None:
        """
        Check if the marginal distribution is fitted.

        A NormalMixture is considered fitted if its underlying joint
        distribution exists and is fitted. This is necessary because
        EM fitting modifies the joint distribution directly, bypassing
        the marginal's setters.
        """
        if self._joint is not None and self._joint._fitted:
            self._fitted = True
            return
        super()._check_fitted()

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
        The returned keys depend on the sufficient statistics of the mixing
        distribution:

        - **GH** (GIG mixing): ``E_Y``, ``E_inv_Y``, ``E_log_Y``
        - **VG** (Gamma mixing): ``E_Y``, ``E_log_Y``
        - **NIG** (InvGauss mixing): ``E_Y``, ``E_inv_Y``
        - **NInvG** (InvGamma mixing): ``E_inv_Y``, ``E_log_Y``

        Parameters
        ----------
        x : array_like
            Observed X values.

        Returns
        -------
        expectations : dict
            Dictionary containing conditional expectations needed for the
            M-step. Always includes at least two of:

            - ``'E_Y'``: :math:`E[Y | X]`
            - ``'E_inv_Y'``: :math:`E[1/Y | X]`
            - ``'E_log_Y'``: :math:`E[\\log Y | X]`
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
        self._fitted = self._joint._fitted
        self._invalidate_cache()
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
        self._fitted = self._joint._fitted
        self._invalidate_cache()
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
        self._fitted = self._joint._fitted
        self._invalidate_cache()
        return self

    # ========================================================================
    # Parameter getters (delegate to joint)
    # ========================================================================

    @property
    def natural_params(self) -> NDArray:
        """
        Natural parameters :math:`\\theta` of the joint distribution.

        Note: The marginal distribution itself is not an exponential family,
        so these are the natural parameters of the joint distribution.

        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
        self._check_fitted()
        return self.joint.natural_params

    @property
    def classical_params(self) -> Dict[str, Any]:
        """
        Classical parameters of the distribution.

        Delegates to the joint distribution's classical parameters.

        Returns
        -------
        params : dict
            Dictionary of classical parameters.
        """
        self._check_fitted()
        return self.joint.classical_params

    @property
    def expectation_params(self) -> NDArray:
        """
        Expectation parameters :math:`\\eta = E[t(X, Y)]` of the joint.

        Returns
        -------
        eta : ndarray
            Expectation parameter vector.
        """
        self._check_fitted()
        return self.joint.expectation_params

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
        self._check_fitted()
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
        return self.joint._create_mixing_distribution()

    # ========================================================================
    # EM helpers
    # ========================================================================

    def _check_convergence(
        self,
        prev_mu: NDArray,
        prev_gamma: NDArray,
        prev_L: NDArray,
        *,
        verbose: int,
        iteration: int,
        X: NDArray,
    ) -> float:
        """
        Compute relative parameter change for EM convergence checking.

        Measures the maximum relative change in the normal parameters
        :math:`(\\mu, \\gamma, L_\\Sigma)` between the current and previous
        EM iteration.  Optionally prints log-likelihood and per-parameter
        diagnostics.

        Parameters
        ----------
        prev_mu : ndarray
            Previous iteration's :math:`\\mu`.
        prev_gamma : ndarray
            Previous iteration's :math:`\\gamma`.
        prev_L : ndarray
            Previous iteration's Cholesky factor :math:`L_\\Sigma`.
        verbose : int
            Verbosity level (0 = silent, 1 = log-likelihood,
            2 = per-parameter relative changes).
        iteration : int
            Current iteration index (0-based).
        X : ndarray
            Data array, shape ``(n_samples, d)``.

        Returns
        -------
        max_rel_change : float
            Maximum of the three relative changes.
        """
        mu_new = self._joint._mu
        gamma_new = self._joint._gamma
        L_new = self._joint._L_Sigma

        rel_mu = np.linalg.norm(mu_new - prev_mu) / max(np.linalg.norm(prev_mu), 1e-10)
        rel_gamma = np.linalg.norm(gamma_new - prev_gamma) / max(np.linalg.norm(prev_gamma), 1e-10)
        rel_sigma = np.linalg.norm(L_new - prev_L, 'fro') / max(np.linalg.norm(prev_L, 'fro'), 1e-10)

        max_rel_change = max(rel_mu, rel_gamma, rel_sigma)

        if verbose >= 1:
            ll = np.mean(self.logpdf(X))
            print(f"Iteration {iteration + 1}: log-likelihood = {ll:.6f}")
            if verbose >= 2:
                print(f"  rel_change: mu={rel_mu:.2e}, gamma={rel_gamma:.2e}, "
                      f"Sigma={rel_sigma:.2e}")

        return max_rel_change

    # ========================================================================
    # Data normalization for EM
    # ========================================================================

    @staticmethod
    def _normalize_data(X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """
        Normalize data columns by median and MAD for numerical stability.

        Computes :math:`X_{\\text{norm}} = \\text{diag}(m)^{-1} (X - c)` where
        :math:`c` is the column-wise median and :math:`m` is the scaled MAD.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Raw data matrix.

        Returns
        -------
        X_norm : ndarray, shape (n, d)
            Normalized data.
        center : ndarray, shape (d,)
            Column medians.
        scale : ndarray, shape (d,)
            Column scaled-MAD values.
        """
        center, scale = column_median_mad(X)
        X_norm = (X - center) / scale
        return X_norm, center, scale

    def _denormalize_params(self, center: NDArray, scale: NDArray) -> None:
        """
        Transform fitted parameters from normalized space back to original.

        If the EM was run on :math:`X_{\\text{norm}} = D^{-1}(X - c)` where
        :math:`D = \\text{diag}(\\text{scale})`, then the original-space
        parameters are:

        .. math::
            \\mu_{\\text{orig}} &= c + D\\,\\mu_{\\text{norm}} \\\\
            \\gamma_{\\text{orig}} &= D\\,\\gamma_{\\text{norm}} \\\\
            L_{\\Sigma,\\text{orig}} &= D\\,L_{\\Sigma,\\text{norm}}

        Mixing distribution parameters are unchanged.

        Parameters
        ----------
        center : ndarray, shape (d,)
            Column medians used for centering.
        scale : ndarray, shape (d,)
            Column scaled-MAD values used for scaling.
        """
        jt = self._joint
        jt._mu = center + scale * jt._mu
        jt._gamma = scale * jt._gamma
        jt._L_Sigma = scale[:, np.newaxis] * jt._L_Sigma
        jt._invalidate_cache()
        self._invalidate_cache()

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
        if not self._fitted:
            return f"{name}(not fitted)"
        try:
            d = self.d
            return f"{name}(d={d})"
        except ValueError:
            return f"{name}(not fitted)"
