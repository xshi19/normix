"""
Joint Variance Gamma distribution :math:`f(x, y)`.

The joint distribution of :math:`(X, Y)` where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{Gamma}(\\alpha, \\beta)

This joint distribution belongs to the exponential family.

Sufficient statistics:

.. math::
    t(x, y) = [\\log y, y^{-1}, y, x, x y^{-1}, \\text{vec}(x x^T y^{-1})]

Natural parameters derived from classical :math:`(\\mu, \\gamma, \\Sigma, \\alpha, \\beta)`:

.. math::
    \\theta_1 &= \\alpha - 1 - d/2 \\\\
    \\theta_2 &= -\\frac{1}{2} \\mu^T \\Sigma^{-1} \\mu \\\\
    \\theta_3 &= -(\\beta + \\frac{1}{2} \\gamma^T \\Sigma^{-1} \\gamma) \\\\
    \\theta_4 &= \\Sigma^{-1} \\gamma \\\\
    \\theta_5 &= \\Sigma^{-1} \\mu \\\\
    \\theta_6 &= -\\frac{1}{2} \\text{vec}(\\Sigma^{-1})
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, List, Optional, Tuple, Type

from scipy.linalg import solve as scipy_solve
from scipy.special import gammaln

from normix.base import JointNormalMixture, ExponentialFamily
from normix.distributions.univariate import Gamma


class JointVarianceGamma(JointNormalMixture):
    """
    Joint Variance Gamma distribution :math:`f(x, y)`.

    The joint distribution where:

    .. math::
        X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

        Y \\sim \\text{Gamma}(\\alpha, \\beta)

    This is a special case of the Joint Generalized Hyperbolic distribution
    with GIG parameter :math:`b \\to 0`.

    Parameters
    ----------
    d : int, optional
        Dimension of X. Inferred from parameters if not provided.

    Attributes
    ----------
    _d : int or None
        Dimension of X.
    _natural_params : tuple or None
        Natural parameters stored as tuple for caching.

    Examples
    --------
    >>> # 1D case
    >>> jvg = JointVarianceGamma.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     shape=2.0,
    ...     rate=1.0
    ... )
    >>> X, Y = jvg.rvs(size=1000, random_state=42)

    >>> # 2D case
    >>> jvg = JointVarianceGamma.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     shape=2.0,
    ...     rate=1.0
    ... )

    See Also
    --------
    VarianceGamma : Marginal distribution (X only)
    Gamma : The mixing distribution for Y
    """

    # ========================================================================
    # Mixing distribution
    # ========================================================================

    @classmethod
    def _get_mixing_distribution_class(cls) -> Type[ExponentialFamily]:
        """Return Gamma as the mixing distribution class."""
        return Gamma

    def _get_mixing_natural_params(self, theta: NDArray) -> NDArray:
        """
        Extract Gamma natural parameters from joint natural params.

        Gamma natural params: :math:`[\\alpha - 1, -\\beta]`

        From joint VG:

        - :math:`\\theta_1 = \\alpha - 1 - d/2`, so :math:`\\alpha - 1 = \\theta_1 + d/2`
        - :math:`\\theta_3 = -(\\beta + \\frac{1}{2} \\gamma^T \\Lambda \\gamma)`, need to solve for :math:`\\beta`

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector for joint distribution.

        Returns
        -------
        theta_gamma : ndarray
            Natural parameters :math:`[\\alpha - 1, -\\beta]` for Gamma distribution.
        """
        d = self.d

        # Extract scalar components
        theta_1 = theta[0]  # α - 1 - d/2
        theta_3 = theta[2]  # -(β + 1/2 γ^T Λ γ)
        theta_4 = theta[3:3 + d]  # Λγ

        # Get normal params using helper
        _, _, _, gamma = self._extract_normal_params_from_theta(theta)

        # Recover α and β using simplified quadratic form: γ^T Λ γ = γ^T θ₄
        alpha_minus_1 = theta_1 + d / 2  # α - 1
        gamma_quad = 0.5 * (gamma @ theta_4)
        beta = -theta_3 - gamma_quad

        return np.array([alpha_minus_1, -beta])

    # ========================================================================
    # Natural parameter support
    # ========================================================================

    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Get support/bounds for natural parameters.

        Natural parameters:

        - :math:`\\theta_1 = \\alpha - 1 - d/2`: must have :math:`\\alpha > 0`, so :math:`\\theta_1 > -1 - d/2`
        - :math:`\\theta_2 = -\\frac{1}{2} \\mu^T \\Lambda \\mu \\leq 0` (since :math:`\\Lambda` is positive definite), can be 0 when :math:`\\mu=0`
        - :math:`\\theta_3 = -(\\beta + \\frac{1}{2} \\gamma^T \\Lambda \\gamma) < 0` (since :math:`\\beta > 0` and quadratic :math:`\\geq 0`)
        - :math:`\\theta_4 = \\Lambda\\gamma`: unbounded
        - :math:`\\theta_5 = \\Lambda\\mu`: unbounded
        - :math:`\\theta_6 = -\\frac{1}{2} \\Lambda`: diagonal elements :math:`< 0`

        Returns
        -------
        bounds : list of tuples
            Bounds for each natural parameter component.
        """
        d = self.d
        bounds = []

        # θ₁: α - 1 - d/2, need α > 0 so θ₁ > -1 - d/2
        bounds.append((-1.0 - d / 2, np.inf))

        # θ₂: -1/2 μ^T Λ μ ≤ 0, use small epsilon to allow 0
        bounds.append((-np.inf, 1e-10))

        # θ₃: -(β + 1/2 γ^T Λ γ) < 0
        bounds.append((-np.inf, 0.0))

        # θ₄: Λγ, unbounded (d components)
        for _ in range(d):
            bounds.append((-np.inf, np.inf))

        # θ₅: Λμ, unbounded (d components)
        for _ in range(d):
            bounds.append((-np.inf, np.inf))

        # θ₆: -1/2 Λ, diagonal < 0, off-diagonal unbounded (d² components)
        for i in range(d):
            for j in range(d):
                if i == j:
                    bounds.append((-np.inf, 0.0))
                else:
                    bounds.append((-np.inf, np.inf))

        return bounds

    # ========================================================================
    # Parameter conversions
    # ========================================================================

    def _classical_to_natural(self, **kwargs) -> NDArray:
        """
        Convert classical parameters to natural parameters.

        Parameters
        ----------
        mu : array_like
            Location parameter :math:`\\mu`, shape (d,).
        gamma : array_like
            Skewness parameter :math:`\\gamma`, shape (d,).
        sigma : array_like
            Covariance scale matrix :math:`\\Sigma`, shape (d, d).
        shape : float
            Gamma shape parameter :math:`\\alpha > 0`.
        rate : float
            Gamma rate parameter :math:`\\beta > 0`.

        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
        mu = np.asarray(kwargs['mu']).flatten()
        gamma = np.asarray(kwargs['gamma']).flatten()
        sigma = np.asarray(kwargs['sigma'])
        alpha = kwargs['shape']
        beta = kwargs['rate']

        d = len(mu)
        self._d = d

        # Validate
        if sigma.shape != (d, d):
            raise ValueError(f"sigma shape {sigma.shape} doesn't match mu dimension {d}")
        if alpha <= 0:
            raise ValueError(f"Shape must be positive, got {alpha}")
        if beta <= 0:
            raise ValueError(f"Rate must be positive, got {beta}")

        # Compute precision matrix (Σ is positive definite)
        Lambda = scipy_solve(sigma, np.eye(d), assume_a='pos')

        # Natural parameters
        theta_1 = alpha - 1 - d / 2
        theta_2 = -0.5 * mu @ Lambda @ mu
        theta_3 = -(beta + 0.5 * gamma @ Lambda @ gamma)
        theta_4 = Lambda @ gamma
        theta_5 = Lambda @ mu
        theta_6 = -0.5 * Lambda

        # Flatten and concatenate
        theta = np.concatenate([
            [theta_1, theta_2, theta_3],
            theta_4,
            theta_5,
            theta_6.flatten()
        ])

        return theta

    def _natural_to_classical(self, theta: NDArray) -> Dict[str, Any]:
        """
        Convert natural parameters to classical parameters.

        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.

        Returns
        -------
        params : dict
            Dictionary with keys: mu, gamma, sigma, shape, rate.
        """
        d = self.d

        # Extract scalar components
        theta_1 = theta[0]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]  # Λγ

        # Get normal params using helper (with symmetrization)
        _, Sigma, mu, gamma = self._extract_normal_params_from_theta(
            theta, symmetrize=True
        )

        # Recover α
        alpha = theta_1 + 1 + d / 2

        # Recover β using simplified quadratic form: γ^T Λ γ = γ^T θ₄
        gamma_quad = 0.5 * (gamma @ theta_4)
        beta = -theta_3 - gamma_quad

        return {
            'mu': mu,
            'gamma': gamma,
            'sigma': Sigma,
            'shape': alpha,
            'rate': beta
        }

    # ========================================================================
    # Log partition function
    # ========================================================================

    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function for Joint Variance Gamma.

        .. math::
            \\psi(\\theta) = \\frac{1}{2} \\log|\\Sigma| + \\log\\Gamma(\\alpha)
            - \\alpha \\log\\beta + \\mu^T \\Sigma^{-1} \\gamma

        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.

        Returns
        -------
        psi : float
            Log partition function value.
        """
        d = self.d

        # Extract scalar components
        theta_1 = theta[0]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]  # Λγ

        # Get normal params using Cholesky (more efficient for log determinant)
        _, log_det_Lambda, mu, gamma = self._extract_normal_params_with_cholesky(
            theta, symmetrize=True
        )

        # Log determinant of Σ = -log|Λ|
        log_det_Sigma = -log_det_Lambda

        # Recover α
        alpha = theta_1 + 1 + d / 2

        # Recover β using simplified quadratic form
        gamma_quad = 0.5 * (gamma @ theta_4)
        beta = -theta_3 - gamma_quad

        # Compute μ^T Λ γ = μ^T θ₄ (since Λγ = θ₄)
        mu_Lambda_gamma = mu @ theta_4

        # Log partition
        psi = 0.5 * log_det_Sigma + gammaln(alpha) - alpha * np.log(beta) + mu_Lambda_gamma

        return float(psi)

    # ========================================================================
    # Expectation parameters (analytical)
    # ========================================================================

    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Convert natural to expectation parameters: :math:`\\eta = E[t(X, Y)]`.

        The expectation parameters are:

        .. math::
            \\eta_1 &= E[\\log Y] = \\psi(\\alpha) - \\log\\beta \\\\
            \\eta_2 &= E[1/Y] = \\text{undefined for Gamma} \\\\
            \\eta_3 &= E[Y] = \\alpha / \\beta \\\\
            \\eta_4 &= E[X] = \\mu + \\gamma E[Y] \\\\
            \\eta_5 &= E[X/Y] = \\mu E[1/Y] + \\gamma \\\\
            \\eta_6 &= E[XX^T/Y] = \\Sigma + \\mu\\mu^T E[1/Y] + \\gamma\\gamma^T E[Y] + ...

        Note: :math:`E[1/Y]` for :math:`\\text{Gamma}(\\alpha, \\beta)` equals :math:`\\beta/(\\alpha-1)` and only exists for :math:`\\alpha > 1`.

        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.

        Returns
        -------
        eta : ndarray
            Expectation parameter vector.
        """
        from scipy.special import digamma

        d = self.d
        classical = self._natural_to_classical(theta)
        mu = classical['mu']
        gamma = classical['gamma']
        Sigma = classical['sigma']
        alpha = classical['shape']
        beta = classical['rate']

        # Gamma expectations
        E_log_Y = digamma(alpha) - np.log(beta)
        E_Y = alpha / beta

        # E[1/Y] exists only for α > 1
        if alpha > 1:
            E_inv_Y = beta / (alpha - 1)
        else:
            E_inv_Y = np.inf

        # E[X] = μ + γ E[Y]
        E_X = mu + gamma * E_Y

        # E[X/Y] = μ E[1/Y] + γ
        if np.isfinite(E_inv_Y):
            E_X_inv_Y = mu * E_inv_Y + gamma
        else:
            E_X_inv_Y = np.full(d, np.inf)

        # E[XX^T/Y] = Σ + μμ^T E[1/Y] + γγ^T + μγ^T + γμ^T
        # This comes from E[(μ + γY + √Y Z)(μ + γY + √Y Z)^T / Y]
        if np.isfinite(E_inv_Y):
            E_XXT_inv_Y = (Sigma + np.outer(mu, mu) * E_inv_Y +
                          np.outer(gamma, gamma) * E_Y +
                          np.outer(mu, gamma) + np.outer(gamma, mu))
        else:
            E_XXT_inv_Y = np.full((d, d), np.inf)

        # Stack all expectations
        eta = np.concatenate([
            [E_log_Y, E_inv_Y, E_Y],
            E_X,
            E_X_inv_Y,
            E_XXT_inv_Y.flatten()
        ])

        return eta

    # ========================================================================
    # String representation
    # ========================================================================

    # ========================================================================
    # Expectation to Natural conversion (for fitting)
    # ========================================================================

    def _expectation_to_natural(self, eta: NDArray, theta0=None) -> NDArray:
        """
        Convert expectation parameters to natural parameters.

        For Joint Variance Gamma, the expectation parameters are:

        .. math::
            \\eta = [E[\\log Y], E[1/Y], E[Y], E[X], E[X/Y], \\text{vec}(E[XX^T/Y])]

        The M-step formulas from the EM algorithm give closed-form solutions
        for the normal parameters :math:`(\\mu, \\gamma, \\Sigma)`, and Newton's
        method for the Gamma parameters :math:`(\\alpha, \\beta)`.

        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector.

        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
        from scipy.special import digamma, polygamma

        d = self.d

        # Extract expectation parameters
        E_log_Y = eta[0]
        E_inv_Y = eta[1]
        E_Y = eta[2]
        E_X = eta[3:3 + d]
        E_X_inv_Y = eta[3 + d:3 + 2 * d]
        E_XXT_inv_Y = eta[3 + 2 * d:].reshape(d, d)

        # Symmetrize E[XX^T/Y] for numerical stability
        E_XXT_inv_Y = (E_XXT_inv_Y + E_XXT_inv_Y.T) / 2

        # ================================================================
        # M-step formulas for normal parameters (closed-form)
        # From em_algorithm.rst equations
        # ================================================================

        # Denominator: 1 - E[1/Y] * E[Y]
        denom = 1.0 - E_inv_Y * E_Y

        # Handle edge case where denom is close to zero
        if abs(denom) < 1e-10:
            # Fall back to simpler estimates
            mu = E_X / E_Y if E_Y > 0 else E_X
            gamma = np.zeros(d)
        else:
            # μ = (E[X] - E[Y] * E[X/Y]) / (1 - E[1/Y] * E[Y])
            mu = (E_X - E_Y * E_X_inv_Y) / denom

            # γ = (E[X/Y] - E[1/Y] * E[X]) / (1 - E[1/Y] * E[Y])
            gamma = (E_X_inv_Y - E_inv_Y * E_X) / denom

        # Σ = E[XX^T/Y] - E[X/Y] μ^T - μ E[X/Y]^T + E[1/Y] μ μ^T - E[Y] γ γ^T
        Sigma = (E_XXT_inv_Y
                 - np.outer(E_X_inv_Y, mu)
                 - np.outer(mu, E_X_inv_Y)
                 + E_inv_Y * np.outer(mu, mu)
                 - E_Y * np.outer(gamma, gamma))

        # Symmetrize and ensure positive definiteness
        Sigma = (Sigma + Sigma.T) / 2

        # Add small regularization if needed
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 1e-8:
            Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)

        # ================================================================
        # M-step for Gamma parameters (Newton's method)
        # From em_algorithm.rst: solve ψ(α) - log(α/E[Y]) = E[log Y]
        # ================================================================

        # Initial guess for α
        alpha = max(E_Y ** 2 / max(E_Y - E_inv_Y * E_Y ** 2, 0.1), 1.0)

        # Newton's method to solve: ψ(α) - log(α) = E[log Y] - log(E[Y])
        target = E_log_Y - np.log(E_Y)

        for _ in range(100):
            psi_val = digamma(alpha)
            psi_prime = polygamma(1, alpha)

            # f(α) = ψ(α) - log(α) - target
            f_val = psi_val - np.log(alpha) - target

            # f'(α) = ψ'(α) - 1/α
            f_prime = psi_prime - 1.0 / alpha

            # Newton step with damping
            step = f_val / f_prime
            alpha_new = alpha - step

            # Keep α positive and bounded
            alpha_new = max(alpha_new, 0.1)
            alpha_new = min(alpha_new, 1000.0)

            if abs(alpha_new - alpha) / max(abs(alpha), 1e-10) < 1e-10:
                alpha = alpha_new
                break

            alpha = alpha_new

        # β = α / E[Y]
        beta = alpha / E_Y

        # ================================================================
        # Convert to natural parameters
        # ================================================================
        return self._classical_to_natural(
            mu=mu, gamma=gamma, sigma=Sigma, shape=alpha, rate=beta
        )

    def _get_initial_natural_params(self, eta: NDArray) -> NDArray:
        """
        Get initial guess for natural parameters from expectation parameters.

        Uses the analytical M-step formulas.
        """
        return self._expectation_to_natural(eta)

    def fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        **kwargs
    ) -> 'JointVarianceGamma':
        """
        Fit distribution to complete data (X, Y) using MLE.

        For exponential families, the MLE has a closed form:

        .. math::
            \\hat{\\eta} = \\frac{1}{n} \\sum_{i=1}^n t(x_i, y_i)

        Then converts to natural parameters using :meth:`_expectation_to_natural`.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d) or (n_samples,) for d=1.
        Y : array_like
            Observed Y data (mixing variable), shape (n_samples,).
        **kwargs
            Additional fitting parameters (currently unused).

        Returns
        -------
        self : JointVarianceGamma
            Fitted distribution (returns self for method chaining).

        Examples
        --------
        >>> # Generate data from known distribution
        >>> true_dist = JointVarianceGamma.from_classical_params(
        ...     mu=np.array([0.0]), gamma=np.array([0.5]),
        ...     sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        ... )
        >>> X, Y = true_dist.rvs(size=5000, random_state=42)
        >>> 
        >>> # Fit new distribution
        >>> fitted = JointVarianceGamma(d=1).fit(X, Y)
        >>> print(fitted.get_classical_params())
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Handle 1D X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, d = X.shape
        self._d = d

        # Validate shapes
        if len(Y) != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but Y has {len(Y)} samples"
            )

        # Compute sufficient statistics
        t_xy = self._sufficient_statistics(X, Y)

        # MLE: sample mean of sufficient statistics
        eta_hat = np.mean(t_xy, axis=0)

        # Convert to natural parameters
        self.set_expectation_params(eta_hat)

        return self

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if self._natural_params is None:
            if self._d is not None:
                return f"JointVarianceGamma(d={self._d}, not fitted)"
            return "JointVarianceGamma(not fitted)"

        classical = self.get_classical_params()
        d = self.d
        alpha = classical['shape']
        beta = classical['rate']

        if d == 1:
            mu = float(classical['mu'][0])
            gamma_val = float(classical['gamma'][0])
            return f"JointVarianceGamma(μ={mu:.3f}, γ={gamma_val:.3f}, α={alpha:.3f}, β={beta:.3f})"
        else:
            return f"JointVarianceGamma(d={d}, α={alpha:.3f}, β={beta:.3f})"
