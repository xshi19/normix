"""
Joint Normal Inverse Gaussian (NIG) distribution :math:`f(x, y)`.

The joint distribution of :math:`(X, Y)` where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{InvGauss}(\\delta, \\eta)

where InvGauss has mean :math:`\\delta` and shape :math:`\\eta`.

This joint distribution belongs to the exponential family. It is a special case
of the Joint Generalized Hyperbolic distribution with GIG parameter :math:`p = -1/2`.

Sufficient statistics:

.. math::
    t(x, y) = [\\log y, y^{-1}, y, x, x y^{-1}, \\text{vec}(x x^T y^{-1})]

Natural parameters derived from classical :math:`(\\mu, \\gamma, \\Sigma, \\delta, \\eta)`:

.. math::
    \\theta_1 &= -3/2 - d/2 \\quad (\\text{since } p = -1/2)\\\\
    \\theta_2 &= -(b + \\frac{1}{2} \\mu^T \\Sigma^{-1} \\mu) \\quad \\text{where } b = \\eta \\\\
    \\theta_3 &= -(a + \\frac{1}{2} \\gamma^T \\Sigma^{-1} \\gamma) \\quad \\text{where } a = \\eta/\\delta^2 \\\\
    \\theta_4 &= \\Sigma^{-1} \\gamma \\\\
    \\theta_5 &= \\Sigma^{-1} \\mu \\\\
    \\theta_6 &= -\\frac{1}{2} \\text{vec}(\\Sigma^{-1})

The GIG parameters for the mixing distribution are:
- :math:`p = -1/2` (Inverse Gaussian special case)
- :math:`a = \\eta / \\delta^2` (coefficient of :math:`y` in exponent)
- :math:`b = \\eta` (coefficient of :math:`1/y` in exponent)
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, List, Optional, Tuple, Type

from normix.base import JointNormalMixture, ExponentialFamily
from normix.distributions.univariate import InverseGaussian
from normix.params import NormalInverseGaussianParams
from normix.utils import log_kv, robust_cholesky


class JointNormalInverseGaussian(JointNormalMixture):
    """
    Joint Normal Inverse Gaussian (NIG) distribution :math:`f(x, y)`.

    The joint distribution where:

    .. math::
        X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

        Y \\sim \\text{InvGauss}(\\delta, \\eta)

    where the Inverse Gaussian has mean :math:`\\delta` and shape :math:`\\eta`.

    This is a special case of the Joint Generalized Hyperbolic distribution
    with GIG parameter :math:`p = -1/2`.

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
    >>> jnig = JointNormalInverseGaussian.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     delta=1.0,
    ...     eta=1.0
    ... )
    >>> X, Y = jnig.rvs(size=1000, random_state=42)

    >>> # 2D case
    >>> jnig = JointNormalInverseGaussian.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     delta=1.0,
    ...     eta=1.0
    ... )

    See Also
    --------
    NormalInverseGaussian : Marginal distribution (X only)
    InverseGaussian : The mixing distribution for Y
    JointGeneralizedHyperbolic : General case with GIG mixing

    Notes
    -----
    The NIG distribution is widely used in finance due to its semi-heavy tails
    and ability to model skewness. It provides a good fit for asset returns
    that exhibit non-Gaussian behavior.

    The Inverse Gaussian mixing distribution has:
    - Mean: :math:`E[Y] = \\delta`
    - Variance: :math:`\\text{Var}[Y] = \\delta^3 / \\eta`
    """

    # ========================================================================
    # Mixing distribution
    # ========================================================================

    def __init__(self, d=None):
        super().__init__(d)
        self._delta: Optional[float] = None
        self._eta: Optional[float] = None

    @classmethod
    def _get_mixing_distribution_class(cls) -> Type[ExponentialFamily]:
        """Return InverseGaussian as the mixing distribution class."""
        return InverseGaussian

    # ========================================================================
    # Mixing parameter management
    # ========================================================================

    def _store_mixing_params(self, *, delta, eta) -> None:
        self._delta = float(delta)
        self._eta = float(eta)

    def _store_mixing_params_from_theta(self, theta: NDArray) -> None:
        d = self.d
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]

        mu_quad = 0.5 * (self._mu @ theta_5)
        gamma_quad = 0.5 * (self._gamma @ theta_4)
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        if a > 1e-12 and b > 1e-12:
            self._eta = float(b)
            self._delta = float(np.sqrt(b / a))
        else:
            self._delta = 1.0
            self._eta = max(float(b), 1e-6)

    def _compute_mixing_theta(self, theta_4, theta_5):
        d = self._d
        delta = self._delta
        eta = self._eta
        a = eta / (delta ** 2)
        b = eta
        p = -0.5
        theta_1 = p - 1 - d / 2
        theta_2 = -(b + 0.5 * (self._mu @ theta_5))
        theta_3 = -(a + 0.5 * (self._gamma @ theta_4))
        return theta_1, theta_2, theta_3

    def _create_mixing_distribution(self):
        return InverseGaussian.from_classical_params(
            mean=self._delta, shape=self._eta
        )

    # ========================================================================
    # Natural parameter support
    # ========================================================================

    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Get support/bounds for natural parameters.

        For NIG with p = -1/2:

        - :math:`\\theta_1 = -3/2 - d/2`: fixed value
        - :math:`\\theta_2 = -(b + \\frac{1}{2} \\mu^T \\Lambda \\mu) < 0`
        - :math:`\\theta_3 = -(a + \\frac{1}{2} \\gamma^T \\Lambda \\gamma) < 0`
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

        # θ₁: -3/2 - d/2 (fixed for NIG)
        theta_1_val = -1.5 - d / 2
        bounds.append((theta_1_val - 0.1, theta_1_val + 0.1))

        # θ₂: -(b + 1/2 μ^T Λ μ) < 0
        bounds.append((-np.inf, 0.0))

        # θ₃: -(a + 1/2 γ^T Λ γ) < 0
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
    # Parameter setters / getters
    # ========================================================================

    def _set_from_classical(self, *, mu, gamma, sigma, delta, eta) -> None:
        if delta <= 0:
            raise ValueError(f"Delta must be positive, got {delta}")
        if eta <= 0:
            raise ValueError(f"Eta must be positive, got {eta}")
        self._store_normal_params(mu=mu, gamma=gamma, sigma=sigma)
        self._store_mixing_params(delta=delta, eta=eta)
        self._fitted = True
        self._invalidate_cache()

    def _compute_classical_params(self):
        Sigma = self._L_Sigma @ self._L_Sigma.T
        return NormalInverseGaussianParams(
            mu=self._mu.copy(), gamma=self._gamma.copy(), sigma=Sigma,
            delta=self._delta, eta=self._eta
        )

    # ========================================================================
    # Log partition function
    # ========================================================================

    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function for Joint Normal Inverse Gaussian.

        .. math::
            \\psi(\\theta) = \\frac{1}{2} \\log|\\Sigma| + \\log 2 + \\log K_{-1/2}(\\eta)
            + \\frac{-1/2}{2} \\log\\left(\\frac{\\eta\\delta}{\\eta/\\delta}\\right) + \\mu^T \\Sigma^{-1} \\gamma

        For :math:`p = -1/2`, :math:`K_{-1/2}(z) = \\sqrt{\\pi/(2z)} e^{-z}`.

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
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]  # Λγ
        theta_5 = theta[3 + d:3 + 2 * d]  # Λμ

        # Get normal params using Cholesky (more efficient for log determinant)
        _, log_det_Lambda, mu, gamma = self._extract_normal_params_with_cholesky(theta)

        # Log determinant of Σ = -log|Λ|
        log_det_Sigma = -log_det_Lambda

        # Recover a and b using simplified quadratic forms
        mu_quad = 0.5 * (mu @ theta_5)
        gamma_quad = 0.5 * (gamma @ theta_4)
        
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        # δ and η from a = η/δ², b = η
        # η = b, δ = √(b/a)
        if a > 1e-12 and b > 1e-12:
            eta = b
            delta = np.sqrt(b / a)
        else:
            delta = 1.0
            eta = 1.0

        # p = -1/2 for NIG
        p = -0.5

        # Log partition for GIG with p = -1/2
        # The argument to K_p is √(ab) = √(η * η/δ²) = η/δ
        sqrt_ab = np.sqrt(a * b)
        # log K_{-1/2}(z) = log(√(π/(2z))) - z = 0.5*log(π/(2z)) - z
        log_K = 0.5 * np.log(np.pi / (2 * sqrt_ab)) - sqrt_ab

        # Compute μ^T Λ γ = μ^T θ₄ (since Λγ = θ₄)
        mu_Lambda_gamma = mu @ theta_4

        # Log partition
        # psi = 0.5 * log|Σ| + log(2) + log K_p(η) + (p/2) log(b/a) + μ^T Λ γ
        psi = (0.5 * log_det_Sigma + np.log(2) + log_K + 
               0.5 * p * np.log(b / a) + mu_Lambda_gamma)

        return float(psi)

    # ========================================================================
    # Expectation parameters (analytical)
    # ========================================================================

    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Convert natural to expectation parameters: :math:`\\eta = E[t(X, Y)]`.

        For NIG with Inverse Gaussian mixing (mean δ, shape η):

        .. math::
            \\eta_1 &= E[\\log Y] \\\\
            \\eta_2 &= E[1/Y] = 1/\\delta + 1/\\eta \\\\
            \\eta_3 &= E[Y] = \\delta \\\\
            \\eta_4 &= E[X] = \\mu + \\gamma \\delta \\\\
            \\eta_5 &= E[X/Y] = \\mu E[1/Y] + \\gamma \\\\
            \\eta_6 &= E[XX^T/Y] = \\Sigma + \\mu\\mu^T E[1/Y] + \\gamma\\gamma^T \\delta + ...

        Parameters
        ----------
        theta : ndarray
            Natural parameter vector.

        Returns
        -------
        eta : ndarray
            Expectation parameter vector.
        """
        d = self.d
        L_Lambda, _, mu, gamma, Sigma = self._extract_normal_params_with_cholesky(
            theta, return_sigma=True
        )
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        mu_quad = 0.5 * (mu @ theta_5)
        gamma_quad = 0.5 * (gamma @ theta_4)
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad
        if a > 1e-12 and b > 1e-12:
            eta_param = b
            delta = np.sqrt(b / a)
        else:
            delta = 1.0
            eta_param = max(b, 1e-6)

        # Inverse Gaussian expectations
        # E[Y] = δ (mean)
        E_Y = delta

        # E[1/Y] = 1/δ + 1/η
        E_inv_Y = 1.0 / delta + 1.0 / eta_param

        # E[log Y] - need numerical approximation or use GIG formula
        # For IG: compute numerically using the derivative of log K
        # E[log Y] = ∂/∂p log(K_p(√(ab))) |_{p=-1/2} + (1/2) log(b/a)
        # where a = η/δ², b = η
        a = eta_param / (delta ** 2)
        b = eta_param
        sqrt_ab = np.sqrt(a * b)  # = η/δ
        
        # Numerical derivative of log K_p at p = -1/2
        p = -0.5
        eps = 1e-6
        log_kv_p_plus = log_kv(p + eps, sqrt_ab)
        log_kv_p_minus = log_kv(p - eps, sqrt_ab)
        d_log_kv_dp = (log_kv_p_plus - log_kv_p_minus) / (2 * eps)
        
        E_log_Y = d_log_kv_dp + 0.5 * np.log(b / a)

        # E[X] = μ + γ E[Y]
        E_X = mu + gamma * E_Y

        # E[X/Y] = μ E[1/Y] + γ
        E_X_inv_Y = mu * E_inv_Y + gamma

        # E[XX^T/Y] = Σ + μμ^T E[1/Y] + γγ^T E[Y] + μγ^T + γμ^T
        E_XXT_inv_Y = (Sigma + np.outer(mu, mu) * E_inv_Y +
                      np.outer(gamma, gamma) * E_Y +
                      np.outer(mu, gamma) + np.outer(gamma, mu))

        # Stack all expectations
        eta = np.concatenate([
            [E_log_Y, E_inv_Y, E_Y],
            E_X,
            E_X_inv_Y,
            E_XXT_inv_Y.flatten()
        ])

        return eta

    # ========================================================================
    # Expectation to Natural conversion (for fitting)
    # ========================================================================

    def _expectation_to_natural(self, eta: NDArray, theta0=None) -> NDArray:
        """
        Convert expectation parameters to natural parameters.

        For Joint NIG, the expectation parameters are:

        .. math::
            \\eta = [E[\\log Y], E[1/Y], E[Y], E[X], E[X/Y], \\text{vec}(E[XX^T/Y])]

        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector.

        Returns
        -------
        theta : ndarray
            Natural parameter vector.
        """
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
        # ================================================================

        # Denominator: 1 - E[1/Y] * E[Y]
        denom = 1.0 - E_inv_Y * E_Y

        # Handle edge case where denom is close to zero
        if abs(denom) < 1e-10:
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

        # Ensure positive definiteness via robust Cholesky
        L = robust_cholesky(Sigma)
        Sigma = L @ L.T

        # ================================================================
        # M-step for Inverse Gaussian parameters
        # For IG: E[Y] = δ, E[1/Y] = 1/δ + 1/η
        # ================================================================

        # δ = E[Y]
        delta = E_Y

        # From E[1/Y] = 1/δ + 1/η, solve for η:
        # η = 1 / (E[1/Y] - 1/δ)
        inv_eta = E_inv_Y - 1.0 / delta
        if inv_eta > 1e-10:
            eta_param = 1.0 / inv_eta
        else:
            # Edge case: large η
            eta_param = 1000.0

        # Bound parameters
        delta = max(delta, 1e-6)
        eta_param = max(eta_param, 1e-6)

        # ================================================================
        # Convert to natural parameters
        # ================================================================
        self._set_from_classical(
            mu=mu, gamma=gamma, sigma=Sigma, delta=delta, eta=eta_param
        )
        return self._compute_natural_params()

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
    ) -> 'JointNormalInverseGaussian':
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
        self : JointNormalInverseGaussian
            Fitted distribution (returns self for method chaining).
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
        if not self._fitted:
            if self._d is not None:
                return f"JointNormalInverseGaussian(d={self._d}, not fitted)"
            return "JointNormalInverseGaussian(not fitted)"

        classical = self.classical_params
        d = self.d
        delta = classical['delta']
        eta = classical['eta']

        if d == 1:
            mu = float(classical['mu'][0])
            gamma_val = float(classical['gamma'][0])
            return f"JointNormalInverseGaussian(μ={mu:.3f}, γ={gamma_val:.3f}, δ={delta:.3f}, η={eta:.3f})"
        else:
            return f"JointNormalInverseGaussian(d={d}, δ={delta:.3f}, η={eta:.3f})"
