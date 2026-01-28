"""
Joint Normal Inverse Gamma distribution :math:`f(x, y)`.

The joint distribution of :math:`(X, Y)` where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{InvGamma}(\\alpha, \\beta)

This joint distribution belongs to the exponential family.

Sufficient statistics:

.. math::
    t(x, y) = [\\log y, y^{-1}, y, x, x y^{-1}, \\text{vec}(x x^T y^{-1})]

Natural parameters derived from classical :math:`(\\mu, \\gamma, \\Sigma, \\alpha, \\beta)`:

.. math::
    \\theta_1 &= -(\\alpha + 1) - d/2 \\\\
    \\theta_2 &= -(\\beta + \\frac{1}{2} \\mu^T \\Sigma^{-1} \\mu) \\\\
    \\theta_3 &= -\\frac{1}{2} \\gamma^T \\Sigma^{-1} \\gamma \\\\
    \\theta_4 &= \\Sigma^{-1} \\gamma \\\\
    \\theta_5 &= \\Sigma^{-1} \\mu \\\\
    \\theta_6 &= -\\frac{1}{2} \\text{vec}(\\Sigma^{-1})

This is a special case of the Joint Generalized Hyperbolic distribution
with GIG parameter :math:`a \\to 0` (inverse gamma mixing).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, List, Optional, Tuple, Type

from scipy.special import gammaln, digamma, polygamma

from normix.base import JointNormalMixture, ExponentialFamily
from normix.distributions.univariate import InverseGamma


class JointNormalInverseGamma(JointNormalMixture):
    """
    Joint Normal Inverse Gamma distribution :math:`f(x, y)`.

    The joint distribution where:

    .. math::
        X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

        Y \\sim \\text{InvGamma}(\\alpha, \\beta)

    This is a special case of the Joint Generalized Hyperbolic distribution
    with GIG parameter :math:`a \\to 0`.

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
    >>> jnig = JointNormalInverseGamma.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     shape=3.0,
    ...     rate=1.0
    ... )
    >>> X, Y = jnig.rvs(size=1000, random_state=42)

    >>> # 2D case
    >>> jnig = JointNormalInverseGamma.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     shape=3.0,
    ...     rate=1.0
    ... )

    See Also
    --------
    NormalInverseGamma : Marginal distribution (X only)
    InverseGamma : The mixing distribution for Y

    Notes
    -----
    For the mean and variance to exist, we need :math:`\\alpha > 1` and
    :math:`\\alpha > 2` respectively.
    """

    # ========================================================================
    # Mixing distribution
    # ========================================================================

    @classmethod
    def _get_mixing_distribution_class(cls) -> Type[ExponentialFamily]:
        """Return InverseGamma as the mixing distribution class."""
        return InverseGamma

    def _get_mixing_natural_params(self, theta: NDArray) -> NDArray:
        """
        Extract InverseGamma natural parameters from joint natural params.

        InverseGamma natural params: :math:`[\\beta, -(\\alpha + 1)]`

        From joint Normal-Inverse Gamma:

        - :math:`\\theta_1 = -(\\alpha + 1) - d/2`, so :math:`-(\\alpha + 1) = \\theta_1 + d/2`
        - :math:`\\theta_2 = -(\\beta + \\frac{1}{2} \\mu^T \\Lambda \\mu)`, need to solve for :math:`\\beta`

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector for joint distribution.

        Returns
        -------
        theta_invgamma : ndarray
            Natural parameters :math:`[\\beta, -(\\alpha + 1)]` for InverseGamma distribution.
        """
        d = self.d

        # Extract components
        theta_1 = theta[0]  # -(α + 1) - d/2
        theta_2 = theta[1]  # -(β + 1/2 μ^T Λ μ)
        theta_5 = theta[3 + d:3 + 2 * d]  # Λμ
        theta_6 = theta[3 + 2 * d:].reshape(d, d)  # -1/2 Λ

        # Recover Λ and Σ
        Lambda = -2 * theta_6
        Sigma = np.linalg.inv(Lambda)

        # Recover μ
        mu = Sigma @ theta_5

        # Recover α and β
        neg_alpha_plus_1 = theta_1 + d / 2  # -(α + 1)
        mu_quad = 0.5 * mu @ Lambda @ mu
        beta = -theta_2 - mu_quad

        return np.array([beta, neg_alpha_plus_1])

    def _get_normal_params(self, theta: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Extract normal distribution parameters :math:`(\\mu, \\gamma, \\Sigma)` from joint natural params.

        Parameters
        ----------
        theta : ndarray
            Full natural parameter vector.

        Returns
        -------
        mu : ndarray
            Location parameter :math:`\\mu`, shape (d,).
        gamma : ndarray
            Skewness parameter :math:`\\gamma`, shape (d,).
        Sigma : ndarray
            Covariance scale matrix :math:`\\Sigma`, shape (d, d).
        """
        d = self.d

        theta_4 = theta[3:3 + d]  # Λγ
        theta_5 = theta[3 + d:3 + 2 * d]  # Λμ
        theta_6 = theta[3 + 2 * d:].reshape(d, d)  # -1/2 Λ

        # Recover Λ and Σ
        Lambda = -2 * theta_6
        Sigma = np.linalg.inv(Lambda)

        # Recover μ and γ
        mu = Sigma @ theta_5
        gamma = Sigma @ theta_4

        return mu, gamma, Sigma

    # ========================================================================
    # Natural parameter support
    # ========================================================================

    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Get support/bounds for natural parameters.

        Natural parameters:

        - :math:`\\theta_1 = -(\\alpha + 1) - d/2`: must have :math:`\\alpha > 0`, so :math:`\\theta_1 < -1 - d/2`
        - :math:`\\theta_2 = -(\\beta + \\frac{1}{2} \\mu^T \\Lambda \\mu) < 0` (since :math:`\\beta > 0`)
        - :math:`\\theta_3 = -\\frac{1}{2} \\gamma^T \\Lambda \\gamma \\leq 0`
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

        # θ₁: -(α + 1) - d/2, need α > 0 so θ₁ < -1 - d/2
        bounds.append((-np.inf, -1.0 - d / 2))

        # θ₂: -(β + 1/2 μ^T Λ μ) < 0
        bounds.append((-np.inf, 0.0))

        # θ₃: -1/2 γ^T Λ γ ≤ 0, allow small positive for numerical reasons
        bounds.append((-np.inf, 1e-10))

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
            InverseGamma shape parameter :math:`\\alpha > 0`.
        rate : float
            InverseGamma rate parameter :math:`\\beta > 0`.

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

        # Compute precision matrix
        Lambda = np.linalg.inv(sigma)

        # Natural parameters
        # For InvGamma mixing: coefficient of log(y) is -(α+1) from f(y) 
        # and -d/2 from the normal, so θ₁ = -(α+1) - d/2
        theta_1 = -(alpha + 1) - d / 2
        
        # Coefficient of 1/y: -β from InvGamma and -1/2 μ^T Λ μ from normal
        theta_2 = -(beta + 0.5 * mu @ Lambda @ mu)
        
        # Coefficient of y: -1/2 γ^T Λ γ from normal
        theta_3 = -0.5 * gamma @ Lambda @ gamma
        
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

        # Extract components
        theta_1 = theta[0]
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        theta_6 = theta[3 + 2 * d:].reshape(d, d)

        # Recover Λ and Σ
        Lambda = -2 * theta_6
        # Symmetrize for numerical stability
        Lambda = (Lambda + Lambda.T) / 2
        Sigma = np.linalg.inv(Lambda)

        # Recover μ and γ
        mu = Sigma @ theta_5
        gamma = Sigma @ theta_4

        # Recover α: θ₁ = -(α+1) - d/2, so α = -(θ₁ + d/2) - 1
        alpha = -(theta_1 + d / 2) - 1

        # Recover β: θ₂ = -(β + 1/2 μ^T Λ μ)
        mu_quad = 0.5 * mu @ Lambda @ mu
        beta = -theta_2 - mu_quad

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
        Log partition function for Joint Normal Inverse Gamma.

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

        # Extract and recover classical parameters
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        theta_6 = theta[3 + 2 * d:].reshape(d, d)

        # Recover Λ and Σ
        Lambda = -2 * theta_6
        Lambda = (Lambda + Lambda.T) / 2

        # Log determinant of Σ = -log|Λ|
        _, logdet_Lambda = np.linalg.slogdet(Lambda)
        log_det_Sigma = -logdet_Lambda

        # Recover α and β from theta
        theta_1 = theta[0]
        theta_2 = theta[1]

        alpha = -(theta_1 + d / 2) - 1

        # Recover Σ for computing β
        Sigma = np.linalg.inv(Lambda)
        mu = Sigma @ theta_5

        mu_quad = 0.5 * mu @ Lambda @ mu
        beta = -theta_2 - mu_quad

        # Compute μ^T Λ γ = θ₅^T Σ θ₄
        mu_Lambda_gamma = theta_5 @ Sigma @ theta_4

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
            \\eta_1 &= E[\\log Y] = \\log\\beta - \\psi(\\alpha) \\\\
            \\eta_2 &= E[1/Y] = \\alpha / \\beta \\\\
            \\eta_3 &= E[Y] = \\beta / (\\alpha - 1) \\text{ for } \\alpha > 1 \\\\
            \\eta_4 &= E[X] = \\mu + \\gamma E[Y] \\\\
            \\eta_5 &= E[X/Y] = \\mu E[1/Y] + \\gamma \\\\
            \\eta_6 &= E[XX^T/Y] = \\Sigma + \\mu\\mu^T E[1/Y] + \\gamma\\gamma^T E[Y] + ...

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
        classical = self._natural_to_classical(theta)
        mu = classical['mu']
        gamma = classical['gamma']
        Sigma = classical['sigma']
        alpha = classical['shape']
        beta = classical['rate']

        # InverseGamma expectations
        # E[log Y] = log(β) - ψ(α)
        E_log_Y = np.log(beta) - digamma(alpha)
        
        # E[1/Y] = α/β
        E_inv_Y = alpha / beta

        # E[Y] = β/(α-1) for α > 1
        if alpha > 1:
            E_Y = beta / (alpha - 1)
        else:
            E_Y = np.inf

        # E[X] = μ + γ E[Y]
        if np.isfinite(E_Y):
            E_X = mu + gamma * E_Y
        else:
            E_X = np.full(d, np.inf)

        # E[X/Y] = μ E[1/Y] + γ
        E_X_inv_Y = mu * E_inv_Y + gamma

        # E[XX^T/Y] = Σ + μμ^T E[1/Y] + γγ^T E[Y] + μγ^T + γμ^T
        if np.isfinite(E_Y):
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
    # Expectation to Natural conversion (for fitting)
    # ========================================================================

    def _expectation_to_natural(self, eta: NDArray) -> NDArray:
        """
        Convert expectation parameters to natural parameters.

        For Joint Normal Inverse Gamma, the expectation parameters are:

        .. math::
            \\eta = [E[\\log Y], E[1/Y], E[Y], E[X], E[X/Y], \\text{vec}(E[XX^T/Y])]

        The M-step formulas from the EM algorithm give closed-form solutions
        for the normal parameters :math:`(\\mu, \\gamma, \\Sigma)`, and Newton's
        method for the InverseGamma parameters :math:`(\\alpha, \\beta)`.

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
        # Same as Variance Gamma
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

        # Symmetrize and ensure positive definiteness
        Sigma = (Sigma + Sigma.T) / 2

        # Add small regularization if needed
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 1e-8:
            Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)

        # ================================================================
        # M-step for InverseGamma parameters (Newton's method)
        # For InvGamma: E[1/Y] = α/β, E[log Y] = log(β) - ψ(α)
        # ================================================================

        # From E[1/Y] = α/β and E[Y] = β/(α-1), we have:
        # β = α / E[1/Y]
        # And: E[log Y] = log(β) - ψ(α) = log(α/E[1/Y]) - ψ(α)
        # So: ψ(α) - log(α) = -E[log Y] - log(E[1/Y])

        target = -E_log_Y - np.log(E_inv_Y)

        # Initial guess for α using method of moments
        # From E[Y] = β/(α-1) and E[1/Y] = α/β
        # E[Y] * E[1/Y] = α/(α-1), so α = E[Y]*E[1/Y] / (E[Y]*E[1/Y] - 1)
        ey_einvy = E_Y * E_inv_Y
        if ey_einvy > 1.001:  # Need ey_einvy > 1 for valid α > 0
            alpha = ey_einvy / (ey_einvy - 1)
        else:
            alpha = 3.0  # Default

        # Bound alpha
        alpha = max(alpha, 1.5)
        alpha = min(alpha, 100.0)

        # Newton's method to solve: ψ(α) - log(α) = target
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

            # Keep α positive and bounded (need α > 1 for finite mean)
            alpha_new = max(alpha_new, 1.5)
            alpha_new = min(alpha_new, 1000.0)

            if abs(alpha_new - alpha) / max(abs(alpha), 1e-10) < 1e-10:
                alpha = alpha_new
                break

            alpha = alpha_new

        # β = α / E[1/Y]
        beta = alpha / E_inv_Y

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
    ) -> 'JointNormalInverseGamma':
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
        self : JointNormalInverseGamma
            Fitted distribution (returns self for method chaining).

        Examples
        --------
        >>> # Generate data from known distribution
        >>> true_dist = JointNormalInverseGamma.from_classical_params(
        ...     mu=np.array([0.0]), gamma=np.array([0.5]),
        ...     sigma=np.array([[1.0]]), shape=3.0, rate=1.0
        ... )
        >>> X, Y = true_dist.rvs(size=5000, random_state=42)
        >>> 
        >>> # Fit new distribution
        >>> fitted = JointNormalInverseGamma(d=1).fit(X, Y)
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
                return f"JointNormalInverseGamma(d={self._d}, not fitted)"
            return "JointNormalInverseGamma(not fitted)"

        classical = self.get_classical_params()
        d = self.d
        alpha = classical['shape']
        beta = classical['rate']

        if d == 1:
            mu = float(classical['mu'][0])
            gamma_val = float(classical['gamma'][0])
            return f"JointNormalInverseGamma(μ={mu:.3f}, γ={gamma_val:.3f}, α={alpha:.3f}, β={beta:.3f})"
        else:
            return f"JointNormalInverseGamma(d={d}, α={alpha:.3f}, β={beta:.3f})"
