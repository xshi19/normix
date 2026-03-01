"""
Joint Generalized Hyperbolic (GH) distribution :math:`f(x, y)`.

The joint distribution of :math:`(X, Y)` where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{GIG}(p, a, b)

where GIG is the Generalized Inverse Gaussian distribution.

This joint distribution belongs to the exponential family.

Sufficient statistics:

.. math::
    t(x, y) = [\\log y, y^{-1}, y, x, x y^{-1}, \\text{vec}(x x^T y^{-1})]

Natural parameters derived from classical :math:`(\\mu, \\gamma, \\Sigma, p, a, b)`:

.. math::
    \\theta_1 &= p - 1 - d/2 \\\\
    \\theta_2 &= -(b + \\frac{1}{2} \\mu^T \\Sigma^{-1} \\mu) \\\\
    \\theta_3 &= -(a + \\frac{1}{2} \\gamma^T \\Sigma^{-1} \\gamma) \\\\
    \\theta_4 &= \\Sigma^{-1} \\gamma \\\\
    \\theta_5 &= \\Sigma^{-1} \\mu \\\\
    \\theta_6 &= -\\frac{1}{2} \\text{vec}(\\Sigma^{-1})

Special cases:
- **Variance Gamma**: :math:`b \\to 0` (GIG :math:`\\to` Gamma)
- **Normal-Inverse Gaussian**: :math:`p = -1/2` (GIG :math:`\\to` InverseGaussian)
- **Normal-Inverse Gamma**: :math:`a \\to 0` (GIG :math:`\\to` InverseGamma)
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from normix.base import JointNormalMixture, ExponentialFamily
from normix.distributions.univariate import GeneralizedInverseGaussian
from normix.params import GHParams
from normix.utils import log_kv, robust_cholesky


class JointGeneralizedHyperbolic(JointNormalMixture):
    """
    Joint Generalized Hyperbolic distribution :math:`f(x, y)`.

    The joint distribution where:

    .. math::
        X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

        Y \\sim \\text{GIG}(p, a, b)

    where GIG is the Generalized Inverse Gaussian distribution with:
    - :math:`p`: shape parameter (real-valued)
    - :math:`a > 0`: rate parameter (coefficient of :math:`y`)
    - :math:`b > 0`: rate parameter (coefficient of :math:`1/y`)

    This is the most general form of the normal variance-mean mixture.

    Parameters
    ----------
    d : int, optional
        Dimension of X. Inferred from parameters if not provided.

    Attributes
    ----------
    _d : int or None
        Dimension of X.
    _p : float or None
        GIG shape parameter.
    _a : float or None
        GIG rate parameter (coefficient of y).
    _b : float or None
        GIG rate parameter (coefficient of 1/y).

    Examples
    --------
    >>> # 1D case
    >>> jgh = JointGeneralizedHyperbolic.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     p=1.0,
    ...     a=1.0,
    ...     b=1.0
    ... )
    >>> X, Y = jgh.rvs(size=1000, random_state=42)

    >>> # 2D case
    >>> jgh = JointGeneralizedHyperbolic.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     p=1.0,
    ...     a=1.0,
    ...     b=1.0
    ... )

    See Also
    --------
    GeneralizedHyperbolic : Marginal distribution (X only)
    GeneralizedInverseGaussian : The subordinator distribution for Y
    JointVarianceGamma : Special case with Gamma subordinator (b → 0)
    JointNormalInverseGaussian : Special case with p = -1/2
    JointNormalInverseGamma : Special case with a → 0

    Notes
    -----
    The GH distribution is widely used in finance for modeling asset returns.
    It includes many important distributions as special cases:

    - **Variance Gamma (VG)**: :math:`b \\to 0, p > 0` (Gamma subordinator)
    - **Normal-Inverse Gaussian (NIG)**: :math:`p = -1/2` (IG subordinator)
    - **Normal-Inverse Gamma (NInvG)**: :math:`a \\to 0, p < 0` (InvGamma subordinator)
    - **Hyperbolic**: :math:`p = 1`
    - **Student-t**: :math:`p = -\\nu/2, a \\to 0, b = \\nu` gives Student-t with ν d.f.
    """

    # ========================================================================
    # Subordinator distribution
    # ========================================================================

    def __init__(self, d=None):
        super().__init__(d)
        self._p: Optional[float] = None
        self._a: Optional[float] = None
        self._b: Optional[float] = None

    @classmethod
    def _get_subordinator_class(cls) -> Type[ExponentialFamily]:
        """Return GeneralizedInverseGaussian as the subordinator class."""
        return GeneralizedInverseGaussian

    # ========================================================================
    # Subordinator parameter management
    # ========================================================================

    def _store_subordinator_params(self, *, p, a, b) -> None:
        self._p = float(p)
        self._a = float(a)
        self._b = float(b)

    def _store_subordinator_params_from_theta(self, theta: NDArray) -> None:
        d = self.d
        theta_1 = theta[0]
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]

        mu_quad = 0.5 * (self._mu @ theta_5)
        gamma_quad = 0.5 * (self._gamma @ theta_4)
        self._p = float(theta_1 + 1 + d / 2)
        self._b = float(-theta_2 - mu_quad)
        self._a = float(-theta_3 - gamma_quad)

    def _compute_subordinator_theta(self, theta_4, theta_5):
        d = self._d
        p = self._p
        a = self._a
        b = self._b
        theta_1 = p - 1 - d / 2
        theta_2 = -(b + 0.5 * (self._mu @ theta_5))
        theta_3 = -(a + 0.5 * (self._gamma @ theta_4))
        return theta_1, theta_2, theta_3

    def _create_subordinator(self):
        return GeneralizedInverseGaussian.from_classical_params(
            p=self._p, a=self._a, b=self._b
        )

    # ========================================================================
    # Natural parameter support
    # ========================================================================

    def _get_natural_param_support(self) -> List[Tuple[float, float]]:
        """
        Get support/bounds for natural parameters.

        Natural parameters:
        - :math:`\\theta_1 = p - 1 - d/2`: unbounded (p can be any real number)
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

        # θ₁: p - 1 - d/2, unbounded (p can be any real number)
        bounds.append((-np.inf, np.inf))

        # θ₂: -(b + 1/2 μ^T Λ μ) < 0 (since b > 0)
        bounds.append((-np.inf, 0.0))

        # θ₃: -(a + 1/2 γ^T Λ γ) < 0 (since a > 0)
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

    def _set_from_classical(self, *, mu, gamma, sigma, p, a, b) -> None:
        if a <= 0:
            raise ValueError(f"Parameter 'a' must be positive, got {a}")
        if b <= 0:
            raise ValueError(f"Parameter 'b' must be positive, got {b}")
        self._store_normal_params(mu=mu, gamma=gamma, sigma=sigma)
        self._store_subordinator_params(p=p, a=a, b=b)
        self._fitted = True
        self._invalidate_cache()

    def _compute_classical_params(self):
        Sigma = self._L_Sigma @ self._L_Sigma.T
        return GHParams(
            mu=self._mu.copy(), gamma=self._gamma.copy(), sigma=Sigma,
            p=self._p, a=self._a, b=self._b
        )

    # ========================================================================
    # Log partition function
    # ========================================================================

    def _log_partition(self, theta: NDArray) -> float:
        """
        Log partition function for Joint Generalized Hyperbolic.

        .. math::
            \\psi(\\theta) = \\frac{1}{2} \\log|\\Sigma| + \\log 2 + \\log K_p(\\sqrt{ab})
            + \\frac{p}{2} \\log\\left(\\frac{b}{a}\\right) + \\mu^T \\Sigma^{-1} \\gamma

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
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]  # Λγ
        theta_5 = theta[3 + d:3 + 2 * d]  # Λμ

        # Get normal params using Cholesky (more efficient for log determinant)
        _, log_det_Lambda, mu, gamma = self._extract_normal_params_with_cholesky(theta)

        # Log determinant of Σ = -log|Λ|
        log_det_Sigma = -log_det_Lambda

        # Recover p, a, b using simplified quadratic forms
        p = theta_1 + 1 + d / 2
        mu_quad = 0.5 * (mu @ theta_5)
        gamma_quad = 0.5 * (gamma @ theta_4)

        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        # Log partition for GIG
        sqrt_ab = np.sqrt(a * b)
        log_K_p = log_kv(p, sqrt_ab)

        # Compute μ^T Λ γ = μ^T θ₄ (since Λγ = θ₄)
        mu_Lambda_gamma = mu @ theta_4

        # Log partition
        psi = (0.5 * log_det_Sigma + np.log(2) + log_K_p +
               0.5 * p * np.log(b / a) + mu_Lambda_gamma)

        return float(psi)

    # ========================================================================
    # Expectation parameters (analytical)
    # ========================================================================

    def _compute_expectation_params(self) -> NDArray:
        """
        Compute expectation parameters directly from internal attributes.

        Avoids the round-trip through natural parameters (which would require
        computing :math:`\\Lambda = \\Sigma^{-1}` and then extracting it back).
        """
        mu = self._mu
        gamma = self._gamma
        Sigma = self._L_Sigma @ self._L_Sigma.T
        p = self._p
        a = self._a
        b = self._b

        sqrt_ab = np.sqrt(a * b)
        sqrt_b_over_a = np.sqrt(b / a)

        log_kv_p_val = log_kv(p, sqrt_ab)
        log_kv_pm1 = log_kv(p - 1, sqrt_ab)
        log_kv_pp1 = log_kv(p + 1, sqrt_ab)

        E_Y = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p_val)
        E_inv_Y = np.exp(log_kv_pm1 - log_kv_p_val) / sqrt_b_over_a

        eps = 1e-6
        d_log_kv_dp = (log_kv(p + eps, sqrt_ab) - log_kv(p - eps, sqrt_ab)) / (2 * eps)
        E_log_Y = d_log_kv_dp + 0.5 * np.log(b / a)

        E_X = mu + gamma * E_Y
        E_X_inv_Y = mu * E_inv_Y + gamma
        E_XXT_inv_Y = (Sigma + np.outer(mu, mu) * E_inv_Y +
                      np.outer(gamma, gamma) * E_Y +
                      np.outer(mu, gamma) + np.outer(gamma, mu))

        return np.concatenate([
            [E_log_Y, E_inv_Y, E_Y],
            E_X, E_X_inv_Y, E_XXT_inv_Y.flatten()
        ])

    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        """
        Convert natural to expectation parameters: :math:`\\eta = E[t(X, Y)]`.

        The expectation parameters are:

        .. math::
            \\eta_1 &= E[\\log Y] \\\\
            \\eta_2 &= E[1/Y] \\\\
            \\eta_3 &= E[Y] \\\\
            \\eta_4 &= E[X] = \\mu + \\gamma E[Y] \\\\
            \\eta_5 &= E[X/Y] = \\mu E[1/Y] + \\gamma \\\\
            \\eta_6 &= E[XX^T/Y] = \\Sigma + \\mu\\mu^T E[1/Y] + \\gamma\\gamma^T E[Y] + \\mu\\gamma^T + \\gamma\\mu^T

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
        theta_1 = theta[0]
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        p = theta_1 + 1 + d / 2
        mu_quad = 0.5 * (mu @ theta_5)
        gamma_quad = 0.5 * (gamma @ theta_4)
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        # GIG expectations using Bessel function ratios
        sqrt_ab = np.sqrt(a * b)
        sqrt_b_over_a = np.sqrt(b / a)

        log_kv_p = log_kv(p, sqrt_ab)
        log_kv_pm1 = log_kv(p - 1, sqrt_ab)
        log_kv_pp1 = log_kv(p + 1, sqrt_ab)

        # E[Y] = √(b/a) · K_{p+1}(√(ab)) / K_p(√(ab))
        E_Y = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)

        # E[1/Y] = √(a/b) · K_{p-1}(√(ab)) / K_p(√(ab))
        E_inv_Y = np.exp(log_kv_pm1 - log_kv_p) / sqrt_b_over_a

        # E[log Y] = ∂/∂p log(K_p(√(ab))) + (1/2) log(b/a)
        eps = 1e-6
        d_log_kv_dp = (log_kv(p + eps, sqrt_ab) - log_kv(p - eps, sqrt_ab)) / (2 * eps)
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

    def _expectation_to_natural(
        self, 
        eta: NDArray, 
        theta0: Optional[Union[NDArray, List[NDArray]]] = None
    ) -> NDArray:
        """
        Convert expectation parameters to natural parameters.

        For Joint GH, the expectation parameters are:

        .. math::
            \\eta = [E[\\log Y], E[1/Y], E[Y], E[X], E[X/Y], \\text{vec}(E[XX^T/Y])]

        The M-step formulas from the EM algorithm give closed-form solutions
        for the normal parameters :math:`(\\mu, \\gamma, \\Sigma)`, and numerical
        optimization for the GIG parameters :math:`(p, a, b)`.

        Parameters
        ----------
        eta : ndarray
            Expectation parameter vector.
        theta0 : ndarray or list of ndarray, optional
            Initial guess(es) for natural parameters. If provided, used to
            extract initial GIG parameters for optimization.

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
        # From em_algorithm.rst equations
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
        # M-step for GIG parameters
        # Use GIG's expectation-to-natural conversion since the first three
        # expectation parameters of JointGH are exactly the GIG expectations:
        #   η_GIG = [E[log Y], E[1/Y], E[Y]]
        # ================================================================

        gig = GeneralizedInverseGaussian()
        gig_eta = np.array([E_log_Y, E_inv_Y, E_Y])
        
        # Extract GIG initial parameters from theta0 if provided
        gig_theta0 = None
        if theta0 is not None:
            theta0_arr = np.asarray(theta0[0] if isinstance(theta0, list) else theta0)
            if len(theta0_arr) >= 3:
                # First 3 components are GIG natural parameters
                gig_theta0 = theta0_arr[:3]

        try:
            gig.set_expectation_params(gig_eta, theta0=gig_theta0)
            gig_classical = gig.classical_params
            p = gig_classical.p
            a = gig_classical.a
            b = gig_classical.b
        except Exception:
            # Fallback to heuristic if optimization fails
            # Try to guess from moment relationships
            # E[Y]/E[1/Y] ≈ b/a for moderate √(ab)
            ratio = E_Y / E_inv_Y if E_inv_Y > 0 else 1.0
            sqrt_b_over_a = np.sqrt(max(ratio, 0.01))

            # Use variance relationship for another equation
            # Var[Y] ≈ E[Y]^2 / (2p) for large p
            # Start with p = 1.0 as default
            p = 1.0
            sqrt_ab = 1.0
            a = sqrt_ab / sqrt_b_over_a
            b = sqrt_ab * sqrt_b_over_a

        # Bound parameters
        a = max(a, 1e-6)
        b = max(b, 1e-6)

        # ================================================================
        # Convert to natural parameters
        # ================================================================
        self._set_from_classical(
            mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b
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
    ) -> 'JointGeneralizedHyperbolic':
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
            Observed Y data (subordinator variable), shape (n_samples,).
        **kwargs
            Additional fitting parameters (currently unused).

        Returns
        -------
        self : JointGeneralizedHyperbolic
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
                return f"JointGeneralizedHyperbolic(d={self._d}, not fitted)"
            return "JointGeneralizedHyperbolic(not fitted)"

        classical = self.classical_params
        d = self.d
        p = classical['p']
        a = classical['a']
        b = classical['b']

        if d == 1:
            mu = float(classical['mu'][0])
            gamma_val = float(classical['gamma'][0])
            return f"JointGeneralizedHyperbolic(μ={mu:.3f}, γ={gamma_val:.3f}, p={p:.3f}, a={a:.3f}, b={b:.3f})"
        else:
            return f"JointGeneralizedHyperbolic(d={d}, p={p:.3f}, a={a:.3f}, b={b:.3f})"
