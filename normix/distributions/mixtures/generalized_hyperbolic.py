"""
Generalized Hyperbolic (GH) marginal distribution :math:`f(x)`.

The marginal distribution of :math:`X` obtained by integrating out :math:`Y`:

.. math::
    f(x) = \\int_0^\\infty f(x, y) \\, dy = \\int_0^\\infty f(x | y) f(y) \\, dy

where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{GIG}(p, a, b)

The marginal PDF has a closed form involving modified Bessel functions.

Note: The marginal distribution is NOT an exponential family, but the joint
distribution :math:`f(x, y)` IS an exponential family (accessible via ``.joint``).

Special cases:
- **Variance Gamma (VG)**: :math:`b \\to 0` (GIG → Gamma)
- **Normal-Inverse Gaussian (NIG)**: :math:`p = -1/2` (GIG → InverseGaussian)  
- **Normal-Inverse Gamma (NInvG)**: :math:`a \\to 0` (GIG → InverseGamma)
- **Hyperbolic**: :math:`p = 1`
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Callable, Dict, Optional, Tuple, Union

from normix.base import NormalMixture, JointNormalMixture
from normix.utils import log_kv
from .joint_generalized_hyperbolic import JointGeneralizedHyperbolic


# ============================================================================
# Parameter Regularization Methods
# ============================================================================

def regularize_det_sigma_one(
    mu: NDArray,
    gamma: NDArray,
    sigma: NDArray,
    p: float,
    a: float,
    b: float,
    d: int
) -> Dict[str, Any]:
    """
    Regularize GH parameters by enforcing :math:`|\\Sigma| = 1`.

    This is the recommended regularization that doesn't affect convergence
    of the EM algorithm. From em_algorithm.rst:

    .. math::
        (\\mu, \\gamma, \\Sigma, p, a, b) \\to
        (\\mu, |\\Sigma|^{-1/d} \\gamma, |\\Sigma|^{-1/d} \\Sigma,
        p, |\\Sigma|^{1/d} a, |\\Sigma|^{-1/d} b)

    Parameters
    ----------
    mu, gamma, sigma, p, a, b : parameters
        Classical GH parameters.
    d : int
        Dimension of X.

    Returns
    -------
    params : dict
        Regularized parameters with |Σ| = 1.
    """
    det_sigma = np.linalg.det(sigma)
    if det_sigma <= 0:
        # Sigma is degenerate, can't regularize
        return {'mu': mu, 'gamma': gamma, 'sigma': sigma, 'p': p, 'a': a, 'b': b}

    scale = det_sigma ** (1.0 / d)

    return {
        'mu': mu,
        'gamma': gamma / scale,
        'sigma': sigma / scale,
        'p': p,
        'a': a * scale,
        'b': b / scale
    }


def regularize_sigma_diagonal_one(
    mu: NDArray,
    gamma: NDArray,
    sigma: NDArray,
    p: float,
    a: float,
    b: float,
    d: int
) -> Dict[str, Any]:
    """
    Regularize by enforcing the first diagonal element of :math:`\\Sigma` to be 1.

    .. math::
        \\Sigma_{11} = 1

    This is an alternative regularization useful in some applications.

    Parameters
    ----------
    mu, gamma, sigma, p, a, b : parameters
        Classical GH parameters.
    d : int
        Dimension of X.

    Returns
    -------
    params : dict
        Regularized parameters with Σ₁₁ = 1.
    """
    scale = sigma[0, 0]
    if scale <= 0:
        return {'mu': mu, 'gamma': gamma, 'sigma': sigma, 'p': p, 'a': a, 'b': b}

    return {
        'mu': mu,
        'gamma': gamma / scale,
        'sigma': sigma / scale,
        'p': p,
        'a': a * scale,
        'b': b / scale
    }


def regularize_fix_p(
    mu: NDArray,
    gamma: NDArray,
    sigma: NDArray,
    p: float,
    a: float,
    b: float,
    d: int,
    p_fixed: float = -0.5
) -> Dict[str, Any]:
    """
    Regularize by fixing the GIG parameter :math:`p`.

    This converts GH to a special case:
    - p = -0.5: Normal-Inverse Gaussian
    - p > 0 with b → 0: Variance Gamma (approximately)

    Parameters
    ----------
    mu, gamma, sigma, p, a, b : parameters
        Classical GH parameters.
    d : int
        Dimension of X.
    p_fixed : float, optional
        Fixed value for p. Default is -0.5 (NIG).

    Returns
    -------
    params : dict
        Regularized parameters with fixed p.
    """
    return {
        'mu': mu,
        'gamma': gamma,
        'sigma': sigma,
        'p': p_fixed,
        'a': a,
        'b': b
    }


def regularize_none(
    mu: NDArray,
    gamma: NDArray,
    sigma: NDArray,
    p: float,
    a: float,
    b: float,
    d: int
) -> Dict[str, Any]:
    """
    No regularization (identity function).

    Parameters
    ----------
    mu, gamma, sigma, p, a, b : parameters
        Classical GH parameters.
    d : int
        Dimension of X.

    Returns
    -------
    params : dict
        Unchanged parameters.
    """
    return {'mu': mu, 'gamma': gamma, 'sigma': sigma, 'p': p, 'a': a, 'b': b}


# Dictionary of available regularization methods
REGULARIZATION_METHODS = {
    'det_sigma_one': regularize_det_sigma_one,
    'sigma_diagonal_one': regularize_sigma_diagonal_one,
    'fix_p': regularize_fix_p,
    'none': regularize_none,
}


class GeneralizedHyperbolic(NormalMixture):
    """
    Generalized Hyperbolic (GH) marginal distribution.

    The Generalized Hyperbolic distribution is a normal variance-mean mixture where
    the mixing distribution is Generalized Inverse Gaussian (GIG). It is the most
    general form of the normal variance-mean mixture family.

    The marginal distribution :math:`f(x)` is NOT an exponential family.
    The joint distribution :math:`f(x, y)` IS an exponential family and can
    be accessed via the ``.joint`` property.

    Parameters
    ----------
    None. Use factory methods to create instances.

    Attributes
    ----------
    _joint : JointGeneralizedHyperbolic
        The underlying joint distribution.

    Examples
    --------
    >>> # 1D case
    >>> gh = GeneralizedHyperbolic.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     p=1.0,
    ...     a=1.0,
    ...     b=1.0
    ... )
    >>> x = gh.rvs(size=1000, random_state=42)  # Samples X only

    >>> # Access joint distribution
    >>> gh.joint.pdf(x, y)  # Joint PDF f(x, y)
    >>> gh.joint.natural_params  # Natural parameters (exponential family)

    >>> # Fit with EM algorithm and regularization
    >>> gh = GeneralizedHyperbolic().fit(
    ...     X, max_iter=100, 
    ...     regularization='det_sigma_one'  # or 'sigma_diagonal_one', 'fix_p', 'none'
    ... )

    See Also
    --------
    JointGeneralizedHyperbolic : Joint distribution (exponential family)
    GeneralizedInverseGaussian : The mixing distribution for Y
    VarianceGamma : Special case with Gamma mixing (b → 0)
    NormalInverseGaussian : Special case with p = -1/2
    NormalInverseGamma : Special case with a → 0

    Notes
    -----
    The GH distribution is widely used in finance for modeling asset returns.
    It includes many important distributions as special cases:

    - **Variance Gamma (VG)**: :math:`b \\to 0, p > 0` (Gamma mixing)
    - **Normal-Inverse Gaussian (NIG)**: :math:`p = -1/2` (IG mixing)
    - **Normal-Inverse Gamma (NInvG)**: :math:`a \\to 0, p < 0` (InvGamma mixing)
    - **Hyperbolic**: :math:`p = 1`
    - **Student-t**: :math:`p = -\\nu/2, a \\to 0, b = \\nu` gives Student-t with ν d.f.

    The model is not identifiable since the parameter sets
    :math:`(\\mu, \\gamma/c, \\Sigma/c, p, c \\cdot b, a/c)` give the same distribution
    for any :math:`c > 0`. Use regularization to fix this.

    Available regularization methods:

    - ``'det_sigma_one'``: Fix :math:`|\\Sigma| = 1` (recommended)
    - ``'sigma_diagonal_one'``: Fix :math:`\\Sigma_{11} = 1`
    - ``'fix_p'``: Fix :math:`p` to a specific value (e.g., -0.5 for NIG)
    - ``'none'``: No regularization
    """

    # ========================================================================
    # Joint distribution factory
    # ========================================================================

    def _create_joint_distribution(self) -> JointNormalMixture:
        """Create the underlying JointGeneralizedHyperbolic instance."""
        return JointGeneralizedHyperbolic()

    # ========================================================================
    # Marginal PDF
    # ========================================================================

    def _marginal_logpdf(self, x: ArrayLike) -> NDArray:
        """
        Compute marginal log PDF: log f(x).

        The marginal PDF for GH has a closed form involving Bessel K functions:

        .. math::
            f(x) = C \\cdot (b + q(x))^{(p - d/2)/2} (a + \\tilde{q})^{(d/2 - p)/2}
            \\cdot \\frac{K_{p - d/2}(\\sqrt{(b + q(x))(a + \\tilde{q})})}{
            K_p(\\sqrt{ab})} \\cdot e^{(x-\\mu)^T\\Lambda\\gamma}

        where:
        - :math:`q(x) = (x-\\mu)^T \\Lambda (x-\\mu)` (Mahalanobis distance squared)
        - :math:`\\tilde{q} = \\gamma^T \\Lambda \\gamma`
        - :math:`C` is the normalizing constant

        Parameters
        ----------
        x : array_like
            Points at which to evaluate log PDF. Shape (d,) or (n, d).

        Returns
        -------
        logpdf : ndarray
            Log PDF values.
        """
        x = np.asarray(x)
        classical = self.get_classical_params()
        mu = classical['mu']
        gamma = classical['gamma']
        Sigma = classical['sigma']
        p = classical['p']
        a = classical['a']
        b = classical['b']
        d = self.d

        # Precision matrix
        Lambda = np.linalg.inv(Sigma)

        # Constants
        gamma_quad = gamma @ Lambda @ gamma
        sqrt_ab = np.sqrt(a * b)

        # Order of Bessel function for marginal: p - d/2
        nu = p - d / 2

        # Log normalizing constant
        _, logdet_Sigma = np.linalg.slogdet(Sigma)

        # C = (a/b)^{p/2} * (a + γ^T Λ γ)^{d/2 - p} / ((2π)^{d/2} |Σ|^{1/2} K_p(√(ab)))
        log_K_p = log_kv(p, sqrt_ab)

        log_C = (0.5 * p * np.log(a / b) +
                 (0.5 * d - p) * np.log(a + gamma_quad) -
                 0.5 * d * np.log(2 * np.pi) -
                 0.5 * logdet_Sigma - log_K_p)

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]
        logpdf = np.zeros(n)

        for i in range(n):
            diff = x[i] - mu

            # Mahalanobis distance squared
            q = diff @ Lambda @ diff

            # Arguments for Bessel function
            arg1 = b + q
            arg2 = a + gamma_quad
            sqrt_arg = np.sqrt(arg1 * arg2)

            # Log Bessel K_{p - d/2}(√((b+q)(a+γ^T Λ γ)))
            log_K_nu = log_kv(nu, sqrt_arg)

            # Linear term: (x-μ)^T Λ γ
            linear = diff @ Lambda @ gamma

            # Log PDF
            # log f(x) = log C + (ν/2) log((b+q)/(a+γ^T Λ γ)) + log K_ν(...) + linear
            # But we already accounted for (d/2 - p) log(a + γ^T Λ γ) in log_C
            # So: log f(x) = log C + (ν/2) log(b+q) - (ν/2) log(a+γ^T Λ γ) + log K_ν + linear
            # With our log_C: (d/2 - p) log(a+γ^T Λ γ) is added, and we need (p - d/2)/2 log(b+q)
            # Actually, let's recalculate more carefully

            # Full formula:
            # f(x) = (a/b)^{p/2} / ((2π)^{d/2} |Σ|^{1/2} K_p(√ab))
            #        × (b + q)^{(p - d/2)/2} × (a + γ^T Λ γ)^{(d/2 - p)/2}
            #        × K_{p-d/2}(√((b+q)(a+γ^T Λ γ))) × exp((x-μ)^T Λ γ)

            # log f(x) = (p/2)(log a - log b) - (d/2) log(2π) - (1/2) log|Σ| - log K_p(√ab)
            #            + ((p - d/2)/2) log(b + q) + ((d/2 - p)/2) log(a + γ^T Λ γ)
            #            + log K_{p-d/2}(√...) + (x-μ)^T Λ γ

            logpdf[i] = (0.5 * p * (np.log(a) - np.log(b)) -
                        0.5 * d * np.log(2 * np.pi) -
                        0.5 * logdet_Sigma - log_K_p +
                        0.5 * nu * np.log(arg1) +
                        0.5 * (-nu) * np.log(arg2) +
                        log_K_nu + linear)

        if single_point:
            return logpdf[0]
        return logpdf

    # ========================================================================
    # EM Algorithm with Regularization
    # ========================================================================

    def fit(
        self,
        X: ArrayLike,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: int = 0,
        regularization: Union[str, Callable] = 'det_sigma_one',
        regularization_params: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ) -> 'GeneralizedHyperbolic':
        """
        Fit distribution to marginal data X using the EM algorithm.

        Since the marginal distribution is not an exponential family, we use
        the EM algorithm treating Y as latent. The joint distribution (X, Y)
        is an exponential family, which makes the EM algorithm tractable.

        Parameters
        ----------
        X : array_like
            Observed data, shape (n_samples, d) or (n_samples,) for d=1.
        max_iter : int, optional
            Maximum number of EM iterations. Default is 100.
        tol : float, optional
            Convergence tolerance for log-likelihood. Default is 1e-6.
        verbose : int, optional
            Verbosity level (0=silent, 1=progress). Default is 0.
        regularization : str or callable, optional
            Regularization method to use. Options:

            - ``'det_sigma_one'``: Fix |Σ| = 1 (recommended, default)
            - ``'sigma_diagonal_one'``: Fix Σ₁₁ = 1
            - ``'fix_p'``: Fix p to a specific value (use regularization_params)
            - ``'none'``: No regularization
            - Or provide a custom callable with signature:
              ``f(mu, gamma, sigma, p, a, b, d) -> dict``

        regularization_params : dict, optional
            Additional parameters for the regularization method.
            For 'fix_p': ``{'p_fixed': -0.5}`` to fix p = -0.5 (NIG).
        random_state : int, optional
            Random seed for initialization.

        Returns
        -------
        self : GeneralizedHyperbolic
            Fitted distribution (returns self for method chaining).

        Notes
        -----
        The EM algorithm alternates between:

        **E-step**: Compute conditional expectations given current parameters:

        .. math::
            E[Y^{-1}|X], E[Y|X], E[\\log Y|X]

        **M-step**: Update parameters using closed-form solutions for normal
        parameters and numerical optimization for GIG parameters.

        After each M-step, regularization is applied to ensure identifiability.
        The recommended regularization is :math:`|\\Sigma| = 1`, which doesn't
        affect EM convergence (see em_algorithm.rst for proof).

        Examples
        --------
        >>> # Default regularization (|Σ| = 1)
        >>> gh = GeneralizedHyperbolic().fit(X)

        >>> # Fix p = -0.5 (NIG)
        >>> gh = GeneralizedHyperbolic().fit(
        ...     X, 
        ...     regularization='fix_p',
        ...     regularization_params={'p_fixed': -0.5}
        ... )

        >>> # Custom regularization
        >>> def my_regularize(mu, gamma, sigma, p, a, b, d):
        ...     return {'mu': mu, 'gamma': gamma, 'sigma': sigma,
        ...             'p': p, 'a': a, 'b': 1.0}  # Fix b = 1
        >>> gh = GeneralizedHyperbolic().fit(X, regularization=my_regularize)
        """
        X = np.asarray(X)

        # Handle 1D X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, d = X.shape

        # Initialize joint distribution if needed
        if self._joint is None:
            self._joint = self._create_joint_distribution()
        self._joint._d = d

        # Get regularization function
        if isinstance(regularization, str):
            if regularization not in REGULARIZATION_METHODS:
                raise ValueError(
                    f"Unknown regularization method: {regularization}. "
                    f"Available: {list(REGULARIZATION_METHODS.keys())}"
                )
            regularize_fn = REGULARIZATION_METHODS[regularization]
        elif callable(regularization):
            regularize_fn = regularization
        else:
            raise ValueError("regularization must be str or callable")

        regularization_params = regularization_params or {}

        # Initialize parameters using method of moments
        if self._joint._natural_params is None:
            self._initialize_params(X, random_state)

        # EM iterations
        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step: compute conditional expectations
            cond_exp = self._conditional_expectation_y_given_x(X)

            # M-step: update parameters
            self._m_step(X, cond_exp)

            # Apply regularization
            classical = self.get_classical_params()
            if regularization == 'fix_p':
                regularized = regularize_fn(
                    classical['mu'], classical['gamma'], classical['sigma'],
                    classical['p'], classical['a'], classical['b'], d,
                    **regularization_params
                )
            else:
                regularized = regularize_fn(
                    classical['mu'], classical['gamma'], classical['sigma'],
                    classical['p'], classical['a'], classical['b'], d
                )

            # Set regularized parameters
            self._joint.set_classical_params(**regularized)

            # Compute log-likelihood
            ll = np.mean(self.logpdf(X))

            if verbose > 0:
                print(f"Iteration {iteration + 1}: log-likelihood = {ll:.6f}")

            # Check convergence
            if abs(ll - prev_ll) < tol:
                if verbose > 0:
                    print(f"Converged after {iteration + 1} iterations.")
                break

            prev_ll = ll

        return self

    def _initialize_params(
        self, 
        X: NDArray, 
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize parameters using method of moments.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        random_state : int, optional
            Random seed.
        """
        n, d = X.shape

        # Estimate mu as sample mean
        mu = np.mean(X, axis=0)

        # Estimate Sigma as sample covariance
        Sigma = np.cov(X, rowvar=False)
        if d == 1:
            Sigma = np.array([[Sigma]])

        # Make sure Sigma is positive definite
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 1e-8:
            Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)

        # Start with symmetric case (gamma = 0)
        gamma = np.zeros(d)

        # Initial GIG parameters (reasonable defaults)
        # Start near the hyperbolic case
        p = 1.0
        a = 1.0
        b = 1.0

        # Set initial parameters
        self._joint.set_classical_params(
            mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b
        )

    def _m_step(self, X: NDArray, cond_exp: Dict[str, NDArray]) -> None:
        """
        M-step: update parameters given conditional expectations.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        cond_exp : dict
            Dictionary with keys 'E_Y', 'E_inv_Y', 'E_log_Y'.
        """
        from normix.distributions.univariate import GeneralizedInverseGaussian

        n, d = X.shape

        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']
        E_log_Y = cond_exp['E_log_Y']

        # Average conditional expectations
        eta_1 = np.mean(E_inv_Y)  # E[E[1/Y|X]]
        eta_2 = np.mean(E_Y)      # E[E[Y|X]]
        eta_3 = np.mean(E_log_Y)  # E[E[log Y|X]]

        # Weighted sums for normal parameters
        eta_4 = np.mean(X, axis=0)  # E[X]
        eta_5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)  # E[X/Y]
        eta_6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n  # E[XX^T/Y]

        # Symmetrize
        eta_6 = (eta_6 + eta_6.T) / 2

        # M-step for normal parameters (closed-form)
        denom = 1.0 - eta_1 * eta_2

        if abs(denom) < 1e-10:
            mu = eta_4 / eta_2 if eta_2 > 0 else eta_4
            gamma = np.zeros(d)
        else:
            mu = (eta_4 - eta_2 * eta_5) / denom
            gamma = (eta_5 - eta_1 * eta_4) / denom

        # Sigma = E[XX^T/Y] - E[X/Y] μ^T - μ E[X/Y]^T + E[1/Y] μ μ^T - E[Y] γ γ^T
        Sigma = (eta_6
                 - np.outer(eta_5, mu)
                 - np.outer(mu, eta_5)
                 + eta_1 * np.outer(mu, mu)
                 - eta_2 * np.outer(gamma, gamma))

        # Symmetrize and ensure positive definiteness
        Sigma = (Sigma + Sigma.T) / 2
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 1e-8:
            Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)

        # M-step for GIG parameters
        gig = GeneralizedInverseGaussian()
        gig_eta = np.array([eta_3, eta_1, eta_2])

        try:
            gig.set_expectation_params(gig_eta)
            gig_classical = gig.get_classical_params()
            p = gig_classical['p']
            a = gig_classical['a']
            b = gig_classical['b']
        except Exception:
            # Fallback: keep current GIG parameters
            current = self.get_classical_params()
            p = current['p']
            a = current['a']
            b = current['b']

        # Bound parameters
        a = max(a, 1e-6)
        b = max(b, 1e-6)

        # Update joint distribution
        self._joint.set_classical_params(
            mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b
        )

    # ========================================================================
    # Conditional Expectations (E-step helper)
    # ========================================================================

    def _conditional_expectation_y_given_x(
        self, 
        X: ArrayLike
    ) -> Dict[str, NDArray]:
        """
        Compute conditional expectations E[Y^α | X] for the E-step.

        The conditional distribution Y | X = x is GIG with parameters:

        .. math::
            Y | X = x \\sim \\text{GIG}\\left(p - \\frac{d}{2}, \\,
            a + \\gamma^T \\Sigma^{-1} \\gamma, \\,
            b + (x-\\mu)^T \\Sigma^{-1}(x-\\mu)\\right)

        Parameters
        ----------
        X : array_like
            Data points, shape (n, d) or (d,).

        Returns
        -------
        expectations : dict
            Dictionary with keys:
            - 'E_Y': E[Y | X], shape (n,)
            - 'E_inv_Y': E[1/Y | X], shape (n,)
            - 'E_log_Y': E[log Y | X], shape (n,)
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        classical = self.get_classical_params()
        mu = classical['mu']
        gamma = classical['gamma']
        Sigma = classical['sigma']
        p = classical['p']
        a = classical['a']
        b = classical['b']
        d = self.d

        # Precision matrix
        Lambda = np.linalg.inv(Sigma)

        # Constants for conditional GIG
        gamma_quad = gamma @ Lambda @ gamma
        a_cond = a + gamma_quad  # a + γ^T Λ γ

        # Conditional GIG order
        p_cond = p - d / 2

        n = X.shape[0]
        E_Y = np.zeros(n)
        E_inv_Y = np.zeros(n)
        E_log_Y = np.zeros(n)

        for i in range(n):
            diff = X[i] - mu
            q = diff @ Lambda @ diff
            b_cond = b + q  # b + q(x)

            # GIG moments
            sqrt_ab = np.sqrt(a_cond * b_cond)
            sqrt_b_over_a = np.sqrt(b_cond / a_cond)

            log_kv_p = log_kv(p_cond, sqrt_ab)
            log_kv_pm1 = log_kv(p_cond - 1, sqrt_ab)
            log_kv_pp1 = log_kv(p_cond + 1, sqrt_ab)

            # E[Y | X=x] = √(b_cond/a_cond) · K_{p_cond+1}(√(ab)) / K_{p_cond}(√(ab))
            E_Y[i] = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)

            # E[1/Y | X=x] = √(a_cond/b_cond) · K_{p_cond-1}(√(ab)) / K_{p_cond}(√(ab))
            E_inv_Y[i] = np.exp(log_kv_pm1 - log_kv_p) / sqrt_b_over_a

            # E[log Y | X=x] = ∂/∂p log(K_p(√(ab))) + (1/2) log(b_cond/a_cond)
            eps = 1e-6
            d_log_kv_dp = (log_kv(p_cond + eps, sqrt_ab) - log_kv(p_cond - eps, sqrt_ab)) / (2 * eps)
            E_log_Y[i] = d_log_kv_dp + 0.5 * np.log(b_cond / a_cond)

        if single_point:
            return {
                'E_Y': float(E_Y[0]),
                'E_inv_Y': float(E_inv_Y[0]),
                'E_log_Y': float(E_log_Y[0])
            }

        return {
            'E_Y': E_Y,
            'E_inv_Y': E_inv_Y,
            'E_log_Y': E_log_Y
        }

    # ========================================================================
    # Special Case Constructors
    # ========================================================================

    @classmethod
    def as_variance_gamma(
        cls,
        mu: ArrayLike,
        gamma: ArrayLike,
        sigma: ArrayLike,
        shape: float,
        rate: float
    ) -> 'GeneralizedHyperbolic':
        """
        Create a GH distribution equivalent to Variance Gamma.

        VG is GH with b → 0 (Gamma mixing).

        Parameters
        ----------
        mu, gamma, sigma : arrays
            Normal parameters.
        shape : float
            Gamma shape parameter α.
        rate : float
            Gamma rate parameter β.

        Returns
        -------
        gh : GeneralizedHyperbolic
            GH distribution equivalent to VG.
        """
        mu = np.asarray(mu).flatten()
        gamma_arr = np.asarray(gamma).flatten()
        sigma = np.asarray(sigma)

        # VG: Y ~ Gamma(α, β)
        # GH: Y ~ GIG(p, a, b) with b → 0, p = α, a = 2β
        # Small b to approximate Gamma limit
        b_small = 1e-10
        return cls.from_classical_params(
            mu=mu, gamma=gamma_arr, sigma=sigma,
            p=shape, a=2 * rate, b=b_small
        )

    @classmethod
    def as_normal_inverse_gaussian(
        cls,
        mu: ArrayLike,
        gamma: ArrayLike,
        sigma: ArrayLike,
        delta: float,
        eta: float
    ) -> 'GeneralizedHyperbolic':
        """
        Create a GH distribution equivalent to Normal-Inverse Gaussian.

        NIG is GH with p = -1/2 (Inverse Gaussian mixing).

        Parameters
        ----------
        mu, gamma, sigma : arrays
            Normal parameters.
        delta : float
            IG mean parameter.
        eta : float
            IG shape parameter.

        Returns
        -------
        gh : GeneralizedHyperbolic
            GH distribution equivalent to NIG.
        """
        mu = np.asarray(mu).flatten()
        gamma_arr = np.asarray(gamma).flatten()
        sigma = np.asarray(sigma)

        # NIG: Y ~ IG(δ, η)
        # GIG: p = -1/2, a = η/δ², b = η
        a = eta / (delta ** 2)
        b = eta
        return cls.from_classical_params(
            mu=mu, gamma=gamma_arr, sigma=sigma,
            p=-0.5, a=a, b=b
        )

    @classmethod
    def as_normal_inverse_gamma(
        cls,
        mu: ArrayLike,
        gamma: ArrayLike,
        sigma: ArrayLike,
        shape: float,
        rate: float
    ) -> 'GeneralizedHyperbolic':
        """
        Create a GH distribution equivalent to Normal-Inverse Gamma.

        NInvG is GH with a → 0 (Inverse Gamma mixing).

        Parameters
        ----------
        mu, gamma, sigma : arrays
            Normal parameters.
        shape : float
            InverseGamma shape parameter α.
        rate : float
            InverseGamma rate parameter β.

        Returns
        -------
        gh : GeneralizedHyperbolic
            GH distribution equivalent to NInvG.
        """
        mu = np.asarray(mu).flatten()
        gamma_arr = np.asarray(gamma).flatten()
        sigma = np.asarray(sigma)

        # NInvG: Y ~ InvGamma(α, β)
        # GIG: a → 0, p = -α, b = 2β
        a_small = 1e-10
        return cls.from_classical_params(
            mu=mu, gamma=gamma_arr, sigma=sigma,
            p=-shape, a=a_small, b=2 * rate
        )

    # ========================================================================
    # Moments
    # ========================================================================

    def mean(self) -> NDArray:
        """
        Mean of the marginal distribution: :math:`E[X] = \\mu + \\gamma E[Y]`.

        Returns
        -------
        mean : ndarray
            Mean vector, shape (d,).
        """
        classical = self.get_classical_params()
        mu = classical['mu']
        gamma = classical['gamma']
        p = classical['p']
        a = classical['a']
        b = classical['b']

        # E[Y] from GIG
        sqrt_ab = np.sqrt(a * b)
        sqrt_b_over_a = np.sqrt(b / a)
        log_kv_p = log_kv(p, sqrt_ab)
        log_kv_pp1 = log_kv(p + 1, sqrt_ab)
        E_Y = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)

        return mu + gamma * E_Y

    def var(self) -> NDArray:
        """
        Variance of the marginal distribution (diagonal elements of covariance).

        .. math::
            \\text{Cov}[X] = E[Y] \\Sigma + \\text{Var}[Y] \\gamma\\gamma^T

        Returns
        -------
        var : ndarray
            Variance vector, shape (d,).
        """
        cov = self.cov()
        return np.diag(cov)

    def cov(self) -> NDArray:
        """
        Covariance matrix of the marginal distribution.

        .. math::
            \\text{Cov}[X] = E[Y] \\Sigma + \\text{Var}[Y] \\gamma\\gamma^T

        Returns
        -------
        cov : ndarray
            Covariance matrix, shape (d, d).
        """
        classical = self.get_classical_params()
        gamma = classical['gamma']
        Sigma = classical['sigma']
        p = classical['p']
        a = classical['a']
        b = classical['b']

        # E[Y] and E[Y²] from GIG
        sqrt_ab = np.sqrt(a * b)
        sqrt_b_over_a = np.sqrt(b / a)
        log_kv_p = log_kv(p, sqrt_ab)
        log_kv_pp1 = log_kv(p + 1, sqrt_ab)
        log_kv_pp2 = log_kv(p + 2, sqrt_ab)

        E_Y = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)
        E_Y2 = (b / a) * np.exp(log_kv_pp2 - log_kv_p)
        Var_Y = E_Y2 - E_Y ** 2

        return E_Y * Sigma + Var_Y * np.outer(gamma, gamma)

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if self._joint is None or self._joint._natural_params is None:
            return "GeneralizedHyperbolic(not fitted)"

        classical = self.get_classical_params()
        d = self.d
        p = classical['p']
        a = classical['a']
        b = classical['b']

        if d == 1:
            mu = float(classical['mu'][0])
            gamma_val = float(classical['gamma'][0])
            return f"GeneralizedHyperbolic(μ={mu:.3f}, γ={gamma_val:.3f}, p={p:.3f}, a={a:.3f}, b={b:.3f})"
        else:
            return f"GeneralizedHyperbolic(d={d}, p={p:.3f}, a={a:.3f}, b={b:.3f})"
