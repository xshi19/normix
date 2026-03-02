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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    from .variance_gamma import VarianceGamma
    from .normal_inverse_gaussian import NormalInverseGaussian
    from .normal_inverse_gamma import NormalInverseGamma

from normix.base import NormalMixture, JointNormalMixture
from normix.utils import log_kv, robust_cholesky
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
    d: int,
    log_det_sigma: Optional[float] = None
) -> Dict[str, Any]:
    """
    Regularize GH parameters by enforcing :math:`|\\Sigma| = 1`.

    This is the recommended regularization that doesn't affect convergence
    of the EM algorithm. Following the legacy implementation:

    .. math::
        (\\mu, \\gamma, \\Sigma, p, a, b) \\to
        (\\mu, |\\Sigma|^{-1/d} \\gamma, |\\Sigma|^{-1/d} \\Sigma,
        p, |\\Sigma|^{-1/d} a, |\\Sigma|^{1/d} b)

    All computations are done in log space to avoid underflow/overflow
    of the determinant in high dimensions.

    Parameters
    ----------
    mu, gamma, sigma, p, a, b : parameters
        Classical GH parameters.
    d : int
        Dimension of X.
    log_det_sigma : float, optional
        Precomputed log determinant of Sigma. If provided, avoids
        recomputing the determinant.

    Returns
    -------
    params : dict
        Regularized parameters with |Σ| = 1.
    """
    # Compute log determinant in a numerically stable way
    if log_det_sigma is not None:
        log_det = log_det_sigma
    else:
        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            # Sigma is degenerate, can't regularize
            return {'mu': mu, 'gamma': gamma, 'sigma': sigma, 'p': p, 'a': a, 'b': b}

    # Work entirely in log space: scale = det^(1/d) = exp(log_det / d)
    log_scale = log_det / d
    scale = np.exp(log_scale)

    # Guard against degenerate scale (e.g., scale=0 or inf)
    if scale <= 0 or not np.isfinite(scale):
        return {'mu': mu, 'gamma': gamma, 'sigma': sigma, 'p': p, 'a': a, 'b': b}

    return {
        'mu': mu,
        'gamma': gamma / scale,
        'sigma': sigma / scale,
        'p': p,
        'a': a / scale,  # Legacy: psi = psi / scale
        'b': b * scale   # Legacy: chi = chi * scale
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
        'a': a / scale,
        'b': b * scale
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
    the subordinator distribution is Generalized Inverse Gaussian (GIG). It is the most
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
    GeneralizedInverseGaussian : The subordinator distribution for Y
    VarianceGamma : Special case with Gamma subordinator (b → 0)
    NormalInverseGaussian : Special case with p = -1/2
    NormalInverseGamma : Special case with a → 0

    Notes
    -----
    The GH distribution is widely used in finance for modeling asset returns.
    It includes many important distributions as special cases:

    - **Variance Gamma (VG)**: :math:`b \\to 0, p > 0` (Gamma subordinator)
    - **Normal-Inverse Gaussian (NIG)**: :math:`p = -1/2` (IG subordinator)
    - **Normal-Inverse Gamma (NInvG)**: :math:`a \\to 0, p < 0` (InvGamma subordinator)
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
        Compute marginal log PDF: log f(x) (vectorized with Cholesky).

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
        mu = self._joint._mu
        gamma = self._joint._gamma
        p = self._joint._p
        a = self._joint._a
        b = self._joint._b
        d = self.d

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

        # Get cached L_Sigma_inv and log_det_Sigma
        L_inv = self._joint.L_Sigma_inv
        logdet_Sigma = self._joint.log_det_Sigma
        
        # Transform data: z = L^{-1}(x - μ)
        # Mahalanobis distance: q(x) = ||z||^2 = (x-μ)^T Σ^{-1} (x-μ)
        diff = x - mu  # (n, d)
        z = L_inv @ diff.T  # (d, n)
        q = np.sum(z ** 2, axis=0)  # (n,)
        
        # Transform gamma: gamma_z = L^{-1} γ
        gamma_z = L_inv @ gamma  # (d,)
        gamma_quad = np.dot(gamma_z, gamma_z)  # γ^T Σ^{-1} γ
        
        # Linear term: (x-μ)^T Σ^{-1} γ = z.T @ gamma_z
        linear = z.T @ gamma_z  # (n,)

        # Constants
        sqrt_ab = np.sqrt(a * b)
        nu = p - d / 2  # Order of Bessel function for marginal

        # Log Bessel K_p(√ab) - constant for all x
        log_K_p = log_kv(p, sqrt_ab)

        # Vectorized log PDF computation:
        # log f(x) = (p/2)(log a - log b) - (d/2) log(2π) - (1/2) log|Σ| - log K_p(√ab)
        #            + (ν/2) log(b + q) + (-ν/2) log(a + γ^T Λ γ)
        #            + log K_{p-d/2}(√((b+q)(a+γ^TΛγ))) + (x-μ)^T Λ γ
        
        # Arguments for Bessel function
        arg1 = b + q  # (n,)
        arg2 = a + gamma_quad  # scalar
        eta = np.sqrt(arg1 * arg2)  # (n,)
        
        # Log Bessel K_{p-d/2}(eta) - vectorized
        log_K_nu = log_kv(nu, eta)  # (n,)
        
        # Constant part (same for all samples)
        log_const = (0.5 * p * (np.log(a) - np.log(b)) -
                     0.5 * d * np.log(2 * np.pi) -
                     0.5 * logdet_Sigma - log_K_p +
                     0.5 * (-nu) * np.log(arg2))
        
        # Variable part (depends on x)
        logpdf = log_const + 0.5 * nu * np.log(arg1) + log_K_nu + linear

        if single_point:
            return float(logpdf[0])
        return logpdf

    # ========================================================================
    # EM Algorithm with Regularization
    # ========================================================================

    def fit(
        self,
        X: ArrayLike,
        *,
        max_iter: int = 100,
        tol: float = 1e-2,
        verbose: int = 0,
        regularization: Union[str, Callable] = 'det_sigma_one',
        regularization_params: Optional[Dict] = None,
        random_state: Optional[int] = None,
        fix_tail: bool = False,
    ) -> 'GeneralizedHyperbolic':
        """
        Fit distribution to marginal data X using the EM algorithm.

        Since the marginal distribution is not an exponential family, we use
        the EM algorithm treating Y as latent. The joint distribution (X, Y)
        is an exponential family, which makes the EM algorithm tractable.

        Convergence is checked based on the relative change in the normal
        parameters :math:`(\\mu, \\gamma, \\Sigma)`, which are more stable than
        the GIG parameters. Log-likelihood is optionally displayed per
        iteration when ``verbose >= 1``.

        Parameters
        ----------
        X : array_like
            Observed data, shape (n_samples, d) or (n_samples,) for d=1.
        max_iter : int, optional
            Maximum number of EM iterations. Default is 100.
        tol : float, optional
            Convergence tolerance for relative parameter change (based on
            :math:`\\mu, \\gamma, \\Sigma`). Default is 1e-4.
        verbose : int, optional
            Verbosity level. 0 = silent, 1 = progress with llh,
            2 = detailed with parameter norms. Default is 0.
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
        fix_tail : bool, optional
            If True, do not update GIG parameters (p, a, b) during fitting.
            Useful for fitting special cases like VG, NIG, NInvG. Default False.

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

        The initial parameters are obtained by running a short EM
        (5 iterations) for each special case (NIG, VG, NInvG) and
        selecting the candidate with the highest log-likelihood as the
        starting point for :math:`(\\mu, \\gamma, \\Sigma)` and the GIG
        parameters.

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

        # Normalize data for numerical stability
        X_norm, center, scale = self._normalize_data(X)

        n_samples, d = X_norm.shape

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

        # Initialize parameters using NIG warm start
        if not self._joint._fitted:
            self._initialize_params(X_norm, random_state)

        if verbose > 0:
            init_ll = np.mean(self.logpdf(X_norm))
            print(f"Initial log-likelihood: {init_ll:.6f}")

        def _apply_regularization():
            """Apply regularization to current parameters."""
            jt = self._joint
            mu = jt._mu
            gamma = jt._gamma
            sigma = jt._L_Sigma @ jt._L_Sigma.T
            p_val, a_val, b_val = jt._p, jt._a, jt._b
            log_det_sigma = jt.log_det_Sigma

            if regularization == 'fix_p':
                regularized = regularize_fn(
                    mu, gamma, sigma, p_val, a_val, b_val, d,
                    **regularization_params
                )
            elif regularization == 'det_sigma_one':
                regularized = regularize_fn(
                    mu, gamma, sigma, p_val, a_val, b_val, d,
                    log_det_sigma=log_det_sigma
                )
            else:
                regularized = regularize_fn(
                    mu, gamma, sigma, p_val, a_val, b_val, d
                )

            # Sanity check: clamp degenerate GIG parameters
            a_reg = regularized['a']
            b_reg = regularized['b']
            if a_reg < 1e-6 or b_reg < 1e-6 or a_reg > 1e6 or b_reg > 1e6:
                regularized['a'] = np.clip(a_reg, 1e-6, 1e6)
                regularized['b'] = np.clip(b_reg, 1e-6, 1e6)

            self._joint.set_classical_params(**regularized)

        # EM iterations
        for iteration in range(max_iter):
            _apply_regularization()

            prev_mu = self._joint._mu.copy()
            prev_gamma = self._joint._gamma.copy()
            prev_L = self._joint._L_Sigma.copy()

            cond_exp = self._conditional_expectation_y_given_x(X_norm)
            self._m_step(X_norm, cond_exp, fix_tail=fix_tail, verbose=verbose)

            _apply_regularization()

            max_rel_change = self._check_convergence(
                prev_mu, prev_gamma, prev_L,
                verbose=verbose, iteration=iteration, X=X_norm,
            )

            if max_rel_change < tol:
                if verbose >= 1:
                    print(f"Converged at iteration {iteration + 1}")
                self.n_iter_ = iteration + 1
                self._denormalize_params(center, scale)
                self._fitted = self._joint._fitted
                self._invalidate_cache()
                return self

        self.n_iter_ = max_iter
        self._denormalize_params(center, scale)
        self._fitted = self._joint._fitted
        self._invalidate_cache()

        if verbose >= 1:
            final_ll = np.mean(self.logpdf(X))
            print(f"Final log-likelihood: {final_ll:.6f}")

        return self

    def _initialize_params(
        self, 
        X: NDArray, 
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize GH parameters by fitting special-case sub-models and
        selecting the one with highest log-likelihood.

        Runs a short EM (5 iterations) for each of three special cases:

        - **NIG** (Normal-Inverse Gaussian): :math:`p = -1/2`
        - **VG** (Variance Gamma): :math:`b \\to 0`
        - **NInvG** (Normal-Inverse Gamma): :math:`a \\to 0`

        Each fitted sub-model is converted to GH parametrisation and scored
        on mean log-likelihood.  The best candidate is used as the starting
        point for the full GH EM.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        random_state : int, optional
            Random seed.
        """
        import warnings
        from .normal_inverse_gaussian import NormalInverseGaussian
        from .variance_gamma import VarianceGamma
        from .normal_inverse_gamma import NormalInverseGamma

        n, d = X.shape
        candidates = []  # (name, mean_llh, mu, gamma, Sigma, p, a, b)

        # --- Candidate: NIG (p = -1/2) ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nig = NormalInverseGaussian()
                nig.fit(X, max_iter=5, verbose=0, random_state=random_state)

            par = nig.classical_params
            p = -0.4999
            a = np.clip(par['eta'] / (par['delta'] ** 2), 1e-6, 1e6)
            b = np.clip(par['eta'], 1e-6, 1e6)
            ll = np.mean(nig.logpdf(X))
            if np.isfinite(ll):
                candidates.append(
                    ('NIG', ll, par['mu'], par['gamma'], par['sigma'], p, a, b)
                )
        except Exception:
            pass

        # --- Candidate: VG (b → 0, Gamma subordinator) ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vg = VarianceGamma()
                vg.fit(X, max_iter=5, verbose=0, random_state=random_state)

            par = vg.classical_params
            p = par['shape']
            a = np.clip(2.0 * par['rate'], 1e-6, 1e6)
            b = 1e-4
            ll = np.mean(vg.logpdf(X))
            if np.isfinite(ll):
                candidates.append(
                    ('VG', ll, par['mu'], par['gamma'], par['sigma'], p, a, b)
                )
        except Exception:
            pass

        # --- Candidate: NInvG (a → 0, InverseGamma subordinator) ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ninvg = NormalInverseGamma()
                ninvg.fit(X, max_iter=5, verbose=0, random_state=random_state)

            par = ninvg.classical_params
            p = -par['shape']
            a = 1e-4
            b = np.clip(2.0 * par['rate'], 1e-6, 1e6)
            ll = np.mean(ninvg.logpdf(X))
            if np.isfinite(ll):
                candidates.append(
                    ('NInvG', ll, par['mu'], par['gamma'], par['sigma'], p, a, b)
                )
        except Exception:
            pass

        if candidates:
            best = max(candidates, key=lambda c: c[1])
            _, _, mu, gamma, Sigma, p, a, b = best
        else:
            mu = np.mean(X, axis=0)
            Sigma = np.cov(X, rowvar=False)
            if d == 1:
                Sigma = np.array([[Sigma]])
            L = robust_cholesky(Sigma)
            Sigma = L @ L.T
            gamma = np.zeros(d)
            p = 1.0
            a = 1.0
            b = 1.0

        self._joint.set_classical_params(
            mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b
        )

    def _m_step(
        self, 
        X: NDArray, 
        cond_exp: Dict[str, NDArray],
        fix_tail: bool = False,
        *,
        verbose: int = 0,
    ) -> None:
        """
        M-step: update parameters given conditional expectations.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        cond_exp : dict
            Dictionary with keys 'E_Y', 'E_inv_Y', 'E_log_Y'.
        fix_tail : bool, optional
            If True, do not update GIG parameters (p, a, b). Default False.
        verbose : int, optional
            Verbosity level for diagnostics.
        """
        n, d = X.shape

        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']
        E_log_Y = cond_exp['E_log_Y']

        # Average conditional expectations (sufficient statistics for GIG)
        s1 = np.mean(E_inv_Y)  # E[E[1/Y|X]]
        s2 = np.mean(E_Y)      # E[E[Y|X]]
        s3 = np.mean(E_log_Y)  # E[E[log Y|X]]

        # Weighted sums for normal parameters
        s4 = np.mean(X, axis=0)  # E[X]
        s5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)  # E[X/Y]
        s6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n  # E[XX^T/Y]

        # Get current GIG parameters for fallback
        current_p = self._joint._p
        current_a = self._joint._a
        current_b = self._joint._b

        # Update GIG parameters if not fixed
        if not fix_tail:
            sub = self._joint.subordinator
            sub_eta = np.array([s3, s1, s2])

            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub.set_expectation_params(sub_eta, theta0=sub.natural_params)

                if verbose >= 1:
                    recovered = sub._compute_expectation_params()
                    eta_diff = np.max(np.abs(recovered - sub_eta))
                    if eta_diff > 1e-6:
                        print(f"  Warning: GIG expectation param roundtrip error = {eta_diff:.2e}")

                sub_classical = sub.classical_params
                p_new = sub_classical.p
                a_new = sub_classical.a
                b_new = sub_classical.b

                if (abs(p_new) > 50 or a_new > 1e10 or b_new > 1e10 or
                    a_new < 1e-10 or b_new < 1e-10):
                    p = current_p
                    a = current_a
                    b = current_b
                else:
                    p = p_new
                    a = a_new
                    b = b_new
            except Exception:
                p = current_p
                a = current_a
                b = current_b
        else:
            p = current_p
            a = current_a
            b = current_b

        # M-step for normal parameters (closed-form, following legacy)
        # mu = (s4 - s2 * s5) / (1 - s1 * s2)
        # gamma = (s5 - s1 * s4) / (1 - s1 * s2)
        denom = 1.0 - s1 * s2

        if abs(denom) < 1e-10:
            mu = s4 / s2 if s2 > 0 else s4
            gamma = np.zeros(d)
        else:
            mu = (s4 - s2 * s5) / denom
            gamma = (s5 - s1 * s4) / denom

        # Sigma = -s5 μ^T - μ s5^T + s6 + s1 μ μ^T - s2 γ γ^T
        # Following legacy code exactly
        Sigma = -np.outer(s5, mu)
        Sigma = Sigma + Sigma.T + s6 + s1 * np.outer(mu, mu) - s2 * np.outer(gamma, gamma)

        L_Sigma = robust_cholesky(Sigma)

        # Bound GIG parameters
        a = max(a, 1e-6)
        b = max(b, 1e-6)

        # Update joint distribution via fast path (no redundant Cholesky)
        self._joint._set_internal(
            mu=mu, gamma=gamma, L_sigma=L_Sigma,
            p=p, a=a, b=b
        )
        self._fitted = True
        self._invalidate_cache()

    # ========================================================================
    # Conditional Expectations (E-step helper)
    # ========================================================================

    def _conditional_expectation_y_given_x(
        self, 
        X: ArrayLike
    ) -> Dict[str, NDArray]:
        """
        Compute conditional expectations E[Y^α | X] for the E-step (vectorized).

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
        from normix.utils import kv_ratio
        
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        mu = self._joint._mu
        gamma = self._joint._gamma
        p = self._joint._p
        a = self._joint._a
        b = self._joint._b
        d = self.d

        L_inv = self._joint.L_Sigma_inv
        
        # Transform data: z = L^{-1}(x - μ)
        # Mahalanobis distance: q(x) = (x-μ)^T Σ^{-1} (x-μ) = ||z||^2
        diff = X - mu  # (n, d)
        z = L_inv @ diff.T  # (d, n)
        
        # Mahalanobis distances: q(x) = (x-μ)^T Σ^{-1} (x-μ) = ||z||^2
        q = np.sum(z ** 2, axis=0)  # (n,)
        
        # Transform gamma: L^{-1} γ
        gamma_z = L_inv @ gamma  # (d,)
        gamma_quad = np.dot(gamma_z, gamma_z)  # γ^T Σ^{-1} γ
        
        # Conditional GIG parameters (vectorized)
        p_cond = p - d / 2  # scalar
        a_cond = a + gamma_quad  # scalar (same for all x)
        b_cond = b + q  # (n,) - varies with x
        
        # delta = sqrt(b_cond / a_cond), eta = sqrt(a_cond * b_cond)
        delta = np.sqrt(b_cond / a_cond)  # (n,)
        eta = np.sqrt(a_cond * b_cond)  # (n,)
        
        # E[Y | X] = delta * K_{p_cond+1}(eta) / K_{p_cond}(eta)
        E_Y = delta * kv_ratio(p_cond + 1, p_cond, eta)
        
        # E[1/Y | X] = (1/delta) * K_{p_cond-1}(eta) / K_{p_cond}(eta)
        E_inv_Y = kv_ratio(p_cond - 1, p_cond, eta) / delta
        
        # E[log Y | X] = ∂/∂p log(K_p(eta)) + log(delta)
        # Numerical derivative for ∂/∂p log(K_p(eta))
        eps = 1e-5
        d_log_kv_dp = (log_kv(p_cond + eps, eta) - log_kv(p_cond - eps, eta)) / (2 * eps)
        E_log_Y = d_log_kv_dp + np.log(delta)

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

        VG is GH with b → 0 (Gamma subordinator).

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

        NIG is GH with p = -1/2 (Inverse Gaussian subordinator).

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

        NInvG is GH with a → 0 (Inverse Gamma subordinator).

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
    # Conversion to special cases (instance methods, using expectation params)
    # ========================================================================

    def to_variance_gamma(self) -> 'VarianceGamma':
        """
        Convert to the closest :class:`VarianceGamma` distribution by matching
        expectation parameters.

        Projects the GH expectation-parameter vector onto the Variance Gamma
        submanifold (:math:`b \\to 0`, Gamma subordinator) by fitting a
        :class:`JointVarianceGamma` to the current expectation parameters and
        wrapping it in a :class:`VarianceGamma` marginal.

        Returns
        -------
        vg : VarianceGamma
            Fitted Variance Gamma distribution.

        See Also
        --------
        JointGeneralizedHyperbolic.to_joint_variance_gamma
        """
        from .variance_gamma import VarianceGamma

        self._check_fitted()
        jvg = self._joint.to_joint_variance_gamma()
        vg = VarianceGamma()
        vg._joint = jvg
        vg._fitted = True
        vg._invalidate_cache()
        return vg

    def to_normal_inverse_gaussian(self) -> 'NormalInverseGaussian':
        """
        Convert to the closest :class:`NormalInverseGaussian` distribution by
        matching expectation parameters.

        Projects the GH expectation-parameter vector onto the Normal-Inverse
        Gaussian submanifold (:math:`p = -1/2`, Inverse Gaussian subordinator)
        by fitting a :class:`JointNormalInverseGaussian` to the current
        expectation parameters and wrapping it in a
        :class:`NormalInverseGaussian` marginal.

        Returns
        -------
        nig : NormalInverseGaussian
            Fitted Normal-Inverse Gaussian distribution.

        See Also
        --------
        JointGeneralizedHyperbolic.to_joint_normal_inverse_gaussian
        """
        from .normal_inverse_gaussian import NormalInverseGaussian

        self._check_fitted()
        jnig = self._joint.to_joint_normal_inverse_gaussian()
        nig = NormalInverseGaussian()
        nig._joint = jnig
        nig._fitted = True
        nig._invalidate_cache()
        return nig

    def to_normal_inverse_gamma(self) -> 'NormalInverseGamma':
        """
        Convert to the closest :class:`NormalInverseGamma` distribution by
        matching expectation parameters.

        Projects the GH expectation-parameter vector onto the Normal-Inverse
        Gamma submanifold (:math:`a \\to 0`, Inverse Gamma subordinator) by
        fitting a :class:`JointNormalInverseGamma` to the current expectation
        parameters and wrapping it in a :class:`NormalInverseGamma` marginal.

        Returns
        -------
        ning : NormalInverseGamma
            Fitted Normal-Inverse Gamma distribution.

        See Also
        --------
        JointGeneralizedHyperbolic.to_joint_normal_inverse_gamma
        """
        from .normal_inverse_gamma import NormalInverseGamma

        self._check_fitted()
        jning = self._joint.to_joint_normal_inverse_gamma()
        ning = NormalInverseGamma()
        ning._joint = jning
        ning._fitted = True
        ning._invalidate_cache()
        return ning

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
        mu = self._joint._mu
        gamma = self._joint._gamma
        p = self._joint._p
        a = self._joint._a
        b = self._joint._b

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
        gamma = self._joint._gamma
        Sigma = self._joint._L_Sigma @ self._joint._L_Sigma.T
        p = self._joint._p
        a = self._joint._a
        b = self._joint._b

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
        if not self._fitted:
            return "GeneralizedHyperbolic(not fitted)"

        classical = self.classical_params
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
