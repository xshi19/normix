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
        classical = self.joint.classical_params
        p = classical['p']
        a = classical['a']
        b = classical['b']
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

        The initial parameters are obtained by running a short NIG EM
        (5 iterations) to get a reasonable starting point for
        :math:`(\\mu, \\gamma, \\Sigma)` and the GIG parameters.

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

        # Initialize parameters using NIG warm start
        if not self._joint._fitted:
            self._initialize_params(X, random_state)

        if verbose > 0:
            init_ll = np.mean(self.logpdf(X))
            print(f"Initial log-likelihood: {init_ll:.6f}")

        def _apply_regularization():
            """Apply regularization to current parameters."""
            classical = self.classical_params
            log_det_sigma = self._joint.log_det_Sigma

            if regularization == 'fix_p':
                regularized = regularize_fn(
                    classical['mu'], classical['gamma'], classical['sigma'],
                    classical['p'], classical['a'], classical['b'], d,
                    **regularization_params
                )
            elif regularization == 'det_sigma_one':
                regularized = regularize_fn(
                    classical['mu'], classical['gamma'], classical['sigma'],
                    classical['p'], classical['a'], classical['b'], d,
                    log_det_sigma=log_det_sigma
                )
            else:
                regularized = regularize_fn(
                    classical['mu'], classical['gamma'], classical['sigma'],
                    classical['p'], classical['a'], classical['b'], d
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
            # Apply regularization BEFORE E-step (following legacy)
            _apply_regularization()

            # Save regularized parameters for convergence check
            classical = self.classical_params
            prev_mu = classical['mu'].copy()
            prev_gamma = classical['gamma'].copy()
            prev_sigma = classical['sigma'].copy()

            # E-step: compute conditional expectations
            cond_exp = self._conditional_expectation_y_given_x(X)

            # M-step: update parameters
            self._m_step(X, cond_exp, fix_tail=fix_tail)

            # Apply regularization AFTER M-step to ensure constraints hold
            _apply_regularization()

            # Check convergence using relative parameter change on (mu, gamma, Sigma)
            new_classical = self.classical_params
            mu_new = new_classical['mu']
            gamma_new = new_classical['gamma']
            sigma_new = new_classical['sigma']

            mu_norm = np.linalg.norm(mu_new - prev_mu)
            mu_denom = max(np.linalg.norm(prev_mu), 1e-10)
            rel_mu = mu_norm / mu_denom

            gamma_norm = np.linalg.norm(gamma_new - prev_gamma)
            gamma_denom = max(np.linalg.norm(prev_gamma), 1e-10)
            rel_gamma = gamma_norm / gamma_denom

            sigma_norm = np.linalg.norm(sigma_new - prev_sigma, 'fro')
            sigma_denom = max(np.linalg.norm(prev_sigma, 'fro'), 1e-10)
            rel_sigma = sigma_norm / sigma_denom

            max_rel_change = max(rel_mu, rel_gamma, rel_sigma)

            if verbose > 0:
                ll = np.mean(self.logpdf(X))
                print(f"Iteration {iteration + 1}: log-likelihood = {ll:.6f}")
                if verbose > 1:
                    print(f"  rel_change: mu={rel_mu:.2e}, gamma={rel_gamma:.2e}, "
                          f"Sigma={rel_sigma:.2e}")

            if max_rel_change < tol:
                if verbose > 0:
                    print('Converged')
                self.n_iter_ = iteration + 1
                self._fitted = self._joint._fitted
                self._invalidate_cache()
                return self

        if verbose > 0:
            print('Not converged (max iterations reached)')
        self.n_iter_ = max_iter
        self._fitted = self._joint._fitted
        self._invalidate_cache()
        return self

    def _initialize_params(
        self, 
        X: NDArray, 
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize GH parameters using a short NIG EM warm start.

        Runs a Normal-Inverse Gaussian EM for a few iterations to obtain
        reasonable initial values for :math:`(\\mu, \\gamma, \\Sigma)` and
        the GIG parameters :math:`(p, a, b)`.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        random_state : int, optional
            Random seed.
        """
        import warnings
        from .normal_inverse_gaussian import NormalInverseGaussian

        n, d = X.shape

        try:
            # Use NIG with a few EM iterations to get a warm start
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nig = NormalInverseGaussian()
                nig.fit(X, max_iter=5, verbose=0, random_state=random_state)

            nig_params = nig.classical_params
            mu = nig_params['mu']
            gamma = nig_params['gamma']
            Sigma = nig_params['sigma']
            delta = nig_params['delta']
            eta = nig_params['eta']

            # Convert NIG (IG) params to GIG params:
            # IG(δ, η) corresponds to GIG(p=-0.5, a=η/δ², b=η)
            p = -0.45
            a = eta / (delta ** 2)
            b = eta

            # Clamp GIG parameters to reasonable range
            a = np.clip(a, 1e-6, 1e6)
            b = np.clip(b, 1e-6, 1e6)

        except Exception:
            # Fallback: method of moments initialization
            mu = np.mean(X, axis=0)
            Sigma = np.cov(X, rowvar=False)
            if d == 1:
                Sigma = np.array([[Sigma]])

            min_eig = np.linalg.eigvalsh(Sigma).min()
            if min_eig < 1e-8:
                Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)

            gamma = np.zeros(d)
            p = 1.0
            a = 1.0
            b = 1.0

        # Set initial parameters
        self._joint.set_classical_params(
            mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b
        )

    def _m_step(
        self, 
        X: NDArray, 
        cond_exp: Dict[str, NDArray],
        fix_tail: bool = False
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
        """
        from normix.distributions.univariate import GeneralizedInverseGaussian

        n, d = X.shape

        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']
        E_log_Y = cond_exp['E_log_Y']

        # Average conditional expectations (sufficient statistics for GIG)
        # Following legacy naming: s1 = E[1/Y], s2 = E[Y], s3 = E[log Y]
        s1 = np.mean(E_inv_Y)  # E[E[1/Y|X]]
        s2 = np.mean(E_Y)      # E[E[Y|X]]
        s3 = np.mean(E_log_Y)  # E[E[log Y|X]]

        # Weighted sums for normal parameters
        # s4 = E[X], s5 = E[X/Y], s6 = E[XX^T/Y]
        s4 = np.mean(X, axis=0)  # E[X]
        s5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)  # E[X/Y]
        s6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n  # E[XX^T/Y]

        # Get current GIG parameters for initialization and fallback
        current = self.joint.classical_params
        
        # Update GIG parameters if not fixed
        if not fix_tail:
            gig = GeneralizedInverseGaussian()
            # Sufficient statistics for GIG: [E[log Y], E[1/Y], E[Y]]
            gig_eta = np.array([s3, s1, s2])
            
            # Use current GIG parameters as initial point
            current_gig_theta = np.array([
                current['p'] - 1, 
                -current['b'] / 2, 
                -current['a'] / 2
            ])
            
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gig.set_expectation_params(gig_eta, theta0=current_gig_theta)
                    
                gig_classical = gig.classical_params
                p_new = gig_classical.p
                a_new = gig_classical.a
                b_new = gig_classical.b
                
                # Sanity check: reject if parameters are extreme or degenerate
                # This prevents the optimization from diverging
                if (abs(p_new) > 50 or a_new > 1e10 or b_new > 1e10 or
                    a_new < 1e-10 or b_new < 1e-10):
                    # Keep current parameters
                    p = current['p']
                    a = current['a']
                    b = current['b']
                else:
                    p = p_new
                    a = a_new
                    b = b_new
            except Exception:
                # Fallback: keep current GIG parameters
                p = current['p']
                a = current['a']
                b = current['b']
        else:
            # Keep current GIG parameters
            p = current['p']
            a = current['a']
            b = current['b']

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

        # Symmetrize and ensure positive definiteness
        Sigma = (Sigma + Sigma.T) / 2
        min_eig = np.linalg.eigvalsh(Sigma).min()
        if min_eig < 1e-8:
            Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)

        # Compute Cholesky factor of Sigma once and cache it
        from scipy.linalg import cholesky
        L_Sigma = cholesky(Sigma, lower=True)

        # Bound GIG parameters
        a = max(a, 1e-6)
        b = max(b, 1e-6)

        # Update joint distribution and cache L_Sigma
        self._joint.set_classical_params(
            mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b
        )
        # Cache the Cholesky factor (set_classical_params clears cache, so set after)
        self._joint.set_L_Sigma(L_Sigma)

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
        classical = self.joint.classical_params
        p = classical['p']
        a = classical['a']
        b = classical['b']
        d = self.d

        # Get cached L_Sigma_inv
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
        mu = self._joint._mu
        gamma = self._joint._gamma
        classical = self.joint.classical_params
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
        gamma = self._joint._gamma
        Sigma = self._joint._L_Sigma @ self._joint._L_Sigma.T
        classical = self.joint.classical_params
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
