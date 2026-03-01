"""
Normal Inverse Gamma (NInvG) marginal distribution :math:`f(x)`.

The marginal distribution of :math:`X` obtained by integrating out :math:`Y`:

.. math::
    f(x) = \\int_0^\\infty f(x, y) \\, dy = \\int_0^\\infty f(x | y) f(y) \\, dy

where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{InvGamma}(\\alpha, \\beta)

The marginal PDF has a closed form involving modified Bessel functions:

.. math::
    f(x) = \\frac{2 \\beta^\\alpha}{(2\\pi)^{d/2} |\\Sigma|^{1/2} \\Gamma(\\alpha)}
    \\left(\\frac{q(x)}{2c}\\right)^{(\\alpha - d/2)/2}
    K_{\\alpha - d/2}\\left(\\sqrt{2 q(x) c}\\right)
    e^{(x-\\mu)^T\\Sigma^{-1}\\gamma}

where :math:`q(x) = (x-\\mu)^T\\Sigma^{-1}(x-\\mu)` is the squared Mahalanobis distance
and :math:`c` depends on the skewness parameter :math:`\\gamma`.

Note: The marginal distribution is NOT an exponential family, but the joint
distribution :math:`f(x, y)` IS an exponential family (accessible via ``.joint``).

This is a special case of the Generalized Hyperbolic distribution with GIG
parameter :math:`a \\to 0` (inverse gamma mixing).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, Optional, Tuple, Union

from normix.base import NormalMixture, JointNormalMixture
from normix.utils import log_kv, robust_cholesky
from .joint_normal_inverse_gamma import JointNormalInverseGamma


class NormalInverseGamma(NormalMixture):
    """
    Normal Inverse Gamma (NInvG) marginal distribution.

    The Normal Inverse Gamma distribution is a normal variance-mean mixture where
    the mixing distribution is InverseGamma. It is a special case of the Generalized
    Hyperbolic distribution with GIG parameter :math:`a \\to 0`.

    The marginal distribution :math:`f(x)` is NOT an exponential family.
    The joint distribution :math:`f(x, y)` IS an exponential family and can
    be accessed via the ``.joint`` property.

    Parameters
    ----------
    None. Use factory methods to create instances.

    Attributes
    ----------
    _joint : JointNormalInverseGamma
        The underlying joint distribution.

    Examples
    --------
    >>> # 1D case
    >>> nig = NormalInverseGamma.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     shape=3.0,
    ...     rate=1.0
    ... )
    >>> x = nig.rvs(size=1000, random_state=42)  # Samples X only

    >>> # Access joint distribution
    >>> nig.joint.pdf(x, y)  # Joint PDF f(x, y)
    >>> nig.joint.natural_params  # Natural parameters (exponential family)

    >>> # 2D case
    >>> nig = NormalInverseGamma.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     shape=3.0,
    ...     rate=1.0
    ... )

    See Also
    --------
    JointNormalInverseGamma : Joint distribution (exponential family)
    GeneralizedHyperbolic : General case with GIG mixing
    VarianceGamma : Special case with Gamma mixing

    Notes
    -----
    The Normal Inverse Gamma distribution has heavier tails than the normal
    distribution, controlled by the InverseGamma shape parameter :math:`\\alpha`.
    Smaller :math:`\\alpha` leads to heavier tails.

    The skewness is controlled by the :math:`\\gamma` parameter:

    - :math:`\\gamma = 0`: symmetric distribution
    - :math:`\\gamma > 0`: right-skewed
    - :math:`\\gamma < 0`: left-skewed

    For the mean to exist, we need :math:`\\alpha > 1`.
    For the variance to exist, we need :math:`\\alpha > 2`.
    """

    # ========================================================================
    # Joint distribution factory
    # ========================================================================

    def _create_joint_distribution(self) -> JointNormalMixture:
        """Create the underlying JointNormalInverseGamma instance."""
        return JointNormalInverseGamma()

    # ========================================================================
    # Marginal PDF
    # ========================================================================

    def _marginal_logpdf(self, x: ArrayLike) -> NDArray:
        """
        Compute marginal log PDF: log f(x).

        The marginal PDF is derived from integrating the joint.
        For Normal Inverse Gamma, the conditional Y|X follows a 
        Generalized Inverse Gaussian (GIG) distribution, and the marginal
        involves Bessel K functions.

        The formula is similar to Variance Gamma but with different parameters:

        .. math::
            f(x) = C \\cdot \\left(\\frac{q(x)}{2c}\\right)^{\\nu/2} \\cdot
            K_{\\nu}(\\sqrt{2 q(x) c}) \\cdot e^{(x-\\mu)^T\\Lambda\\gamma}

        where:
        - :math:`q(x) = (x-\\mu)^T \\Lambda (x-\\mu)` (Mahalanobis distance squared)
        - :math:`c = \\beta` (rate parameter)
        - :math:`\\nu = \\alpha - d/2` (order of Bessel function)
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
        alpha = self._joint._alpha
        beta = self._joint._beta
        d = self.d

        L_inv = self._joint.L_Sigma_inv
        logdet_Sigma = self._joint.log_det_Sigma

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

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

        # Log normalizing constant (without log(2) - that comes from the GIG integral)
        log_C = (- 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet_Sigma
                 - gammaln(alpha) + alpha * np.log(beta))

        # GIG parameters: p = -(α + d/2), a = γ^T Σ^{-1} γ, b = q + 2β
        p_gig = -(alpha + d / 2)
        a_gig = gamma_quad
        b_gig = q + 2 * beta  # (n,)

        # Initialize logpdf
        logpdf = np.zeros(n)

        # Handle the case where a_gig ≈ 0 (symmetric case)
        if a_gig < 1e-12:
            # Symmetric case: reduces to Student-t like
            # For a → 0 in GIG with p < 0:
            # ∫ y^{p-1} exp(-b/(2y)) dy = (b/2)^p × Γ(-p)
            if p_gig < 0:
                # For negative p, the integral is (b/2)^p × Γ(-p)
                # log_integral = p × log(b/2) + log Γ(-p)
                #              = (-p) × log(2/b) + log Γ(-p)
                log_integral = (-p_gig) * np.log(2.0 / b_gig) + gammaln(-p_gig)
                logpdf = log_C + linear + log_integral
            else:
                # This shouldn't happen for typical α > 0
                logpdf[:] = -np.inf
        else:
            # General case with γ ≠ 0 - vectorized
            sqrt_ab = np.sqrt(a_gig * b_gig)  # (n,)
            
            # The GIG normalization integral is:
            # ∫ y^{p-1} exp(-b/(2y) - a*y/2) dy = 2 (b/a)^{p/2} K_p(√(ab))
            log_bessel = log_kv(p_gig, sqrt_ab)  # (n,)
            log_integral = np.log(2) + 0.5 * p_gig * np.log(b_gig / a_gig) + log_bessel
            logpdf = log_C + linear + log_integral

        if single_point:
            return float(logpdf[0])

        return logpdf

    def _conditional_expectation_y_given_x(
        self, x: ArrayLike
    ) -> Dict[str, NDArray]:
        """
        Compute conditional expectations :math:`E[g(Y) | X = x]` for EM algorithm.

        For Normal Inverse Gamma, the conditional distribution of :math:`Y | X = x` is
        a Generalized Inverse Gaussian (GIG):

        .. math::
            Y | X = x \\sim \\text{GIG}\\left(-(\\alpha + \\frac{d}{2}), \\,
            \\gamma^T \\Sigma^{-1} \\gamma, \\,
            2\\beta + (x-\\mu)^T \\Sigma^{-1} (x-\\mu)\\right)

        where the GIG parameters are :math:`(p, a, b)` in our notation.

        Parameters
        ----------
        x : array_like
            Observed X values. Shape (d,) for single point or (n, d) for n points.

        Returns
        -------
        expectations : dict
            Dictionary with:
            - 'E_Y': :math:`E[Y | X]`, shape (n,)
            - 'E_inv_Y': :math:`E[1/Y | X]`, shape (n,)
            - 'E_log_Y': :math:`E[\\log Y | X]`, shape (n,)
        """
        x = np.asarray(x)
        mu = self._joint._mu
        gamma = self._joint._gamma
        alpha = self._joint._alpha
        beta = self._joint._beta
        d = self.d

        L_inv = self._joint.L_Sigma_inv

        # GIG parameters for Y | X = x
        # p = -(α + d/2)
        # a = γ^T Σ^{-1} γ
        # b = 2β + (x-μ)^T Σ^{-1} (x-μ)
        p_cond = -(alpha + d / 2)

        # a is same for all x
        gamma_z = L_inv @ gamma  # (d,)
        gamma_quad = np.dot(gamma_z, gamma_z)  # γ^T Σ^{-1} γ
        a_cond = gamma_quad

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

        # b = 2β + (x - μ)^T Σ^{-1} (x - μ) for each x
        diff = x - mu  # (n, d)
        z = L_inv @ diff.T  # (d, n)
        q_x = np.sum(z ** 2, axis=0)  # (n,)
        b_cond = 2 * beta + q_x

        # Ensure b > 0 (add small epsilon for numerical stability)
        b_cond = np.maximum(b_cond, 1e-10)

        # Handle the case where a ≈ 0 (symmetric case, γ ≈ 0)
        if a_cond < 1e-12:
            # When a → 0, GIG(p, a, b) → InvGamma if p < 0
            # For GIG with a → 0 and p < 0:
            # E[Y] = (b/2) / (|p| - 1) for |p| > 1
            # E[1/Y] = 2|p| / b
            # E[log Y] = log(b/2) - ψ(|p|)
            
            abs_p = np.abs(p_cond)
            
            if abs_p > 1:
                E_Y = (b_cond / 2) / (abs_p - 1)
            else:
                E_Y = np.full(n, np.inf)
            
            E_inv_Y = 2 * abs_p / b_cond
            E_log_Y = np.log(b_cond / 2) - digamma(abs_p)
        else:
            # General case with a > 0
            # Compute GIG expectations using moment formula
            # E[Y^α] = (b/a)^(α/2) * K_{p+α}(√(ab)) / K_p(√(ab))
            sqrt_ab = np.sqrt(a_cond * b_cond)  # (n,)
            sqrt_b_over_a = np.sqrt(b_cond / a_cond)  # (n,)

            # Log Bessel function values
            log_kv_p = log_kv(p_cond, sqrt_ab)
            log_kv_pm1 = log_kv(p_cond - 1, sqrt_ab)
            log_kv_pp1 = log_kv(p_cond + 1, sqrt_ab)

            # E[Y] = √(b/a) * K_{p+1}(√(ab)) / K_p(√(ab))
            E_Y = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)

            # E[1/Y] = √(a/b) * K_{p-1}(√(ab)) / K_p(√(ab))
            E_inv_Y = np.exp(log_kv_pm1 - log_kv_p) / sqrt_b_over_a

            # E[log Y] = ∂/∂p log(K_p(√(ab))) + (1/2) log(b/a)
            # Numerical derivative for ∂/∂p log(K_p(z))
            eps = 1e-6
            log_kv_p_plus = log_kv(p_cond + eps, sqrt_ab)
            log_kv_p_minus = log_kv(p_cond - eps, sqrt_ab)
            d_log_kv_dp = (log_kv_p_plus - log_kv_p_minus) / (2 * eps)
            E_log_Y = d_log_kv_dp + 0.5 * np.log(b_cond / a_cond)

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
    # Fitting via EM algorithm
    # ========================================================================

    def _initialize_params(
        self,
        X: NDArray,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> None:
        """
        Initialize NInvG parameters using method of moments.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        random_state : int or Generator, optional
            Random state (reserved for future use).
        """
        n, d = X.shape

        mu_init = np.mean(X, axis=0)
        Sigma_init = np.cov(X, rowvar=False)
        if Sigma_init.ndim == 0:
            Sigma_init = np.array([[Sigma_init]])

        alpha_init = 3.0
        beta_init = 1.0
        gamma_init = np.zeros(d)

        L = robust_cholesky(Sigma_init, eps=1e-6)
        Sigma_init = L @ L.T

        self.set_classical_params(
            mu=mu_init,
            gamma=gamma_init,
            sigma=Sigma_init,
            shape=alpha_init,
            rate=beta_init
        )

    def _m_step(
        self,
        X: NDArray,
        cond_exp: Dict[str, NDArray],
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
            Dictionary with keys ``'E_Y'``, ``'E_inv_Y'``, ``'E_log_Y'``.
        verbose : int, optional
            Verbosity level for diagnostics.
        """
        from normix.distributions.univariate import InverseGamma

        n, d = X.shape

        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']
        E_log_Y = cond_exp['E_log_Y']

        s1 = np.mean(E_inv_Y)
        s2 = np.mean(E_Y)
        s3 = np.mean(E_log_Y)
        s4 = np.mean(X, axis=0)
        s5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)
        s6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n

        # Normal parameters (closed-form)
        denom = 1.0 - s1 * s2
        if abs(denom) < 1e-10:
            denom = np.sign(denom) * 1e-10 if denom != 0 else 1e-10

        mu = (s4 - s2 * s5) / denom
        gamma = (s5 - s1 * s4) / denom

        Sigma = -np.outer(s5, mu)
        Sigma = Sigma + Sigma.T + s6 + s1 * np.outer(mu, mu) - s2 * np.outer(gamma, gamma)

        L_Sigma = robust_cholesky(Sigma)

        # InverseGamma parameters via set_expectation_params
        # InvGamma expectation params: [-alpha/beta, log(beta) - digamma(alpha)] = [E[-1/Y], E[log Y]]
        # From EM: E[-1/Y] = -s1, E[log Y] = s3
        ig_dist = InverseGamma()
        ig_eta = np.array([-s1, s3])
        current_theta = np.array([
            self._joint._beta,
            -(self._joint._alpha + 1)
        ])
        ig_dist.set_expectation_params(ig_eta, theta0=current_theta)

        if verbose >= 1:
            recovered = ig_dist._compute_expectation_params()
            eta_diff = np.max(np.abs(recovered - ig_eta))
            if eta_diff > 1e-6:
                print(f"  Warning: InverseGamma expectation param roundtrip error = {eta_diff:.2e}")

        alpha_new = ig_dist.classical_params.shape
        beta_new = ig_dist.classical_params.rate

        self._joint._set_internal(
            mu=mu, gamma=gamma, L_sigma=L_Sigma,
            shape=alpha_new, rate=beta_new
        )
        self._fitted = True
        self._invalidate_cache()

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-3,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> 'NormalInverseGamma':
        """
        Fit distribution to data using the EM algorithm.

        Convergence is checked based on the relative change in the normal
        parameters :math:`(\\mu, \\gamma, \\Sigma)`. Log-likelihood is optionally
        displayed per iteration when ``verbose >= 1``.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d) or (n_samples,) for d=1.
        y : array_like, optional
            Ignored (for sklearn API compatibility).
        max_iter : int, optional
            Maximum number of EM iterations. Default is 100.
        tol : float, optional
            Convergence tolerance for relative parameter change.
            Default is 1e-3.
        verbose : int, optional
            Verbosity level. 0 = silent, 1 = progress with llh,
            2 = detailed with parameter norms. Default is 0.
        random_state : int or Generator, optional
            Random state for initialization.

        Returns
        -------
        self : NormalInverseGamma
            Fitted distribution (returns self for method chaining).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Normalize data for numerical stability
        X_norm, center, scale = self._normalize_data(X)

        n_samples, d = X_norm.shape

        # Initialize joint distribution if needed
        if self._joint is None:
            self._joint = self._create_joint_distribution()
        self._joint._d = d

        if not self._joint._fitted:
            self._initialize_params(X_norm, random_state)

        if verbose >= 1:
            init_ll = np.mean(self.logpdf(X_norm))
            print(f"Initial log-likelihood: {init_ll:.6f}")

        # EM iterations
        for iteration in range(max_iter):
            prev_mu = self._joint._mu.copy()
            prev_gamma = self._joint._gamma.copy()
            prev_L = self._joint._L_Sigma.copy()

            cond_exp = self._conditional_expectation_y_given_x(X_norm)
            self._m_step(X_norm, cond_exp, verbose=verbose)

            max_rel_change = self._check_convergence(
                prev_mu, prev_gamma, prev_L,
                verbose=verbose, iteration=iteration, X=X_norm,
            )

            if max_rel_change < tol:
                if verbose >= 1:
                    print(f"Converged at iteration {iteration + 1}")
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = max_iter

        # Transform parameters back to original scale
        self._denormalize_params(center, scale)

        self._fitted = self._joint._fitted
        self._invalidate_cache()

        if verbose >= 1:
            final_ll = np.mean(self.logpdf(X))
            print(f"Final log-likelihood: {final_ll:.6f}")

        return self

    def fit_complete(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        **kwargs
    ) -> 'NormalInverseGamma':
        """
        Fit distribution from complete data (both X and Y observed).

        Since the joint distribution is an exponential family, this uses
        the closed-form MLE via the joint distribution's fit method.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d) or (n_samples,) for d=1.
        Y : array_like
            Observed Y data (mixing variable), shape (n_samples,).
        **kwargs
            Additional fitting parameters passed to joint.fit().

        Returns
        -------
        self : NormalInverseGamma
            Fitted distribution (returns self for method chaining).
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Handle 1D X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Initialize joint if needed
        if self._joint is None:
            self._joint = self._create_joint_distribution()

        # Use joint distribution's fit method
        self._joint.fit(X, Y, **kwargs)

        self._fitted = self._joint._fitted
        self._invalidate_cache()

        return self

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if not self._fitted:
            return "NormalInverseGamma(not fitted)"

        try:
            classical = self.classical_params
            d = self.d
            alpha = classical['shape']
            beta = classical['rate']

            if d == 1:
                mu = float(classical['mu'][0])
                gamma_val = float(classical['gamma'][0])
                return f"NormalInverseGamma(μ={mu:.3f}, γ={gamma_val:.3f}, α={alpha:.3f}, β={beta:.3f})"
            else:
                return f"NormalInverseGamma(d={d}, α={alpha:.3f}, β={beta:.3f})"
        except ValueError:
            return "NormalInverseGamma(not fitted)"


# Import gammaln and digamma at module level
from scipy.special import gammaln, digamma
