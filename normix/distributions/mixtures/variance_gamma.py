"""
Variance Gamma (VG) marginal distribution :math:`f(x)`.

The marginal distribution of :math:`X` obtained by integrating out :math:`Y`:

.. math::
    f(x) = \\int_0^\\infty f(x, y) \\, dy = \\int_0^\\infty f(x | y) f(y) \\, dy

where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{Gamma}(\\alpha, \\beta)

The marginal PDF has a closed form involving modified Bessel functions:

.. math::
    f(x) = \\frac{2 \\beta^\\alpha}{(2\\pi)^{d/2} |\\Sigma|^{1/2} \\Gamma(\\alpha)}
    \\left(\\frac{q(x)}{2c}\\right)^{(\\alpha - d/2)/2}
    K_{\\alpha - d/2}\\left(\\sqrt{2 q(x) c}\\right)
    e^{(x-\\mu)^T\\Sigma^{-1}\\gamma}

where :math:`q(x) = (x-\\mu)^T\\Sigma^{-1}(x-\\mu)` is the squared Mahalanobis distance
and :math:`c = \\beta + \\frac{1}{2}\\gamma^T\\Sigma^{-1}\\gamma`.

Note: The marginal distribution is NOT an exponential family, but the joint
distribution :math:`f(x, y)` IS an exponential family (accessible via ``.joint``).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, Optional, Tuple, Union

from normix.base import NormalMixture, JointNormalMixture
from normix.utils import log_kv, robust_cholesky
from .joint_variance_gamma import JointVarianceGamma


class VarianceGamma(NormalMixture):
    """
    Variance Gamma (VG) marginal distribution.

    The Variance Gamma distribution is a normal variance-mean mixture where
    the subordinator distribution is Gamma. It is a special case of the Generalized
    Hyperbolic distribution with GIG parameter :math:`b \\to 0`.

    The marginal distribution :math:`f(x)` is NOT an exponential family.
    The joint distribution :math:`f(x, y)` IS an exponential family and can
    be accessed via the ``.joint`` property.

    Parameters
    ----------
    None. Use factory methods to create instances.

    Attributes
    ----------
    _joint : JointVarianceGamma
        The underlying joint distribution.

    Examples
    --------
    >>> # 1D case
    >>> vg = VarianceGamma.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     shape=2.0,
    ...     rate=1.0
    ... )
    >>> x = vg.rvs(size=1000, random_state=42)  # Samples X only

    >>> # Access joint distribution
    >>> vg.joint.pdf(x, y)  # Joint PDF f(x, y)
    >>> vg.joint.natural_params  # Natural parameters (exponential family)

    >>> # 2D case
    >>> vg = VarianceGamma.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     shape=2.0,
    ...     rate=1.0
    ... )

    See Also
    --------
    JointVarianceGamma : Joint distribution (exponential family)
    GeneralizedHyperbolic : General case with GIG subordinator
    NormalInverseGaussian : Special case with Inverse Gaussian subordinator

    Notes
    -----
    The Variance Gamma distribution has heavier tails than the normal
    distribution, controlled by the Gamma shape parameter :math:`\\alpha`.
    Smaller :math:`\\alpha` leads to heavier tails.

    The skewness is controlled by the :math:`\\gamma` parameter:

    - :math:`\\gamma = 0`: symmetric distribution
    - :math:`\\gamma > 0`: right-skewed
    - :math:`\\gamma < 0`: left-skewed
    """

    # ========================================================================
    # Joint distribution factory
    # ========================================================================

    def _create_joint_distribution(self) -> JointNormalMixture:
        """Create the underlying JointVarianceGamma instance."""
        return JointVarianceGamma()

    # ========================================================================
    # Marginal PDF
    # ========================================================================

    def _marginal_logpdf(self, x: ArrayLike) -> NDArray:
        """
        Compute marginal log PDF: log f(x).

        The marginal PDF is derived from integrating the joint:

        .. math::
            f(x) = C \\cdot \\left(\\frac{q(x)}{2c}\\right)^{(\\alpha - d/2)/2} \\cdot
            K_{\\alpha - d/2}(\\sqrt{2 q(x) c}) \\cdot e^{(x-\\mu)^T\\Lambda\\gamma}

        where:
        - :math:`q(x) = (x-\\mu)^T \\Lambda (x-\\mu)` (Mahalanobis distance squared)
        - :math:`c = \\beta + \\frac{1}{2}\\gamma^T\\Lambda\\gamma`
        - :math:`C = \\frac{2 \\beta^\\alpha}{(2\\pi)^{d/2} |\\Sigma|^{1/2} \\Gamma(\\alpha)}`

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

        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

        diff = x - mu
        z = L_inv @ diff.T
        q = np.sum(z ** 2, axis=0)

        gamma_z = L_inv @ gamma
        gamma_quad = np.dot(gamma_z, gamma_z)

        linear = z.T @ gamma_z

        c = beta + 0.5 * gamma_quad

        log_C = (np.log(2) - 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet_Sigma
                 - gammaln(alpha) + alpha * np.log(beta))

        # Order of Bessel function
        nu = alpha - d / 2

        # Initialize logpdf
        logpdf = np.zeros(n)

        # Handle q = 0 case (x = μ) specially
        small_q_mask = q <= 1e-14
        if np.any(small_q_mask):
            if nu > 0:
                # Limit: log C + log(Γ(ν) * 2^{ν-1} / (2c)^ν) + linear
                logpdf[small_q_mask] = (log_C + gammaln(nu) + (nu - 1) * np.log(2) 
                                        - nu * np.log(2.0 * c) + linear[small_q_mask])
            else:
                # For ν ≤ 0, the PDF diverges at q=0
                logpdf[small_q_mask] = np.inf if nu < 0 else -np.inf

        # Handle normal case (q > 0) - vectorized
        normal_mask = ~small_q_mask
        if np.any(normal_mask):
            q_normal = q[normal_mask]
            linear_normal = linear[normal_mask]

            # Bessel function argument: sqrt(2 * q * c)
            z_arg = np.sqrt(2.0 * q_normal * c)

            # Log Bessel K (vectorized)
            log_K = log_kv(nu, z_arg)

            # Combine: log C + (ν/2) log(q/(2c)) + log K_ν(z) + linear
            logpdf[normal_mask] = (log_C + 0.5 * nu * np.log(q_normal / (2.0 * c)) 
                                   + log_K + linear_normal)

        if single_point:
            return float(logpdf[0])

        return logpdf

    def _conditional_expectation_y_given_x(
        self, x: ArrayLike
    ) -> Dict[str, NDArray]:
        """
        Compute conditional expectations :math:`E[g(Y) | X = x]` for EM algorithm.

        For Variance Gamma, the conditional distribution of :math:`Y | X = x` is
        a Generalized Inverse Gaussian (GIG):

        .. math::
            Y | X = x \\sim \\text{GIG}\\left(\\alpha - \\frac{d}{2}, \\,
            \\beta + \\frac{1}{2} \\gamma^T \\Sigma^{-1} \\gamma, \\,
            (x-\\mu)^T \\Sigma^{-1} (x-\\mu)\\right)

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
        # From the derivation:
        # f(Y|X) ∝ Y^{p-1} exp(-(a Y + b/Y)/2)
        # where:
        #   p = α - d/2
        #   a = 2β + γ^T Σ^{-1} γ
        #   b = (x-μ)^T Σ^{-1} (x-μ)
        p_cond = alpha - d / 2

        # a = 2β + γ^T Σ^{-1} γ (same for all x)
        gamma_quad = self._joint.gamma_mahal_sq  # γ^T Σ^{-1} γ
        a_cond = 2 * beta + gamma_quad

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

        # b = (x - μ)^T Σ^{-1} (x - μ) for each x
        diff = x - mu  # (n, d)
        z = L_inv @ diff.T  # (d, n)
        b_cond = np.sum(z ** 2, axis=0)  # (n,)

        # Ensure b > 0 (add small epsilon for numerical stability)
        b_cond = np.maximum(b_cond, 1e-10)

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

    def _initialize_params(self, X: NDArray) -> None:
        """
        Initialize VG parameters using method of moments.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        """
        n, d = X.shape

        mu_init = np.mean(X, axis=0)
        Sigma_init = np.cov(X, rowvar=False)
        if Sigma_init.ndim == 0:
            Sigma_init = np.array([[Sigma_init]])

        L = robust_cholesky(Sigma_init, eps=1e-6)

        self._joint._set_internal(
            mu=mu_init, gamma=np.zeros(d), L_sigma=L,
            shape=2.0, rate=1.0,
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

        # Gamma parameters via subordinator.set_expectation_params
        # Gamma expectation params: [digamma(alpha) - log(beta), alpha/beta] = [E[log Y], E[Y]]
        sub = self._joint.subordinator
        sub_eta = np.array([s3, s2])
        sub.set_expectation_params(sub_eta)

        if verbose >= 1:
            recovered = sub._compute_expectation_params()
            eta_diff = np.max(np.abs(recovered - sub_eta))
            if eta_diff > 1e-6:
                print(f"  Warning: Gamma expectation param roundtrip error = {eta_diff:.2e}")

        alpha_new = sub.classical_params.shape
        beta_new = sub.classical_params.rate

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
    ) -> 'VarianceGamma':
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
        self : VarianceGamma
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
            self._initialize_params(X_norm)

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
    ) -> 'VarianceGamma':
        """
        Fit distribution from complete data (both X and Y observed).

        Since the joint distribution is an exponential family, this uses
        the closed-form MLE via the joint distribution's fit method.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d) or (n_samples,) for d=1.
        Y : array_like
            Observed Y data (subordinator variable), shape (n_samples,).
        **kwargs
            Additional fitting parameters passed to joint.fit().

        Returns
        -------
        self : VarianceGamma
            Fitted distribution (returns self for method chaining).

        Examples
        --------
        >>> # Generate complete data
        >>> true_dist = VarianceGamma.from_classical_params(
        ...     mu=np.array([0.0]), gamma=np.array([0.5]),
        ...     sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        ... )
        >>> X, Y = true_dist.rvs_joint(size=5000, random_state=42)
        >>>
        >>> # Fit with complete data
        >>> fitted = VarianceGamma().fit_complete(X, Y)
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
            return "VarianceGamma(not fitted)"

        try:
            classical = self.classical_params
            d = self.d
            alpha = classical['shape']
            beta = classical['rate']

            if d == 1:
                mu = float(classical['mu'][0])
                gamma_val = float(classical['gamma'][0])
                return f"VarianceGamma(μ={mu:.3f}, γ={gamma_val:.3f}, α={alpha:.3f}, β={beta:.3f})"
            else:
                return f"VarianceGamma(d={d}, α={alpha:.3f}, β={beta:.3f})"
        except ValueError:
            return "VarianceGamma(not fitted)"


# Import gammaln at module level
from scipy.special import gammaln
