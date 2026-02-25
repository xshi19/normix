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

        When only X is observed (Y is latent), uses the EM algorithm described
        in the theory documentation. The E-step computes conditional expectations
        of Y given X, and the M-step updates parameters using closed-form formulas.

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
            Default is 1e-6.
        verbose : int, optional
            Verbosity level. 0 = silent, 1 = progress with llh,
            2 = detailed with parameter norms. Default is 0.
        random_state : int or Generator, optional
            Random state for initialization.

        Returns
        -------
        self : NormalInverseGamma
            Fitted distribution (returns self for method chaining).

        Notes
        -----
        The EM algorithm iterates between:

        **E-step**: Compute conditional expectations

        .. math::
            E[Y | X = x_j], \\quad E[1/Y | X = x_j], \\quad E[\\log Y | X = x_j]

        **M-step**: Update parameters using closed-form formulas

        The InverseGamma parameters :math:`(\\alpha, \\beta)` are updated via Newton's method.

        For valid estimates, we need :math:`\\alpha > 1` for the mean to exist.
        """
        from scipy.special import digamma, polygamma

        X = np.asarray(X)

        # Handle 1D X
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, d = X.shape

        # Initialize joint distribution if needed
        if self._joint is None:
            self._joint = self._create_joint_distribution()
        self._joint._d = d

        # Set up RNG for initialization
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state

        # ================================================================
        # Initialize parameters using method of moments
        # ================================================================
        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X, rowvar=False)
        if X_cov.ndim == 0:
            X_cov = np.array([[X_cov]])

        # For NInvG: E[X] = μ + γ E[Y] = μ + γ β/(α-1)
        # Var[X] = E[Y] Σ + Var[Y] γγ^T
        # 
        # Initial guesses using heuristics:
        # - Start with α = 3, β = 1 (so E[Y] = 0.5, Var[Y] = 0.25)
        # - Estimate γ from skewness (if available)
        # - μ = E[X] - γ E[Y]
        # - Σ from residual variance

        alpha_init = 3.0  # Need α > 2 for finite variance
        beta_init = 1.0
        E_Y_init = beta_init / (alpha_init - 1)

        # Estimate skewness to get initial gamma
        X_centered = X - X_mean
        X_std = np.std(X, axis=0)
        X_std = np.maximum(X_std, 1e-10)  # Avoid division by zero

        # Skewness: E[(X - μ_X)³] / σ³
        skewness = np.mean((X_centered / X_std) ** 3, axis=0)

        # Heuristic: γ ≈ skewness * σ / (some factor)
        gamma_init = skewness * X_std * 0.1

        # μ = E[X] - γ E[Y]
        mu_init = X_mean - gamma_init * E_Y_init

        # Σ from sample covariance, adjusted for gamma contribution
        Var_Y_init = beta_init**2 / ((alpha_init - 1)**2 * (alpha_init - 2))
        Sigma_init = (X_cov - Var_Y_init * np.outer(gamma_init, gamma_init)) / E_Y_init

        # Ensure Sigma is positive definite via robust Cholesky
        L = robust_cholesky(Sigma_init, eps=1e-6)
        Sigma_init = L @ L.T

        # Set initial parameters
        self.set_classical_params(
            mu=mu_init,
            gamma=gamma_init,
            sigma=Sigma_init,
            shape=alpha_init,
            rate=beta_init
        )

        if verbose >= 1:
            init_ll = np.mean(self.logpdf(X))
            print(f"Initial log-likelihood: {init_ll:.6f}")

        # ================================================================
        # EM iterations
        # ================================================================
        for iteration in range(max_iter):
            prev_mu = self._joint._mu.copy()
            prev_gamma = self._joint._gamma.copy()
            prev_L = self._joint._L_Sigma.copy()
            prev_shape = self._joint._alpha
            prev_rate = self._joint._beta

            # ==============================================================
            # E-step: Compute conditional expectations E[g(Y) | X]
            # ==============================================================
            cond_exp = self._conditional_expectation_y_given_x(X)
            E_Y = cond_exp['E_Y']          # (n,)
            E_inv_Y = cond_exp['E_inv_Y']  # (n,)
            E_log_Y = cond_exp['E_log_Y']  # (n,)

            # Compute weighted statistics
            eta_1 = np.mean(E_inv_Y)
            eta_2 = np.mean(E_Y)
            eta_3 = np.mean(E_log_Y)
            eta_4 = np.mean(X, axis=0)  # (d,)
            eta_5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)  # (d,)

            # η̂₆ = (1/n) Σ X_j X_j^T E[1/Y | X_j]
            eta_6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n_samples

            # ==============================================================
            # M-step: Update parameters
            # ==============================================================

            # Denominator: 1 - η̂₁ η̂₂
            denom = 1.0 - eta_1 * eta_2

            # Handle edge case
            if abs(denom) < 1e-10:
                if verbose >= 1:
                    print(f"Warning: denominator near zero at iteration {iteration}")
                denom = np.sign(denom) * 1e-10 if denom != 0 else 1e-10

            # μ = (η̂₄ - η̂₂ η̂₅) / (1 - η̂₁ η̂₂)
            mu_new = (eta_4 - eta_2 * eta_5) / denom

            # γ = (η̂₅ - η̂₁ η̂₄) / (1 - η̂₁ η̂₂)
            gamma_new = (eta_5 - eta_1 * eta_4) / denom

            # Σ = η̂₆ - η̂₅ μ^T - μ η̂₅^T + η̂₁ μ μ^T - η̂₂ γ γ^T
            Sigma_new = (eta_6
                         - np.outer(eta_5, mu_new)
                         - np.outer(mu_new, eta_5)
                         + eta_1 * np.outer(mu_new, mu_new)
                         - eta_2 * np.outer(gamma_new, gamma_new))

            L = robust_cholesky(Sigma_new)

            # InverseGamma parameters via Newton's method
            # For InvGamma: E[1/Y] = α/β, E[log Y] = log(β) - ψ(α)
            # Solve: ψ(α) - log(α) = -η̂₃ - log(η̂₁)
            target = -eta_3 - np.log(eta_1)

            alpha_new = prev_shape

            for _ in range(50):
                psi_val = digamma(alpha_new)
                psi_prime = polygamma(1, alpha_new)

                f_val = psi_val - np.log(alpha_new) - target
                f_prime = psi_prime - 1.0 / alpha_new

                step = f_val / f_prime
                alpha_candidate = alpha_new - step

                # Keep α > 1.5 (need α > 1 for finite mean)
                alpha_candidate = max(alpha_candidate, 1.5)
                alpha_candidate = min(alpha_candidate, 1000.0)

                if abs(alpha_candidate - alpha_new) / max(abs(alpha_new), 1e-10) < 1e-10:
                    alpha_new = alpha_candidate
                    break

                alpha_new = alpha_candidate

            # β = α / η̂₁
            beta_new = alpha_new / eta_1

            # ==============================================================
            # Update parameters via _set_internal
            # ==============================================================
            try:
                self._joint._set_internal(
                    mu=mu_new, gamma=gamma_new, L_sigma=L,
                    shape=alpha_new, rate=beta_new
                )
                self._fitted = True
                self._invalidate_cache()
            except (ValueError, np.linalg.LinAlgError) as e:
                if verbose >= 1:
                    print(f"Warning: parameter update failed at iteration {iteration}: {e}")
                self._joint._set_internal(
                    mu=prev_mu, gamma=prev_gamma,
                    L_sigma=prev_L,
                    shape=prev_shape, rate=prev_rate
                )
                self._fitted = True
                self._invalidate_cache()
                self.n_iter_ = iteration + 1
                break

            # ==============================================================
            # Check convergence using relative parameter change
            # ==============================================================
            mu_norm = np.linalg.norm(mu_new - prev_mu)
            mu_denom_val = max(np.linalg.norm(prev_mu), 1e-10)
            rel_mu = mu_norm / mu_denom_val

            gamma_norm = np.linalg.norm(gamma_new - prev_gamma)
            gamma_denom_val = max(np.linalg.norm(prev_gamma), 1e-10)
            rel_gamma = gamma_norm / gamma_denom_val

            L_norm = np.linalg.norm(L - prev_L, 'fro')
            L_denom = max(np.linalg.norm(prev_L, 'fro'), 1e-10)
            rel_sigma = L_norm / L_denom

            max_rel_change = max(rel_mu, rel_gamma, rel_sigma)

            if verbose >= 1:
                current_ll = np.mean(self.logpdf(X))
                print(f"Iteration {iteration + 1}: log-likelihood = {current_ll:.6f}")
                if verbose >= 2:
                    print(f"  rel_change: mu={rel_mu:.2e}, gamma={rel_gamma:.2e}, "
                          f"Sigma={rel_sigma:.2e}")

            if max_rel_change < tol:
                if verbose >= 1:
                    print(f"Converged at iteration {iteration + 1}")
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = max_iter

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
