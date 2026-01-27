"""
Normal Inverse Gaussian (NIG) marginal distribution :math:`f(x)`.

The marginal distribution of :math:`X` obtained by integrating out :math:`Y`:

.. math::
    f(x) = \\int_0^\\infty f(x, y) \\, dy = \\int_0^\\infty f(x | y) f(y) \\, dy

where:

.. math::
    X | Y \\sim N(\\mu + \\gamma Y, \\Sigma Y)

    Y \\sim \\text{InvGauss}(\\delta, \\eta)

The marginal PDF has a closed form involving modified Bessel functions :math:`K_1`:

.. math::
    f(x) = \\frac{\\eta \\delta}{\\pi} \\frac{K_1(\\eta \\sqrt{\\delta^2 + q(x)})}{\\sqrt{\\delta^2 + q(x)}}
    e^{\\delta \\eta + (x-\\mu)^T \\Sigma^{-1} \\gamma}

for the univariate case, where :math:`q(x) = (x-\\mu)^T \\Sigma^{-1}(x-\\mu)`.

This is a special case of the Generalized Hyperbolic distribution with
GIG parameter :math:`p = -1/2`.

Note: The marginal distribution is NOT an exponential family, but the joint
distribution :math:`f(x, y)` IS an exponential family (accessible via ``.joint``).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, Dict, Optional, Tuple, Union

from normix.base import NormalMixture, JointNormalMixture
from normix.utils import log_kv
from .joint_normal_inverse_gaussian import JointNormalInverseGaussian


class NormalInverseGaussian(NormalMixture):
    """
    Normal Inverse Gaussian (NIG) marginal distribution.

    The Normal Inverse Gaussian distribution is a normal variance-mean mixture where
    the mixing distribution is Inverse Gaussian. It is a special case of the Generalized
    Hyperbolic distribution with GIG parameter :math:`p = -1/2`.

    The marginal distribution :math:`f(x)` is NOT an exponential family.
    The joint distribution :math:`f(x, y)` IS an exponential family and can
    be accessed via the ``.joint`` property.

    Parameters
    ----------
    None. Use factory methods to create instances.

    Attributes
    ----------
    _joint : JointNormalInverseGaussian
        The underlying joint distribution.

    Examples
    --------
    >>> # 1D case
    >>> nig = NormalInverseGaussian.from_classical_params(
    ...     mu=np.array([0.0]),
    ...     gamma=np.array([0.5]),
    ...     sigma=np.array([[1.0]]),
    ...     delta=1.0,
    ...     eta=1.0
    ... )
    >>> x = nig.rvs(size=1000, random_state=42)  # Samples X only

    >>> # Access joint distribution
    >>> nig.joint.pdf(x, y)  # Joint PDF f(x, y)
    >>> nig.joint.natural_params  # Natural parameters (exponential family)

    >>> # 2D case
    >>> nig = NormalInverseGaussian.from_classical_params(
    ...     mu=np.array([0.0, 0.0]),
    ...     gamma=np.array([0.5, -0.3]),
    ...     sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
    ...     delta=1.0,
    ...     eta=1.0
    ... )

    See Also
    --------
    JointNormalInverseGaussian : Joint distribution (exponential family)
    GeneralizedHyperbolic : General case with GIG mixing
    VarianceGamma : Special case with Gamma mixing
    NormalInverseGamma : Special case with InverseGamma mixing

    Notes
    -----
    The NIG distribution is widely used in finance due to its semi-heavy tails
    and ability to model both skewness and excess kurtosis. It was introduced
    by Barndorff-Nielsen (1977) and provides a good fit for asset returns.

    The skewness is controlled by the :math:`\\gamma` parameter:

    - :math:`\\gamma = 0`: symmetric distribution
    - :math:`\\gamma > 0`: right-skewed
    - :math:`\\gamma < 0`: left-skewed

    The tail heaviness is controlled by :math:`\\eta`: smaller :math:`\\eta`
    gives heavier tails.
    """

    # ========================================================================
    # Joint distribution factory
    # ========================================================================

    def _create_joint_distribution(self) -> JointNormalMixture:
        """Create the underlying JointNormalInverseGaussian instance."""
        return JointNormalInverseGaussian()

    # ========================================================================
    # Marginal PDF
    # ========================================================================

    def _marginal_logpdf(self, x: ArrayLike) -> NDArray:
        """
        Compute marginal log PDF: log f(x).

        The marginal PDF for NIG has a closed form involving Bessel K functions.
        For the general multivariate case:

        .. math::
            f(x) = C \\cdot \\frac{K_{-1/2 - d/2}(\\sqrt{(b + q(x))(a + \\gamma^T\\Lambda\\gamma)})}{
            (\\sqrt{(b + q(x))(a + \\gamma^T\\Lambda\\gamma)})^{d/2 + 1/2}}
            e^{(x-\\mu)^T\\Lambda\\gamma}

        where:
        - :math:`q(x) = (x-\\mu)^T \\Lambda (x-\\mu)`
        - :math:`a = \\eta/\\delta`, :math:`b = \\eta\\delta`
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
        delta = classical['delta']
        eta = classical['eta']
        d = self.d

        # GIG parameters for IG(δ, η): p = -1/2, a = η/δ², b = η
        p = -0.5
        a = eta / (delta ** 2)
        b = eta

        # Precision matrix
        Lambda = np.linalg.inv(Sigma)

        # Constants
        gamma_quad = gamma @ Lambda @ gamma

        # Order of Bessel function for marginal: p - d/2 = -1/2 - d/2
        nu = p - d / 2

        # Log normalizing constant
        _, logdet_Sigma = np.linalg.slogdet(Sigma)
        
        # C = (a/b)^{p/2} * (a + γ^T Λ γ)^{d/2 - p} / ((2π)^{d/2} |Σ|^{1/2} K_p(√(ab)))
        # For NIG: √(ab) = √(η · η/δ²) = η/δ
        sqrt_ab = np.sqrt(a * b)  # = eta / delta
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
            # sqrt((b + q)(a + γ^T Λ γ))
            arg_inner = (b + q) * (a + gamma_quad)
            
            if arg_inner <= 0:
                logpdf[i] = -np.inf
                continue
                
            arg = np.sqrt(arg_inner)

            # Log Bessel K
            log_K = log_kv(nu, arg)

            # Linear term
            linear = diff @ Lambda @ gamma

            # Combine: log f(x) = log C + (d/2 - p) * log(sqrt((b+q)/(a+γ^TΛγ))) + log K_ν(arg) + linear
            # Simplify: (d/2 - p) * 0.5 * log((b+q)/(a+γ^TΛγ)) = (d/2 - p)/2 * log(b+q) - (d/2-p)/2 * log(a+γ^TΛγ)
            # The log(a+γ^TΛγ) part is already in log_C
            
            # Full formula:
            # log f(x) = log C + (d/2 - p) * 0.5 * log((b+q)/(a+γ^TΛγ)) + log K_ν(arg) + linear
            #          = log C + (d/2 - p) * 0.5 * log(b+q) - (d/2-p) * 0.5 * log(a+γ^TΛγ) + log K_ν(arg) + linear
            # But the (d/2-p)*log(a+γ^TΛγ) term is already in log_C with positive sign, so we need:
            
            half_nu_term = (0.5 * d - p) * 0.5 * np.log(b + q) - (0.5 * d - p) * 0.5 * np.log(a + gamma_quad)
            
            # Actually the correct formula is:
            # f(x) = C * (sqrt((b+q)/(a+γ^TΛγ)))^{d/2-p} * K_ν(arg) * exp(linear)
            # log f(x) = log C + (d/2-p)/2 * log(b+q) - (d/2-p)/2 * log(a+γ^TΛγ) + log K_ν + linear
            # Since log_C already has + (d/2-p) * log(a+γ^TΛγ), we need to cancel half of it
            
            # Let me recalculate more carefully:
            # From the GH marginal formula:
            # f(x) = c * K_{p-d/2}(sqrt((b+q)(a+γ^TΛγ))) / (sqrt((b+q)(a+γ^TΛγ)))^{d/2-p} * exp(linear)
            # where c = (a/b)^{p/2} * (a+γ^TΛγ)^{d/2-p} / ((2π)^{d/2} |Σ|^{1/2} K_p(sqrt(ab)))
            
            # So: log f = log c + log K_ν(arg) - (d/2-p) * log(arg) + linear
            #           = log c + log K_ν(arg) - (d/2-p) * 0.5 * log((b+q)(a+γ^TΛγ)) + linear
            #           = log c + log K_ν(arg) - (d/2-p)/2 * log(b+q) - (d/2-p)/2 * log(a+γ^TΛγ) + linear
            
            # log c = p/2 * log(a/b) + (d/2-p) * log(a+γ^TΛγ) - d/2 * log(2π) - 1/2 * log|Σ| - log K_p(η)
            
            # Combining the (a+γ^TΛγ) terms: (d/2-p) * log(...) - (d/2-p)/2 * log(...) = (d/2-p)/2 * log(...)
            
            logpdf[i] = (log_C + log_K 
                        - (0.5 * d - p) * 0.5 * np.log(b + q)
                        - (0.5 * d - p) * 0.5 * np.log(a + gamma_quad)
                        + linear)

        if single_point:
            return float(logpdf[0])

        return logpdf

    def _conditional_expectation_y_given_x(
        self, x: ArrayLike
    ) -> Dict[str, NDArray]:
        """
        Compute conditional expectations :math:`E[g(Y) | X = x]` for EM algorithm.

        For Normal Inverse Gaussian, the conditional distribution of :math:`Y | X = x` is
        a Generalized Inverse Gaussian (GIG):

        .. math::
            Y | X = x \\sim \\text{GIG}\\left(-\\frac{1}{2} - \\frac{d}{2}, \\,
            a + \\gamma^T \\Sigma^{-1} \\gamma, \\,
            b + (x-\\mu)^T \\Sigma^{-1} (x-\\mu)\\right)

        where :math:`a = \\eta/\\delta^2`, :math:`b = \\eta`, and the GIG parameters are
        :math:`(p, a, b)` in our notation.

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
        classical = self.get_classical_params()
        mu = classical['mu']
        gamma = classical['gamma']
        Sigma = classical['sigma']
        delta = classical['delta']
        eta = classical['eta']
        d = self.d

        # GIG parameters for mixing: p = -1/2, a = η/δ², b = η
        a_mix = eta / (delta ** 2)
        b_mix = eta

        # Precision matrix
        Lambda = np.linalg.inv(Sigma)

        # GIG parameters for Y | X = x
        # p_cond = p - d/2 = -1/2 - d/2
        p_cond = -0.5 - d / 2

        # a_cond = a + γ^T Λ γ (same for all x)
        gamma_quad = gamma @ Lambda @ gamma
        a_cond = a_mix + gamma_quad

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

        # b_cond = b + (x - μ)^T Λ (x - μ) for each x
        diff = x - mu  # (n, d)
        q_x = np.einsum('ni,ij,nj->n', diff, Lambda, diff)  # (n,)
        b_cond = b_mix + q_x

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

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.Generator]] = None
    ) -> 'NormalInverseGaussian':
        """
        Fit distribution to data using the EM algorithm.

        When only X is observed (Y is latent), uses the EM algorithm described
        in the theory documentation. The E-step computes conditional expectations
        of Y given X, and the M-step updates parameters using closed-form formulas.

        Parameters
        ----------
        X : array_like
            Observed X data, shape (n_samples, d) or (n_samples,) for d=1.
        y : array_like, optional
            Ignored (for sklearn API compatibility).
        max_iter : int, optional
            Maximum number of EM iterations. Default is 100.
        tol : float, optional
            Convergence tolerance for log-likelihood. Default is 1e-6.
        verbose : int, optional
            Verbosity level. 0 = silent, 1 = progress, 2 = detailed. Default is 0.
        random_state : int or Generator, optional
            Random state for initialization.

        Returns
        -------
        self : NormalInverseGaussian
            Fitted distribution (returns self for method chaining).

        Notes
        -----
        The EM algorithm iterates between:

        **E-step**: Compute conditional expectations

        .. math::
            E[Y | X = x_j], \\quad E[1/Y | X = x_j], \\quad E[\\log Y | X = x_j]

        **M-step**: Update parameters using closed-form formulas

        The Inverse Gaussian parameters :math:`(\\delta, \\eta)` are updated analytically:
        - :math:`\\delta = E[Y]`
        - :math:`\\eta = 1 / (E[1/Y] - 1/\\delta)`
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

        # For NIG: E[X] = μ + γ δ, Var[X] = δ Σ + (δ³/η) γγ^T
        # Initial guesses:
        delta_init = 1.0
        eta_init = 1.0

        # Estimate skewness to get initial gamma
        X_centered = X - X_mean
        X_std = np.std(X, axis=0)
        X_std = np.maximum(X_std, 1e-10)

        skewness = np.mean((X_centered / X_std) ** 3, axis=0)
        gamma_init = skewness * X_std * 0.1

        # μ = E[X] - γ δ
        mu_init = X_mean - gamma_init * delta_init

        # Σ from sample covariance, adjusted for gamma contribution
        Var_Y_init = delta_init**3 / eta_init
        Sigma_init = (X_cov - Var_Y_init * np.outer(gamma_init, gamma_init)) / delta_init

        # Ensure Sigma is positive definite
        Sigma_init = (Sigma_init + Sigma_init.T) / 2
        min_eig = np.linalg.eigvalsh(Sigma_init).min()
        if min_eig < 1e-6:
            Sigma_init = Sigma_init + (1e-6 - min_eig + 1e-6) * np.eye(d)

        # Set initial parameters
        self.set_classical_params(
            mu=mu_init,
            gamma=gamma_init,
            sigma=Sigma_init,
            delta=delta_init,
            eta=eta_init
        )

        # Compute initial log-likelihood
        prev_ll = np.mean(self.logpdf(X))

        if verbose >= 1:
            print(f"Initial log-likelihood: {prev_ll:.6f}")

        # ================================================================
        # EM iterations
        # ================================================================
        for iteration in range(max_iter):
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
            eta_6 = np.zeros((d, d))
            for j in range(n_samples):
                eta_6 += np.outer(X[j], X[j]) * E_inv_Y[j]
            eta_6 /= n_samples

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

            # Symmetrize
            Sigma_new = (Sigma_new + Sigma_new.T) / 2

            # Ensure positive definiteness
            min_eig = np.linalg.eigvalsh(Sigma_new).min()
            if min_eig < 1e-8:
                Sigma_new = Sigma_new + (1e-8 - min_eig + 1e-8) * np.eye(d)

            # Inverse Gaussian parameters
            # For IG: E[Y] = δ, E[1/Y] = 1/δ + 1/η
            # δ = η̂₂ = E[Y]
            delta_new = eta_2

            # η = 1 / (E[1/Y] - 1/δ) = 1 / (η̂₁ - 1/η̂₂)
            inv_eta = eta_1 - 1.0 / delta_new
            if inv_eta > 1e-10:
                eta_new = 1.0 / inv_eta
            else:
                # Edge case: large η
                eta_new = 1000.0

            # Bound parameters
            delta_new = max(delta_new, 1e-6)
            eta_new = max(eta_new, 1e-6)

            # ==============================================================
            # Update parameters
            # ==============================================================
            old_params = self.get_classical_params()

            try:
                self.set_classical_params(
                    mu=mu_new,
                    gamma=gamma_new,
                    sigma=Sigma_new,
                    delta=delta_new,
                    eta=eta_new
                )
            except ValueError as e:
                if verbose >= 1:
                    print(f"Warning: parameter update failed at iteration {iteration}: {e}")
                self.set_classical_params(**old_params)
                break

            # ==============================================================
            # Check convergence
            # ==============================================================
            current_ll = np.mean(self.logpdf(X))

            if verbose >= 2:
                print(f"Iteration {iteration + 1}: log-likelihood = {current_ll:.6f}")

            # Check for convergence
            ll_change = current_ll - prev_ll
            if abs(ll_change) < tol:
                if verbose >= 1:
                    print(f"Converged at iteration {iteration + 1}")
                break

            # Check for decrease (shouldn't happen in EM, but numerical issues)
            if ll_change < -1e-8:
                if verbose >= 1:
                    print(f"Warning: log-likelihood decreased at iteration {iteration + 1}")

            prev_ll = current_ll

        if verbose >= 1:
            print(f"Final log-likelihood: {prev_ll:.6f}")

        return self

    def fit_complete(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        **kwargs
    ) -> 'NormalInverseGaussian':
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
        self : NormalInverseGaussian
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

        return self

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if self._joint is None:
            return "NormalInverseGaussian(not fitted)"

        try:
            classical = self.get_classical_params()
            d = self.d
            delta = classical['delta']
            eta = classical['eta']

            if d == 1:
                mu = float(classical['mu'][0])
                gamma_val = float(classical['gamma'][0])
                return f"NormalInverseGaussian(μ={mu:.3f}, γ={gamma_val:.3f}, δ={delta:.3f}, η={eta:.3f})"
            else:
                return f"NormalInverseGaussian(d={d}, δ={delta:.3f}, η={eta:.3f})"
        except ValueError:
            return "NormalInverseGaussian(not fitted)"
