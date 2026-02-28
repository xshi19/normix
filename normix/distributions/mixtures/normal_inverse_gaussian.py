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
from normix.utils import log_kv, robust_cholesky
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
        mu = self._joint._mu
        gamma = self._joint._gamma
        delta = self._joint._delta
        eta = self._joint._eta
        d = self.d

        p = -0.5
        a = eta / (delta ** 2)
        b = eta

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

        # Order of Bessel function for marginal: p - d/2 = -1/2 - d/2
        nu = p - d / 2

        # Constants
        sqrt_ab = np.sqrt(a * b)  # = eta / delta
        log_K_p = log_kv(p, sqrt_ab)

        # Arguments for Bessel function (vectorized)
        arg1 = b + q  # (n,)
        arg2 = a + gamma_quad  # scalar
        eta_arg = np.sqrt(arg1 * arg2)  # (n,)

        # Log Bessel K_{p-d/2}(eta) - vectorized
        log_K_nu = log_kv(nu, eta_arg)  # (n,)

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

        Since the Inverse Gaussian sufficient statistics are :math:`t_Y(y) = (y, y^{-1})`,
        only :math:`E[Y | X]` and :math:`E[1/Y | X]` are needed for the M-step.
        :math:`E[\\log Y | X]` is not required and is omitted to avoid unnecessary
        Bessel function evaluations.

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
        """
        x = np.asarray(x)
        mu = self._joint._mu
        gamma = self._joint._gamma
        delta = self._joint._delta
        eta = self._joint._eta
        d = self.d

        a_mix = eta / (delta ** 2)
        b_mix = eta

        L_inv = self._joint.L_Sigma_inv

        # GIG parameters for Y | X = x
        # p_cond = p - d/2 = -1/2 - d/2
        p_cond = -0.5 - d / 2

        # a_cond = a + γ^T Σ^{-1} γ (same for all x)
        gamma_z = L_inv @ gamma  # (d,)
        gamma_quad = np.dot(gamma_z, gamma_z)  # γ^T Σ^{-1} γ
        a_cond = a_mix + gamma_quad

        # Handle single point vs multiple points
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        n = x.shape[0]

        # b_cond = b + (x - μ)^T Σ^{-1} (x - μ) for each x
        diff = x - mu  # (n, d)
        z = L_inv @ diff.T  # (d, n)
        q_x = np.sum(z ** 2, axis=0)  # (n,)
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

        if single_point:
            return {
                'E_Y': float(E_Y[0]),
                'E_inv_Y': float(E_inv_Y[0]),
            }

        return {
            'E_Y': E_Y,
            'E_inv_Y': E_inv_Y,
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
        Initialize NIG parameters using method of moments.

        Estimates initial values of :math:`(\\mu, \\gamma, \\Sigma, \\delta, \\eta)`
        from sample statistics.

        Parameters
        ----------
        X : ndarray
            Data array, shape (n_samples, d).
        random_state : int or Generator, optional
            Random state (reserved for future use).
        """
        n, d = X.shape

        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X, rowvar=False)
        if X_cov.ndim == 0:
            X_cov = np.array([[X_cov]])

        delta_init = 1.0
        eta_init = 1.0

        gamma_init = np.zeros(d)
        mu_init = X_mean
        Sigma_init = X_cov / delta_init

        L = robust_cholesky(Sigma_init, eps=1e-6)
        Sigma_init = L @ L.T

        self.set_classical_params(
            mu=mu_init,
            gamma=gamma_init,
            sigma=Sigma_init,
            delta=delta_init,
            eta=eta_init
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
            Dictionary with keys ``'E_Y'``, ``'E_inv_Y'``.
        verbose : int, optional
            Verbosity level for diagnostics.
        """
        from normix.distributions.univariate import InverseGaussian

        n, d = X.shape

        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']

        s1 = np.mean(E_inv_Y)
        s2 = np.mean(E_Y)
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

        # InverseGaussian parameters via set_expectation_params
        # IG expectation params: [delta, 1/delta + 1/eta] = [E[Y], E[1/Y]]
        ig_dist = InverseGaussian()
        ig_eta = np.array([s2, s1])
        ig_dist.set_expectation_params(ig_eta)

        if verbose >= 1:
            recovered = ig_dist._compute_expectation_params()
            eta_diff = np.max(np.abs(recovered - ig_eta))
            if eta_diff > 1e-6:
                print(f"  Warning: InverseGaussian expectation param roundtrip error = {eta_diff:.2e}")

        delta_new = ig_dist.classical_params.delta
        eta_new = ig_dist.classical_params.eta

        # Bound parameters
        delta_new = max(delta_new, 1e-6)
        eta_new = max(eta_new, 1e-6)

        self._joint._set_internal(
            mu=mu, gamma=gamma, L_sigma=L_Sigma,
            delta=delta_new, eta=eta_new
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
    ) -> 'NormalInverseGaussian':
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
        self : NormalInverseGaussian
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
        self._fitted = self._joint._fitted
        self._invalidate_cache()

        return self

    # ========================================================================
    # String representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if not self._fitted:
            return "NormalInverseGaussian(not fitted)"

        try:
            classical = self.classical_params
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
