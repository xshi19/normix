"""
Generalized Inverse Gaussian (GIG) distribution.

The GIG distribution belongs to the exponential family and serves as a
unifying framework for many important distributions.

The GIG distribution has PDF (Wikipedia parameterization):

.. math::
    f(x|p,a,b) = \\frac{(a/b)^{p/2}}{2 K_p(\\sqrt{ab})} x^{p-1} 
    \\exp\\left(-\\frac{ax + b/x}{2}\\right)

for :math:`x > 0`, where:

- :math:`p`: real-valued shape parameter (any real number)
- :math:`a > 0`: rate parameter (coefficient of :math:`x`)
- :math:`b > 0`: rate parameter (coefficient of :math:`1/x`)
- :math:`K_p` is the modified Bessel function of the second kind

Exponential family form:

- :math:`h(x) = 1` for :math:`x > 0` (base measure)
- :math:`t(x) = [\\log x, 1/x, x]` (sufficient statistics)
- :math:`\\theta = [p-1, -b/2, -a/2]` (natural parameters)
- :math:`\\psi(\\theta) = \\log(2) + \\log K_p(\\sqrt{ab}) + \\frac{p}{2}\\log(b/a)` (log partition)

Natural parameters: :math:`\\theta = (\\theta_1, \\theta_2, \\theta_3)`:

- :math:`\\theta_1 = p - 1` (unbounded)
- :math:`\\theta_2 = -b/2 < 0`
- :math:`\\theta_3 = -a/2 < 0`

Scipy parametrization (``scipy.stats.geninvgauss``):

- scipy uses :math:`(p, b_{scipy}, \\text{scale})` where 
  :math:`b_{scipy} = \\sqrt{ab}` and :math:`\\text{scale} = \\sqrt{b/a}`

Special cases:

- **Gamma**: :math:`b \\to 0` gives :math:`\\text{Gamma}(p, a/2)` for :math:`p > 0`
- **Inverse Gamma**: :math:`a \\to 0` gives :math:`\\text{InvGamma}(-p, b/2)` for :math:`p < 0`
- **Inverse Gaussian**: :math:`p = -1/2`

References
----------
https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Union, List
from scipy.interpolate import interp1d

from normix.params import GIGParams
from normix.base import ExponentialFamily
from normix.utils import log_kv, log_kv_derivative_z

# When √(ab) falls below this threshold, switch to Gamma/InvGamma limit
# formulas to avoid ∞−∞ and ∞×0 in the Bessel-based expressions.
_DEGEN_THRESHOLD = 1e-10

# Module-level helpers for degenerate-case computations.
# Import lazily to avoid circular imports at module load time.
_gamma_helper = None
_invgamma_helper = None


def _get_gamma():
    global _gamma_helper
    if _gamma_helper is None:
        from normix.distributions.univariate.gamma import Gamma
        _gamma_helper = Gamma()
    return _gamma_helper


def _get_invgamma():
    global _invgamma_helper
    if _invgamma_helper is None:
        from normix.distributions.univariate.inverse_gamma import InverseGamma
        _invgamma_helper = InverseGamma()
    return _invgamma_helper


class GeneralizedInverseGaussian(ExponentialFamily):
    """
    Generalized Inverse Gaussian (GIG) distribution in exponential family form.
    
    The GIG distribution has PDF (Wikipedia parameterization):
    
    .. math::
        f(x|p,a,b) = \\frac{(a/b)^{p/2}}{2 K_p(\\sqrt{ab})} x^{p-1} 
        \\exp\\left(-\\frac{ax + b/x}{2}\\right)
    
    for :math:`x > 0`, where :math:`K_p` is the modified Bessel function 
    of the second kind.
    
    Parameters
    ----------
    p : float, optional
        Shape parameter (any real number). Use ``from_classical_params(p=..., a=..., b=...)``.
    a : float, optional
        Rate parameter :math:`a > 0` (coefficient of :math:`x`).
    b : float, optional
        Rate parameter :math:`b > 0` (coefficient of :math:`1/x`).
    
    Attributes
    ----------
    _natural_params : tuple or None
        Internal storage for natural parameters :math:`\\theta = [p-1, -b/2, -a/2]`.
    
    Examples
    --------
    >>> # Create from classical parameters
    >>> dist = GeneralizedInverseGaussian.from_classical_params(p=1.0, a=1.0, b=1.0)
    >>> dist.mean()
    
    >>> # Create from natural parameters
    >>> dist = GeneralizedInverseGaussian.from_natural_params(np.array([0.0, -0.5, -0.5]))
    
    >>> # Fit from data
    >>> from scipy.stats import geninvgauss
    >>> data = geninvgauss.rvs(p=1.0, b=1.0, size=1000)
    >>> dist = GeneralizedInverseGaussian().fit(data)
    
    See Also
    --------
    Gamma : Special case when :math:`b \\to 0` for :math:`p > 0`
    InverseGamma : Special case when :math:`a \\to 0` for :math:`p < 0`
    InverseGaussian : Special case when :math:`p = -1/2`
    
    Notes
    -----
    The GIG distribution belongs to the exponential family with:
    
    - Sufficient statistics: :math:`t(x) = [\\log x, 1/x, x]`
    - Natural parameters: :math:`\\theta = [p-1, -b/2, -a/2]`
    - Log partition: :math:`\\psi(\\theta) = \\log(2) + \\log K_p(\\sqrt{ab}) + \\frac{p}{2}\\log(b/a)`
    
    Moments are given by:
    
    .. math::
        E[X^\\alpha] = \\left(\\frac{b}{a}\\right)^{\\alpha/2} 
        \\frac{K_{p+\\alpha}(\\sqrt{ab})}{K_p(\\sqrt{ab})}
    
    Special cases:
    
    - :math:`b \\to 0, p > 0`: :math:`\\text{GIG} \\to \\text{Gamma}(p, a/2)`
    - :math:`a \\to 0, p < 0`: :math:`\\text{GIG} \\to \\text{InvGamma}(-p, b/2)`
    - :math:`p = -1/2`: :math:`\\text{GIG} \\to \\text{InverseGaussian}`
    
    References
    ----------
    Barndorff-Nielsen, O. E. (1978). Information and exponential families.

    https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution
    """
    
    def __init__(self):
        super().__init__()
        self._p = None
        self._a = None
        self._b = None
    
    # ================================================================
    # New interface: internal state management
    # ================================================================
    
    def _set_from_classical(self, *, p, a, b) -> None:
        """Set internal state from classical parameters.

        Allows ``a = 0`` (Inverse-Gamma limit, requires ``p < 0``) or
        ``b = 0`` (Gamma limit, requires ``p > 0``).
        """
        if a < 0:
            raise ValueError(f"Parameter 'a' must be non-negative, got {a}")
        if b < 0:
            raise ValueError(f"Parameter 'b' must be non-negative, got {b}")
        if a == 0 and b == 0:
            raise ValueError("Parameters 'a' and 'b' cannot both be zero")
        if a == 0 and p >= 0:
            raise ValueError(
                f"Inverse-Gamma limit (a=0) requires p < 0, got p={p}"
            )
        if b == 0 and p <= 0:
            raise ValueError(
                f"Gamma limit (b=0) requires p > 0, got p={p}"
            )
        self._p = float(p)
        self._a = float(a)
        self._b = float(b)
        self._fitted = True
        self._invalidate_cache()
    
    def _set_from_natural(self, theta) -> None:
        """Set internal state from natural parameters.

        Allows ``θ₂ = 0`` (Inverse-Gamma limit) or ``θ₃ = 0``
        (Gamma limit).  Delegates to ``_set_from_classical`` which
        validates the degenerate-case constraints.
        """
        theta = np.asarray(theta)
        p = float(theta[0] + 1)
        b = float(max(-2 * theta[1], 0.0))
        a = float(max(-2 * theta[2], 0.0))
        self._set_from_classical(p=p, a=a, b=b)
    
    def _compute_natural_params(self):
        """Compute natural parameters from internal state: θ = [p-1, -b/2, -a/2]."""
        return np.array([self._p - 1, -self._b / 2, -self._a / 2])
    
    def _compute_classical_params(self):
        """Compute classical parameters from internal state."""
        return GIGParams(p=self._p, a=self._a, b=self._b)
    
    def _get_natural_param_support(self):
        """
        Natural parameter support (boundaries included for Gamma/InvGamma
        limits).

        - :math:`\\theta_1 = p - 1`: unbounded
        - :math:`\\theta_2 = -b/2 \\le 0`
        - :math:`\\theta_3 = -a/2 \\le 0`

        The base-class validator uses *strict* inequality, so we set the
        upper bound to a tiny positive epsilon to effectively include 0.
        """
        _EPS = 1e-30
        return [(-np.inf, np.inf), (-np.inf, _EPS), (-np.inf, _EPS)]
    
    def _sufficient_statistics(self, x: ArrayLike) -> NDArray:
        """
        Sufficient statistics: t(x) = [log(x), 1/x, x].
        
        Returns
        -------
        t : ndarray
            Shape (3,) for scalar input, (n, 3) for array input.
        """
        x = np.asarray(x)
        if x.ndim == 0 or x.shape == ():
            return np.array([np.log(x), 1.0/x, x])
        else:
            return np.column_stack([np.log(x), 1.0/x, x])
    
    def _log_partition(self, theta: NDArray) -> float:
        r"""
        Log partition function.

        .. math::
            A(\theta) = \log 2 + \log K_p(\sqrt{ab})
                        + \tfrac{p}{2}\bigl(\log b - \log a\bigr)

        Falls back to closed-form Gamma / Inverse-Gamma limits when
        :math:`\sqrt{ab}` is below ``_DEGEN_THRESHOLD``.
        """
        p = theta[0] + 1
        b = max(-2 * theta[1], 0.0)
        a = max(-2 * theta[2], 0.0)
        sqrt_ab = np.sqrt(a * b)

        if sqrt_ab < _DEGEN_THRESHOLD:
            if a <= b:
                # a ≈ 0  →  InvGamma(α=-p, β=b/2), θ_IG = [β, -(α+1)]
                return _get_invgamma()._log_partition(np.array([b / 2, p - 1]))
            else:
                # b ≈ 0  →  Gamma(α=p, β=a/2), θ_Γ = [α-1, -β]
                return _get_gamma()._log_partition(np.array([p - 1, -a / 2]))

        return np.log(2) + log_kv(p, sqrt_ab) + (p / 2) * (np.log(b) - np.log(a))
    
    def _log_base_measure(self, x: ArrayLike) -> NDArray:
        """
        Log base measure: log h(x) = 0 for x > 0, -∞ otherwise.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)
        result[x <= 0] = -np.inf
        return result
    
    def _natural_to_expectation(self, theta: NDArray) -> NDArray:
        r"""
        Convert natural parameters to expectation parameters: η = ∇A(θ).

        .. math::
            \eta = [E[\log X],\; E[1/X],\; E[X]]

        Uses Bessel-ratio formulas in log-space for the general case,
        with closed-form Inverse-Gamma / Gamma limits when
        :math:`\sqrt{ab}` falls below ``_DEGEN_THRESHOLD``.
        """
        p = theta[0] + 1
        b = max(-2 * theta[1], 0.0)
        a = max(-2 * theta[2], 0.0)
        sqrt_ab = np.sqrt(a * b)

        # ---------- degenerate limits -----------------------------------
        if sqrt_ab < _DEGEN_THRESHOLD:
            if a <= b:
                # a ≈ 0  →  InvGamma(α=-p, β=b/2), θ_IG = [β, -(α+1)]
                ig_theta = np.array([b / 2, p - 1])
                ig_eta = _get_invgamma()._natural_to_expectation(ig_theta)
                # ig_eta = [-α/β, log(β)-ψ(α)]
                E_log_x = ig_eta[1]
                E_inv_x = -ig_eta[0]
                alpha_ig = -p
                E_x = (b / 2) / (alpha_ig - 1) if alpha_ig > 1 else np.inf
            else:
                # b ≈ 0  →  Gamma(α=p, β=a/2), θ_Γ = [α-1, -β]
                g_theta = np.array([p - 1, -a / 2])
                g_eta = _get_gamma()._natural_to_expectation(g_theta)
                # g_eta = [ψ(α)-log(β), α/β]
                E_log_x = g_eta[0]
                E_x = g_eta[1]
                E_inv_x = (a / 2) / (p - 1) if p > 1 else np.inf
            return np.array([E_log_x, E_inv_x, E_x])

        # ---------- general Bessel case --------------------------------
        log_kv_p = log_kv(p, sqrt_ab)
        log_kv_pm1 = log_kv(p - 1, sqrt_ab)
        log_kv_pp1 = log_kv(p + 1, sqrt_ab)

        log_sqrt_ba = 0.5 * (np.log(b) - np.log(a))

        E_inv_x = np.exp(log_kv_pm1 - log_kv_p - log_sqrt_ba)
        E_x = np.exp(log_kv_pp1 - log_kv_p + log_sqrt_ba)

        eps = 1e-6
        d_log_kv_dp = (log_kv(p + eps, sqrt_ab) - log_kv(p - eps, sqrt_ab)) / (2 * eps)
        E_log_x = d_log_kv_dp + log_sqrt_ba

        return np.array([E_log_x, E_inv_x, E_x])
    
    def _get_initial_natural_params(self, eta: NDArray) -> List[NDArray]:
        """
        Get initial guesses for natural parameters from expectation parameters.

        Fits the three special-case sub-distributions (Inverse Gaussian,
        Gamma, Inverse Gamma) to the relevant subsets of ``eta`` and
        converts their parameters to GIG natural parameters.  Each
        solution is also perturbed to provide additional starting points
        away from the exact boundary.
        """
        from normix.distributions.univariate.inverse_gaussian import InverseGaussian

        E_log_x, E_inv_x, E_x = eta
        starting_points: List[NDArray] = []
        eps = 1e-4

        # ── 1. Inverse Gaussian (p = -1/2): match E[X] and E[1/X] ────
        #    IG θ = [-η/(2δ²), -η/2]
        #    → GIG θ = [-3/2, ig_θ₂, ig_θ₁]
        if E_x > 0 and E_inv_x > 1.0 / E_x:
            try:
                ig_eta = np.array([E_x, E_inv_x])
                ig_theta = InverseGaussian()._expectation_to_natural(ig_eta)
                starting_points.append(
                    np.array([-1.5, ig_theta[1], ig_theta[0]])
                )
            except (ValueError, FloatingPointError):
                pass

        # ── 2. Gamma (b ≈ 0): match E[log X] and E[X] ───────────────
        #    Γ θ = [α-1, -β]
        #    → GIG θ = [γ_θ₁, -ε/2, γ_θ₂]
        if E_x > 0:
            try:
                g_eta = np.array([E_log_x, E_x])
                g_theta = _get_gamma()._expectation_to_natural(g_eta)
                starting_points.append(
                    np.array([g_theta[0], -eps / 2, g_theta[1]])
                )
            except (ValueError, FloatingPointError):
                pass

        # ── 3. Inverse Gamma (a ≈ 0): match E[1/X] and E[log X] ─────
        #    IG θ = [β, -(α+1)]
        #    → GIG θ = [ig_θ₂, -ig_θ₁, -ε/2]
        if E_inv_x > 0:
            try:
                ig_eta = np.array([-E_inv_x, E_log_x])
                ig_theta = _get_invgamma()._expectation_to_natural(ig_eta)
                starting_points.append(
                    np.array([ig_theta[1], -ig_theta[0], -eps / 2])
                )
            except (ValueError, FloatingPointError):
                pass

        # ── 4. Perturbed copies — move boundary params toward interior ─
        for sp in list(starting_points):
            for scale in [0.1, 0.5, 2.0, 10.0]:
                perturbed = sp.copy()
                perturbed[1] = min(perturbed[1], -eps * scale / 2)
                perturbed[2] = min(perturbed[2], -eps * scale / 2)
                starting_points.append(perturbed)

        # ── 5. Fallback ──────────────────────────────────────────────
        if not starting_points:
            starting_points.append(np.array([0.0, -0.5, -0.5]))

        return starting_points
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Fisher information matrix I(θ) = ∇²A(θ).
        
        Uses numerical differentiation since the analytical form is complex.
        """
        if theta is None:
            theta = self.natural_params
        return super().fisher_information(theta)
    
    # ============================================================
    # Conversion methods for scipy.stats.geninvgauss
    # ============================================================
    
    def to_scipy_params(self):
        """
        Convert to scipy.stats.geninvgauss parameters (p, b, scale).
        
        scipy parameterization:
            f(x|p,b,scale) ∝ (x/scale)^{p-1} exp(-b((x/scale) + scale/x)/2)
        
        Conversion:
            p_scipy = p
            b_scipy = √(ab)
            scale = √(b/a)
        
        Returns
        -------
        params : dict
            Dictionary with keys 'p', 'b', 'scale'.
        """
        self._check_fitted()
        
        return {
            'p': self._p,
            'b': np.sqrt(self._a * self._b),
            'scale': np.sqrt(self._b / self._a)
        }
    
    @classmethod
    def from_scipy_params(cls, p: float, b: float, scale: float = 1.0) -> 'GeneralizedInverseGaussian':
        """
        Create from scipy.stats.geninvgauss parameters.
        
        Conversion:
            p = p_scipy
            a = b_scipy / scale
            b = b_scipy * scale
        
        Parameters
        ----------
        p : float
            Shape parameter.
        b : float
            Shape parameter b > 0 (this is √(ab) in our notation).
        scale : float, optional
            Scale parameter > 0. Default is 1.0.
        
        Returns
        -------
        dist : GeneralizedInverseGaussian
            Distribution instance.
        """
        a_ours = b / scale
        b_ours = b * scale
        return cls.from_classical_params(p=p, a=a_ours, b=b_ours)
    
    # ============================================================
    # Moments using Bessel function ratios
    # ============================================================
    
    def moment_alpha(self, alpha: float) -> float:
        """
        Compute the α-th moment E[X^α].
        
        E[X^α] = (b/a)^(α/2) · K_{p+α}(√(ab)) / K_p(√(ab))
        
        Parameters
        ----------
        alpha : float
            Order of the moment.
        
        Returns
        -------
        moment : float
            The α-th moment.
        """
        self._check_fitted()
        
        sqrt_b_over_a = np.sqrt(self._b / self._a)
        sqrt_ab = np.sqrt(self._a * self._b)
        
        log_ratio = log_kv(self._p + alpha, sqrt_ab) - log_kv(self._p, sqrt_ab)
        
        return (sqrt_b_over_a ** alpha) * np.exp(log_ratio)
    
    def mean(self) -> float:
        """
        Mean of GIG distribution using Bessel function ratio.
        
        .. math::
            E[X] = \\sqrt{\\frac{b}{a}} \\frac{K_{p+1}(\\sqrt{ab})}{K_p(\\sqrt{ab})}
        
        Returns
        -------
        mean : float
            Mean of the distribution.
        """
        return self.moment_alpha(1.0)
    
    def var(self) -> float:
        """
        Variance of GIG distribution: Var[X] = E[X^2] - E[X]^2.
        
        .. math::
            \\text{Var}[X] = E[X^2] - (E[X])^2
        
        Uses the moment formula :math:`E[X^\\alpha] = (b/a)^{\\alpha/2} K_{p+\\alpha}(\\sqrt{ab})/K_p(\\sqrt{ab})`.
        
        Returns
        -------
        var : float
            Variance of the distribution.
        """
        return self.moment_alpha(2.0) - self.moment_alpha(1.0) ** 2
    
    # ============================================================
    # Random variate generation
    # ============================================================
    
    def rvs(self, size=None, random_state=None, method='scipy'):
        """
        Generate random samples from the GIG distribution.
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Shape of samples to generate.
        random_state : int or Generator, optional
            Random number generator seed or instance.
        method : str, optional
            Sampling method: 'scipy' uses scipy.stats.geninvgauss (default),
            'naive' uses inverse CDF method.
        
        Returns
        -------
        samples : float or ndarray
            Random samples from the distribution.
        """
        self._check_fitted()
        
        if random_state is None:
            rng = np.random.default_rng()
        elif isinstance(random_state, int):
            rng = np.random.default_rng(random_state)
        else:
            rng = random_state
        
        if method == 'scipy':
            return self._rvs_scipy(size, rng)
        elif method == 'naive':
            return self._rvs_naive(size, rng)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'scipy' or 'naive'.")
    
    def _rvs_scipy(self, size, rng):
        """Generate samples using scipy.stats.geninvgauss."""
        from scipy.stats import geninvgauss
        
        params = self.to_scipy_params()
        return geninvgauss.rvs(p=params['p'], b=params['b'], scale=params['scale'],
                               size=size, random_state=rng)
    
    def _rvs_naive(self, size, rng, n_grids: int = 100000):
        """
        Generate samples using naive inverse CDF method.
        """
        mu = self.mean()
        
        z1 = np.linspace(0., mu, n_grids + 1)[1:]
        z2 = np.linspace(0., 1. / mu, n_grids + 1)[1:][::-1] ** (-1)
        z = np.concatenate([z1, z2])
        
        dz = np.diff(z)
        z_mid = z[1:] - dz / 2
        
        pdf_vals = self.pdf(z_mid)
        pdf_vals = np.maximum(pdf_vals, 0)
        
        y = np.cumsum(pdf_vals * dz)
        y = y / y[-1]
        y = np.clip(y, 0, 1)
        
        y_unique, idx = np.unique(y, return_index=True)
        z_unique = z_mid[idx]
        
        inv_cdf = interp1d(y_unique, z_unique, kind='linear',
                          bounds_error=False, fill_value=(z_unique[0], z_unique[-1]))
        
        if size is None:
            u = rng.random()
        else:
            u = rng.random(size)
        
        return inv_cdf(u)
    
    def cdf(self, x: ArrayLike) -> NDArray:
        """
        Cumulative distribution function.
        
        Uses scipy.stats.geninvgauss.
        """
        self._check_fitted()
        
        from scipy.stats import geninvgauss
        
        x = np.asarray(x)
        params = self.to_scipy_params()
        
        result = geninvgauss.cdf(x, p=params['p'], b=params['b'], scale=params['scale'])
        
        if np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return float(result)
        return result
    
    def ppf(self, q: ArrayLike) -> NDArray:
        """
        Percent point function (inverse of CDF).
        
        Uses scipy.stats.geninvgauss.
        """
        self._check_fitted()
        
        from scipy.stats import geninvgauss
        
        q = np.asarray(q)
        params = self.to_scipy_params()
        
        result = geninvgauss.ppf(q, p=params['p'], b=params['b'], scale=params['scale'])
        
        if np.isscalar(q) or (hasattr(q, 'shape') and q.shape == ()):
            return float(result)
        return result


# Alias for convenience
GIG = GeneralizedInverseGaussian
