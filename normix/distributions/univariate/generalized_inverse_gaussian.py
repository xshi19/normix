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
from scipy.special import digamma

from normix.base import ExponentialFamily
from normix.utils import log_kv, log_kv_derivative_z


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
        """Set internal state from classical parameters."""
        if a <= 0:
            raise ValueError(f"Parameter 'a' must be positive, got {a}")
        if b <= 0:
            raise ValueError(f"Parameter 'b' must be positive, got {b}")
        self._p = float(p)
        self._a = float(a)
        self._b = float(b)
        self._natural_params = tuple(np.array([p - 1, -b / 2, -a / 2]))
        self._fitted = True
        self._invalidate_cache()
    
    def _set_from_natural(self, theta) -> None:
        """Set internal state from natural parameters."""
        theta = np.asarray(theta)
        self._validate_natural_params(theta)
        self._p = float(theta[0] + 1)
        self._b = float(-2 * theta[1])
        self._a = float(-2 * theta[2])
        self._natural_params = tuple(theta)
        self._fitted = True
        self._invalidate_cache()
    
    def _compute_natural_params(self):
        """Compute natural parameters from internal state: θ = [p-1, -b/2, -a/2]."""
        return np.array([self._p - 1, -self._b / 2, -self._a / 2])
    
    def _compute_classical_params(self):
        """Compute classical parameters from internal state."""
        return {'p': self._p, 'a': self._a, 'b': self._b}
    
    def _get_natural_param_support(self):
        """
        Natural parameter support.
        
        θ₁ = p - 1: unbounded
        θ₂ = -b/2: < 0 (since b > 0)
        θ₃ = -a/2: < 0 (since a > 0)
        """
        return [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
    
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
        """
        Log partition function: A(θ) = log(2) + log(K_p(√(ab))) + (p/2)log(b/a).
        
        where:
            p = θ₁ + 1
            b = -2θ₂
            a = -2θ₃
        """
        p = theta[0] + 1
        b = -2 * theta[1]
        a = -2 * theta[2]
        
        sqrt_ab = np.sqrt(a * b)
        
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
        """
        Convert natural parameters to expectation parameters: η = ∇A(θ).
        
        η = [E[log X], E[1/X], E[X]]
        
        Using analytical formulas from Wikipedia:
        - E[X] = √(b/a) · K_{p+1}(√(ab)) / K_p(√(ab))
        - E[1/X] = √(a/b) · K_{p-1}(√(ab)) / K_p(√(ab)) - 2p/b  [Note: Wikipedia formula]
        - E[log X] = log(√(b/a)) + ∂/∂p log(K_p(√(ab)))
        """
        p = theta[0] + 1
        b = -2 * theta[1]
        a = -2 * theta[2]
        sqrt_ab = np.sqrt(a * b)
        sqrt_b_over_a = np.sqrt(b / a)
        
        log_kv_p = log_kv(p, sqrt_ab)
        log_kv_pm1 = log_kv(p - 1, sqrt_ab)
        log_kv_pp1 = log_kv(p + 1, sqrt_ab)
        
        # E[1/X] = √(a/b) · K_{p-1}(√(ab)) / K_p(√(ab))
        # Note: Using the recurrence relation form which is more stable
        E_inv_x = np.exp(log_kv_pm1 - log_kv_p) / sqrt_b_over_a
        
        # E[X] = √(b/a) · K_{p+1}(√(ab)) / K_p(√(ab))
        E_x = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)
        
        # E[log X] = ∂A/∂θ₁ = ∂/∂p[log K_p(√(ab))] + (1/2)(log b - log a)
        eps = 1e-6
        d_log_kv_dp = (log_kv(p + eps, sqrt_ab) - log_kv(p - eps, sqrt_ab)) / (2 * eps)
        E_log_x = d_log_kv_dp + 0.5 * np.log(b / a)
        
        return np.array([E_log_x, E_inv_x, E_x])
    
    def _get_initial_natural_params(self, eta: NDArray) -> List[NDArray]:
        """
        Get initial guesses for natural parameters from expectation parameters.
        
        Uses the three special cases of GIG distribution:
        1. Inverse Gaussian (p = -1/2): uses E[X] and E[1/X]
        2. Gamma (b → small): uses E[log X] and E[X]
        3. Inverse Gamma (a → small): uses E[log X] and E[1/X]
        
        Returns multiple starting points for multi-start optimization.
        """
        E_log_x, E_inv_x, E_x = eta
        starting_points = []
        
        # Small value for nearly-degenerate cases
        eps = 1e-6
        
        # === Special case 1: Inverse Gaussian (p = -1/2) ===
        # For IG: E[X] = μ, E[1/X] = 1/μ + 1/(μ²λ)
        # With GIG params: a = λ/μ, b = λμ, p = -1/2
        # E[X] = √(b/a), so b/a = E[X]²
        # E[1/X] = √(a/b) · K_{-3/2}(√(ab)) / K_{-1/2}(√(ab))
        if E_x > 0 and E_inv_x > 0:
            p_ig = -0.5
            # Approximate: for IG, √(b/a) ≈ E[X]
            sqrt_b_over_a = E_x
            # Product ab from the ratio E[1/X]/E[X] ≈ a/b for moderate √(ab)
            # Simple heuristic: set √(ab) = 1
            sqrt_ab = max(1.0, np.sqrt(E_x * E_inv_x))
            a_ig = sqrt_ab / sqrt_b_over_a
            b_ig = sqrt_ab * sqrt_b_over_a
            a_ig = max(a_ig, eps)
            b_ig = max(b_ig, eps)
            starting_points.append(np.array([p_ig - 1, -b_ig / 2, -a_ig / 2]))
        
        # === Special case 2: Gamma (b → small) ===
        # Gamma(α, β) with α = p, β = a/2, b → 0
        # E[X] = α/β = 2p/a
        # E[log X] = ψ(α) - log(β) = ψ(p) - log(a/2)
        # From E[X]: a = 2p/E[X]
        # From E[log X]: p satisfies ψ(p) = E[log X] + log(a/2)
        if E_x > 0:
            # Try a few values of p > 0 for Gamma
            for p_gamma in [0.5, 1.0, 2.0, 5.0]:
                a_gamma = 2 * p_gamma / E_x
                a_gamma = max(a_gamma, eps)
                b_gamma = eps  # Small b for Gamma limit
                starting_points.append(np.array([p_gamma - 1, -b_gamma / 2, -a_gamma / 2]))
        
        # === Special case 3: Inverse Gamma (a → small) ===
        # InvGamma(α, β) with α = -p, β = b/2, a → 0
        # E[1/X] = α/β = -2p/b (for p < 0)
        # E[log X] = log(β) - ψ(α) = log(b/2) - ψ(-p)
        # From E[1/X]: b = -2p/E[1/X]
        if E_inv_x > 0:
            # Try a few values of p < 0 for Inverse Gamma
            for p_invgamma in [-0.5, -1.0, -2.0, -5.0]:
                b_invgamma = -2 * p_invgamma / E_inv_x
                b_invgamma = max(b_invgamma, eps)
                a_invgamma = eps  # Small a for Inverse Gamma limit
                starting_points.append(np.array([p_invgamma - 1, -b_invgamma / 2, -a_invgamma / 2]))
        
        # === General heuristic: moment matching ===
        # E[X]/E[1/X] ≈ b/a for moderate √(ab)
        if E_x > 0 and E_inv_x > 0:
            ratio = E_x / E_inv_x  # ≈ b/a
            sqrt_b_over_a = np.sqrt(max(ratio, eps))
            
            for sqrt_ab in [0.5, 1.0, 2.0, 5.0]:
                a_heur = sqrt_ab / sqrt_b_over_a
                b_heur = sqrt_ab * sqrt_b_over_a
                for p_heur in [-1.0, 0.0, 1.0, 2.0]:
                    starting_points.append(np.array([p_heur - 1, -b_heur / 2, -a_heur / 2]))
        
        # Fallback: simple default
        if len(starting_points) == 0:
            starting_points.append(np.array([0.0, -0.5, -0.5]))  # p=1, a=1, b=1
        
        return starting_points
    
    def fisher_information(self, theta: Optional[NDArray] = None) -> NDArray:
        """
        Fisher information matrix I(θ) = ∇²A(θ).
        
        Uses numerical differentiation since the analytical form is complex.
        """
        if theta is None:
            theta = self.get_natural_params()
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
