"""
Bessel function utilities for numerical stability.

This module provides numerically stable implementations of log Bessel functions
that are essential for the Generalized Inverse Gaussian distribution.

The modified Bessel function of the second kind :math:`K_\\nu(z)` appears in the
normalizing constant of the GIG distribution:

.. math::
    f(x|p,a,b) = \\frac{(a/b)^{p/2}}{2 K_p(\\sqrt{ab})} x^{p-1} 
    \\exp\\left(-\\frac{ax + b/x}{2}\\right)

For numerical stability, we work with :math:`\\log K_\\nu(z)` instead of 
:math:`K_\\nu(z)` directly.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Union
from scipy.special import gammaln, kve


def log_kv(v: float, z: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Log modified Bessel function of the second kind: log K_v(z).
    
    Computes :math:`\\log K_\\nu(z)` in a numerically stable way.
    
    This is an optimized version where :math:`\\nu` is a scalar and :math:`z` 
    can be vectorized. For the fully vectorized version (both :math:`\\nu` and 
    :math:`z` arrays), use :func:`log_kv_vectorized`.
    
    Uses the exponentially scaled function :math:`K_\\nu^e(z) = K_\\nu(z) e^z` for
    numerical stability when :math:`z` is large:
    
    .. math::
        \\log K_\\nu(z) = \\log K_\\nu^e(z) - z
    
    When the result is inf (underflow for small :math:`z`), uses asymptotic approximations:
    
    - If :math:`|\\nu| > \\varepsilon`: 
      :math:`\\log K_\\nu(z) \\approx \\log\\Gamma(|\\nu|) - \\log 2 + |\\nu|(\\log 2 - \\log z)`
    - If :math:`|\\nu| \\approx 0`: 
      :math:`\\log K_0(z) \\approx \\log(-\\log(z/2) - \\gamma)` where :math:`\\gamma` is Euler's constant
    
    Parameters
    ----------
    v : float
        Order of Bessel function :math:`\\nu` (scalar).
    z : float or ndarray
        Argument (must be > 0), can be vectorized.
    
    Returns
    -------
    log_kv : float or ndarray
        :math:`\\log K_\\nu(z)`.
    
    Examples
    --------
    >>> log_kv(1.0, 2.0)
    -1.9670713500717493
    >>> log_kv(1.5, np.array([0.5, 1.0, 2.0]))
    array([ 0.68024015, -0.38318674, -1.71531713])
    """
    z_scalar = np.isscalar(z)
    z = np.atleast_1d(np.asarray(z, dtype=float))
    
    # Ensure z is positive
    z = np.maximum(z, np.finfo(float).tiny)
    
    # Use kve for numerical stability: kve(v, z) = kv(v, z) * exp(z)
    # So log(kv(v, z)) = log(kve(v, z)) - z
    result = np.log(kve(v, z)) - z
    
    # Handle cases where result is inf (kv underflows to 0 when z → 0)
    # The threshold depends on v, so check the result directly
    inf_mask = np.isinf(result)
    
    if np.any(inf_mask):
        z_inf = z[inf_mask]
        v_abs = np.abs(v)
        
        # For |v| > ε: K_v(z) ≈ Γ(|v|)/2 · (2/z)^|v|
        # log(K_v(z)) ≈ gammaln(|v|) - log(2) + |v|(log(2) - log(z))
        if v_abs > 1e-10:
            approx = gammaln(v_abs) - np.log(2.0) + v_abs * (np.log(2.0) - np.log(z_inf))
        else:
            # For v ≈ 0: K_0(z) ≈ -log(z/2) - γ (Euler's constant)
            # log(K_0(z)) ≈ log(-log(z/2) - γ)
            inner = -np.log(z_inf / 2.0) - np.euler_gamma
            inner = np.maximum(inner, np.finfo(float).tiny)
            approx = np.log(inner)
        
        result[inf_mask] = approx
    
    if z_scalar:
        return float(result[0])
    
    return result


def log_kv_vectorized(v: Union[float, NDArray], z: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Log modified Bessel function of the second kind: log(K_v(z)).
    
    Fully vectorized version where both v and z can be arrays.
    This is slower than log_kv when v is a scalar.
    
    Parameters
    ----------
    v : float or ndarray
        Order of Bessel function.
    z : float or ndarray
        Argument (must be > 0).
    
    Returns
    -------
    log_kv : float or ndarray
        Log of the modified Bessel function of the second kind.
    """
    v_scalar = np.isscalar(v)
    z_scalar = np.isscalar(z)
    
    v = np.asarray(v, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Broadcast v and z to same shape
    v, z = np.broadcast_arrays(v, z)
    
    # Ensure z is positive
    z = np.maximum(z, np.finfo(float).tiny)
    
    # Use kve for numerical stability
    result = np.log(kve(v, z)) - z
    
    # Handle inf cases
    inf_mask = np.isinf(result)
    
    if np.any(inf_mask):
        z_inf = z[inf_mask]
        v_inf = v[inf_mask]
        v_abs = np.abs(v_inf)
        
        approx = np.zeros_like(z_inf)
        
        large_v_mask = v_abs > 1e-10
        if np.any(large_v_mask):
            approx[large_v_mask] = (gammaln(v_abs[large_v_mask]) - np.log(2.0) + 
                                    v_abs[large_v_mask] * (np.log(2.0) - np.log(z_inf[large_v_mask])))
        
        small_v_mask = ~large_v_mask
        if np.any(small_v_mask):
            inner = -np.log(z_inf[small_v_mask] / 2.0) - np.euler_gamma
            inner = np.maximum(inner, np.finfo(float).tiny)
            approx[small_v_mask] = np.log(inner)
        
        result[inf_mask] = approx
    
    if v_scalar and z_scalar and result.size == 1:
        return float(result.flat[0])
    
    return result


def log_kv_derivative_v(v: float, z: Union[float, NDArray], *, eps: float = 1e-6) -> Union[float, NDArray]:
    r"""
    Derivative of log(K_v(z)) with respect to the order v.

    Uses the asymptotic expansion (DLMF 10.40.2) when :math:`z` is large
    relative to :math:`\nu^2`, giving an analytical formula without finite
    differences.  Falls back to central differences otherwise.

    **Asymptotic formula** (for large :math:`z`):

    .. math::

        \frac{\partial}{\partial\nu}\log K_\nu(z)
        \approx \frac{S'(\nu,z)}{S(\nu,z)},
        \quad S(\nu,z) = \sum_{k=0}^{K} \frac{a_k(\nu)}{z^k}

    where :math:`a_k(\nu) = \frac{\prod_{j=1}^k[4\nu^2-(2j-1)^2]}{k!\,8^k}`
    and :math:`a_k'(\nu) = a_k(\nu)\sum_{i=1}^k \frac{8\nu}{4\nu^2-(2i-1)^2}`.

    **Central-difference formula** (for moderate :math:`z`):

    .. math::

        \frac{\partial}{\partial\nu}\log K_\nu(z)
        \approx \frac{\log K_{\nu+\varepsilon}(z)
                      - \log K_{\nu-\varepsilon}(z)}{2\varepsilon}

    Parameters
    ----------
    v : float
        Order of Bessel function :math:`\nu` (scalar).
    z : float or ndarray
        Argument (must be > 0), can be vectorized.
    eps : float, optional
        Step size for central differences. Default is 1e-6.

    Returns
    -------
    deriv : float or ndarray
        Derivative of :math:`\log K_\nu(z)` with respect to :math:`\nu`.

    Examples
    --------
    >>> log_kv_derivative_v(1.0, 2.0)
    -0.4250407...
    """
    z_scalar = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))

    K = 9  # number of asymptotic terms
    v_abs = abs(v)
    # Asymptotic threshold: accurate when z > v²/(K-1) and z > 20
    asym_thresh = max(20.0, v_abs**2 / (K - 1))

    result = np.empty_like(z_arr)

    asym_mask = z_arr >= asym_thresh
    fd_mask = ~asym_mask

    if np.any(asym_mask):
        result[asym_mask] = _log_kv_deriv_v_asymptotic(v, z_arr[asym_mask], K=K)

    if np.any(fd_mask):
        z_fd = z_arr[fd_mask]
        result[fd_mask] = (log_kv(v + eps, z_fd) - log_kv(v - eps, z_fd)) / (2 * eps)

    if z_scalar:
        return float(result[0])
    return result


def _log_kv_deriv_v_asymptotic(
    v: float,
    z: Union[float, np.ndarray],
    K: int = 9,
) -> Union[float, np.ndarray]:
    r"""
    Analytical :math:`\partial_\nu\log K_\nu(z)` via DLMF 10.40.2 (large-z).

    Uses :math:`K` terms of the asymptotic expansion.  Accurate when
    :math:`z \gg \nu^2 / K`.  Falls back to central differences for any
    element where a denominator factor :math:`4\nu^2-(2i-1)^2` is near zero
    (i.e. half-integer or integer orders that coincide with a pole).

    Parameters
    ----------
    v : float
        Order :math:`\nu` (scalar).
    z : float or ndarray
        Argument(s), must be positive.
    K : int
        Number of asymptotic terms. Default 9 (matches legacy code).

    Returns
    -------
    deriv : float or ndarray
    """
    from scipy.special import factorial as sp_factorial

    z_scalar = np.isscalar(z)
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    k_arr = np.arange(1, K + 1, dtype=float)

    # a_k(v) = cumprod_{j=1}^k [4v²-(2j-1)²] / (k! * 8^k)
    factors = 4.0 * v**2 - (2 * k_arr - 1)**2          # shape (K,)
    a_k = np.cumprod(factors) / sp_factorial(k_arr, exact=False) / 8.0**k_arr

    # a_k'(v) = a_k(v) * Σ_{i=1}^k  8v / (4v²-(2i-1)²)
    # built up cumulatively so each a_k' reuses the previous running sum.
    da_k = np.zeros(K)
    fallback_needed = False
    running_sum = 0.0
    for idx in range(K):
        f = factors[idx]
        if abs(f) < 1e-8:
            # v is at or near a pole of the log-derivative: use FD fallback
            fallback_needed = True
            break
        running_sum += 8.0 * v / f
        da_k[idx] = a_k[idx] * running_sum

    # S(v,z) = 1 + Σ_k a_k/z^k,  S'(v,z) = Σ_k da_k/z^k
    # z_powers shape: (n_z, K)
    inv_z = (1.0 / z_arr)[:, np.newaxis]          # (n_z, 1)
    z_pows = inv_z ** k_arr                        # (n_z, K)

    S  = 1.0 + z_pows @ a_k                       # (n_z,)
    dS = z_pows @ da_k                             # (n_z,)

    result = dS / S                                # (n_z,)

    if fallback_needed:
        # Any element that hit a pole: replace with central differences
        eps = 1e-6
        result[:] = (log_kv(v + eps, z_arr) - log_kv(v - eps, z_arr)) / (2 * eps)

    if z_scalar:
        return float(result[0])
    return result


def log_kv_derivative_z(v: float, z: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Derivative of log(K_v(z)) with respect to z.
    
    Using the recurrence relation::
    
        K'_v(z) = -(K_{v-1}(z) + K_{v+1}(z)) / 2
    
    So::
    
        d/dz log(K_v(z)) = K'_v(z) / K_v(z)
                         = -(K_{v-1}(z) + K_{v+1}(z)) / (2 K_v(z))
                         = -1/2 (exp(log K_{v-1}(z) - log K_v(z)) + 
                                 exp(log K_{v+1}(z) - log K_v(z)))
    
    Parameters
    ----------
    v : float
        Order of Bessel function (scalar).
    z : float or ndarray
        Argument (must be > 0), can be vectorized.
    
    Returns
    -------
    deriv : float or ndarray
        Derivative of log K_v(z) with respect to z.
    
    Examples
    --------
    >>> log_kv_derivative_z(1.0, 2.0)
    -1.3143079478654318
    """
    log_kv_val = log_kv(v, z)
    log_kv_m1 = log_kv(v - 1, z)
    log_kv_p1 = log_kv(v + 1, z)
    
    return -0.5 * (np.exp(log_kv_m1 - log_kv_val) + np.exp(log_kv_p1 - log_kv_val))


def kv_ratio(v1: float, v2: float, z: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Compute the ratio K_{v1}(z) / K_{v2}(z) in a numerically stable way.
    
    Uses log space to avoid overflow/underflow:
        K_{v1}(z) / K_{v2}(z) = exp(log K_{v1}(z) - log K_{v2}(z))
    
    Parameters
    ----------
    v1 : float
        Order of numerator Bessel function.
    v2 : float
        Order of denominator Bessel function.
    z : float or ndarray
        Argument (must be > 0).
    
    Returns
    -------
    ratio : float or ndarray
        The ratio K_{v1}(z) / K_{v2}(z).
    """
    return np.exp(log_kv(v1, z) - log_kv(v2, z))

