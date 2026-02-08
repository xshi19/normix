"""
Frozen dataclass parameter containers for all distributions.

Each distribution's classical parameters are represented as a frozen dataclass
with ``slots=True`` for memory efficiency. This provides:

- **IDE autocompletion**: ``params.mu`` instead of ``params['mu']``
- **Immutability**: Prevents accidental mutation of fitted parameters
- **Type safety**: Type hints for all fields
- **Dict conversion**: ``dataclasses.asdict(params)`` when needed

Examples
--------
>>> from normix.params import ExponentialParams, GHParams
>>> p = ExponentialParams(rate=2.0)
>>> p.rate
2.0
>>> p.rate = 3.0  # Raises FrozenInstanceError

>>> import dataclasses
>>> dataclasses.asdict(p)
{'rate': 2.0}

Notes
-----
The ``frozen=True`` flag prevents attribute reassignment, but numpy arrays
are internally mutable (``params.mu[0] = 999`` still works at the Python level).
For full immutability, callers should not modify returned arrays in-place.
"""

from dataclasses import dataclass
import numpy as np


# ============================================================================
# Univariate distribution parameters
# ============================================================================

@dataclass(frozen=True, slots=True)
class ExponentialParams:
    """
    Classical parameters for the Exponential distribution.

    Attributes
    ----------
    rate : float
        Rate parameter :math:`\\lambda > 0`.
    """
    rate: float


@dataclass(frozen=True, slots=True)
class GammaParams:
    """
    Classical parameters for the Gamma distribution.

    Attributes
    ----------
    shape : float
        Shape parameter :math:`\\alpha > 0`.
    rate : float
        Rate parameter :math:`\\beta > 0`.
    """
    shape: float
    rate: float


@dataclass(frozen=True, slots=True)
class InverseGammaParams:
    """
    Classical parameters for the Inverse Gamma distribution.

    Attributes
    ----------
    shape : float
        Shape parameter :math:`\\alpha > 0`.
    scale : float
        Scale parameter :math:`\\beta > 0`.
    """
    shape: float
    scale: float


@dataclass(frozen=True, slots=True)
class InverseGaussianParams:
    """
    Classical parameters for the Inverse Gaussian distribution.

    Attributes
    ----------
    mean : float
        Mean parameter :math:`\\mu > 0`.
    shape : float
        Shape parameter :math:`\\lambda > 0`.
    """
    mean: float
    shape: float


@dataclass(frozen=True, slots=True)
class GIGParams:
    """
    Classical parameters for the Generalized Inverse Gaussian distribution.

    Attributes
    ----------
    p : float
        Index parameter :math:`p \\in \\mathbb{R}`.
    a : float
        Concentration parameter :math:`a > 0`.
    b : float
        Scale parameter :math:`b > 0`.
    """
    p: float
    a: float
    b: float


# ============================================================================
# Multivariate distribution parameters
# ============================================================================

@dataclass(frozen=True, slots=True)
class MultivariateNormalParams:
    """
    Classical parameters for the Multivariate Normal distribution.

    Attributes
    ----------
    mu : np.ndarray
        Mean vector, shape ``(d,)``.
    sigma : np.ndarray
        Covariance matrix, shape ``(d, d)``.
    """
    mu: np.ndarray
    sigma: np.ndarray


# ============================================================================
# Normal mixture distribution parameters
# ============================================================================

@dataclass(frozen=True, slots=True)
class VarianceGammaParams:
    """
    Classical parameters for the Variance Gamma distribution.

    The mixing distribution is :math:`Y \\sim \\text{Gamma}(\\alpha, \\beta)`.

    Attributes
    ----------
    mu : np.ndarray
        Location parameter, shape ``(d,)``.
    gamma : np.ndarray
        Skewness parameter, shape ``(d,)``.
    sigma : np.ndarray
        Covariance scale matrix, shape ``(d, d)``.
    shape : float
        Gamma shape parameter :math:`\\alpha > 0`.
    rate : float
        Gamma rate parameter :math:`\\beta > 0`.
    """
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    shape: float
    rate: float


@dataclass(frozen=True, slots=True)
class NormalInverseGammaParams:
    """
    Classical parameters for the Normal-Inverse Gamma distribution.

    The mixing distribution is :math:`Y \\sim \\text{InverseGamma}(\\alpha, \\beta)`.

    Attributes
    ----------
    mu : np.ndarray
        Location parameter, shape ``(d,)``.
    gamma : np.ndarray
        Skewness parameter, shape ``(d,)``.
    sigma : np.ndarray
        Covariance scale matrix, shape ``(d, d)``.
    shape : float
        Inverse Gamma shape parameter :math:`\\alpha > 0`.
    rate : float
        Inverse Gamma rate parameter :math:`\\beta > 0`.
    """
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    shape: float
    rate: float


@dataclass(frozen=True, slots=True)
class NormalInverseGaussianParams:
    """
    Classical parameters for the Normal-Inverse Gaussian distribution.

    The mixing distribution is :math:`Y \\sim \\text{InverseGaussian}(\\delta, \\eta)`.

    Attributes
    ----------
    mu : np.ndarray
        Location parameter, shape ``(d,)``.
    gamma : np.ndarray
        Skewness parameter, shape ``(d,)``.
    sigma : np.ndarray
        Covariance scale matrix, shape ``(d, d)``.
    delta : float
        Scale parameter :math:`\\delta > 0`.
    eta : float
        Shape parameter :math:`\\eta > 0`.
    """
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    delta: float
    eta: float


@dataclass(frozen=True, slots=True)
class GHParams:
    """
    Classical parameters for the Generalized Hyperbolic distribution.

    The mixing distribution is :math:`Y \\sim \\text{GIG}(p, a, b)`.

    Attributes
    ----------
    mu : np.ndarray
        Location parameter, shape ``(d,)``.
    gamma : np.ndarray
        Skewness parameter, shape ``(d,)``.
    sigma : np.ndarray
        Covariance scale matrix, shape ``(d, d)``.
    p : float
        GIG index parameter :math:`p \\in \\mathbb{R}`.
    a : float
        GIG concentration parameter :math:`a > 0`.
    b : float
        GIG scale parameter :math:`b > 0`.
    """
    mu: np.ndarray
    gamma: np.ndarray
    sigma: np.ndarray
    p: float
    a: float
    b: float


__all__ = [
    "ExponentialParams",
    "GammaParams",
    "InverseGammaParams",
    "InverseGaussianParams",
    "GIGParams",
    "MultivariateNormalParams",
    "VarianceGammaParams",
    "NormalInverseGammaParams",
    "NormalInverseGaussianParams",
    "GHParams",
]
