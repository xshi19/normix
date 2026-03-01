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

import dataclasses
from dataclasses import dataclass, fields
import numpy as np


class _ParamsBase:
    """Mixin providing dict-style access on frozen dataclass params.

    Allows both ``params.mu`` and ``params['mu']`` access styles,
    plus ``items()``, ``keys()``, ``values()`` for iteration.
    """

    __slots__ = ()

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self):
        """Yield field names."""
        return (f.name for f in fields(self))

    def values(self):
        """Yield field values."""
        return (getattr(self, f.name) for f in fields(self))

    def items(self):
        """Yield ``(name, value)`` pairs."""
        return ((f.name, getattr(self, f.name)) for f in fields(self))


# ============================================================================
# Univariate distribution parameters
# ============================================================================

@dataclass(frozen=True, slots=True)
class ExponentialParams(_ParamsBase):
    """
    Classical parameters for the Exponential distribution.

    Attributes
    ----------
    rate : float
        Rate parameter :math:`\\lambda > 0`.
    """
    rate: float


@dataclass(frozen=True, slots=True)
class GammaParams(_ParamsBase):
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
class InverseGammaParams(_ParamsBase):
    """
    Classical parameters for the Inverse Gamma distribution.

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
class InverseGaussianParams(_ParamsBase):
    """
    Classical parameters for the Inverse Gaussian distribution.

    Attributes
    ----------
    delta : float
        Mean parameter :math:`\\delta > 0`.
    eta : float
        Shape parameter :math:`\\eta > 0`.
    """
    delta: float
    eta: float


@dataclass(frozen=True, slots=True)
class GIGParams(_ParamsBase):
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
class MultivariateNormalParams(_ParamsBase):
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
class VarianceGammaParams(_ParamsBase):
    """
    Classical parameters for the Variance Gamma distribution.

    The subordinator distribution is :math:`Y \\sim \\text{Gamma}(\\alpha, \\beta)`.

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
class NormalInverseGammaParams(_ParamsBase):
    """
    Classical parameters for the Normal-Inverse Gamma distribution.

    The subordinator distribution is :math:`Y \\sim \\text{InverseGamma}(\\alpha, \\beta)`.

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
class NormalInverseGaussianParams(_ParamsBase):
    """
    Classical parameters for the Normal-Inverse Gaussian distribution.

    The subordinator distribution is :math:`Y \\sim \\text{InverseGaussian}(\\delta, \\eta)`.

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
class GHParams(_ParamsBase):
    """
    Classical parameters for the Generalized Hyperbolic distribution.

    The subordinator distribution is :math:`Y \\sim \\text{GIG}(p, a, b)`.

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
