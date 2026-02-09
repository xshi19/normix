"""
normix: Generalized Hyperbolic distributions as exponential families.

Implements Generalized Hyperbolic distributions and their special cases
(Variance Gamma, Normal-Inverse Gaussian, Normal-Inverse Gamma) with
sklearn-style API and exponential family structure.

Key features:
- Three parametrizations: classical, natural, expectation
- Cached derived quantities (Cholesky factors, parameter conversions)
- EM algorithm for marginal distribution fitting
- Frozen dataclass parameter containers (normix.params)
"""

from normix.params import (
    ExponentialParams,
    GammaParams,
    InverseGammaParams,
    InverseGaussianParams,
    GIGParams,
    MultivariateNormalParams,
    VarianceGammaParams,
    NormalInverseGammaParams,
    NormalInverseGaussianParams,
    GHParams,
)

__all__ = [
    # Parameter dataclasses
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
