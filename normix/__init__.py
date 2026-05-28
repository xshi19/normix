"""
normix — JAX package for Generalized Hyperbolic distributions as exponential families.

Built on Equinox. Float64 precision strongly recommended.

Quick start
-----------
>>> import jax
>>> jax.config.update("jax_enable_x64", True)  # recommended
>>> import jax.numpy as jnp
>>> from normix import GeneralizedHyperbolic

>>> key = jax.random.PRNGKey(0)
>>> X = jax.random.normal(key, (1000, 2))
>>> model = GeneralizedHyperbolic.default_init(X)
>>> result = model.fit(X, max_iter=50)
>>> log_p = jax.vmap(result.model.log_prob)(X)
"""
from __future__ import annotations

import warnings
import jax

if not jax.config.jax_enable_x64:
    warnings.warn(
        "normix: float64 is not enabled. Bessel functions, GIG optimization, "
        "and log-density evaluation may lose accuracy under float32. "
        "Enable before importing normix:\n"
        "  jax.config.update('jax_enable_x64', True)",
        stacklevel=2,
    )

# jaxopt is unmaintained upstream and emits a DeprecationWarning on import.
# We use it intentionally for JAX-native LBFGS/BFGS in fitting/solvers.py
# and suppress the warning here so library users are not affected.
# Migration path: see exponential_family design doc § jaxopt migration (D4).
warnings.filterwarnings(
    "ignore",
    message="JAXopt is no longer maintained",
    category=DeprecationWarning,
    module="jaxopt",
)

from normix.exponential_family import ExponentialFamily
from normix.utils.bessel import log_kv

from normix.distributions.gamma import Gamma
from normix.distributions.inverse_gamma import InverseGamma
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.distributions.generalized_inverse_gaussian import GIG, GeneralizedInverseGaussian
from normix.distributions.normal import MultivariateNormal
from normix.distributions.variance_gamma import (
    VarianceGamma, JointVarianceGamma, FactorVarianceGamma,
    UnivariateVarianceGamma)
from normix.distributions.normal_inverse_gamma import (
    NormalInverseGamma, JointNormalInverseGamma, FactorNormalInverseGamma,
    UnivariateNormalInverseGamma)
from normix.distributions.normal_inverse_gaussian import (
    NormalInverseGaussian, JointNormalInverseGaussian,
    FactorNormalInverseGaussian, UnivariateNormalInverseGaussian)
from normix.distributions.generalized_hyperbolic import (
    GeneralizedHyperbolic, JointGeneralizedHyperbolic,
    FactorGeneralizedHyperbolic, UnivariateGeneralizedHyperbolic)

from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import MarginalMixture, NormalMixture
from normix.mixtures.factor import FactorNormalMixture

from normix.fitting.em import BatchEMFitter, IncrementalEMFitter
from normix.fitting.eta import NormalMixtureEta, FactorMixtureStats, affine_combine
from normix.fitting.eta_rules import (
    EtaUpdateRule,
    AffineRule,
    IdentityUpdate,
    RobbinsMonroUpdate,
    SampleWeightedUpdate,
    EWMAUpdate,
    Shrinkage,
    AffineUpdate,
)
from normix.fitting.shrinkage_targets import (
    eta0_from_model,
    eta0_isotropic,
    eta0_diagonal,
    eta0_with_sigma,
)

from normix import finance

from normix.divergences import (
    squared_hellinger,
    kl_divergence,
    squared_hellinger_from_psi,
    kl_divergence_from_psi,
)

__version__ = "0.2.3"  # x-release-please-version

__all__ = [
    # Exponential family base
    "ExponentialFamily",
    # Bessel
    "log_kv",
    # Univariate distributions
    "Gamma",
    "InverseGamma",
    "InverseGaussian",
    "GIG",
    "GeneralizedInverseGaussian",
    # Multivariate normal
    "MultivariateNormal",
    # Mixture distributions (marginal)
    "VarianceGamma",
    "NormalInverseGamma",
    "NormalInverseGaussian",
    "GeneralizedHyperbolic",
    # Univariate (d=1) marginal mixtures — scipy-style scalar API + cdf/ppf
    "UnivariateVarianceGamma",
    "UnivariateNormalInverseGamma",
    "UnivariateNormalInverseGaussian",
    "UnivariateGeneralizedHyperbolic",
    # Mixture distributions (joint)
    "JointVarianceGamma",
    "JointNormalInverseGamma",
    "JointNormalInverseGaussian",
    "JointGeneralizedHyperbolic",
    # Base mixture classes
    "JointNormalMixture",
    "MarginalMixture",
    "NormalMixture",
    # Factor-analysis mixtures
    "FactorNormalMixture",
    "FactorVarianceGamma",
    "FactorNormalInverseGamma",
    "FactorNormalInverseGaussian",
    "FactorGeneralizedHyperbolic",
    # Fitters
    "BatchEMFitter",
    "IncrementalEMFitter",
    # Eta parametrization
    "NormalMixtureEta",
    "FactorMixtureStats",
    "affine_combine",
    # Eta update rules
    "EtaUpdateRule",
    "AffineRule",
    "IdentityUpdate",
    "RobbinsMonroUpdate",
    "SampleWeightedUpdate",
    "EWMAUpdate",
    "Shrinkage",
    "AffineUpdate",
    # Shrinkage targets
    "eta0_from_model",
    "eta0_isotropic",
    "eta0_diagonal",
    "eta0_with_sigma",
    # Finance subpackage
    "finance",
    # Divergences
    "squared_hellinger",
    "kl_divergence",
    "squared_hellinger_from_psi",
    "kl_divergence_from_psi",
]
