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
# Migration path: see docs/design/design.md § "jaxopt migration (D4)".
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
from normix.distributions.variance_gamma import VarianceGamma, JointVarianceGamma
from normix.distributions.normal_inverse_gamma import (
    NormalInverseGamma, JointNormalInverseGamma)
from normix.distributions.normal_inverse_gaussian import (
    NormalInverseGaussian, JointNormalInverseGaussian)
from normix.distributions.generalized_hyperbolic import (
    GeneralizedHyperbolic, JointGeneralizedHyperbolic)

from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import MarginalMixture, NormalMixture

from normix.fitting.em import BatchEMFitter, IncrementalEMFitter
from normix.fitting.eta import NormalMixtureEta, affine_combine
from normix.fitting.eta_rules import (
    EtaUpdateRule,
    AffineRule,
    IdentityUpdate,
    RobbinsMonroUpdate,
    SampleWeightedUpdate,
    EWMAUpdate,
    ShrinkageUpdate,
    AffineUpdate,
)

from normix.divergences import (
    squared_hellinger,
    kl_divergence,
    squared_hellinger_from_psi,
    kl_divergence_from_psi,
)

__version__ = "0.2.0"

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
    # Mixture distributions (joint)
    "JointVarianceGamma",
    "JointNormalInverseGamma",
    "JointNormalInverseGaussian",
    "JointGeneralizedHyperbolic",
    # Base mixture classes
    "JointNormalMixture",
    "MarginalMixture",
    "NormalMixture",
    # Fitters
    "BatchEMFitter",
    "IncrementalEMFitter",
    # Eta parametrization
    "NormalMixtureEta",
    "affine_combine",
    # Eta update rules
    "EtaUpdateRule",
    "AffineRule",
    "IdentityUpdate",
    "RobbinsMonroUpdate",
    "SampleWeightedUpdate",
    "EWMAUpdate",
    "ShrinkageUpdate",
    "AffineUpdate",
    # Divergences
    "squared_hellinger",
    "kl_divergence",
    "squared_hellinger_from_psi",
    "kl_divergence_from_psi",
]
