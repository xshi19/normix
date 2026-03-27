"""
normix — JAX package for Generalized Hyperbolic distributions as exponential families.

Built on Equinox. Float64 precision throughout.

Quick start
-----------
>>> import jax.numpy as jnp
>>> from normix import GeneralizedHyperbolic, GIG, Gamma

>>> # Fit GH to data
>>> import jax
>>> key = jax.random.PRNGKey(0)
>>> X = jax.random.normal(key, (1000, 2))
>>> model = GeneralizedHyperbolic.fit(X, key=key, max_iter=50)
>>> log_p = jax.vmap(model.log_prob)(X)
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

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
from normix.mixtures.marginal import NormalMixture

from normix.fitting.em import BatchEMFitter, OnlineEMFitter, MiniBatchEMFitter

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
    "NormalMixture",
    # Fitters
    "BatchEMFitter",
    "OnlineEMFitter",
    "MiniBatchEMFitter",
]
