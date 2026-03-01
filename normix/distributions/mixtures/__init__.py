"""
Mixture distributions: joint f(x,y) and marginal f(x) distributions.

Normal mixture distributions have the form:

.. math::
    X = \\mu + \\gamma Y + \\sqrt{Y} Z

where :math:`Z \\sim N(0, \\Sigma)` and :math:`Y` follows a subordinator distribution.

Joint distributions :math:`f(x, y)` are exponential families.
Marginal distributions :math:`f(x)` are NOT exponential families.

Available distributions:
- GeneralizedHyperbolic: Y ~ GIG (most general case)
- JointGeneralizedHyperbolic: Joint distribution for Generalized Hyperbolic
- VarianceGamma: Y ~ Gamma (special case of GH with b → 0)
- JointVarianceGamma: Joint distribution for Variance Gamma
- NormalInverseGamma: Y ~ InverseGamma (special case of GH with a → 0)
- JointNormalInverseGamma: Joint distribution for Normal Inverse Gamma
- NormalInverseGaussian: Y ~ InverseGaussian (special case of GH with p = -1/2)
- JointNormalInverseGaussian: Joint distribution for Normal Inverse Gaussian
"""

from .generalized_hyperbolic import (
    GeneralizedHyperbolic,
    REGULARIZATION_METHODS,
    regularize_det_sigma_one,
    regularize_sigma_diagonal_one,
    regularize_fix_p,
    regularize_none,
)
from .joint_generalized_hyperbolic import JointGeneralizedHyperbolic
from .variance_gamma import VarianceGamma
from .joint_variance_gamma import JointVarianceGamma
from .normal_inverse_gamma import NormalInverseGamma
from .joint_normal_inverse_gamma import JointNormalInverseGamma
from .normal_inverse_gaussian import NormalInverseGaussian
from .joint_normal_inverse_gaussian import JointNormalInverseGaussian

__all__ = [
    "GeneralizedHyperbolic",
    "JointGeneralizedHyperbolic",
    "REGULARIZATION_METHODS",
    "regularize_det_sigma_one",
    "regularize_sigma_diagonal_one",
    "regularize_fix_p",
    "regularize_none",
    "VarianceGamma",
    "JointVarianceGamma",
    "NormalInverseGamma",
    "JointNormalInverseGamma",
    "NormalInverseGaussian",
    "JointNormalInverseGaussian",
]
