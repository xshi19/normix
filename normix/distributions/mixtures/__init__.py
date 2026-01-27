"""
Mixture distributions: joint f(x,y) and marginal f(x) distributions.

Normal mixture distributions have the form:

.. math::
    X = \\mu + \\gamma Y + \\sqrt{Y} Z

where :math:`Z \\sim N(0, \\Sigma)` and :math:`Y` follows a mixing distribution.

Joint distributions :math:`f(x, y)` are exponential families.
Marginal distributions :math:`f(x)` are NOT exponential families.

Available distributions:
- VarianceGamma: Y ~ Gamma (special case of GH with b → 0)
- JointVarianceGamma: Joint distribution for Variance Gamma
"""

from .variance_gamma import VarianceGamma
from .joint_variance_gamma import JointVarianceGamma

__all__ = [
    "VarianceGamma",
    "JointVarianceGamma",
]
