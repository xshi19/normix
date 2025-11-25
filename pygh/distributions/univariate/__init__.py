"""Univariate distributions (exponential family)."""

from .exponential import Exponential
from .gamma import Gamma
from .inverse_gamma import InverseGamma
from .inverse_gaussian import InverseGaussian

__all__ = ['Exponential', 'Gamma', 'InverseGamma', 'InverseGaussian']

