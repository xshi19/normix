"""Compatibility imports for the former conditional Monte Carlo helpers.

Arrays still represent equal-weight draws. The implementation now accepts
weighted measures in :mod:`normix.finance._conditional`.
"""
from normix.finance._conditional import (
    bracket_raw as bracket_cmc_raw,
    cdf as cdf_cmc,
    cdf_raw as cdf_cmc_raw,
    quantile as quantile_cmc,
    quantile_raw as quantile_cmc_raw,
)

__all__ = [
    "bracket_cmc_raw",
    "cdf_cmc",
    "cdf_cmc_raw",
    "quantile_cmc",
    "quantile_cmc_raw",
]
