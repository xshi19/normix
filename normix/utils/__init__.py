"""
Utilities for normix.

Bessel functions in ``bessel``, shared constants in ``constants``,
plotting helpers in ``plotting``, validation/EM helpers in ``validation``.
"""
from normix.utils.bessel import log_kv
from normix.utils.constants import LOG_EPS
from normix.utils.plotting import (
    plot_pdf_cdf_comparison,
    plot_sample_histograms,
    plot_mle_fit,
    plot_joint_1d,
    plot_marginal_2d,
    plot_em_convergence,
    PHI,
    FIG_W,
    FIG_H,
)

from normix.utils.validation import (
    validate_moments,
    print_moment_validation,
    print_exp_family_params,
)

__all__ = [
    "log_kv",
    "LOG_EPS",
    "plot_pdf_cdf_comparison",
    "plot_sample_histograms",
    "plot_mle_fit",
    "plot_joint_1d",
    "plot_marginal_2d",
    "plot_em_convergence",
    "PHI",
    "FIG_W",
    "FIG_H",
    "validate_moments",
    "print_moment_validation",
    "print_exp_family_params",
]
