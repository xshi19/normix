"""
Notebook utilities for normix.

Plotting helpers live in ``plotting``, validation/EM helpers in ``validation``.
"""
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
    run_em,
)

__all__ = [
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
    "run_em",
]
