"""
Utilities for normix.

Bessel functions in ``bessel``, shared constants in ``constants``,
RVS generation in ``rvs``, plotting helpers in ``plotting``,
validation/EM helpers in ``validation``.
"""
from importlib import import_module

from normix.utils.bessel import log_kv
from normix.utils.constants import LOG_EPS
from normix.utils.rvs import build_pinv_table, rvs_pinv
from normix.utils.validation import (
    validate_moments,
    print_moment_validation,
    print_exp_family_params,
)

_PLOTTING_EXPORTS = {
    "plot_pdf_cdf_comparison",
    "plot_sample_histograms",
    "plot_mle_fit",
    "plot_joint_1d",
    "plot_marginal_2d",
    "plot_em_convergence",
    "PHI",
    "FIG_W",
    "FIG_H",
}


def __getattr__(name: str):
    """Load plotting helpers lazily so matplotlib remains optional."""
    if name == "plotting":
        module = import_module("normix.utils.plotting")
        globals()[name] = module
        return module
    if name in _PLOTTING_EXPORTS:
        plotting = import_module("normix.utils.plotting")
        value = getattr(plotting, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

__all__ = [
    "log_kv",
    "LOG_EPS",
    "build_pinv_table",
    "rvs_pinv",
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
