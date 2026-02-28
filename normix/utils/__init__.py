"""Utility functions for normix package."""

from .bessel import log_kv, log_kv_vectorized, log_kv_derivative_z, kv_ratio
from .linalg import robust_cholesky
from .scaling import column_median_mad
from .mixture_viz import (
    plot_joint_distribution_1d,
    plot_marginal_distribution_2d,
    validate_moments,
    print_moment_validation,
    fit_and_track_convergence,
    plot_em_convergence,
    test_joint_fitting,
    print_fitting_results,
    comprehensive_distribution_test,
    EMConvergenceResult
)

__all__ = [
    'log_kv', 'log_kv_vectorized', 'log_kv_derivative_z', 'kv_ratio',
    'robust_cholesky',
    'column_median_mad',
    'plot_joint_distribution_1d', 'plot_marginal_distribution_2d',
    'validate_moments', 'print_moment_validation',
    'fit_and_track_convergence', 'plot_em_convergence',
    'test_joint_fitting', 'print_fitting_results',
    'comprehensive_distribution_test', 'EMConvergenceResult'
]
