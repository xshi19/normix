"""
Visualization and testing utilities for normal mixture distributions.

This module provides standardized visualization and validation functions for
normal variance-mean mixture distributions (VG, NInvG, NIG, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Type, Any
from dataclasses import dataclass


@dataclass
class EMConvergenceResult:
    """Container for EM algorithm convergence results."""
    iterations: List[int]
    log_likelihoods: List[float]
    converged: bool
    final_params: Dict[str, Any]


def plot_joint_distribution_1d(
    joint_dist,
    n_samples: int = 5000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot joint distribution f(x, y) for 1D X with marginal histograms.
    
    Creates a figure with:
    - Center: 2D contour plot of joint PDF with scatter plot of samples
    - Top: Marginal histogram and PDF of X
    - Right: Marginal histogram and PDF of Y (mixing variable)
    
    Parameters
    ----------
    joint_dist : JointNormalMixture
        Joint distribution object (JointVarianceGamma, JointNormalInverseGamma, etc.)
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Generate samples
    X_samples, Y_samples = joint_dist.rvs(size=n_samples, random_state=random_state)
    X_samples = X_samples.flatten()
    
    # Create figure with custom grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    
    # Main scatter/contour plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    
    # Top marginal
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    
    # Right marginal
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # Compute PDF on grid for contour plot
    x_margin = 0.1 * (np.percentile(X_samples, 99) - np.percentile(X_samples, 1))
    y_margin = 0.1 * (np.percentile(Y_samples, 99) - np.percentile(Y_samples, 1))
    
    x_range = np.linspace(
        np.percentile(X_samples, 1) - x_margin,
        np.percentile(X_samples, 99) + x_margin,
        80
    )
    y_range = np.linspace(
        max(0.01, np.percentile(Y_samples, 1) - y_margin),
        np.percentile(Y_samples, 99) + y_margin,
        80
    )
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Compute joint PDF
    Z = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            Z[i, j] = float(joint_dist.pdf(
                np.array([[X_grid[i, j]]]),
                np.array([Y_grid[i, j]])
            ))
    
    # Main plot: contour with scatter
    ax_main.contourf(X_grid, Y_grid, Z, levels=20, cmap='Blues', alpha=0.7)
    ax_main.scatter(X_samples, Y_samples, alpha=0.1, s=3, c='darkblue', label='Samples')
    ax_main.set_xlabel('X', fontsize=12)
    ax_main.set_ylabel('Y (mixing variable)', fontsize=12)
    
    # Top marginal: X histogram and PDF
    ax_top.hist(X_samples, bins=50, density=True, alpha=0.7, color='steelblue')
    # Compute marginal X PDF by integrating joint over Y
    x_pdf = np.sum(Z, axis=0) * (y_range[1] - y_range[0])
    x_pdf = x_pdf / np.trapezoid(x_pdf, x_range)  # Normalize
    ax_top.plot(x_range, x_pdf, 'r-', linewidth=2, label='PDF')
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc='upper right', fontsize=9)
    
    # Right marginal: Y histogram and PDF
    ax_right.hist(Y_samples, bins=50, density=True, alpha=0.7, color='steelblue',
                  orientation='horizontal')
    # Compute marginal Y PDF by integrating joint over X
    y_pdf = np.sum(Z, axis=1) * (x_range[1] - x_range[0])
    y_pdf = y_pdf / np.trapezoid(y_pdf, y_range)  # Normalize
    ax_right.plot(y_pdf, y_range, 'r-', linewidth=2, label='PDF')
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.legend(loc='upper right', fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    return fig


def plot_marginal_distribution_2d(
    marginal_dist,
    n_samples: int = 5000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D marginal distribution f(x) with marginal histograms.
    
    Creates a figure with:
    - Center: 2D contour plot of marginal PDF with scatter plot of samples
    - Top: Marginal histogram and PDF of X₁
    - Right: Marginal histogram and PDF of X₂
    
    Parameters
    ----------
    marginal_dist : NormalMixture
        2D marginal distribution object
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
        
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Generate samples
    X_samples = marginal_dist.rvs(size=n_samples, random_state=random_state)
    
    # Create figure with custom grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    
    # Main scatter/contour plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    
    # Top marginal
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    
    # Right marginal
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # Compute PDF on grid for contour plot
    x1_margin = 0.1 * (np.percentile(X_samples[:, 0], 99) - np.percentile(X_samples[:, 0], 1))
    x2_margin = 0.1 * (np.percentile(X_samples[:, 1], 99) - np.percentile(X_samples[:, 1], 1))
    
    x1_range = np.linspace(
        np.percentile(X_samples[:, 0], 1) - x1_margin,
        np.percentile(X_samples[:, 0], 99) + x1_margin,
        80
    )
    x2_range = np.linspace(
        np.percentile(X_samples[:, 1], 1) - x2_margin,
        np.percentile(X_samples[:, 1], 99) + x2_margin,
        80
    )
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    # Compute marginal PDF
    Z = np.zeros_like(X1_grid)
    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            point = np.array([[X1_grid[i, j], X2_grid[i, j]]])
            result = marginal_dist.pdf(point)
            Z[i, j] = float(result.flat[0]) if hasattr(result, 'flat') else float(result)
    
    # Main plot: contour with scatter
    ax_main.contourf(X1_grid, X2_grid, Z, levels=20, cmap='Blues', alpha=0.7)
    ax_main.scatter(X_samples[:, 0], X_samples[:, 1], alpha=0.1, s=3, c='darkblue', 
                    label='Samples')
    ax_main.set_xlabel('$X_1$', fontsize=12)
    ax_main.set_ylabel('$X_2$', fontsize=12)
    
    # Top marginal: X₁ histogram
    ax_top.hist(X_samples[:, 0], bins=50, density=True, alpha=0.7, color='steelblue')
    # Marginal PDF of X₁
    x1_pdf = np.sum(Z, axis=0) * (x2_range[1] - x2_range[0])
    x1_pdf = x1_pdf / np.trapezoid(x1_pdf, x1_range)  # Normalize
    ax_top.plot(x1_range, x1_pdf, 'r-', linewidth=2, label='PDF')
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(loc='upper right', fontsize=9)
    
    # Right marginal: X₂ histogram
    ax_right.hist(X_samples[:, 1], bins=50, density=True, alpha=0.7, color='steelblue',
                  orientation='horizontal')
    # Marginal PDF of X₂
    x2_pdf = np.sum(Z, axis=1) * (x1_range[1] - x1_range[0])
    x2_pdf = x2_pdf / np.trapezoid(x2_pdf, x2_range)  # Normalize
    ax_right.plot(x2_pdf, x2_range, 'r-', linewidth=2, label='PDF')
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.legend(loc='upper right', fontsize=9)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    return fig


def validate_moments(
    dist,
    n_samples: int = 50000,
    random_state: int = 42,
    is_joint: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Compare sample moments with theoretical moments.
    
    Parameters
    ----------
    dist : Distribution
        Distribution object
    n_samples : int
        Number of samples for empirical estimation
    random_state : int
        Random seed
    is_joint : bool
        If True, dist is a joint distribution returning (X, Y) tuples
        
    Returns
    -------
    results : dict
        Dictionary with 'mean' and 'variance' sub-dicts containing
        'sample', 'theoretical', and 'rel_error' values.
    """
    if is_joint:
        X_samples, Y_samples = dist.rvs(size=n_samples, random_state=random_state)
        X_samples = X_samples.flatten() if X_samples.ndim > 1 else X_samples
        
        # Get theoretical moments
        theory_mean = dist.mean()
        if isinstance(theory_mean, tuple):
            x_mean_theory = float(theory_mean[0].flat[0])
            y_mean_theory = float(theory_mean[1])
        else:
            x_mean_theory = float(np.asarray(theory_mean[0]).flat[0])
            y_mean_theory = float(theory_mean[1])
        
        theory_var = dist.var()
        if isinstance(theory_var, tuple):
            x_var_theory = float(theory_var[0].flat[0])
            y_var_theory = float(theory_var[1])
        else:
            x_var_theory = float(np.asarray(theory_var[0]).flat[0])
            y_var_theory = float(theory_var[1])
        
        return {
            'X_mean': {
                'sample': float(np.mean(X_samples)),
                'theoretical': x_mean_theory,
                'rel_error': abs(np.mean(X_samples) - x_mean_theory) / max(abs(x_mean_theory), 1e-10)
            },
            'Y_mean': {
                'sample': float(np.mean(Y_samples)),
                'theoretical': y_mean_theory,
                'rel_error': abs(np.mean(Y_samples) - y_mean_theory) / max(abs(y_mean_theory), 1e-10)
            },
            'X_var': {
                'sample': float(np.var(X_samples)),
                'theoretical': x_var_theory,
                'rel_error': abs(np.var(X_samples) - x_var_theory) / max(abs(x_var_theory), 1e-10)
            },
            'Y_var': {
                'sample': float(np.var(Y_samples)),
                'theoretical': y_var_theory,
                'rel_error': abs(np.var(Y_samples) - y_var_theory) / max(abs(y_var_theory), 1e-10)
            }
        }
    else:
        X_samples = dist.rvs(size=n_samples, random_state=random_state)
        
        sample_mean = np.mean(X_samples, axis=0)
        sample_var = np.var(X_samples, axis=0)
        
        theory_mean = np.asarray(dist.mean())
        theory_var = np.asarray(dist.var())
        
        mean_rel_error = np.abs(sample_mean - theory_mean) / np.maximum(np.abs(theory_mean), 1e-10)
        var_rel_error = np.abs(sample_var - theory_var) / np.maximum(np.abs(theory_var), 1e-10)
        
        return {
            'mean': {
                'sample': sample_mean,
                'theoretical': theory_mean,
                'rel_error': mean_rel_error
            },
            'variance': {
                'sample': sample_var,
                'theoretical': theory_var,
                'rel_error': var_rel_error
            }
        }


def print_moment_validation(results: Dict, name: str = "Distribution"):
    """Print formatted moment validation results."""
    print(f"\n{'='*60}")
    print(f"Moment Validation: {name}")
    print(f"{'='*60}")
    
    for key, value in results.items():
        sample = value['sample']
        theory = value['theoretical']
        rel_err = value['rel_error']
        
        if np.ndim(sample) == 0:
            print(f"{key:10s}: sample = {sample:10.4f}, theory = {theory:10.4f}, rel_err = {rel_err:.2e}")
        else:
            sample_str = np.array2string(np.asarray(sample), precision=4, suppress_small=True)
            theory_str = np.array2string(np.asarray(theory), precision=4, suppress_small=True)
            rel_err_str = np.array2string(np.asarray(rel_err), precision=2, suppress_small=True)
            print(f"{key:10s}:")
            print(f"  sample = {sample_str}")
            print(f"  theory = {theory_str}")
            print(f"  rel_err = {rel_err_str}")


def fit_and_track_convergence(
    marginal_dist_class,
    X_data: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42
) -> Tuple[Any, EMConvergenceResult]:
    """
    Fit marginal distribution with EM and track convergence.
    
    Parameters
    ----------
    marginal_dist_class : type
        Marginal distribution class (VarianceGamma, NormalInverseGamma, etc.)
    X_data : ndarray
        Data to fit
    max_iter : int
        Maximum EM iterations
    tol : float
        Convergence tolerance
    random_state : int
        Random seed
        
    Returns
    -------
    fitted_dist : NormalMixture
        Fitted distribution
    convergence : EMConvergenceResult
        Convergence history
    """
    # Create distribution instance
    dist = marginal_dist_class()
    
    # Track log-likelihood at each iteration
    iterations = []
    log_likelihoods = []
    
    # Custom fit with tracking
    X = np.asarray(X_data)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, d = X.shape
    
    # Initialize
    if dist._joint is None:
        dist._joint = dist._create_joint_distribution()
    dist._joint._d = d
    
    # Call fit with verbose to capture progress (we'll parse it)
    class IterationTracker:
        def __init__(self):
            self.iterations = []
            self.log_likelihoods = []
            
    tracker = IterationTracker()
    
    # Fit with tracking - use verbose=2 and capture
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        dist.fit(X, max_iter=max_iter, tol=tol, verbose=2, random_state=random_state)
    finally:
        sys.stdout = old_stdout
    
    # Parse output
    # Format: "Iteration X: log-likelihood = Y" or "Initial log-likelihood: Y"
    output = buffer.getvalue()
    for line in output.split('\n'):
        if 'Iteration' in line and 'log-likelihood' in line:
            try:
                # Format: "Iteration X: log-likelihood = Y"
                iter_part = line.split(':')[0].split()[-1]
                ll_part = line.split('=')[-1].strip()
                tracker.iterations.append(int(iter_part))
                tracker.log_likelihoods.append(float(ll_part))
            except (IndexError, ValueError):
                pass
        elif 'Initial log-likelihood' in line:
            try:
                ll = float(line.split(':')[1].strip())
                tracker.iterations.append(0)
                tracker.log_likelihoods.append(ll)
            except (IndexError, ValueError):
                pass
    
    converged = 'Converged' in output
    
    return dist, EMConvergenceResult(
        iterations=tracker.iterations,
        log_likelihoods=tracker.log_likelihoods,
        converged=converged,
        final_params=dist.get_classical_params()
    )


def plot_em_convergence(
    convergence: EMConvergenceResult,
    title: str = "EM Algorithm Convergence",
    figsize: Tuple[int, int] = (10, 5)
) -> plt.Figure:
    """Plot EM algorithm convergence."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(convergence.iterations, convergence.log_likelihoods, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Log-likelihood', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add convergence annotation
    status = "Converged" if convergence.converged else "Not converged"
    ax.annotate(status, xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=11,
                color='green' if convergence.converged else 'red')
    
    plt.tight_layout()
    return fig


def test_joint_fitting(
    joint_dist_class,
    true_params: Dict[str, Any],
    n_samples: int = 5000,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Test joint distribution fitting (exponential family MLE).
    
    Parameters
    ----------
    joint_dist_class : type
        Joint distribution class
    true_params : dict
        True parameters to generate data from
    n_samples : int
        Number of samples
    random_state : int
        Random seed
        
    Returns
    -------
    fitted_dist : JointNormalMixture
        Fitted distribution
    fitted_params : dict
        Fitted parameters
    param_errors : dict
        Relative errors for each parameter
    """
    # Create true distribution
    true_dist = joint_dist_class.from_classical_params(**true_params)
    
    # Generate data
    X_data, Y_data = true_dist.rvs(size=n_samples, random_state=random_state)
    
    # Fit
    fitted_dist = joint_dist_class().fit(X_data, Y_data)
    fitted_params = fitted_dist.get_classical_params()
    
    # Compute errors
    param_errors = {}
    for key in true_params:
        true_val = np.asarray(true_params[key])
        fitted_val = np.asarray(fitted_params[key])
        if true_val.size == 1:
            true_val = float(true_val.flat[0]) if hasattr(true_val, 'flat') else float(true_val)
            fitted_val = float(fitted_val.flat[0]) if hasattr(fitted_val, 'flat') else float(fitted_val)
            param_errors[key] = abs(true_val - fitted_val) / max(abs(true_val), 1e-10)
        else:
            param_errors[key] = float(np.max(np.abs(true_val - fitted_val) / np.maximum(np.abs(true_val), 1e-10)))
    
    return fitted_dist, fitted_params, param_errors


def print_fitting_results(
    true_params: Dict[str, Any],
    fitted_params: Dict[str, Any],
    param_errors: Dict[str, float],
    name: str = "Distribution"
):
    """Print formatted fitting results."""
    print(f"\n{'='*60}")
    print(f"Fitting Results: {name}")
    print(f"{'='*60}")
    print(f"{'Parameter':15s} {'True':>15s} {'Fitted':>15s} {'Rel.Error':>12s}")
    print(f"{'-'*60}")
    
    for key in true_params:
        true_val = true_params[key]
        fitted_val = fitted_params[key]
        error = param_errors[key]
        
        if np.ndim(true_val) == 0 or (hasattr(true_val, 'size') and true_val.size == 1):
            true_str = f"{float(np.asarray(true_val).flat[0]):>15.4f}"
            fitted_str = f"{float(np.asarray(fitted_val).flat[0]):>15.4f}"
        else:
            true_str = np.array2string(np.asarray(true_val), precision=3)
            fitted_str = np.array2string(np.asarray(fitted_val), precision=3)
        
        print(f"{key:15s} {true_str} {fitted_str} {error:>12.2e}")


def comprehensive_distribution_test(
    joint_dist_class,
    marginal_dist_class,
    param_sets: List[Dict[str, Any]],
    n_samples: int = 5000,
    em_max_iter: int = 100,
    random_state: int = 42,
    show_plots: bool = True
):
    """
    Run comprehensive tests on a mixture distribution.
    
    Parameters
    ----------
    joint_dist_class : type
        Joint distribution class
    marginal_dist_class : type
        Marginal distribution class
    param_sets : list of dict
        List of parameter dictionaries to test
    n_samples : int
        Number of samples for tests
    em_max_iter : int
        Max EM iterations
    random_state : int
        Random seed
    show_plots : bool
        Whether to display plots
    """
    for i, params in enumerate(param_sets):
        print(f"\n{'#'*70}")
        print(f"# Parameter Set {i+1}")
        print(f"{'#'*70}")
        
        for key, val in params.items():
            print(f"  {key}: {val}")
        
        # 1. Joint distribution tests (1D X)
        print(f"\n{'='*60}")
        print("1. Joint Distribution (1D X)")
        print(f"{'='*60}")
        
        # Create 1D version of params
        params_1d = params.copy()
        if 'mu' in params_1d:
            params_1d['mu'] = np.array([params_1d['mu'][0]]) if np.ndim(params_1d['mu']) > 0 else np.array([params_1d['mu']])
        if 'gamma' in params_1d:
            params_1d['gamma'] = np.array([params_1d['gamma'][0]]) if np.ndim(params_1d['gamma']) > 0 else np.array([params_1d['gamma']])
        if 'sigma' in params_1d:
            sigma = np.asarray(params_1d['sigma'])
            params_1d['sigma'] = np.array([[sigma[0, 0]]]) if sigma.ndim > 1 else np.array([[sigma]])
        
        joint_dist = joint_dist_class.from_classical_params(**params_1d)
        
        # Moment validation
        results = validate_moments(joint_dist, n_samples=n_samples, random_state=random_state, is_joint=True)
        print_moment_validation(results, f"Joint {joint_dist_class.__name__}")
        
        # Joint fitting test
        fitted_dist, fitted_params, param_errors = test_joint_fitting(
            joint_dist_class, params_1d, n_samples=n_samples, random_state=random_state
        )
        print_fitting_results(params_1d, fitted_params, param_errors, "Joint Fitting")
        
        # Plot joint distribution
        if show_plots:
            fig = plot_joint_distribution_1d(
                joint_dist, n_samples=n_samples, random_state=random_state,
                title=f"Joint {joint_dist_class.__name__} (Parameter Set {i+1})"
            )
            plt.show()
        
        # 2. Marginal distribution tests (2D X)
        print(f"\n{'='*60}")
        print("2. Marginal Distribution (2D X)")
        print(f"{'='*60}")
        
        marginal_dist = marginal_dist_class.from_classical_params(**params)
        
        # Moment validation
        results = validate_moments(marginal_dist, n_samples=n_samples, random_state=random_state, is_joint=False)
        print_moment_validation(results, f"Marginal {marginal_dist_class.__name__}")
        
        # EM fitting test
        print("\nEM Algorithm Fitting:")
        X_data = marginal_dist.rvs(size=n_samples, random_state=random_state)
        fitted_marginal, convergence = fit_and_track_convergence(
            marginal_dist_class, X_data, max_iter=em_max_iter, random_state=random_state+1
        )
        
        print(f"  Converged: {convergence.converged}")
        print(f"  Iterations: {len(convergence.iterations)}")
        if convergence.log_likelihoods:
            print(f"  Final log-likelihood: {convergence.log_likelihoods[-1]:.4f}")
        
        # Print fitted params comparison
        print("\n  Fitted Parameters:")
        for key, val in convergence.final_params.items():
            true_val = params.get(key)
            if true_val is not None:
                print(f"    {key}: true = {true_val}, fitted = {val}")
        
        # Plot marginal distribution
        if show_plots:
            fig = plot_marginal_distribution_2d(
                marginal_dist, n_samples=n_samples, random_state=random_state,
                title=f"Marginal {marginal_dist_class.__name__} (Parameter Set {i+1})"
            )
            plt.show()
            
            # Plot EM convergence
            if convergence.iterations:
                fig = plot_em_convergence(
                    convergence,
                    title=f"EM Convergence - {marginal_dist_class.__name__} (Parameter Set {i+1})"
                )
                plt.show()
