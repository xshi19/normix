"""Plotting utilities for normix notebooks.

Requires the ``plotting`` extra: ``pip install normix[plotting]``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as mgs
except ImportError as e:
    raise ImportError(
        "matplotlib is required for normix.utils.plotting. "
        "Install it with: pip install normix[plotting]"
    ) from e

from scipy.stats import gaussian_kde

import jax
import jax.numpy as jnp


_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Golden ratio for figure proportions
PHI = (1 + np.sqrt(5)) / 2
FIG_W = 12
FIG_H = FIG_W / PHI


def plot_pdf_cdf_comparison(
    configs: List[Dict[str, Any]],
    x: np.ndarray,
    xlabel: str = "x",
    title: str = "",
) -> plt.Figure:
    """
    Plot PDF and CDF comparing normix vs scipy for multiple distributions.

    Parameters
    ----------
    configs : list of dicts with keys {label, dist, scipy}
    x       : 1-D evaluation grid
    """
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    for i, cfg in enumerate(configs):
        color = _COLORS[i % len(_COLORS)]
        lbl = cfg["label"]
        dist = cfg["dist"]
        sp = cfg["scipy"]

        normix_pdf = np.asarray(jnp.exp(jax.vmap(dist.log_prob)(jnp.asarray(x))))
        scipy_pdf = sp.pdf(x)
        scipy_cdf = sp.cdf(x)

        try:
            normix_cdf = np.asarray(jax.vmap(dist.cdf)(jnp.asarray(x)))
        except NotImplementedError:
            from scipy.integrate import cumulative_trapezoid
            normix_cdf = cumulative_trapezoid(normix_pdf, x, initial=0.0)
            normix_cdf = np.clip(normix_cdf, 0.0, 1.0)

        axes[0].plot(x, normix_pdf, "-", lw=2, label=lbl, color=color, alpha=0.85)
        axes[0].plot(x, scipy_pdf, "o", ms=2.5, color=color, alpha=0.35)
        axes[1].plot(x, normix_cdf, "-", lw=2, label=lbl, color=color, alpha=0.85)
        axes[1].plot(x, scipy_cdf, "o", ms=2.5, color=color, alpha=0.35)

    axes[0].set(xlabel=xlabel, ylabel="PDF",
                title="Probability Density Functions\n(normix — , scipy ·)")
    axes[1].set(xlabel=xlabel, ylabel="CDF",
                title="Cumulative Distribution Functions\n(normix — , scipy ·)")
    for ax in axes:
        ax.legend(fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_sample_histograms(
    configs: List[Dict[str, Any]],
    ncols: int = 2,
) -> plt.Figure:
    """
    Grid of histograms with theoretical PDF overlay.

    Parameters
    ----------
    configs : list of dicts with keys {label, dist, samples, x_plot (optional)}
    """
    n = len(configs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_W, FIG_H * nrows))
    axes = np.asarray(axes).flatten()

    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        samples = cfg["samples"]
        x_plot = cfg.get("x_plot",
                         np.linspace(samples.min() * 0.9,
                                     np.percentile(samples, 99), 200))
        pdf = np.asarray(jnp.exp(jax.vmap(cfg["dist"].log_prob)(jnp.asarray(x_plot))))

        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples))

        ax.hist(samples, bins=60, density=True, alpha=0.55, color="steelblue",
                edgecolor="white", linewidth=0.3, label="Samples")
        ax.plot(x_plot, pdf, "r-", lw=2.5, label="Theoretical PDF")

        extra = f"  mean={sample_mean:.3f}, std={sample_std:.3f}"
        ax.set(xlabel="x", ylabel="Density", title=cfg["label"] + extra)
        ax.legend(fontsize=9)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    return fig


def plot_mle_fit(
    data: np.ndarray,
    fit_results: List[Dict[str, Any]],
    xlabel: str = "x",
    title: str = "Maximum Likelihood Estimation",
) -> plt.Figure:
    """
    Histogram of data with multiple PDF overlays for MLE comparison.

    Parameters
    ----------
    data        : 1-D sample array
    fit_results : list of dicts {label, dist, ls, color}
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(data, bins=60, density=True, alpha=0.5, color="lightsteelblue",
            edgecolor="white", linewidth=0.3, label="Data")
    x_plot = np.linspace(data.min(), np.percentile(data, 99.5), 400)
    for fr in fit_results:
        pdf = np.asarray(jnp.exp(jax.vmap(fr["dist"].log_prob)(jnp.asarray(x_plot))))
        ax.plot(x_plot, pdf, lw=2.5, ls=fr.get("ls", "-"),
                color=fr.get("color", None), label=fr["label"])
    ax.set(xlabel=xlabel, ylabel="Density", title=title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_joint_1d(
    joint_dist,
    n_samples: int = 5000,
    seed: int = 42,
    title: str = "",
) -> plt.Figure:
    """
    Visualize a 1-D joint distribution f(x, y).

    Left: scatter X vs Y.  Right: marginal histogram of X.
    """
    X, Y = joint_dist.rvs(n_samples, seed)
    x_vals = X[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    sc = axes[0].scatter(x_vals, Y, c=Y, cmap="viridis", s=5, alpha=0.4)
    plt.colorbar(sc, ax=axes[0], label="Y")
    axes[0].set(xlabel="X", ylabel="Y (mixing variable)",
                title="Joint scatter: X vs Y")

    x_grid = np.linspace(np.percentile(x_vals, 0.5),
                         np.percentile(x_vals, 99.5), 300)
    axes[1].hist(x_vals, bins=60, density=True, alpha=0.55, color="steelblue",
                 edgecolor="white", linewidth=0.3, label="Samples")
    kde = gaussian_kde(x_vals, bw_method="scott")
    axes[1].plot(x_grid, kde(x_grid), "r-", lw=2.5, label="KDE density")
    axes[1].set(xlabel="X", ylabel="Density", title="Marginal histogram of X")
    axes[1].legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


def plot_marginal_2d(
    marginal_dist,
    n_samples: int = 5000,
    seed: int = 42,
    title: str = "",
) -> plt.Figure:
    """
    Visualize a 2-D marginal distribution: scatter + marginal histograms.
    """
    X = marginal_dist.rvs(n_samples, seed)

    fig = plt.figure(figsize=(10, 10 / PHI * 1.2))
    gs = mgs.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                      hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(X[:, 0], X[:, 1], s=4, alpha=0.3, color="steelblue")
    ax_main.set(xlabel="X₁", ylabel="X₂")

    ax_top.hist(X[:, 0], bins=50, density=True, color="steelblue",
                alpha=0.6, edgecolor="white")
    ax_right.hist(X[:, 1], bins=50, density=True, color="steelblue",
                  alpha=0.6, edgecolor="white", orientation="horizontal")

    for ax_h, vals, orient in [(ax_top, X[:, 0], "v"), (ax_right, X[:, 1], "h")]:
        kde = gaussian_kde(vals)
        grid = np.linspace(vals.min(), vals.max(), 200)
        if orient == "v":
            ax_h.plot(grid, kde(grid), "r-", lw=1.8)
        else:
            ax_h.plot(kde(grid), grid, "r-", lw=1.8)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_em_convergence(
    log_likelihoods: List[float],
    title: str = "EM Convergence",
    true_ll: Optional[float] = None,
) -> plt.Figure:
    """Plot EM log-likelihood convergence curve."""
    fig, ax = plt.subplots(figsize=(FIG_W * 0.7, FIG_H * 0.85))
    iters = np.arange(1, len(log_likelihoods) + 1)
    ax.plot(iters, log_likelihoods, "b-o", ms=4, lw=2, label="EM log-likelihood")
    if true_ll is not None:
        ax.axhline(true_ll, color="green", ls="--", lw=1.8,
                   label=f"True LL = {true_ll:.4f}")
    ax.set(xlabel="Iteration", ylabel="Mean log-likelihood", title=title)
    ax.legend()
    plt.tight_layout()
    return fig
