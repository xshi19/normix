"""Moment validation and parameter printing utilities for normix notebooks."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def validate_moments(
    dist,
    n_samples: int = 20000,
    seed: int = 42,
    is_joint: bool = True,
) -> Dict[str, Any]:
    """
    Validate E[X] and E[Y] by comparing samples vs analytical values.

    Uses dist.mean() for theoretical E[X] and subordinator().mean() for E[Y].
    """
    if is_joint:
        X, Y = dist.rvs(n_samples, seed)
        theory_EY = float(dist.subordinator().mean())
        theory_EX = np.asarray(dist.mu) + np.asarray(dist.gamma) * theory_EY
        sample_EX = np.mean(X, axis=0)
        sample_EY = float(np.mean(Y))
        return {
            "sample_mean_X": sample_EX,
            "theory_mean_X": theory_EX,
            "mean_error_X": np.abs(sample_EX - theory_EX),
            "sample_mean_Y": sample_EY,
            "theory_mean_Y": theory_EY,
            "mean_error_Y": abs(sample_EY - theory_EY),
            "n_samples": n_samples,
        }
    else:
        X = dist.rvs(n_samples, seed)
        theory_EX = np.asarray(dist.mean())
        sample_EX = np.mean(X, axis=0)
        sample_cov = np.atleast_2d(np.cov(X.T))
        theory_cov = np.asarray(dist.cov())
        return {
            "sample_mean_X": sample_EX,
            "theory_mean_X": theory_EX,
            "mean_error_X": np.abs(sample_EX - theory_EX),
            "sample_cov_X": sample_cov,
            "theory_cov_X": theory_cov,
            "n_samples": n_samples,
        }


def print_moment_validation(results: Dict[str, Any], title: str = "") -> None:
    """Print moment validation results."""
    sep = "=" * 60
    if title:
        print(f"\n{sep}")
        print(f"  Moment Validation — {title}")
        print(sep)
    print(f"\nSample size: {results['n_samples']:,}")
    print(f"\nE[X]  theory  : {results['theory_mean_X']}")
    print(f"E[X]  sample  : {results['sample_mean_X']}")
    print(f"E[X]  |error| : {results['mean_error_X']}")
    if "sample_mean_Y" in results:
        print(f"\nE[Y]  theory  : {results['theory_mean_Y']:.6f}")
        print(f"E[Y]  sample  : {results['sample_mean_Y']:.6f}")
        print(f"E[Y]  |error| : {results['mean_error_Y']:.6f}")
    if "theory_cov_X" in results:
        print(f"\nCov[X] (theory):\n{results['theory_cov_X']}")
        print(f"Cov[X] (sample):\n{results['sample_cov_X']}")


def print_exp_family_params(dist, label: str = "") -> None:
    """Print natural and expectation parameters."""
    if label:
        print(f"\n{'='*60}")
        print(f"  Exponential Family Parameters — {label}")
        print("=" * 60)
    theta = np.asarray(dist.natural_params())
    eta = np.asarray(dist.expectation_params())
    print(f"\nNatural params   θ : {theta}")
    print(f"Expectation params η: {eta}")


