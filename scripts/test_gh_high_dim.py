"""
Test script to reproduce and verify the fix for the GH det_sigma_one
regularization bug in high dimensions.

Problem: In high dimensions (e.g., d=468), np.exp(log_det_sigma) underflows
to 0.0 because log_det_sigma is very negative. This causes the regularization
to silently skip, so det(Sigma) != 1 after fitting.

The fix: work entirely in log space in regularize_det_sigma_one.
"""

import numpy as np
import time


def test_regularize_det_sigma_one_high_dim():
    """Test that regularize_det_sigma_one works in high dimensions."""
    from normix.distributions.mixtures.generalized_hyperbolic import (
        regularize_det_sigma_one,
    )

    # Simulate a high-dimensional Sigma with small eigenvalues
    # (typical in financial return data with d=468)
    rng = np.random.default_rng(42)
    d = 468

    # Create a realistic covariance matrix with small eigenvalues
    # Eigenvalues roughly in [0.0001, 0.1] range
    eigenvalues = rng.uniform(0.0001, 0.1, size=d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    sigma = Q @ np.diag(eigenvalues) @ Q.T
    sigma = (sigma + sigma.T) / 2  # ensure symmetry

    det_sigma = np.linalg.det(sigma)
    sign, log_det_sigma = np.linalg.slogdet(sigma)
    print(f"d = {d}")
    print(f"np.linalg.det(sigma) = {det_sigma}")
    print(f"np.linalg.slogdet(sigma) = (sign={sign}, logdet={log_det_sigma:.2f})")
    print(f"np.exp(log_det_sigma) = {np.exp(log_det_sigma)}")
    print()

    # Dummy parameters
    mu = np.zeros(d)
    gamma = rng.standard_normal(d) * 0.01
    p, a, b = 1.0, 1.0, 1.0

    # Test with log_det_sigma provided (the path used during EM fitting)
    result = regularize_det_sigma_one(
        mu, gamma, sigma, p, a, b, d, log_det_sigma=log_det_sigma
    )

    # Check if regularization was actually applied
    result_det = np.linalg.det(result['sigma'])
    _, result_log_det = np.linalg.slogdet(result['sigma'])
    print("After regularization:")
    print(f"  np.linalg.det(result_sigma) = {result_det}")
    print(f"  slogdet(result_sigma) = logdet={result_log_det:.6f}")
    print(f"  Expected: logdet ≈ 0.0 (i.e., det ≈ 1.0)")

    # The bug: if log_det_sigma is very negative, np.exp(log_det_sigma) = 0.0
    # and regularization is skipped
    if abs(result_log_det) > 0.1:
        print(f"\n  *** BUG: det(Sigma) is NOT close to 1! log_det = {result_log_det:.2f} ***")
        return False
    else:
        print(f"\n  OK: det(Sigma) ≈ 1 (log_det = {result_log_det:.6f})")
        return True


def test_gh_fit_medium_dim():
    """Test GH fit with det_sigma_one regularization at moderate dimension."""
    from normix.distributions.mixtures.generalized_hyperbolic import (
        GeneralizedHyperbolic,
    )

    rng = np.random.default_rng(42)
    d = 50  # moderate dimension where det already underflows
    n = 500

    # Generate synthetic data from a multivariate normal (simple case)
    mu_true = rng.standard_normal(d) * 0.01
    L = np.eye(d) * 0.1  # diagonal L for simple covariance
    X = rng.standard_normal((n, d)) @ L.T + mu_true

    print(f"\nFitting GH with d={d}, n={n}...")
    start = time.time()
    gh = GeneralizedHyperbolic()
    gh.fit(X, max_iter=10, tol=1e-2, verbose=1, regularization='det_sigma_one')
    elapsed = time.time() - start
    print(f"Fitting time: {elapsed:.2f}s")

    sigma = gh.classical_params['sigma']
    det_sigma = np.linalg.det(sigma)
    _, log_det_sigma = np.linalg.slogdet(sigma)
    print(f"\nAfter fitting:")
    print(f"  np.linalg.det(sigma) = {det_sigma}")
    print(f"  slogdet(sigma) = logdet={log_det_sigma:.6f}")

    if abs(log_det_sigma) > 0.1:
        print(f"\n  *** BUG: det(Sigma) is NOT close to 1! ***")
        return False
    else:
        print(f"\n  OK: det(Sigma) ≈ 1")
        return True


if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: regularize_det_sigma_one with high-dim Sigma")
    print("=" * 70)
    ok1 = test_regularize_det_sigma_one_high_dim()

    print("\n" + "=" * 70)
    print("Test 2: GH fit with det_sigma_one at moderate dimension")
    print("=" * 70)
    ok2 = test_gh_fit_medium_dim()

    print("\n" + "=" * 70)
    if ok1 and ok2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
