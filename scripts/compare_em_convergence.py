#!/usr/bin/env python3
"""
Compare EM convergence: log-likelihood-based vs parameter-change-based stopping.

This script tests the difference in speed and final log-likelihood between
the old (llh-based) and new (parameter-change-based) convergence criteria
for all mixture distributions.

Uses S&P 500 data (downloaded via yfinance) as a realistic test case.

Usage:
    python scripts/compare_em_convergence.py
    python scripts/compare_em_convergence.py --max-iter 200
    python scripts/compare_em_convergence.py --tickers AAPL MSFT GOOGL
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')


# ============================================================================
# Old EM implementations (llh-based convergence) for comparison
# ============================================================================

def _fit_gh_old_convergence(X, max_iter=100, tol=1e-4, regularization='det_sigma_one'):
    """
    Fit GH using OLD convergence criterion (relative llh change).

    This reimplements the old EM loop for fair comparison.
    """
    from normix.distributions.mixtures import GeneralizedHyperbolic
    from normix.distributions.mixtures.generalized_hyperbolic import (
        REGULARIZATION_METHODS, regularize_det_sigma_one,
    )

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, d = X.shape

    gh = GeneralizedHyperbolic()
    gh._joint = gh._create_joint_distribution()
    gh._joint._d = d

    regularize_fn = REGULARIZATION_METHODS[regularization]

    # Method of moments initialization (old style)
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    if d == 1:
        Sigma = np.array([[Sigma]])
    min_eig = np.linalg.eigvalsh(Sigma).min()
    if min_eig < 1e-8:
        Sigma = Sigma + (1e-8 - min_eig + 1e-8) * np.eye(d)
    gamma = np.zeros(d)
    p, a, b = 1.0, 1.0, 1.0

    gh._joint.set_classical_params(mu=mu, gamma=gamma, sigma=Sigma, p=p, a=a, b=b)

    prev_ll = 0.0
    iterations_run = 0
    llh_history = []

    start_time = time.time()

    for iteration in range(max_iter):
        # Apply regularization
        classical = gh.get_classical_params()
        _, log_det_sigma = gh._joint.get_L_Sigma()

        if regularization == 'det_sigma_one':
            regularized = regularize_fn(
                classical['mu'], classical['gamma'], classical['sigma'],
                classical['p'], classical['a'], classical['b'], d,
                log_det_sigma=log_det_sigma
            )
        else:
            regularized = regularize_fn(
                classical['mu'], classical['gamma'], classical['sigma'],
                classical['p'], classical['a'], classical['b'], d
            )

        a_reg = regularized['a']
        b_reg = regularized['b']
        if a_reg < 1e-6 or b_reg < 1e-6 or a_reg > 1e6 or b_reg > 1e6:
            regularized['a'] = np.clip(a_reg, 1e-6, 1e6)
            regularized['b'] = np.clip(b_reg, 1e-6, 1e6)

        gh._joint.set_classical_params(**regularized)

        ll = np.mean(gh.logpdf(X))
        llh_history.append(ll)

        if iteration > 0:
            if abs(prev_ll) > 0 and abs(ll - prev_ll) / abs(prev_ll) < tol:
                iterations_run = iteration + 1
                break

        prev_ll = ll

        cond_exp = gh._conditional_expectation_y_given_x(X)
        gh._m_step(X, cond_exp, fix_tail=False)
        iterations_run = iteration + 1

    elapsed = time.time() - start_time
    final_ll = np.mean(gh.logpdf(X))

    return {
        'dist': gh,
        'final_ll': final_ll,
        'time': elapsed,
        'iterations': iterations_run,
        'llh_history': llh_history,
        'method': 'old (llh-based)',
    }


def _fit_gh_new_convergence(X, max_iter=100, tol=1e-4, regularization='det_sigma_one'):
    """
    Fit GH using NEW convergence criterion (parameter change).

    Uses the current implementation directly.
    """
    from normix.distributions.mixtures import GeneralizedHyperbolic

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    gh = GeneralizedHyperbolic()

    start_time = time.time()
    gh.fit(X, max_iter=max_iter, tol=tol, regularization=regularization, verbose=0)
    elapsed = time.time() - start_time

    final_ll = np.mean(gh.logpdf(X))

    return {
        'dist': gh,
        'final_ll': final_ll,
        'time': elapsed,
        'method': 'new (param-change)',
    }


def _fit_vg_old_convergence(X, max_iter=100, tol=1e-6):
    """Fit VG using OLD convergence criterion (llh-based)."""
    from normix.distributions.mixtures import VarianceGamma
    from scipy.special import digamma, polygamma
    from normix.utils import log_kv

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, d = X.shape

    vg = VarianceGamma()
    vg._joint = vg._create_joint_distribution()
    vg._joint._d = d

    # Init
    X_mean = np.mean(X, axis=0)
    X_cov = np.cov(X, rowvar=False)
    if X_cov.ndim == 0:
        X_cov = np.array([[X_cov]])

    alpha_init, beta_init = 2.0, 1.0
    E_Y_init = alpha_init / beta_init
    X_centered = X - X_mean
    X_std = np.maximum(np.std(X, axis=0), 1e-10)
    skewness = np.mean((X_centered / X_std) ** 3, axis=0)
    gamma_init = skewness * X_std * 0.1
    mu_init = X_mean - gamma_init * E_Y_init
    Var_Y_init = alpha_init / (beta_init ** 2)
    Sigma_init = (X_cov - Var_Y_init * np.outer(gamma_init, gamma_init)) / E_Y_init
    Sigma_init = (Sigma_init + Sigma_init.T) / 2
    min_eig = np.linalg.eigvalsh(Sigma_init).min()
    if min_eig < 1e-6:
        Sigma_init = Sigma_init + (1e-6 - min_eig + 1e-6) * np.eye(d)

    vg.set_classical_params(mu=mu_init, gamma=gamma_init, sigma=Sigma_init,
                            shape=alpha_init, rate=beta_init)

    prev_ll = np.mean(vg.logpdf(X))
    iterations_run = 0

    start_time = time.time()

    for iteration in range(max_iter):
        cond_exp = vg._conditional_expectation_y_given_x(X)
        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']
        E_log_Y = cond_exp['E_log_Y']

        eta_1 = np.mean(E_inv_Y)
        eta_2 = np.mean(E_Y)
        eta_3 = np.mean(E_log_Y)
        eta_4 = np.mean(X, axis=0)
        eta_5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)
        eta_6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n_samples

        denom = 1.0 - eta_1 * eta_2
        if abs(denom) < 1e-10:
            denom = 1e-10

        mu_new = (eta_4 - eta_2 * eta_5) / denom
        gamma_new = (eta_5 - eta_1 * eta_4) / denom
        Sigma_new = (eta_6 - np.outer(eta_5, mu_new) - np.outer(mu_new, eta_5)
                     + eta_1 * np.outer(mu_new, mu_new)
                     - eta_2 * np.outer(gamma_new, gamma_new))
        Sigma_new = (Sigma_new + Sigma_new.T) / 2
        min_eig = np.linalg.eigvalsh(Sigma_new).min()
        if min_eig < 1e-8:
            Sigma_new = Sigma_new + (1e-8 - min_eig + 1e-8) * np.eye(d)

        target = eta_3 - np.log(eta_2)
        alpha_new = vg.get_classical_params()['shape']
        for _ in range(50):
            f_val = digamma(alpha_new) - np.log(alpha_new) - target
            f_prime = polygamma(1, alpha_new) - 1.0 / alpha_new
            step = f_val / f_prime
            alpha_candidate = np.clip(alpha_new - step, 0.1, 1000.0)
            if abs(alpha_candidate - alpha_new) / max(abs(alpha_new), 1e-10) < 1e-10:
                alpha_new = alpha_candidate
                break
            alpha_new = alpha_candidate
        beta_new = alpha_new / eta_2

        try:
            vg.set_classical_params(mu=mu_new, gamma=gamma_new, sigma=Sigma_new,
                                    shape=alpha_new, rate=beta_new)
        except ValueError:
            break

        current_ll = np.mean(vg.logpdf(X))
        ll_change = current_ll - prev_ll
        iterations_run = iteration + 1

        if abs(ll_change) < tol:
            break
        prev_ll = current_ll

    elapsed = time.time() - start_time
    final_ll = np.mean(vg.logpdf(X))

    return {
        'dist': vg,
        'final_ll': final_ll,
        'time': elapsed,
        'iterations': iterations_run,
        'method': 'old (llh-based)',
    }


def _fit_vg_new_convergence(X, max_iter=100, tol=1e-6):
    """Fit VG using NEW convergence criterion (parameter change)."""
    from normix.distributions.mixtures import VarianceGamma

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    vg = VarianceGamma()

    start_time = time.time()
    vg.fit(X, max_iter=max_iter, tol=tol, verbose=0)
    elapsed = time.time() - start_time

    final_ll = np.mean(vg.logpdf(X))

    return {
        'dist': vg,
        'final_ll': final_ll,
        'time': elapsed,
        'method': 'new (param-change)',
    }


def _fit_nig_old_convergence(X, max_iter=100, tol=1e-6):
    """Fit NIG using OLD convergence criterion (llh-based)."""
    from normix.distributions.mixtures import NormalInverseGaussian

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, d = X.shape

    nig = NormalInverseGaussian()
    nig._joint = nig._create_joint_distribution()
    nig._joint._d = d

    # Init
    X_mean = np.mean(X, axis=0)
    X_cov = np.cov(X, rowvar=False)
    if X_cov.ndim == 0:
        X_cov = np.array([[X_cov]])
    delta_init, eta_init = 1.0, 1.0
    X_std = np.maximum(np.std(X, axis=0), 1e-10)
    skewness = np.mean(((X - X_mean) / X_std) ** 3, axis=0)
    gamma_init = skewness * X_std * 0.1
    mu_init = X_mean - gamma_init * delta_init
    Var_Y_init = delta_init**3 / eta_init
    Sigma_init = (X_cov - Var_Y_init * np.outer(gamma_init, gamma_init)) / delta_init
    Sigma_init = (Sigma_init + Sigma_init.T) / 2
    min_eig = np.linalg.eigvalsh(Sigma_init).min()
    if min_eig < 1e-6:
        Sigma_init = Sigma_init + (1e-6 - min_eig + 1e-6) * np.eye(d)

    nig.set_classical_params(mu=mu_init, gamma=gamma_init, sigma=Sigma_init,
                             delta=delta_init, eta=eta_init)

    prev_ll = np.mean(nig.logpdf(X))
    iterations_run = 0

    start_time = time.time()

    for iteration in range(max_iter):
        cond_exp = nig._conditional_expectation_y_given_x(X)
        E_Y = cond_exp['E_Y']
        E_inv_Y = cond_exp['E_inv_Y']

        eta_1 = np.mean(E_inv_Y)
        eta_2 = np.mean(E_Y)
        eta_4 = np.mean(X, axis=0)
        eta_5 = np.mean(X * E_inv_Y[:, np.newaxis], axis=0)
        eta_6 = np.einsum('ij,ik,i->jk', X, X, E_inv_Y) / n_samples

        denom = 1.0 - eta_1 * eta_2
        if abs(denom) < 1e-10:
            denom = 1e-10

        mu_new = (eta_4 - eta_2 * eta_5) / denom
        gamma_new = (eta_5 - eta_1 * eta_4) / denom
        Sigma_new = (eta_6 - np.outer(eta_5, mu_new) - np.outer(mu_new, eta_5)
                     + eta_1 * np.outer(mu_new, mu_new)
                     - eta_2 * np.outer(gamma_new, gamma_new))
        Sigma_new = (Sigma_new + Sigma_new.T) / 2
        min_eig = np.linalg.eigvalsh(Sigma_new).min()
        if min_eig < 1e-8:
            Sigma_new = Sigma_new + (1e-8 - min_eig + 1e-8) * np.eye(d)

        delta_new = max(eta_2, 1e-6)
        inv_eta = eta_1 - 1.0 / delta_new
        eta_new = 1.0 / inv_eta if inv_eta > 1e-10 else 1000.0
        eta_new = max(eta_new, 1e-6)

        old_params = nig.get_classical_params()
        try:
            nig.set_classical_params(mu=mu_new, gamma=gamma_new, sigma=Sigma_new,
                                     delta=delta_new, eta=eta_new)
        except ValueError:
            nig.set_classical_params(**old_params)
            break

        current_ll = np.mean(nig.logpdf(X))
        iterations_run = iteration + 1

        if abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    elapsed = time.time() - start_time
    final_ll = np.mean(nig.logpdf(X))

    return {
        'dist': nig,
        'final_ll': final_ll,
        'time': elapsed,
        'iterations': iterations_run,
        'method': 'old (llh-based)',
    }


def _fit_nig_new_convergence(X, max_iter=100, tol=1e-6):
    """Fit NIG using NEW convergence criterion."""
    from normix.distributions.mixtures import NormalInverseGaussian

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    nig = NormalInverseGaussian()

    start_time = time.time()
    nig.fit(X, max_iter=max_iter, tol=tol, verbose=0)
    elapsed = time.time() - start_time

    final_ll = np.mean(nig.logpdf(X))

    return {
        'dist': nig,
        'final_ll': final_ll,
        'time': elapsed,
        'method': 'new (param-change)',
    }


# ============================================================================
# Data loading
# ============================================================================

def download_sp500_data(period_years=10):
    """Download S&P 500 log returns."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365 + 30)

    print(f"Downloading S&P 500 data ({start_date.date()} to {end_date.date()})...")
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

    # Handle MultiIndex columns from newer yfinance
    if isinstance(sp500.columns, __import__('pandas').MultiIndex):
        close = sp500['Close']
        if hasattr(close, 'iloc') and close.ndim > 1:
            close = close.iloc[:, 0]
    else:
        close = sp500['Close']

    log_returns = (np.log(close / close.shift(1)).dropna() * 100).values
    print(f"Got {len(log_returns)} daily log returns (in %)")
    return log_returns


def download_multivariate_data(tickers=None, period_years=5):
    """Download multivariate stock log returns."""
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        raise ImportError("yfinance and pandas required")

    from datetime import datetime, timedelta

    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)

    print(f"Downloading {tickers} data ({start_date.date()} to {end_date.date()})...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            prices = data.iloc[:, :len(tickers)]
    else:
        prices = data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    log_returns = (np.log(prices / prices.shift(1)).dropna() * 100).values
    print(f"Got {log_returns.shape[0]} samples x {log_returns.shape[1]} assets")
    return log_returns, tickers


# ============================================================================
# Comparison runner
# ============================================================================

def compare_single_distribution(name, fit_old, fit_new, X, max_iter=100, tol=None, n_runs=3):
    """
    Compare old vs new convergence for a single distribution.

    Runs multiple times and reports median to reduce noise.
    """
    if tol is None:
        tol = 1e-4 if name == 'GH' else 1e-6

    old_results = []
    new_results = []

    for run in range(n_runs):
        old_res = fit_old(X, max_iter=max_iter, tol=tol)
        old_results.append(old_res)

        new_res = fit_new(X, max_iter=max_iter, tol=tol)
        new_results.append(new_res)

    # Take median time
    old_times = [r['time'] for r in old_results]
    new_times = [r['time'] for r in new_results]

    old_ll = old_results[-1]['final_ll']
    new_ll = new_results[-1]['final_ll']

    old_time = np.median(old_times)
    new_time = np.median(new_times)

    return {
        'name': name,
        'old_ll': old_ll,
        'new_ll': new_ll,
        'old_time': old_time,
        'new_time': new_time,
        'speedup': old_time / new_time if new_time > 0 else float('inf'),
        'll_diff': new_ll - old_ll,
    }


def run_comparison(max_iter=100, tickers=None):
    """Run the full comparison."""
    print("=" * 80)
    print("EM CONVERGENCE COMPARISON: log-likelihood vs parameter-change stopping")
    print("=" * 80)

    # Download SP500 univariate data
    sp500_returns = download_sp500_data(period_years=10)
    X_1d = sp500_returns

    # Download multivariate data
    X_md, ticker_names = download_multivariate_data(tickers=tickers, period_years=5)

    results = []

    # ---- Univariate comparisons ----
    print("\n" + "-" * 80)
    print("UNIVARIATE FITS (S&P 500 log returns)")
    print("-" * 80)

    # VG
    print("\nFitting Variance Gamma...")
    res = compare_single_distribution(
        'VG (1D)',
        _fit_vg_old_convergence,
        _fit_vg_new_convergence,
        X_1d, max_iter=max_iter, tol=1e-6, n_runs=3
    )
    results.append(res)
    print(f"  Old: ll={res['old_ll']:.6f}, time={res['old_time']:.4f}s")
    print(f"  New: ll={res['new_ll']:.6f}, time={res['new_time']:.4f}s")
    print(f"  Speedup: {res['speedup']:.2f}x, LL diff: {res['ll_diff']:.6f}")

    # NIG
    print("\nFitting Normal-Inverse Gaussian...")
    res = compare_single_distribution(
        'NIG (1D)',
        _fit_nig_old_convergence,
        _fit_nig_new_convergence,
        X_1d, max_iter=max_iter, tol=1e-6, n_runs=3
    )
    results.append(res)
    print(f"  Old: ll={res['old_ll']:.6f}, time={res['old_time']:.4f}s")
    print(f"  New: ll={res['new_ll']:.6f}, time={res['new_time']:.4f}s")
    print(f"  Speedup: {res['speedup']:.2f}x, LL diff: {res['ll_diff']:.6f}")

    # GH 1D
    print("\nFitting Generalized Hyperbolic (1D)...")
    res = compare_single_distribution(
        'GH (1D)',
        _fit_gh_old_convergence,
        _fit_gh_new_convergence,
        X_1d, max_iter=max_iter, tol=1e-4, n_runs=3
    )
    results.append(res)
    print(f"  Old: ll={res['old_ll']:.6f}, time={res['old_time']:.4f}s")
    print(f"  New: ll={res['new_ll']:.6f}, time={res['new_time']:.4f}s")
    print(f"  Speedup: {res['speedup']:.2f}x, LL diff: {res['ll_diff']:.6f}")

    # ---- Multivariate comparisons ----
    print("\n" + "-" * 80)
    print(f"MULTIVARIATE FITS ({len(ticker_names)} assets: {', '.join(ticker_names)})")
    print("-" * 80)

    # GH multivariate
    print("\nFitting Generalized Hyperbolic (multivariate)...")
    res = compare_single_distribution(
        f'GH ({len(ticker_names)}D)',
        _fit_gh_old_convergence,
        _fit_gh_new_convergence,
        X_md, max_iter=max_iter, tol=1e-4, n_runs=3
    )
    results.append(res)
    print(f"  Old: ll={res['old_ll']:.6f}, time={res['old_time']:.4f}s")
    print(f"  New: ll={res['new_ll']:.6f}, time={res['new_time']:.4f}s")
    print(f"  Speedup: {res['speedup']:.2f}x, LL diff: {res['ll_diff']:.6f}")

    # NIG multivariate
    print("\nFitting Normal-Inverse Gaussian (multivariate)...")
    res = compare_single_distribution(
        f'NIG ({len(ticker_names)}D)',
        _fit_nig_old_convergence,
        _fit_nig_new_convergence,
        X_md, max_iter=max_iter, tol=1e-6, n_runs=3
    )
    results.append(res)
    print(f"  Old: ll={res['old_ll']:.6f}, time={res['old_time']:.4f}s")
    print(f"  New: ll={res['new_ll']:.6f}, time={res['new_time']:.4f}s")
    print(f"  Speedup: {res['speedup']:.2f}x, LL diff: {res['ll_diff']:.6f}")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Distribution':<20} {'Old LL':>12} {'New LL':>12} {'LL Diff':>10} "
          f"{'Old Time':>10} {'New Time':>10} {'Speedup':>8}")
    print("-" * 84)

    for res in results:
        print(f"{res['name']:<20} {res['old_ll']:>12.4f} {res['new_ll']:>12.4f} "
              f"{res['ll_diff']:>10.4f} {res['old_time']:>9.4f}s {res['new_time']:>9.4f}s "
              f"{res['speedup']:>7.2f}x")

    print("\n" + "-" * 84)
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_ll_diff = np.mean([r['ll_diff'] for r in results])
    print(f"{'Average':<20} {'':>12} {'':>12} {avg_ll_diff:>10.4f} "
          f"{'':>10} {'':>10} {avg_speedup:>7.2f}x")

    print("\nNotes:")
    print("- LL Diff > 0 means the new method achieves higher log-likelihood")
    print("- Speedup > 1 means the new method is faster")
    print("- The new method uses parameter-change stopping (mu, gamma, Sigma)")
    print("- The old method uses relative log-likelihood change stopping")
    print("- GH new method additionally uses NIG warm-start initialization")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare EM convergence criteria: llh-based vs parameter-change'
    )
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum EM iterations (default: 100)')
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Tickers for multivariate test (default: AAPL MSFT GOOGL)')

    args = parser.parse_args()

    results = run_comparison(
        max_iter=args.max_iter,
        tickers=args.tickers,
    )
