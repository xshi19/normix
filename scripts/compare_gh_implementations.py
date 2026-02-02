#!/usr/bin/env python3
"""
Compare the new GeneralizedHyperbolic implementation (normix.distributions.mixtures)
with the legacy GH implementation (normix.legacy.gh).

This script:
1. Downloads MAG7 stock prices from Yahoo Finance (last 5 years)
2. Computes log returns
3. Fits a MULTIVARIATE GH distribution to all stocks jointly
4. Compares: parameters, likelihoods, and fitting speed

Parameterization mapping between implementations:
- New (normix.distributions): mu, gamma, sigma, p, a, b
- Legacy (normix.legacy):     mu, gamma, sigma, lam, chi, psi

Mapping: p = lam, a = psi, b = chi
"""

import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime, timedelta

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def download_stock_data(tickers, period_years=5):
    """
    Download stock data from Yahoo Finance.
    
    Parameters
    ----------
    tickers : list of str
        Stock tickers to download.
    period_years : int
        Number of years of historical data.
        
    Returns
    -------
    prices : pd.DataFrame
        Adjusted close prices for each ticker.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required. Install with: pip install yfinance"
        )
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    
    print(f"Downloading {len(tickers)} stocks from {start_date.date()} to {end_date.date()}...")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Handle both old and new yfinance API formats
    if isinstance(data.columns, pd.MultiIndex):
        # New format: MultiIndex columns (Price, Ticker)
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            prices = data.xs('Close', axis=1, level=0) if 'Close' in data.columns.get_level_values(0) else data.iloc[:, :len(tickers)]
    else:
        # Old format: simple columns
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
    
    # If single ticker, ensure it's a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    
    print(f"Downloaded {len(prices)} days of data.")
    return prices


def compute_log_returns(prices):
    """
    Compute log returns from price series.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data with tickers as columns.
        
    Returns
    -------
    returns : pd.DataFrame
        Log returns (percentage).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna() * 100  # in percentage
    return log_returns


def fit_new_implementation(returns_matrix, max_iter=100, verbose=0):
    """
    Fit the new GeneralizedHyperbolic implementation (MULTIVARIATE).
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2D array of log returns, shape (n_samples, n_assets).
    max_iter : int
        Maximum EM iterations.
    verbose : int
        Verbosity level.
        
    Returns
    -------
    result : dict
        Dictionary with fitted parameters, log-likelihood, and timing.
    """
    from normix.distributions.mixtures import GeneralizedHyperbolic
    
    n_samples, d = returns_matrix.shape
    
    gh = GeneralizedHyperbolic()
    
    start_time = time.time()
    gh.fit(returns_matrix, max_iter=max_iter, verbose=verbose, regularization='det_sigma_one')
    elapsed = time.time() - start_time
    
    # Get parameters
    params = gh.get_classical_params()
    
    # Compute log-likelihood (per sample)
    ll = np.mean(gh.logpdf(returns_matrix))
    
    return {
        'mu': params['mu'],
        'gamma': params['gamma'],
        'sigma': params['sigma'],
        'p': params['p'],
        'a': params['a'],
        'b': params['b'],
        'log_likelihood': ll,
        'time': elapsed,
        'dist': gh,
        'd': d,
        'n_samples': n_samples
    }


def fit_legacy_implementation(returns_matrix, max_iter=100, verbose=False):
    """
    Fit the legacy GH implementation (MULTIVARIATE).
    
    Parameters
    ----------
    returns_matrix : np.ndarray
        2D array of log returns, shape (n_samples, n_assets).
    max_iter : int
        Maximum EM iterations.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    result : dict
        Dictionary with fitted parameters, log-likelihood, and timing.
    """
    # Import legacy GH - need to handle its internal imports
    from scipy.special import kv, gammaln, factorial
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize
    from scipy.linalg import solve_triangular
    
    # Define legacy helper functions inline (from normix/legacy/func.py)
    def logkv(v, z):
        """Log modified Bessel function of the second kind"""
        y = np.log(kv(v, z))
        
        # y -> -infinity when z -> infinity
        if np.any(y == -np.inf):
            k = np.arange(1, 10)
            akv = np.cumprod(4*v**2 - (2*k-1)**2) / factorial(k) / 8**k
            y[y == -np.inf] = -z[y == -np.inf] - 0.5*np.log(z[y == -np.inf]) + 0.5*np.log(np.pi/2) \
                            + np.log(1 + np.sum(akv/np.power.outer(z[y == -np.inf], k), axis=1))
        
        # y -> infinity when z -> 0
        if np.any(y == np.inf):
            if np.abs(v) > 1e-10:
                y[y == np.inf] = gammaln(np.abs(v)) - np.log(2.) \
                                + np.abs(v) * (np.log(2.) - np.log(z[y == np.inf]))
            else:
                y[y == np.inf] = np.log(-np.log(z[y == np.inf]/2) - np.euler_gamma)
        
        return y
    
    def logkvp(v, z):
        return -0.5*(np.exp(logkv(v+1, z)-logkv(v, z))+np.exp(logkv(v-1, z)-logkv(v, z)))
    
    def kvratio(v1, v2, z):
        return np.exp(logkv(v1, z) - logkv(v2, z))
    
    # Define legacy GIG class inline
    class GIG:
        """Generalized Inverse Gaussian (GIG) - Legacy"""
        
        def __init__(self, lam, chi, psi):
            self.lam = lam
            self.chi = chi
            self.psi = psi
        
        def moment(self, alpha):
            delta = np.sqrt(self.chi/self.psi)
            eta = np.sqrt(self.chi*self.psi)
            m = delta**alpha * np.exp(logkv(self.lam+alpha, eta) - logkv(self.lam, eta))
            return m
        
        def mean(self):
            return self.moment(1)
        
        def var(self):
            return self.moment(2) - self.moment(1)**2
        
        def suffstats2param(self, s1, s2, s3):
            def llh(param):
                lam = param[0]
                chi = param[1]
                psi = param[2]
                eta = np.array([np.sqrt(chi*psi)])
                delta = np.sqrt(chi/psi)
                l = 0.5*(chi*s1 + psi*s2) - (lam-1)*s3 + lam*np.log(delta) + logkv(lam, eta)
                return l[0]
            
            def grad(param):
                lam = param[0]
                chi = param[1]
                psi = param[2]
                eta = np.array([np.sqrt(chi*psi)])
                delta = np.sqrt(chi/psi)
                g = np.array([(-s3 + np.log(delta) + (logkv(lam+1e-10, eta)[0] - logkv(lam-1e-10, eta)[0])/2e-10),
                              0.5*(s1 + lam/chi + logkvp(lam, eta)[0]/delta),
                              0.5*(s2 - lam/psi + logkvp(lam, eta)[0]*delta)])
                return g
            
            x0 = np.array([self.lam, self.chi, self.psi])
            bounds = [(-20., 20.), (1e-20, None), (1e-20, None)]
            res = minimize(fun=llh, x0=x0, jac=grad, bounds=bounds)
            
            if res.success:
                self.lam = res.x[0]
                self.chi = res.x[1]
                self.psi = res.x[2]
            return res
    
    # Define legacy GH class inline
    class GH:
        """Generalized Hyperbolic (GH) - Legacy"""
        
        def __init__(self, mu, gamma, sigma, lam, chi, psi):
            self.mu = mu
            self.gamma = gamma
            self.sigma = sigma
            self.lam = lam
            self.chi = chi
            self.psi = psi
            self.dim = len(mu)
        
        def llh(self, x):
            l = np.linalg.cholesky(self.sigma)
            z = solve_triangular(l, (x-self.mu).T, lower=True)
            gamma = solve_triangular(l, self.gamma, lower=True)
            lam = self.lam - self.dim/2
            chi = self.chi + np.sum(z**2, axis=0)
            psi = self.psi + np.dot(gamma, gamma)
            
            delta = np.sqrt(chi/psi)
            eta = np.sqrt(chi*psi)
            
            llh = self.lam/2*np.log(self.psi/self.chi) \
                - logkv(self.lam, np.sqrt(self.chi*self.psi)) \
                + logkv(lam, eta) + np.dot(z.T, gamma) + lam*np.log(delta)
            return llh, lam, delta, eta
        
        def regulate(self, method='|sigma|=1'):
            if method == 'chi=1':
                self.gamma = self.gamma * self.chi
                self.sigma = self.sigma * self.chi
                self.psi = self.psi * self.chi
                self.chi = 1.0
            elif method == 'psi=1':
                self.gamma = self.gamma / self.psi
                self.sigma = self.sigma / self.psi
                self.chi = self.chi * self.psi
                self.psi = 1.0
            elif method == 'chi=psi':
                self.gamma = self.gamma * np.sqrt(self.chi/self.psi)
                self.sigma = self.sigma * np.sqrt(self.chi/self.psi)
                self.chi = np.sqrt(self.chi*self.psi)
                self.psi = self.chi
            elif method == '|sigma|=1':
                _, logd = np.linalg.slogdet(self.sigma)
                const = np.exp(logd/self.dim)
                self.gamma = self.gamma / const
                self.sigma = self.sigma / const
                self.chi = self.chi * const
                self.psi = self.psi / const
        
        def fit_em(self, x, max_iter=100, fix_tail=False, eps=1e-4, reg='|sigma|=1', diff=1e-5, disp=True):
            if self.dim != x.shape[1]:
                raise ValueError('x dimension must be (, {})'.format(self.dim))
            
            suff_stats = [0]*6
            suff_stats[3] = np.mean(x, axis=0)
            
            llh_last = 0
            for i in range(max_iter):
                self.regulate(reg)
                llh, lam, delta, eta = self.llh(x)
                llh = np.mean(llh)
                
                if i > 0:
                    if disp:
                        print('iter=%s, llh=%.5f, change=%.5f' % (i, llh, llh-llh_last))
                    
                    if (np.abs(llh-llh_last)/np.abs(llh_last)) < eps:
                        if disp:
                            print('success')
                        return True
                llh_last = llh
                
                # E-step
                a = kvratio(lam-1, lam, eta) / delta
                b = kvratio(lam+1, lam, eta) * delta
                c = (kvratio(lam+diff, lam, eta) - kvratio(lam-diff, lam, eta)) / (2*diff) + np.log(delta)
                
                suff_stats[0] = np.mean(a)
                suff_stats[1] = np.mean(b)
                suff_stats[2] = np.mean(c)
                suff_stats[4] = np.mean(x.T * a, axis=1)
                suff_stats[5] = np.dot((x.T * a) / x.shape[0], x)
                
                if not fix_tail:
                    gig = GIG(self.lam, self.chi, self.psi)
                    res = gig.suffstats2param(suff_stats[0], suff_stats[1], suff_stats[2])
                    if res.success:
                        self.lam = gig.lam
                        self.chi = gig.chi
                        self.psi = gig.psi
                
                self.mu = (suff_stats[3] - suff_stats[1]*suff_stats[4]) / (1 - suff_stats[0]*suff_stats[1])
                self.gamma = (suff_stats[4] - suff_stats[0]*suff_stats[3]) / (1 - suff_stats[0]*suff_stats[1])
                self.sigma = -np.outer(suff_stats[4], self.mu)
                self.sigma = self.sigma + self.sigma.T + suff_stats[5] \
                    + suff_stats[0]*np.outer(self.mu, self.mu) \
                    - suff_stats[1]*np.outer(self.gamma, self.gamma)
            
            if disp:
                print('fail to converge')
            return False
    
    n_samples, d = returns_matrix.shape
    
    # Initialize with reasonable defaults
    mu_init = np.mean(returns_matrix, axis=0)
    gamma_init = np.zeros(d)
    sigma_init = np.cov(returns_matrix, rowvar=False)
    
    # Ensure sigma is positive definite
    min_eig = np.linalg.eigvalsh(sigma_init).min()
    if min_eig < 1e-8:
        sigma_init = sigma_init + (1e-8 - min_eig + 1e-8) * np.eye(d)
    
    lam_init = 1.0
    chi_init = 1.0
    psi_init = 1.0
    
    gh = GH(mu_init, gamma_init, sigma_init, lam_init, chi_init, psi_init)
    
    start_time = time.time()
    converged = gh.fit_em(returns_matrix, max_iter=max_iter, reg='|sigma|=1', disp=verbose)
    elapsed = time.time() - start_time
    
    # Compute log-likelihood (per sample)
    llh_result = gh.llh(returns_matrix)
    ll = np.mean(llh_result[0])
    
    return {
        'mu': gh.mu,
        'gamma': gh.gamma,
        'sigma': gh.sigma,
        'lam': gh.lam,
        'chi': gh.chi,
        'psi': gh.psi,
        # Mapped to new parameterization
        'p': gh.lam,
        'a': gh.psi,
        'b': gh.chi,
        'log_likelihood': ll,
        'time': elapsed,
        'converged': converged,
        'dist': gh,
        'd': d,
        'n_samples': n_samples
    }


def print_comparison(new_result, legacy_result, tickers):
    """
    Print comparison between new and legacy implementations.
    
    Parameters
    ----------
    new_result : dict
        Results from new implementation.
    legacy_result : dict
        Results from legacy implementation.
    tickers : list of str
        Ticker symbols (for labeling dimensions).
    """
    d = new_result['d']
    
    print("\n" + "=" * 100)
    print("MULTIVARIATE GH COMPARISON RESULTS")
    print(f"Dimension: {d} assets, Samples: {new_result['n_samples']}")
    print("=" * 100)
    
    # GIG parameters comparison
    print("\n--- GIG Parameter Comparison ---")
    print(f"{'Parameter':<12} {'New':>15} {'Legacy':>15} {'Difference':>15}")
    print("-" * 60)
    
    params = [
        ('p (lam)', new_result['p'], legacy_result['p']),
        ('a (psi)', new_result['a'], legacy_result['a']),
        ('b (chi)', new_result['b'], legacy_result['b']),
    ]
    
    for name, new_val, legacy_val in params:
        diff = new_val - legacy_val
        print(f"{name:<12} {new_val:>15.6f} {legacy_val:>15.6f} {diff:>15.6f}")
    
    # Mu comparison
    print("\n--- Location Parameter (mu) Comparison ---")
    print(f"{'Asset':<8} {'New':>12} {'Legacy':>12} {'Difference':>12}")
    print("-" * 48)
    
    for i, ticker in enumerate(tickers):
        new_val = new_result['mu'][i]
        legacy_val = legacy_result['mu'][i]
        diff = new_val - legacy_val
        print(f"{ticker:<8} {new_val:>12.4f} {legacy_val:>12.4f} {diff:>12.4f}")
    
    # Gamma comparison
    print("\n--- Skewness Parameter (gamma) Comparison ---")
    print(f"{'Asset':<8} {'New':>12} {'Legacy':>12} {'Difference':>12}")
    print("-" * 48)
    
    for i, ticker in enumerate(tickers):
        new_val = new_result['gamma'][i]
        legacy_val = legacy_result['gamma'][i]
        diff = new_val - legacy_val
        print(f"{ticker:<8} {new_val:>12.4f} {legacy_val:>12.4f} {diff:>12.4f}")
    
    # Sigma diagonal comparison
    print("\n--- Covariance Diagonal (Sigma) Comparison ---")
    print(f"{'Asset':<8} {'New':>12} {'Legacy':>12} {'Difference':>12}")
    print("-" * 48)
    
    for i, ticker in enumerate(tickers):
        new_val = new_result['sigma'][i, i]
        legacy_val = legacy_result['sigma'][i, i]
        diff = new_val - legacy_val
        print(f"{ticker:<8} {new_val:>12.4f} {legacy_val:>12.4f} {diff:>12.4f}")
    
    # Sigma correlation comparison
    print("\n--- Correlation Matrix Comparison (New vs Legacy) ---")
    
    # Convert covariance to correlation
    new_sigma = new_result['sigma']
    legacy_sigma = legacy_result['sigma']
    
    new_std = np.sqrt(np.diag(new_sigma))
    legacy_std = np.sqrt(np.diag(legacy_sigma))
    
    new_corr = new_sigma / np.outer(new_std, new_std)
    legacy_corr = legacy_sigma / np.outer(legacy_std, legacy_std)
    
    print("\nNew Implementation Correlations:")
    print("        " + "".join([f"{t:>8}" for t in tickers]))
    for i, t1 in enumerate(tickers):
        row = f"{t1:<8}"
        for j, t2 in enumerate(tickers):
            row += f"{new_corr[i, j]:>8.3f}"
        print(row)
    
    print("\nLegacy Implementation Correlations:")
    print("        " + "".join([f"{t:>8}" for t in tickers]))
    for i, t1 in enumerate(tickers):
        row = f"{t1:<8}"
        for j, t2 in enumerate(tickers):
            row += f"{legacy_corr[i, j]:>8.3f}"
        print(row)
    
    print("\nCorrelation Difference (New - Legacy):")
    print("        " + "".join([f"{t:>8}" for t in tickers]))
    for i, t1 in enumerate(tickers):
        row = f"{t1:<8}"
        for j, t2 in enumerate(tickers):
            row += f"{new_corr[i, j] - legacy_corr[i, j]:>8.3f}"
        print(row)
    
    # Log-likelihood comparison
    print("\n--- Log-Likelihood Comparison ---")
    print(f"{'Metric':<25} {'New':>15} {'Legacy':>15}")
    print("-" * 60)
    print(f"{'Log-likelihood (per sample)':<25} {new_result['log_likelihood']:>15.4f} {legacy_result['log_likelihood']:>15.4f}")
    print(f"{'Fitting time (seconds)':<25} {new_result['time']:>15.4f} {legacy_result['time']:>15.4f}")
    print(f"{'Speed ratio (legacy/new)':<25} {legacy_result['time']/new_result['time']:>15.2f}x")
    
    # Summary
    print("\n--- Summary ---")
    mu_diff = np.linalg.norm(new_result['mu'] - legacy_result['mu'])
    gamma_diff = np.linalg.norm(new_result['gamma'] - legacy_result['gamma'])
    sigma_diff = np.linalg.norm(new_result['sigma'] - legacy_result['sigma'], 'fro')
    
    print(f"Mu difference (L2 norm): {mu_diff:.6f}")
    print(f"Gamma difference (L2 norm): {gamma_diff:.6f}")
    print(f"Sigma difference (Frobenius norm): {sigma_diff:.6f}")
    print(f"p difference: {abs(new_result['p'] - legacy_result['p']):.6f}")
    print(f"a difference: {abs(new_result['a'] - legacy_result['a']):.6f}")
    print(f"b difference: {abs(new_result['b'] - legacy_result['b']):.6f}")


def analyze_potential_issues(new_result, legacy_result):
    """
    Analyze potential issues causing differences between implementations.
    """
    print("\n" + "=" * 100)
    print("POTENTIAL ISSUES ANALYSIS")
    print("=" * 100)
    
    print("\n" + "-" * 80)
    print("OBSERVATION: GIG Parameter Comparison")
    print("-" * 80)
    print(f"""
New implementation:  p={new_result['p']:.4f}, a={new_result['a']:.6f}, b={new_result['b']:.6f}
Legacy implementation: p={legacy_result['p']:.4f}, a={legacy_result['a']:.6f}, b={legacy_result['b']:.6f}

If 'a' is very large and 'b' is near 0 in the new implementation, this indicates
the GIG expectation_to_natural conversion is diverging.
""")
    
    # Check for extreme values
    if new_result['a'] > 1e6 or new_result['b'] < 1e-6:
        print("WARNING: New implementation shows extreme GIG parameters!")
        print("This suggests the expectation_to_natural conversion failed.")
    
    print("\n--- ROOT CAUSE ANALYSIS ---")
    print("""
================================================================================
ISSUE #1: GIG PARAMETER UPDATE DIVERGENCE
================================================================================

The new implementation's M-step for GIG parameters uses:
    gig.set_expectation_params(gig_eta)  # gig_eta = [E[log Y], E[1/Y], E[Y]]
    
This calls expectation_to_natural() which uses Newton's method to invert:
    η = ∇ψ(θ)  →  θ = (∇ψ)^{-1}(η)

PROBLEM: The GIG expectation-to-natural mapping is highly non-convex.
For certain η values (especially in high dimensions), the Newton iterations 
fail to converge, leading to extreme parameter values.

The legacy implementation's approach is more robust:
    gig.suffstats2param(s1, s2, s3)  # Direct optimization with bounds
    
It uses scipy.optimize.minimize with explicit bounds:
    bounds = [(-20., 20.), (1e-20, None), (1e-20, None)]

================================================================================
ISSUE #2: REGULARIZATION TIMING
================================================================================

Legacy: regulate() called at START of each iteration
New:    regularization applied at END of each iteration (after M-step)

================================================================================
ISSUE #3: E-STEP COMPUTATION
================================================================================

Legacy uses Cholesky-based computation (more numerically stable):
    z = L^{-1}(x - μ)  where L = chol(Σ)
    
New computes directly with precision matrix:
    Λ = Σ^{-1}

================================================================================
ISSUE #4: MULTIVARIATE SCALING
================================================================================

In high dimensions (d=7), numerical issues become more pronounced:
- Matrix conditioning affects stability
- More parameters to estimate
- GIG parameter space becomes harder to navigate

================================================================================
RECOMMENDATIONS
================================================================================

1. Replace GIG expectation_to_natural with bounded optimization fallback
2. Use Cholesky decomposition instead of matrix inversion in E-step
3. Add parameter bounds to prevent divergence
4. Consider fixing p (lambda) initially for better stability
""")


def run_comparison(tickers=None, period_years=5, max_iter=100, verbose=0):
    """
    Run the full comparison between implementations.
    
    Parameters
    ----------
    tickers : list of str, optional
        Stock tickers. Defaults to MAG7.
    period_years : int
        Years of historical data.
    max_iter : int
        Maximum EM iterations.
    verbose : int
        Verbosity level.
        
    Returns
    -------
    results : dict
        Dictionary of comparison results.
    """
    if tickers is None:
        # MAG7 stocks (Magnificent 7)
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    print("=" * 100)
    print("MULTIVARIATE GH Implementation Comparison")
    print("New (normix.distributions) vs Legacy (normix.legacy)")
    print("=" * 100)
    
    # Download data
    prices = download_stock_data(tickers, period_years)
    log_returns = compute_log_returns(prices)
    
    # Drop any rows with missing values
    log_returns = log_returns.dropna()
    
    print(f"\nLog returns statistics:")
    print(log_returns.describe())
    
    # Convert to numpy array for fitting
    returns_matrix = log_returns.values
    n_samples, d = returns_matrix.shape
    
    print(f"\nFitting {d}-dimensional GH distribution to {n_samples} samples...")
    
    # Fit new implementation
    print(f"\n--- Fitting NEW implementation ---")
    try:
        new_result = fit_new_implementation(returns_matrix, max_iter=max_iter, verbose=verbose)
        print(f"  Time: {new_result['time']:.4f}s")
        print(f"  Log-likelihood (per sample): {new_result['log_likelihood']:.4f}")
        print(f"  GIG params: p={new_result['p']:.4f}, a={new_result['a']:.6f}, b={new_result['b']:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        new_result = None
    
    # Fit legacy implementation
    print(f"\n--- Fitting LEGACY implementation ---")
    try:
        legacy_result = fit_legacy_implementation(returns_matrix, max_iter=max_iter, verbose=(verbose > 0))
        print(f"  Time: {legacy_result['time']:.4f}s")
        print(f"  Log-likelihood (per sample): {legacy_result['log_likelihood']:.4f}")
        print(f"  GIG params: lam={legacy_result['lam']:.4f}, chi={legacy_result['chi']:.6f}, psi={legacy_result['psi']:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        legacy_result = None
    
    # Print comparison if both succeeded
    if new_result is not None and legacy_result is not None:
        print_comparison(new_result, legacy_result, tickers)
        analyze_potential_issues(new_result, legacy_result)
    
    return {
        'new_result': new_result,
        'legacy_result': legacy_result,
        'log_returns': log_returns,
        'tickers': tickers
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare multivariate GH implementations')
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Stock tickers to analyze (default: MAG7)')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data (default: 5)')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum EM iterations (default: 100)')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level (default: 0)')
    
    args = parser.parse_args()
    
    results = run_comparison(
        tickers=args.tickers,
        period_years=args.years,
        max_iter=args.max_iter,
        verbose=args.verbose
    )
