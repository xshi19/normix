"""
Debug script for GH parameter instability across dimensions.

The GH distribution shows a parameter regime change around d ≈ 100:
- d < 100: p ≈ -2, a ≈ 0.01 (close to Student-t, captures heavy tails well)
- d >= 100: p ≈ -0.45, a ≈ 1-2 (different mixing regime, poorer tail capture)

This script investigates the transition and its impact on fit quality.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis, anderson_ksamp

from normix.distributions.mixtures.generalized_hyperbolic import GeneralizedHyperbolic

import warnings
warnings.filterwarnings("ignore")


def load_data(n_stocks):
    data_path = Path(__file__).parent.parent / "data" / "sp500_sample.csv"
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    tickers = sorted(df.columns.tolist())[:n_stocks]
    return df[tickers].values


def analyze_gh_across_dims():
    """Track GH parameters and fit quality across dimensions."""
    print("GH Parameter Stability Analysis")
    print("=" * 90)
    print(f"{'d':>4} | {'iters':>5} | {'p':>7} | {'a':>7} | {'b':>7} | "
          f"{'AD mean':>8} | {'kurt_d':>7} | {'kurt_s':>7} | {'LL_train':>10}")
    print("-" * 90)

    for d in [30, 50, 70, 80, 90, 100, 110, 120, 150, 200]:
        X = load_data(d)
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(d), size=30)
        w_single = rng.dirichlet(np.ones(d))

        gh = GeneralizedHyperbolic()
        gh.fit(X, max_iter=100, tol=1e-3,
               regularization="det_sigma_one", random_state=42)

        samples = gh.rvs(size=10000, random_state=42)
        data_proj = X @ w_single
        sample_proj = samples @ w_single

        ad_stats = []
        for i in range(30):
            w = weights[i]
            try:
                ad = anderson_ksamp([X @ w, samples @ w])
                ad_stats.append(ad.statistic)
            except Exception:
                ad_stats.append(np.nan)

        p = gh.classical_params
        ll = np.mean(gh.logpdf(X))

        print(f"{d:4d} | {gh.n_iter_:5d} | {p.p:7.3f} | {p.a:7.3f} | "
              f"{p.b:7.3f} | {np.nanmean(ad_stats):8.2f} | "
              f"{kurtosis(data_proj):7.2f} | {kurtosis(sample_proj):7.2f} | "
              f"{ll:10.2f}")

    print("-" * 90)


def compare_gh_regularizations():
    """Compare different GH regularization strategies."""
    print("\n\nGH Regularization Comparison at d=100")
    print("=" * 80)

    X = load_data(100)
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(100), size=30)
    w_single = rng.dirichlet(np.ones(100))
    data_proj = X @ w_single

    regs = [
        ("det_sigma_one", {}),
        ("fix_p", {"p_fixed": -0.5}),
        ("fix_p", {"p_fixed": -1.0}),
        ("fix_p", {"p_fixed": -2.0}),
    ]

    for reg_name, reg_params in regs:
        gh = GeneralizedHyperbolic()
        gh.fit(X, max_iter=100, tol=1e-3,
               regularization=reg_name,
               regularization_params=reg_params,
               random_state=42)

        samples = gh.rvs(size=10000, random_state=42)
        sample_proj = samples @ w_single

        ad_stats = []
        for i in range(30):
            w = weights[i]
            try:
                ad = anderson_ksamp([X @ w, samples @ w])
                ad_stats.append(ad.statistic)
            except Exception:
                ad_stats.append(np.nan)

        p = gh.classical_params
        ll = np.mean(gh.logpdf(X))
        label = f"{reg_name}"
        if reg_params:
            label += f"(p={reg_params.get('p_fixed', '')})"

        print(f"  {label:25s}: iters={gh.n_iter_:2d}, p={p.p:.3f}, "
              f"a={p.a:.3f}, b={p.b:.3f}, "
              f"AD={np.nanmean(ad_stats):.2f}, "
              f"kurt_s={kurtosis(sample_proj):.2f}, LL={ll:.2f}")


def main():
    analyze_gh_across_dims()
    compare_gh_regularizations()

    print("\n\nSUMMARY:")
    print("""
Key findings:
1. GH with det_sigma_one regularization shows a parameter regime change
   around d ≈ 100 where p jumps from about -2 to -0.45.

2. At d < 100 (p ≈ -2, a ≈ 0), GH behaves like a Student-t distribution,
   which captures heavy tails very effectively.

3. At d >= 100 (p ≈ -0.45, a > 1), GH moves to a different mixing regime
   that captures less tail kurtosis. This leads to higher AD stats.

4. The fix_p regularization with p = -2.0 may help maintain the
   Student-t-like behavior at higher dimensions.

5. Unlike the VG bifurcation (which is catastrophic), the GH transition
   is more gradual and the model still captures some tail behavior.
""")


if __name__ == "__main__":
    main()
