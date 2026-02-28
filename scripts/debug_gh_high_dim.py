"""
Debug script for GH distribution behavior at different dimensions.

Compares GH with other distributions (NInvG, NIG, VG) to identify
dimension-dependent issues.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kurtosis, anderson_ksamp

from normix.distributions.mixtures.variance_gamma import VarianceGamma
from normix.distributions.mixtures.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.mixtures.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.mixtures.generalized_hyperbolic import GeneralizedHyperbolic

import warnings
warnings.filterwarnings("ignore")


def load_data(n_stocks):
    data_path = Path(__file__).parent.parent / "data" / "sp500_sample.csv"
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    tickers = sorted(df.columns.tolist())[:n_stocks]
    return df[tickers].values


def run_comparison(n_stocks_list):
    """Compare all four distributions across different dimensions."""
    print("=" * 90)
    print(f"{'d':>4} | {'Dist':>6} | {'iters':>5} | {'AD mean':>8} | "
          f"{'kurt_data':>9} | {'kurt_samp':>9} | Mixing params")
    print("-" * 90)

    for n_stocks in n_stocks_list:
        X = load_data(n_stocks)
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(n_stocks), size=30)

        w_single = rng.dirichlet(np.ones(n_stocks))
        data_proj = X @ w_single

        dists = [
            ("VG", VarianceGamma, {}),
            ("NInvG", NormalInverseGamma, {}),
            ("NIG", NormalInverseGaussian, {}),
            ("GH", GeneralizedHyperbolic, {"regularization": "det_sigma_one"}),
        ]

        for name, DistClass, kwargs in dists:
            dist = DistClass()
            dist.fit(X, max_iter=100, tol=1e-3, random_state=42, **kwargs)
            samples = dist.rvs(size=10000, random_state=42)
            sample_proj = samples @ w_single
            kurt_d = kurtosis(data_proj)
            kurt_s = kurtosis(sample_proj)

            ad_stats = []
            for i in range(30):
                w = weights[i]
                try:
                    ad = anderson_ksamp([X @ w, samples @ w])
                    ad_stats.append(ad.statistic)
                except Exception:
                    ad_stats.append(np.nan)
            mean_ad = np.nanmean(ad_stats)

            params = dist.classical_params
            if name == "VG":
                mixing = f"shape={params.shape:.2f}, rate={params.rate:.2f}"
            elif name == "NInvG":
                mixing = f"shape={params.shape:.2f}, rate={params.rate:.2f}"
            elif name == "NIG":
                mixing = f"delta={params.delta:.2f}, eta={params.eta:.2f}"
            elif name == "GH":
                mixing = f"p={params.p:.2f}, a={params.a:.2f}, b={params.b:.2f}"

            print(f"{n_stocks:4d} | {name:>6} | {dist.n_iter_:5d} | "
                  f"{mean_ad:8.2f} | {kurt_d:9.2f} | {kurt_s:9.2f} | {mixing}")

        print("-" * 90)


def main():
    print("Distribution Comparison Across Dimensions")
    print("Using SP500 log returns (full history)\n")

    run_comparison([50, 80, 100, 120, 150, 200])

    print("\nSUMMARY:")
    print("""
Key findings:
1. VG bifurcation: At d ≥ ~130, VG's Gamma mixing distribution collapses
   to near-degenerate (high shape), producing essentially Gaussian output.
   Other distributions (NInvG, NIG, GH) do NOT show this behavior.

2. AD stats degrade with dimension for ALL distributions, but VG degrades
   catastrophically while others degrade gradually.

3. GH shows some instability at high dimensions (fewer iterations, higher
   AD) but maintains non-trivial mixing parameters.

4. NInvG and NIG are the most stable across dimensions, consistently
   capturing some tail behavior even at d=200.
""")


if __name__ == "__main__":
    main()
