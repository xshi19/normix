"""
Quick comparison of normix_numpy fit() times for all distribution types.
Uses .fit() directly for reliability, with per-iteration timing via verbose mode.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_returns.csv")
df = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna(axis=1)
X_full = df.values.astype(np.float64)
n_obs, n_stocks = X_full.shape
print(f"Data: {n_obs} obs × {n_stocks} stocks\n", flush=True)

from normix_numpy.distributions.mixtures.generalized_hyperbolic import GeneralizedHyperbolic as GH_np
from normix_numpy.distributions.mixtures.variance_gamma import VarianceGamma as VG_np
from normix_numpy.distributions.mixtures.normal_inverse_gaussian import NormalInverseGaussian as NIG_np
from normix_numpy.distributions.mixtures.normal_inverse_gamma import NormalInverseGamma as NIGa_np

N_ITER = 10

configs = [
    ("10 stocks", X_full[:, :10]),
    (f"{n_stocks} stocks", X_full),
]

dists = [
    ("NIG", NIG_np),
    ("VG", VG_np),
    ("NIG-alpha", NIGa_np),
    ("GH", GH_np),
]

print("=" * 80, flush=True)
print("normix_numpy EM fit() timing", flush=True)
print("=" * 80, flush=True)

results = {}

for config_name, X_data in configs:
    print(f"\n--- {config_name} ({X_data.shape[0]} × {X_data.shape[1]}) ---", flush=True)
    results[config_name] = {}

    for dist_name, dist_cls in dists:
        print(f"\n  {dist_name}:", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = dist_cls()
                t0 = time.perf_counter()
                model.fit(X_data, max_iter=N_ITER, verbose=1, tol=1e-12)
                t_total = time.perf_counter() - t0

            n_iter_done = getattr(model, 'n_iter_', N_ITER)
            avg_iter = t_total / n_iter_done if n_iter_done > 0 else t_total
            print(f"  Total: {t_total:.3f}s, {n_iter_done} iters, avg={avg_iter:.4f}s/iter",
                  flush=True)
            results[config_name][dist_name] = {
                'total': t_total,
                'n_iter': n_iter_done,
                'avg_iter': avg_iter,
            }
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results[config_name][dist_name] = {'error': str(e)}

# Summary table
print("\n" + "=" * 80, flush=True)
print("SUMMARY TABLE (normix_numpy)", flush=True)
print("=" * 80, flush=True)

for config_name, _ in configs:
    print(f"\n--- {config_name} ---", flush=True)
    print(f"  {'Dist':<12} {'Total(s)':<12} {'Iters':<8} {'Avg iter(s)':<14}", flush=True)
    print(f"  {'-'*46}", flush=True)
    for dn, _ in dists:
        r = results.get(config_name, {}).get(dn, {})
        if 'error' in r:
            print(f"  {dn:<12} ERROR: {r['error'][:40]}", flush=True)
        else:
            print(f"  {dn:<12} {r['total']:<12.3f} {r['n_iter']:<8} {r['avg_iter']:<14.4f}",
                  flush=True)

# Cross-reference with JAX results from earlier
print("\n" + "=" * 80, flush=True)
print("CROSS-REFERENCE with JAX (from earlier profiling)", flush=True)
print("=" * 80, flush=True)
print("""
Previous JAX results (avg per iteration, 468 stocks):
  Distribution    JAX CPU iter    JAX GPU iter    NumPy iter
  NIG             1.42s           0.10s           ???
  VG              1.45s           0.31s           ???
  NIG-alpha       1.47s           0.32s           ???
  GH              44.1s           28.3s           0.66s

Fill in ??? from the results above.
""", flush=True)
