"""
Verify GIG fit() after Bug 1-3 fixes. Compare with scipy floc=0.
"""

import numpy as np
import pandas as pd
from scipy.stats import geninvgauss

from normix.distributions.univariate import GeneralizedInverseGaussian

y = pd.read_csv("data/gig_test.csv").iloc[:, 0].values
n = len(y)
eta_hat = np.array([np.mean(np.log(y)), np.mean(y ** -1), np.mean(y)])

# ── scipy reference ──────────────────────────────────────────────────
scipy_params = geninvgauss.fit(y, floc=0)
p_sc, b_sc, _, scale_sc = scipy_params
gig_scipy = GeneralizedInverseGaussian.from_scipy_params(p_sc, b_sc, scale_sc)
ll_scipy = np.mean(gig_scipy.logpdf(y))

print("=" * 72)
print("scipy fit (floc=0)")
print("=" * 72)
print(f"  (p, a, b)    = ({gig_scipy._p:.8f}, {gig_scipy._a:.10f}, {gig_scipy._b:.8f})")
print(f"  mean ℓ       = {ll_scipy:.10f}")
print(f"  η(θ) - η̂    = {gig_scipy.expectation_params - eta_hat}")
print()

# ── Our fit ──────────────────────────────────────────────────────────
print("=" * 72)
print("GeneralizedInverseGaussian().fit(y)")
print("=" * 72)
try:
    gig_fit = GeneralizedInverseGaussian()
    gig_fit.fit(y)
    ll_fit = np.mean(gig_fit.logpdf(y))
    print(f"  (p, a, b)    = ({gig_fit._p:.8f}, {gig_fit._a:.10f}, {gig_fit._b:.8f})")
    print(f"  mean ℓ       = {ll_fit:.10f}")
    print(f"  η(θ) - η̂    = {gig_fit.expectation_params - eta_hat}")
    print(f"  ℓ(fit) - ℓ(scipy) = {ll_fit - ll_scipy:.6e}")
    if ll_fit >= ll_scipy - 1e-8:
        print(f"  ✓ fit() log-likelihood >= scipy")
    else:
        print(f"  ✗ fit() log-likelihood < scipy by {ll_scipy - ll_fit:.6e}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# ── Also test with well-posed synthetic data ─────────────────────────
print("=" * 72)
print("Synthetic GIG(p=-1, a=2, b=3) fit test")
print("=" * 72)
gig_true = GeneralizedInverseGaussian.from_classical_params(p=-1.0, a=2.0, b=3.0)
np.random.seed(42)
y_synth = gig_true.rvs(size=5000, random_state=42)

gig_synth = GeneralizedInverseGaussian()
gig_synth.fit(y_synth)
print(f"  True:  (p=-1.0, a=2.0, b=3.0)")
print(f"  Fit:   (p={gig_synth._p:.4f}, a={gig_synth._a:.4f}, b={gig_synth._b:.4f})")
eta_synth = np.array([np.mean(np.log(y_synth)), np.mean(y_synth**-1), np.mean(y_synth)])
print(f"  |η(θ)-η̂|∞ = {np.max(np.abs(gig_synth.expectation_params - eta_synth)):.2e}")
print()

# ── Boundary case: data that genuinely fits InvGamma ─────────────────
print("=" * 72)
print("InvGamma data → GIG.fit() should degenerate gracefully")
print("=" * 72)
from scipy.stats import invgamma as invgamma_scipy
np.random.seed(123)
y_ig = invgamma_scipy.rvs(a=3.0, scale=2.0, size=2000)

gig_ig = GeneralizedInverseGaussian()
gig_ig.fit(y_ig)
print(f"  True InvGamma: α=3.0, β=2.0  →  GIG(p=-3.0, a=0, b=4.0)")
print(f"  Fit:           (p={gig_ig._p:.4f}, a={gig_ig._a:.6f}, b={gig_ig._b:.4f})")
ll_gig = np.mean(gig_ig.logpdf(y_ig))
print(f"  mean ℓ(GIG) = {ll_gig:.10f}")

# Compare to fitting InvGamma directly
from normix.distributions.univariate import InverseGamma
ig_fit = InverseGamma()
ig_fit.fit(y_ig)
ll_ig = np.mean(ig_fit.logpdf(y_ig))
print(f"  mean ℓ(InvGamma) = {ll_ig:.10f}")
print(f"  ℓ(GIG) - ℓ(IG) = {ll_gig - ll_ig:.6e}")
print()
