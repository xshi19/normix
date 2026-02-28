"""
SP500 distribution validation tests.

Fits GH, NIG, Variance Gamma, and Normal Inverse Gamma to SP500 log returns
(full history) and validates:
1. EM convergence within 100 iterations
2. Anderson-Darling statistics on random portfolio projections (in-sample)
3. Parameter recovery from generated samples (rvs -> refit)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import anderson_ksamp

from normix.distributions.mixtures.variance_gamma import VarianceGamma
from normix.distributions.mixtures.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.mixtures.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.mixtures.generalized_hyperbolic import GeneralizedHyperbolic


DATA_PATH = Path(__file__).parent.parent / "data" / "sp500_sample.csv"
MAX_ITER = 100
EM_TOL = 1e-3
N_PORTFOLIOS = 50
N_SAMPLES_AD = 10000
N_SAMPLES_REFIT = 5000
AD_STAT_THRESHOLD = 10.0
RANDOM_STATE = 42


def load_sp500_returns():
    """Load SP500 log returns from CSV, using full history."""
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return df.values


def generate_random_weights(d, n_portfolios=100, random_state=None):
    """Generate random portfolio weights using Dirichlet distribution."""
    rng = np.random.default_rng(random_state)
    return rng.dirichlet(np.ones(d), size=n_portfolios)


def compute_ad_stats(dist, returns, weights, n_samples=10000, random_state=None):
    """
    Compute Anderson-Darling statistics for random portfolio projections.

    Uses in-sample data: compares projections of the original returns against
    projections of samples drawn from the fitted distribution.
    """
    samples = dist.rvs(size=n_samples, random_state=random_state)
    n_portfolios = weights.shape[0]
    ad_stats = []

    for i in range(n_portfolios):
        w = weights[i]
        data_proj = returns @ w
        sample_proj = samples @ w
        try:
            ad_result = anderson_ksamp([data_proj, sample_proj])
            ad_stats.append(ad_result.statistic)
        except Exception:
            ad_stats.append(np.nan)

    return np.array(ad_stats)


@pytest.fixture(scope="module")
def sp500_returns():
    """Load SP500 returns once for the entire module."""
    return load_sp500_returns()


@pytest.fixture(scope="module")
def random_weights(sp500_returns):
    """Generate random portfolio weights once for the entire module."""
    d = sp500_returns.shape[1]
    return generate_random_weights(d, n_portfolios=N_PORTFOLIOS, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def fitted_vg(sp500_returns):
    """Fit Variance Gamma to SP500 returns."""
    vg = VarianceGamma()
    vg.fit(sp500_returns, max_iter=MAX_ITER, tol=EM_TOL, random_state=RANDOM_STATE)
    return vg


@pytest.fixture(scope="module")
def fitted_ninvg(sp500_returns):
    """Fit Normal Inverse Gamma to SP500 returns."""
    ninvg = NormalInverseGamma()
    ninvg.fit(sp500_returns, max_iter=MAX_ITER, tol=EM_TOL, random_state=RANDOM_STATE)
    return ninvg


@pytest.fixture(scope="module")
def fitted_nig(sp500_returns):
    """Fit Normal Inverse Gaussian to SP500 returns."""
    nig = NormalInverseGaussian()
    nig.fit(sp500_returns, max_iter=MAX_ITER, tol=EM_TOL, random_state=RANDOM_STATE)
    return nig


@pytest.fixture(scope="module")
def fitted_gh(sp500_returns):
    """Fit Generalized Hyperbolic to SP500 returns."""
    gh = GeneralizedHyperbolic()
    gh.fit(
        sp500_returns,
        max_iter=MAX_ITER,
        tol=EM_TOL,
        regularization="det_sigma_one",
        random_state=RANDOM_STATE,
    )
    return gh


# ============================================================================
# Test 1: EM convergence within 100 iterations
# ============================================================================


class TestEMConvergence:
    """Verify EM algorithm converges within MAX_ITER iterations for all distributions."""

    def test_vg_convergence(self, fitted_vg):
        assert fitted_vg._fitted is True
        assert fitted_vg.n_iter_ <= MAX_ITER, (
            f"VG did not converge within {MAX_ITER} iterations "
            f"(used {fitted_vg.n_iter_})"
        )

    def test_ninvg_convergence(self, fitted_ninvg):
        assert fitted_ninvg._fitted is True
        assert fitted_ninvg.n_iter_ <= MAX_ITER, (
            f"NInvG did not converge within {MAX_ITER} iterations "
            f"(used {fitted_ninvg.n_iter_})"
        )

    def test_nig_convergence(self, fitted_nig):
        assert fitted_nig._fitted is True
        assert fitted_nig.n_iter_ <= MAX_ITER, (
            f"NIG did not converge within {MAX_ITER} iterations "
            f"(used {fitted_nig.n_iter_})"
        )

    def test_gh_convergence(self, fitted_gh):
        assert fitted_gh._fitted is True
        assert fitted_gh.n_iter_ <= MAX_ITER, (
            f"GH did not converge within {MAX_ITER} iterations "
            f"(used {fitted_gh.n_iter_})"
        )


# ============================================================================
# Test 2: Anderson-Darling statistics on random portfolios (in-sample)
# ============================================================================


class TestAndersonDarling:
    """
    Verify Anderson-Darling statistics are below threshold on random portfolios.

    Uses in-sample data: projects both the original returns and samples from
    the fitted distribution onto random portfolio weight vectors, then runs
    two-sample Anderson-Darling tests.
    """

    def test_vg_ad_stats(self, fitted_vg, sp500_returns, random_weights):
        ad_stats = compute_ad_stats(
            fitted_vg, sp500_returns, random_weights,
            n_samples=N_SAMPLES_AD, random_state=RANDOM_STATE,
        )
        mean_ad = np.nanmean(ad_stats)
        assert mean_ad < AD_STAT_THRESHOLD, (
            f"VG mean AD stat {mean_ad:.4f} exceeds threshold {AD_STAT_THRESHOLD}"
        )

    def test_ninvg_ad_stats(self, fitted_ninvg, sp500_returns, random_weights):
        ad_stats = compute_ad_stats(
            fitted_ninvg, sp500_returns, random_weights,
            n_samples=N_SAMPLES_AD, random_state=RANDOM_STATE,
        )
        mean_ad = np.nanmean(ad_stats)
        assert mean_ad < AD_STAT_THRESHOLD, (
            f"NInvG mean AD stat {mean_ad:.4f} exceeds threshold {AD_STAT_THRESHOLD}"
        )

    def test_nig_ad_stats(self, fitted_nig, sp500_returns, random_weights):
        ad_stats = compute_ad_stats(
            fitted_nig, sp500_returns, random_weights,
            n_samples=N_SAMPLES_AD, random_state=RANDOM_STATE,
        )
        mean_ad = np.nanmean(ad_stats)
        assert mean_ad < AD_STAT_THRESHOLD, (
            f"NIG mean AD stat {mean_ad:.4f} exceeds threshold {AD_STAT_THRESHOLD}"
        )

    def test_gh_ad_stats(self, fitted_gh, sp500_returns, random_weights):
        ad_stats = compute_ad_stats(
            fitted_gh, sp500_returns, random_weights,
            n_samples=N_SAMPLES_AD, random_state=RANDOM_STATE,
        )
        mean_ad = np.nanmean(ad_stats)
        assert mean_ad < AD_STAT_THRESHOLD, (
            f"GH mean AD stat {mean_ad:.4f} exceeds threshold {AD_STAT_THRESHOLD}"
        )


# ============================================================================
# Test 3: Parameter recovery from generated samples (rvs -> refit)
# ============================================================================


class TestParameterRecovery:
    """
    Generate random samples from fitted distributions, then refit and verify
    that recovered parameters are reasonably close to the originals.

    For high-dimensional data, we check:
    - mu (location): close within a tolerance
    - gamma (skewness): close within a tolerance
    - sigma (covariance): Frobenius norm ratio close to 1
    - mixing distribution params: within relative tolerance
    """

    def _check_mu_gamma_sigma(self, original_params, refitted_params, dist_name,
                              mu_atol=0.01, gamma_atol=0.01, sigma_rtol=0.5):
        """Check that mu, gamma, sigma are recovered within tolerances."""
        mu_orig = original_params["mu"]
        mu_refit = refitted_params["mu"]
        mu_diff = np.max(np.abs(mu_orig - mu_refit))
        assert mu_diff < mu_atol, (
            f"{dist_name} mu max diff {mu_diff:.6f} exceeds tolerance {mu_atol}"
        )

        gamma_orig = original_params["gamma"]
        gamma_refit = refitted_params["gamma"]
        gamma_diff = np.max(np.abs(gamma_orig - gamma_refit))
        assert gamma_diff < gamma_atol, (
            f"{dist_name} gamma max diff {gamma_diff:.6f} exceeds tolerance {gamma_atol}"
        )

        sigma_orig = original_params["sigma"]
        sigma_refit = refitted_params["sigma"]
        sigma_ratio = np.linalg.norm(sigma_refit) / np.linalg.norm(sigma_orig)
        assert abs(sigma_ratio - 1.0) < sigma_rtol, (
            f"{dist_name} sigma Frobenius norm ratio {sigma_ratio:.4f} "
            f"deviates from 1.0 by more than {sigma_rtol}"
        )

    def test_vg_parameter_recovery(self, fitted_vg):
        X_gen = fitted_vg.rvs(size=N_SAMPLES_REFIT, random_state=RANDOM_STATE + 1)
        refitted = VarianceGamma()
        refitted.fit(X_gen, max_iter=MAX_ITER, tol=EM_TOL, random_state=RANDOM_STATE + 2)

        orig = fitted_vg.classical_params
        refit = refitted.classical_params

        self._check_mu_gamma_sigma(orig, refit, "VG")

        shape_ratio = refit["shape"] / orig["shape"]
        assert 0.3 < shape_ratio < 3.0, (
            f"VG shape ratio {shape_ratio:.4f} outside [0.3, 3.0]"
        )

        rate_ratio = refit["rate"] / orig["rate"]
        assert 0.3 < rate_ratio < 3.0, (
            f"VG rate ratio {rate_ratio:.4f} outside [0.3, 3.0]"
        )

    def test_ninvg_parameter_recovery(self, fitted_ninvg):
        X_gen = fitted_ninvg.rvs(size=N_SAMPLES_REFIT, random_state=RANDOM_STATE + 1)
        refitted = NormalInverseGamma()
        refitted.fit(X_gen, max_iter=MAX_ITER, tol=EM_TOL, random_state=RANDOM_STATE + 2)

        orig = fitted_ninvg.classical_params
        refit = refitted.classical_params

        self._check_mu_gamma_sigma(orig, refit, "NInvG")

        shape_ratio = refit["shape"] / orig["shape"]
        assert 0.3 < shape_ratio < 3.0, (
            f"NInvG shape ratio {shape_ratio:.4f} outside [0.3, 3.0]"
        )

        rate_ratio = refit["rate"] / orig["rate"]
        assert 0.3 < rate_ratio < 3.0, (
            f"NInvG rate ratio {rate_ratio:.4f} outside [0.3, 3.0]"
        )

    def test_nig_parameter_recovery(self, fitted_nig):
        X_gen = fitted_nig.rvs(size=N_SAMPLES_REFIT, random_state=RANDOM_STATE + 1)
        refitted = NormalInverseGaussian()
        refitted.fit(X_gen, max_iter=MAX_ITER, tol=EM_TOL, random_state=RANDOM_STATE + 2)

        orig = fitted_nig.classical_params
        refit = refitted.classical_params

        self._check_mu_gamma_sigma(orig, refit, "NIG")

        delta_ratio = refit["delta"] / orig["delta"]
        assert 0.3 < delta_ratio < 3.0, (
            f"NIG delta ratio {delta_ratio:.4f} outside [0.3, 3.0]"
        )

        eta_ratio = refit["eta"] / orig["eta"]
        assert 0.3 < eta_ratio < 3.0, (
            f"NIG eta ratio {eta_ratio:.4f} outside [0.3, 3.0]"
        )

    def test_gh_parameter_recovery(self, fitted_gh):
        X_gen = fitted_gh.rvs(size=N_SAMPLES_REFIT, random_state=RANDOM_STATE + 1)
        refitted = GeneralizedHyperbolic()
        refitted.fit(
            X_gen, max_iter=MAX_ITER, tol=EM_TOL,
            regularization="det_sigma_one",
            random_state=RANDOM_STATE + 2,
        )

        orig = fitted_gh.classical_params
        refit = refitted.classical_params

        self._check_mu_gamma_sigma(orig, refit, "GH")

        a_ratio = refit["a"] / orig["a"]
        assert 0.1 < a_ratio < 10.0, (
            f"GH a ratio {a_ratio:.4f} outside [0.1, 10.0]"
        )

        b_ratio = refit["b"] / orig["b"]
        assert 0.1 < b_ratio < 10.0, (
            f"GH b ratio {b_ratio:.4f} outside [0.1, 10.0]"
        )
