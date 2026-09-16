"""S1: GIG Devroye TDR in (p, z, s) coordinates.

Stable log-mode, e^{-1} envelope, while_loop that never emits a reject.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from normix.distributions.gamma import Gamma
from normix.distributions.generalized_inverse_gaussian import (
    GIG,
    _gig_log_mode,
    _gig_tdr_propose,
    _gig_tdr_setup,
)
from normix.distributions.inverse_gamma import InverseGamma

pytestmark = pytest.mark.contract

_ACCEPT_GRID = [
    (p, z, z)
    for p in (0.0, 2.0, -2.0)
    for z in (1e-12, 1e-8, 1e-4, 1.0, 1e2)
]


class TestGIGReviewSampling:
    """Verification-script pass bar."""

    @pytest.mark.parametrize("p,a,b", [
        (-2.0, 1e-8, 1e-8),
        (0.0, 1e-8, 1e-8),
    ])
    def test_rvs_finite_positive(self, p, a, b):
        draws = np.asarray(GIG(p, a, b).rvs(1000, seed=0), dtype=np.float64)
        assert draws.shape == (1000,)
        assert np.all(np.isfinite(draws))
        assert np.all(draws > 0.0)

    def test_no_impossible_tail(self):
        draws = np.asarray(
            GIG(0.0, 1e-4, 1e-4).rvs(10000, seed=42), dtype=np.float64,
        )
        assert np.all(np.isfinite(draws) & (draws > 0.0))
        assert float(np.mean(np.abs(np.log(draws)) > 20.0)) == 0.0


class TestGIGLogMode:
    def test_mode_no_cancellation(self):
        gig = GIG(0.0, 1e-8, 1e-8)
        q = -1.0
        r = math.sqrt(q * q + 1e-16)
        exact = 1e-8 / (abs(q) + r)
        assert_allclose(float(gig.mode()), exact, rtol=1e-12)
        assert float(gig.mode()) > 0.0

    def test_neg_p_mode(self):
        gig = GIG(-2.0, 1e-8, 1e-8)
        q = -3.0
        r = math.sqrt(q * q + 1e-16)
        exact = 1e-8 / (abs(q) + r)
        assert_allclose(float(gig.mode()), exact, rtol=1e-12)

    def test_log_mode_matches_asinh(self):
        p, a, b = -2.0, 1e-8, 1e-8
        z = math.sqrt(a * b)
        s = 0.5 * (math.log(b) - math.log(a))
        w0 = s + math.asinh(p / z)
        assert_allclose(float(_gig_log_mode(p, a, b)), w0, rtol=1e-12)

    def test_pinv_finite(self):
        draws = np.asarray(
            GIG(-2.0, 1e-8, 1e-8).rvs(200, seed=0, method="pinv"),
            dtype=np.float64,
        )
        assert np.all(np.isfinite(draws) & (draws > 0.0))

    def test_cdf_at_mode_interior(self):
        gig = GIG(0.0, 1e-8, 1e-8)
        c = float(gig.cdf(gig.mode()))
        assert 0.0 < c < 1.0

    def test_mode_gamma_boundary(self):
        gig = GIG(2.0, 3.0, 0.0)
        gamma = Gamma(alpha=2.0, beta=1.5)
        assert_allclose(float(gig.mode()), float(gamma.mode()), rtol=1e-12)
        assert_allclose(float(gig.mode()), 2.0 / 3.0, rtol=1e-12)

    def test_mode_invgamma_boundary(self):
        gig = GIG(-2.0, 0.0, 3.0)
        ig = InverseGamma(alpha=2.0, beta=1.5)
        assert_allclose(float(gig.mode()), float(ig.mode()), rtol=1e-12)
        assert_allclose(float(gig.mode()), 0.5, rtol=1e-12)

    def test_rvs_gamma_boundary_finite(self):
        draws = np.asarray(GIG(2.0, 3.0, 0.0).rvs(500, seed=0), dtype=np.float64)
        assert np.all(np.isfinite(draws) & (draws > 0.0))
        # Gamma(2, 1.5) mean = 4/3
        assert_allclose(float(np.mean(draws)), 4.0 / 3.0, rtol=0.15)

    def test_rvs_invgamma_boundary_finite(self):
        draws = np.asarray(GIG(-2.0, 0.0, 3.0).rvs(500, seed=0), dtype=np.float64)
        assert np.all(np.isfinite(draws) & (draws > 0.0))

    def test_exhausted_tdr_rounds_are_nan(self, monkeypatch):
        import normix.distributions.generalized_inverse_gaussian as gig_mod
        monkeypatch.setattr(gig_mod, "_GIG_TDR_MAX_ROUNDS", 0)
        draws = np.asarray(
            gig_mod._gig_rvs_tdr(
                jax.random.PRNGKey(0),
                jnp.float64(0.0), jnp.float64(1.0), jnp.float64(1.0),
                8,
            ),
            dtype=np.float64,
        )
        assert np.all(np.isnan(draws))


class TestGIGTDRAcceptance:
    @pytest.mark.parametrize("p,a,b", _ACCEPT_GRID)
    def test_acceptance_at_least_one_over_e(self, p, a, b):
        env = _gig_tdr_setup(
            jnp.float64(p), jnp.float64(a), jnp.float64(b),
        )
        _, accept = _gig_tdr_propose(jax.random.PRNGKey(0), env, 4000)
        rate = float(np.mean(np.asarray(accept)))
        assert rate >= math.exp(-1.0) - 0.03, (
            f"acceptance {rate:.3f} < 1/e at GIG({p},{a},{b})"
        )


class TestGIGTDRVsScipy:
    @pytest.mark.parametrize("p,a,b", _ACCEPT_GRID)
    def test_ks_against_scipy(self, p, a, b):
        n = 2000
        x = np.asarray(GIG(p, a, b).rvs(n, seed=0), dtype=np.float64)
        b_sp = math.sqrt(a * b)
        scale = math.sqrt(b / a)
        y = stats.geninvgauss.rvs(
            p=p, b=b_sp, scale=scale, size=n, random_state=1,
        )
        ks = stats.ks_2samp(np.log(x), np.log(y))
        assert ks.statistic < 0.08, (
            f"KS={ks.statistic:.3f} at GIG({p},{a},{b})"
        )
