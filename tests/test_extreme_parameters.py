"""
T4: Extreme-parameter tests for all distributions.

Verifies that distributions handle extreme parameter regimes without
NaN or overflow: large/small shape parameters, near-boundary cases,
concentrated distributions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from normix.distributions.gamma import Gamma
from normix.distributions.inverse_gamma import InverseGamma
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.distributions.generalized_inverse_gaussian import GIG
from normix.distributions import (
    VarianceGamma,
    NormalInverseGamma,
    NormalInverseGaussian,
    GeneralizedHyperbolic,
)


# ============================================================
# Gamma extreme
# ============================================================

class TestGammaExtreme:

    @pytest.mark.parametrize("alpha,beta", [
        (0.1, 1.0), (100.0, 1.0), (1.0, 100.0), (50.0, 50.0),
    ])
    def test_log_prob_finite(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        x = jnp.array(alpha / beta)  # at the mode
        assert np.isfinite(float(g.log_prob(x)))

    @pytest.mark.parametrize("alpha,beta", [
        (0.1, 1.0), (100.0, 1.0), (1.0, 100.0),
    ])
    def test_log_partition_finite(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        assert np.isfinite(float(g.log_partition()))

    @pytest.mark.parametrize("alpha,beta", [
        (100.0, 1.0), (1.0, 100.0),
    ])
    def test_expectation_params_finite(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        eta = g.expectation_params()
        assert jnp.all(jnp.isfinite(eta))


# ============================================================
# InverseGamma extreme
# ============================================================

class TestInverseGammaExtreme:

    @pytest.mark.parametrize("alpha,beta", [
        (2.1, 1.0), (100.0, 1.0), (3.0, 100.0), (50.0, 50.0),
    ])
    def test_log_prob_finite(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        x = jnp.array(beta / (alpha + 1))  # at the mode
        assert np.isfinite(float(ig.log_prob(x)))

    @pytest.mark.parametrize("alpha,beta", [
        (2.1, 1.0), (100.0, 1.0),
    ])
    def test_mean_var_finite(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        assert np.isfinite(float(ig.mean()))
        if alpha > 2:
            assert np.isfinite(float(ig.var()))


# ============================================================
# InverseGaussian extreme
# ============================================================

class TestInverseGaussianExtreme:

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1000.0), (1.0, 0.01), (100.0, 1.0), (0.01, 1.0),
    ])
    def test_log_prob_finite(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        x = jnp.array(mu)
        assert np.isfinite(float(ig.log_prob(x)))

    @pytest.mark.parametrize("mu,lam,x", [
        (1.0, 1000.0, 1.0),
        (0.5, 1000.0, 0.5),
        (2.0, 500.0, 2.0),
    ])
    def test_cdf_extreme_shape(self, mu, lam, x):
        """CDF must be finite and match scipy in high-shape regime."""
        ig = InverseGaussian(mu=mu, lam=lam)
        our = float(ig.cdf(jnp.array(x)))
        ref = float(stats.invgauss.cdf(x, mu=mu / lam, scale=lam))
        assert np.isfinite(our), f"CDF non-finite for mu={mu}, lam={lam}, x={x}"
        np.testing.assert_allclose(our, ref, rtol=1e-5)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1000.0), (0.01, 1.0), (100.0, 1.0),
    ])
    def test_rvs_no_nan(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        samples = ig.rvs(100, seed=42)
        assert jnp.all(jnp.isfinite(samples))
        assert jnp.all(samples > 0)


# ============================================================
# GIG extreme
# ============================================================

class TestGIGExtreme:

    @pytest.mark.parametrize("p,a,b", [
        (0.5, 100.0, 0.01),   # near Gamma limit
        (-2.0, 0.01, 100.0),  # near InverseGamma limit
        (10.0, 1.0, 1.0),     # large p
        (-10.0, 1.0, 1.0),    # very negative p
        (0.5, 0.01, 0.01),    # both small
        (1.0, 50.0, 50.0),    # both large
    ])
    def test_log_prob_finite(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        x = jnp.array(float(gig.mean()))
        assert np.isfinite(float(gig.log_prob(x)))

    @pytest.mark.parametrize("p,a,b", [
        (0.5, 100.0, 0.01),
        (-2.0, 0.01, 100.0),
        (10.0, 1.0, 1.0),
        (-10.0, 1.0, 1.0),
    ])
    def test_log_partition_finite(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        assert np.isfinite(float(gig.log_partition()))

    @pytest.mark.parametrize("p,a,b", [
        (1.0, 1.0, 1.0),
        (2.0, 0.5, 0.5),
        (-0.5, 2.0, 1.0),
    ])
    def test_rvs_positive_finite(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        samples = gig.rvs(100, seed=42)
        assert jnp.all(jnp.isfinite(samples))
        assert jnp.all(samples > 0)


# ============================================================
# Mixture distribution extreme parameters
# ============================================================

class TestMixtureExtreme:

    def test_vg_large_alpha(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=50.0, beta=25.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_vg_small_alpha(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.1]),
            sigma=jnp.array([[1.0]]), alpha=0.5, beta=0.5,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_ninvg_large_alpha(self):
        nig = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=50.0, beta=25.0,
        )
        X = nig.rvs(20, seed=42)
        lps = jax.vmap(nig.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_nig_large_lam(self):
        nig = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), mu_ig=1.0, lam=50.0,
        )
        X = nig.rvs(20, seed=42)
        lps = jax.vmap(nig.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_nig_small_lam(self):
        nig = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.1]),
            sigma=jnp.array([[1.0]]), mu_ig=1.0, lam=0.1,
        )
        X = nig.rvs(20, seed=42)
        lps = jax.vmap(nig.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_gh_large_p(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), p=20.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_gh_negative_p(self):
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), p=-5.0, a=1.0, b=1.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_gh_near_vg_limit(self):
        """GH with b ≈ 0 should still produce finite log_prob."""
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), p=2.0, a=2.0, b=0.01,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_gh_near_ninvg_limit(self):
        """GH with a ≈ 0 should still produce finite log_prob."""
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), p=-3.0, a=0.01, b=2.0,
        )
        X = gh.rvs(20, seed=42)
        lps = jax.vmap(gh.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))
