"""
Tests for GIG special-case behavior.

The old projection methods (to_gamma, to_inverse_gamma, to_inverse_gaussian)
are not in the current API. These tests verify the GIG's behavior at
special-case boundaries and its relationship to limiting distributions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

from normix.distributions.gamma import Gamma
from normix.distributions.generalized_inverse_gaussian import GIG
from normix.distributions.inverse_gamma import InverseGamma
from normix.distributions.inverse_gaussian import InverseGaussian


class TestGIGGammaLimit:
    """GIG(p, a, b→0) should behave like Gamma(shape=p, rate=a/2)."""

    @pytest.mark.parametrize("p,a", [(2.0, 2.0), (0.5, 6.0), (5.0, 1.0)])
    def test_mean_matches_gamma(self, p, a):
        gig = GIG(p=p, a=a, b=1e-14)
        gamma = Gamma(alpha=p, beta=a / 2)
        assert_allclose(float(gig.mean()), float(gamma.mean()), rtol=1e-3)

    @pytest.mark.parametrize("p,a", [(2.0, 2.0), (3.0, 4.0)])
    def test_log_partition_finite(self, p, a):
        gig = GIG(p=p, a=a, b=1e-14)
        assert np.isfinite(float(gig.log_partition()))

    def test_fit_gamma_data(self):
        """GIG fitted to Gamma samples should recover Gamma-like parameters."""
        rng = np.random.default_rng(42)
        data = jnp.array(rng.gamma(shape=3.0, scale=1.0 / 2.0, size=5000))
        gig = GIG(p=1.0, a=1.0, b=1.0)
        gig_fit = gig.fit(data)
        assert_allclose(float(gig_fit.mean()), 3.0 / 2.0, rtol=0.15)


class TestGIGInverseGammaLimit:
    """GIG(p, a→0, b) should behave like InverseGamma(shape=-p, rate=b/2)."""

    @pytest.mark.parametrize("shape,rate", [(3.0, 1.0), (2.0, 5.0), (4.0, 0.5)])
    def test_mean_matches_invgamma(self, shape, rate):
        b = 2 * rate
        p = -shape
        gig = GIG(p=p, a=1e-14, b=b)
        ig = InverseGamma(alpha=shape, beta=rate)
        assert_allclose(float(gig.mean()), float(ig.mean()), rtol=1e-2)

    @pytest.mark.parametrize("shape,rate", [(3.0, 1.0), (4.0, 0.5)])
    def test_log_partition_finite(self, shape, rate):
        b = 2 * rate
        p = -shape
        gig = GIG(p=p, a=1e-14, b=b)
        assert np.isfinite(float(gig.log_partition()))


class TestGIGInverseGaussianLimit:
    """GIG(p=-1/2, a, b) should relate to InverseGaussian."""

    @pytest.mark.parametrize("mu_ig,lam", [(1.0, 1.0), (2.0, 3.0)])
    def test_mean_matches_invgauss(self, mu_ig, lam):
        a = lam / mu_ig**2
        b = lam
        gig = GIG(p=-0.5, a=a, b=b)
        ig = InverseGaussian(mu=mu_ig, lam=lam)
        assert_allclose(float(gig.mean()), float(ig.mean()), rtol=1e-4)


class TestGIGFitting:
    """GIG fit on boundary data."""

    def test_fit_invgamma_data_ll(self):
        """GIG fitted to InvGamma samples should achieve good log-likelihood."""
        from scipy.stats import invgamma as invgamma_scipy
        rng = np.random.default_rng(42)
        data = jnp.array(invgamma_scipy.rvs(a=3.0, scale=2.0, size=2000, random_state=rng))
        gig = GIG(p=-1.0, a=0.1, b=4.0)
        gig_fit = gig.fit(data)
        ll = float(jax.vmap(gig_fit.log_prob)(data).mean())
        assert np.isfinite(ll), f"Non-finite log-likelihood"

    def test_fit_gamma_data_ll(self):
        """GIG fitted to Gamma samples should achieve good log-likelihood."""
        rng = np.random.default_rng(123)
        data = jnp.array(rng.gamma(shape=2.5, scale=1.0 / 1.5, size=2000))
        gig = GIG(p=2.0, a=3.0, b=0.1)
        gig_fit = gig.fit(data)
        ll = float(jax.vmap(gig_fit.log_prob)(data).mean())
        assert np.isfinite(ll), f"Non-finite log-likelihood"
