"""
Tests for GIG → special-case projection via expectation parameters.

Each test creates a GIG at (or near) a special-case boundary, projects to
the special case, and checks that the projected distribution's parameters
and moments match.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from normix.distributions.univariate import (
    Gamma,
    GeneralizedInverseGaussian as GIG,
    InverseGamma,
    InverseGaussian,
)


class TestGIGToGamma:
    """GIG(p, a, b→0) should project to Gamma(shape=p, rate=a/2)."""

    @pytest.mark.parametrize("shape, rate", [
        (2.0, 1.0),
        (0.5, 3.0),
        (5.0, 0.5),
        (1.0, 2.0),
    ])
    def test_roundtrip_from_gamma(self, shape, rate):
        """Gamma → GIG → to_gamma() should recover the original Gamma."""
        g_orig = Gamma.from_classical_params(shape=shape, rate=rate)

        a = 2 * rate
        p = shape
        gig = GIG.from_classical_params(p=p, a=a, b=1e-14)

        g_proj = gig.to_gamma()
        g_params = g_proj.classical_params
        assert_allclose(g_params.shape, shape, rtol=1e-4)
        assert_allclose(g_params.rate, rate, rtol=1e-4)

    @pytest.mark.parametrize("shape, rate", [
        (2.0, 1.0),
        (3.0, 2.0),
    ])
    def test_moments_match(self, shape, rate):
        """Projected Gamma should share E[log X] and E[X] with the GIG."""
        a = 2 * rate
        p = shape
        gig = GIG.from_classical_params(p=p, a=a, b=1e-14)

        g_proj = gig.to_gamma()
        gig_eta = gig.expectation_params

        g_eta = g_proj.expectation_params
        assert_allclose(g_eta[0], gig_eta[0], rtol=1e-4,
                        err_msg="E[log X] mismatch")
        assert_allclose(g_eta[1], gig_eta[2], rtol=1e-4,
                        err_msg="E[X] mismatch")


class TestGIGToInverseGamma:
    """GIG(p, a→0, b) should project to InverseGamma(shape=-p, rate=b/2)."""

    @pytest.mark.parametrize("shape, rate", [
        (3.0, 1.0),
        (2.0, 5.0),
        (4.0, 0.5),
        (1.5, 2.0),
    ])
    def test_roundtrip_from_inverse_gamma(self, shape, rate):
        """InverseGamma → GIG → to_inverse_gamma() should recover the original."""
        ig_orig = InverseGamma.from_classical_params(shape=shape, rate=rate)

        b = 2 * rate
        p = -shape
        gig = GIG.from_classical_params(p=p, a=1e-14, b=b)

        ig_proj = gig.to_inverse_gamma()
        ig_params = ig_proj.classical_params
        assert_allclose(ig_params.shape, shape, rtol=1e-4)
        assert_allclose(ig_params.rate, rate, rtol=1e-4)

    @pytest.mark.parametrize("shape, rate", [
        (3.0, 1.0),
        (2.0, 3.0),
    ])
    def test_moments_match(self, shape, rate):
        """Projected InverseGamma should share E[1/X] and E[log X] with the GIG."""
        b = 2 * rate
        p = -shape
        gig = GIG.from_classical_params(p=p, a=1e-14, b=b)

        ig_proj = gig.to_inverse_gamma()
        gig_eta = gig.expectation_params

        ig_eta = ig_proj.expectation_params
        assert_allclose(-ig_eta[0], gig_eta[1], rtol=1e-4,
                        err_msg="E[1/X] mismatch")
        assert_allclose(ig_eta[1], gig_eta[0], rtol=1e-4,
                        err_msg="E[log X] mismatch")


class TestGIGToInverseGaussian:
    """GIG(p=-1/2, a, b) should project to InverseGaussian."""

    @pytest.mark.parametrize("delta, eta", [
        (1.0, 1.0),
        (2.0, 3.0),
        (0.5, 5.0),
        (3.0, 0.5),
    ])
    def test_roundtrip_from_inverse_gaussian(self, delta, eta):
        """InverseGaussian → GIG → to_inverse_gaussian() should recover original."""
        ig_orig = InverseGaussian.from_classical_params(delta=delta, eta=eta)

        a = eta / delta**2
        b = eta
        gig = GIG.from_classical_params(p=-0.5, a=a, b=b)

        ig_proj = gig.to_inverse_gaussian()
        ig_params = ig_proj.classical_params
        assert_allclose(ig_params.delta, delta, rtol=1e-4)
        assert_allclose(ig_params.eta, eta, rtol=1e-4)

    @pytest.mark.parametrize("delta, eta", [
        (1.0, 1.0),
        (2.0, 3.0),
    ])
    def test_moments_match(self, delta, eta):
        """Projected IG should share E[X] and E[1/X] with the GIG."""
        a = eta / delta**2
        b = eta
        gig = GIG.from_classical_params(p=-0.5, a=a, b=b)

        ig_proj = gig.to_inverse_gaussian()
        gig_eta = gig.expectation_params

        ig_eta = ig_proj.expectation_params
        assert_allclose(ig_eta[0], gig_eta[2], rtol=1e-4,
                        err_msg="E[X] mismatch")
        assert_allclose(ig_eta[1], gig_eta[1], rtol=1e-4,
                        err_msg="E[1/X] mismatch")


class TestGeneralGIGProjection:
    """Projection from a general GIG (not at boundary) to special cases."""

    def test_general_gig_to_gamma_preserves_moments(self):
        """For a general GIG, projected Gamma preserves E[log X] and E[X]."""
        gig = GIG.from_classical_params(p=1.5, a=2.0, b=1.0)
        g = gig.to_gamma()

        gig_eta = gig.expectation_params
        g_eta = g.expectation_params
        assert_allclose(g_eta[0], gig_eta[0], rtol=1e-6,
                        err_msg="E[log X] mismatch")
        assert_allclose(g_eta[1], gig_eta[2], rtol=1e-6,
                        err_msg="E[X] mismatch")

    def test_general_gig_to_inverse_gamma_preserves_moments(self):
        """For a general GIG, projected InverseGamma preserves E[1/X] and E[log X]."""
        gig = GIG.from_classical_params(p=-1.5, a=1.0, b=2.0)
        ig = gig.to_inverse_gamma()

        gig_eta = gig.expectation_params
        ig_eta = ig.expectation_params
        assert_allclose(-ig_eta[0], gig_eta[1], rtol=1e-6,
                        err_msg="E[1/X] mismatch")
        assert_allclose(ig_eta[1], gig_eta[0], rtol=1e-6,
                        err_msg="E[log X] mismatch")

    def test_general_gig_to_inverse_gaussian_preserves_moments(self):
        """For a general GIG, projected IG preserves E[X] and E[1/X]."""
        gig = GIG.from_classical_params(p=1.0, a=2.0, b=1.0)
        ig = gig.to_inverse_gaussian()

        gig_eta = gig.expectation_params
        ig_eta = ig.expectation_params
        assert_allclose(ig_eta[0], gig_eta[2], rtol=1e-6,
                        err_msg="E[X] mismatch")
        assert_allclose(ig_eta[1], gig_eta[1], rtol=1e-6,
                        err_msg="E[1/X] mismatch")

    def test_not_fitted_raises(self):
        """Projection on unfitted GIG raises ValueError."""
        gig = GIG()
        with pytest.raises(ValueError):
            gig.to_gamma()
        with pytest.raises(ValueError):
            gig.to_inverse_gamma()
        with pytest.raises(ValueError):
            gig.to_inverse_gaussian()


class TestFittedGIGProjection:
    """Fit a GIG to samples drawn from a special case, then project back."""

    def test_fit_gamma_samples_project_back(self):
        """GIG fitted to Gamma samples should project back close to the original."""
        rng = np.random.default_rng(42)
        shape, rate = 3.0, 2.0
        data = rng.gamma(shape=shape, scale=1.0 / rate, size=5000)

        gig = GIG().fit(data)
        g = gig.to_gamma()
        g_params = g.classical_params
        assert_allclose(g_params.shape, shape, rtol=0.15)
        assert_allclose(g_params.rate, rate, rtol=0.15)

    def test_fit_invgauss_samples_project_back(self):
        """GIG fitted to InverseGaussian samples should project back close."""
        rng = np.random.default_rng(123)
        delta, eta_param = 2.0, 3.0
        data = rng.wald(mean=delta, scale=eta_param, size=5000)

        gig = GIG().fit(data)
        ig = gig.to_inverse_gaussian()
        ig_params = ig.classical_params
        assert_allclose(ig_params.delta, delta, rtol=0.15)
        assert_allclose(ig_params.eta, eta_param, rtol=0.15)
