r"""
Tests for KL projection / exact embedding between distribution families.

Two operations:

* **Special -> General** (e.g. ``Gamma.to_gig``): exact embedding;
  the special case is a sub-manifold of the general family. Round-tripping
  through the general family must recover the original.
* **General -> Special** (e.g. ``GIG.to_gamma``): KL projection; the
  general distribution is approximated by the closest member of the
  special-case family in :math:`D_{\mathrm{KL}}(p\,\|\,q)` sense.

When the source distribution lies on the special-case manifold the
projection is also exact, so

::

    Source -> General -> Source

is a numerical identity for every pair (Gamma / InvGamma / InvGauss
through GIG; VG / NInvG / NIG through GH). Tested at both subordinator
and joint/marginal levels.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

from normix.distributions import (
    Gamma,
    InverseGamma,
    InverseGaussian,
    GIG,
    GeneralizedHyperbolic,
    JointGeneralizedHyperbolic,
    JointNormalInverseGamma,
    JointNormalInverseGaussian,
    JointVarianceGamma,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)


# ---------------------------------------------------------------------------
# Subordinator level: Gamma / InverseGamma / InverseGaussian <-> GIG
# ---------------------------------------------------------------------------


class TestSubordinatorRoundtrip:
    """``special.to_gig().to_special()`` must recover the original parameters."""

    @pytest.mark.parametrize("alpha,beta", [
        (2.5, 1.7), (0.7, 2.0), (5.0, 0.5), (1.2, 3.4),
    ])
    def test_gamma_roundtrip(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        g_back = g.to_gig().to_gamma()
        assert_allclose(float(g_back.alpha), alpha, rtol=1e-6, atol=1e-8)
        assert_allclose(float(g_back.beta), beta, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("alpha,beta", [
        (3.0, 2.0), (2.5, 0.8), (4.5, 1.5), (1.1, 2.0),
    ])
    def test_inverse_gamma_roundtrip(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        ig_back = ig.to_gig().to_inverse_gamma()
        assert_allclose(float(ig_back.alpha), alpha, rtol=1e-6, atol=1e-7)
        assert_allclose(float(ig_back.beta), beta, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("mu,lam", [
        (1.5, 2.0), (1.0, 1.0), (3.0, 5.0), (0.5, 0.7),
    ])
    def test_inverse_gaussian_roundtrip(self, mu, lam):
        igauss = InverseGaussian(mu=mu, lam=lam)
        igauss_back = igauss.to_gig().to_inverse_gaussian()
        assert_allclose(float(igauss_back.mu), mu, rtol=1e-12)
        assert_allclose(float(igauss_back.lam), lam, rtol=1e-12)


class TestSubordinatorLift:
    """Direct check of the algebraic embedding (special -> GIG)."""

    @pytest.mark.parametrize("alpha,beta", [(2.5, 1.0), (3.0, 2.0)])
    def test_gamma_to_gig_params(self, alpha, beta):
        gig = Gamma(alpha=alpha, beta=beta).to_gig()
        assert float(gig.p) == pytest.approx(alpha)
        assert float(gig.a) == pytest.approx(2.0 * beta)
        assert float(gig.b) == 0.0

    def test_gamma_to_gig_boundary_eps(self):
        gig = Gamma(alpha=2.0, beta=1.0).to_gig(boundary_eps=1e-10)
        assert float(gig.b) == pytest.approx(1e-10)

    @pytest.mark.parametrize("alpha,beta", [(3.0, 1.5), (2.5, 0.8)])
    def test_inverse_gamma_to_gig_params(self, alpha, beta):
        gig = InverseGamma(alpha=alpha, beta=beta).to_gig()
        assert float(gig.p) == pytest.approx(-alpha)
        assert float(gig.a) == 0.0
        assert float(gig.b) == pytest.approx(2.0 * beta)

    @pytest.mark.parametrize("mu,lam", [(1.5, 2.0), (3.0, 1.0)])
    def test_inverse_gaussian_to_gig_params(self, mu, lam):
        gig = InverseGaussian(mu=mu, lam=lam).to_gig()
        assert float(gig.p) == pytest.approx(-0.5)
        assert float(gig.a) == pytest.approx(lam / mu ** 2)
        assert float(gig.b) == pytest.approx(lam)


class TestGIGProjectionMomentMatch:
    """``GIG.to_<family>`` must satisfy the KL first-order conditions."""

    @pytest.mark.parametrize("p,a,b", [
        (1.0, 2.0, 1.0), (2.5, 0.5, 3.0), (-0.5, 1.0, 1.0), (-2.0, 4.0, 2.0),
    ])
    def test_gamma_projection_matches_E_logX_and_E_X(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        g = gig.to_gamma()
        eta_g = g.expectation_params()
        # Gamma matches E[log X] and E[X] to its precision; clamps the unused stat.
        assert_allclose(float(eta_g[0]), float(eta[0]), rtol=1e-6, atol=1e-7)
        assert_allclose(float(eta_g[1]), float(eta[2]), rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("p,a,b", [
        (-1.0, 2.0, 1.0), (-2.5, 0.5, 3.0), (0.5, 1.0, 1.0),
    ])
    def test_inverse_gamma_projection_matches_E_invX_and_E_logX(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        ig = gig.to_inverse_gamma()
        eta_ig = ig.expectation_params()
        # InverseGamma's eta = [-alpha/beta, log beta - psi(alpha)]
        # corresponds to [-E[1/X], E[log X]].
        assert_allclose(float(eta_ig[0]), -float(eta[1]), rtol=1e-6, atol=1e-7)
        assert_allclose(float(eta_ig[1]), float(eta[0]), rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("p,a,b", [
        (-0.5, 2.0, 3.0), (1.0, 1.0, 2.0), (-1.0, 0.5, 1.5),
    ])
    def test_inverse_gaussian_projection_matches_E_X_and_E_invX(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        igauss = gig.to_inverse_gaussian()
        eta_ig = igauss.expectation_params()
        # InverseGaussian's eta = [E[X], E[1/X]] = [eta[2], eta[1]] of GIG.
        assert_allclose(float(eta_ig[0]), float(eta[2]), rtol=1e-10)
        assert_allclose(float(eta_ig[1]), float(eta[1]), rtol=1e-10)


class TestGIGProjectionIdentityOnManifold:
    """A GIG that *is* a special case must project back to itself exactly."""

    @pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (3.5, 0.7)])
    def test_gamma_identity(self, alpha, beta):
        # Gamma -> GIG is exact; GIG -> Gamma should recover (alpha, beta).
        gig = Gamma(alpha=alpha, beta=beta).to_gig()
        g = gig.to_gamma()
        assert_allclose(float(g.alpha), alpha, rtol=1e-6, atol=1e-8)
        assert_allclose(float(g.beta), beta, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("alpha,beta", [(3.0, 2.0), (2.5, 1.0)])
    def test_inverse_gamma_identity(self, alpha, beta):
        gig = InverseGamma(alpha=alpha, beta=beta).to_gig()
        ig = gig.to_inverse_gamma()
        assert_allclose(float(ig.alpha), alpha, rtol=1e-6, atol=1e-7)
        assert_allclose(float(ig.beta), beta, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("mu,lam", [(1.5, 2.0), (3.0, 1.0)])
    def test_inverse_gaussian_identity(self, mu, lam):
        gig = InverseGaussian(mu=mu, lam=lam).to_gig()
        igauss = gig.to_inverse_gaussian()
        assert_allclose(float(igauss.mu), mu, rtol=1e-12)
        assert_allclose(float(igauss.lam), lam, rtol=1e-12)


# ---------------------------------------------------------------------------
# Joint level: JointGH <-> JointVG / JointNInvG / JointNIG
# ---------------------------------------------------------------------------


def _normal_block_2d():
    return (jnp.array([0.1, -0.2]),
            jnp.array([0.05, 0.1]),
            jnp.array([[1.0, 0.3], [0.3, 1.5]]))


class TestJointRoundtripPreservesNormalBlock:
    """Lift then project must preserve mu, gamma, Sigma exactly."""

    def test_vg_normal_block_unchanged(self):
        mu, gamma, sigma = _normal_block_2d()
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.5, beta=1.5)
        vg_back = vg.to_generalized_hyperbolic().to_variance_gamma()
        assert_allclose(np.asarray(vg_back.mu), np.asarray(mu), atol=0)
        assert_allclose(np.asarray(vg_back.gamma), np.asarray(gamma), atol=0)
        assert_allclose(np.asarray(vg_back.sigma()), np.asarray(sigma), atol=0)

    def test_nig_normal_block_unchanged(self):
        mu, gamma, sigma = _normal_block_2d()
        nig = NormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.5, lam=2.0)
        nig_back = nig.to_generalized_hyperbolic().to_normal_inverse_gaussian()
        assert_allclose(np.asarray(nig_back.mu), np.asarray(mu), atol=0)
        assert_allclose(np.asarray(nig_back.gamma), np.asarray(gamma), atol=0)
        assert_allclose(np.asarray(nig_back.sigma()), np.asarray(sigma), atol=0)

    def test_ninvg_normal_block_unchanged(self):
        mu, gamma, sigma = _normal_block_2d()
        ninvg = NormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=3.0, beta=2.0)
        ninvg_back = ninvg.to_generalized_hyperbolic().to_normal_inverse_gamma()
        assert_allclose(np.asarray(ninvg_back.mu), np.asarray(mu), atol=0)
        assert_allclose(np.asarray(ninvg_back.gamma), np.asarray(gamma), atol=0)
        assert_allclose(np.asarray(ninvg_back.sigma()), np.asarray(sigma), atol=0)


class TestJointRoundtripKLZero:
    """KL between original and round-tripped joint must be ~0."""

    KL_TOL = 1e-10  # accounts for digamma Newton tolerance in Gamma/InvGamma inverse

    def test_vg_roundtrip_kl_zero(self):
        mu, gamma, sigma = _normal_block_2d()
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.5, beta=1.5)
        vg_back = vg.to_generalized_hyperbolic().to_variance_gamma()
        kl = float(vg.kl_divergence(vg_back))
        assert abs(kl) < self.KL_TOL

    def test_nig_roundtrip_kl_zero(self):
        mu, gamma, sigma = _normal_block_2d()
        nig = NormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.5, lam=2.0)
        nig_back = nig.to_generalized_hyperbolic().to_normal_inverse_gaussian()
        kl = float(nig.kl_divergence(nig_back))
        # NIG embedding has no boundary; KL should be machine-epsilon small.
        assert abs(kl) < 1e-12

    def test_ninvg_roundtrip_kl_zero(self):
        mu, gamma, sigma = _normal_block_2d()
        ninvg = NormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=3.0, beta=2.0)
        ninvg_back = ninvg.to_generalized_hyperbolic().to_normal_inverse_gamma()
        kl = float(ninvg.kl_divergence(ninvg_back))
        assert abs(kl) < self.KL_TOL


class TestJointBoundaryEps:
    """boundary_eps lifts the lifted GIG away from the exact boundary."""

    def test_vg_boundary_eps_propagates(self):
        mu, gamma, sigma = _normal_block_2d()
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0)
        gh = vg.to_generalized_hyperbolic(boundary_eps=1e-12)
        assert float(gh._joint.b) == pytest.approx(1e-12)
        # Round-trip is still essentially exact.
        vg_back = gh.to_variance_gamma()
        assert float(vg.kl_divergence(vg_back)) < 1e-10


class TestGHProjectionIdentity:
    """A GH that lies on a special-case manifold projects back exactly."""

    def test_gh_built_from_vg_projects_back_to_vg(self):
        mu, gamma, sigma = _normal_block_2d()
        vg_orig = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.5, beta=1.5)
        gh = vg_orig.to_generalized_hyperbolic()
        vg_proj = gh.to_variance_gamma()
        assert_allclose(float(vg_proj.alpha), 2.5, rtol=1e-6, atol=1e-8)
        assert_allclose(float(vg_proj.beta), 1.5, rtol=1e-6, atol=1e-8)

    def test_gh_built_from_nig_projects_back_to_nig(self):
        mu, gamma, sigma = _normal_block_2d()
        nig_orig = NormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.5, lam=2.0)
        gh = nig_orig.to_generalized_hyperbolic()
        nig_proj = gh.to_normal_inverse_gaussian()
        assert_allclose(float(nig_proj.mu_ig), 1.5, rtol=1e-12)
        assert_allclose(float(nig_proj.lam), 2.0, rtol=1e-12)

    def test_gh_built_from_ninvg_projects_back_to_ninvg(self):
        mu, gamma, sigma = _normal_block_2d()
        ninvg_orig = NormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=3.0, beta=2.0)
        gh = ninvg_orig.to_generalized_hyperbolic()
        ninvg_proj = gh.to_normal_inverse_gamma()
        assert_allclose(float(ninvg_proj.alpha), 3.0, rtol=1e-6, atol=1e-7)
        assert_allclose(float(ninvg_proj.beta), 2.0, rtol=1e-6, atol=1e-7)


class TestGHProjectionMinimisesKL:
    """The projection must minimise KL inside the target family.

    For a GH whose subordinator is *not* on the manifold (a true GIG with
    p, a, b all in the interior), the projected VG must achieve a KL
    divergence smaller than every nearby VG.
    """

    def test_projected_vg_is_local_kl_minimum(self):
        mu, gamma, sigma = _normal_block_2d()
        gh = GeneralizedHyperbolic.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, p=1.5, a=2.0, b=1.0)

        vg_proj = gh.to_variance_gamma()

        # KL(p || q) for cross-family is computed via Monte Carlo since
        # GH and VG have different log-partition functions. Use sampling.
        n_samples = 20000
        X, _ = gh._joint.rvs(n=n_samples, seed=0)
        log_p = jax.vmap(gh.log_prob)(X)

        def avg_kl(vg):
            return float(jnp.mean(log_p - jax.vmap(vg.log_prob)(X)))

        kl_proj = avg_kl(vg_proj)
        # Perturb each parameter and check KL increases.
        for delta_alpha in [0.85, 1.15]:
            vg_pert = VarianceGamma.from_classical(
                mu=mu, gamma=gamma, sigma=sigma,
                alpha=float(vg_proj.alpha) * delta_alpha,
                beta=float(vg_proj.beta))
            assert avg_kl(vg_pert) > kl_proj - 1e-3, (
                f"projection isn't a local KL min: KL_proj={kl_proj}, "
                f"alpha*={delta_alpha}: KL={avg_kl(vg_pert)}")
        for delta_beta in [0.85, 1.15]:
            vg_pert = VarianceGamma.from_classical(
                mu=mu, gamma=gamma, sigma=sigma,
                alpha=float(vg_proj.alpha),
                beta=float(vg_proj.beta) * delta_beta)
            assert avg_kl(vg_pert) > kl_proj - 1e-3, (
                f"projection isn't a local KL min: KL_proj={kl_proj}, "
                f"beta*={delta_beta}: KL={avg_kl(vg_pert)}")


# ---------------------------------------------------------------------------
# Composition: cross-family conversions via GIG / GH
# ---------------------------------------------------------------------------


class TestComposition:
    """Cross-special-case projections via the general family must succeed."""

    def test_invgauss_to_invgamma_via_gig(self):
        # InverseGaussian -> GIG (exact) -> InverseGamma (KL projection).
        igauss = InverseGaussian(mu=1.5, lam=2.0)
        ig_proj = igauss.to_gig().to_inverse_gamma()
        assert float(ig_proj.alpha) > 0
        assert float(ig_proj.beta) > 0
        # First-order condition: matched stats under the source IG.
        eta_source = igauss.to_gig().expectation_params()
        eta_target = ig_proj.expectation_params()
        assert_allclose(float(eta_target[0]), -float(eta_source[1]), rtol=1e-6, atol=1e-7)
        assert_allclose(float(eta_target[1]), float(eta_source[0]), rtol=1e-6, atol=1e-7)

    def test_nig_to_vg_via_gh(self):
        # NIG -> GH (exact) -> VG (KL projection on Gamma subordinator).
        mu, gamma, sigma = _normal_block_2d()
        nig = NormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.5, lam=2.0)
        vg_proj = nig.to_generalized_hyperbolic().to_variance_gamma()
        assert float(vg_proj.alpha) > 0
        assert float(vg_proj.beta) > 0
        # Normal block must still pass through unchanged.
        assert_allclose(np.asarray(vg_proj.mu), np.asarray(mu), atol=0)
        assert_allclose(np.asarray(vg_proj.sigma()), np.asarray(sigma), atol=0)
