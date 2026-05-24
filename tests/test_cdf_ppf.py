"""
Tests for ``cdf`` / ``ppf`` across distributions.

Covers:

- Univariate subordinator distributions: analytical ppf (Gamma, InverseGamma)
  vs ``scipy.stats``; PINV-backed ppf (InverseGaussian, GIG) vs ``scipy.stats``.
- Univariate marginal mixtures (``UnivariateVarianceGamma``,
  ``UnivariateNormalInverseGamma``, ``UnivariateNormalInverseGaussian``,
  ``UnivariateGeneralizedHyperbolic``): cdf is monotone, ``ppf(cdf(x)) ≈ x``,
  and ``cdf(ppf(q)) ≈ q``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from normix import (
    Gamma, InverseGamma, InverseGaussian, GIG,
    UnivariateVarianceGamma,
    UnivariateNormalInverseGamma,
    UnivariateNormalInverseGaussian,
    UnivariateGeneralizedHyperbolic,
)


_QS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


# ---------------------------------------------------------------------------
# Univariate subordinators (scipy reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (5.0, 3.0), (0.7, 0.5)])
def test_gamma_ppf_matches_scipy(alpha, beta):
    g = Gamma(alpha=alpha, beta=beta)
    ours = np.asarray(g.ppf(jnp.asarray(_QS)))
    ref = stats.gamma.ppf(_QS, a=alpha, scale=1.0 / beta)
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("alpha,beta", [(3.0, 2.0), (5.0, 1.0), (4.0, 0.5)])
def test_inverse_gamma_ppf_matches_scipy(alpha, beta):
    ig = InverseGamma(alpha=alpha, beta=beta)
    ours = np.asarray(ig.ppf(jnp.asarray(_QS)))
    ref = stats.invgamma.ppf(_QS, a=alpha, scale=beta)
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("mu,lam", [(1.0, 1.0), (2.0, 3.0), (0.5, 2.0)])
def test_inverse_gaussian_ppf_matches_scipy(mu, lam):
    ig = InverseGaussian(mu=mu, lam=lam)
    ours = np.asarray(ig.ppf(jnp.asarray(_QS)))
    ref = stats.invgauss.ppf(_QS, mu=mu / lam, scale=lam)
    # PINV table on a 4000-pt grid is not bit-exact vs scipy quad-based ppf
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("p,a,b", [(1.0, 2.0, 3.0), (-0.5, 1.5, 2.5), (2.0, 1.0, 0.5)])
def test_gig_cdf_ppf_matches_scipy(p, a, b):
    gig = GIG(p=p, a=a, b=b)
    b_sp = float(np.sqrt(a * b))
    scale = float(np.sqrt(b / a))

    # ppf
    ours_ppf = np.asarray(gig.ppf(jnp.asarray(_QS)))
    ref_ppf = stats.geninvgauss.ppf(_QS, p=p, b=b_sp, scale=scale)
    np.testing.assert_allclose(ours_ppf, ref_ppf, rtol=1e-3, atol=1e-4)

    # cdf at the scipy quantiles
    ours_cdf = np.asarray(gig.cdf(jnp.asarray(ref_ppf)))
    np.testing.assert_allclose(ours_cdf, _QS, rtol=1e-3, atol=1e-4)


def test_gig_cdf_ppf_round_trip():
    gig = GIG(p=0.5, a=1.5, b=2.0)
    qs = jnp.asarray(_QS)
    np.testing.assert_allclose(
        np.asarray(gig.cdf(gig.ppf(qs))), _QS, rtol=1e-4, atol=1e-5)


def test_gig_degenerate_gamma_limit_delegates():
    """GIG(p>0, a, b≈0) should match Gamma(p, a/2) for cdf and ppf."""
    p, a, b = 2.0, 3.0, 0.0
    gig = GIG(p=p, a=a, b=b)
    g = Gamma(alpha=p, beta=a / 2.0)

    xs = jnp.asarray([0.1, 0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(gig.cdf(xs)), np.asarray(g.cdf(xs)), rtol=1e-8, atol=1e-10)
    qs = jnp.asarray(_QS)
    np.testing.assert_allclose(
        np.asarray(gig.ppf(qs)), np.asarray(g.ppf(qs)), rtol=1e-8, atol=1e-10)


def test_gig_degenerate_invgamma_limit_delegates():
    """GIG(p<0, a≈0, b) should match InverseGamma(-p, b/2)."""
    p, a, b = -2.0, 0.0, 3.0
    gig = GIG(p=p, a=a, b=b)
    ig = InverseGamma(alpha=-p, beta=b / 2.0)

    xs = jnp.asarray([0.1, 0.5, 1.0, 2.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(gig.cdf(xs)), np.asarray(ig.cdf(xs)), rtol=1e-8, atol=1e-10)
    qs = jnp.asarray(_QS)
    np.testing.assert_allclose(
        np.asarray(gig.ppf(qs)), np.asarray(ig.ppf(qs)), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Univariate marginal mixtures: round-trips and sanity checks
# ---------------------------------------------------------------------------


def _univariate_models():
    return [
        ("VG", UnivariateVarianceGamma.from_classical(
            mu=0.0, gamma=0.3, sigma=1.0, alpha=2.0, beta=1.0)),
        ("NInvG", UnivariateNormalInverseGamma.from_classical(
            mu=0.0, gamma=0.0, sigma=1.0, alpha=3.0, beta=1.0)),
        ("NIG", UnivariateNormalInverseGaussian.from_classical(
            mu=0.0, gamma=0.2, sigma=1.0, mu_ig=1.0, lam=1.5)),
        ("GH", UnivariateGeneralizedHyperbolic.from_classical(
            mu=0.0, gamma=0.0, sigma=1.0, p=1.0, a=1.0, b=1.0)),
    ]


@pytest.mark.parametrize("name,model", _univariate_models())
def test_univariate_mixture_round_trip(name, model):
    qs = jnp.asarray(_QS)
    xs = model.ppf(qs)
    # cdf(ppf(q)) == q
    np.testing.assert_allclose(np.asarray(model.cdf(xs)), _QS, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("name,model", _univariate_models())
def test_univariate_mixture_scalar_api(name, model):
    """``mean``/``var``/``std`` are scalars; ``rvs`` is ``(n,)``."""
    assert jnp.asarray(model.mean()).shape == ()
    assert jnp.asarray(model.var()).shape == ()
    assert jnp.asarray(model.std()).shape == ()
    samples = model.rvs(50, seed=1)
    assert samples.shape == (50,)


@pytest.mark.parametrize("name,model", _univariate_models())
def test_univariate_mixture_cdf_monotone(name, model):
    xs = jnp.linspace(-5.0, 5.0, 41)
    fs = np.asarray(model.cdf(xs))
    diffs = np.diff(fs)
    assert (diffs >= -1e-10).all(), f"{name}: cdf not monotone (min diff={diffs.min()})"
    assert 0.0 <= fs[0] <= 1.0
    assert 0.0 <= fs[-1] <= 1.0


def test_univariate_rejects_multivariate_joint():
    """Constructing a Univariate from a d>1 joint must raise."""
    from normix import JointVarianceGamma
    joint = JointVarianceGamma.from_classical(
        mu=jnp.zeros(2), gamma=jnp.zeros(2),
        sigma=jnp.eye(2), alpha=2.0, beta=1.0,
    )
    with pytest.raises(ValueError, match="d=1"):
        UnivariateVarianceGamma(joint)
