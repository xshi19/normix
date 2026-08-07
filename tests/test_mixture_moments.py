"""Skewness / excess kurtosis of normal variance-mean mixtures (review F1)."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from normix.distributions.gamma import Gamma
from normix.distributions.generalized_hyperbolic import (
    GeneralizedHyperbolic,
    UnivariateGeneralizedHyperbolic,
)
from normix.distributions.generalized_inverse_gaussian import GIG
from normix.distributions.inverse_gamma import InverseGamma
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.distributions.normal_inverse_gamma import (
    NormalInverseGamma,
    UnivariateNormalInverseGamma,
)
from normix.distributions.normal_inverse_gaussian import (
    NormalInverseGaussian,
    UnivariateNormalInverseGaussian,
)
from normix.distributions.variance_gamma import (
    UnivariateVarianceGamma,
    VarianceGamma,
)


# ---------------------------------------------------------------------------
# Subordinator raw moments
# ---------------------------------------------------------------------------

def test_gamma_raw_moments_match_closed_form():
    g = Gamma(alpha=2.5, beta=1.3)
    ks = jnp.array([1.0, 2.0, 3.0, 4.0])
    got = np.asarray(g.raw_moments(ks))
    alpha, beta = 2.5, 1.3
    want = np.array([
        alpha / beta,
        alpha * (alpha + 1) / beta ** 2,
        alpha * (alpha + 1) * (alpha + 2) / beta ** 3,
        alpha * (alpha + 1) * (alpha + 2) * (alpha + 3) / beta ** 4,
    ])
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_invgamma_raw_moments_match_closed_form():
    ig = InverseGamma(alpha=6.0, beta=2.0)
    ks = jnp.array([1.0, 2.0, 3.0, 4.0])
    got = np.asarray(ig.raw_moments(ks))
    alpha, beta = 6.0, 2.0
    want = np.array([
        beta / (alpha - 1),
        beta ** 2 / ((alpha - 1) * (alpha - 2)),
        beta ** 3 / ((alpha - 1) * (alpha - 2) * (alpha - 3)),
        beta ** 4 / ((alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4)),
    ])
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_gig_raw_moment_matches_mean_var():
    gig = GIG(p=0.7, a=1.4, b=0.9)
    m1, m2 = np.asarray(gig.raw_moments(jnp.array([1.0, 2.0])))
    np.testing.assert_allclose(m1, float(gig.mean()), rtol=1e-10)
    np.testing.assert_allclose(m2 - m1 ** 2, float(gig.var()), rtol=1e-10)


def test_ig_raw_moments_match_gig_embedding():
    ig = InverseGaussian(mu=1.2, lam=0.8)
    ks = jnp.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        np.asarray(ig.raw_moments(ks)),
        np.asarray(ig.to_gig().raw_moments(ks)),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Closed-form special cases
# ---------------------------------------------------------------------------

def test_symmetric_vg_excess_kurtosis_is_3_over_alpha():
    """γ = 0 ⇒ excess kurtosis = 3/α for VarianceGamma."""
    alpha = 2.0
    vg = UnivariateVarianceGamma.from_classical(
        mu=0.0, gamma=0.0, sigma=1.0, alpha=alpha, beta=1.0)
    np.testing.assert_allclose(float(vg.skewness()), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(vg.kurtosis()), 3.0 / alpha, rtol=1e-12)


def test_symmetric_ninvg_excess_kurtosis_is_3_over_alpha_minus_2():
    """γ = 0 ⇒ excess kurtosis = 3/(α−2) for NormalInverseGamma (α > 2)."""
    alpha = 5.0
    ninvg = UnivariateNormalInverseGamma.from_classical(
        mu=0.0, gamma=0.0, sigma=1.0, alpha=alpha, beta=2.0)
    np.testing.assert_allclose(float(ninvg.skewness()), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        float(ninvg.kurtosis()), 3.0 / (alpha - 2.0), rtol=1e-12)


# ---------------------------------------------------------------------------
# Multivariate consistency with univariate projection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: VarianceGamma.from_classical(
        mu=jnp.array([0.1, -0.2]),
        gamma=jnp.array([0.4, -0.3]),
        sigma=jnp.array([[1.2, 0.2], [0.2, 0.9]]),
        alpha=2.5, beta=1.0),
    lambda: NormalInverseGaussian.from_classical(
        mu=jnp.array([0.0, 0.1]),
        gamma=jnp.array([0.5, -0.2]),
        sigma=jnp.array([[1.0, 0.3], [0.3, 1.1]]),
        mu_ig=1.0, lam=1.5),
    lambda: NormalInverseGamma.from_classical(
        mu=jnp.array([0.0, 0.0]),
        gamma=jnp.array([0.2, 0.3]),
        sigma=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
        alpha=6.0, beta=2.0),
    lambda: GeneralizedHyperbolic.from_classical(
        mu=jnp.array([0.0, 0.2]),
        gamma=jnp.array([-0.3, 0.4]),
        sigma=jnp.array([[1.0, 0.25], [0.25, 1.2]]),
        p=-0.5, a=1.5, b=1.0),
])
def test_component_skew_kurt_match_projection(factory):
    model = factory()
    skew = np.asarray(model.skewness())
    kurt = np.asarray(model.kurtosis())
    for i in range(model.d):
        e_i = jnp.zeros(model.d).at[i].set(1.0)
        uni = model.project(e_i)
        np.testing.assert_allclose(
            skew[i], float(uni.skewness()), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            kurt[i], float(uni.kurtosis()), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Sample-moment agreement (contract)
# ---------------------------------------------------------------------------

@pytest.mark.contract
@pytest.mark.parametrize("name,model", [
    ("VG", UnivariateVarianceGamma.from_classical(
        mu=0.0, gamma=0.5, sigma=1.0, alpha=3.0, beta=1.5)),
    ("NIG", UnivariateNormalInverseGaussian.from_classical(
        mu=0.0, gamma=0.4, sigma=1.0, mu_ig=1.0, lam=2.0)),
    ("NInvG", UnivariateNormalInverseGamma.from_classical(
        mu=0.0, gamma=0.3, sigma=1.0, alpha=6.0, beta=2.0)),
    ("GH", UnivariateGeneralizedHyperbolic.from_classical(
        mu=0.0, gamma=0.35, sigma=1.0, p=0.5, a=1.2, b=1.0)),
])
def test_univariate_skew_kurt_vs_sample(name, model):
    del name
    x = np.asarray(model.rvs(80_000, seed=11), dtype=np.float64)
    np.testing.assert_allclose(
        float(model.skewness()), stats.skew(x), rtol=0.08, atol=0.05)
    np.testing.assert_allclose(
        float(model.kurtosis()), stats.kurtosis(x, fisher=True),
        rtol=0.12, atol=0.08,
    )
