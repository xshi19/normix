"""
Tests for information-theoretic quantities: entropy, varentropy, renyi.

Fast tests validate against scipy closed forms, the Fisher quadratic form
:math:`\\theta^\\top I\\theta`, the analytical Gaussian baseline
:math:`V_H(\\mathcal N_d)=d/2`, delegation consistency for the embedded
special cases (InverseGaussian, NIG), and the Rényi <-> varentropy
differential identity. Slow tests cross-check against Monte Carlo.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

jax.config.update("jax_enable_x64", True)

from normix import (
    Gamma, InverseGamma, InverseGaussian, GIG, MultivariateNormal,
    GeneralizedHyperbolic, VarianceGamma, NormalInverseGaussian,
    NormalInverseGamma,
)


# ---------------------------------------------------------------------------
# Entropy vs scipy closed forms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist, scipy_dist", [
    (Gamma(alpha=2.0, beta=3.0), stats.gamma(a=2.0, scale=1.0 / 3.0)),
    (Gamma(alpha=5.0, beta=0.5), stats.gamma(a=5.0, scale=2.0)),
    (InverseGamma(alpha=3.0, beta=2.0), stats.invgamma(a=3.0, scale=2.0)),
    (InverseGaussian(mu=1.5, lam=2.0), stats.invgauss(mu=1.5 / 2.0, scale=2.0)),
])
def test_entropy_vs_scipy(dist, scipy_dist):
    np.testing.assert_allclose(
        float(dist.entropy()), scipy_dist.entropy(), rtol=1e-6, atol=1e-8)


def test_mvn_entropy_and_varentropy_closed_form():
    mu = jnp.array([0.5, -1.0, 2.0])
    Sigma = jnp.array([[2.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.5]])
    mvn = MultivariateNormal.from_classical(mu, Sigma)
    d = 3
    _, logdet = np.linalg.slogdet(np.asarray(Sigma))
    H = 0.5 * d * (1.0 + np.log(2.0 * np.pi)) + 0.5 * logdet
    np.testing.assert_allclose(float(mvn.entropy()), H, rtol=1e-10)
    # Gaussian varentropy is exactly d/2, independent of mu, Sigma.
    np.testing.assert_allclose(float(mvn.varentropy()), d / 2.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Varentropy == Fisher quadratic form θ·I·θ (constant base measure)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist, rtol", [
    (Gamma(alpha=2.0, beta=3.0), 1e-5),
    (InverseGamma(alpha=3.0, beta=2.0), 1e-5),
    (GIG(p=-0.5, a=2.0, b=3.0), 1e-4),
    (GIG(p=2.5, a=0.5, b=4.0), 1e-4),
])
def test_varentropy_equals_fisher_quadratic_form(dist, rtol):
    theta = np.asarray(dist.natural_params())
    # CPU Fisher (analytic for Gamma/InvGamma, accurate FD for GIG)
    I = np.asarray(dist.fisher_information(backend="cpu"))
    np.testing.assert_allclose(
        float(dist.varentropy()), float(theta @ I @ theta), rtol=rtol)


# ---------------------------------------------------------------------------
# Rényi entropy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist", [
    Gamma(alpha=2.0, beta=3.0),
    GIG(p=-0.5, a=2.0, b=3.0),
    InverseGaussian(mu=1.5, lam=2.0),
])
def test_renyi_at_one_is_shannon_entropy(dist):
    np.testing.assert_allclose(
        float(dist.renyi(1.0)), float(dist.entropy()), rtol=1e-10)


@pytest.mark.parametrize("dist", [
    Gamma(alpha=2.0, beta=3.0),
    GIG(p=-0.5, a=2.0, b=3.0),
])
def test_varentropy_is_second_derivative_of_R(dist):
    """H = -R'(1), V_H = R''(1) where R(α) = log ∫ p^α."""
    R = dist.log_density_power
    Hp = float(-jax.grad(R)(jnp.asarray(1.0)))
    Vpp = float(jax.grad(jax.grad(R))(jnp.asarray(1.0)))
    np.testing.assert_allclose(Hp, float(dist.entropy()), rtol=1e-8)
    np.testing.assert_allclose(Vpp, float(dist.varentropy()), rtol=1e-8)


@pytest.mark.parametrize("dist", [
    Gamma(alpha=2.0, beta=3.0),
    GIG(p=-0.5, a=2.0, b=3.0),
])
def test_varentropy_is_minus_twice_renyi_slope(dist):
    """H_α = H - (V_H/2)(α-1) + O((α-1)²)  ⇒  V_H = -2 H_α'(1)."""
    eps = 1e-3
    slope = (float(dist.renyi(1.0 + eps)) - float(dist.renyi(1.0 - eps))) / (2.0 * eps)
    np.testing.assert_allclose(-2.0 * slope, float(dist.varentropy()), rtol=1e-3)


# ---------------------------------------------------------------------------
# Delegation consistency for embedded special cases
# ---------------------------------------------------------------------------
def test_inverse_gaussian_delegates_to_gig():
    ig = InverseGaussian(mu=1.5, lam=2.0)
    gig = ig.to_gig()
    for a in (0.5, 2.0):
        np.testing.assert_allclose(
            float(ig.renyi(a)), float(gig.renyi(a)), rtol=1e-10)
    np.testing.assert_allclose(float(ig.entropy()), float(gig.entropy()), rtol=1e-10)
    np.testing.assert_allclose(
        float(ig.varentropy()), float(gig.varentropy()), rtol=1e-10)


def test_nig_joint_delegates_to_gh_embedding():
    nig = NormalInverseGaussian.from_classical(
        mu=jnp.array([0.0, 0.0]), gamma=jnp.array([0.3, 0.2]),
        sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]), mu_ig=1.0, lam=2.0)
    gh = nig._joint.to_joint_generalized_hyperbolic()
    np.testing.assert_allclose(
        float(nig.joint_varentropy()), float(gh.varentropy()), rtol=1e-10)
    np.testing.assert_allclose(
        float(nig.joint_entropy()), float(gh.entropy()), rtol=1e-10)


# ---------------------------------------------------------------------------
# Joint varentropy: analytic operator formula (independent of mu, gamma, Sigma)
# ---------------------------------------------------------------------------
def _joint_gh_varentropy_operator(p, a, b, d, h=1e-4):
    """d/2 + (L_d² - L_d) log K_p(t), t = √(ab), via scipy FD Bessel."""
    from scipy.special import kve
    def lk(pp, tt):
        return np.log(kve(pp, tt)) - tt
    t = np.sqrt(a * b)
    fpp = (lk(p + h, t) - 2 * lk(p, t) + lk(p - h, t)) / h**2
    ftt = (lk(p, t + h) - 2 * lk(p, t) + lk(p, t - h)) / h**2
    fpt = (lk(p + h, t + h) - lk(p + h, t - h)
           - lk(p - h, t + h) + lk(p - h, t - h)) / (4 * h**2)
    c = p - 1.0 - d / 2.0
    return d / 2.0 + c * c * fpp + 2 * c * t * fpt + t * t * ftt


def test_joint_gh_varentropy_matches_operator_formula():
    gh = GeneralizedHyperbolic.from_classical(
        mu=jnp.array([0.0, 1.0]), gamma=jnp.array([0.5, -0.3]),
        sigma=jnp.array([[1.5, 0.4], [0.4, 0.8]]), p=-0.5, a=2.0, b=3.0)
    expected = _joint_gh_varentropy_operator(-0.5, 2.0, 3.0, d=2)
    np.testing.assert_allclose(float(gh.joint_varentropy()), expected, rtol=1e-4)


def test_joint_varentropy_independent_of_normal_block():
    """V_H(X,Y) depends on (mu, gamma, Sigma) only through the dimension d."""
    base = dict(p=-0.5, a=2.0, b=3.0)
    v_ref = None
    for mu, gamma, sig in [
        ([0.0, 0.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([5.0, -3.0], [2.0, 1.0], [[4.0, 1.5], [1.5, 2.0]]),
    ]:
        gh = GeneralizedHyperbolic.from_classical(
            mu=jnp.array(mu), gamma=jnp.array(gamma), sigma=jnp.array(sig), **base)
        v = float(gh.joint_varentropy())
        if v_ref is None:
            v_ref = v
        else:
            np.testing.assert_allclose(v, v_ref, rtol=1e-8)


# ---------------------------------------------------------------------------
# JIT / vmap compatibility
# ---------------------------------------------------------------------------
def test_varentropy_entropy_jittable():
    g = Gamma(alpha=2.0, beta=3.0)
    v_jit = float(jax.jit(lambda d: d.varentropy())(g))
    h_jit = float(jax.jit(lambda d: d.entropy())(g))
    np.testing.assert_allclose(v_jit, float(g.varentropy()), rtol=1e-10)
    np.testing.assert_allclose(h_jit, float(g.entropy()), rtol=1e-10)


# ---------------------------------------------------------------------------
# Monte Carlo cross-checks (slow)
# ---------------------------------------------------------------------------
def _mc_entropy_varentropy(dist, n, seed=0):
    Y = np.asarray(dist.rvs(n, seed=seed))
    s = -np.asarray(jax.vmap(dist.log_prob)(jnp.asarray(Y)))
    return s.mean(), s.var()


def _mc_joint_entropy_varentropy(joint, n, seed=0):
    X, Y = joint.rvs(n, seed=seed)
    s = -np.asarray(jax.vmap(joint.log_prob_joint)(X, Y))
    return s.mean(), s.var()


@pytest.mark.slow
@pytest.mark.parametrize("dist", [
    Gamma(alpha=2.0, beta=3.0),
    InverseGamma(alpha=3.0, beta=2.0),
    GIG(p=-0.5, a=2.0, b=3.0),
    GIG(p=2.5, a=0.5, b=4.0),
    InverseGaussian(mu=1.5, lam=2.0),
])
def test_entropy_varentropy_vs_monte_carlo(dist):
    Hm, Vm = _mc_entropy_varentropy(dist, n=4_000_000, seed=0)
    np.testing.assert_allclose(float(dist.entropy()), Hm, rtol=0.02)
    np.testing.assert_allclose(float(dist.varentropy()), Vm, rtol=0.03)


@pytest.mark.slow
def test_joint_varentropy_vs_monte_carlo():
    configs = [
        GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 1.0]), gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.5, 0.4], [0.4, 0.8]]), p=-0.5, a=2.0, b=3.0),
        VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]), gamma=jnp.array([0.3, 0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]), alpha=1.5, beta=1.0),
        NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.0]), gamma=jnp.array([0.3, 0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]), alpha=3.0, beta=1.0),
        NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0, 0.0]), gamma=jnp.array([0.3, 0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]), mu_ig=1.0, lam=2.0),
    ]
    for m in configs:
        Hm, Vm = _mc_joint_entropy_varentropy(m._joint, n=3_000_000, seed=0)
        np.testing.assert_allclose(float(m.joint_entropy()), Hm, rtol=0.02)
        np.testing.assert_allclose(float(m.joint_varentropy()), Vm, rtol=0.03)
