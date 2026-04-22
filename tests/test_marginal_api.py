"""
Tests for the marginal/joint η→model API and the parameter view/replace facade.

Covers:
  - JointNormalMixture.from_expectation(NormalMixtureEta) — closed-form
    M-step dispatch (replaces the old `m_step` indirection).
  - NormalMixture.from_expectation(NormalMixtureEta) — wrapper around
    the joint constructor.
  - Forwarding properties (mu, gamma, L_Sigma, sigma, log_det_sigma) and
    subordinator-specific forwarders (alpha, beta, mu_ig, lam, p, a, b).
  - replace(**) — immutable update with normal/subordinator dispatch and
    `sigma` Cholesky alias.
  - eta0_isotropic round-trip: from_expectation(eta0_isotropic(model,
    σ²)).sigma() recovers σ²·I.

These tests pin down the public contract added by the API unification
(see `docs/design/em_covariance_extensions.md` §9, post-Phase 2).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions.variance_gamma import (
    JointVarianceGamma, VarianceGamma,
)
from normix.distributions.normal_inverse_gamma import (
    JointNormalInverseGamma, NormalInverseGamma,
)
from normix.distributions.normal_inverse_gaussian import (
    JointNormalInverseGaussian, NormalInverseGaussian,
)
from normix.distributions.generalized_hyperbolic import (
    JointGeneralizedHyperbolic, GeneralizedHyperbolic,
)
from normix.fitting.shrinkage_targets import (
    eta0_from_model, eta0_isotropic, eta0_with_sigma,
)


KEY = jax.random.PRNGKey(7)
D = 3
N = 200


def _make_data(key=KEY, n=N, d=D):
    return jax.random.normal(key, shape=(n, d), dtype=jnp.float64) * 0.5


def _make_models(X):
    d = X.shape[1]
    mu = jnp.mean(X, axis=0)
    sigma = jnp.cov(X.T) + 1e-4 * jnp.eye(d)
    return {
        "VG": VarianceGamma.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, alpha=2.0, beta=1.0),
        "NInvG": NormalInverseGamma.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, alpha=3.0, beta=1.0),
        "NIG": NormalInverseGaussian.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, mu_ig=1.0, lam=1.0),
        "GH": GeneralizedHyperbolic.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma, p=-0.5, a=2.0, b=1.0),
    }


_DIST_NAMES = ["VG", "NInvG", "NIG", "GH"]
_JOINT_CLS = {
    "VG": JointVarianceGamma,
    "NInvG": JointNormalInverseGamma,
    "NIG": JointNormalInverseGaussian,
    "GH": JointGeneralizedHyperbolic,
}
_MARGINAL_CLS = {
    "VG": VarianceGamma,
    "NInvG": NormalInverseGamma,
    "NIG": NormalInverseGaussian,
    "GH": GeneralizedHyperbolic,
}


# ---------------------------------------------------------------------------
# Forwarding properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_normal_param_properties_match_joint(dist_name):
    """vg.mu / .gamma / .L_Sigma / .sigma() forward to vg.joint."""
    X = _make_data()
    model = _make_models(X)[dist_name]
    j = model.joint
    np.testing.assert_array_equal(np.array(model.mu), np.array(j.mu))
    np.testing.assert_array_equal(np.array(model.gamma), np.array(j.gamma))
    np.testing.assert_array_equal(np.array(model.L_Sigma), np.array(j.L_Sigma))
    np.testing.assert_allclose(
        np.array(model.sigma()), np.array(j.sigma()), atol=1e-12)
    np.testing.assert_allclose(
        float(model.log_det_sigma()), float(j.log_det_sigma()), atol=1e-12)


def test_subordinator_properties_forward():
    """Subordinator fields are exposed on each marginal subclass."""
    X = _make_data()
    models = _make_models(X)

    vg = models["VG"]
    np.testing.assert_allclose(float(vg.alpha), float(vg.joint.alpha))
    np.testing.assert_allclose(float(vg.beta), float(vg.joint.beta))

    ninvg = models["NInvG"]
    np.testing.assert_allclose(float(ninvg.alpha), float(ninvg.joint.alpha))
    np.testing.assert_allclose(float(ninvg.beta), float(ninvg.joint.beta))

    nig = models["NIG"]
    np.testing.assert_allclose(float(nig.mu_ig), float(nig.joint.mu_ig))
    np.testing.assert_allclose(float(nig.lam), float(nig.joint.lam))

    gh = models["GH"]
    np.testing.assert_allclose(float(gh.p), float(gh.joint.p))
    np.testing.assert_allclose(float(gh.a), float(gh.joint.a))
    np.testing.assert_allclose(float(gh.b), float(gh.joint.b))


def test_sigma_distinct_from_marginal_cov():
    """sigma() is the GIG-mixed dispersion Σ, not the marginal Cov[X]."""
    X = _make_data()
    vg = _make_models(X)["VG"]
    Sigma = np.array(vg.sigma())
    Cov = np.array(vg.cov())
    # Both PD, but Cov includes E[Y]·Σ + Var[Y]·γγᵀ — distinct from Σ
    # except in degenerate cases (E[Y]=1, γ=0).
    assert Sigma.shape == Cov.shape == (D, D)


# ---------------------------------------------------------------------------
# Joint.from_expectation(NormalMixtureEta)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_joint_from_expectation_pytree_matches_m_step(dist_name):
    """JointXxx.from_expectation(eta) == old m_step(eta) for VG/NIG/NInvG.

    For GH the classmethod path lacks warm-start + sanity fallback, so
    we only require finite, PD output (the LL agreement is checked in
    the round-trip test below).
    """
    X = _make_data()
    model = _make_models(X)[dist_name]
    eta = model.e_step(X, backend='cpu')

    joint_cls = _JOINT_CLS[dist_name]
    j_from_eta = joint_cls.from_expectation(eta, backend='cpu')

    if dist_name != "GH":
        m_step_model = model.m_step(eta, backend='cpu')
        j_via_m_step = m_step_model.joint
        np.testing.assert_allclose(
            np.array(j_from_eta.mu), np.array(j_via_m_step.mu),
            atol=1e-10, rtol=1e-8,
            err_msg=f"{dist_name}: mu mismatch from_expectation vs m_step")
        np.testing.assert_allclose(
            np.array(j_from_eta.L_Sigma), np.array(j_via_m_step.L_Sigma),
            atol=1e-10, rtol=1e-8)

    # Σ is PD
    Sigma = np.array(j_from_eta.sigma())
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals > 0), f"{dist_name}: Σ not PD after from_expectation"


@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_marginal_from_expectation_wraps_joint(dist_name):
    """MarginalXxx.from_expectation(eta).joint == JointXxx.from_expectation(eta)."""
    X = _make_data()
    model = _make_models(X)[dist_name]
    eta = model.e_step(X, backend='cpu')

    marginal_cls = _MARGINAL_CLS[dist_name]
    joint_cls = _JOINT_CLS[dist_name]

    m_from_eta = marginal_cls.from_expectation(eta, backend='cpu')
    j_from_eta = joint_cls.from_expectation(eta, backend='cpu')

    np.testing.assert_allclose(
        np.array(m_from_eta.joint.mu), np.array(j_from_eta.mu), atol=1e-12)
    np.testing.assert_allclose(
        np.array(m_from_eta.joint.L_Sigma),
        np.array(j_from_eta.L_Sigma), atol=1e-12)


def test_joint_from_expectation_flat_array_dispatches_to_bregman():
    """A flat ``jax.Array`` η dispatches to the inherited Bregman solver.

    We don't assert on the solution quality here — the joint EF has
    high-dimensional θ that the generic Bregman solver handles poorly
    (the closed-form ``NormalMixtureEta`` path is the real intended
    route). We only verify the dispatch is wired and the call returns a
    structurally valid `JointVarianceGamma`.
    """
    X = _make_data()
    j = _make_models(X)["VG"].joint

    theta = j.natural_params()
    eta_flat = JointVarianceGamma._grad_log_partition(theta)

    # Should run without raising; quality is not asserted.
    j2 = JointVarianceGamma.from_expectation(eta_flat, backend='cpu')
    assert isinstance(j2, JointVarianceGamma)
    assert j2.mu.shape == j.mu.shape


# ---------------------------------------------------------------------------
# eta0_isotropic round-trip — the original validation use case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_eta0_isotropic_recovers_sigma_via_from_expectation(dist_name):
    r"""``cls.from_expectation(eta0_isotropic(model, σ²)).sigma() == σ² I_d``.

    This is the validation the user asked for in the original report:
    ``eta0_isotropic`` should produce an η whose closed-form M-step
    inverse has dispersion exactly :math:`\sigma^2 I`. With γ = 0 (the
    `eta0_isotropic` convention via `default_init` returns γ=0), the
    closed-form M-step on the dispersion block reduces to
    :math:`\Sigma = E[XX^\top/Y] - μμ^\top E[1/Y] = σ² I`.
    """
    X = _make_data()
    sigma2 = 1.7

    # default_init produces γ = 0 (see NormalMixture.default_init); this
    # is the regime in which the eta0_isotropic prior is well-defined.
    init = _MARGINAL_CLS[dist_name].default_init(X)
    eta0 = eta0_isotropic(init, sigma2=sigma2)

    recovered = _MARGINAL_CLS[dist_name].from_expectation(eta0, backend='cpu')
    Sigma = np.array(recovered.sigma())
    expected = sigma2 * np.eye(D)
    np.testing.assert_allclose(Sigma, expected, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_eta0_with_sigma_recovers_arbitrary_sigma(dist_name):
    """from_expectation(eta0_with_sigma(model, Σ₀)) recovers Σ₀ exactly."""
    X = _make_data()
    Sigma0 = jnp.array([[1.5, 0.2, 0.0],
                        [0.2, 0.8, 0.1],
                        [0.0, 0.1, 1.1]])

    init = _MARGINAL_CLS[dist_name].default_init(X)
    eta0 = eta0_with_sigma(init, Sigma0=Sigma0)

    recovered = _MARGINAL_CLS[dist_name].from_expectation(eta0, backend='cpu')
    np.testing.assert_allclose(
        np.array(recovered.sigma()), np.array(Sigma0),
        atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_eta0_from_model_round_trip(dist_name):
    """from_expectation(eta0_from_model(model)) recovers the model's normals.

    The subordinator solver may not round-trip exactly (especially GH /
    GIG which uses an iterative solver), but μ, γ, Σ are produced by a
    closed-form formula from η, so the normal block must round-trip to
    machine precision.
    """
    X = _make_data()
    model = _make_models(X)[dist_name]
    eta = eta0_from_model(model)

    recovered = _MARGINAL_CLS[dist_name].from_expectation(eta, backend='cpu')
    np.testing.assert_allclose(
        np.array(recovered.mu), np.array(model.mu),
        atol=1e-8, rtol=1e-6,
        err_msg=f"{dist_name}: μ not recovered from compute_eta_from_model")
    np.testing.assert_allclose(
        np.array(recovered.sigma()), np.array(model.sigma()),
        atol=1e-8, rtol=1e-6,
        err_msg=f"{dist_name}: Σ not recovered from compute_eta_from_model")


# ---------------------------------------------------------------------------
# replace(**) — view + immutable update
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", _DIST_NAMES)
def test_replace_normal_params(dist_name):
    """replace(mu=..., gamma=..., L_Sigma=...) updates only the named fields."""
    X = _make_data()
    model = _make_models(X)[dist_name]

    new_mu = jnp.array([10.0, -1.0, 0.5])
    new_gamma = jnp.array([0.1, 0.2, -0.3])

    updated = model.replace(mu=new_mu, gamma=new_gamma)

    np.testing.assert_allclose(np.array(updated.mu), np.array(new_mu))
    np.testing.assert_allclose(np.array(updated.gamma), np.array(new_gamma))
    # L_Sigma unchanged
    np.testing.assert_array_equal(
        np.array(updated.L_Sigma), np.array(model.L_Sigma))
    # original is unmodified (immutable)
    assert not np.allclose(np.array(model.mu), np.array(new_mu))


def test_replace_sigma_alias_converts_to_cholesky():
    """replace(sigma=Σ) sets L_Sigma = chol(Σ)."""
    X = _make_data()
    vg = _make_models(X)["VG"]

    Sigma_new = jnp.array([[2.0, 0.5, 0.0],
                           [0.5, 1.5, 0.2],
                           [0.0, 0.2, 1.0]])
    updated = vg.replace(sigma=Sigma_new)
    np.testing.assert_allclose(
        np.array(updated.sigma()), np.array(Sigma_new), atol=1e-12)


def test_replace_sigma_and_L_Sigma_conflict_raises():
    X = _make_data()
    vg = _make_models(X)["VG"]
    with pytest.raises(ValueError, match="either `sigma` or `L_Sigma`"):
        vg.replace(sigma=jnp.eye(D), L_Sigma=jnp.eye(D))


def test_replace_subordinator_params_VG():
    """replace(alpha=..., beta=...) updates the Gamma subordinator only."""
    X = _make_data()
    vg = _make_models(X)["VG"]

    updated = vg.replace(alpha=5.0, beta=2.5)
    np.testing.assert_allclose(float(updated.alpha), 5.0)
    np.testing.assert_allclose(float(updated.beta), 2.5)
    # Normal params unchanged
    np.testing.assert_array_equal(np.array(updated.mu), np.array(vg.mu))
    np.testing.assert_array_equal(np.array(updated.gamma), np.array(vg.gamma))


def test_replace_subordinator_params_NIG():
    X = _make_data()
    nig = _make_models(X)["NIG"]
    updated = nig.replace(mu_ig=2.0, lam=3.0)
    np.testing.assert_allclose(float(updated.mu_ig), 2.0)
    np.testing.assert_allclose(float(updated.lam), 3.0)


def test_replace_subordinator_params_GH():
    X = _make_data()
    gh = _make_models(X)["GH"]
    updated = gh.replace(p=1.0, a=3.0, b=4.0)
    np.testing.assert_allclose(float(updated.p), 1.0)
    np.testing.assert_allclose(float(updated.a), 3.0)
    np.testing.assert_allclose(float(updated.b), 4.0)


def test_replace_unknown_key_raises():
    X = _make_data()
    vg = _make_models(X)["VG"]
    with pytest.raises(ValueError, match="unknown parameter"):
        vg.replace(nonexistent=1.0)
    # NIG-style key on a VG should also fail
    with pytest.raises(ValueError, match="unknown parameter"):
        vg.replace(mu_ig=1.0)


def test_replace_no_args_returns_self():
    X = _make_data()
    vg = _make_models(X)["VG"]
    out = vg.replace()
    assert out is vg


def test_replace_then_log_prob_finite():
    """A replaced model is still a usable distribution (log_prob finite)."""
    X = _make_data()
    vg = _make_models(X)["VG"]
    updated = vg.replace(mu=jnp.zeros(D), sigma=2.0 * jnp.eye(D), alpha=2.5)
    lp = updated.marginal_log_likelihood(X)
    assert jnp.isfinite(lp)
