"""
Tests for IncrementalEMFitter and eta update infrastructure.

Covers:
  - compute_eta_from_model round-trip consistency
  - affine_combine arithmetic
  - IncrementalEMFitter with each eta update rule
  - BatchEMFitter with ShrinkageUpdate
  - LL improves (or at least stays finite) after incremental EM
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions.variance_gamma import VarianceGamma
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.fitting.em import BatchEMFitter, IncrementalEMFitter
from normix.fitting.eta import NormalMixtureEta, affine_combine
from normix.fitting.eta_rules import (
    IdentityUpdate,
    RobbinsMonroUpdate,
    SampleWeightedUpdate,
    EWMAUpdate,
    ShrinkageUpdate,
    AffineUpdate,
)

KEY = jax.random.PRNGKey(42)
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


# ---------------------------------------------------------------------------
# compute_eta_from_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_compute_eta_from_model_fields_finite(dist_name):
    """compute_eta_from_model returns finite values for all fields."""
    X = _make_data()
    model = _make_models(X)[dist_name]
    eta = model.compute_eta_from_model()
    for field in ("E_log_Y", "E_inv_Y", "E_Y", "E_X", "E_X_inv_Y", "E_XXT_inv_Y"):
        val = getattr(eta, field)
        assert jnp.all(jnp.isfinite(val)), (
            f"{dist_name}.compute_eta_from_model().{field} not finite: {val}")


@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_compute_eta_round_trip(dist_name):
    """M-step on model's own eta should approximately recover the model."""
    X = _make_data()
    model = _make_models(X)[dist_name]
    eta = model.compute_eta_from_model()
    recovered = model.m_step_normal(eta)

    np.testing.assert_allclose(
        np.array(recovered._joint.mu), np.array(model._joint.mu),
        atol=1e-6, rtol=1e-4,
        err_msg=f"{dist_name} mu not recovered from compute_eta_from_model",
    )
    np.testing.assert_allclose(
        np.array(recovered._joint.gamma), np.array(model._joint.gamma),
        atol=1e-6, rtol=1e-4,
        err_msg=f"{dist_name} gamma not recovered from compute_eta_from_model",
    )


# ---------------------------------------------------------------------------
# affine_combine
# ---------------------------------------------------------------------------

def test_affine_combine_identity():
    """affine_combine with b=0, c=1 returns eta_new."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X)
    result = affine_combine(eta1, eta2, b=0.0, c=1.0)
    np.testing.assert_allclose(
        np.array(result.E_X), np.array(eta2.E_X), atol=1e-12)


def test_affine_combine_midpoint():
    """affine_combine with b=0.5, c=0.5 gives midpoint."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X)
    result = affine_combine(eta1, eta2, b=0.5, c=0.5)
    expected = 0.5 * np.array(eta1.E_X) + 0.5 * np.array(eta2.E_X)
    np.testing.assert_allclose(np.array(result.E_X), expected, atol=1e-12)


def test_affine_combine_with_shift():
    """affine_combine with additive shift a."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X)
    shift = jax.tree.map(lambda x: 0.1 * jnp.ones_like(x), eta1)
    result = affine_combine(eta1, eta2, b=0.0, c=1.0, a=shift)
    expected = np.array(eta2.E_Y) + 0.1
    np.testing.assert_allclose(float(result.E_Y), float(expected), atol=1e-12)


# ---------------------------------------------------------------------------
# Eta update rules — weight contracts
# ---------------------------------------------------------------------------

def test_identity_update_weights():
    rule = IdentityUpdate()
    a, b, c, state = rule.weights(0, 100, {})
    assert a is None
    assert float(b) == 0.0
    assert float(c) == 1.0


def test_robbins_monro_weights():
    rule = RobbinsMonroUpdate(tau0=10.0)
    a, b, c, state = rule.weights(0, 100, {})
    assert a is None
    assert abs(float(c) - 1.0 / 10.0) < 1e-12
    assert abs(float(b) - (1.0 - 1.0 / 10.0)) < 1e-12


def test_sample_weighted_cumulative():
    rule = SampleWeightedUpdate()
    state = rule.initial_state()
    _, b1, c1, state = rule.weights(0, 100, state)
    assert float(state['cumulative_n']) == 100.0
    assert abs(float(b1)) < 1e-12
    assert abs(float(c1) - 1.0) < 1e-12

    _, b2, c2, state = rule.weights(1, 50, state)
    assert float(state['cumulative_n']) == 150.0
    assert abs(float(b2) - 100.0 / 150.0) < 1e-12
    assert abs(float(c2) - 50.0 / 150.0) < 1e-12


def test_ewma_weights():
    rule = EWMAUpdate(w=0.3)
    a, b, c, _ = rule.weights(5, 100, {})
    assert a is None
    assert abs(float(b) - 0.7) < 1e-12
    assert abs(float(c) - 0.3) < 1e-12


def test_shrinkage_weights():
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = model.compute_eta_from_model()
    rule = ShrinkageUpdate(eta0, tau=0.5)
    a, b, c, _ = rule.weights(0, 100, {})
    assert float(b) == 0.0
    assert abs(float(c) - 1.0 / 1.5) < 1e-12
    assert a is not None
    expected_factor = 0.5 / 1.5
    np.testing.assert_allclose(
        float(a.E_Y), float(eta0.E_Y) * expected_factor, atol=1e-12)


def test_rules_are_pytrees():
    """All rules are eqx.Module pytrees with JAX array leaves."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = model.compute_eta_from_model()

    rules = [
        IdentityUpdate(),
        RobbinsMonroUpdate(tau0=5.0),
        SampleWeightedUpdate(),
        EWMAUpdate(w=0.2),
        ShrinkageUpdate(eta0, tau=0.5),
        AffineUpdate(b=0.9, c=0.1),
    ]
    for rule in rules:
        leaves = jax.tree.leaves(rule)
        name = type(rule).__name__
        for leaf in leaves:
            assert hasattr(leaf, 'shape'), (
                f"{name} leaf {leaf!r} is not a JAX array")


# ---------------------------------------------------------------------------
# IncrementalEMFitter — smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_incremental_em_robbins_monro(dist_name):
    """IncrementalEMFitter with RobbinsMonroUpdate produces a finite model."""
    X = _make_data()
    model = _make_models(X)[dist_name]

    fitter = IncrementalEMFitter(
        eta_update=RobbinsMonroUpdate(tau0=10.0),
        batch_size=50, max_steps=20,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"{dist_name} incremental EM LL not finite: {ll}"


@pytest.mark.parametrize("rule_name,rule", [
    ("identity", IdentityUpdate()),
    ("sample_weighted", SampleWeightedUpdate()),
    ("ewma", EWMAUpdate(w=0.2)),
])
def test_incremental_em_various_rules(rule_name, rule):
    """IncrementalEMFitter works with each non-shrinkage rule."""
    X = _make_data()
    model = _make_models(X)["VG"]

    fitter = IncrementalEMFitter(
        eta_update=rule,
        batch_size=50, max_steps=15,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"rule={rule_name}: LL not finite after incremental EM"


def test_incremental_em_fine_tuning():
    """IncrementalEMFitter with inner_iter > 1 (fine-tuning mode)."""
    X = _make_data()
    model = _make_models(X)["VG"]

    fitter = IncrementalEMFitter(
        eta_update=IdentityUpdate(),
        batch_size=100, max_steps=5, inner_iter=3,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"fine-tuning LL not finite: {ll}"


def test_incremental_em_shrinkage():
    """IncrementalEMFitter with ShrinkageUpdate."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = model.compute_eta_from_model()

    fitter = IncrementalEMFitter(
        eta_update=ShrinkageUpdate(eta0, tau=0.5),
        batch_size=50, max_steps=15,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"shrinkage LL not finite: {ll}"


# ---------------------------------------------------------------------------
# BatchEMFitter + eta_update (shrinkage)
# ---------------------------------------------------------------------------

def test_batch_em_with_shrinkage():
    """BatchEMFitter with ShrinkageUpdate penalises toward prior."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta_prior = model.compute_eta_from_model()

    fitter = BatchEMFitter(
        eta_update=ShrinkageUpdate(eta_prior, tau=0.5),
        max_iter=10, tol=1e-3,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"batch EM + shrinkage LL not finite: {ll}"


def test_batch_em_eta_update_none_unchanged():
    """BatchEMFitter with eta_update=None behaves identically to default."""
    X = _make_data()
    model = _make_models(X)["VG"]

    result1 = BatchEMFitter(
        max_iter=5, tol=1e-6,
        e_step_backend='jax', m_step_backend='jax',
    ).fit(model, X)

    result2 = BatchEMFitter(
        eta_update=None,
        max_iter=5, tol=1e-6,
        e_step_backend='jax', m_step_backend='jax',
    ).fit(model, X)

    np.testing.assert_allclose(
        np.array(result1.model._joint.mu),
        np.array(result2.model._joint.mu),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# IncrementalEMFitter — LL improves from default init
# ---------------------------------------------------------------------------

def test_incremental_em_ll_improves():
    """Incremental EM on enough steps should improve LL over the init (VG)."""
    X = _make_data(n=300)
    model = _make_models(X)["VG"]

    ll_before = float(model.marginal_log_likelihood(X))

    fitter = IncrementalEMFitter(
        eta_update=SampleWeightedUpdate(),
        batch_size=150, max_steps=30,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll_after = float(result.model.marginal_log_likelihood(X))

    assert np.isfinite(ll_after), "VG LL not finite"
    assert ll_after > ll_before - 1.0, (
        f"VG LL degraded too much: before={ll_before:.4f} after={ll_after:.4f}")
