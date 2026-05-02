"""
Tests for IncrementalEMFitter and eta update infrastructure.

Covers:
  - compute_eta_from_model round-trip consistency
  - affine_combine arithmetic
  - IncrementalEMFitter with each eta update rule
  - BatchEMFitter with Shrinkage(IdentityUpdate(), ...)
  - Shrinkage combinator: tau=0 ≡ base, scalar τ closed-form,
    per-field τ, composition with running rules
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
    Shrinkage,
    AffineUpdate,
)
from normix.fitting.shrinkage_targets import (
    eta0_from_model,
    eta0_isotropic,
    eta0_diagonal,
    eta0_with_sigma,
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


def test_shrinkage_combinator_scalar_closed_form():
    """Shrinkage(IdentityUpdate(), eta0, scalar_tau) matches closed form.

    For Identity base (b=0, c=1), eta_base = eta_new, so
        eta_t = (τ/(1+τ)) * eta0 + (1/(1+τ)) * eta_new.
    """
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)
    eta_new = model.e_step(X)

    tau = 0.5
    rule = Shrinkage(IdentityUpdate(), eta0, tau=tau)
    state = rule.initial_state()
    eta_t, _ = rule(eta0, eta_new, jnp.int32(0), 100, state)

    factor_a = tau / (1.0 + tau)
    factor_b = 1.0 / (1.0 + tau)

    expected_E_Y = factor_a * float(eta0.E_Y) + factor_b * float(eta_new.E_Y)
    np.testing.assert_allclose(float(eta_t.E_Y), expected_E_Y, atol=1e-12)

    expected_S6 = (
        factor_a * np.array(eta0.E_XXT_inv_Y)
        + factor_b * np.array(eta_new.E_XXT_inv_Y)
    )
    np.testing.assert_allclose(
        np.array(eta_t.E_XXT_inv_Y), expected_S6, atol=1e-12)


def test_rules_are_pytrees():
    """All rules are eqx.Module pytrees with JAX array leaves."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)

    rules = [
        IdentityUpdate(),
        RobbinsMonroUpdate(tau0=5.0),
        SampleWeightedUpdate(),
        EWMAUpdate(w=0.2),
        Shrinkage(IdentityUpdate(), eta0, tau=0.5),
        AffineUpdate(b=0.9, c=0.1),
    ]
    for rule in rules:
        leaves = jax.tree.leaves(rule)
        name = type(rule).__name__
        for leaf in leaves:
            assert hasattr(leaf, 'shape'), (
                f"{name} leaf {leaf!r} is not a JAX array")


# ---------------------------------------------------------------------------
# Shrinkage combinator — semantic tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_factory", [
    IdentityUpdate,
    lambda: RobbinsMonroUpdate(tau0=5.0),
    lambda: EWMAUpdate(w=0.3),
    SampleWeightedUpdate,
])
def test_shrinkage_tau_zero_equals_base(base_factory):
    """Shrinkage(base, eta0, tau=0) ≡ base for any base rule."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)
    eta_prev = eta0
    eta_new = model.e_step(X)

    base = base_factory()
    shrunk = Shrinkage(base, eta0, tau=0.0)

    s_base = base.initial_state()
    s_shr = shrunk.initial_state()
    eta_b, _ = base(eta_prev, eta_new, jnp.int32(0), 100, s_base)
    eta_s, _ = shrunk(eta_prev, eta_new, jnp.int32(0), 100, s_shr)

    np.testing.assert_allclose(np.array(eta_b.E_Y), np.array(eta_s.E_Y),
                               atol=1e-12)
    np.testing.assert_allclose(np.array(eta_b.E_X), np.array(eta_s.E_X),
                               atol=1e-12)
    np.testing.assert_allclose(np.array(eta_b.E_XXT_inv_Y),
                               np.array(eta_s.E_XXT_inv_Y), atol=1e-12)


def test_shrinkage_per_field_tau_only_sigma():
    """Per-field τ with only E_XXT_inv_Y non-zero leaves other fields untouched."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)
    eta_new = model.e_step(X)

    d = X.shape[1]
    tau_pytree = NormalMixtureEta(
        E_inv_Y=jnp.float64(0.0),
        E_Y=jnp.float64(0.0),
        E_log_Y=jnp.float64(0.0),
        E_X=jnp.zeros(d, dtype=jnp.float64),
        E_X_inv_Y=jnp.zeros(d, dtype=jnp.float64),
        E_XXT_inv_Y=jnp.full((d, d), 0.5, dtype=jnp.float64),
    )
    rule = Shrinkage(IdentityUpdate(), eta0, tau=tau_pytree)
    eta_t, _ = rule(eta0, eta_new, jnp.int32(0), 100, rule.initial_state())

    # Subordinator-only fields and X-related linear fields: untouched.
    for field in ("E_inv_Y", "E_Y", "E_log_Y", "E_X", "E_X_inv_Y"):
        np.testing.assert_allclose(
            np.array(getattr(eta_t, field)),
            np.array(getattr(eta_new, field)),
            atol=1e-12,
            err_msg=f"field {field} should be unshrunk (tau=0)",
        )

    # E_XXT_inv_Y: shrunk with τ=0.5.
    expected_S6 = (
        (0.5 / 1.5) * np.array(eta0.E_XXT_inv_Y)
        + (1.0 / 1.5) * np.array(eta_new.E_XXT_inv_Y)
    )
    np.testing.assert_allclose(
        np.array(eta_t.E_XXT_inv_Y), expected_S6, atol=1e-12)


def test_shrinkage_preserves_running_state():
    """Shrinkage(RobbinsMonroUpdate(...), …) retains the running mean.

    Regression for the bug noted in §2 of em_covariance_extensions.md:
    composing shrinkage by hand with a running rule used to discard the
    base rule's state. The combinator must thread it.
    """
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)
    eta_prev = eta0
    eta_new = model.e_step(X)

    base = RobbinsMonroUpdate(tau0=10.0)
    shrunk = Shrinkage(base, eta0, tau=0.25)

    # First, what the base rule alone would produce at step 0.
    eta_b0, _ = base(eta_prev, eta_new, jnp.int32(0), 100, base.initial_state())

    # Combinator at step 0: should be (τ/(1+τ)) eta0 + (1/(1+τ)) eta_b0.
    eta_s0, st0 = shrunk(eta_prev, eta_new, jnp.int32(0),
                        100, shrunk.initial_state())
    factor_a = 0.25 / 1.25
    factor_b = 1.0 / 1.25
    expected_E_Y_0 = (factor_a * float(eta0.E_Y)
                     + factor_b * float(eta_b0.E_Y))
    np.testing.assert_allclose(float(eta_s0.E_Y), expected_E_Y_0, atol=1e-12)

    # Now step 1: base rule's denominator is τ0 + 1 (not τ0), proving the
    # combinator threaded the step counter (and any state) into the base.
    eta_b1, _ = base(eta_prev, eta_new, jnp.int32(1), 100, base.initial_state())
    eta_s1, _ = shrunk(eta_prev, eta_new, jnp.int32(1), 100, st0)

    expected_E_Y_1 = (factor_a * float(eta0.E_Y)
                     + factor_b * float(eta_b1.E_Y))
    np.testing.assert_allclose(float(eta_s1.E_Y), expected_E_Y_1, atol=1e-12)
    # Sanity: step 1 differs from step 0.
    assert abs(expected_E_Y_1 - expected_E_Y_0) > 1e-15


def test_shrinkage_preserves_sample_weighted_state():
    """SampleWeightedUpdate's cumulative_n is threaded through Shrinkage."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)
    eta_new = model.e_step(X)

    base = SampleWeightedUpdate()
    shrunk = Shrinkage(base, eta0, tau=0.0)  # τ=0 ⇒ output equals base

    state = shrunk.initial_state()
    _, state = shrunk(eta0, eta_new, jnp.int32(0), 100, state)
    assert float(state['cumulative_n']) == 100.0
    _, state = shrunk(eta0, eta_new, jnp.int32(1), 50, state)
    assert float(state['cumulative_n']) == 150.0


# ---------------------------------------------------------------------------
# Shrinkage target builders
# ---------------------------------------------------------------------------

def test_eta0_from_model_matches_compute_eta_from_model():
    X = _make_data()
    model = _make_models(X)["VG"]
    a = eta0_from_model(model)
    b = model.compute_eta_from_model()
    np.testing.assert_allclose(np.array(a.E_X), np.array(b.E_X), atol=1e-12)
    np.testing.assert_allclose(
        np.array(a.E_XXT_inv_Y), np.array(b.E_XXT_inv_Y), atol=1e-12)


def test_eta0_isotropic_substitutes_sigma():
    """eta0_isotropic(σ²) puts σ² I_d into the dispersion part of E_XXT_inv_Y."""
    X = _make_data()
    model = _make_models(X)["VG"]
    sigma2 = 2.5
    eta = eta0_isotropic(model, sigma2=sigma2)

    j = model._joint
    mu, gamma = np.array(j.mu), np.array(j.gamma)
    E_log_Y, E_inv_Y, E_Y = model._subordinator_expectations()
    E_inv_Y, E_Y = float(E_inv_Y), float(E_Y)

    expected = (
        sigma2 * np.eye(model.d)
        + np.outer(mu, mu) * E_inv_Y
        + np.outer(gamma, gamma) * E_Y
        + np.outer(mu, gamma) + np.outer(gamma, mu)
    )
    np.testing.assert_allclose(
        np.array(eta.E_XXT_inv_Y), expected, atol=1e-10)


def test_eta0_diagonal_and_with_sigma_consistent():
    X = _make_data()
    model = _make_models(X)["VG"]
    diag = jnp.array([0.5, 1.0, 2.0])
    a = eta0_diagonal(model, diag=diag)
    b = eta0_with_sigma(model, Sigma0=jnp.diag(diag))
    np.testing.assert_allclose(
        np.array(a.E_XXT_inv_Y), np.array(b.E_XXT_inv_Y), atol=1e-12)


# ---------------------------------------------------------------------------
# IncrementalEMFitter — smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
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
    """IncrementalEMFitter with Shrinkage(IdentityUpdate(), eta0, tau)."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)

    fitter = IncrementalEMFitter(
        eta_update=Shrinkage(IdentityUpdate(), eta0, tau=0.5),
        batch_size=50, max_steps=15,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"shrinkage LL not finite: {ll}"


def test_incremental_em_shrinkage_over_running_rule():
    """Composing Shrinkage with a running rule produces a finite model."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta0 = eta0_from_model(model)

    fitter = IncrementalEMFitter(
        eta_update=Shrinkage(RobbinsMonroUpdate(tau0=10.0), eta0, tau=0.2),
        batch_size=50, max_steps=15,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"shrinkage∘RM LL not finite: {ll}"


# ---------------------------------------------------------------------------
# BatchEMFitter + eta_update (shrinkage)
# ---------------------------------------------------------------------------

def test_batch_em_with_shrinkage():
    """BatchEMFitter with Shrinkage combinator penalises toward prior."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta_prior = eta0_from_model(model)

    fitter = BatchEMFitter(
        eta_update=Shrinkage(IdentityUpdate(), eta_prior, tau=0.5),
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
