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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scipy.special import digamma

from normix import FactorNormalInverseGamma, FactorVarianceGamma, Gamma
from normix.distributions.variance_gamma import VarianceGamma
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.fitting.em import BatchEMFitter, IncrementalEMFitter
from normix.fitting.eta import NormalMixtureEta, affine_combine, _pack_gig_eta, _unpack_gig_eta
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


@pytest.mark.contract
@pytest.mark.parametrize("batched", [False, True])
def test_pack_gig_eta_positions_and_round_trip(batched):
    moments = {"E_inv_Y": jnp.array(2.0), "E_Y": jnp.array(5.0), "E_log_Y": jnp.array(-0.5)}
    if batched:
        moments = {name: jnp.stack((value, value + 1.0)) for name, value in moments.items()}
    packed = jax.jit(_pack_gig_eta)(**moments)
    np.testing.assert_array_equal(packed[..., 0], moments["E_log_Y"])
    np.testing.assert_array_equal(packed[..., 1], moments["E_inv_Y"])
    np.testing.assert_array_equal(packed[..., 2], moments["E_Y"])
    for name, value in _unpack_gig_eta(packed).items():
        np.testing.assert_array_equal(value, moments[name])


@pytest.mark.contract
@pytest.mark.parametrize("batched", [False, True])
def test_unpack_gig_eta_names_and_round_trip(batched):
    packed = jnp.array([[-0.5, 2.0, 5.0], [0.5, 3.0, 6.0]]) if batched else jnp.array([-0.5, 2.0, 5.0])
    moments = jax.jit(_unpack_gig_eta)(packed)
    assert set(moments) == {"E_log_Y", "E_inv_Y", "E_Y"}
    np.testing.assert_array_equal(moments["E_log_Y"], packed[..., 0])
    np.testing.assert_array_equal(moments["E_inv_Y"], packed[..., 1])
    np.testing.assert_array_equal(moments["E_Y"], packed[..., 2])
    np.testing.assert_array_equal(_pack_gig_eta(**moments), packed)


@pytest.mark.contract
def test_gig_eta_helpers_vmap_grad():
    def weighted(eta):
        moments = _unpack_gig_eta(eta)
        return 2.0 * moments["E_log_Y"] + 3.0 * moments["E_inv_Y"] + 7.0 * moments["E_Y"]

    values = jnp.array([[-0.5, 2.0, 5.0], [0.5, 3.0, 6.0]])
    np.testing.assert_array_equal(jax.jit(jax.vmap(weighted))(values), jnp.array([40.0, 52.0]))
    np.testing.assert_array_equal(jax.jit(jax.grad(weighted))(values[0]), jnp.array([2.0, 3.0, 7.0]))
    packed = jax.jit(jax.vmap(lambda row: _pack_gig_eta(E_log_Y=row[0], E_inv_Y=row[1], E_Y=row[2])))(values)
    np.testing.assert_array_equal(packed, values)

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

@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 1.0, 1.01, 1.05, 1.1, 2.0])
@pytest.mark.parametrize("full,factor", [(VarianceGamma, FactorVarianceGamma),
                                         (NormalInverseGamma, FactorNormalInverseGamma)])
def test_prior_moments_and_em_reference(alpha, full, factor):
    from normix.fitting.shrinkage_targets import _em_reference_eta

    params = dict(mu=jnp.array([1.0]), gamma=jnp.array([0.5]), alpha=alpha, beta=2.0)
    model = full.from_classical(**params, sigma=jnp.array([[2.0]]))
    fa = factor.from_classical(**params, F=jnp.array([[1.0]]), D=jnp.array([1.0]))

    def check(eta, a):
        log_y = digamma(a) - np.log(2.0)
        divergent = np.inf if a <= 1.0 else 2.0 / (a - 1.0)
        expected = ((log_y, divergent, a / 2.0) if full is VarianceGamma
                    else (-log_y, a / 2.0, divergent))
        np.testing.assert_allclose([eta.E_log_Y, eta.E_inv_Y, eta.E_Y], expected, rtol=1e-12)
        assert eta.E_inv_Y * eta.E_Y > 1
        assert -jnp.log(eta.E_inv_Y) <= eta.E_log_Y <= jnp.log(eta.E_Y)
        assert all(not np.isnan(v).any() for v in jax.tree.leaves(eta))

    for m in (model, model.project(jnp.ones(1)), fa):
        eta = m.compute_eta_from_model()
        check(eta, alpha)
        check(eta0_from_model(m), alpha)
        if m is not fa:
            for target in (eta0_with_sigma(m, jnp.eye(1)), eta0_diagonal(m, [1.0]),
                           eta0_isotropic(m, 1.0)):
                check(target, alpha)
        ref = _em_reference_eta(m)
        check(ref, max(alpha, 1.1))
        if alpha >= 1.1:
            for field in vars(eta):
                np.testing.assert_array_equal(getattr(eta, field), getattr(ref, field))
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(ref))
        for finite_eta in ([ref, eta] if alpha > 1.0 else [ref]):
            fitted = m.m_step_normal(finite_eta)
            np.testing.assert_allclose(fitted.mu, m.mu, atol=1e-12)
            np.testing.assert_allclose(fitted.gamma, m.gamma, atol=1e-12)
            np.testing.assert_allclose(fitted.sigma(), m.sigma(), atol=2e-8, rtol=0)


@pytest.mark.parametrize("full,factor", [(VarianceGamma, FactorVarianceGamma),
                                         (NormalInverseGamma, FactorNormalInverseGamma)])
@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.75])
def test_half_moment_existence(full, factor, alpha):
    params = dict(mu=jnp.array([1e9, 0.0]), gamma=jnp.array([0.0, 0.5]),
                  alpha=alpha, beta=2.0)
    F = jnp.array([[0.5], [1.0]])
    D = jnp.ones(2)
    m = full.from_classical(**params, sigma=F @ F.T + jnp.diag(D))
    fa = factor.from_classical(**params, F=F, D=D)
    joint_eta = m.joint.expectation_params()
    np.testing.assert_array_equal(joint_eta[3:5], m.mean())
    eta = fa.compute_eta_from_model()
    np.testing.assert_array_equal(eta.origin, m.mu)
    if full is VarianceGamma:
        expected = np.inf if alpha <= 0.5 else 0.5
        assert joint_eta[6] == expected
        assert eta.E_X_inv_Y[1] == expected
        np.testing.assert_allclose(eta.E_XZT_inv_sqrtY, F)
        assert eta.E_Z_inv_sqrtY[0] == (np.inf if alpha <= 0.5 else 0.0)
        np.testing.assert_allclose(eta.E_XXT_inv_Y, m.sigma() + jnp.outer(m.gamma, m.gamma)*eta.E_Y)
    else:
        assert eta.E_X[0] == (np.inf if alpha <= 0.5 else 0.0)
        assert eta.E_Z_sqrtY[0] == (np.inf if alpha <= 0.5 else 0.0)
        assert eta.E_XXT_inv_Y[0, 0] == fa.sigma()[0, 0]
        assert eta.E_XXT_inv_Y[0, 1] == (np.inf if alpha <= 0.5 else fa.sigma()[0, 1])
    np.testing.assert_allclose(eta.E_ZZT, jnp.eye(1))


@pytest.mark.parametrize("factor", [FactorVarianceGamma, FactorNormalInverseGamma])
def test_factor_public_prior_under_vmap(factor):
    def prior(a):
        model = factor.from_classical(mu=jnp.zeros(1), gamma=jnp.zeros(1), F=jnp.ones((1, 1)),
                                      D=jnp.ones(1), alpha=a, beta=2.0)
        return model.compute_eta_from_model()

    shapes = jnp.array([0.2, 0.75, 1.0, 1.05, 2.0])
    batched = jax.jit(jax.vmap(prior))(shapes)
    for i, a in enumerate(shapes):
        expected = prior(a)
        for field in vars(batched):
            np.testing.assert_allclose(getattr(batched, field)[i], getattr(expected, field),
                                       atol=1e-14, rtol=1e-13)


def test_zero_dispersion_target_has_no_noise_moment_requirement():
    model = NormalInverseGamma.from_classical(
        mu=jnp.ones(2), gamma=jnp.array([0.0, 1.0]), sigma=jnp.eye(2), alpha=0.2, beta=2.0)
    eta = eta0_diagonal(model, [0.0, 1.0])
    np.testing.assert_array_equal(eta.E_X, [1.0, np.inf])
    np.testing.assert_allclose(eta.E_XXT_inv_Y, [[0.1, 1.1], [1.1, np.inf]])


def test_gamma_negative_raw_moments():
    sub = Gamma(alpha=0.05, beta=2.0)
    assert np.isposinf(sub.raw_moment(-1.0))
    np.testing.assert_array_equal(sub.raw_moments(jnp.array([-1.0, -0.05, 0.0])),
                                  [np.inf, np.inf, 1.0])


@pytest.mark.parametrize("full", [VarianceGamma, NormalInverseGamma])
def test_prior_pole_under_transforms(full):
    model = full.from_classical(mu=jnp.zeros(1), gamma=jnp.zeros(1), sigma=jnp.eye(1),
                                alpha=2.0, beta=2.0)

    def moment(alpha):
        eta = model.replace(alpha=alpha).compute_eta_from_model()
        return eta.E_inv_Y if full is VarianceGamma else eta.E_Y

    alphas = jnp.array([0.5, 1.0, 1.05])
    values, grads = jax.jit(jax.vmap(jax.value_and_grad(moment)))(alphas)
    np.testing.assert_allclose(values, [np.inf, np.inf, 40.0])
    np.testing.assert_allclose(grads, [0.0, 0.0, -800.0])


def test_incremental_em_heavy_peaked_vg_finite():
    """A finite EM reference starts the stream for heavy-peaked VG data."""
    vg_true = VarianceGamma.from_classical(
        mu=jnp.array([0.0]), gamma=jnp.array([0.2]),
        sigma=jnp.array([[1.0]]), alpha=0.7, beta=1.0,
    )
    X = vg_true.rvs(2000, seed=0).reshape(-1, 1)
    # Keep a near-mode observation in the stream throughout.
    X = jnp.concatenate([jnp.mean(X, axis=0, keepdims=True), X], axis=0)
    model = VarianceGamma.default_init(X)

    fitter = IncrementalEMFitter(
        eta_update=RobbinsMonroUpdate(tau0=10.0),
        batch_size=200, max_steps=30,
        e_step_backend='jax', m_step_backend='jax',
    )
    result = fitter.fit(model, X, key=KEY)

    j = result.model._joint
    for leaf in (j.mu, j.gamma, j.L_Sigma, j.alpha, j.beta):
        assert jnp.all(jnp.isfinite(leaf)), "incremental EM produced a non-finite param"
    eta = result.model.compute_eta_from_model()
    assert float(eta.E_inv_Y) > 0.0 and float(eta.E_Y) > 0.0

# ---------------------------------------------------------------------------
# affine_combine
# ---------------------------------------------------------------------------

def test_affine_combine_identity():
    """affine_combine with b=0, c=1 returns eta_new."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X).eta
    result = affine_combine(eta1, eta2, b=0.0, c=1.0)
    np.testing.assert_allclose(
        np.array(result.E_X), np.array(eta2.E_X), atol=1e-12)

def test_affine_combine_midpoint():
    """affine_combine with b=0.5, c=0.5 gives midpoint."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X).eta
    result = affine_combine(eta1, eta2, b=0.5, c=0.5)
    expected = 0.5 * np.array(eta1.E_X) + 0.5 * np.array(eta2.E_X)
    np.testing.assert_allclose(np.array(result.E_X), expected, atol=1e-12)

def test_affine_combine_with_shift():
    """affine_combine with additive shift a."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X).eta
    shift = jax.tree.map(lambda x: 0.1 * jnp.ones_like(x), eta1)
    result = affine_combine(eta1, eta2, b=0.0, c=1.0, a=shift)
    for name in ("E_inv_Y", "E_Y", "E_log_Y", "E_X", "E_X_inv_Y", "E_XXT_inv_Y"):
        np.testing.assert_allclose(getattr(result, name), getattr(eta2, name) + 0.1,
                                   rtol=0, atol=1e-12)
    np.testing.assert_array_equal(result.origin, eta2.origin)


def test_affine_combine_weight_pytree_origin_is_not_a_weight():
    """A per-field weight built with tree.map matches the scalar weight."""
    X = _make_data()
    model = _make_models(X)["VG"]
    eta1 = model.compute_eta_from_model()
    eta2 = model.e_step(X).eta
    b = jax.tree.map(lambda x: jnp.full_like(x, 0.3), eta1)
    c = jax.tree.map(lambda x: jnp.full_like(x, 0.7), eta1)
    actual = affine_combine(eta1, eta2, b=b, c=c)
    expected = affine_combine(eta1, eta2, b=0.3, c=0.7)
    for got, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, ref)

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
    eta_new = model.e_step(X).eta

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
    eta_new = model.e_step(X).eta

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
    eta_new = model.e_step(X).eta

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
    eta_new = model.e_step(X).eta

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
    eta_new = model.e_step(X).eta

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
# IncrementalEMFitter — lax.scan parity (JAX backends, verbose==0 path)
# ---------------------------------------------------------------------------

def test_incremental_em_scan_matches_python_loop():
    """Pre-materialized RNG + scan replay matches the Python minibatch loop."""
    from normix.fitting.em import _materialize_incremental_subkeys

    X = _make_data(key=jax.random.key(17), n=120, d=3)
    model_basis = _make_models(X)["VG"]
    bs = min(35, X.shape[0])

    rule = RobbinsMonroUpdate(tau0=11.0)
    fitter = IncrementalEMFitter(
        eta_update=rule,
        batch_size=bs,
        max_steps=9,
        inner_iter=2,
        e_step_backend='jax',
        m_step_backend='jax',
        verbose=0,
    )
    mk = jax.random.key(701)
    step_keys = _materialize_incremental_subkeys(mk, fitter.max_steps)

    def clone_leaves(mod):
        return jax.tree.map(lambda z: jnp.array(z), mod)

    r_scan = fitter._fit_incremental_scan(
        clone_leaves(model_basis),
        jnp.asarray(X, dtype=jnp.float64),
        int(X.shape[0]),
        int(bs),
        step_keys,
        None,
    )
    r_py = fitter._fit_incremental_python(
        clone_leaves(model_basis),
        jnp.asarray(X, dtype=jnp.float64),
        int(X.shape[0]),
        int(bs),
        step_keys,
        type(model_basis).__name__,
        None,
    )

    np.testing.assert_allclose(
        np.asarray(r_scan.model._joint.mu),
        np.asarray(r_py.model._joint.mu),
        atol=5e-10,
        rtol=5e-10,
    )
    np.testing.assert_allclose(
        np.asarray(r_scan.param_changes),
        np.asarray(r_py.param_changes),
        atol=1e-9,
        rtol=1e-9,
    )

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


@pytest.fixture(params=[False, True], ids=["full", "factor"])
def vg_model(request):
    params = dict(mu=jnp.ones(1), gamma=jnp.ones(1), alpha=2.0, beta=2.0)
    if request.param:
        return FactorVarianceGamma.from_classical(**params, F=jnp.ones((1, 1)), D=jnp.ones(1))
    return VarianceGamma.from_classical(**params, sigma=jnp.eye(1))


@pytest.mark.contract
def test_infeasible_mstep(vg_model):
    eta = vg_model.compute_eta_from_model()
    invalid_stats = [
        eqx.tree_at(lambda e: e.E_Y, eta, jnp.array(0.25)),
        eqx.tree_at(lambda e: e.E_XXT_inv_Y, eta, jnp.array([[jnp.inf]])),
    ]
    if isinstance(vg_model, FactorVarianceGamma):
        invalid_stats.append(eqx.tree_at(lambda e: e.E_inv_Y, eta, 1.0 / eta.E_Y))
        for cross_moment in (1.0, 2.0):
            invalid_stats.append(eqx.tree_at(lambda e: e.E_Z_sqrtY, eta, jnp.array([cross_moment])))

    def fit(stats):
        return vg_model.m_step_normal(stats).mu

    batched_fit = jax.jit(jax.vmap(fit))
    pair = jax.tree.map(lambda x: jnp.stack((x, x)), eta)
    np.testing.assert_allclose(batched_fit(pair), jnp.stack((vg_model.mu, vg_model.mu)), atol=1e-12)
    for invalid in invalid_stats:
        with pytest.raises(ValueError, match="Infeasible normal M-step"):
            fit(invalid)
        pair = jax.tree.map(lambda x, y: jnp.stack((x, y)), eta, invalid)
        with pytest.raises((RuntimeError, ValueError), match="Infeasible normal M-step"):
            batched_fit(pair).block_until_ready()


@pytest.mark.contract
def test_incremental_scan_rejects_infeasible_update():
    X = _make_data()
    model = _make_models(X)["VG"]
    fitter = IncrementalEMFitter(eta_update=AffineUpdate(b=0.0, c=0.0), inner_iter=2,
                                 batch_size=32, max_steps=1,
                                 e_step_backend="jax", m_step_backend="jax")
    with pytest.raises((RuntimeError, ValueError), match="Infeasible normal M-step"):
        jax.block_until_ready(fitter.fit(model, X, key=KEY).model)


def test_shrinkage_feasibility():
    model = VarianceGamma.from_classical(mu=jnp.zeros(1), gamma=jnp.ones(1),
                                         alpha=2.0, beta=0.25, sigma=jnp.eye(1))
    target = model.compute_eta_from_model()
    other = model.replace(beta=8.0).compute_eta_from_model()
    with pytest.raises(ValueError, match="finite target"):
        Shrinkage(IdentityUpdate(), model.replace(alpha=0.2).compute_eta_from_model())
    for tau in (0.0, 1.0, 100.0):
        rule = Shrinkage(IdentityUpdate(), target, tau)
        eta, _ = rule(other, other, 0, 1, {})
        assert eta.E_inv_Y * eta.E_Y > 1


@pytest.mark.contract
@pytest.mark.parametrize("field", ["E_inv_Y", "E_Y", "E_log_Y"])
def test_shrinkage_rejects_unequal_subordinator_tau(vg_model, field):
    target = eta0_from_model(vg_model)
    zeros = jax.tree.map(lambda _: 0.0, target)

    def build(weight, prior=None):
        tau = eqx.tree_at(lambda e: getattr(e, field), zeros, weight)
        return Shrinkage(IdentityUpdate(), prior, tau).tau.E_X

    message = "E_inv_Y, E_Y, and E_log_Y"
    with pytest.raises(ValueError, match=message):
        build(0.5, target)
    with pytest.raises(eqx.EquinoxRuntimeError, match=message):
        jax.block_until_ready(eqx.filter_jit(build)(jnp.array(0.5)))


@pytest.mark.contract
@pytest.mark.parametrize("shared_tau", [0.0, 0.5])
def test_shrinkage_shared_subordinator_tau(vg_model, shared_tau):
    target = eta0_from_model(vg_model)
    other = vg_model.e_step(jnp.array([[-1.0], [0.5], [3.0]]), backend="jax").eta
    tau = jax.tree.map(jnp.zeros_like, target)
    tau = eqx.tree_at(lambda e: (e.E_inv_Y, e.E_Y, e.E_log_Y, e.E_X), tau,
                      (shared_tau, shared_tau, shared_tau, jnp.ones_like(tau.E_X)))
    if isinstance(vg_model, FactorVarianceGamma) and shared_tau:
        tau = eqx.tree_at(lambda e: e.E_Z_sqrtY, tau, jnp.full_like(tau.E_Z_sqrtY, 2.0))

    @jax.jit
    def update(weights):
        rule = Shrinkage(IdentityUpdate(), target, weights)
        return rule(other, other, 0, 1, {})[0]

    result = update(tau)
    expected = jax.tree.map(lambda t, a, b: (t * a + b) / (1.0 + t), tau, target, other)
    for actual, want in zip(jax.tree.leaves(result), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, want, atol=1e-12)
    assert result.E_inv_Y * result.E_Y >= 1.0


@pytest.mark.parametrize("cls", [VarianceGamma, NormalInverseGamma])
def test_small_shape_em_paths(cls):
    model = cls.from_classical(mu=jnp.array([1.0]), gamma=jnp.array([0.5]),
                               sigma=jnp.array([[2.0]]), alpha=0.05, beta=2.0)
    rng = np.random.default_rng(2026)
    g = rng.gamma(0.65, 1.0, 64)
    y = g if cls is VarianceGamma else 1.0 / g
    X = jnp.asarray((1.0 + 0.5 * y + np.sqrt(2 * y) * rng.normal(size=y.size))[:, None])
    rule = Shrinkage(EWMAUpdate(w=0.25), tau=100.0)
    options = dict(eta_update=rule, e_step_backend='jax', m_step_backend='jax')
    results = [BatchEMFitter(algorithm=alg, max_iter=2, **options).fit(model, X)
               for alg in ('em', 'mcecm')]
    for inner in (1, 2):
        for verbose in (0, 1):
            fitter = IncrementalEMFitter(inner_iter=inner, batch_size=32, max_steps=2,
                                         verbose=verbose, **options)
            results.append(fitter.fit(model, X, key=KEY))
    for result in results:
        assert result.n_iter == 2 and not result.diverged and not result.solver_failed
        assert all(np.isfinite(v).all() for v in jax.tree.leaves(result.model))
        assert result.model.mu[0] > 0 and result.model.gamma[0] > 0
