"""Translation, support, and fit_mle robustness.

These pin the three failures verified on master 85f835e: the normal M-step
loses Σ when moments are stored uncentered, positive-support densities are
NaN off the support, and fit_mle returns a Newton cap or a clamp instead of
raising on data with no finite MLE.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import GIG, Gamma, InverseGamma, InverseGaussian, VarianceGamma, FactorVarianceGamma
from normix.fitting.em import BatchEMFitter
from normix.fitting.eta_rules import IdentityUpdate, Shrinkage
from normix.fitting.shrinkage_targets import eta0_from_model, eta0_with_sigma
from normix.fitting.eta import affine_combine
from normix.mixtures.joint import JointNormalMixture
from normix.utils.constants import SIGMA_REG

pytestmark = pytest.mark.contract

_FAMILIES = [Gamma, InverseGamma, InverseGaussian, GIG]


def _location_model(factor, shift=0.0):
    common = dict(mu=jnp.array([shift]), gamma=jnp.array([0.3]), alpha=3.0, beta=2.0)
    if factor:
        return FactorVarianceGamma.from_classical(**common, F=jnp.array([[0.5]]), D=jnp.array([0.75]))
    return VarianceGamma.from_classical(**common, sigma=jnp.eye(1))


@pytest.mark.parametrize("shift", [0.0, 1e8])
def test_large_location_model_roundtrip(shift):
    model = VarianceGamma.from_classical(
        mu=jnp.array([shift]), gamma=jnp.array([0.3]),
        sigma=jnp.eye(1), alpha=3.0, beta=2.0,
    )
    eta = model.compute_eta_from_model()
    np.testing.assert_allclose(eta.E_X + eta.origin, [shift + 0.45], rtol=0, atol=1e-8)
    recovered = VarianceGamma.from_expectation(eta)
    np.testing.assert_allclose(recovered.mu, model.mu, rtol=0, atol=1e-8)
    np.testing.assert_allclose(recovered.gamma, model.gamma, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered.sigma(), model.sigma() + SIGMA_REG,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("backend", ["jax", "cpu"])
@pytest.mark.parametrize("factor", [False, True])
def test_large_location_estep_roundtrip(weighted, backend, factor):
    base = _location_model(factor)
    X = jnp.array([[-2.0], [-0.5], [0.0], [0.25], [1.0], [3.0]])
    weights = jnp.arange(1.0, 7.0) if weighted else None
    shifted = base.replace(mu=base.mu + 1e8)
    actual = shifted.m_step(shifted.e_step(
        X + 1e8, backend=backend, sample_weight=weights).eta)
    expected = base.m_step(base.e_step(X, backend=backend, sample_weight=weights).eta)
    np.testing.assert_allclose(actual.mu, expected.mu + 1e8, rtol=0, atol=2e-8)
    np.testing.assert_allclose(actual.gamma, expected.gamma, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.sigma(), expected.sigma(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("algorithm,backend", [("em", "jax"), ("mcecm", "cpu")])
def test_large_location_distinct_shrinkage_prior(factor, weighted, algorithm, backend):
    base = _location_model(factor)
    X = jnp.array([[-2.0], [-0.5], [0.0], [0.25], [1.0], [3.0]])
    weights = jnp.arange(1.0, 7.0) if weighted else None
    results = []
    for shift in [0.0, 1e8]:
        init = base.replace(mu=base.mu + shift)
        prior = base.replace(mu=base.mu + shift + 0.25)
        result = BatchEMFitter(
            max_iter=3, tol=0, algorithm=algorithm, track_ll=True,
            eta_update=Shrinkage(IdentityUpdate(), prior.compute_eta_from_model(), 0.2),
            e_step_backend=backend, m_step_backend=backend,
        ).fit(init, X + shift, sample_weight=weights)
        assert not result.diverged and not result.solver_failed
        results.append(result)
    expected, actual = results
    np.testing.assert_allclose(actual.model.mu, expected.model.mu + 1e8, rtol=0, atol=3e-8)
    np.testing.assert_allclose(actual.model.gamma, expected.model.gamma, rtol=0, atol=5e-8)
    np.testing.assert_allclose(actual.model.sigma(), expected.model.sigma(), rtol=0, atol=5e-8)
    np.testing.assert_allclose(actual.log_likelihoods, expected.log_likelihoods, rtol=0, atol=5e-8)


@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("per_field", [False, True])
def test_shrinkage_aligns_explicit_origins_under_jit(factor, per_field):
    model = _location_model(factor)
    eta = model.compute_eta_from_model().rebase(jnp.array([0.25]))
    prior = model.replace(mu=model.mu + 0.5).compute_eta_from_model()
    tau = jax.tree.map(lambda x: jnp.full_like(x, 0.2), prior) if per_field else 0.2
    rule = Shrinkage(IdentityUpdate(), prior, tau)
    result, _ = jax.jit(lambda e: rule(e, e, 0, 6, {}))(eta)
    expected = affine_combine(prior, eta.rebase(jnp.zeros(1)), 1/6, 5/6)
    actual = result.rebase(jnp.zeros(1))
    np.testing.assert_array_equal(result.origin, eta.origin)
    for got, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-14)


def test_large_location_custom_sigma_prior_and_vmap_roundtrip():
    model = _location_model(False, 1e8)
    target = eta0_with_sigma(model, jnp.array([[2.0]]))
    recovered = VarianceGamma.from_expectation(target)
    np.testing.assert_allclose(recovered.mu, model.mu, rtol=0, atol=1e-8)
    np.testing.assert_allclose(recovered.sigma(), [[2.0 + SIGMA_REG]], rtol=0, atol=1e-12)
    roundtrip = jax.jit(jax.vmap(lambda shift: JointNormalMixture._mstep_normal_params(
        _location_model(False, shift).compute_eta_from_model())[0]))
    shifts = jnp.array([0.0, 1e8, -1e8])
    np.testing.assert_allclose(roundtrip(shifts)[:, 0], shifts, rtol=0, atol=1e-8)


def _package_sigma(model) -> np.ndarray:
    _, _, L = JointNormalMixture._mstep_normal_params(model.compute_eta_from_model())
    return np.asarray(L @ L.T)


@pytest.mark.parametrize("shift", [0.0, 1e8, 1e10])
@pytest.mark.parametrize("gamma", [0.0, 0.3])
def test_exact_vg_sigma_is_translation_invariant(shift, gamma):
    """Σ = 1 survives the M-step at a large location, including γ = 0.

    γ = 0 is the silent failure: the uncentered second moment rounds to
    c², the M-step returns only SIGMA_REG, and the fit still reports success.
    """
    model = VarianceGamma.from_classical(
        mu=jnp.array([shift]),
        gamma=jnp.array([gamma]),
        sigma=jnp.eye(1),
        alpha=3.0,
        beta=2.0,
    )
    sigma = _package_sigma(model)
    assert np.isfinite(sigma).all()
    np.testing.assert_allclose(sigma, np.eye(1), atol=1e-6, rtol=0.0)
    assert float(sigma[0, 0]) > 100.0 * SIGMA_REG


@pytest.mark.parametrize("shift", [1e8, 1e10])
def test_fit_gamma_zero_does_not_collapse_sigma(shift):
    """A shifted γ = 0 sample must not fit a jitter covariance."""
    base = VarianceGamma.from_classical(
        mu=jnp.zeros(1),
        gamma=jnp.zeros(1),
        sigma=jnp.eye(1),
        alpha=3.0,
        beta=2.0,
    )
    X = jnp.asarray(base.rvs(400, seed=1)) + shift
    result = VarianceGamma.default_init(X).fit(
        X, max_iter=15, tol=1e-8,
        e_step_backend="jax", m_step_backend="jax",
    )
    sigma = np.asarray(result.model.sigma())
    assert result.diverged is False
    assert np.isfinite(sigma).all()
    assert float(sigma[0, 0]) > 0.1


def test_shrinkage_from_model_keeps_sigma_at_large_location():
    """A self-prior must not be translated as if it still contained μ.

    ``compute_eta_from_model`` stores (s4, s5, s6) about μ at this
    location. Subtracting the sample mean from those moments puts
    s1 c c^T back into s6, and the fit reports success with Σ ~ 1e15.
    """
    shift = 1e8
    base = VarianceGamma.from_classical(
        mu=jnp.zeros(1),
        gamma=jnp.zeros(1),
        sigma=jnp.eye(1),
        alpha=3.0,
        beta=2.0,
    )
    X = jnp.asarray(base.rvs(200, seed=0)) + shift
    init = VarianceGamma.default_init(X)
    result = BatchEMFitter(
        max_iter=8,
        tol=1e-6,
        eta_update=Shrinkage(IdentityUpdate(), eta0_from_model(init), tau=0.3),
        e_step_backend="jax",
        m_step_backend="jax",
    ).fit(init, X)
    sigma = float(np.asarray(result.model.sigma())[0, 0])
    assert result.diverged is False
    assert math.isfinite(sigma)
    assert 0.1 < sigma < 100.0


def test_log_prob_pdf_cdf_off_support_and_at_zero():
    """x < 0 is off support; x = 0 is the one-sided limit. Grads stay non-NaN."""
    cases = [
        (Gamma(0.5, 2.0), math.inf, lambda t: Gamma(t, 2.0), 0.5),
        (Gamma(1.0, 2.0), math.log(2.0), lambda t: Gamma(t, 2.0), 1.0),
        (Gamma(3.0, 2.0), -math.inf, lambda t: Gamma(t, 2.0), 3.0),
        (InverseGamma(2.0, 1.0), -math.inf, lambda t: InverseGamma(t, 1.0), 2.0),
        (InverseGaussian(1.0, 2.0), -math.inf, lambda t: InverseGaussian(t, 2.0), 1.0),
        (GIG(0.5, 1.0, 1.0), -math.inf, lambda t: GIG(t, 1.0, 1.0), 0.5),
        (GIG(1.0, 1.0, 1.0), -math.inf, lambda t: GIG(t, 1.0, 1.0), 1.0),
        (GIG(2.0, 1.0, 1.0), -math.inf, lambda t: GIG(t, 1.0, 1.0), 2.0),
    ]
    for dist, log_at_zero, build, t0 in cases:
        # One compiled gradient per case, reused for both x values.
        grad_x = jax.jit(jax.grad(lambda z, dist=dist: dist.log_prob(z)))
        grad_t = jax.jit(jax.grad(lambda t, x, build=build: build(t).log_prob(x)))
        for x in (-1.0, -1e-300):
            lp = float(dist.log_prob(x))
            assert lp == -math.inf
            assert float(dist.pdf(x)) == 0.0
            assert float(dist.cdf(x)) == 0.0
            gx = float(grad_x(jnp.float64(x)))
            gt = float(grad_t(jnp.float64(t0), jnp.float64(x)))
            assert not math.isnan(gx)
            assert not math.isnan(gt)
        lp0 = float(dist.log_prob(0.0))
        assert not math.isnan(lp0)
        if math.isinf(log_at_zero):
            assert lp0 == log_at_zero
        else:
            assert lp0 == pytest.approx(log_at_zero, rel=1e-12, abs=1e-12)
        assert float(dist.cdf(0.0)) == 0.0
        pdf0 = float(dist.pdf(0.0))
        assert not math.isnan(pdf0)


@pytest.mark.parametrize("cls", _FAMILIES)
@pytest.mark.parametrize("label,x", [
    ("empty", np.array([])),
    ("n=1", np.array([2.0])),
    ("constant", np.full(50, 2.0)),
    ("nonpositive", np.array([0.0, 1.0, 2.0])),
    ("negative", np.array([-1.0, 1.0, 2.0])),
    ("nan", np.array([np.nan, 1.0, 2.0])),
    ("inf", np.array([np.inf, 1.0, 2.0])),
])
def test_fit_mle_rejects_invalid_data(cls, label, x):
    with pytest.raises(ValueError):
        cls.fit_mle(jnp.asarray(x))


def test_fit_mle_constant_message_names_jensen_gap():
    with pytest.raises(ValueError, match="Jensen"):
        Gamma.fit_mle(jnp.full(20, 3.7))


def test_fit_mle_vmap_on_valid_samples():
    """The host-side check must not block a traced fit of a valid sample."""
    x = jnp.array([0.5, 1.0, 2.0, 3.0, 0.7, 1.4])
    fitted = jax.vmap(Gamma.fit_mle)(jnp.stack([x, x * 1.1]))
    assert np.isfinite(np.asarray(fitted.alpha)).all()
    assert np.isfinite(np.asarray(fitted.beta)).all()


def _vg_init():
    return VarianceGamma.from_classical(
        mu=jnp.zeros(2), gamma=jnp.zeros(2),
        sigma=jnp.eye(2), alpha=2.0, beta=1.0,
    )


def test_fit_rejects_empty_and_nonfinite_data():
    model = _vg_init()
    X = jnp.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.0, 0.1]])
    with pytest.raises(ValueError, match="empty"):
        model.fit(jnp.zeros((0, 2)))
    with pytest.raises(ValueError, match="empty"):
        model.fit(jnp.array([]))
    with pytest.raises(ValueError, match="finite"):
        model.fit(X.at[1, 0].set(jnp.nan))
    with pytest.raises(ValueError, match="finite"):
        model.fit(X.at[0, 1].set(jnp.inf))


def test_incremental_fit_rejects_empty_data():
    from normix.fitting.em import IncrementalEMFitter

    with pytest.raises(ValueError, match="empty"):
        IncrementalEMFitter().fit(
            _vg_init(), jnp.zeros((0, 2)), key=jax.random.PRNGKey(0),
        )


def test_fit_rejects_unknown_backend():
    model = _vg_init()
    X = jnp.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]])
    with pytest.raises(ValueError, match="e_step_backend"):
        model.fit(X, e_step_backend="gpu")
    with pytest.raises(ValueError, match="m_step_backend"):
        BatchEMFitter(m_step_backend="torch")


@pytest.mark.parametrize("dist", [
    Gamma(alpha=2.0, beta=1.0),
    InverseGamma(alpha=2.0, beta=1.0),
    InverseGaussian(mu=1.5, lam=2.0),
])
def test_unknown_backend_raises_on_expectation_api(dist):
    eta = dist.expectation_params()
    with pytest.raises(ValueError, match="backend"):
        dist.expectation_params(backend="foo")
    with pytest.raises(ValueError, match="backend"):
        type(dist).from_expectation(eta, backend="foo")


@pytest.mark.parametrize("eta", [
    np.array([0.0, 1.0]),
    np.array([1.0, 1.0]),
    np.array([0.0, -1.0]),
    np.array([np.nan, 1.0]),
    np.array([0.0]),
])
def test_gamma_from_expectation_rejects_impossible_eta(eta):
    with pytest.raises(ValueError, match="no finite Gamma"):
        Gamma.from_expectation(jnp.asarray(eta))


def test_gamma_from_expectation_accepts_true_moments_and_traced_eta():
    g = Gamma(alpha=2.5, beta=1.3)
    eta = g.expectation_params()
    back = Gamma.from_expectation(eta)
    np.testing.assert_allclose(float(back.alpha), 2.5, rtol=1e-8)
    np.testing.assert_allclose(float(back.beta), 1.3, rtol=1e-8)
    traced = jax.jit(lambda e: Gamma.from_expectation(e).alpha)(eta)
    np.testing.assert_allclose(float(traced), 2.5, rtol=1e-8)


def test_fit_mle_valid_sample_unchanged():
    x = jnp.array([0.5, 1.0, 2.0, 3.0, 0.7, 1.4])
    g = Gamma.fit_mle(x)
    assert math.isfinite(float(g.alpha)) and math.isfinite(float(g.beta))
    ig = InverseGaussian.fit_mle(x)
    mu = float(np.mean(np.asarray(x)))
    lam = 1.0 / float(np.mean(1.0 / np.asarray(x) - 1.0 / mu))
    np.testing.assert_allclose(float(ig.mu), mu, rtol=1e-9)
    np.testing.assert_allclose(float(ig.lam), lam, rtol=1e-9)
