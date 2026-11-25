"""Observation-weight contracts for batch EM, initialization, and MLE."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import (
    BatchEMFitter,
    FactorGeneralizedHyperbolic,
    FactorNormalInverseGaussian,
    Gamma,
    GeneralizedHyperbolic,
    IdentityUpdate,
    MultivariateNormal,
    NormalInverseGaussian,
    Shrinkage,
)
from normix.exponential_family import _normalize_sample_weight
from normix.fitting.em import _validate_fit_data

pytestmark = pytest.mark.contract


def _sample():
    rng = np.random.default_rng(273)
    y = rng.wald(1.0, 0.7, 96)
    X = np.array([0.3, -0.2]) + y[:, None] * np.array([0.5, -0.3])
    X += np.sqrt(y[:, None]) * rng.normal(size=(96, 2))
    return jnp.asarray(X)


def _model(family):
    params = dict(mu=jnp.zeros(2), gamma=jnp.array([0.2, -0.1]))
    if family == 'factor':
        return FactorNormalInverseGaussian.from_classical(
            **params, F=jnp.array([[0.4], [0.2]]), D=jnp.ones(2),
            mu_ig=1.0, lam=1.0)
    if family == 'gh':
        return GeneralizedHyperbolic.from_classical(
            **params, sigma=jnp.eye(2), p=-0.5, a=1.0, b=1.0)
    return NormalInverseGaussian.from_classical(
        **params, sigma=jnp.eye(2), mu_ig=1.0, lam=1.0)


def _assert_tree_close(actual, expected, *, rtol=2e-10, atol=2e-11):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize('family', ['nig', 'gh', 'factor'])
@pytest.mark.parametrize('backend', ['jax', 'cpu'])
def test_weighted_estep_matches_replication_and_density(family, backend):
    X = _sample()[:12]
    weights = np.array([0, 1, 3, 2] * 3)
    model = _model(family)
    actual = model.e_step(X, backend=backend, sample_weight=weights)
    expected = model.e_step(jnp.repeat(X, weights, axis=0), backend=backend)
    _assert_tree_close(actual.eta, expected.eta)
    np.testing.assert_allclose(actual.log_lik, expected.log_lik, rtol=0, atol=2e-12)
    density_mean = np.average(jax.vmap(model.log_prob)(X), weights=weights)
    np.testing.assert_allclose(actual.log_lik, density_mean, rtol=0, atol=2e-11)


@pytest.mark.parametrize('family', ['nig', 'gh', 'factor'])
@pytest.mark.parametrize('algorithm,backend', [('em', 'jax'), ('em', 'cpu'), ('mcecm', 'cpu')])
def test_none_and_uniform_weights(family, algorithm, backend):
    X = _sample()[:16]
    fitter = BatchEMFitter(
        algorithm=algorithm, max_iter=4, tol=0, track_ll=True,
        e_step_backend=backend, m_step_backend=backend)
    unweighted = fitter.fit(_model(family), X)
    explicit_none = fitter.fit(_model(family), X, sample_weight=None)
    uniform = fitter.fit(_model(family), X, sample_weight=jnp.full(len(X), 7.0))
    _assert_tree_close(unweighted.model, explicit_none.model, rtol=0, atol=0)
    np.testing.assert_array_equal(unweighted.log_likelihoods, explicit_none.log_likelihoods)
    np.testing.assert_array_equal(unweighted.param_changes, explicit_none.param_changes)
    _assert_tree_close(unweighted.model, uniform.model)
    np.testing.assert_allclose(unweighted.log_likelihoods, uniform.log_likelihoods,
                               rtol=0, atol=2e-11)
    assert not uniform.solver_failed and not uniform.diverged


@pytest.mark.parametrize('family', ['nig', 'gh', 'factor'])
@pytest.mark.parametrize('algorithm,backend', [('em', 'jax'), ('mcecm', 'jax'), ('em', 'cpu')])
def test_weighted_iterations_match_replication(family, algorithm, backend):
    X = _sample()[:16]
    weights = np.array([0, 1, 2, 3] * 4)
    fitter = BatchEMFitter(
        algorithm=algorithm, max_iter=5, tol=0, track_ll=True,
        e_step_backend=backend, m_step_backend=backend,
        regularization='det_sigma_x')
    actual = fitter.fit(_model(family), X, sample_weight=weights)
    expected = fitter.fit(_model(family), jnp.repeat(X, weights, axis=0))
    assert not actual.solver_failed and not actual.diverged
    _assert_tree_close(actual.model, expected.model, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual.log_likelihoods, expected.log_likelihoods,
                               rtol=0, atol=2e-10)


@pytest.mark.parametrize('family', ['nig', 'gh', 'factor'])
def test_weighted_fixed_point_matches_replication(family):
    X = _sample()
    weights = np.tile([1, 2, 3], len(X) // 3)
    fitter = BatchEMFitter(
        max_iter=600, tol=1e-6, track_ll=True,
        e_step_backend='cpu', m_step_backend='cpu', regularization='a_eq_b')
    actual = fitter.fit(_model(family), X, sample_weight=weights)
    expected = fitter.fit(_model(family), jnp.repeat(X, weights, axis=0))
    assert actual.converged and expected.converged
    assert not actual.solver_failed and not actual.diverged
    _assert_tree_close(actual.model, expected.model, rtol=2e-7, atol=2e-8)
    actual_ll = actual.model.marginal_log_likelihood(X, sample_weight=weights)
    expected_ll = expected.model.marginal_log_likelihood(jnp.repeat(X, weights, axis=0))
    np.testing.assert_allclose(actual_ll, expected_ll, rtol=0, atol=2e-10)


@pytest.mark.parametrize('backend', ['jax', 'cpu'])
def test_ewma_converges_with_nondecreasing_weighted_likelihood(backend):
    X = _sample()
    weights = 0.98 ** np.arange(len(X) - 1, -1, -1)
    model = NormalInverseGaussian.default_init(X, sample_weight=weights)
    result = model.fit(
        X, sample_weight=weights, max_iter=400, tol=1e-5, track_ll=True,
        e_step_backend=backend, m_step_backend=backend)
    assert result.converged
    assert not result.solver_failed and not result.diverged
    assert np.min(np.diff(result.log_likelihoods)) >= -2e-11
    final_ll = result.model.marginal_log_likelihood(X, sample_weight=weights)
    assert final_ll >= result.log_likelihoods[-1] - 2e-11


@pytest.mark.parametrize('cls', [NormalInverseGaussian, FactorNormalInverseGaussian,
                               GeneralizedHyperbolic, FactorGeneralizedHyperbolic])
def test_weighted_default_init_matches_replication(cls):
    X = _sample()[:16]
    weights = np.array([0, 1, 2, 3] * 4)
    actual = cls.default_init(X, sample_weight=weights)
    expected = cls.default_init(jnp.repeat(X, weights, axis=0))
    _assert_tree_close(actual, expected, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize('cls', [Gamma, MultivariateNormal])
@pytest.mark.parametrize('method', ['fit', 'fit_mle', 'default_init'])
def test_weighted_exponential_family_fit(cls, method):
    X = jnp.array([0.4, 0.8, 1.3, 2.5, 4.0]) if cls is Gamma else _sample()[:5]
    weights = np.array([0, 1, 3, 2, 1])
    owner = cls.default_init(X) if method == 'fit' else cls
    actual = getattr(owner, method)(X, sample_weight=weights)
    expected = getattr(owner, method)(jnp.repeat(X, weights, axis=0))
    _assert_tree_close(actual, expected)


@pytest.mark.parametrize('weights', [
    [-1, 1, 2], [0, 0, 0], [1, np.nan, 2], [1, np.inf, 2],
    [1, 2], [[1], [2], [3]], 1.0,
])
@pytest.mark.parametrize('entry', ['fitter', 'wrapper', 'estep', 'init', 'mle'])
def test_invalid_weights_raise(weights, entry):
    X = _sample()[:3]
    model = _model('nig')
    with pytest.raises(ValueError, match='sample_weight'):
        if entry == 'fitter':
            BatchEMFitter(max_iter=0).fit(model, X, sample_weight=weights)
        elif entry == 'wrapper':
            model.fit(X, sample_weight=weights, max_iter=0)
        elif entry == 'estep':
            model.e_step(X, sample_weight=weights)
        elif entry == 'init':
            NormalInverseGaussian.default_init(X, sample_weight=weights)
        else:
            Gamma.fit_mle(jnp.array([1.0, 2.0, 3.0]), sample_weight=weights)


def test_weight_normalization_is_scale_invariant_and_overflow_safe():
    X = _sample()[:3]
    weights = jnp.array([1.0, 2.0, 0.0])
    expected = _normalize_sample_weight(weights, 3)
    for scale in [1e-200, 8e307]:
        scaled = weights * scale
        np.testing.assert_array_equal(_validate_fit_data(X, scaled), expected)
    jit_normalize = jax.jit(lambda w: _normalize_sample_weight(w, 3))
    np.testing.assert_allclose(jit_normalize(weights), expected, rtol=0, atol=1e-16)
    with pytest.raises(Exception, match='sample_weight'):
        jit_normalize(jnp.array([0.0, 0.0, 0.0])).block_until_ready()


@pytest.mark.parametrize('family', ['nig', 'factor'])
@pytest.mark.parametrize('algorithm', ['em', 'mcecm'])
def test_weighted_shrinkage_translation_and_effective_sample_size(family, algorithm):
    X = _sample()[:16]
    weights = np.arange(1, len(X) + 1, dtype=float)
    center = jnp.asarray(np.average(X, axis=0, weights=weights))
    model = _model(family)
    counts = []

    class RecordCount:
        def initial_state(self):
            return {}

        def __call__(self, eta_prev, eta_new, step, batch_size, state):
            counts.append(float(batch_size))
            return eta_new, state

    rule = Shrinkage(RecordCount(), model.compute_eta_from_model(), 0.2)
    fitter = BatchEMFitter(
        algorithm=algorithm, eta_update=rule, max_iter=3, tol=0,
        e_step_backend='cpu', m_step_backend='cpu', track_ll=True)
    actual = fitter.fit(model, X, sample_weight=weights)
    np.testing.assert_allclose(counts, weights.sum() ** 2 / (weights @ weights),
                               rtol=2e-15, atol=0)
    assert fitter.eta_update is rule
    centered_model = model.replace(mu=model.mu - center)
    centered_rule = Shrinkage(IdentityUpdate(), centered_model.compute_eta_from_model(), 0.2)
    expected = BatchEMFitter(
        algorithm=algorithm, eta_update=centered_rule, max_iter=3, tol=0,
        e_step_backend='cpu', m_step_backend='cpu', track_ll=True,
    ).fit(centered_model, X - center, sample_weight=weights)
    _assert_tree_close(actual.model, expected.model.replace(mu=expected.model.mu + center))
    assert not actual.solver_failed and not actual.diverged


def test_jitted_weighted_estep():
    X = _sample()[:8]
    weight = jnp.arange(1.0, 9.0)
    model = _model('factor')
    actual = jax.jit(lambda x, w: model.e_step(x, sample_weight=w))(X, weight)
    expected = model.e_step(X, sample_weight=weight)
    _assert_tree_close(actual.eta, expected.eta)
    np.testing.assert_allclose(actual.log_lik, expected.log_lik, rtol=0, atol=2e-12)


@pytest.mark.parametrize('family', ['nig', 'gh', 'factor'])
@pytest.mark.parametrize('backend', ['jax', 'cpu'])
def test_zero_weight_discards_extreme_observation(family, backend):
    X = _sample()[:8]
    padded = jnp.concatenate([X, jnp.full((1, 2), 1e200)])
    weights = jnp.concatenate([jnp.ones(len(X)), jnp.zeros(1)])
    model = _model(family)
    actual = model.e_step(padded, sample_weight=weights, backend=backend)
    expected = model.e_step(X, backend=backend)
    _assert_tree_close(actual.eta, expected.eta)
    np.testing.assert_allclose(actual.log_lik, expected.log_lik, rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        model.marginal_log_likelihood(padded, sample_weight=weights),
        model.marginal_log_likelihood(X), rtol=0, atol=2e-12)
