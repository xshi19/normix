"""Necessary moment-domain checks at the public GIG boundary."""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import GIG, Gamma, InverseGamma


_INVALID = [
    ([0.0, -1.0, 2.0], 'E[1/X] > 0'),
    ([0.0, 0.0, 2.0], 'E[1/X] > 0'),
    ([0.0, 2.0, -1.0], 'E[X] > 0'),
    ([0.0, 2.0, 0.0], 'E[X] > 0'),
    ([0.0, 0.5, 0.5], 'E[X] * E[1/X] > 1'),
    ([-1.0, 2.0, 2.0], '-log(E[1/X]) < E[log X]'),
    ([1.0, 2.0, 2.0], 'E[log X] < log(E[X])'),
    ([np.nan, 2.0, 2.0], 'finite'),
    ([0.0, np.inf, 2.0], 'finite'),
    ([0.0, 2.0, np.inf], 'finite'),
    ([0.0, 1.0, 1.0], 'E[X] * E[1/X] > 1'),
    ([-np.log(2.0), 2.0, 2.0], '-log(E[1/X]) < E[log X]'),
    ([np.log(2.0), 2.0, 2.0], 'E[log X] < log(E[X])'),
]


@pytest.mark.contract
@pytest.mark.parametrize('eta,condition', _INVALID)
@pytest.mark.parametrize('backend', ['jax', 'cpu'])
def test_gig_invalid_expectations_raise_before_solve(eta, condition, backend, monkeypatch):
    import normix.distributions.generalized_inverse_gaussian as gig_module

    def unexpected_solve(*args, **kwargs):
        pytest.fail('invalid expectation target reached the solver')

    monkeypatch.setattr(gig_module, '_gig_jax_newton_jit', unexpected_solve)
    monkeypatch.setattr(gig_module, 'solve_bregman', unexpected_solve)
    with pytest.raises(ValueError, match=re.escape(condition)):
        GIG.from_expectation(jnp.array(eta), theta0=jnp.array([-0.5, -0.5, -0.5]), backend=backend)


@pytest.mark.contract
def test_gig_invalid_expectations_jit_vmap_return_nan():
    theta0 = GIG(0.5, 1.0, 1.0).natural_params()
    invert = lambda eta: GIG.from_expectation(eta, theta0=theta0, maxiter=30).natural_params()
    invalid = jnp.array([eta for eta, _ in _INVALID])
    actual = jax.jit(jax.vmap(invert))(invalid)
    assert np.isnan(actual).all()
    for eta in invalid:
        assert np.isnan(jax.jit(invert)(eta)).all()
    assert np.isnan(jax.jit(jax.jacrev(invert))(invalid[0])).all()


@pytest.mark.contract
@pytest.mark.parametrize('backend', ['jax', 'cpu'])
def test_gig_valid_expectations_still_invert(backend):
    model = GIG(0.5, 1.0, 1.0)
    recovered = GIG.from_expectation(model.expectation_params(),
                                    theta0=model.natural_params() * 0.9, backend=backend)
    np.testing.assert_allclose(recovered.natural_params(), model.natural_params(), rtol=1e-5, atol=1e-7)


@pytest.mark.contract
@pytest.mark.parametrize('dist', [Gamma(3.0, 2.0), InverseGamma(3.0, 2.0)])
def test_gig_nonpoint_boundary_moments_are_not_rejected(dist):
    gig = dist.to_gig()
    eta_fin, moment, direction = gig._gig_layout_eta()
    eta = eta_fin + moment * direction
    recovered = GIG.from_expectation(eta, theta0=gig.natural_params(), backend='cpu', method='lbfgs')
    np.testing.assert_allclose(recovered.natural_params(), gig.natural_params(), atol=2e-5)
