"""
Tests for the MCECM algorithm.

Verifies that MCECM converges to the same MLE as EM for all mixture
distributions (VG, NInvG, NIG, GH) on SP500 data.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.variance_gamma import VarianceGamma
from normix.fitting.em import BatchEMFitter


def _make_models(X):
    d = X.shape[1]
    mu = jnp.mean(X, axis=0)
    sigma_emp = jnp.cov(X.T) + 1e-4 * jnp.eye(d)

    return {
        "VG": VarianceGamma.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma_emp,
            alpha=2.0, beta=1.0,
        ),
        "NInvG": NormalInverseGamma.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma_emp,
            alpha=3.0, beta=1.0,
        ),
        "NIG": NormalInverseGaussian.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma_emp,
            mu_ig=1.0, lam=1.0,
        ),
        "GH": GeneralizedHyperbolic.from_classical(
            mu=mu, gamma=jnp.zeros(d), sigma=sigma_emp,
            p=-0.5, a=2.0, b=1.0,
        ),
    }


# ---------------------------------------------------------------------------
# MCECM one step: LL should improve (or at least not blow up)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_mcecm_one_step_finite(dist_name, sp500_returns):
    """One MCECM iteration produces finite parameters and LL."""
    X = sp500_returns
    model = _make_models(X)[dist_name]

    regularization = 'det_sigma_one' if dist_name == 'GH' else 'none'

    fitter = BatchEMFitter(
        algorithm='mcecm', max_iter=1, tol=1e-12,
        e_step_backend='cpu', m_step_backend='cpu', m_step_method='newton',
        regularization=regularization,
    )
    result = fitter.fit(model, X)
    ll = float(result.model.marginal_log_likelihood(X))

    assert np.isfinite(ll), (
        f"{dist_name} MCECM: LL not finite after one step: {ll}"
    )


# ---------------------------------------------------------------------------
# m_step_normal: only normal params change, subordinator unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_m_step_normal_preserves_subordinator(dist_name, sp500_returns):
    """m_step_normal should not change subordinator parameters."""
    X = sp500_returns
    model = _make_models(X)[dist_name]

    eta = model.e_step(X, backend='cpu')
    model_after = model.m_step_normal(eta)

    j_before = model._joint
    j_after = model_after._joint

    np.testing.assert_array_equal(
        np.array(j_after.mu) != np.array(j_before.mu),
        [True] * j_before.d,
        err_msg=f"{dist_name}: mu should change after m_step_normal",
    )

    before_leaves = jax.tree.leaves(j_before)
    after_leaves = jax.tree.leaves(j_after)
    shared = {id(l) for l in [j_before.mu, j_before.gamma, j_before.L_Sigma]}

    for lb, la in zip(before_leaves, after_leaves):
        if id(lb) in shared:
            continue
        np.testing.assert_array_equal(
            np.array(la), np.array(lb),
            err_msg=f"{dist_name}: subordinator param changed after m_step_normal",
        )


# ---------------------------------------------------------------------------
# Full m_step == m_step_normal + m_step_subordinator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG"])
def test_full_m_step_matches_split(dist_name, sp500_returns):
    """m_step should produce the same result as m_step_normal + m_step_subordinator."""
    X = sp500_returns
    model = _make_models(X)[dist_name]

    eta = model.e_step(X, backend='cpu')

    model_full = model.m_step(eta)

    model_split = model.m_step_normal(eta)
    model_split = model_split.m_step_subordinator(eta)

    j_full = model_full._joint
    j_split = model_split._joint

    np.testing.assert_allclose(
        np.array(j_split.mu), np.array(j_full.mu),
        rtol=1e-10, atol=1e-12,
        err_msg=f"{dist_name}: split vs full mu mismatch",
    )
    np.testing.assert_allclose(
        np.array(j_split.gamma), np.array(j_full.gamma),
        rtol=1e-10, atol=1e-12,
        err_msg=f"{dist_name}: split vs full gamma mismatch",
    )
    np.testing.assert_allclose(
        np.array(j_split.L_Sigma), np.array(j_full.L_Sigma),
        rtol=1e-10, atol=1e-12,
        err_msg=f"{dist_name}: split vs full L_Sigma mismatch",
    )


# ---------------------------------------------------------------------------
# MCECM via .fit() interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_mcecm_via_fit_interface(dist_name, sp500_returns):
    """model.fit(algorithm='mcecm') should work for all distributions."""
    X = sp500_returns
    model = _make_models(X)[dist_name]

    regularization = 'det_sigma_one' if dist_name == 'GH' else 'none'

    result = model.fit(
        X, algorithm='mcecm', max_iter=3, tol=1e-6,
        e_step_backend='cpu', m_step_backend='cpu',
        regularization=regularization,
    )

    assert result.n_iter >= 1
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"{dist_name} MCECM .fit() LL not finite"


# ---------------------------------------------------------------------------
# MCECM ≡ EM at the MLE (T5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG"])
def test_mcecm_matches_em_at_mle(dist_name, sp500_returns):
    """From the same init, MCECM and EM converge to the same LL and params."""
    X = sp500_returns
    init = _make_models(X)[dist_name]
    kwargs = dict(
        max_iter=40, tol=1e-6, verbose=0,
        e_step_backend='cpu', m_step_backend='cpu',
    )

    em = BatchEMFitter(algorithm='em', **kwargs).fit(init, X)
    mcecm = BatchEMFitter(algorithm='mcecm', **kwargs).fit(init, X)

    ll_em = float(em.model.marginal_log_likelihood(X))
    ll_mcecm = float(mcecm.model.marginal_log_likelihood(X))
    np.testing.assert_allclose(ll_mcecm, ll_em, rtol=1e-4, atol=1e-4)

    np.testing.assert_allclose(
        np.array(mcecm.model.mu), np.array(em.model.mu), rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(
        np.array(mcecm.model.gamma), np.array(em.model.gamma),
        rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Invalid algorithm raises
# ---------------------------------------------------------------------------

def test_invalid_algorithm_raises():
    """BatchEMFitter should reject unknown algorithm names."""
    with pytest.raises(ValueError, match="algorithm"):
        BatchEMFitter(algorithm='bogus')
