"""
Tests for the MCECM algorithm.

Verifies that MCECM converges to the same MLE as EM for all mixture
distributions (VG, NInvG, NIG, GH) on SP500 data.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.variance_gamma import VarianceGamma
from normix.fitting.em import BatchEMFitter

DATA_PATH = Path(__file__).parent.parent / "data" / "sp500_returns.csv"
N_STOCKS = 5

_sp500_cache = None


def _load_sp500():
    global _sp500_cache
    if _sp500_cache is not None:
        return _sp500_cache
    if not DATA_PATH.exists():
        pytest.skip(f"SP500 data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna(axis=1)
    X = jnp.asarray(df.values[:, :N_STOCKS], dtype=jnp.float64)
    _sp500_cache = X
    return X


def _make_models(X):
    n, d = X.shape
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
def test_mcecm_one_step_finite(dist_name):
    """One MCECM iteration produces finite parameters and LL."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

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
# MCECM vs EM: both converge to similar log-likelihood
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_mcecm_vs_em_ll_sp500(dist_name):
    """MCECM and EM (10 iters) converge to similar log-likelihood on SP500."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

    regularization = 'det_sigma_one' if dist_name == 'GH' else 'none'

    fitter_em = BatchEMFitter(
        algorithm='em', max_iter=10, tol=1e-6,
        e_step_backend='cpu', m_step_backend='cpu', m_step_method='newton',
        regularization=regularization,
    )
    fitter_mcecm = BatchEMFitter(
        algorithm='mcecm', max_iter=10, tol=1e-6,
        e_step_backend='cpu', m_step_backend='cpu', m_step_method='newton',
        regularization=regularization,
    )

    result_em = fitter_em.fit(model, X)
    result_mcecm = fitter_mcecm.fit(model, X)

    ll_em = float(result_em.model.marginal_log_likelihood(X))
    ll_mcecm = float(result_mcecm.model.marginal_log_likelihood(X))

    assert np.isfinite(ll_em), f"{dist_name} EM LL not finite: {ll_em}"
    assert np.isfinite(ll_mcecm), f"{dist_name} MCECM LL not finite: {ll_mcecm}"

    assert abs(ll_em - ll_mcecm) < 0.5, (
        f"{dist_name} EM vs MCECM LL too different: "
        f"em={ll_em:.4f}, mcecm={ll_mcecm:.4f}"
    )


# ---------------------------------------------------------------------------
# m_step_normal: only normal params change, subordinator unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_m_step_normal_preserves_subordinator(dist_name):
    """m_step_normal should not change subordinator parameters."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

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
def test_full_m_step_matches_split(dist_name):
    """m_step should produce the same result as m_step_normal + m_step_subordinator."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

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
def test_mcecm_via_fit_interface(dist_name):
    """model.fit(algorithm='mcecm') should work for all distributions."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

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
# Invalid algorithm raises
# ---------------------------------------------------------------------------

def test_invalid_algorithm_raises():
    """BatchEMFitter should reject unknown algorithm names."""
    with pytest.raises(ValueError, match="algorithm"):
        BatchEMFitter(algorithm='bogus')
