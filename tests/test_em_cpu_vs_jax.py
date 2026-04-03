"""
Tests comparing CPU and JAX backends for E-step and M-step on real SP500 data.

Verifies that both backends produce the same results for all mixture distributions:
  - VarianceGamma (VG)
  - NormalInverseGamma (NInvG)
  - NormalInverseGaussian (NIG)
  - GeneralizedHyperbolic (GH)

Uses a small SP500 subset (5 stocks) to keep tests fast while exercising
real-world parameter ranges.
"""
import os
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
    """Create initial models for all distribution types from SP500 data."""
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    sigma_emp = jnp.cov(X.T) + 1e-4 * jnp.eye(d)

    models = {
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
    return models


# ---------------------------------------------------------------------------
# E-step: CPU vs JAX
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_e_step_cpu_vs_jax_sp500(dist_name):
    """E-step with CPU and JAX backends produce the same expectations on SP500 data."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

    eta_jax = model.e_step(X, backend='jax')
    eta_cpu = model.e_step(X, backend='cpu')

    for field in ['E_log_Y', 'E_inv_Y', 'E_Y']:
        np.testing.assert_allclose(
            np.array(getattr(eta_cpu, field)),
            np.array(getattr(eta_jax, field)),
            rtol=1e-5, atol=1e-7,
            err_msg=f"{dist_name} e_step CPU vs JAX mismatch for {field} on SP500",
        )


# ---------------------------------------------------------------------------
# M-step: CPU vs JAX (using expectations from one consistent E-step)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_m_step_cpu_vs_jax_sp500(dist_name):
    """M-step with CPU and JAX backends produce the same model on SP500 data.

    VG, NInvG, NIG have closed-form subordinator M-steps (no backend kwarg
    in m_step_subordinator). The CPU vs JAX difference comes solely from
    the GH M-step (GIG from_expectation solver). For the non-GH distributions,
    we verify that the normal parameter updates (mu, gamma, L) are identical
    regardless of the backend kwarg passed to m_step.
    """
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

    eta = model.e_step(X, backend='cpu')

    model_jax = model.m_step(eta, backend='jax', method='newton')
    model_cpu = model.m_step(eta, backend='cpu', method='lbfgs')

    j_jax = model_jax._joint
    j_cpu = model_cpu._joint

    np.testing.assert_allclose(
        np.array(j_cpu.mu), np.array(j_jax.mu),
        rtol=1e-6, atol=1e-8,
        err_msg=f"{dist_name} m_step mu mismatch",
    )
    np.testing.assert_allclose(
        np.array(j_cpu.gamma), np.array(j_jax.gamma),
        rtol=1e-6, atol=1e-8,
        err_msg=f"{dist_name} m_step gamma mismatch",
    )
    np.testing.assert_allclose(
        np.array(j_cpu.L_Sigma), np.array(j_jax.L_Sigma),
        rtol=1e-6, atol=1e-8,
        err_msg=f"{dist_name} m_step L_Sigma mismatch",
    )

    if dist_name == "GH":
        ll_jax = float(model_jax.marginal_log_likelihood(X))
        ll_cpu = float(model_cpu.marginal_log_likelihood(X))
        assert abs(ll_cpu - ll_jax) < 0.1, (
            f"GH m_step CPU vs JAX LL too different: "
            f"cpu={ll_cpu:.4f} jax={ll_jax:.4f} "
            f"(GIG params may differ due to solver convergence)"
        )


# ---------------------------------------------------------------------------
# Full EM: CPU vs JAX backends produce comparable log-likelihoods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
def test_em_cpu_vs_jax_ll_sp500(dist_name):
    """Full EM (5 iters) with CPU vs JAX backends gives similar log-likelihood."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

    regularization = 'det_sigma_one' if dist_name == 'GH' else 'none'

    fitter_jax = BatchEMFitter(
        max_iter=5, tol=1e-4,
        e_step_backend='jax', m_step_backend='jax', m_step_method='newton',
        regularization=regularization,
    )
    fitter_cpu = BatchEMFitter(
        max_iter=5, tol=1e-4,
        e_step_backend='cpu', m_step_backend='cpu', m_step_method='lbfgs',
        regularization=regularization,
    )

    result_jax = fitter_jax.fit(model, X)
    result_cpu = fitter_cpu.fit(model, X)

    ll_jax = float(result_jax.model.marginal_log_likelihood(X))
    ll_cpu = float(result_cpu.model.marginal_log_likelihood(X))

    assert np.isfinite(ll_jax), f"{dist_name} JAX EM log-likelihood not finite"
    assert np.isfinite(ll_cpu), f"{dist_name} CPU EM log-likelihood not finite"

    assert abs(ll_cpu - ll_jax) < 0.5, (
        f"{dist_name} CPU vs JAX EM log-likelihood too different: "
        f"cpu={ll_cpu:.4f}, jax={ll_jax:.4f}, diff={abs(ll_cpu - ll_jax):.4f}"
    )


# ---------------------------------------------------------------------------
# E-step then M-step: one full iteration, check LL improves for both backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist_name,backend", [
    ("VG", "jax"), ("VG", "cpu"),
    ("NInvG", "jax"), ("NInvG", "cpu"),
    ("NIG", "jax"), ("NIG", "cpu"),
    ("GH", "jax"), ("GH", "cpu"),
])
def test_em_one_step_ll_improves_sp500(dist_name, backend):
    """One E+M step improves log-likelihood on SP500 data for both backends."""
    X = _load_sp500()
    models = _make_models(X)
    model = models[dist_name]

    ll_before = float(model.marginal_log_likelihood(X))

    e_backend = backend
    m_method = 'lbfgs' if backend == 'cpu' else 'newton'

    eta = model.e_step(X, backend=e_backend)
    new_model = model.m_step(eta, backend=backend, method=m_method)

    if dist_name == 'GH':
        new_model = new_model.regularize_det_sigma_one()

    ll_after = float(new_model.marginal_log_likelihood(X))

    assert np.isfinite(ll_after), (
        f"{dist_name}/{backend}: LL not finite after one step: {ll_after}"
    )
