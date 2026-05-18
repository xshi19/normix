"""
Tests for `normix.finance.projection` and `normix.finance.risk`.

Cover the Phase D contract:

- ``PortfolioProjection`` matches the analytic projection of a normal
  mixture onto a weight vector.
- ``CVaR.var`` solves the Rao-Blackwellised CDF equation.
- Analytic ``CVaR.value`` matches a large-N direct-sample Monte Carlo
  estimate within Monte Carlo error.
- Analytic gradients and Hessian in :math:`(\\tilde\\mu, \\tilde\\gamma,
  \\tilde\\sigma)` match finite differences of the analytic value under
  common random numbers.
- The portfolio chain rule reproduces directional derivatives of the
  analytic value along an arbitrary direction.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.finance import CVaR, PortfolioProjection, project_portfolio
from normix.finance._mc import cdf_rb


def _model():
    mu = jnp.array([0.001, 0.0005, 0.0002])
    gamma = jnp.array([0.0008, 0.0003, 0.0001])
    sigma = jnp.array(
        [[4e-4, 5e-5, 2e-5],
         [5e-5, 3e-4, 3e-5],
         [2e-5, 3e-5, 2e-4]]
    )
    return NormalInverseGaussian.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.5,
    )


def test_projection_matches_analytic():
    model = _model()
    w = jnp.array([0.4, 0.3, 0.3])
    proj = project_portfolio(model, w)

    j = model._joint
    np.testing.assert_allclose(proj.mu_p, jnp.dot(w, j.mu), rtol=1e-12)
    np.testing.assert_allclose(proj.gamma_p, jnp.dot(w, j.gamma), rtol=1e-12)
    np.testing.assert_allclose(
        proj.sigma_p, jnp.sqrt(w @ j.sigma() @ w), rtol=1e-12,
    )

    np.testing.assert_allclose(
        proj.mean(), proj.mu_p + proj.gamma_p * proj.subordinator.mean(),
        rtol=1e-12,
    )


def test_var_inversion():
    model = _model()
    proj = project_portfolio(model, jnp.array([0.4, 0.3, 0.3]))
    cvar = CVaR(0.05)
    Y = CVaR.sample_Y(proj, n_mc=20_000, seed=0)

    v = cvar.var(proj, Y)
    F_at_neg_v = cdf_rb(-v, proj.mu_p, proj.gamma_p, proj.sigma_p, Y)
    np.testing.assert_allclose(float(F_at_neg_v), 0.05, atol=1e-8)


@pytest.mark.slow
def test_cvar_value_vs_direct_mc():
    model = _model()
    w = jnp.array([0.4, 0.3, 0.3])
    proj = project_portfolio(model, w)
    cvar = CVaR(0.05)

    Y = CVaR.sample_Y(proj, n_mc=20_000, seed=0)
    analytic = float(cvar.value(proj, Y))

    n_sim = 400_000
    X = proj.rvs(n_sim, seed=123)
    Xs = jnp.sort(X)
    k = int(0.05 * n_sim)
    simulated = float(-jnp.mean(Xs[:k]))

    se = float(jnp.std(Xs[:k]) / jnp.sqrt(k))
    assert abs(analytic - simulated) < 5.0 * se, (
        f"analytic={analytic}, simulated={simulated}, se={se}"
    )


def test_gradient_scalar_vs_fd():
    model = _model()
    proj = project_portfolio(model, jnp.array([0.4, 0.3, 0.3]))
    cvar = CVaR(0.05)
    Y = CVaR.sample_Y(proj, n_mc=20_000, seed=0)
    g = cvar.gradient_scalar(proj, Y)

    def perturb(idx, eps):
        params = [proj.mu_p, proj.gamma_p, proj.sigma_p]
        params[idx] = params[idx] + eps
        return PortfolioProjection(*params, proj.subordinator)

    eps = jnp.array([1e-5, 1e-5, 1e-6])
    g_fd = jnp.array([
        float((cvar.value(perturb(i, eps[i]), Y) - cvar.value(perturb(i, -eps[i]), Y))
              / (2 * eps[i]))
        for i in range(3)
    ])
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_fd), rtol=1e-3, atol=1e-5)


def test_hessian_scalar_vs_fd():
    model = _model()
    proj = project_portfolio(model, jnp.array([0.4, 0.3, 0.3]))
    cvar = CVaR(0.05)
    Y = CVaR.sample_Y(proj, n_mc=20_000, seed=0)
    H = cvar.hessian_scalar(proj, Y)

    np.testing.assert_array_equal(np.asarray(H[0, :]), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(H[:, 0]), np.zeros(3))

    def perturb(idx, eps):
        params = [proj.mu_p, proj.gamma_p, proj.sigma_p]
        params[idx] = params[idx] + eps
        return PortfolioProjection(*params, proj.subordinator)

    # FD column at index 1 (gamma) and 2 (sigma)
    for idx, eps in [(1, 1e-4), (2, 1e-5)]:
        col_fd = (cvar.gradient_scalar(perturb(idx, eps), Y)
                  - cvar.gradient_scalar(perturb(idx, -eps), Y)) / (2 * eps)
        np.testing.assert_allclose(
            np.asarray(H[:, idx]), np.asarray(col_fd),
            rtol=2e-3, atol=1e-4,
        )


def test_gradient_w_directional_vs_fd():
    model = _model()
    w0 = jnp.array([0.4, 0.3, 0.3])
    cvar = CVaR(0.05)
    proj = project_portfolio(model, w0)
    Y = CVaR.sample_Y(proj, n_mc=20_000, seed=0)

    gw = cvar.gradient_w(model, w0, Y)
    d = jnp.array([1.0, -0.5, -0.5])

    h = 1e-5
    fd = (cvar.value_w(model, w0 + h * d, Y) - cvar.value_w(model, w0 - h * d, Y)) / (2 * h)
    np.testing.assert_allclose(float(jnp.dot(gw, d)), float(fd), rtol=1e-3, atol=1e-6)


def test_hessian_w_directional_vs_fd():
    model = _model()
    w0 = jnp.array([0.4, 0.3, 0.3])
    cvar = CVaR(0.05)
    proj = project_portfolio(model, w0)
    Y = CVaR.sample_Y(proj, n_mc=20_000, seed=0)

    H = cvar.hessian_w(model, w0, Y)
    d = jnp.array([1.0, -0.5, -0.5])
    dHd = float(d @ H @ d)

    h = 1e-3
    v_plus = float(cvar.value_w(model, w0 + h * d, Y))
    v_mid = float(cvar.value_w(model, w0, Y))
    v_minus = float(cvar.value_w(model, w0 - h * d, Y))
    dHd_fd = (v_plus - 2 * v_mid + v_minus) / h ** 2

    np.testing.assert_allclose(dHd, dHd_fd, rtol=5e-3, atol=1e-5)
