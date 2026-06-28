r"""
Tests for ``normix.finance.optimization`` (Phase E: mean-risk optimization).

Cover the reduced-coordinate reduction of the mean-risk problem:

- the minimum-dispersion ``weights`` match an independent KKT solve, sum to
  one, and realise the requested :math:`(\tilde\mu, \tilde\gamma)`;
- ``dispersion`` equals :math:`w^\top\Sigma w` and the projected variance;
- ``min_variance_point`` matches the closed-form min-variance portfolio;
- the vectorized surface risk ``risk_at`` matches the object-based
  ``CVaR.value`` and ``CVaR.value_reduced``;
- the efficient frontier minimises risk along each return-constraint line
  and its weights realise the target return;
- the reduction runs for all four normal-mixture families.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import (
    GeneralizedHyperbolic, NormalInverseGamma, NormalInverseGaussian,
    UnivariateGeneralizedHyperbolic, VarianceGamma,
)
from normix.finance import CVaR, MeanRiskProblem
from normix.finance.optimization import EfficientFrontier, EfficientSurface


def _sigma(d: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(d, d))
    return jnp.asarray(a @ a.T / d + 0.5 * np.eye(d), dtype=jnp.float64) * 3e-4


def _mu_gamma(d: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Non-collinear location/skewness vectors (so ``[mu gamma e]`` is rank 3)."""
    rng = np.random.default_rng(1)
    mu = jnp.asarray(2e-4 + 8e-4 * rng.random(d), dtype=jnp.float64)
    gamma = jnp.asarray(1e-4 + 7e-4 * rng.random(d), dtype=jnp.float64)
    return mu, gamma


def _nig_model(d: int = 5):
    mu, gamma = _mu_gamma(d)
    return NormalInverseGaussian.from_classical(
        mu=mu, gamma=gamma, sigma=_sigma(d), mu_ig=1.0, lam=1.5)


def _vg_model(d):
    mu, gamma = _mu_gamma(d)
    return VarianceGamma.from_classical(
        mu=mu, gamma=gamma, sigma=_sigma(d), alpha=2.0, beta=2.0)


def _ninvg_model(d):
    mu, gamma = _mu_gamma(d)
    return NormalInverseGamma.from_classical(
        mu=mu, gamma=gamma, sigma=_sigma(d), alpha=3.0, beta=2.0)


def _gh_model(d):
    mu, gamma = _mu_gamma(d)
    return GeneralizedHyperbolic.from_classical(
        mu=mu, gamma=gamma, sigma=_sigma(d), p=-0.5, a=1.0, b=1.0)


_FAMILIES = {"VG": _vg_model, "NIG": _nig_model, "NInvG": _ninvg_model, "GH": _gh_model}


def _kkt_min_variance(Sigma, M, c):
    """Independent equality-constrained min-variance solve (full KKT system)."""
    d, k = M.shape
    KKT = np.block([[2 * Sigma, M], [M.T, np.zeros((k, k))]])
    rhs = np.concatenate([np.zeros(d), c])
    return np.linalg.solve(KKT, rhs)[:d]


def test_weights_match_kkt_and_constraints():
    model = _nig_model(5)
    prob = MeanRiskProblem(model, CVaR(0.05))
    mu_t, gamma_t = 6e-4, 4e-4

    w = np.asarray(prob.weights(mu_t, gamma_t))
    M = np.column_stack([np.asarray(model.mu), np.asarray(model.gamma),
                         np.ones(model.d)])
    w_kkt = _kkt_min_variance(np.asarray(model.sigma()), M,
                              np.array([mu_t, gamma_t, 1.0]))

    np.testing.assert_allclose(w, w_kkt, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(float(model.mu @ w), mu_t, rtol=1e-9)
    np.testing.assert_allclose(float(model.gamma @ w), gamma_t, rtol=1e-9)


def test_dispersion_matches_quadratic_and_projection():
    model = _nig_model(5)
    prob = MeanRiskProblem(model, CVaR(0.05))
    mu_t, gamma_t = 6e-4, 4e-4

    w = prob.weights(mu_t, gamma_t)
    g = float(prob.dispersion(mu_t, gamma_t))

    np.testing.assert_allclose(g, float(w @ model.sigma() @ w), rtol=1e-9)

    proj = prob.projection_at(mu_t, gamma_t)
    np.testing.assert_allclose(float(proj._mu_scalar), mu_t, rtol=1e-9)
    np.testing.assert_allclose(float(proj._gamma_scalar), gamma_t, rtol=1e-9)
    np.testing.assert_allclose(float(proj._sigma_scalar) ** 2, g, rtol=1e-9)


def test_min_variance_point_closed_form():
    model = _nig_model(5)
    prob = MeanRiskProblem(model, CVaR(0.05))

    Sigma = np.asarray(model.sigma())
    e = np.ones(model.d)
    w_mv = np.linalg.solve(Sigma, e)
    w_mv = w_mv / (e @ w_mv)

    mu_t, gamma_t = prob.min_variance_point()
    np.testing.assert_allclose(float(mu_t), float(model.mu @ w_mv), rtol=1e-8)
    np.testing.assert_allclose(float(gamma_t), float(model.gamma @ w_mv), rtol=1e-8)
    # the min-variance weights minimise dispersion among all w^T e = 1
    np.testing.assert_allclose(
        float(prob.dispersion(mu_t, gamma_t)), float(w_mv @ Sigma @ w_mv),
        rtol=1e-8)


def test_risk_at_matches_object_and_value_reduced():
    model = _nig_model(4)
    cvar = CVaR(0.05)
    prob = MeanRiskProblem(model, cvar)
    mu_t, gamma_t = 6e-4, 4e-4
    Y = model.joint.subordinator().rvs(40_000, seed=0)

    proj = prob.projection_at(mu_t, gamma_t)
    r_object = float(cvar.value(proj, Y))                      # PINV-seeded quantile
    r_reduced = float(prob.risk_at(mu_t, gamma_t, Y))          # analytic-bracket quantile
    r_value_reduced = float(cvar.value_reduced(
        proj._mu_scalar, proj._gamma_scalar, proj._sigma_scalar, Y))

    np.testing.assert_allclose(r_reduced, r_object, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(r_reduced, r_value_reduced, rtol=1e-10)


def test_efficient_surface_shape_and_finite():
    model = _nig_model(5)
    prob = MeanRiskProblem(model, CVaR(0.05))
    Y = model.joint.subordinator().rvs(20_000, seed=1)

    mu_grid = jnp.linspace(2e-4, 1e-3, 7)
    gamma_grid = jnp.linspace(1e-4, 8e-4, 6)
    surf = prob.efficient_surface(mu_grid, gamma_grid, Y)

    assert isinstance(surf, EfficientSurface)
    assert surf.risk.shape == (7, 6)
    assert np.all(np.isfinite(np.asarray(surf.risk)))
    assert np.all(np.asarray(surf.risk) > 0.0)
    # expected_return = mu_tilde + gamma_tilde * E[Y]
    EY = float(prob.E_Y())
    MU, GA = np.meshgrid(np.asarray(mu_grid), np.asarray(gamma_grid), indexing="ij")
    np.testing.assert_allclose(np.asarray(surf.expected_return), MU + GA * EY, rtol=1e-9)


def test_efficient_frontier_minimises_and_realises_return():
    model = _nig_model(5)
    prob = MeanRiskProblem(model, CVaR(0.05))
    Y = model.joint.subordinator().rvs(30_000, seed=2)

    targets = jnp.linspace(7e-4, 1.2e-3, 5)
    front = prob.efficient_frontier(targets, Y, gamma_bounds=(-2e-3, 3e-3), n_iter=48)

    assert isinstance(front, EfficientFrontier)
    # weights realise the target expected return through the full projection
    for k in range(len(targets)):
        m_real = float(model.project(front.weights[k]).mean())
        np.testing.assert_allclose(m_real, float(targets[k]), rtol=1e-6)

    # frontier risk is the minimum along its constraint line: any other gamma
    # on the same line gives >= risk
    EY = float(prob.E_Y())
    for k in range(len(targets)):
        for gamma_alt in (-5e-4, 1.5e-3):
            mu_alt = float(targets[k]) - gamma_alt * EY
            r_alt = float(prob.risk_at(mu_alt, gamma_alt, Y))
            assert r_alt >= float(front.risk[k]) - 1e-7


@pytest.mark.parametrize("name", list(_FAMILIES))
def test_reduction_runs_for_all_families(name):
    model = _FAMILIES[name](5)
    prob = MeanRiskProblem(model, CVaR(0.05))
    Y = model.joint.subordinator().rvs(15_000, seed=3)

    surf = prob.efficient_surface(
        jnp.linspace(3e-4, 9e-4, 5), jnp.linspace(2e-4, 7e-4, 5), Y)
    assert np.all(np.isfinite(np.asarray(surf.risk)))
    assert np.all(np.asarray(surf.risk) > 0.0)

    front = prob.efficient_frontier(
        jnp.linspace(7e-4, 1.1e-3, 4), Y, gamma_bounds=(-1e-3, 2e-3), n_iter=40)
    assert np.all(np.isfinite(np.asarray(front.risk)))
    np.testing.assert_allclose(np.asarray(front.weights).sum(axis=1),
                               np.ones(4), atol=1e-8)


# Parameters rounded from the generalized-hyperbolic fit to the S&P 500 basket
# in docs/tutorials/finance/05 (a = b gauge): the GIG(p, a, b) subordinator
# (E[Y] ≈ 0.27) and the minimum-variance reduced coordinates (μ̃, γ̃, σ̃).
_GH_SUBORDINATOR = dict(p=-1.75, a=0.5, b=0.5)
_VERTEX = dict(mu=1.1e-3, gamma=-2.0e-3, sigma=1.6e-2)


def test_cvar_monotonicity_theorem_signs():
    r"""Theorem 1/2 monotonicity signs hold for CVaR on a normal mixture.

    For a coherent risk measure, CVaR is decreasing in :math:`\tilde\mu` (slope
    exactly :math:`-1`), non-increasing in :math:`\tilde\gamma`, and
    non-decreasing in :math:`\tilde\sigma`. These three signs are what make the
    efficient surface (Fig. 8) a convex reduction of the mean-risk problem.
    Parameters are rounded from the S&P 500 GH fit used in the finance tutorial.
    """
    mt, gt, st = _VERTEX["mu"], _VERTEX["gamma"], _VERTEX["sigma"]
    uni = UnivariateGeneralizedHyperbolic.from_classical(
        mu=mt, gamma=gt, sigma=st ** 2, **_GH_SUBORDINATOR)
    cvar = CVaR(0.05)
    Y = uni.subordinator.rvs(50_000, seed=0)

    # (1) local sensitivity signs at the vertex (analytic gradient in (μ̃, γ̃, σ̃))
    d_mu, d_gamma, d_sigma = np.asarray(cvar.gradient_scalar(uni, Y))
    np.testing.assert_allclose(d_mu, -1.0, atol=1e-9)    # translation invariance
    assert d_gamma <= 1e-9                                # non-increasing in γ̃
    assert d_sigma >= -1e-9                               # non-decreasing in σ̃

    # (2) global monotonicity along each axis (low-variance reduced CMC, common Y)
    def cv(mu, gamma, sigma):
        return float(cvar.value_reduced(
            jnp.asarray(mu), jnp.asarray(gamma), jnp.asarray(sigma), Y))

    mu_sweep = [cv(m, gt, st) for m in np.linspace(mt - 1e-3, mt + 1e-3, 9)]
    ga_sweep = [cv(mt, g, st) for g in np.linspace(gt - 2e-3, gt + 2e-3, 9)]
    sg_sweep = [cv(mt, gt, s) for s in np.linspace(0.5 * st, 2.0 * st, 9)]
    assert np.all(np.diff(mu_sweep) < 0.0)               # strictly decreasing in μ̃
    assert np.all(np.diff(ga_sweep) <= 1e-12)            # non-increasing in γ̃
    assert np.all(np.diff(sg_sweep) >= -1e-12)           # non-decreasing in σ̃
