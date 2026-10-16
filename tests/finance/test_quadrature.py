"""Weighted-scenario invariants and independent quadrature accuracy checks."""
from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, norminvgauss, t

from normix import (
    Gamma, GeneralizedInverseGaussian, InverseGamma, InverseGaussian,
    NormalInverseGamma, NormalInverseGaussian, UnivariateNormalInverseGamma,
    UnivariateNormalInverseGaussian, UnivariateGeneralizedHyperbolic,
)
from normix.finance import (
    CVaR, DiscreteMeasure, GeneralizedENB, MeanRiskProblem,
    TransactionCostProblem, WeightFunctional,
)
from normix.finance._conditional import bracket_raw, cdf_raw, quantile


@pytest.mark.parametrize("Y,w", [
    ([], []), ([0.0], [1.0]), ([-1.0], [1.0]), ([np.inf], [1.0]),
    ([np.nan], [1.0]), ([1.0], [-1.0]), ([1.0], [np.nan]),
    ([1.0], [np.inf]), ([1.0], [0.0]), ([1.0, 2.0], [1.0]),
    ([[1.0]], [[1.0]]),
])
def test_invalid_measure_rejected(Y, w):
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        DiscreteMeasure.from_weights(jnp.asarray(Y), jnp.asarray(w))


def test_measure_validation_under_jit():
    build = eqx.filter_jit(DiscreteMeasure.from_weights)
    with pytest.raises(Exception, match="nonnegative"):
        jax.block_until_ready(build(jnp.ones(2), jnp.array([1.0, -1.0])))


def test_measure_log_normalization_and_large_weights():
    Q = DiscreteMeasure.from_weights(jnp.array([1.0, 3.0, 1e200]),
                                     jnp.array([1e308, 1e308, 0.0]))
    np.testing.assert_allclose(Q.p, [0.5, 0.5, 0.0], atol=2e-14)
    np.testing.assert_allclose(Q.mean(), 2.0, atol=2e-14)
    assert DiscreteMeasure.coerce(Q) is Q
    assert np.isinf(Q.tail_index)
    with pytest.raises(ValueError, match="required"):
        DiscreteMeasure.coerce(None)


def _model():
    return NormalInverseGaussian.from_classical(
        mu=jnp.array([0.01, 0.03, -0.01]),
        gamma=jnp.array([-0.2, 0.1, 0.05]),
        sigma=jnp.array([[0.7, 0.1, 0.0], [0.1, 0.4, 0.05], [0.0, 0.05, 0.2]]),
        mu_ig=1.0, lam=1.0,
    )


def test_integer_weights_equal_repeated_atoms_and_zero_mass_bracket():
    model, risk, w = _model(), CVaR(0.05), jnp.array([0.4, 0.3, 0.3])
    Y = jnp.array([0.2, 1.0, 4.0])
    Q = DiscreteMeasure.from_weights(Y, jnp.array([2.0, 3.0, 1.0]))
    repeated = jnp.array([0.2, 0.2, 1.0, 1.0, 1.0, 4.0])
    for a, b in zip(risk.value_grad_hess_w(model, w, Q),
                    risk.value_grad_hess_w(model, w, repeated)):
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-14)
    Q0 = DiscreteMeasure.from_weights(jnp.array([1.0, 1e100]), jnp.array([1.0, 0.0]))
    lo, hi = bracket_raw(0.05, 0.0, -0.5, 1.0, Q0)
    np.testing.assert_allclose([lo, hi], [-0.5 + norm.ppf(0.05)] * 2, atol=2e-14)


@pytest.mark.parametrize("alpha", [1e-5, 0.05, 0.5, 0.99])
def test_single_atom_normal_closed_form(alpha):
    Q = DiscreteMeasure.from_sample(jnp.array([2.0]))
    mu, gamma, sigma = 0.1, -0.2, 0.7
    risk = CVaR(alpha)
    expected = -mu - gamma * 2 + sigma * np.sqrt(2) * norm.pdf(norm.ppf(alpha)) / alpha
    np.testing.assert_allclose(risk.value_reduced(mu, gamma, sigma, Q), expected, rtol=2e-13)


@pytest.mark.parametrize("sub", [
    Gamma(2.0, 2.0), Gamma(0.1, 1.0), InverseGamma(3.0, 2.0),
    InverseGamma(2.0, 1.0), InverseGaussian(1.0, 1.0),
    GeneralizedInverseGaussian(1.0, 2.0, 1.0),
    GeneralizedInverseGaussian(-1.5, 1e-8, 1.0),
    GeneralizedInverseGaussian(1.0, 1e4, 1e4),
])
def test_quadrature_moments_and_jit(sub):
    Q = eqx.filter_jit(lambda d: d.quadrature())(sub)
    assert Q.Y.shape == (512,)
    assert np.all(np.diff(Q.Y) > 0)
    assert np.all(np.isfinite(Q.Y)) and np.all(Q.Y > 0)
    assert np.all(Q.p >= 0)
    np.testing.assert_allclose(Q.p.sum(), 1.0, atol=5e-15)
    for k in [0.5, 1.0]:
        np.testing.assert_allclose(Q.p @ Q.Y**k, sub.raw_moment(k), rtol=3e-11, atol=2e-14)
    for bad in [0, 7, 63]:
        with pytest.raises(ValueError, match="multiple"):
            sub.quadrature(bad)


def test_quadrature_vmap_over_parameters():
    rates = jnp.array([0.5, 1.0, 2.0])
    means = jax.jit(jax.vmap(lambda b: Gamma(2.0, b).quadrature().mean()))(rates)
    np.testing.assert_allclose(means, 2 / rates, rtol=2e-13)


def test_weight_autodiff_and_convexity():
    model, risk = _model(), CVaR(0.05)
    Q = model.subordinator().quadrature()
    u, v = jnp.array([0.4, 0.3, 0.3]), jnp.array([0.1, 0.7, 0.2])
    derivative = jax.grad(lambda w: risk.value_w(model, w, Q))(u)
    np.testing.assert_allclose(derivative, risk.gradient_w(model, u, Q), rtol=1e-11, atol=1e-12)
    H = risk.hessian_w(model, u, Q)
    assert np.linalg.eigvalsh(H).min() > -1e-12
    np.testing.assert_allclose(H @ u, 0.0, atol=2e-13)
    assert risk.value_w(model, (u + v) / 2, Q) <= (risk.value_w(model, u, Q) + risk.value_w(model, v, Q)) / 2
    eps = 1e-5
    direction = v - u
    fd = (risk.gradient_w(model, u + eps * direction, Q)
          - risk.gradient_w(model, u - eps * direction, Q)) / (2 * eps)
    np.testing.assert_allclose(H @ direction, fd, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("params,index", [
    ((2.0, 4.0, 0.0), 0), ((2.0, 4.0, 0.0), 1),
    ((-3.0, 0.0, 4.0), 0), ((1.0, 2.0, 1.0), 0),
    ((1.0, 2.0, 1.0), 2),
])
def test_subordinator_parameter_envelope(params, index):
    theta = jnp.array(params)
    risk = CVaR(0.05)
    def value(v):
        p, a, b = theta.at[index].set(v)
        return risk.value_reduced(0.0, -0.5, 1.0,
                                 GeneralizedInverseGaussian(p, a, b).quadrature())
    center = theta[index]
    eps = 1e-4
    fd = (value(center + eps) - value(center - eps)) / (2 * eps)
    np.testing.assert_allclose(jax.jit(jax.grad(value))(center), fd, rtol=1e-8, atol=1e-9)


def test_student_t_and_nig_reference():
    alpha = 0.05
    student = UnivariateNormalInverseGamma.from_classical(
        mu=0.0, gamma=0.0, sigma=1.0, alpha=3.0, beta=3.0)
    x = t.ppf(alpha, 6)
    expected = (6 + x*x) / 5 * t.pdf(x, 6) / alpha
    np.testing.assert_allclose(CVaR(alpha).value(student), expected, rtol=1e-12)
    nig = UnivariateNormalInverseGaussian.from_classical(
        mu=0.0, gamma=-0.5, sigma=1.0, mu_ig=1.0, lam=1.0)
    dist = norminvgauss(np.sqrt(1.25), -0.5)
    x = dist.ppf(alpha)
    expected = -quad(lambda z: z * dist.pdf(z), -np.inf, x, epsabs=1e-13)[0] / alpha
    np.testing.assert_allclose(CVaR(alpha).value(nig), expected, rtol=1e-12)


@pytest.mark.parametrize("shape,gamma,infinite", [(1.0, -0.5, True), (0.5, 0.0, True), (1.5, 0.0, False)])
def test_moment_domains(shape, gamma, infinite):
    model = UnivariateNormalInverseGamma.from_classical(
        mu=0.0, gamma=gamma, sigma=1.0, alpha=shape, beta=0.5)
    risk = CVaR(0.05)
    Q = model.subordinator().quadrature()
    value = risk.value(model, Q)
    assert bool(jnp.isposinf(value)) == infinite
    np.testing.assert_equal(np.asarray(value), np.asarray(risk.value_reduced(0.0, gamma, 1.0, Q)))
    if gamma == 0:
        H = risk.hessian_scalar(model, Q)
        assert jnp.isposinf(H[1, 1])
        np.testing.assert_array_equal(H[0], np.zeros(3))
        assert H[1, 2] == 0 and H[2, 2] == 0
    if shape <= 1:
        assert jnp.isneginf(risk.gradient_scalar(model, Q)[1])


@pytest.mark.contract
@pytest.mark.parametrize("shape", [0.25, 0.5])
@pytest.mark.parametrize("w,expected_grad,expected_hess", [
    ([1.0, 0.0, 0.0], [np.inf, 0.0, 0.0],
     [[0.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, np.inf]]),
    ([-1.0, 0.0, 0.0], [-np.inf, 0.0, 0.0],
     [[0.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, np.inf]]),
    ([3.0, 4.0, 0.0], [np.inf, np.inf, 0.0],
     [[np.inf, -np.inf, 0.0], [-np.inf, np.inf, 0.0], [0.0, 0.0, np.inf]]),
    ([-3.0, 4.0, 0.0], [-np.inf, np.inf, 0.0],
     [[np.inf, np.inf, 0.0], [np.inf, np.inf, 0.0], [0.0, 0.0, np.inf]]),
])
def test_divergent_zero_skew_portfolio_derivatives(shape, w, expected_grad, expected_hess):
    r"""Preserve zeros and infinity signs in limits of truncated-tail derivatives.

    With zero location and skew and :math:`\Sigma=I`, truncated CVaR is
    :math:`c\lVert w\rVert`, where :math:`c\to+\infty` for shape at most
    :math:`1/2`. Gradient signs follow the weights; Hessian off-diagonal
    signs oppose their products. Radial curvature and uncoupled entries
    stay exactly zero. These limits are not classical derivatives of an
    infinite-valued function.
    """
    model = NormalInverseGamma.from_classical(
        mu=jnp.zeros(3), gamma=jnp.zeros(3), sigma=jnp.eye(3),
        alpha=shape, beta=0.5)
    risk, w = CVaR(0.05), jnp.asarray(w)
    Q = model.subordinator().quadrature()
    assert Q.tail_index <= 0.5
    grad = risk.gradient_w(model, w, Q)
    hess = risk.hessian_w(model, w, Q)
    value, fused_grad, fused_hess = risk.value_grad_hess_w(model, w, Q)
    assert np.isposinf(value)
    assert np.isposinf(risk.value_w(model, w, Q))
    for actual, expected in [(grad, expected_grad), (hess, expected_hess),
                             (fused_grad, expected_grad), (fused_hess, expected_hess)]:
        assert not np.isnan(actual).any()
        np.testing.assert_array_equal(actual, expected)


def test_default_measure_through_finance_consumers():
    model, risk, w = _model(), CVaR(0.05), jnp.array([0.4, 0.3, 0.3])
    Q = model.subordinator().quadrature()
    problem = MeanRiskProblem(model, risk)
    wf = WeightFunctional(risk, model)
    np.testing.assert_allclose(wf(w), risk.value_w(model, w, Q), rtol=1e-13)
    mu, gamma = w @ model.mu, w @ model.gamma
    np.testing.assert_allclose(problem.risk_at(mu, gamma), risk.value_w(model, w, Q), rtol=1e-12)
    surface = problem.efficient_surface(jnp.array([mu]), jnp.array([gamma]))
    np.testing.assert_allclose(surface.risk[0, 0], wf(w), rtol=1e-12)
    returns = jnp.array([float(model.mean() @ w)])
    a = problem.efficient_frontier(returns, gamma_bounds=(-0.5, 0.5))
    b = problem.efficient_frontier(returns, Q, gamma_bounds=(-0.5, 0.5))
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-12)
    tc = TransactionCostProblem(model, risk, c1=1.0, c2=0.05)
    np.testing.assert_allclose(tc.approximate(w).value, wf(w), rtol=1e-13)
    np.testing.assert_allclose(tc.solve(w).weights, tc.solve(w, Q).weights, atol=1e-10)
    enb = GeneralizedENB(risk)
    np.testing.assert_allclose(enb.evaluate(model, w).enb, enb.evaluate(model, w, Q).enb, rtol=1e-12)
    batch = jax.vmap(lambda v: enb.evaluate(model, v, Q).enb)(jnp.stack([w, w]))
    np.testing.assert_allclose(batch, enb.evaluate(model, w, Q).enb, rtol=1e-12)


def test_step_resolution_detects_underresolved_transition():
    risk = CVaR(0.05)
    def model(sigma):
        return UnivariateNormalInverseGaussian.from_classical(
            mu=0.0, gamma=-0.5, sigma=sigma**2, mu_ig=1.0, lam=1.0)
    Q = model(1.0).subordinator().quadrature()
    assert risk.step_resolution(model(1.0), Q) > 2
    assert risk.step_resolution(model(0.003), Q) < 1
    fine = model(1.0).subordinator().quadrature(4096)
    assert risk.step_resolution(model(0.003), fine) > risk.step_resolution(model(0.003), Q)


_TABLE = json.loads((Path(__file__).parents[1] / "data/quadrature_cvar_reference.json").read_text())


@pytest.mark.slow
@pytest.mark.parametrize("row", _TABLE, ids=[f'{r["case"]}:{r["gamma"]}' for r in _TABLE])
def test_frozen_adaptive_reference(row):
    nu, A, B = row["params"]
    model = UnivariateGeneralizedHyperbolic.from_classical(
        mu=row["mu"], gamma=row["gamma"], sigma=row["sigma"]**2,
        p=nu, a=2*A, b=2*B)
    Q = GeneralizedInverseGaussian(nu, 2*A, 2*B).quadrature()
    risk = CVaR(row["alpha"])
    x = quantile(model, risk.alpha, Q)
    np.testing.assert_allclose(-x, row["var"], rtol=1e-12, atol=2e-13)
    np.testing.assert_allclose(cdf_raw(x, row["mu"], row["gamma"], row["sigma"], Q), risk.alpha, atol=2e-14)
    np.testing.assert_allclose(risk.value(model, Q), row["cvar"], rtol=1e-12, atol=2e-13)
    np.testing.assert_allclose(risk.gradient_scalar(model, Q), row["gradient"], rtol=1e-12, atol=2e-13)
    np.testing.assert_allclose(risk.hessian_scalar(model, Q), row["hessian"], rtol=1e-9, atol=2e-12)


def test_finite_lower_tail_without_mixing_mean():
    model = UnivariateNormalInverseGamma.from_classical(
        mu=0.0, gamma=0.5, sigma=1.0, alpha=0.5, beta=0.5)
    Q = model.subordinator().quadrature()
    risk = CVaR(0.05)
    x = quantile(model, risk.alpha, Q)
    np.testing.assert_allclose(cdf_raw(x, 0.0, 0.5, 1.0, Q), 0.05, atol=1e-13)
    assert np.isfinite(risk.value(model, Q))
