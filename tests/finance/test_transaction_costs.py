r"""
Tests for ``normix.finance.transaction_costs`` (Phase E: transaction costs).

Cover the local-quadratic / buy-sell QP reduction:

- QP matrix blocks match the theory identities;
- ``weights_from_v`` reconstructs :math:`w = w_0 + v^+ - v^-` and preserves
  the budget;
- large turnover penalty keeps the solution at :math:`w_0`;
- a rebalancing step with moderate costs improves the approximate objective
  and stays near :math:`w_0`;
- long-only inequalities are respected;
- the exact Monte Carlo objective is consistent with the local model at
  :math:`w_0`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import NormalInverseGaussian
from normix.finance import CVaR, TransactionCostProblem
from normix.finance.transaction_costs import (
    build_quadratic_approximation,
    build_transaction_cost_qp,
    solve_transaction_cost_qp,
)


def _model(d: int = 4):
    rng = np.random.default_rng(2)
    mu = jnp.asarray(2e-4 + 6e-4 * rng.random(d), dtype=jnp.float64)
    gamma = jnp.asarray(1e-4 + 5e-4 * rng.random(d), dtype=jnp.float64)
    a = rng.normal(size=(d, d))
    sigma = jnp.asarray(a @ a.T / d + 0.4 * np.eye(d), dtype=jnp.float64) * 3e-4
    return NormalInverseGaussian.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.5,
    )


def _equal_weight(d: int) -> jnp.ndarray:
    return jnp.full(d, 1.0 / d, dtype=jnp.float64)


def test_qp_blocks_match_theory():
    model = _model(4)
    w0 = _equal_weight(4)
    Y = model.joint.subordinator().rvs(4_000, seed=0)
    risk = CVaR(0.05)
    approx = build_quadratic_approximation(risk, model, w0, Y)
    m = model.mean()
    c1, c2 = 2.0, 1e-4
    qp = build_transaction_cost_qp(approx, m, c1, c2, hess_reg=0.0)

    g = np.asarray(approx.gradient)
    H = np.asarray(approx.hessian)
    m_np = np.asarray(m)
    e = np.ones(4)
    lin = m_np - c1 * g
    m_tilde = np.concatenate([lin - c2 * e, -lin - c2 * e])
    H_tilde = np.block([[H, -H], [-H, H]])

    np.testing.assert_allclose(qp.m_tilde, m_tilde, rtol=1e-12)
    np.testing.assert_allclose(qp.H_tilde, H_tilde, rtol=1e-12)
    np.testing.assert_allclose(qp.e_tilde, np.concatenate([e, -e]), rtol=1e-12)
    assert qp.A_tilde is None and qp.b_tilde is None


def test_weights_from_v_budget_and_reconstruction():
    model = _model(3)
    w0 = _equal_weight(3)
    Y = model.joint.subordinator().rvs(2_000, seed=1)
    prob = TransactionCostProblem(model, CVaR(0.05), c1=1.0, c2=1e-4)
    qp = prob.build_qp(w0, Y, hess_reg=1e-8)

    v = jnp.array([0.05, 0.0, 0.0, 0.0, 0.02, 0.03], dtype=jnp.float64)
    w = qp.weights_from_v(v)
    np.testing.assert_allclose(w, w0 + jnp.array([0.05, -0.02, -0.03]), rtol=1e-12)
    np.testing.assert_allclose(float(w.sum()), 1.0, atol=1e-12)


def test_large_c2_holds_current_portfolio():
    model = _model(4)
    w0 = _equal_weight(4)
    Y = model.joint.subordinator().rvs(6_000, seed=3)
    # Dominating turnover cost: any move is penalised more than risk/return gain.
    prob = TransactionCostProblem(model, CVaR(0.05), c1=1.0, c2=10.0)
    result = prob.solve(w0, Y, hess_reg=1e-6)

    np.testing.assert_allclose(result.weights, w0, atol=1e-8)
    assert float(result.turnover) < 1e-8
    assert not bool(result.improved)


def test_moderate_costs_rebalance_near_w0():
    model = _model(4)
    # Start away from equal weight so there is something to trade toward.
    w0 = jnp.array([0.55, 0.20, 0.15, 0.10], dtype=jnp.float64)
    Y = model.joint.subordinator().rvs(8_000, seed=4)
    A = -jnp.eye(4, dtype=jnp.float64)
    b = jnp.zeros(4, dtype=jnp.float64)
    # Long-only + moderate turnover cost keeps the QP in the local regime.
    prob = TransactionCostProblem(model, CVaR(0.05), c1=5.0, c2=5e-2)
    result = prob.solve(w0, Y, A=A, b=b, hess_reg=1e-6)

    np.testing.assert_allclose(float(result.weights.sum()), 1.0, atol=1e-8)
    assert np.all(np.asarray(result.weights) >= -1e-8)
    assert bool(result.improved)
    assert float(result.turnover) > 1e-3
    assert float(result.turnover) < 0.4
    assert float(jnp.max(jnp.abs(result.weights - w0))) < 0.2
    np.testing.assert_allclose(
        float(result.approx_objective),
        float(result.qp.approx_objective(result.weights)),
        rtol=1e-10,
    )


def test_long_only_constraints():
    model = _model(3)
    w0 = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64)
    Y = model.joint.subordinator().rvs(6_000, seed=5)
    A = -jnp.eye(3, dtype=jnp.float64)
    b = jnp.zeros(3, dtype=jnp.float64)
    prob = TransactionCostProblem(model, CVaR(0.05), c1=3.0, c2=1e-5)
    result = prob.solve(w0, Y, A=A, b=b, hess_reg=1e-6)

    np.testing.assert_allclose(float(result.weights.sum()), 1.0, atol=1e-8)
    assert np.all(np.asarray(result.weights) >= -1e-8)


def test_hold_objective_matches_true_at_w0():
    model = _model(3)
    w0 = _equal_weight(3)
    Y = model.joint.subordinator().rvs(5_000, seed=6)
    prob = TransactionCostProblem(model, CVaR(0.05), c1=2.0, c2=1e-4)
    qp = prob.build_qp(w0, Y)
    hold = float(qp.hold_objective())
    true_hold = float(prob.true_objective_at(w0, w0, Y))
    np.testing.assert_allclose(hold, true_hold, rtol=1e-12)


def test_reject_nonpositive_coefficients():
    model = _model(2)
    with pytest.raises(ValueError, match="c1 and c2"):
        TransactionCostProblem(model, CVaR(0.05), c1=0.0, c2=1e-4)
    with pytest.raises(ValueError, match="c1 and c2"):
        TransactionCostProblem(model, CVaR(0.05), c1=1.0, c2=-1e-4)


def test_solve_qp_direct_api():
    model = _model(3)
    w0 = _equal_weight(3)
    Y = model.joint.subordinator().rvs(4_000, seed=7)
    risk = CVaR(0.05)
    approx = build_quadratic_approximation(risk, model, w0, Y)
    qp = build_transaction_cost_qp(approx, model.mean(), 1.0, 1.0, hess_reg=1e-6)
    result = solve_transaction_cost_qp(qp)
    np.testing.assert_allclose(result.weights, w0, atol=1e-8)
