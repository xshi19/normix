r"""
Local-quadratic transaction-cost portfolio rebalancing.

The mean-risk problem with an :math:`\ell_1` turnover penalty

.. math::

    \max_w \; w^\top m - c_1\, r(w) - c_2 \|w - w_0\|_1
    \quad\text{s.t.}\quad w^\top e = 1,\; A w \le b

cannot use the two-dimensional efficient-surface reduction, because the
turnover term breaks translation invariance in the reduced coordinates.
When costs keep the solution near the current portfolio :math:`w_0`, a
second-order Taylor expansion of the convex risk :math:`r` yields a
convex quadratic program in buy/sell variables
:math:`v = (v^+; v^-)` with :math:`w = w_0 + v^+ - v^-`
(see :doc:`../theory/transaction_costs`).

This module builds the QP matrices from any :class:`~normix.finance.risk.RiskMeasure`
that supplies :meth:`~normix.finance.risk.CVaR.gradient_w` /
:meth:`~normix.finance.risk.CVaR.hessian_w`. Solving is optional and uses
``scipy.optimize`` (already a core dependency); heavier QP backends can
consume the same matrices later.
"""
from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import LinearConstraint, minimize

from normix.finance.risk import RiskMeasure
from normix.mixtures.marginal import NormalMixture
from normix.utils.constants import HESSIAN_DAMPING


class QuadraticApproximation(eqx.Module):
    r"""Local Taylor model of :math:`r` at the current portfolio :math:`w_0`.

    .. math::

        r(w) \approx r(w_0) + (w - w_0)^\top \nabla r(w_0)
        + \tfrac12 (w - w_0)^\top H_r(w_0)\,(w - w_0).
    """

    w0: Array
    value: Array
    gradient: Array
    hessian: Array


class TransactionCostQP(eqx.Module):
    r"""Buy/sell quadratic program for the local transaction-cost problem.

    Maximises :math:`v^\top \tilde m - (c_1/2)\, v^\top \tilde H\, v` over
    :math:`v \ge 0` subject to :math:`v^\top \tilde e = 0` and optional
    :math:`\tilde A v \le \tilde b`. Reconstruct
    :math:`w^* = w_0 + (I\;-I)\,v^*`.

    Attributes
    ----------
    m_tilde :
        Linear term :math:`\tilde m \in \mathbb{R}^{2d}`.
    H_tilde :
        Block Hessian :math:`\tilde H` (optionally Tikhonov-regularised).
    e_tilde :
        Budget dual vector :math:`(e; -e)`.
    A_tilde, b_tilde :
        Optional inequality block; ``None`` when only the budget constraint
        is active.
    w0, c1, c2 :
        Anchor portfolio and objective coefficients.
    approx :
        Underlying risk Taylor model (for objective bookkeeping).
    m :
        Expected-return vector used to build :math:`\tilde m`.
    """

    m_tilde: Array
    H_tilde: Array
    e_tilde: Array
    A_tilde: Array | None
    b_tilde: Array | None
    w0: Array
    c1: Array
    c2: Array
    approx: QuadraticApproximation
    m: Array

    @property
    def d(self) -> int:
        return int(self.w0.shape[0])

    def weights_from_v(self, v: Array) -> Array:
        r"""Map buy/sell variables to portfolio weights :math:`w_0 + v^+ - v^-`."""
        v = jnp.asarray(v, dtype=jnp.float64)
        d = self.d
        return self.w0 + v[:d] - v[d:]

    def approx_objective(self, w: Array) -> Array:
        r"""Local quadratic objective (constants retained) at weights ``w``."""
        w = jnp.asarray(w, dtype=jnp.float64)
        dw = w - self.w0
        r_approx = (
            self.approx.value
            + dw @ self.approx.gradient
            + 0.5 * dw @ (self.approx.hessian @ dw)
        )
        return w @ self.m - self.c1 * r_approx - self.c2 * jnp.sum(jnp.abs(dw))

    def hold_objective(self) -> Array:
        r"""Objective of holding :math:`w_0` (zero turnover)."""
        return self.w0 @ self.m - self.c1 * self.approx.value


class TransactionCostResult(eqx.Module):
    r"""Solution of the local transaction-cost QP.

    ``improved`` is ``True`` when the approximate objective at ``weights``
    exceeds the hold objective; otherwise the theory recommends keeping
    :math:`w_0` (see the note in :doc:`../theory/transaction_costs`).
    """

    weights: Array
    v: Array
    turnover: Array
    approx_objective: Array
    hold_objective: Array
    improved: Array
    qp: TransactionCostQP


def build_quadratic_approximation(
    risk: RiskMeasure,
    model: NormalMixture,
    w0: Array,
    Y: Array,
) -> QuadraticApproximation:
    r"""Evaluate :math:`r(w_0)`, :math:`\nabla r(w_0)`, and :math:`H_r(w_0)`."""
    w0 = jnp.asarray(w0, dtype=jnp.float64)
    return QuadraticApproximation(
        w0=w0,
        value=risk.value_w(model, w0, Y),
        gradient=risk.gradient_w(model, w0, Y),
        hessian=risk.hessian_w(model, w0, Y),
    )


def build_transaction_cost_qp(
    approx: QuadraticApproximation,
    m: Array,
    c1: float | Array,
    c2: float | Array,
    A: Array | None = None,
    b: Array | None = None,
    hess_reg: float = HESSIAN_DAMPING,
) -> TransactionCostQP:
    r"""Assemble the buy/sell QP matrices from a risk Taylor model.

    Parameters
    ----------
    approx :
        Local approximation of :math:`r` at :math:`w_0`.
    m :
        Expected-return vector (:math:`E[X]` for a normal mixture).
    c1, c2 :
        Risk and turnover coefficients.
    A, b :
        Optional inequality :math:`A w \le b` (same :math:`w_0` must be
        feasible). Pass ``None`` for the budget-only problem.
    hess_reg :
        Tikhonov damping added to :math:`\tilde H` so the QP is strictly
        convex (:math:`\tilde H` has a nontrivial nullspace along
        :math:`v^+ = v^-`).
    """
    w0 = approx.w0
    m = jnp.asarray(m, dtype=jnp.float64)
    g = approx.gradient
    H = approx.hessian
    c1_arr = jnp.asarray(c1, dtype=jnp.float64)
    c2_arr = jnp.asarray(c2, dtype=jnp.float64)
    e = jnp.ones_like(w0)

    lin = m - c1_arr * g
    m_tilde = jnp.concatenate([lin - c2_arr * e, -lin - c2_arr * e])
    H_tilde = jnp.block([[H, -H], [-H, H]])
    if hess_reg > 0.0:
        H_tilde = H_tilde + hess_reg * jnp.eye(H_tilde.shape[0], dtype=H_tilde.dtype)
    e_tilde = jnp.concatenate([e, -e])

    A_tilde = b_tilde = None
    if A is not None:
        if b is None:
            raise ValueError("b must be provided when A is given")
        A = jnp.asarray(A, dtype=jnp.float64)
        b = jnp.asarray(b, dtype=jnp.float64)
        A_tilde = jnp.concatenate([A, -A], axis=1)
        b_tilde = b - A @ w0

    return TransactionCostQP(
        m_tilde=m_tilde,
        H_tilde=H_tilde,
        e_tilde=e_tilde,
        A_tilde=A_tilde,
        b_tilde=b_tilde,
        w0=w0,
        c1=c1_arr,
        c2=c2_arr,
        approx=approx,
        m=m,
    )


def solve_transaction_cost_qp(
    qp: TransactionCostQP,
    *,
    x0: Array | None = None,
    options: dict[str, Any] | None = None,
) -> TransactionCostResult:
    r"""Solve the local QP with ``scipy.optimize.minimize`` (SLSQP).

    Minimises :math:`(c_1/2)\, v^\top \tilde H\, v - \tilde m^\top v` subject
    to :math:`v \ge 0`, :math:`\tilde e^\top v = 0`, and optional
    :math:`\tilde A v \le \tilde b`. If the approximate objective does not
    beat holding :math:`w_0`, returns the hold portfolio.
    """
    d = qp.d
    n = 2 * d
    m_tilde = np.asarray(qp.m_tilde, dtype=np.float64)
    H_tilde = np.asarray(qp.H_tilde, dtype=np.float64)
    e_tilde = np.asarray(qp.e_tilde, dtype=np.float64)
    c1 = float(qp.c1)

    def fun(v: np.ndarray) -> float:
        return float(0.5 * c1 * v @ H_tilde @ v - m_tilde @ v)

    def jac(v: np.ndarray) -> np.ndarray:
        return c1 * (H_tilde @ v) - m_tilde

    constraints: list[Any] = [
        LinearConstraint(e_tilde, 0.0, 0.0),
    ]
    if qp.A_tilde is not None:
        A_tilde = np.asarray(qp.A_tilde, dtype=np.float64)
        b_tilde = np.asarray(qp.b_tilde, dtype=np.float64)
        constraints.append(LinearConstraint(A_tilde, -np.inf, b_tilde))

    v0 = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
    bounds = [(0.0, None)] * n
    opt = {"ftol": 1e-12, "maxiter": 500}
    if options:
        opt.update(options)

    res = minimize(
        fun, v0, jac=jac, method="SLSQP", bounds=bounds,
        constraints=constraints, options=opt,
    )
    if not res.success:
        raise RuntimeError(f"transaction-cost QP failed: {res.message}")

    v = jnp.asarray(res.x, dtype=jnp.float64)
    # Numerical noise can leave tiny complementary buy/sell pairs.
    v = jnp.maximum(v, 0.0)
    w_star = qp.weights_from_v(v)
    hold_obj = qp.hold_objective()
    approx_obj = qp.approx_objective(w_star)
    improved = approx_obj > hold_obj
    weights = jnp.where(improved, w_star, qp.w0)
    v_out = jnp.where(improved, v, jnp.zeros_like(v))
    return TransactionCostResult(
        weights=weights,
        v=v_out,
        turnover=jnp.sum(v_out),
        approx_objective=jnp.where(improved, approx_obj, hold_obj),
        hold_objective=hold_obj,
        improved=improved,
        qp=qp,
    )


class TransactionCostProblem(eqx.Module):
    r"""Local-quadratic transaction-cost rebalancing for a normal mixture.

    Bundles a fitted :class:`~normix.mixtures.marginal.NormalMixture` and a
    :class:`~normix.finance.risk.RiskMeasure` with coefficients
    :math:`(c_1, c_2)`. The risk object is reused unchanged — only its
    weight-space gradient and Hessian at :math:`w_0` enter the QP.
    """

    model: NormalMixture
    risk: RiskMeasure
    c1: float = eqx.field(static=True)
    c2: float = eqx.field(static=True)

    def __init__(
        self,
        model: NormalMixture,
        risk: RiskMeasure,
        c1: float,
        c2: float,
    ):
        if float(c1) <= 0.0 or float(c2) <= 0.0:
            raise ValueError(f"c1 and c2 must be positive, got c1={c1}, c2={c2}")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "c1", float(c1))
        object.__setattr__(self, "c2", float(c2))

    def expected_return_vector(self) -> Array:
        r"""Asset expected returns :math:`m = E[X] = \mu + \gamma\, E[Y]`."""
        return self.model.mean()

    def approximate(self, w0: Array, Y: Array) -> QuadraticApproximation:
        """Local Taylor model of risk at ``w0`` under subordinator sample ``Y``."""
        return build_quadratic_approximation(self.risk, self.model, w0, Y)

    def build_qp(
        self,
        w0: Array,
        Y: Array,
        *,
        m: Array | None = None,
        A: Array | None = None,
        b: Array | None = None,
        hess_reg: float = HESSIAN_DAMPING,
    ) -> TransactionCostQP:
        r"""Build the buy/sell QP at ``w0``.

        ``m`` defaults to :meth:`expected_return_vector`. Optional ``A``, ``b``
        encode :math:`A w \le b` (e.g. long-only via ``A = -I``, ``b = 0``).
        """
        approx = self.approximate(w0, Y)
        if m is None:
            m = self.expected_return_vector()
        return build_transaction_cost_qp(
            approx, m, self.c1, self.c2, A=A, b=b, hess_reg=hess_reg,
        )

    def solve(
        self,
        w0: Array,
        Y: Array,
        *,
        m: Array | None = None,
        A: Array | None = None,
        b: Array | None = None,
        hess_reg: float = HESSIAN_DAMPING,
        options: dict[str, Any] | None = None,
    ) -> TransactionCostResult:
        """Build and solve the local QP; see :func:`solve_transaction_cost_qp`."""
        qp = self.build_qp(w0, Y, m=m, A=A, b=b, hess_reg=hess_reg)
        return solve_transaction_cost_qp(qp, options=options)

    def true_objective_at(
        self,
        w: Array,
        w0: Array,
        Y: Array,
        m: Array | None = None,
    ) -> Array:
        r"""Exact objective :math:`w^\top m - c_1 r(w) - c_2 \|w - w_0\|_1`."""
        w = jnp.asarray(w, dtype=jnp.float64)
        w0 = jnp.asarray(w0, dtype=jnp.float64)
        if m is None:
            m = self.expected_return_vector()
        r = self.risk.value_w(self.model, w, Y)
        return w @ m - self.c1 * r - self.c2 * jnp.sum(jnp.abs(w - w0))
