"""
normix.finance — finance application layer on top of normix distributions.

This subpackage is downstream of the distribution and mixture layers.
Given a multivariate normal-mixture model and weights :math:`w`,
:func:`project_portfolio` returns the ``Univariate*`` distribution of
:math:`w^\\top X`. Risk measures (:class:`CVaR`) act on that univariate
object; :class:`WeightFunctional` exposes risk as a JIT-able function of
:math:`w` for optimisation. :class:`MeanRiskProblem` solves the
reduced mean-risk surface; :class:`TransactionCostProblem` builds the
local-quadratic turnover QP.

See ``docs/theory/cvar_derivatives.md``,
``docs/theory/mean_risk_optimization.md``, and
``docs/theory/transaction_costs.md``.
"""
from __future__ import annotations

from normix.finance.functional import WeightFunctional
from normix.finance.optimization import (
    EfficientFrontier, EfficientSurface, MeanRiskProblem)
from normix.finance.projection import project_portfolio
from normix.finance.risk import RiskMeasure, CVaR
from normix.finance.transaction_costs import (
    QuadraticApproximation,
    TransactionCostProblem,
    TransactionCostQP,
    TransactionCostResult,
)

__all__ = [
    "project_portfolio",
    "RiskMeasure",
    "CVaR",
    "WeightFunctional",
    "MeanRiskProblem",
    "EfficientSurface",
    "EfficientFrontier",
    "QuadraticApproximation",
    "TransactionCostProblem",
    "TransactionCostQP",
    "TransactionCostResult",
]
