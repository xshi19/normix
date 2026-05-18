"""
normix.finance — finance application layer on top of normix distributions.

This subpackage is downstream of the distribution and mixture layers.
The core abstraction is :class:`PortfolioProjection`: given a multivariate
normal-mixture model and a weight vector :math:`w`, build the univariate
normal-mixture representation of the portfolio return :math:`w^\\top X`.
Risk measures (:class:`CVaR`) then act on the projection.

See ``docs/plans/finance_architecture.md`` for the design and
``docs/theory/cvar_derivatives.rst`` for the CVaR derivative formulas.
"""
from __future__ import annotations

from normix.finance.projection import PortfolioProjection, project_portfolio
from normix.finance.risk import RiskMeasure, CVaR

__all__ = [
    "PortfolioProjection",
    "project_portfolio",
    "RiskMeasure",
    "CVaR",
]
