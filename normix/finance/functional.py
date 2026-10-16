r"""
Risk measures as JIT-able functions of portfolio weights.

:class:`WeightFunctional` bundles a :class:`~normix.finance.risk.RiskMeasure`,
a multivariate normal-mixture model, and a fixed subordinator measure into
a callable ``w -> ℝ`` with gradient and Hessian companions for optimisation.
"""
from __future__ import annotations

import equinox as eqx
import jax

from normix.utils.quadrature import DiscreteMeasure
from normix.finance.risk import RiskMeasure
from normix.mixtures.marginal import NormalMixture


class WeightFunctional(eqx.Module):
    r"""Risk measure as a function of weights, with frozen model and scenarios.

    Bundles a :class:`RiskMeasure`, a :class:`NormalMixture` model, and a
    probability measure of the subordinator into JIT-able callables intended
    for mean-risk optimisation (Phase E).
    """

    risk: RiskMeasure
    model: NormalMixture
    scenarios: DiscreteMeasure

    def __init__(self, risk: RiskMeasure, model: NormalMixture,
                 scenarios: DiscreteMeasure | jax.Array | None = None) -> None:
        self.risk = risk
        self.model = model
        self.scenarios = DiscreteMeasure.coerce(scenarios, model.subordinator())

    @eqx.filter_jit
    def __call__(self, w: jax.Array) -> jax.Array:
        return self.risk.value_w(self.model, w, self.scenarios)

    @eqx.filter_jit
    def grad(self, w: jax.Array) -> jax.Array:
        return self.risk.gradient_w(self.model, w, self.scenarios)

    @eqx.filter_jit
    def hess(self, w: jax.Array) -> jax.Array:
        return self.risk.hessian_w(self.model, w, self.scenarios)
