r"""
Risk measures as JIT-able functions of portfolio weights.

:class:`WeightFunctional` bundles a :class:`~normix.finance.risk.RiskMeasure`,
a multivariate normal-mixture model, and a fixed subordinator sample ``Y`` into
a callable ``w -> ℝ`` with gradient and Hessian companions for optimisation.
"""
from __future__ import annotations

import equinox as eqx
import jax

from normix.finance.risk import RiskMeasure
from normix.mixtures.marginal import NormalMixture


class WeightFunctional(eqx.Module):
    r"""Risk measure as a function of weights, with frozen model and ``Y``.

    Bundles a :class:`RiskMeasure`, a :class:`NormalMixture` model, and a
    realisation ``Y`` of the subordinator into JIT-able callables intended
    for mean-risk optimisation (Phase E).
    """

    risk: RiskMeasure
    model: NormalMixture
    Y: jax.Array

    @eqx.filter_jit
    def __call__(self, w: jax.Array) -> jax.Array:
        return self.risk.value_w(self.model, w, self.Y)

    @eqx.filter_jit
    def grad(self, w: jax.Array) -> jax.Array:
        return self.risk.gradient_w(self.model, w, self.Y)

    @eqx.filter_jit
    def hess(self, w: jax.Array) -> jax.Array:
        return self.risk.hessian_w(self.model, w, self.Y)
