"""Warm conditional CVaR bundle and scenario construction on CPU or GPU."""
from __future__ import annotations

from benchmarks import require_requested_device
import equinox as eqx
import jax
import jax.numpy as jnp

from normix import NormalInverseGaussian
from normix.finance import CVaR, DiscreteMeasure


class ConditionalCVaR:
    """Fixed-rule quadrature versus the existing conditional Monte Carlo path."""

    params = ["quadrature512", "MC20000"]
    param_names = ["scenarios"]
    timeout = 120.0
    warmup_time = 0.0

    def setup(self, scenarios: str) -> None:
        require_requested_device()
        self.model = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.01, 0.03, -0.01]), gamma=jnp.array([-0.2, 0.1, 0.05]),
            sigma=jnp.diag(jnp.array([0.7, 0.4, 0.2])), mu_ig=1.0, lam=1.0)
        self.w = jnp.array([0.4, 0.3, 0.3])
        self.risk = CVaR(0.05)
        self.sub = self.model.subordinator()
        self.build = eqx.filter_jit(
            (lambda sub: sub.quadrature()) if scenarios == "quadrature512"
            else (lambda sub: DiscreteMeasure.from_sample(sub.rvs(20000, seed=0))))
        self.Q = self.build(self.sub)
        jax.block_until_ready(self.risk.value_grad_hess_w(self.model, self.w, self.Q))

    def time_bundle(self, scenarios: str) -> None:
        jax.block_until_ready(self.risk.value_grad_hess_w(self.model, self.w, self.Q))

    def time_generation(self, scenarios: str) -> None:
        jax.block_until_ready(self.build(self.sub))
