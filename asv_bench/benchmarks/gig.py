"""Warm-start GIG ``η → θ`` timing (EM M-step hot path).

Source: ``benchmarks/bench_gig_solvers.py``. Easy = symmetric GIG;
hard = ``a ≫ b`` (the ill-conditioned case η-rescaling targets).
``theta0`` is passed so the JAX path hits ``_gig_jax_newton_jit`` rather
than the cold-start multi-start CPU solver.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from normix.distributions.generalized_inverse_gaussian import (
    GeneralizedInverseGaussian as GIG,
)

_CASES = {
    "easy": (0.5, 1.0, 1.0),
    "hard": (0.5, 10.0, 0.1),
}


class GIGFromExpectation:
    """``GIG.from_expectation`` with a warm start, jax vs cpu."""

    params = [["jax", "cpu"], list(_CASES)]
    param_names = ["backend", "case"]
    timeout = 180.0
    warmup_time = 0.0

    def setup(self, backend: str, case: str) -> None:
        p, a, b = _CASES[case]
        gig = GIG(
            p=jnp.asarray(p, dtype=jnp.float64),
            a=jnp.asarray(a, dtype=jnp.float64),
            b=jnp.asarray(b, dtype=jnp.float64),
        )
        self.eta = gig.expectation_params()
        self.theta0 = gig.natural_params()
        warm = GIG.from_expectation(
            self.eta, theta0=self.theta0, backend=backend
        )
        jax.block_until_ready(warm.p)

    def time_from_expectation(self, backend: str, case: str) -> None:
        gig = GIG.from_expectation(
            self.eta, theta0=self.theta0, backend=backend
        )
        jax.block_until_ready(gig.p)
