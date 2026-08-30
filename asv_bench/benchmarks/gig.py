"""Warm-start GIG ``η → θ`` timing (EM M-step hot path).

Source: ``benchmarks/bench_gig_solvers.py``. Easy = symmetric GIG;
hard = ``a ≫ b`` (the ill-conditioned case η-rescaling targets).
``theta0`` is passed so the JAX path hits ``_gig_jax_newton_jit`` rather
than the cold-start multi-start CPU solver.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmarks import require_requested_device
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from normix.distributions.generalized_inverse_gaussian import (
    GeneralizedInverseGaussian as GIG,
)

_RVS_N = 10_000
_RVS_SEED = 0
_REPEAT = (1, 3, 3.0)

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
        require_requested_device()
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


class Sampling:
    """Steady-state Devroye TDR ``GIG.rvs`` (default method), n=10_000.

    Source: ``scripts/benchmark_gig_rvs.py``. Jitted in ``setup`` so the
    timed body is the kernel, not dispatch.
    """

    timeout = 180.0
    warmup_time = 0.0
    rounds = 1
    repeat = _REPEAT

    def setup(self) -> None:
        require_requested_device()
        gig = GIG(
            p=jnp.asarray(0.5, dtype=jnp.float64),
            a=jnp.asarray(1.0, dtype=jnp.float64),
            b=jnp.asarray(1.0, dtype=jnp.float64),
        )

        def _rvs():
            return gig.rvs(_RVS_N, seed=_RVS_SEED, method="devroye")

        self._rvs = jax.jit(_rvs)
        self._rvs().block_until_ready()

    def time_gig_rvs(self) -> None:
        self._rvs().block_until_ready()
