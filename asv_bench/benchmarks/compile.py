"""First-call JIT compile of the η → θ solvers.

Source: ``benchmarks/bench_jit_solvers.py``. ``timeraw_*`` runs a fresh
interpreter per sample so the XLA cache cannot leak from ``setup``. The
timed statement is the public ``from_expectation`` call; building η lives
in timeit's setup (not timed). ``number=1`` is enforced by ASV.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmarks import require_requested_device

_SETUP_HEAD = """
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
"""

_TIMERAW = {
    "gig": (
        """
out = GIG.from_expectation(
    eta, theta0=theta0, backend="jax", method="newton"
)
jax.block_until_ready(out.p)
""",
        _SETUP_HEAD
        + """
from normix.distributions.generalized_inverse_gaussian import (
    GeneralizedInverseGaussian as GIG,
)
gig = GIG(
    p=jnp.asarray(0.5, dtype=jnp.float64),
    a=jnp.asarray(1.0, dtype=jnp.float64),
    b=jnp.asarray(1.0, dtype=jnp.float64),
)
eta = gig.expectation_params()
theta0 = gig.natural_params()
""",
    ),
    "gamma": (
        """
out = Gamma.from_expectation(eta, backend="jax")
jax.block_until_ready(out.alpha)
""",
        _SETUP_HEAD
        + """
from normix.distributions.gamma import Gamma
g = Gamma(
    alpha=jnp.asarray(2.0, dtype=jnp.float64),
    beta=jnp.asarray(1.0, dtype=jnp.float64),
)
eta = g.expectation_params()
""",
    ),
}


class Compile:
    """Cold-start JAX Newton / digamma compile, GIG vs Gamma."""

    params = [list(_TIMERAW)]
    param_names = ["dist"]
    timeout = 180.0
    warmup_time = 0.0
    rounds = 1
    repeat = 3
    number = 1

    def setup(self, dist: str) -> None:
        require_requested_device()

    def timeraw_from_expectation(self, dist: str) -> tuple[str, str]:
        stmt, setup = _TIMERAW[dist]
        return stmt, setup
