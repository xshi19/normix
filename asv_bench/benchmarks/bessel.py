"""Steady-state ``log_kv`` timing by Bessel regime.

Source: ``benchmarks/bench_bessel.py``, shrunk to synthetic scalars / a
small batch. ``(v, z)`` points are chosen so ``lax.cond`` actually takes
the named branch (the deep-dive script's "small-z" case at ``z=1e-3``
falls through to quadrature).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmarks import require_requested_device
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from normix.utils.bessel import log_kv

# (v, z_scalar, z_batch_lo, z_batch_hi) — batch range stays in-regime.
_REGIMES = {
    "smallz": (2.0, 1e-7, 1e-8, 1e-7),
    "mid": (1.0, 2.0, 0.5, 5.0),
    "hankel": (1.0, 30.0, 30.0, 50.0),
}

_BATCH = 128


def _log_kv(v, z):
    return log_kv(v, z)


class Bessel:
    """``log_kv`` and ``grad log_kv`` at one point per regime, plus a small batch."""

    params = [list(_REGIMES)]
    param_names = ["regime"]
    timeout = 120.0
    warmup_time = 0.0

    def setup(self, regime: str) -> None:
        require_requested_device()
        v, z, z_lo, z_hi = _REGIMES[regime]
        self.v = jnp.asarray(v, dtype=jnp.float64)
        self.z = jnp.asarray(z, dtype=jnp.float64)
        self.v_batch = jnp.full(_BATCH, v, dtype=jnp.float64)
        self.z_batch = jnp.linspace(z_lo, z_hi, _BATCH, dtype=jnp.float64)

        self._eval = jax.jit(_log_kv)
        self._eval_batch = jax.jit(jax.vmap(_log_kv))
        self._grad = jax.jit(jax.grad(_log_kv, argnums=(0, 1)))

        self._eval(self.v, self.z).block_until_ready()
        self._eval_batch(self.v_batch, self.z_batch).block_until_ready()
        gv, gz = self._grad(self.v, self.z)
        gv.block_until_ready()
        gz.block_until_ready()

    def time_log_kv_scalar(self, regime: str) -> None:
        self._eval(self.v, self.z).block_until_ready()

    def time_log_kv_batch(self, regime: str) -> None:
        self._eval_batch(self.v_batch, self.z_batch).block_until_ready()

    def time_grad_log_kv(self, regime: str) -> None:
        gv, gz = self._grad(self.v, self.z)
        gv.block_until_ready()
        gz.block_until_ready()
