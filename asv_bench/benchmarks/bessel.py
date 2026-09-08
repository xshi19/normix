"""Steady-state ``log_kv`` / ``log_kv_moments`` timing.

Source: ``benchmarks/bench_bessel.py``. Points keep the historical ASV
param names (``smallz`` / ``mid`` / ``hankel``) so the dashboard can compare
against pre-S10 runs; all three now use the same moment-quadrature kernel.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmarks import require_requested_device
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from normix.utils.bessel import log_kv, log_kv_moments

# (v, z_scalar, z_batch_lo, z_batch_hi)
_POINTS = {
    "smallz": (2.0, 1e-7, 1e-8, 1e-7),
    "mid": (1.0, 2.0, 0.5, 5.0),
    "hankel": (1.0, 30.0, 30.0, 50.0),
}

_BATCH = 128


def _log_kv(v, z):
    return log_kv(v, z)


def _moments(v, z):
    return log_kv_moments(v, z)


class Bessel:
    """``log_kv``, moments, and Hessian at one point per domain region."""

    params = [list(_POINTS)]
    param_names = ["regime"]
    timeout = 120.0
    warmup_time = 0.0

    def setup(self, regime: str) -> None:
        require_requested_device()
        v, z, z_lo, z_hi = _POINTS[regime]
        self.v = jnp.asarray(v, dtype=jnp.float64)
        self.z = jnp.asarray(z, dtype=jnp.float64)
        self.v_batch = jnp.full(_BATCH, v, dtype=jnp.float64)
        self.z_batch = jnp.linspace(z_lo, z_hi, _BATCH, dtype=jnp.float64)

        self._eval = jax.jit(_log_kv)
        self._eval_batch = jax.jit(jax.vmap(_log_kv))
        self._grad = jax.jit(jax.grad(_log_kv, argnums=(0, 1)))
        self._moments = jax.jit(_moments)
        self._hess = jax.jit(jax.hessian(lambda vz: _log_kv(vz[0], vz[1])))

        self._eval(self.v, self.z).block_until_ready()
        self._eval_batch(self.v_batch, self.z_batch).block_until_ready()
        gv, gz = self._grad(self.v, self.z)
        gv.block_until_ready()
        gz.block_until_ready()
        m = self._moments(self.v, self.z)
        jax.block_until_ready(m.log_k)
        vz = jnp.stack([self.v, self.z])
        self._hess(vz).block_until_ready()

    def time_log_kv_scalar(self, regime: str) -> None:
        self._eval(self.v, self.z).block_until_ready()

    def time_log_kv_batch(self, regime: str) -> None:
        self._eval_batch(self.v_batch, self.z_batch).block_until_ready()

    def time_grad_log_kv(self, regime: str) -> None:
        gv, gz = self._grad(self.v, self.z)
        gv.block_until_ready()
        gz.block_until_ready()

    def time_log_kv_moments(self, regime: str) -> None:
        m = self._moments(self.v, self.z)
        jax.block_until_ready(m.log_k)

    def time_hessian_log_kv(self, regime: str) -> None:
        vz = jnp.stack([self.v, self.z])
        self._hess(vz).block_until_ready()
