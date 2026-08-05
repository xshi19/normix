"""
Pure-JAX inverse of the regularised incomplete gamma function.

JAX exposes :func:`jax.scipy.special.gammainc` (the regularised lower
incomplete gamma :math:`P(a, x) = \\gamma(a, x)/\\Gamma(a)`) but does not
ship its inverse.  :func:`gammaincinv` solves :math:`P(a, x) = q` by
Newton iteration with a Wilson--Hilferty starting guess — fully JIT- and
vmap-compatible.  This is the JAX analogue of
:func:`scipy.special.gammaincinv`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.utils.constants import LOG_EPS


@jax.jit
def gammaincinv(
    a: jax.Array, q: jax.Array, max_iter: int = 20, tol: float = 1e-12,
) -> jax.Array:
    r"""Solve :math:`P(a, x) = q` for :math:`x` by Newton iteration.

    JAX equivalent of :func:`scipy.special.gammaincinv`.

    Parameters
    ----------
    a
        Shape parameter, :math:`a > 0`.
    q
        Probability, :math:`q \in (0, 1)`.
    max_iter
        Maximum Newton iterations (typically converges in <10 steps).
    tol
        Absolute residual tolerance on :math:`|P(a, x) - q|`.

    Notes
    -----
    Starts from the Wilson--Hilferty cube-root normal approximation and
    iterates :math:`x \leftarrow x - (P(a, x) - q) / p(a, x)`, where
    :math:`p(a, x) = x^{a-1} e^{-x} / \Gamma(a)` is the Gamma density.
    The density is evaluated in log space to avoid overflow for large
    :math:`a`.
    """
    a_b, q_b = jnp.broadcast_arrays(
        jnp.asarray(a, dtype=jnp.float64),
        jnp.asarray(q, dtype=jnp.float64),
    )
    q_clip = jnp.clip(q_b, 1e-300, 1.0 - 1e-300)
    z = jax.scipy.special.ndtri(q_clip)
    x0 = a_b * (1.0 - 1.0 / (9.0 * a_b) + z / jnp.sqrt(9.0 * a_b)) ** 3
    x0 = jnp.maximum(x0, LOG_EPS)

    def _residual(x):
        return jax.scipy.special.gammainc(a_b, x) - q_b

    def cond(state):
        i, x, f = state
        return (i < max_iter) & (jnp.max(jnp.abs(f)) > tol)

    def body(state):
        i, x, f = state
        log_fp = (a_b - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a_b)
        x_new = jnp.maximum(x - f / jnp.exp(log_fp), LOG_EPS)
        return i + 1, x_new, _residual(x_new)

    _, x, _ = jax.lax.while_loop(cond, body, (0, x0, _residual(x0)))
    return x
