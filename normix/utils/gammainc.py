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
    a: jax.Array, q: jax.Array, n_iter: int = 50,
) -> jax.Array:
    r"""Solve :math:`P(a, x) = q` for :math:`x` by Newton iteration.

    JAX equivalent of :func:`scipy.special.gammaincinv`.

    Parameters
    ----------
    a
        Shape parameter, :math:`a > 0`.
    q
        Probability, :math:`q \in (0, 1)`.
    n_iter
        Newton iteration count.  The default of 50 is safe; convergence
        is typically reached in <10 steps.

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

    def body(_, x):
        f = jax.scipy.special.gammainc(a_b, x) - q_b
        log_fp = (a_b - 1.0) * jnp.log(x) - x - jax.scipy.special.gammaln(a_b)
        fp = jnp.exp(log_fp)
        return jnp.maximum(x - f / fp, LOG_EPS)

    return jax.lax.fori_loop(0, n_iter, body, x0)
