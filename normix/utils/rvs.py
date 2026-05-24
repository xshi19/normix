"""
Generic RVS utilities for univariate distributions.

:func:`build_pinv_table` builds a quantile table from any univariate
log-kernel in pure JAX (trapezoidal CDF on a :math:`w`-grid).  Distributions
supply ``log_kernel(w)`` from their own ``log_prob`` (plus a Jacobian when
working in :math:`w = \\log x`).  :func:`rvs_pinv` samples via inverse lookup.
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


def _bisect_w(
    log_kernel: Callable[[jax.Array], jax.Array],
    mode: jax.Array,
    thresh: jax.Array,
    *,
    shrink_hi: bool,
) -> jax.Array:
    """Bisect in :math:`w`-space to find where ``log_kernel`` crosses ``thresh``."""
    lo0 = jnp.where(shrink_hi, mode, mode - 200.0)
    hi0 = jnp.where(shrink_hi, mode + 200.0, mode)

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        below = log_kernel(mid) < thresh
        lo = jnp.where(below, jnp.where(shrink_hi, lo, mid), jnp.where(shrink_hi, mid, lo))
        hi = jnp.where(below, jnp.where(shrink_hi, mid, hi), jnp.where(shrink_hi, hi, mid))
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, 200, body, (lo0, hi0))
    return jnp.where(shrink_hi, hi, lo)


def build_pinv_table(
    log_kernel: Callable[[jax.Array], jax.Array],
    mode: jax.Array,
    *,
    x_of_w: Optional[Callable[[jax.Array], jax.Array]] = None,
    n_grid: int = 4000,
    tail_eps: float = 1e-14,
) -> tuple[jax.Array, jax.Array]:
    r"""Build a PINV quantile table in pure JAX.

    Parameters
    ----------
    log_kernel
        Callable ``w -> log f(w)`` for a univariate density on the
        *internal* :math:`w`-axis.  For support :math:`(0, \infty)` with
        :math:`w = \log x`, pass ``log_kernel(w) = log_prob(exp(w)) + w``.
        For support :math:`\mathbb{R}` with :math:`w = x`, pass
        ``log_kernel(w) = log_prob(w)``.
    mode
        Mode of the density on the :math:`w`-axis (starting point for tail
        bisection).
    x_of_w
        Map internal :math:`w` to the observation axis (default identity).
        Use ``jnp.exp`` when :math:`w = \log x`.
    n_grid
        Number of grid points for the trapezoidal CDF.
    tail_eps
        Tail mass below which bisection stops.

    Returns
    -------
    u_grid, x_grid
        JAX arrays of shape ``(n_grid,)`` — trapezoidal CDF values and
        corresponding :math:`x` values for ``jnp.interp`` in ``ppf`` /
        ``cdf``.
    """
    if x_of_w is None:
        x_of_w = lambda w: w

    mode = jnp.asarray(mode, dtype=jnp.float64)
    log_f_mode = log_kernel(mode)
    thresh = log_f_mode + jnp.log(jnp.asarray(tail_eps, dtype=jnp.float64))

    w_min = _bisect_w(log_kernel, mode, thresh, shrink_hi=False)
    w_max = _bisect_w(log_kernel, mode, thresh, shrink_hi=True)

    w_grid = jnp.linspace(w_min, w_max, n_grid)
    log_f = jax.vmap(log_kernel)(w_grid)
    log_f0 = log_kernel(mode)
    f = jnp.exp(log_f - log_f0)
    dw = jnp.diff(w_grid)
    avg_f = 0.5 * (f[:-1] + f[1:])
    cdf_inner = jnp.cumsum(avg_f * dw)
    cdf = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), cdf_inner])
    cdf = cdf / cdf[-1]
    cdf = cdf.at[0].set(0.0).at[-1].set(1.0)
    x_grid = jax.vmap(x_of_w)(w_grid)
    return cdf, x_grid


def rvs_pinv(
    key: jax.Array,
    u_grid: jax.Array,
    x_grid: jax.Array,
    n: int,
) -> jax.Array:
    r"""Sample *n* observations via numerical inverse CDF.

    Parameters
    ----------
    key
        JAX PRNG key.
    u_grid, x_grid
        Arrays returned by :func:`build_pinv_table`.
    n
        Sample size.

    Returns
    -------
    samples
        Array of shape ``(n,)``.
    """
    u = jax.random.uniform(key, (n,), dtype=jnp.float64)
    return jnp.interp(u, u_grid, x_grid)
