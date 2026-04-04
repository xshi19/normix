"""
Generic random variate generation utilities for univariate distributions.

Currently provides **PINV** (Polynomial-Interpolation-based Numerical Inversion):
build a quantile-function table on CPU, then sample via ``jnp.interp`` on GPU.

The PINV method is distribution-agnostic — any univariate density whose
(possibly unnormalized) log-kernel can be evaluated as a Python callable
is supported.  No normalising constant is needed; the CDF is normalised
numerically.

Future methods (e.g. generic TDR for log-concave densities) may be added here.
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np



# ---------------------------------------------------------------------------
# PINV — Numerical Inverse CDF
# ---------------------------------------------------------------------------

def build_pinv_table(
    log_kernel: Callable[[np.ndarray], np.ndarray],
    mode: float,
    *,
    x_of_w: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_grid: int = 4000,
    tail_eps: float = 1e-14,
) -> tuple[jax.Array, jax.Array]:
    """Build a PINV quantile table on CPU for any univariate density.

    The density need not be normalised — the CDF is obtained by numerically
    integrating ``exp(log_kernel(w))`` over a grid in *w*-space and dividing
    by the total.  This avoids computing the normalising constant analytically
    (e.g. no Bessel function for the GIG).

    Parameters
    ----------
    log_kernel : callable  (numpy float or 1-D array) → same shape
        Log of the (possibly unnormalized) density evaluated on the
        internal grid variable *w*.  Must be bounded above and integrable.
        Will be called once with the full grid array.
    mode : float
        Approximate mode of *log_kernel* in w-space.  Used as the starting
        point for the bisection that finds the effective support.
    x_of_w : callable or None
        Maps the grid variable *w* to the sample variable *x*.
        Use ``np.exp`` for densities on (0, ∞) whose internal variable is
        w = log(x).  ``None`` ≡ identity (x = w).
    n_grid : int
        Number of equi-spaced grid points in w-space.
    tail_eps : float
        Density-ratio threshold for truncating tails:
        the grid extends to where ``log_kernel(w) < log_kernel(mode) + log(tail_eps)``.

    Returns
    -------
    (u_grid, x_grid) : pair of JAX float64 arrays, each shape ``(n_grid,)``
        ``u_grid`` are the CDF values; ``x_grid`` the corresponding sample values.
        Pass both to :func:`rvs_pinv` for sampling.
    """
    mode = float(mode)
    g0 = float(log_kernel(np.float64(mode)))
    thresh = g0 + np.log(tail_eps)

    # left boundary
    lo, hi = mode - 200.0, mode
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(log_kernel(np.float64(mid))) < thresh:
            lo = mid
        else:
            hi = mid
    w_min = lo

    # right boundary
    lo, hi = mode, mode + 200.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if float(log_kernel(np.float64(mid))) < thresh:
            hi = mid
        else:
            lo = mid
    w_max = hi

    # evaluate on grid
    w_grid = np.linspace(w_min, w_max, n_grid)
    g_vals = log_kernel(w_grid)
    h_vals = np.exp(g_vals - g0)

    # trapezoidal CDF
    dw = np.diff(w_grid)
    avg_h = 0.5 * (h_vals[:-1] + h_vals[1:])
    cdf = np.empty(n_grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(avg_h * dw)

    u_grid = cdf / cdf[-1]
    u_grid[0] = 0.0
    u_grid[-1] = 1.0

    x_grid = x_of_w(w_grid) if x_of_w is not None else w_grid

    return (
        jnp.asarray(u_grid, dtype=jnp.float64),
        jnp.asarray(x_grid, dtype=jnp.float64),
    )


def rvs_pinv(
    key: jax.Array,
    u_grid: jax.Array,
    x_grid: jax.Array,
    n: int,
) -> jax.Array:
    """Sample *n* variates from a precomputed PINV table.

    Fully vectorised — a single ``jnp.interp`` call, no control flow.
    JIT-able and GPU-friendly.

    Parameters
    ----------
    key : JAX PRNG key
    u_grid, x_grid : arrays returned by :func:`build_pinv_table`
    n : number of samples
    """
    u = jax.random.uniform(key, shape=(n,), dtype=jnp.float64)
    return jnp.interp(u, u_grid, x_grid)
