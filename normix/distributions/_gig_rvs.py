"""
JAX-based random variate generation for the Generalized Inverse Gaussian.

Two methods, neither requires Bessel function evaluation:

1. **devroye** — Transformed density rejection (TDR) with log-concavity.
   Works in w = log(x) where the GIG log-kernel g(w) = p·w − (a·eʷ + b·e⁻ʷ)/2
   is strictly concave.  A three-piece hat (tangent lines at mode ± σ joined by
   a flat cap) gives ≈ 80–90 % acceptance.  All proposals generated in parallel
   — no ``while_loop``, GPU-friendly.

2. **pinv** — Numerical inverse CDF via the generic ``utils.rvs`` module.
   Builds the quantile function F⁻¹ on CPU using the GIG log-kernel (no
   Bessel needed), then samples via ``jnp.interp``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.rvs import build_pinv_table, rvs_pinv

jax.config.update("jax_enable_x64", True)

_TINY64 = jnp.finfo(jnp.float64).tiny

_MAX_REJECT_ROUNDS = 20


# ---------------------------------------------------------------------------
# Method 1 — Devroye (2014): TDR on the log-transformed variable
# ---------------------------------------------------------------------------

def _tdr_setup(p, a, b):
    """Compute the three-piece TDR envelope for g(w) = p·w − (a·eʷ + b·e⁻ʷ)/2.

    Returns a dict of envelope constants (all JAX scalars).
    """
    t0 = (p + jnp.sqrt(p * p + a * b)) / a
    w0 = jnp.log(t0)
    g0 = p * w0 - 0.5 * (a * t0 + b / t0)

    neg_gpp = 0.5 * (a * t0 + b / t0)
    sigma = 1.0 / jnp.sqrt(neg_gpp)

    wL, wR = w0 - sigma, w0 + sigma
    tL, tR = jnp.exp(wL), jnp.exp(wR)

    gL = p * wL - 0.5 * (a * tL + b / tL)
    gR = p * wR - 0.5 * (a * tR + b / tR)
    gpL = p - 0.5 * a * tL + 0.5 * b / tL      # g′(wL) > 0
    gpR = p - 0.5 * a * tR + 0.5 * b / tR      # g′(wR) < 0

    wsL = wL + (g0 - gL) / gpL
    wsR = wR + (g0 - gR) / gpR

    inv_lL = 1.0 / gpL
    inv_lR = 1.0 / (-gpR)
    width  = wsR - wsL
    D      = inv_lL + width + inv_lR

    return dict(
        g0=g0, wsL=wsL, wsR=wsR, gpL=gpL, gpR=gpR,
        width=width, pL=inv_lL / D, pM=width / D, lR=-gpR,
    )


def gig_rvs_devroye(key: jax.Array, p, a, b, n: int) -> jax.Array:
    """Sample *n* GIG(p, a, b) variates via transformed density rejection.

    The GIG density in w = log(x) is h(w) = exp(g(w)) where
    g(w) = p·w − (a·eʷ + b·e⁻ʷ)/2 is strictly concave.

    Envelope: three-piece hat from tangent lines at mode ± σ (flat cap
    at the mode value, exponential tails from tangent lines).
    Acceptance rate ≈ 80–90 % for typical parameters.

    All ``_MAX_REJECT_ROUNDS × n`` proposals are generated in a single
    batch — no ``while_loop`` or ``fori_loop``, fully GPU-parallel.

    Parameters
    ----------
    key : JAX PRNG key
    p, a, b : GIG parameters (scalars).  a > 0, b > 0 required.
    n : number of samples

    Returns
    -------
    jax.Array of shape (n,)
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)

    env = _tdr_setup(p, a, b)
    g0, wsL, wsR = env["g0"], env["wsL"], env["wsR"]
    gpL, gpR, width = env["gpL"], env["gpR"], env["width"]
    pL, pM, lR = env["pL"], env["pM"], env["lR"]

    M = _MAX_REJECT_ROUNDS

    # ---- generate ALL proposals at once: shape (M, n) ----
    k1, k2, k3 = jax.random.split(key, 3)
    all_up = jax.random.uniform(k1, (M, n), dtype=jnp.float64)
    all_u  = jnp.maximum(
        jax.random.uniform(k2, (M, n), dtype=jnp.float64), _TINY64)
    all_ua = jnp.maximum(
        jax.random.uniform(k3, (M, n), dtype=jnp.float64), _TINY64)

    log_u = jnp.log(all_u)

    # three-piece proposal
    w_l = wsL + log_u / gpL
    w_m = wsL + all_u * width
    w_r = wsR - log_u / lR

    h_l = g0 + gpL * (w_l - wsL)
    h_m = g0
    h_r = g0 + gpR * (w_r - wsR)

    left = all_up < pL
    mid  = (all_up >= pL) & (all_up < pL + pM)

    w = jnp.where(left, w_l, jnp.where(mid, w_m, w_r))
    h = jnp.where(left, h_l, jnp.where(mid, h_m, h_r))

    ew = jnp.exp(w)
    gw = p * w - 0.5 * (a * ew + b / ew)
    ok = jnp.log(all_ua) <= gw - h               # (M, n) bool

    # for each sample column, pick the first accepted row
    first_idx = jnp.argmax(ok, axis=0)            # (n,)
    return ew[first_idx, jnp.arange(n)]


# ---------------------------------------------------------------------------
# Method 2 — PINV wrappers for GIG
# ---------------------------------------------------------------------------

def _gig_log_kernel(w, p, a, b):
    """GIG log-kernel in w-space (numpy, vectorised)."""
    w_c = np.clip(w, -500.0, 500.0)
    ew = np.exp(w_c)
    return p * w_c - 0.5 * (a * ew + b / ew)


def _gig_mode_w(p, a, b):
    """Mode of the GIG log-kernel in w = log(x) space."""
    t0 = (p + np.sqrt(p ** 2 + a * b)) / a
    return np.log(t0)


def gig_build_pinv_table(p, a, b, **kwargs):
    """Build PINV table for GIG(p, a, b).  Delegates to ``utils.rvs``."""
    pf, af, bf = float(p), float(a), float(b)
    return build_pinv_table(
        log_kernel=lambda w: _gig_log_kernel(w, pf, af, bf),
        mode=_gig_mode_w(pf, af, bf),
        x_of_w=np.exp,
        **kwargs,
    )


def gig_rvs_pinv(key, u_grid, x_grid, n):
    """Sample *n* GIG variates from a precomputed PINV table."""
    return rvs_pinv(key, u_grid, x_grid, n)
