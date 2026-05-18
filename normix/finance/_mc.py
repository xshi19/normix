r"""
Shared Monte-Carlo helpers for the finance layer.

We use Rao-Blackwellization over the subordinator :math:`Y`: the conditional
distribution of :math:`X = \mu + \gamma Y + \sigma \sqrt{Y} Z` given
:math:`Y` is :math:`\mathcal{N}(\mu + \gamma Y, \sigma^2 Y)`, so the
quantile, CDF, and CVaR functionals can be evaluated by averaging closed-form
Gaussian expressions over i.i.d. samples of :math:`Y` alone. Sampling
:math:`X` directly is wasteful and noisier for the same compute.

All functions in this module accept a pre-sampled array ``Y`` so the same
random draws can be reused across :func:`value`, :func:`gradient_scalar`,
and :func:`hessian_scalar` (common random numbers).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


_TINY = 1e-300


def sample_Y(subordinator, n_mc: int, seed: int) -> jax.Array:
    """Draw ``n_mc`` i.i.d. samples of the subordinator."""
    Y = subordinator.rvs(n_mc, seed=int(seed))
    return jnp.asarray(Y, dtype=jnp.float64)


def cdf_rb(x: jax.Array, mu_p: jax.Array, gamma_p: jax.Array, sigma_p: jax.Array,
           Y: jax.Array) -> jax.Array:
    r"""Rao-Blackwellised CDF :math:`F(x) = E_Y[\Phi((x - \mu - \gamma Y)/(\sigma\sqrt{Y}))]`."""
    sqY = jnp.sqrt(Y)
    z = (x - mu_p - gamma_p * Y) / (sigma_p * sqY)
    return jnp.mean(jax.scipy.stats.norm.cdf(z))


def solve_var(mu_p: jax.Array, gamma_p: jax.Array, sigma_p: jax.Array,
              Y: jax.Array, alpha: float | jax.Array,
              n_bisect: int = 60) -> jax.Array:
    r"""Solve :math:`F(x_\alpha) = \alpha` by bisection; return :math:`\mathrm{VaR}_\alpha = -x_\alpha`.

    The bracket is centred on the analytical mean :math:`\tilde\mu + \tilde\gamma E[Y]`
    and spans :math:`\pm 30` empirical standard deviations of the portfolio
    return, which is a safe over-cover for normal-mixture tails in float64.
    """
    EY = jnp.mean(Y)
    VY = jnp.mean((Y - EY) ** 2)
    var_X = EY * sigma_p ** 2 + VY * gamma_p ** 2
    std_X = jnp.sqrt(var_X + _TINY)
    mean_X = mu_p + gamma_p * EY
    lo = mean_X - 30.0 * std_X
    hi = mean_X + 30.0 * std_X

    def body(_, bracket):
        lo_, hi_ = bracket
        mid = 0.5 * (lo_ + hi_)
        F_mid = cdf_rb(mid, mu_p, gamma_p, sigma_p, Y)
        lo_new = jnp.where(F_mid < alpha, mid, lo_)
        hi_new = jnp.where(F_mid < alpha, hi_, mid)
        return (lo_new, hi_new)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
    x_alpha = 0.5 * (lo + hi)
    return -x_alpha
