r"""
Shared Monte-Carlo helpers for the finance layer.

We use conditional Monte Carlo (Rao-Blackwellisation) over the subordinator
:math:`Y`: the conditional distribution of
:math:`X = \mu + \gamma Y + \sigma \sqrt{Y} Z` given :math:`Y` is
:math:`\mathcal{N}(\mu + \gamma Y, \sigma^2 Y)`, so quantile, CDF, and CVaR
functionals can be evaluated by averaging closed-form Gaussian expressions
over i.i.d. samples of :math:`Y` alone. Sampling :math:`X` directly is
wasteful and noisier for the same compute.

See Asmussen & Glynn (2007) §V.4 and Glasserman (2004) §4.2 for the
conditional Monte Carlo framework.

All functions in this module accept a pre-sampled array ``Y`` so the same
random draws can be reused across :func:`value`, :func:`gradient_scalar`,
and :func:`hessian_scalar` (common random numbers).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.mixtures.marginal import _UnivariateNormalMixtureMixin


def cdf_cmc_raw(
    x: jax.Array,
    mu: jax.Array,
    gamma: jax.Array,
    sigma: jax.Array,
    Y: jax.Array,
) -> jax.Array:
    r"""Conditional Monte Carlo CDF from raw scalar parameters.

    :math:`\hat F(x) = \mathbb{E}_Y[\Phi(z_Y)]` with
    :math:`z_Y = (x - \mu - \gamma Y)/(\sigma\sqrt{Y})`. Bessel- and
    PINV-free, so it is cheap to :func:`jax.vmap` across many
    :math:`(\mu, \gamma, \sigma)` triples that share the same draws ``Y``.
    """
    sqY = jnp.sqrt(Y)
    z = (x - mu - gamma * Y) / (sigma * sqY)
    return jnp.mean(jax.scipy.stats.norm.cdf(z))


def quantile_cmc_raw(
    q: float | jax.Array,
    mu: jax.Array,
    gamma: jax.Array,
    sigma: jax.Array,
    Y: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    n_bisect: int = 60,
) -> jax.Array:
    r"""Solve :math:`\hat F(x_q) = q` by bisection on the CMC CDF.

    Bisects within the explicit bracket ``[lo, hi]`` (which the caller must
    choose to contain the root). Operates on raw scalar parameters and is
    Bessel-/PINV-free, hence :func:`jax.vmap`-friendly.
    """
    def body(_, bracket):
        lo_, hi_ = bracket
        mid = 0.5 * (lo_ + hi_)
        F_mid = cdf_cmc_raw(mid, mu, gamma, sigma, Y)
        lo_new = jnp.where(F_mid < q, mid, lo_)
        hi_new = jnp.where(F_mid < q, hi_, mid)
        return (lo_new, hi_new)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (lo, hi))
    return 0.5 * (lo + hi)


def cdf_cmc(
    univariate: _UnivariateNormalMixtureMixin,
    x: jax.Array,
    Y: jax.Array,
) -> jax.Array:
    r"""Conditional Monte Carlo CDF :math:`\hat F(x) = \mathbb{E}_Y[\Phi(z_Y)]`."""
    return cdf_cmc_raw(
        x, univariate._mu_scalar, univariate._gamma_scalar,
        univariate._sigma_scalar, Y,
    )


def quantile_cmc(
    univariate: _UnivariateNormalMixtureMixin,
    q: float | jax.Array,
    Y: jax.Array,
    n_bisect: int = 60,
) -> jax.Array:
    r"""Solve :math:`\hat F(x_q) = q` by bisection on the CMC CDF.

    Returns the :math:`q`-quantile :math:`x_q` of the portfolio return.
    The bracket is centred on the PINV quantile
    :math:`F^{-1}(q)` with width :math:`\pm 5\sigma`.
    """
    x_seed = univariate.ppf(q)
    half_width = 5.0 * univariate.std()
    return quantile_cmc_raw(
        q, univariate._mu_scalar, univariate._gamma_scalar,
        univariate._sigma_scalar, Y,
        x_seed - half_width, x_seed + half_width, n_bisect,
    )
