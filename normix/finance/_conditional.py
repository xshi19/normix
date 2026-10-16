r"""Conditional normal CDFs and quantiles under a fixed probability measure.

The normal part is integrated analytically; only the positive mixing
variable is discretized. See :doc:`/theory/quadrature_scenarios`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtr, ndtri

from normix.mixtures.marginal import _UnivariateNormalMixtureMixin
from normix.utils.constants import CVAR_BISECT_STEPS
from normix.utils.quadrature import DiscreteMeasure


def cdf_raw(
    x: Array, mu: Array, gamma: Array, sigma: Array,
    scenarios: DiscreteMeasure | Array,
) -> Array:
    r"""Return :math:`F_Q(x)=\sum_j p_j\Phi((x-\mu-\gamma Y_j)/(\sigma\sqrt{Y_j}))`."""
    Q = DiscreteMeasure.coerce(scenarios)
    z = (x - mu - gamma * Q.Y) / (sigma * jnp.sqrt(Q.Y))
    return Q.p @ ndtr(z)


def bracket_raw(
    q: float | Array, mu: Array, gamma: Array, sigma: Array,
    scenarios: DiscreteMeasure | Array,
) -> tuple[Array, Array]:
    r"""Bracket the quantile by the extreme positive-mass component quantiles.

    Each component CDF brackets :math:`q` on this interval, so their
    convex combination does too. Zero-mass atoms do not affect the bracket.
    """
    Q = DiscreteMeasure.coerce(scenarios)
    q_i = mu + gamma * Q.Y + sigma * jnp.sqrt(Q.Y) * ndtri(q)
    active = jnp.isfinite(Q.log_p)
    return jnp.min(jnp.where(active, q_i, jnp.inf)), jnp.max(jnp.where(active, q_i, -jnp.inf))


def quantile_raw(
    q: float | Array, mu: Array, gamma: Array, sigma: Array,
    scenarios: DiscreteMeasure | Array, lo: Array, hi: Array,
    n_bisect: int = CVAR_BISECT_STEPS,
) -> Array:
    r"""Invert the conditional CDF in :math:`x=c+h\sinh t` coordinates.

    The explicit bracket must contain the root. The center is the
    component quantile at :math:`E_Q[Y]`; the scale combines its conditional
    standard deviation and drift. For a source law without a mean, the
    weighted geometric mean supplies a finite central scale instead.
    The returned root is frozen under AD.
    """
    Q = DiscreteMeasure.coerce(scenarios)
    mean = jnp.where(Q.tail_index > 1, Q.mean(), jnp.exp(Q.p @ jnp.log(Q.Y)))
    c = mu + gamma * mean + sigma * jnp.sqrt(mean) * ndtri(q)
    h = sigma * jnp.sqrt(mean) + jnp.abs(gamma) * mean
    t_lo, t_hi = jnp.arcsinh((lo - c) / h), jnp.arcsinh((hi - c) / h)

    def body(_, bracket):
        left, right = bracket
        mid = 0.5 * (left + right)
        F = cdf_raw(c + h * jnp.sinh(mid), mu, gamma, sigma, Q)
        return jnp.where(F < q, mid, left), jnp.where(F < q, right, mid)

    left, right = jax.lax.fori_loop(0, n_bisect, body, (t_lo, t_hi))
    x = jnp.clip(c + h * jnp.sinh(0.5 * (left + right)), lo, hi)
    return jax.lax.stop_gradient(x)


def cdf(
    univariate: _UnivariateNormalMixtureMixin, x: Array,
    scenarios: DiscreteMeasure | Array,
) -> Array:
    r"""Conditional normal CDF under the supplied measure of :math:`Y`."""
    return cdf_raw(x, univariate._mu_scalar, univariate._gamma_scalar,
                   univariate._sigma_scalar, scenarios)


def quantile(
    univariate: _UnivariateNormalMixtureMixin, q: float | Array,
    scenarios: DiscreteMeasure | Array,
    n_bisect: int = CVAR_BISECT_STEPS,
) -> Array:
    """Return the quantile under the supplied measure of the mixing variable."""
    Q = DiscreteMeasure.coerce(scenarios)
    mu, gamma, sigma = (univariate._mu_scalar, univariate._gamma_scalar,
                        univariate._sigma_scalar)
    lo, hi = bracket_raw(q, mu, gamma, sigma, Q)
    return quantile_raw(q, mu, gamma, sigma, Q, lo, hi, n_bisect)
