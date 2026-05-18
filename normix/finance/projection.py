r"""
Portfolio projection: bridge from multivariate normal mixtures to univariate.

For a normal mixture :math:`X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z` with
:math:`Z \sim \mathcal{N}(0, \Sigma)` and weights :math:`w \in \mathbb{R}^d`,
the portfolio return is itself a univariate normal mixture:

.. math::

    w^\top X \stackrel{d}{=} w^\top \mu
        + w^\top \gamma \, Y + \sqrt{w^\top \Sigma w} \, \sqrt{Y} \, Z_1,

where :math:`Z_1 \sim \mathcal{N}(0, 1)`. This module wraps that projection
as a JAX pytree.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily


class PortfolioProjection(eqx.Module):
    r"""Univariate normal-mixture representation of a portfolio return.

    Stored: scalar :math:`\tilde\mu = w^\top \mu`, scalar
    :math:`\tilde\gamma = w^\top \gamma`, scalar
    :math:`\tilde\sigma = \sqrt{w^\top \Sigma w}`, and the subordinator
    distribution :math:`Y` (an :class:`ExponentialFamily` instance).

    The portfolio return is :math:`\tilde\mu + \tilde\gamma Y + \tilde\sigma \sqrt{Y} Z`
    with :math:`Z \sim \mathcal{N}(0, 1)`.
    """

    mu_p: jax.Array
    gamma_p: jax.Array
    sigma_p: jax.Array
    subordinator: ExponentialFamily

    def __init__(self, mu_p, gamma_p, sigma_p, subordinator):
        object.__setattr__(self, 'mu_p', jnp.asarray(mu_p, dtype=jnp.float64))
        object.__setattr__(self, 'gamma_p', jnp.asarray(gamma_p, dtype=jnp.float64))
        object.__setattr__(self, 'sigma_p', jnp.asarray(sigma_p, dtype=jnp.float64))
        object.__setattr__(self, 'subordinator', subordinator)

    @classmethod
    def from_model(cls, model, w: jax.Array) -> "PortfolioProjection":
        r"""Build the projection from a :class:`NormalMixture` and weights ``w``."""
        j = model._joint
        w = jnp.asarray(w, dtype=jnp.float64)
        mu_p = jnp.dot(w, j.mu)
        gamma_p = jnp.dot(w, j.gamma)
        Sigma_w = j.sigma() @ w
        sigma_p = jnp.sqrt(jnp.dot(w, Sigma_w))
        return cls(mu_p=mu_p, gamma_p=gamma_p, sigma_p=sigma_p,
                   subordinator=j.subordinator())

    def mean(self) -> jax.Array:
        r""":math:`E[w^\top X] = \tilde\mu + \tilde\gamma \, E[Y]`."""
        return self.mu_p + self.gamma_p * self.subordinator.mean()

    def var(self) -> jax.Array:
        r""":math:`\mathrm{Var}[w^\top X]
        = E[Y]\tilde\sigma^2 + \mathrm{Var}[Y]\tilde\gamma^2`."""
        EY = self.subordinator.mean()
        VY = self.subordinator.var()
        return EY * self.sigma_p ** 2 + VY * self.gamma_p ** 2

    def std(self) -> jax.Array:
        return jnp.sqrt(self.var())

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        r"""Sample ``n`` portfolio returns :math:`w^\top X`."""
        key = jax.random.PRNGKey(seed)
        kZ, _ = jax.random.split(key)
        # Use the subordinator's own sampler with a derived seed.
        Y = self.subordinator.rvs(n, seed=int(seed) + 1)
        Z = jax.random.normal(kZ, shape=(n,), dtype=jnp.float64)
        return self.mu_p + self.gamma_p * Y + self.sigma_p * jnp.sqrt(Y) * Z


def project_portfolio(model, w: jax.Array) -> PortfolioProjection:
    """Convenience wrapper for :meth:`PortfolioProjection.from_model`."""
    return PortfolioProjection.from_model(model, w)
