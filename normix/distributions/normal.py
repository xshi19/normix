"""
Multivariate Normal distribution.

Stored: ``mu`` (d,), ``L_Sigma`` (d×d) lower-triangular Cholesky of :math:`\\Sigma`.
All linear algebra via ``L_Sigma`` — never form :math:`\\Sigma^{-1}` explicitly.

PDF:

.. math::

    f(x) = (2\\pi)^{-d/2} |\\Sigma|^{-1/2}
    \\exp\\!\\left(-\\tfrac{1}{2}(x-\\mu)^\\top\\Sigma^{-1}(x-\\mu)\\right)

Exponential family:

.. math::

    t(x) = [x, \\operatorname{vec}(xx^\\top)],\\quad
    \\theta = [\\Sigma^{-1}\\mu,\\; -\\tfrac{1}{2}\\operatorname{vec}(\\Sigma^{-1})],\\quad
    \\psi(\\theta) = \\tfrac{1}{2}\\mu^\\top\\Sigma^{-1}\\mu + \\tfrac{1}{2}\\log|\\Sigma| + \\tfrac{d}{2}\\log(2\\pi)

In practice we use the Cholesky log_prob formula directly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import equinox as eqx

jax.config.update("jax_enable_x64", True)


class MultivariateNormal(eqx.Module):
    r"""
    Multivariate Normal with Cholesky parametrization.

    Parameters
    ----------
    mu : jax.Array
        ``(d,)`` mean vector.
    L_Sigma : jax.Array
        ``(d, d)`` lower-triangular Cholesky factor of :math:`\Sigma`.
    """

    mu: jax.Array         # (d,)
    L_Sigma: jax.Array    # (d, d) lower-triangular

    def __init__(self, mu, L_Sigma):
        self.mu = jnp.asarray(mu, dtype=jnp.float64)
        self.L_Sigma = jnp.asarray(L_Sigma, dtype=jnp.float64)

    @classmethod
    def from_classical(cls, mu, sigma) -> "MultivariateNormal":
        """Construct from mean μ and covariance matrix Σ."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L_Sigma = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, L_Sigma=L_Sigma)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """log f(x) for a single observation x of shape (d,)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        d = x.shape[0]
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, x - self.mu, lower=True)
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(self.L_Sigma)))
        return (-0.5 * d * jnp.log(2.0 * jnp.pi)
                - 0.5 * log_det_sigma
                - 0.5 * jnp.dot(z, z))

    def sample(self, key: jax.Array, shape: tuple = ()) -> jax.Array:
        """Draw samples of shape (*shape, d)."""
        d = self.mu.shape[0]
        z = jax.random.normal(key, shape=(*shape, d), dtype=jnp.float64)
        return self.mu + (z @ self.L_Sigma.T)

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    @property
    def sigma(self) -> jax.Array:
        r"""Covariance matrix :math:`\Sigma = L_\Sigma L_\Sigma^\top`."""
        return self.L_Sigma @ self.L_Sigma.T
