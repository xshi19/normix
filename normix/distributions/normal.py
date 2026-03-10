"""
Multivariate Normal distribution.

Stored: mu (d,), L (d,d) lower-triangular Cholesky of Σ.
All linear algebra via L — never form Σ⁻¹ explicitly.

PDF: f(x) = (2π)^{-d/2} |Σ|^{-1/2} exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

Exponential family:
  t(x)  = [x, vec(xxᵀ)]   (d + d² components)
  θ     = [Σ⁻¹μ, -½vec(Σ⁻¹)]
  ψ(θ)  = ½μᵀΣ⁻¹μ + ½log|Σ| + d/2 log(2π)

In practice we use the Cholesky log_prob formula directly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import equinox as eqx

jax.config.update("jax_enable_x64", True)


class MultivariateNormal(eqx.Module):
    """
    Multivariate Normal with Cholesky parametrization.

    Parameters
    ----------
    mu : (d,) array
    L  : (d,d) lower-triangular Cholesky factor of Σ
    """

    mu: jax.Array   # (d,)
    L: jax.Array    # (d, d) lower-triangular

    def __init__(self, mu, L):
        self.mu = jnp.asarray(mu, dtype=jnp.float64)
        self.L = jnp.asarray(L, dtype=jnp.float64)

    @classmethod
    def from_classical(cls, mu, sigma) -> "MultivariateNormal":
        """Construct from mean μ and covariance matrix Σ."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, L=L)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """log f(x) for a single observation x of shape (d,)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        d = x.shape[0]
        # Solve L z = x - μ
        z = jax.scipy.linalg.solve_triangular(self.L, x - self.mu, lower=True)
        # log|Σ| = 2 Σ_i log L_{ii}
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(self.L)))
        return (-0.5 * d * jnp.log(2.0 * jnp.pi)
                - 0.5 * log_det_sigma
                - 0.5 * jnp.dot(z, z))

    def sample(self, key: jax.Array, shape: tuple = ()) -> jax.Array:
        """Draw samples of shape (*shape, d)."""
        d = self.mu.shape[0]
        z = jax.random.normal(key, shape=(*shape, d), dtype=jnp.float64)
        return self.mu + (z @ self.L.T)

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    @property
    def sigma(self) -> jax.Array:
        """Covariance matrix Σ = LLᵀ."""
        return self.L @ self.L.T
