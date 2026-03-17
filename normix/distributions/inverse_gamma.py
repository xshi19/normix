"""
InverseGamma distribution as an exponential family.

PDF: p(x|α,β) = β^α/Γ(α) · x^{-α-1} · exp(-β/x),  x > 0

Exponential family:
  h(x)  = 1
  t(x)  = [-1/x, log x]
  θ     = [β, -(α+1)]        (θ₁ > 0, θ₂ < -1)
  ψ(θ)  = log Γ(-θ₂-1) − (-θ₂-1) log(θ₁)
         = log Γ(α) − α log β
  η     = [-α/β,  log β − ψ(α)]   (E[-1/X], E[log X])
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.utils.constants import LOG_EPS

jax.config.update("jax_enable_x64", True)


class InverseGamma(ExponentialFamily):
    """InverseGamma(α, β) — shape α > 0, rate β > 0."""

    alpha: jax.Array
    beta: jax.Array

    def __init__(self, alpha, beta):
        self.alpha = jnp.asarray(alpha, dtype=jnp.float64)
        self.beta = jnp.asarray(beta, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Exponential family interface
    # ------------------------------------------------------------------

    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        # α = -θ₂-1,  β = θ₁
        alpha = -theta[1] - 1.0
        beta = theta[0]
        return jax.scipy.special.gammaln(alpha) - alpha * jnp.log(beta)

    def natural_params(self) -> jax.Array:
        # θ = [β, -(α+1)]
        return jnp.array([self.beta, -(self.alpha + 1.0)])

    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([-1.0 / x, jnp.log(x)])

    def log_base_measure(self, x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, jnp.zeros((), jnp.float64), -jnp.inf)

    def expectation_params(self) -> jax.Array:
        """Analytical η = [-α/β,  log β − ψ(α)]."""
        return jnp.array([
            -self.alpha / self.beta,
            jnp.log(self.beta) - jax.scipy.special.digamma(self.alpha),
        ])

    def mean(self) -> jax.Array:
        return self.beta / (self.alpha - 1.0)

    def var(self) -> jax.Array:
        return self.beta**2 / ((self.alpha - 1.0)**2 * (self.alpha - 2.0))

    def cdf(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return 1.0 - jax.scipy.special.gammainc(self.alpha, self.beta / x)

    def rvs(self, n: int, seed: int = 42):
        import numpy as np
        from scipy import stats
        return stats.invgamma.rvs(a=float(self.alpha), scale=float(self.beta),
                                  size=n, random_state=seed)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "InverseGamma":
        theta = jnp.asarray(theta, dtype=jnp.float64)
        alpha = -theta[1] - 1.0
        beta = theta[0]
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_expectation(
        cls,
        eta: jax.Array,
        *,
        theta0=None,
        maxiter: int = 100,
        tol: float = 1e-12,
    ) -> "InverseGamma":
        """
        η = [-α/β, log β − ψ(α)].

        β = α/(-η₁),  solve ψ(α) − log α = −η₂ − log(-η₁) via Newton.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2 = eta[0], eta[1]
        # From η = [-α/β, log β − ψ(α)]: β = α/(-η₁), so ψ(α) − log α = −η₂ − log(−η₁).
        target = -eta2 - jnp.log(-eta1)
        alpha = _newton_digamma_ig(target)
        beta = alpha / (-eta1)
        alpha = jnp.maximum(alpha, LOG_EPS)
        beta = jnp.maximum(beta, LOG_EPS)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def fit_mle(
        cls,
        X: jax.Array,
        *,
        theta0=None,
        maxiter: int = 500,
        tol: float = 1e-10,
    ) -> "InverseGamma":
        X = jnp.asarray(X, dtype=jnp.float64)
        eta_hat = jnp.array([jnp.mean(-1.0 / X), jnp.mean(jnp.log(X))])
        return cls.from_expectation(eta_hat, theta0=theta0, maxiter=maxiter, tol=tol)

    @classmethod
    def _dummy_instance(cls) -> "InverseGamma":
        return cls(alpha=jnp.ones(()), beta=jnp.ones(()))


def _newton_digamma_ig(target: jax.Array, n_iter: int = 50) -> jax.Array:
    """Solve ψ(α) − log(α) = target for α > 0."""
    from normix.distributions.gamma import _newton_digamma
    return _newton_digamma(target, n_iter)
