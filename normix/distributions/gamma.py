"""
Gamma distribution as an exponential family.

PDF: p(x|α,β) = β^α/Γ(α) · x^{α-1} · exp(-βx),  x > 0

Exponential family:
  h(x)    = 1
  t(x)    = [log x, x]
  θ       = [α-1, -β]       (θ₁ > -1, θ₂ < 0)
  ψ(θ)    = log Γ(θ₁+1) − (θ₁+1) log(−θ₂)
  η       = [ψ(α) − log β,  α/β]   (digamma, mean)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxopt

from normix.exponential_family import ExponentialFamily

jax.config.update("jax_enable_x64", True)

_EPS = 1e-30


class Gamma(ExponentialFamily):
    """Gamma(α, β) distribution — shape α > 0, rate β > 0."""

    alpha: jax.Array   # shape
    beta: jax.Array    # rate

    def __init__(self, alpha, beta):
        self.alpha = jnp.asarray(alpha, dtype=jnp.float64)
        self.beta = jnp.asarray(beta, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Exponential family interface
    # ------------------------------------------------------------------

    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        alpha = theta[0] + 1.0        # α = θ₁ + 1
        beta = -theta[1]              # β = −θ₂
        return jax.scipy.special.gammaln(alpha) - alpha * jnp.log(beta)

    def natural_params(self) -> jax.Array:
        return jnp.array([self.alpha - 1.0, -self.beta])

    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([jnp.log(x), x])

    def log_base_measure(self, x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, jnp.zeros((), jnp.float64), -jnp.inf)

    def expectation_params(self) -> jax.Array:
        """Analytical η = [ψ(α)−log β,  α/β]."""
        return jnp.array([
            jax.scipy.special.digamma(self.alpha) - jnp.log(self.beta),
            self.alpha / self.beta,
        ])

    def mean(self) -> jax.Array:
        return self.alpha / self.beta

    def var(self) -> jax.Array:
        return self.alpha / self.beta**2

    def std(self) -> jax.Array:
        return jnp.sqrt(self.alpha) / self.beta

    def cdf(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jax.scipy.special.gammainc(self.alpha, self.beta * x)

    def rvs(self, n: int, seed: int = 42) -> "np.ndarray":
        import numpy as np
        return np.random.default_rng(seed).gamma(
            float(self.alpha), 1.0 / float(self.beta), size=n)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "Gamma":
        theta = jnp.asarray(theta, dtype=jnp.float64)
        return cls(alpha=theta[0] + 1.0, beta=-theta[1])

    @classmethod
    def from_expectation(
        cls,
        eta: jax.Array,
        *,
        theta0=None,
        maxiter: int = 100,
        tol: float = 1e-12,
    ) -> "Gamma":
        """
        Closed-form η → θ via Newton's method on ψ(α) − log(α) = η₁ − log(η₂).

        η = [E[log X], E[X]]  →  α from digamma inversion, β = α/η₂.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2 = eta[0], eta[1]

        # Solve ψ(α) − log α = η₁ − log η₂ (fixed-point target)
        target = eta1 - jnp.log(eta2)
        alpha = _newton_digamma(target)
        beta = alpha / eta2
        alpha = jnp.maximum(alpha, _EPS)
        beta = jnp.maximum(beta, _EPS)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def fit_mle(cls, X: jax.Array) -> "Gamma":
        X = jnp.asarray(X, dtype=jnp.float64)
        eta_hat = jnp.array([jnp.mean(jnp.log(X)), jnp.mean(X)])
        return cls.from_expectation(eta_hat)

    @classmethod
    def _dummy_instance(cls) -> "Gamma":
        return cls(alpha=jnp.ones(()), beta=jnp.ones(()))


def _newton_digamma(target: jax.Array, n_iter: int = 50) -> jax.Array:
    """
    Solve ψ(α) − log(α) = target for α > 0 via Newton iterations.

    Uses jax.lax.fori_loop for JIT-compatibility.
    Initial guess from alpha ≈ exp(target + log(exp(-target) - 1)).
    """
    # Reasonable initial guess for α
    alpha0 = jnp.where(
        target >= -2.22,
        1.0 / (2.0 * (-target)),  # rough: ψ(α)-log(α) ≈ -1/(2α) for large α
        jnp.exp(-target),
    )
    alpha0 = jnp.maximum(alpha0, 0.1)

    def body(_, alpha):
        psi = jax.scipy.special.digamma(alpha)
        # trigamma = polygamma(1, α)
        psi_prime = jax.scipy.special.polygamma(1, alpha)
        f = psi - jnp.log(alpha) - target
        fp = psi_prime - 1.0 / alpha
        alpha_new = alpha - f / fp
        return jnp.maximum(alpha_new, _EPS)

    return jax.lax.fori_loop(0, n_iter, body, alpha0)
