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

from normix.exponential_family import ExponentialFamily
from normix.utils.constants import LOG_EPS

jax.config.update("jax_enable_x64", True)


class Gamma(ExponentialFamily):
    """Gamma(α, β) distribution — shape α > 0, rate β > 0."""

    alpha: jax.Array   # shape
    beta: jax.Array    # rate

    def __init__(self, alpha, beta):
        self.alpha = jnp.asarray(alpha, dtype=jnp.float64)
        self.beta = jnp.asarray(beta, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Tier 1: Exponential family interface
    # ------------------------------------------------------------------

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        alpha = theta[0] + 1.0        # α = θ₁ + 1
        beta = -theta[1]              # β = −θ₂
        return jax.scipy.special.gammaln(alpha) - alpha * jnp.log(beta)

    def natural_params(self) -> jax.Array:
        return jnp.array([self.alpha - 1.0, -self.beta])

    @staticmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([jnp.log(x), x])

    @staticmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, jnp.zeros((), jnp.float64), -jnp.inf)

    # ------------------------------------------------------------------
    # Tier 2: Analytical gradient and Hessian of log-partition
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        """∇ψ(θ) = [digamma(α) − log β,  α/β].  Analytical."""
        alpha = theta[0] + 1.0
        beta = -theta[1]
        return jnp.array([
            jax.scipy.special.digamma(alpha) - jnp.log(beta),
            alpha / beta,
        ])

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        """∇²ψ(θ) = [[trigamma(α), 1/β], [1/β, α/β²]].  Analytical."""
        alpha = theta[0] + 1.0
        beta = -theta[1]
        H00 = jax.scipy.special.polygamma(1, alpha)   # trigamma
        H01 = 1.0 / beta
        H11 = alpha / beta ** 2
        return jnp.array([[H00, H01], [H01, H11]])

    # ------------------------------------------------------------------
    # Moments and sampling
    # ------------------------------------------------------------------

    def mean(self) -> jax.Array:
        return self.alpha / self.beta

    def var(self) -> jax.Array:
        return self.alpha / self.beta**2

    def cdf(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jax.scipy.special.gammainc(self.alpha, self.beta * x)

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        """Sample n observations from Gamma(α, β) via JAX PRNG."""
        key = jax.random.PRNGKey(seed)
        return jax.random.gamma(key, self.alpha, shape=(n,), dtype=jnp.float64) / self.beta

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
        **kwargs,
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
        alpha = jnp.maximum(alpha, LOG_EPS)
        beta = jnp.maximum(beta, LOG_EPS)
        return cls(alpha=alpha, beta=beta)


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
        return jnp.maximum(alpha_new, LOG_EPS)

    return jax.lax.fori_loop(0, n_iter, body, alpha0)
