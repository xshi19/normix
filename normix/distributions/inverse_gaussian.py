"""
Inverse Gaussian (Wald) distribution as an exponential family.

PDF: f(x|μ,λ) = √(λ/(2π)) · x^{-3/2} · exp(−λ(x−μ)²/(2μ²x)),  x > 0

Exponential family:
  h(x)  = (2π)^{-1/2} · x^{-3/2}
  t(x)  = [x, 1/x]
  θ     = [−λ/(2μ²), −λ/2]   (θ₁ < 0, θ₂ < 0)
  ψ(θ)  = ½log(2π) − ½log(−2θ₂) + √((−2θ₁)(−2θ₂))
         (the ½log(2π) is absorbed into log h(x) = −½log(2π) − 3/2 log x)
  η     = [E[X], E[1/X]] = [μ, 1/μ + 1/λ]
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.utils.constants import LOG_EPS

jax.config.update("jax_enable_x64", True)


class InverseGaussian(ExponentialFamily):
    """InverseGaussian(μ, λ) — mean μ > 0, shape λ > 0."""

    mu: jax.Array
    lam: jax.Array

    def __init__(self, mu, lam):
        self.mu = jnp.asarray(mu, dtype=jnp.float64)
        self.lam = jnp.asarray(lam, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Tier 1: Exponential family interface
    # ------------------------------------------------------------------

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        a = -2.0 * theta[0]   # λ/μ²
        b = -2.0 * theta[1]   # λ
        b = jnp.maximum(b, LOG_EPS)
        sqrt_ab = jnp.sqrt(jnp.maximum(a * b, 0.0))
        return -0.5 * jnp.log(b) - sqrt_ab

    def natural_params(self) -> jax.Array:
        # θ₁ = -λ/(2μ²), θ₂ = -λ/2
        return jnp.array([-self.lam / (2.0 * self.mu**2), -self.lam / 2.0])

    @staticmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([x, 1.0 / x])

    @staticmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.where(
            x > 0,
            -0.5 * jnp.log(2.0 * jnp.pi) - 1.5 * jnp.log(x),
            -jnp.inf,
        )

    # ------------------------------------------------------------------
    # Tier 2: Analytical gradient and Hessian of log-partition
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        """∇ψ(θ) = [E[X], E[1/X]] = [√(b/a), 1/b + √(a/b)].  Analytical."""
        a = -2.0 * theta[0]   # λ/μ²
        b = jnp.maximum(-2.0 * theta[1], LOG_EPS)   # λ
        sqrt_ab = jnp.sqrt(jnp.maximum(a * b, 0.0))
        return jnp.array([
            sqrt_ab / jnp.maximum(a, LOG_EPS),       # E[X] = μ = √(b/a)
            1.0 / b + jnp.sqrt(jnp.maximum(a / b, 0.0)),  # E[1/X] = 1/λ + 1/μ
        ])

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        """∇²ψ(θ).  Analytical.

        H = [[(b/a)^{1/2}/a,    -1/√(ab)         ],
             [-1/√(ab),          2/b² + √a/b^{3/2}]]

        In terms of μ, λ:  H = [[μ³/λ, -μ/λ], [-μ/λ, 2/λ² + 1/(μλ)]]
        """
        a = jnp.maximum(-2.0 * theta[0], LOG_EPS)   # λ/μ²
        b = jnp.maximum(-2.0 * theta[1], LOG_EPS)   # λ
        sqrt_ab = jnp.sqrt(a * b)
        H00 = jnp.sqrt(b / a) / a                             # μ³/λ
        H01 = -1.0 / jnp.maximum(sqrt_ab, LOG_EPS)           # -μ/λ
        H11 = 2.0 / b**2 + jnp.sqrt(a) / b**1.5             # 2/λ² + 1/(μλ)
        return jnp.array([[H00, H01], [H01, H11]])

    # ------------------------------------------------------------------
    # Moments and sampling
    # ------------------------------------------------------------------

    def mean(self) -> jax.Array:
        return self.mu

    def var(self) -> jax.Array:
        return self.mu**3 / self.lam

    def cdf(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        sqrt_lam_over_x = jnp.sqrt(self.lam / x)
        t1 = sqrt_lam_over_x * (x / self.mu - 1.0)
        t2 = sqrt_lam_over_x * (x / self.mu + 1.0)
        return (jax.scipy.stats.norm.cdf(t1)
                + jnp.exp(2.0 * self.lam / self.mu)
                * jax.scipy.stats.norm.cdf(-t2))

    def rvs(self, n: int, seed: int = 42) -> "np.ndarray":
        from scipy import stats
        return stats.invgauss.rvs(mu=float(self.mu) / float(self.lam),
                                  scale=float(self.lam),
                                  size=n, random_state=seed)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "InverseGaussian":
        theta = jnp.asarray(theta, dtype=jnp.float64)
        # λ = -2θ₂, μ = √(θ₂/θ₁) = √(-2θ₂ / (-2θ₁)) = √(λ/a)
        lam = -2.0 * theta[1]
        mu = jnp.sqrt(theta[1] / theta[0])   # both negative, ratio positive
        return cls(mu=mu, lam=lam)

    @classmethod
    def from_expectation(
        cls,
        eta: jax.Array,
        *,
        theta0=None,
        maxiter: int = 100,
        tol: float = 1e-12,
        **kwargs,
    ) -> "InverseGaussian":
        """
        Closed-form from η = [E[X], E[1/X]] = [μ, 1/μ + 1/λ]:
          μ = η₁,   1/λ = η₂ - 1/η₁  →  λ = 1/(η₂ - 1/η₁)
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        mu = eta[0]
        lam = 1.0 / (eta[1] - 1.0 / eta[0])
        mu = jnp.maximum(mu, LOG_EPS)
        lam = jnp.maximum(lam, LOG_EPS)
        return cls(mu=mu, lam=lam)
