"""
Gamma distribution as an exponential family.

.. math::

    p(x \\mid \\alpha, \\beta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)}
    x^{\\alpha-1} e^{-\\beta x}, \\quad x > 0

**Exponential family structure:**

.. math::

    h(x) = 1, \\quad t(x) = [\\log x,\\; x]

.. math::

    \\theta = [\\alpha-1,\\; -\\beta], \\quad \\theta_1 > -1,\\; \\theta_2 < 0

.. math::

    \\psi(\\theta) = \\log\\Gamma(\\theta_1+1) - (\\theta_1+1)\\log(-\\theta_2)

.. math::

    \\eta = [\\psi(\\alpha) - \\log\\beta,\\; \\alpha/\\beta]
    \\quad \\text{(digamma, mean)}
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.utils.constants import LOG_EPS



class Gamma(ExponentialFamily):
    r"""Gamma(:math:`\alpha`, :math:`\beta`) distribution — shape :math:`\alpha > 0`, rate :math:`\beta > 0`."""

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
        r""":math:`\nabla\psi(\theta) = [\psi(\alpha) - \log\beta,\; \alpha/\beta]`. Analytical."""
        alpha = theta[0] + 1.0
        beta = -theta[1]
        return jnp.array([
            jax.scipy.special.digamma(alpha) - jnp.log(beta),
            alpha / beta,
        ])

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        r""":math:`\nabla^2\psi(\theta) = [[\psi'(\alpha),\; 1/\beta],\; [1/\beta,\; \alpha/\beta^2]]`. Analytical."""
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
        r"""Sample *n* observations from :math:`\mathrm{Gamma}(\alpha, \beta)` via JAX PRNG."""
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
        r"""
        Closed-form :math:`\eta \to \theta` via Newton on
        :math:`\psi(\alpha) - \log\alpha = \eta_1 - \log\eta_2`.

        :math:`\eta = [E[\log X],\; E[X]]` → :math:`\alpha` from digamma inversion,
        :math:`\beta = \alpha / \eta_2`.
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
    r"""
    Solve :math:`\psi(\alpha) - \log\alpha = \text{target}` for :math:`\alpha > 0`
    via Newton iterations.

    Uses ``jax.lax.fori_loop`` for JIT-compatibility.
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
