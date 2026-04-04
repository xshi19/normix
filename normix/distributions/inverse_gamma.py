"""
InverseGamma distribution as an exponential family.

.. math::

    p(x \\mid \\alpha, \\beta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)}
    x^{-\\alpha-1} e^{-\\beta/x}, \\quad x > 0

**Exponential family structure:**

.. math::

    h(x) = 1, \\quad t(x) = [-1/x,\\; \\log x]

.. math::

    \\theta = [\\beta,\\; -(\\alpha+1)], \\quad \\theta_1 > 0,\\; \\theta_2 < -1

.. math::

    \\psi(\\theta) = \\log\\Gamma(-\\theta_2-1) - (-\\theta_2-1)\\log\\theta_1
    = \\log\\Gamma(\\alpha) - \\alpha\\log\\beta

.. math::

    \\eta = [-\\alpha/\\beta,\\; \\log\\beta - \\psi(\\alpha)]
    \\quad (E[-1/X],\\; E[\\log X])
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.utils.constants import LOG_EPS



class InverseGamma(ExponentialFamily):
    r"""InverseGamma(:math:`\alpha`, :math:`\beta`) — shape :math:`\alpha > 0`, rate :math:`\beta > 0`."""

    alpha: jax.Array
    beta: jax.Array

    def __init__(self, alpha, beta):
        self.alpha = jnp.asarray(alpha, dtype=jnp.float64)
        self.beta = jnp.asarray(beta, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Tier 1: Exponential family interface
    # ------------------------------------------------------------------

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        # α = -θ₂-1,  β = θ₁
        alpha = -theta[1] - 1.0
        beta = theta[0]
        return jax.scipy.special.gammaln(alpha) - alpha * jnp.log(beta)

    def natural_params(self) -> jax.Array:
        # θ = [β, -(α+1)]
        return jnp.array([self.beta, -(self.alpha + 1.0)])

    @staticmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([-1.0 / x, jnp.log(x)])

    @staticmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, jnp.zeros((), jnp.float64), -jnp.inf)

    # ------------------------------------------------------------------
    # Tier 2: Analytical gradient and Hessian of log-partition
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        r""":math:`\nabla\psi(\theta) = [-\alpha/\beta,\; \log\beta - \psi(\alpha)]`. Analytical."""
        alpha = -theta[1] - 1.0
        beta = theta[0]
        return jnp.array([
            -alpha / beta,
            jnp.log(beta) - jax.scipy.special.digamma(alpha),
        ])

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        r""":math:`\nabla^2\psi(\theta) = [[\alpha/\beta^2,\; 1/\beta],\; [1/\beta,\; \psi'(\alpha)]]`. Analytical.

        :math:`H_{00} = \partial^2\psi/\partial\theta_1^2 = \alpha/\beta^2`,
        :math:`H_{01} = 1/\beta` (:math:`\partial\alpha/\partial\theta_2 = -1`),
        :math:`H_{11} = \psi'(\alpha)` (trigamma).
        """
        alpha = -theta[1] - 1.0
        beta = theta[0]
        H00 = alpha / beta ** 2
        H01 = 1.0 / beta
        H11 = jax.scipy.special.polygamma(1, alpha)   # trigamma
        return jnp.array([[H00, H01], [H01, H11]])

    # ------------------------------------------------------------------
    # Moments and sampling
    # ------------------------------------------------------------------

    def mean(self) -> jax.Array:
        return self.beta / (self.alpha - 1.0)

    def var(self) -> jax.Array:
        return self.beta**2 / ((self.alpha - 1.0)**2 * (self.alpha - 2.0))

    def cdf(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return 1.0 - jax.scipy.special.gammainc(self.alpha, self.beta / x)

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        r"""Sample *n* observations from :math:`\mathrm{InvGamma}(\alpha, \beta)` via JAX PRNG.

        Uses the relation: if :math:`X \sim \mathrm{Gamma}(\alpha, 1/\beta)` then
        :math:`1/X \sim \mathrm{InvGamma}(\alpha, \beta)`.
        """
        key = jax.random.PRNGKey(seed)
        g = jax.random.gamma(key, self.alpha, shape=(n,), dtype=jnp.float64)
        return self.beta / g

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
        **kwargs,
    ) -> "InverseGamma":
        r"""
        :math:`\eta = [-\alpha/\beta,\; \log\beta - \psi(\alpha)]`.

        :math:`\beta = \alpha / (-\eta_1)`;  solve
        :math:`\psi(\alpha) - \log\alpha = -\eta_2 - \log(-\eta_1)` via Newton.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2 = eta[0], eta[1]
        # From η = [-α/β, log β − ψ(α)]: β = α/(-η₁), so ψ(α) − log α = −η₂ − log(−η₁).
        from normix.distributions.gamma import _newton_digamma
        target = -eta2 - jnp.log(-eta1)
        alpha = _newton_digamma(target)
        beta = alpha / (-eta1)
        alpha = jnp.maximum(alpha, LOG_EPS)
        beta = jnp.maximum(beta, LOG_EPS)
        return cls(alpha=alpha, beta=beta)
