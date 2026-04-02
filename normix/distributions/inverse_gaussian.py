"""
Inverse Gaussian (Wald) distribution as an exponential family.

.. math::

    f(x \\mid \\mu, \\lambda) = \\sqrt{\\frac{\\lambda}{2\\pi}}\\,
    x^{-3/2} \\exp\\!\\left(-\\frac{\\lambda(x-\\mu)^2}{2\\mu^2 x}\\right),
    \\quad x > 0

**Exponential family structure:**

.. math::

    h(x) = (2\\pi)^{-1/2} x^{-3/2}, \\quad t(x) = [x,\\; 1/x]

.. math::

    \\theta = \\Bigl[-\\tfrac{\\lambda}{2\\mu^2},\\; -\\tfrac{\\lambda}{2}\\Bigr],
    \\quad \\theta_1 < 0,\\; \\theta_2 < 0

.. math::

    \\psi(\\theta) = -\\tfrac{1}{2}\\log(-2\\theta_2)
    - \\sqrt{(-2\\theta_1)(-2\\theta_2)}

    \\bigl(\\tfrac{1}{2}\\log(2\\pi)\\ \\text{is absorbed into}\\
    \\log h(x) = -\\tfrac{1}{2}\\log(2\\pi) - \\tfrac{3}{2}\\log x\\bigr)

.. math::

    \\eta = [E[X],\\; E[1/X]] = [\\mu,\\; 1/\\mu + 1/\\lambda]
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.utils.constants import LOG_EPS

jax.config.update("jax_enable_x64", True)


class InverseGaussian(ExponentialFamily):
    r"""InverseGaussian(:math:`\mu`, :math:`\lambda`) — mean :math:`\mu > 0`, shape :math:`\lambda > 0`."""

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
        r""":math:`\nabla\psi(\theta) = [E[X],\; E[1/X]] = [\sqrt{b/a},\; 1/b + \sqrt{a/b}]`. Analytical."""
        a = -2.0 * theta[0]   # λ/μ²
        b = jnp.maximum(-2.0 * theta[1], LOG_EPS)   # λ
        sqrt_ab = jnp.sqrt(jnp.maximum(a * b, 0.0))
        return jnp.array([
            sqrt_ab / jnp.maximum(a, LOG_EPS),       # E[X] = μ = √(b/a)
            1.0 / b + jnp.sqrt(jnp.maximum(a / b, 0.0)),  # E[1/X] = 1/λ + 1/μ
        ])

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        r""":math:`\nabla^2\psi(\theta)`. Analytical.

        .. math::

            H = \begin{pmatrix}
                (b/a)^{1/2}/a & -1/\sqrt{ab} \\
                -1/\sqrt{ab}  & 2/b^2 + \sqrt{a}/b^{3/2}
            \end{pmatrix}
            = \begin{pmatrix}
                \mu^3/\lambda & -\mu/\lambda \\
                -\mu/\lambda  & 2/\lambda^2 + 1/(\mu\lambda)
            \end{pmatrix}
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
        r"""CDF of the Inverse Gaussian distribution (log-space stable).

        .. math::

            F(x) = \Phi(t_1) + \exp\!\bigl(2\lambda/\mu + \log\Phi(-t_2)\bigr)

        where :math:`t_1 = \sqrt{\lambda/x}\,(x/\mu - 1)` and
        :math:`t_2 = \sqrt{\lambda/x}\,(x/\mu + 1)`.  The second term uses
        ``log_ndtr`` to avoid overflow when :math:`\lambda/\mu` is large.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        sqrt_lam_over_x = jnp.sqrt(self.lam / x)
        t1 = sqrt_lam_over_x * (x / self.mu - 1.0)
        t2 = sqrt_lam_over_x * (x / self.mu + 1.0)
        log_term2 = 2.0 * self.lam / self.mu + jax.scipy.special.log_ndtr(-t2)
        return jax.scipy.stats.norm.cdf(t1) + jnp.exp(log_term2)

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        r"""Sample *n* observations from :math:`\mathrm{InvGaussian}(\mu, \lambda)` via JAX PRNG.

        Uses the algorithm from Michael, Schucany & Haas (1976):

        1. :math:`\nu \sim \mathcal{N}(0,1)`, :math:`y = \nu^2`
        2. :math:`x = \mu + \frac{\mu^2 y}{2\lambda}
           - \frac{\mu}{2\lambda}\sqrt{4\mu\lambda y + \mu^2 y^2}`
        3. :math:`z \sim \mathrm{Uniform}(0,1)`; return :math:`x` if
           :math:`z \le \mu/(\mu+x)`, else :math:`\mu^2/x`

        Uses ``jnp.where`` for vectorized branching over the full sample array.
        """
        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)

        nu = jax.random.normal(key1, shape=(n,), dtype=jnp.float64)
        y = nu * nu

        mu, lam = self.mu, self.lam
        x = mu + (mu * mu * y) / (2.0 * lam) \
            - (mu / (2.0 * lam)) * jnp.sqrt(4.0 * mu * lam * y + mu * mu * y * y)

        z = jax.random.uniform(key2, shape=(n,), dtype=jnp.float64)
        return jnp.where(z <= mu / (mu + x), x, mu * mu / x)

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
        r"""
        Closed-form from :math:`\eta = [E[X],\; E[1/X]] = [\mu,\; 1/\mu + 1/\lambda]`:

        :math:`\mu = \eta_1`, :math:`\lambda = 1/(\eta_2 - 1/\eta_1)`.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        mu = eta[0]
        lam = 1.0 / (eta[1] - 1.0 / eta[0])
        mu = jnp.maximum(mu, LOG_EPS)
        lam = jnp.maximum(lam, LOG_EPS)
        return cls(mu=mu, lam=lam)
