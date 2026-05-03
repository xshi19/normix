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
        backend: str = "jax",
        **kwargs,
    ) -> "Gamma":
        r"""
        Closed-form :math:`\eta \to \theta` via Newton on
        :math:`\psi(\alpha) - \log\alpha = \eta_1 - \log\eta_2`.

        :math:`\eta = [E[\log X],\; E[X]]` → :math:`\alpha` from digamma inversion,
        :math:`\beta = \alpha / \eta_2`.

        Parameters
        ----------
        backend : str
            ``'jax'`` (default): ``lax.fori_loop`` Newton (JIT-compatible).
            ``'cpu'``: ``scipy.special`` digamma/polygamma (no XLA tracing).
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2 = eta[0], eta[1]
        target = eta1 - jnp.log(eta2)

        if backend == "cpu":
            alpha = _newton_digamma_cpu(float(target))
        else:
            alpha = _newton_digamma(target)

        alpha = jnp.maximum(jnp.asarray(alpha, dtype=jnp.float64), LOG_EPS)
        beta = jnp.maximum(alpha / eta2, LOG_EPS)
        return cls(alpha=alpha, beta=beta)


@jax.jit
def _newton_digamma(target: jax.Array, n_iter: int = 50) -> jax.Array:
    r"""
    Solve :math:`\psi(\alpha) - \log\alpha = \text{target}` for :math:`\alpha > 0`
    via Newton iterations.

    Uses ``jax.lax.fori_loop`` for JIT-compatibility. Decorated with
    ``@jax.jit`` so repeated Python-loop calls (e.g. inside the VG/NInvG
    M-step) hit the XLA cache instead of re-tracing the body every time.
    ``n_iter`` defaults are baked in; pass non-default values only inside an
    enclosing jit to avoid retracing per call.
    """
    alpha0 = jnp.where(
        target >= -2.22,
        1.0 / (2.0 * (-target)),
        jnp.exp(-target),
    )
    alpha0 = jnp.maximum(alpha0, 0.1)

    def body(_, alpha):
        psi = jax.scipy.special.digamma(alpha)
        psi_prime = jax.scipy.special.polygamma(1, alpha)
        f = psi - jnp.log(alpha) - target
        fp = psi_prime - 1.0 / alpha
        alpha_new = alpha - f / fp
        return jnp.maximum(alpha_new, LOG_EPS)

    return jax.lax.fori_loop(0, n_iter, body, alpha0)


def _newton_digamma_cpu(target: float, n_iter: int = 50, tol: float = 1e-12) -> float:
    r"""CPU variant using :func:`scipy.special.digamma` / :func:`scipy.special.polygamma`.

    No XLA tracing — suitable for the Python-loop EM path.
    """
    import math
    from scipy.special import digamma, polygamma

    if target >= -2.22:
        alpha = max(1.0 / (2.0 * (-target)), 0.1) if target < 0 else 0.1
    else:
        alpha = max(math.exp(-target), 0.1)

    for _ in range(n_iter):
        f = float(digamma(alpha)) - math.log(alpha) - target
        fp = float(polygamma(1, alpha)) - 1.0 / alpha
        if abs(fp) < 1e-30:
            break
        alpha_new = alpha - f / fp
        alpha = max(alpha_new, 1e-10)
        if abs(f) < tol:
            break

    return alpha
