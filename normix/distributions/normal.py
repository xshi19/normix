"""
Multivariate Normal distribution.

Stored: ``mu`` (d,), ``L_Sigma`` (d×d) lower-triangular Cholesky of :math:`\\Sigma`.
All linear algebra via ``L_Sigma`` — never form :math:`\\Sigma^{-1}` explicitly.

Exponential family structure
-----------------------------

.. math::

    t(x) = [x,\\; \\operatorname{vec}(xx^\\top)], \\quad
    \\theta = [\\Sigma^{-1}\\mu,\\; -\\tfrac{1}{2}\\operatorname{vec}(\\Sigma^{-1})], \\quad
    \\log h(x) = 0

.. math::

    \\psi(\\theta) = \\tfrac{1}{2}\\mu^\\top\\Sigma^{-1}\\mu
    - \\tfrac{1}{2}\\log|\\Sigma^{-1}|
    + \\tfrac{d}{2}\\log(2\\pi)

where :math:`\\operatorname{vec}` uses row-major order (:func:`numpy.ndarray.ravel`).

All parametrization conversions are analytical (closed-form):

- classical :math:`\\leftrightarrow` natural: ``natural_params`` / ``from_natural``
- natural :math:`\\to` expectation: ``_grad_log_partition`` (analytical override)
- expectation :math:`\\to` classical: ``from_expectation``

No Bregman solver is ever invoked.  ``fit_mle`` computes
:math:`\\hat\\eta = n^{-1}\\sum_i t(x_i)` and calls ``from_expectation``
(closed-form).  ``log_prob`` overrides the inherited EF formula with a
direct Cholesky computation for efficiency.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily

jax.config.update("jax_enable_x64", True)


class MultivariateNormal(ExponentialFamily):
    r"""
    Multivariate Normal distribution as an exponential family.

    Parameters
    ----------
    mu : jax.Array
        ``(d,)`` mean vector.
    L_Sigma : jax.Array
        ``(d, d)`` lower-triangular Cholesky factor of :math:`\Sigma`.

    Notes
    -----
    Natural parameters: :math:`\theta = [\Sigma^{-1}\mu,\; -\tfrac{1}{2}\operatorname{vec}(\Sigma^{-1})]`.
    Sufficient statistics: :math:`t(x) = [x,\; \operatorname{vec}(xx^\top)]`.
    Log-partition: :math:`\psi(\theta) = \tfrac{1}{2}\mu^\top\Sigma^{-1}\mu - \tfrac{1}{2}\log|\Sigma^{-1}| + \tfrac{d}{2}\log(2\pi)`.
    Log base measure: :math:`\log h(x) = 0`.
    """

    mu: jax.Array         # (d,)
    L_Sigma: jax.Array    # (d, d) lower-triangular

    def __init__(self, mu, L_Sigma):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))

    # ------------------------------------------------------------------
    # Tier 1: ExponentialFamily abstract methods
    # ------------------------------------------------------------------

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta) = \tfrac{1}{2}\theta_1^\top\Lambda^{-1}\theta_1 - \tfrac{1}{2}\log|\Lambda| + \tfrac{d}{2}\log(2\pi)`

        where :math:`\Lambda = -2\,\mathrm{reshape}(\theta_2, d, d)` is the precision matrix
        and :math:`\theta = [\theta_1,\; \theta_2]` with :math:`\theta_1 \in \mathbb{R}^d`,
        :math:`\theta_2 \in \mathbb{R}^{d^2}`.
        """
        n = theta.shape[0]
        d = int(round((-1.0 + (1.0 + 4.0 * n) ** 0.5) / 2.0))
        theta_1 = theta[:d]
        theta_2 = theta[d:].reshape(d, d)
        Lambda = -2.0 * theta_2
        Lambda = 0.5 * (Lambda + Lambda.T)          # symmetrize
        L_Lambda = jnp.linalg.cholesky(Lambda)
        z = jax.scipy.linalg.solve_triangular(L_Lambda, theta_1, lower=True)
        log_det_Lambda = 2.0 * jnp.sum(jnp.log(jnp.diag(L_Lambda)))
        return (0.5 * jnp.dot(z, z)
                - 0.5 * log_det_Lambda
                + 0.5 * d * jnp.log(2.0 * jnp.pi))

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [\Sigma^{-1}\mu,\; -\tfrac{1}{2}\operatorname{vec}(\Sigma^{-1})]`
        """
        d = self.d
        L_inv = jax.scipy.linalg.solve_triangular(
            self.L_Sigma, jnp.eye(d, dtype=jnp.float64), lower=True)
        Lambda = L_inv.T @ L_inv                    # Σ⁻¹
        theta_1 = Lambda @ self.mu
        theta_2 = -0.5 * Lambda
        return jnp.concatenate([theta_1, theta_2.ravel()])

    @staticmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        r""":math:`t(x) = [x,\; \operatorname{vec}(xx^\top)]`."""
        return jnp.concatenate([x, jnp.outer(x, x).ravel()])

    @staticmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        r""":math:`\log h(x) = 0` (base measure is Lebesgue)."""
        return jnp.array(0.0, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Tier 2: analytical gradient (no autodiff needed)
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        r"""
        :math:`\nabla\psi(\theta) = [\mu,\; \operatorname{vec}(\Sigma + \mu\mu^\top)]`

        Analytical — recovers :math:`\mu = \Lambda^{-1}\theta_1` and
        :math:`\Sigma = \Lambda^{-1}` from :math:`\theta`, then assembles
        the expectation parameters directly.
        """
        n = theta.shape[0]
        d = int(round((-1.0 + (1.0 + 4.0 * n) ** 0.5) / 2.0))
        theta_1 = theta[:d]
        theta_2 = theta[d:].reshape(d, d)
        Lambda = -2.0 * theta_2
        Lambda = 0.5 * (Lambda + Lambda.T)
        L_Lambda = jnp.linalg.cholesky(Lambda)
        mu = jax.scipy.linalg.solve_triangular(
            L_Lambda.T,
            jax.scipy.linalg.solve_triangular(L_Lambda, theta_1, lower=True),
            lower=False,
        )
        L_inv = jax.scipy.linalg.solve_triangular(
            L_Lambda, jnp.eye(d, dtype=jnp.float64), lower=True)
        Sigma = L_inv.T @ L_inv
        return jnp.concatenate([mu, (Sigma + jnp.outer(mu, mu)).ravel()])

    # ------------------------------------------------------------------
    # Constructors — all analytical (no Bregman solver)
    # ------------------------------------------------------------------

    @classmethod
    def from_classical(cls, mu, sigma) -> "MultivariateNormal":
        """Construct from mean μ and covariance matrix Σ."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L_Sigma = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, L_Sigma=L_Sigma)

    @classmethod
    def from_expectation(cls, eta: jax.Array, **_kwargs) -> "MultivariateNormal":
        r"""
        Closed-form inversion :math:`\eta \to \theta \to (\mu, L_\Sigma)`.

        :math:`\eta = [E[X],\; \operatorname{vec}(E[XX^\top])]`, so

        .. math::

            \mu = \eta_1, \qquad
            \Sigma = \operatorname{reshape}(\eta_2, d, d) - \mu\mu^\top

        Parameters
        ----------
        eta : jax.Array
            Expectation parameters of shape ``(d + d²,)``.
        **_kwargs
            Ignored (accepts ``backend``, ``theta0``, etc. for API compatibility).
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        n = eta.shape[0]
        d = int(round((-1.0 + (1.0 + 4.0 * n) ** 0.5) / 2.0))
        mu = eta[:d]
        E_xxt = eta[d:].reshape(d, d)
        Sigma = E_xxt - jnp.outer(mu, mu)
        Sigma = 0.5 * (Sigma + Sigma.T)            # symmetrize
        L_Sigma = jnp.linalg.cholesky(Sigma)
        return cls(mu=mu, L_Sigma=L_Sigma)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "MultivariateNormal":
        r"""
        Construct from natural parameters :math:`\theta = [\theta_1, \theta_2]`.

        Recovers :math:`\Lambda = -2\,\mathrm{reshape}(\theta_2, d, d)`,
        then :math:`\mu = \Lambda^{-1}\theta_1` and
        :math:`L_\Sigma = \mathrm{chol}(\Lambda^{-1})`.
        """
        n = theta.shape[0]
        d = int(round((-1.0 + (1.0 + 4.0 * n) ** 0.5) / 2.0))
        theta_1 = theta[:d]
        theta_2 = theta[d:].reshape(d, d)
        Lambda = -2.0 * theta_2
        Lambda = 0.5 * (Lambda + Lambda.T)
        L_Lambda = jnp.linalg.cholesky(Lambda)
        # μ = Λ⁻¹θ₁
        mu = jax.scipy.linalg.solve_triangular(
            L_Lambda.T,
            jax.scipy.linalg.solve_triangular(L_Lambda, theta_1, lower=True),
            lower=False,
        )
        # Σ = Λ⁻¹: form via L_inv to avoid explicit inversion
        L_inv = jax.scipy.linalg.solve_triangular(
            L_Lambda, jnp.eye(d, dtype=jnp.float64), lower=True)
        Sigma = L_inv.T @ L_inv
        L_Sigma = jnp.linalg.cholesky(Sigma)
        return cls(mu=mu, L_Sigma=L_Sigma)

    # ------------------------------------------------------------------
    # Standard distribution API
    # ------------------------------------------------------------------

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        :math:`\log f(x) = -\tfrac{d}{2}\log(2\pi) - \tfrac{1}{2}\log|\Sigma| - \tfrac{1}{2}\|L_\Sigma^{-1}(x-\mu)\|^2`.

        Overrides the inherited EF formula for numerical efficiency (Cholesky-direct).
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        d = x.shape[0]
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, x - self.mu, lower=True)
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(self.L_Sigma)))
        return (-0.5 * d * jnp.log(2.0 * jnp.pi)
                - 0.5 * log_det_sigma
                - 0.5 * jnp.dot(z, z))

    def mean(self) -> jax.Array:
        r""":math:`E[X] = \mu`."""
        return self.mu

    def cov(self) -> jax.Array:
        r""":math:`\mathrm{Cov}[X] = \Sigma = L_\Sigma L_\Sigma^\top`."""
        return self.L_Sigma @ self.L_Sigma.T

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        """
        Draw ``n`` i.i.d. samples via JAX PRNG.

        Returns
        -------
        jax.Array
            Shape ``(n, d)``.
        """
        key = jax.random.PRNGKey(seed)
        d = self.d
        z = jax.random.normal(key, shape=(n, d), dtype=jnp.float64)
        return self.mu[None, :] + z @ self.L_Sigma.T

    def sample(self, key: jax.Array, shape: tuple = ()) -> jax.Array:
        """Draw samples via an explicit JAX key.  Legacy API — prefer ``rvs``."""
        d = self.mu.shape[0]
        z = jax.random.normal(key, shape=(*shape, d), dtype=jnp.float64)
        return self.mu + (z @ self.L_Sigma.T)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        """Dimensionality."""
        return int(self.mu.shape[0])

    @property
    def dim(self) -> int:
        """Dimensionality (alias for ``d``)."""
        return self.d

    @property
    def sigma(self) -> jax.Array:
        r"""Covariance matrix :math:`\Sigma = L_\Sigma L_\Sigma^\top` (alias for ``cov()``)."""
        return self.cov()
