"""
Variance Gamma (VG) distribution.

Special case of GH with GIG → Gamma subordinator
(:math:`b \\to 0`, :math:`p > 0`).
:math:`Y \\sim \\mathrm{Gamma}(\\alpha, \\beta)`,
i.e. GIG(:math:`p = \\alpha`, :math:`a = 2\\beta`, :math:`b \\to 0`).

Stored: :math:`\\mu`, :math:`\\gamma`, :math:`L_\\Sigma` (Cholesky of :math:`\\Sigma`),
:math:`\\alpha` (shape), :math:`\\beta` (rate) of Gamma.
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture

jax.config.update("jax_enable_x64", True)

from normix.utils.constants import LOG_EPS


class JointVarianceGamma(JointNormalMixture):
    r"""
    Joint :math:`f(x,y)`: :math:`X\mid Y \sim \mathcal{N}(\mu+\gamma y, \Sigma y)`,
    :math:`Y \sim \mathrm{Gamma}(\alpha, \beta)`.

    GIG limit: :math:`p = \alpha`, :math:`a = 2\beta`, :math:`b \to 0`.
    """

    alpha: jax.Array   # Gamma shape
    beta: jax.Array    # Gamma rate

    def __init__(self, mu, gamma, L_Sigma, alpha, beta):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'alpha', jnp.asarray(alpha, dtype=jnp.float64))
        object.__setattr__(self, 'beta', jnp.asarray(beta, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.gamma import Gamma
        return Gamma(alpha=self.alpha, beta=self.beta)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        r"""
        Posterior :math:`Y\mid X=x \sim \mathrm{GIG}(p_{\mathrm{post}}, a_{\mathrm{post}}, b_{\mathrm{post}})`:

        .. math::

            p_{\mathrm{post}} = \alpha - d/2, \quad
            a_{\mathrm{post}} = 2\beta + \gamma^\top\Sigma^{-1}\gamma, \quad
            b_{\mathrm{post}} = (x-\mu)^\top\Sigma^{-1}(x-\mu)
        """
        from normix.distributions.generalized_inverse_gaussian import GIG
        d = self.d
        z, w, z2, w2, zw = self._quad_forms(x)

        p_post = self.alpha - d / 2.0
        a_post = 2.0 * self.beta + w2
        b_post = z2

        gig = GIG(p=p_post, a=a_post, b=b_post)
        eta = gig.expectation_params()
        return {
            'E_log_Y': eta[0],
            'E_inv_Y': eta[1],
            'E_Y': eta[2],
        }

    def _posterior_gig_params(
        self, z2: jax.Array, w2: jax.Array
    ):
        """Posterior GIG (p, a, b) given quad-form scalars."""
        return (self.alpha - self.d / 2.0,
                2.0 * self.beta + w2,
                z2)

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [\alpha-1-d/2,\; -\tfrac{1}{2}\mu^\top\Lambda\mu,\;
        -(\beta+\tfrac{1}{2}\gamma^\top\Lambda\gamma),\;
        \Lambda\gamma,\; \Lambda\mu,\; -\tfrac{1}{2}\mathrm{vec}(\Lambda)]`

        (Gamma subordinator: :math:`p=\alpha`, :math:`a=2\beta`, :math:`b\to 0`).
        """
        _, _, mu_quad, gamma_quad, _ = self._precision_quantities()
        return self._assemble_natural_params(
            self.alpha - 1.0 - self.d / 2.0,
            -mu_quad,
            -(self.beta + gamma_quad),
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta) = \tfrac{1}{2}\log|\Sigma| + \log\Gamma(\alpha) - \alpha\log\beta + \mu^\top\Lambda\gamma`.

        Analytical — no Bessel function needed (Gamma subordinator).
        """
        from normix.mixtures.joint import JointNormalMixture
        (d, theta_1, _, theta_3, *_, log_det_Sigma, _, _,
         _, gamma_quad, mu_Lambda_gamma) = JointNormalMixture._parse_joint_theta(theta)

        alpha = theta_1 + 1.0 + d / 2.0
        beta = -theta_3 - gamma_quad

        return (0.5 * log_det_Sigma
                + jax.scipy.special.gammaln(alpha)
                - alpha * jnp.log(beta)
                + mu_Lambda_gamma)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta):
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, alpha=alpha, beta=beta)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointVarianceGamma":
        raise NotImplementedError("Use from_classical or m_step.")



class VarianceGamma(NormalMixture):
    """Marginal Variance Gamma distribution f(x)."""

    def __init__(self, joint: JointVarianceGamma):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta) -> "VarianceGamma":
        joint = JointVarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=alpha, beta=beta)
        return cls(joint)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        Marginal VG log-density (own formula, no GH delegation).

        .. math::

            f(x) \propto \left(\frac{q}{2c}\right)^{\nu/2}
            K_\nu\!\left(\sqrt{2qc}\right) \exp(\gamma^\top\Sigma^{-1}(x-\mu))

        where :math:`\nu = \alpha - d/2`,
        :math:`c = \beta + \tfrac{1}{2}\gamma^\top\Lambda\gamma`,
        :math:`q = (x-\mu)^\top\Lambda(x-\mu)`.
        """
        from normix.utils.bessel import log_kv

        j = self._joint
        d = j.d
        alpha = j.alpha
        beta = j.beta

        z, w, z2, w2, zw = j._quad_forms(x)
        q = z2
        gamma_quad = w2
        linear = zw

        c = beta + 0.5 * gamma_quad
        nu = alpha - d / 2.0

        log_det_sigma = j.log_det_sigma()

        log_C = (jnp.log(2.0)
                 - 0.5 * d * jnp.log(2.0 * jnp.pi)
                 - 0.5 * log_det_sigma
                 - jax.scipy.special.gammaln(alpha)
                 + alpha * jnp.log(beta))

        z_arg = jnp.sqrt(2.0 * q * c)
        log_K = log_kv(nu, z_arg)

        log_f = (log_C
                 + 0.5 * nu * jnp.log(q / (2.0 * c + LOG_EPS) + LOG_EPS)
                 + log_K
                 + linear)
        return log_f

    def _subordinator_expectations(self):
        j = self._joint
        E_log_Y = jax.scipy.special.digamma(j.alpha) - jnp.log(j.beta)
        E_inv_Y = j.beta / (j.alpha - 1.0)
        E_Y = j.alpha / j.beta
        return E_log_Y, E_inv_Y, E_Y

    def m_step_subordinator(self, eta, **kwargs):
        from normix.distributions.gamma import Gamma
        j = self._joint
        gamma_dist = Gamma.from_expectation(
            jnp.array([eta.E_log_Y, eta.E_Y]))
        joint_new = JointVarianceGamma(
            mu=j.mu, gamma=j.gamma, L_Sigma=j.L_Sigma,
            alpha=gamma_dist.alpha, beta=gamma_dist.beta,
        )
        return VarianceGamma(joint_new)

    def _build_rescaled(self, mu, gamma_new, L_new, scale):
        j = self._joint
        joint_new = JointVarianceGamma(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            alpha=j.alpha, beta=j.beta / scale,
        )
        return VarianceGamma(joint_new)

    def fit(self, X, *, algorithm='em', verbose=0, max_iter=200, tol=1e-3,
            regularization='none',
            e_step_backend='cpu', m_step_backend='cpu',
            m_step_method='newton'):
        """Fit VG using EM or MCECM.  Defaults to CPU E-step (faster than JAX
        vmap for the degenerate-GIG posterior arising from the Gamma subordinator)."""
        return super().fit(
            X, algorithm=algorithm,
            verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0)
