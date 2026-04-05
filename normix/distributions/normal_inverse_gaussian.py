"""
Normal-Inverse Gaussian (NIG) distribution.

Special case of GH with GIG → InverseGaussian subordinator (:math:`p = -1/2`).
:math:`Y \\sim \\mathrm{InvGaussian}(\\mu_{IG}, \\lambda)`,
i.e. GIG(:math:`p = -1/2`, :math:`a = \\lambda/\\mu_{IG}^2`, :math:`b = \\lambda`).

Stored: :math:`\\mu`, :math:`\\gamma`, :math:`L_\\Sigma` (Cholesky of :math:`\\Sigma`),
:math:`\\mu_{IG}` (IG mean), :math:`\\lambda` (IG shape).
"""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture


from normix.utils.constants import LOG_EPS


class JointNormalInverseGaussian(JointNormalMixture):
    r"""
    Joint :math:`f(x,y)`: :math:`X\mid Y \sim \mathcal{N}(\mu+\gamma y, \Sigma y)`,
    :math:`Y \sim \mathrm{InvGaussian}(\mu_{IG}, \lambda)`.

    Stored: :math:`\mu_{IG}` (IG mean) and :math:`\lambda` (IG shape) directly.
    GIG params: :math:`p = -1/2`, :math:`a = \lambda/\mu_{IG}^2`, :math:`b = \lambda`.
    """

    mu_ig: jax.Array   # IG mean parameter
    lam: jax.Array     # IG shape parameter

    def __init__(self, mu, gamma, L_Sigma, mu_ig, lam):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'mu_ig', jnp.asarray(mu_ig, dtype=jnp.float64))
        object.__setattr__(self, 'lam', jnp.asarray(lam, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.inverse_gaussian import InverseGaussian
        return InverseGaussian(mu=self.mu_ig, lam=self.lam)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        r"""
        Posterior :math:`Y\mid X=x \sim \mathrm{GIG}(-1/2-d/2, a_{\mathrm{post}}, b_{\mathrm{post}})`:

        .. math::

            a_{\mathrm{post}} = \lambda/\mu_{IG}^2 + \gamma^\top\Sigma^{-1}\gamma, \quad
            b_{\mathrm{post}} = \lambda + (x-\mu)^\top\Sigma^{-1}(x-\mu)
        """
        from normix.distributions.generalized_inverse_gaussian import GIG
        d = self.d
        z, w, z2, w2, zw = self._quad_forms(x)

        # GIG params from IG subordinator
        a_ig = self.lam / (self.mu_ig**2)
        b_ig = self.lam

        p_post = -0.5 - d / 2.0
        a_post = a_ig + w2
        b_post = b_ig + z2

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
        a_ig = self.lam / (self.mu_ig ** 2)
        return (-0.5 - self.d / 2.0,
                a_ig + w2,
                self.lam + z2)

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [-3/2-d/2,\; -(\lambda/2+\tfrac{1}{2}\mu^\top\Lambda\mu),\;
        -(\lambda/(2\mu_{IG}^2)+\tfrac{1}{2}\gamma^\top\Lambda\gamma),\;
        \Lambda\gamma,\; \Lambda\mu,\; -\tfrac{1}{2}\mathrm{vec}(\Lambda)]`

        where :math:`p=-1/2`, :math:`a=\lambda/\mu_{IG}^2`, :math:`b=\lambda`, aligned
        with GIG natural parameters on :math:`[\log y,\,1/y,\,y]`.
        """
        a_ig = self.lam / (self.mu_ig ** 2)
        _, _, mu_quad, gamma_quad, _ = self._precision_quantities()
        return self._assemble_natural_params(
            -1.5 - self.d / 2.0,
            -(self.lam / 2.0 + mu_quad),
            -(a_ig / 2.0 + gamma_quad),
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta)` for Joint NIG. Uses :math:`K_{-1/2}(z) = \sqrt{\pi/(2z)}\,e^{-z}` —
        no Bessel function evaluation needed.
        """
        from normix.mixtures.joint import JointNormalMixture
        (d, _, theta_2, theta_3, *_, log_det_Sigma, _, _,
         mu_quad, gamma_quad, mu_Lambda_gamma) = JointNormalMixture._parse_joint_theta(theta)

        b = 2.0 * (-theta_2 - mu_quad)
        a = 2.0 * (-theta_3 - gamma_quad)

        p = -0.5
        sqrt_ab = jnp.sqrt(a * b)
        log_K = 0.5 * jnp.log(jnp.pi / (2.0 * sqrt_ab + LOG_EPS)) - sqrt_ab

        return (0.5 * log_det_Sigma + jnp.log(2.0) + log_K
                + 0.5 * p * jnp.log((b + LOG_EPS) / (a + LOG_EPS))
                + mu_Lambda_gamma)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, mu_ig, lam):
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, mu_ig=mu_ig, lam=lam)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointNormalInverseGaussian":
        r"""Recover classical parameters from :math:`\theta`.

        From :math:`b = -2\theta_2 - 2\mu_{\mathrm{quad}} = \lambda` and
        :math:`a = -2\theta_3 - 2\gamma_{\mathrm{quad}} = \lambda/\mu_{IG}^2`,
        so :math:`\mu_{IG} = \sqrt{b/a}`.
        """
        from normix.mixtures.joint import JointNormalMixture
        (d, mu, gamma, L_Sigma,
         _, theta_2, theta_3, mu_quad, gamma_quad,
         ) = JointNormalMixture._recover_normal_params(jnp.asarray(theta, dtype=jnp.float64))
        lam = 2.0 * (-theta_2 - mu_quad)           # b = λ
        a_ig = 2.0 * (-theta_3 - gamma_quad)       # a = λ/μ_IG²
        mu_ig = jnp.sqrt(lam / jnp.maximum(a_ig, 1e-30))
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma, mu_ig=mu_ig, lam=lam)



class NormalInverseGaussian(NormalMixture):
    """Marginal Normal-Inverse Gaussian distribution f(x)."""

    def __init__(self, joint: JointNormalInverseGaussian):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, mu_ig, lam):
        joint = JointNormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=mu_ig, lam=lam)
        return cls(joint)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        Marginal NIG log-density.

        Uses :math:`K_{-1/2}(z) = \sqrt{\pi/(2z)}\,e^{-z}` for the normalisation,
        leaving only one :math:`\log K_\nu` call at order :math:`\nu = -1/2 - d/2`.
        """
        from normix.utils.bessel import log_kv

        j = self._joint
        d = j.d
        p = -0.5
        a = j.lam / (j.mu_ig ** 2)
        b = j.lam

        z, w, z2, w2, zw = j._quad_forms(x)
        Q = z2
        A = a + w2
        nu = p - d / 2.0

        log_det_sigma = j.log_det_sigma()

        sqrt_ab = jnp.sqrt(a * b)
        log_K_p = 0.5 * jnp.log(jnp.pi / (2.0 * sqrt_ab + LOG_EPS)) - sqrt_ab

        sqrt_A_Qb = jnp.sqrt(A * (Q + b))

        log_f = (
            -0.5 * d * jnp.log(2.0 * jnp.pi)
            - 0.5 * log_det_sigma
            + 0.5 * p * (jnp.log(a + LOG_EPS) - jnp.log(b + LOG_EPS))
            - log_K_p
            + 0.5 * (d / 2.0 - p) * jnp.log(A / (Q + b + LOG_EPS))
            + log_kv(nu, sqrt_A_Qb)
            + zw
        )
        return log_f

    def _subordinator_expectations(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        from normix.distributions.generalized_inverse_gaussian import GIG
        j = self._joint
        gig = GIG(p=jnp.float64(-0.5), a=j.lam / j.mu_ig**2, b=j.lam)
        eta = gig.expectation_params()
        return eta[0], eta[1], eta[2]

    def m_step_subordinator(self, eta, **kwargs) -> "NormalInverseGaussian":
        from normix.distributions.inverse_gaussian import InverseGaussian
        j = self._joint
        ig_new = InverseGaussian.from_expectation(
            jnp.array([eta.E_Y, eta.E_inv_Y]))
        joint_new = JointNormalInverseGaussian(
            mu=j.mu, gamma=j.gamma, L_Sigma=j.L_Sigma,
            mu_ig=ig_new.mu, lam=ig_new.lam,
        )
        return NormalInverseGaussian(joint_new)

    def _build_rescaled(self, mu, gamma_new, L_new, scale) -> "NormalInverseGaussian":
        j = self._joint
        joint_new = JointNormalInverseGaussian(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            mu_ig=j.mu_ig / scale, lam=j.lam / scale,
        )
        return NormalInverseGaussian(joint_new)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.0)
