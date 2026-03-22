"""
Normal-Inverse Gamma (NInvG) distribution.

Special case of GH with GIG → InverseGamma subordinator (a → 0, p < 0).
Y ~ InverseGamma(α, β), i.e. GIG(p = −α, a → 0, b = 2β).

Stored: μ, γ, L_Σ (Cholesky of Σ), α (shape), β (rate) of InverseGamma.
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


class JointNormalInverseGamma(JointNormalMixture):
    """
    Joint f(x,y): X|Y ~ N(μ+γy, Σy), Y ~ InverseGamma(α, β).

    GIG limit: p = −α, a → 0, b = 2β.
    """

    alpha: jax.Array
    beta: jax.Array

    def __init__(self, mu, gamma, L_Sigma, alpha, beta):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'alpha', jnp.asarray(alpha, dtype=jnp.float64))
        object.__setattr__(self, 'beta', jnp.asarray(beta, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.inverse_gamma import InverseGamma
        return InverseGamma(alpha=self.alpha, beta=self.beta)

    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        from normix.distributions.inverse_gamma import InverseGamma
        alpha_ig = -p_eff
        beta_ig = b_eff / 2.0
        theta = jnp.array([-beta_ig, -(alpha_ig + 1.0)])
        return InverseGamma._log_partition_from_theta(theta)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """
        Posterior Y|X=x ~ GIG(-alpha-d/2, a_post, b_post)
        with a_post = γᵀΣ⁻¹γ, b_post = 2*beta + (x-μ)ᵀΣ⁻¹(x-μ).
        """
        from normix.distributions.generalized_inverse_gaussian import GIG
        d = self.d
        z, w, z2, w2, zw = self._quad_forms(x)

        p_post = -self.alpha - d / 2.0
        a_post = w2
        b_post = 2.0 * self.beta + z2

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
        return (-self.alpha - self.d / 2.0,
                w2,
                2.0 * self.beta + z2)

    def natural_params(self) -> jax.Array:
        """
        θ = [-(α+1)-d/2, -(β+½μᵀΛμ), -½γᵀΛγ, Λγ, Λμ, -½vec(Λ)]
        (InverseGamma subordinator: p=-α, a→0, b=2β).
        """
        _, _, mu_quad, gamma_quad, _ = self._precision_quantities()
        return self._assemble_natural_params(
            -(self.alpha + 1.0) - self.d / 2.0,
            -(self.beta + mu_quad),
            -gamma_quad,
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        """
        ψ(θ) = ½ log|Σ| + log Γ(α) − α log β + μᵀΛγ
        Analytical — no Bessel function needed (InverseGamma subordinator).
        """
        from normix.mixtures.joint import JointNormalMixture
        (d, theta_1, theta_2, _, *_, log_det_Sigma, _, _,
         mu_quad, _, mu_Lambda_gamma) = JointNormalMixture._parse_joint_theta(theta)

        alpha = -(theta_1 + d / 2.0) - 1.0
        beta = -theta_2 - mu_quad

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
    def from_natural(cls, theta):
        raise NotImplementedError("Use from_classical or m_step.")



class NormalInverseGamma(NormalMixture):
    """Marginal Normal-Inverse Gamma distribution f(x)."""

    def __init__(self, joint: JointNormalInverseGamma):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta):
        joint = JointNormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=alpha, beta=beta)
        return cls(joint)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """
        Marginal NInvG log-density (own formula, no GH delegation).

        GIG params: p=-α, a=γᵀΛγ, b=2β+Q(x).
        The normalising integral is 2(b/a)^{p/2} K_p(sqrt(ab)).
        For the symmetric case (γ≈0, a→0), uses Γ-function closed form.
        """
        from normix.utils.bessel import log_kv

        j = self._joint
        d = j.d
        alpha = j.alpha
        beta = j.beta

        z, w, z2, w2, zw = j._quad_forms(x)
        q = z2
        a_gig = w2
        b_gig = 2.0 * beta + q
        p_gig = -(alpha + d / 2.0)
        linear = zw

        log_det_sigma = j.log_det_sigma()

        log_C = (-0.5 * d * jnp.log(2.0 * jnp.pi)
                 - 0.5 * log_det_sigma
                 - jax.scipy.special.gammaln(alpha)
                 + alpha * jnp.log(beta))

        sqrt_ab = jnp.sqrt(a_gig * b_gig)
        log_bessel = log_kv(p_gig, sqrt_ab)
        log_integral = (jnp.log(2.0)
                        + 0.5 * p_gig * jnp.log((b_gig + LOG_EPS) / (a_gig + LOG_EPS))
                        + log_bessel)

        log_f = log_C + linear + log_integral
        return log_f

    def _m_step_subordinator(self, mu_new, gamma_new, L_new, gig_eta, **kwargs):
        from normix.distributions.inverse_gamma import InverseGamma
        ig_eta = jnp.array([-gig_eta[1], gig_eta[0]])
        ig_new = InverseGamma.from_expectation(ig_eta)
        joint_new = JointNormalInverseGamma(
            mu=mu_new, gamma=gamma_new, L_Sigma=L_new,
            alpha=ig_new.alpha, beta=ig_new.beta,
        )
        return NormalInverseGamma(joint_new)

    def _build_rescaled(self, mu, gamma_new, L_new, scale):
        j = self._joint
        joint_new = JointNormalInverseGamma(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            alpha=j.alpha, beta=j.beta * scale,
        )
        return NormalInverseGamma(joint_new)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=3.0, beta=1.0)
