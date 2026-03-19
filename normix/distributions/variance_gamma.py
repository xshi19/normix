"""
Variance Gamma (VG) distribution.

Special case of GH with GIG → Gamma subordinator (b → 0, p > 0).
Y ~ Gamma(α, β), i.e. GIG(p = α, a = 2β, b → 0).

Stored: μ, γ, L_Σ (Cholesky of Σ), α (shape), β (rate) of Gamma.
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
    """
    Joint f(x,y): X|Y ~ N(μ+γy, Σy), Y ~ Gamma(α, β).

    GIG limit: p = α, a = 2β, b → 0.
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

    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        from normix.distributions.gamma import Gamma
        theta = jnp.array([p_eff - 1.0, -a_eff / 2.0])
        return Gamma._log_partition_from_theta(theta)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """
        Posterior Y|X=x ~ GIG(alpha-d/2, beta_post_a, beta_post_b)
        where beta_post_a = 2*beta + γᵀΣ⁻¹γ,  beta_post_b = (x-μ)ᵀΣ⁻¹(x-μ)

        Actually for Gamma subordinator (b→0), the posterior is GIG:
          p_post = alpha - d/2,  a_post = 2β + γᵀΣ⁻¹γ,  b_post = (x-μ)ᵀΣ⁻¹(x-μ)
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
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        j = JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L_Sigma=self.L_Sigma,
            p=self.alpha, a=2.0 * self.beta, b=jnp.array(LOG_EPS),
        )
        return j.natural_params()

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        return JointGeneralizedHyperbolic._log_partition_from_theta(theta)

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
        """
        Marginal VG log-density via GH formula with b→0.
        VG: p=alpha, a=2*beta, b→0.
        Use GeneralizedHyperbolic.log_prob with GH params.
        """
        from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
        j = self._joint
        gh = GeneralizedHyperbolic.from_classical(
            mu=j.mu, gamma=j.gamma, sigma=j.sigma(),
            p=j.alpha, a=2.0 * j.beta, b=jnp.array(LOG_EPS),
        )
        return gh.log_prob(x)

    def m_step(self, X, expectations) -> "VarianceGamma":
        from normix.distributions.gamma import Gamma

        X = jnp.asarray(X, dtype=jnp.float64)
        j = self._joint

        E_log_Y = expectations['E_log_Y']
        E_inv_Y = expectations['E_inv_Y']
        E_Y = expectations['E_Y']

        mean_E_inv_Y = jnp.mean(E_inv_Y)
        mean_E_Y = jnp.mean(E_Y)
        E_X = jnp.mean(X, axis=0)
        E_X_inv_Y = jnp.mean(X * E_inv_Y[:, None], axis=0)
        E_XXT_inv_Y = jnp.mean(
            jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y), axis=0)

        mu_new, gamma_new, L_new = JointNormalMixture._mstep_normal_params(
            E_X, E_X_inv_Y, E_XXT_inv_Y, mean_E_inv_Y, mean_E_Y)

        gig_eta = jnp.array([jnp.mean(E_log_Y), mean_E_inv_Y, mean_E_Y])
        gamma_dist = Gamma.from_expectation(jnp.array([gig_eta[0], gig_eta[2]]))

        joint_new = JointVarianceGamma(
            mu=mu_new, gamma=gamma_new, L_Sigma=L_new,
            alpha=gamma_dist.alpha, beta=gamma_dist.beta,
        )
        return VarianceGamma(joint_new)

    def regularize_det_sigma_one(self) -> "VarianceGamma":
        j = self._joint
        d = j.d
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L_Sigma)))
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)
        L_new = j.L_Sigma / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        beta_new = j.beta / scale
        joint_new = JointVarianceGamma(
            mu=j.mu, gamma=gamma_new, L_Sigma=L_new,
            alpha=j.alpha, beta=beta_new,
        )
        return VarianceGamma(joint_new)

    def marginal_log_likelihood(self, X):
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))
