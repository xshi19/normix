"""
Normal-Inverse Gaussian (NIG) distribution.

Special case of GH with GIG → InverseGaussian subordinator (p = −½).
Y ~ InverseGaussian(μ_IG, λ), i.e. GIG(p = −½, a = λ/μ_IG², b = λ).

Stored: μ, γ, L_Σ (Cholesky of Σ), μ_IG (IG mean), λ (IG shape).
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


class JointNormalInverseGaussian(JointNormalMixture):
    """
    Joint f(x,y): X|Y ~ N(μ+γy, Σy), Y ~ InverseGaussian(μ_IG, λ).

    Stored: μ_IG (IG mean) and λ (IG shape) directly.
    GIG params: p = −½, a = λ/μ_IG², b = λ.
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

    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        from normix.distributions.inverse_gaussian import InverseGaussian
        lam_p = b_eff     # b = lam
        mu_p = jnp.sqrt(b_eff / a_eff)
        theta = jnp.array([-lam_p / (2.0 * mu_p**2), -lam_p / 2.0])
        return InverseGaussian._log_partition_from_theta(theta)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """
        Posterior Y|X=x ~ GIG(-1/2-d/2, a_post, b_post)
        where a_post = lam/mu_ig² + γᵀΣ⁻¹γ,  b_post = lam + (x-μ)ᵀΣ⁻¹(x-μ).
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
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        a_ig = self.lam / (self.mu_ig**2)
        j = JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L_Sigma=self.L_Sigma,
            p=jnp.array(-0.5), a=a_ig, b=self.lam,
        )
        return j.natural_params()

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        return JointGeneralizedHyperbolic._log_partition_from_theta(theta)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, mu_ig, lam):
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, mu_ig=mu_ig, lam=lam)

    @classmethod
    def from_natural(cls, theta):
        raise NotImplementedError("Use from_classical or m_step.")



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
        from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
        j = self._joint
        a_ig = j.lam / (j.mu_ig**2)
        gh = GeneralizedHyperbolic.from_classical(
            mu=j.mu, gamma=j.gamma, sigma=j.sigma(),
            p=jnp.array(-0.5), a=a_ig, b=j.lam,
        )
        return gh.log_prob(x)

    def m_step(self, X, expectations) -> "NormalInverseGaussian":
        from normix.distributions.inverse_gaussian import InverseGaussian

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

        ig_new = InverseGaussian.from_expectation(
            jnp.array([mean_E_Y, mean_E_inv_Y]))

        joint_new = JointNormalInverseGaussian(
            mu=mu_new, gamma=gamma_new, L_Sigma=L_new,
            mu_ig=ig_new.mu, lam=ig_new.lam,
        )
        return NormalInverseGaussian(joint_new)

    def regularize_det_sigma_one(self) -> "NormalInverseGaussian":
        j = self._joint
        d = j.d
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L_Sigma)))
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)
        L_new = j.L_Sigma / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        # Y → Y/scale: mu_ig → mu_ig/scale, lam → lam/scale²
        mu_ig_new = j.mu_ig / scale
        lam_new = j.lam / scale
        joint_new = JointNormalInverseGaussian(
            mu=j.mu, gamma=gamma_new, L_Sigma=L_new,
            mu_ig=mu_ig_new, lam=lam_new,
        )
        return NormalInverseGaussian(joint_new)

    def marginal_log_likelihood(self, X):
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))
