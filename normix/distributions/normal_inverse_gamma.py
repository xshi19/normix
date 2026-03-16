"""
Normal-Inverse Gamma (NInvG) distribution.

Special case of GH with GIG → InverseGamma subordinator (a → 0, p < 0).
Y ~ InverseGamma(alpha, beta), i.e. GIG(p=-alpha, a→0, b=2*beta).

Stored: mu, gamma, L (Cholesky of Σ), alpha (shape), beta (rate) of InverseGamma.
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture

jax.config.update("jax_enable_x64", True)

_EPS = 1e-30


class JointNormalInverseGamma(JointNormalMixture):
    """
    Joint f(x,y): X|Y~N(μ+γy, Σy), Y~InverseGamma(alpha, beta).

    GIG limit: p=-alpha, a→0, b=2*beta.
    """

    alpha: jax.Array
    beta: jax.Array

    def __init__(self, mu, gamma, L, alpha, beta):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L', jnp.asarray(L, dtype=jnp.float64))
        object.__setattr__(self, 'alpha', jnp.asarray(alpha, dtype=jnp.float64))
        object.__setattr__(self, 'beta', jnp.asarray(beta, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.inverse_gamma import InverseGamma
        return InverseGamma(alpha=self.alpha, beta=self.beta)

    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        from normix.distributions.inverse_gamma import InverseGamma
        dummy = InverseGamma(alpha=jnp.ones(()), beta=jnp.ones(()))
        alpha_ig = -p_eff
        beta_ig = b_eff / 2.0
        theta = jnp.array([-beta_ig, -(alpha_ig + 1.0)])
        return dummy._log_partition_from_theta(theta)

    def _conditional_expectations_impl(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """
        Posterior Y|X=x ~ GIG(-alpha-d/2, a_post, b_post)
        with a_post = γᵀΣ⁻¹γ, b_post = 2*beta + (x-μ)ᵀΣ⁻¹(x-μ).
        """
        from normix.distributions.gig import GIG
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
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        j = JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L=self.L,
            p=-self.alpha, a=jnp.array(_EPS), b=2.0 * self.beta,
        )
        return j.natural_params()

    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        dummy = JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L=self.L,
            p=-self.alpha, a=jnp.array(_EPS), b=2.0 * self.beta,
        )
        return dummy._log_partition_from_theta(theta)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta):
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L=L, alpha=alpha, beta=beta)

    @classmethod
    def from_natural(cls, theta):
        raise NotImplementedError("Use from_classical or m_step.")

    @classmethod
    def _dummy_instance(cls):
        d = 1
        return cls(mu=jnp.zeros(d), gamma=jnp.zeros(d), L=jnp.eye(d),
                   alpha=jnp.ones(()), beta=jnp.ones(()))


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
        from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
        j = self._joint
        gh = GeneralizedHyperbolic.from_classical(
            mu=j.mu, gamma=j.gamma, sigma=j.sigma(),
            p=-j.alpha, a=jnp.array(_EPS), b=2.0 * j.beta,
        )
        return gh.log_prob(x)

    def m_step(self, X, expectations) -> "NormalInverseGamma":
        from normix.distributions.inverse_gamma import InverseGamma

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

        ig_eta = jnp.array([-mean_E_inv_Y, jnp.mean(E_log_Y)])
        ig_new = InverseGamma.from_expectation(ig_eta)

        joint_new = JointNormalInverseGamma(
            mu=mu_new, gamma=gamma_new, L=L_new,
            alpha=ig_new.alpha, beta=ig_new.beta,
        )
        return NormalInverseGamma(joint_new)

    def regularize_det_sigma_one(self) -> "NormalInverseGamma":
        j = self._joint
        d = j.d
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L)))
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)
        L_new = j.L / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        beta_new = j.beta * scale
        joint_new = JointNormalInverseGamma(
            mu=j.mu, gamma=gamma_new, L=L_new,
            alpha=j.alpha, beta=beta_new,
        )
        return NormalInverseGamma(joint_new)

    def marginal_log_likelihood(self, X):
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))
