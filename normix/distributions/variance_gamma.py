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
        """
        θ = [α-1-d/2, -½μᵀΛμ, -(β+½γᵀΛγ), Λγ, Λμ, -½vec(Λ)]
        (Gamma subordinator: p=α, a=2β, b→0).
        """
        _, _, mu_quad, gamma_quad, _ = self._precision_quantities()
        return self._assemble_natural_params(
            self.alpha - 1.0 - self.d / 2.0,
            -mu_quad,
            -(self.beta + gamma_quad),
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        """
        ψ(θ) = ½ log|Σ| + log Γ(α) − α log β + μᵀΛγ
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
        """
        Marginal VG log-density (own formula, no GH delegation).

        f(x) = C * (q/(2c))^{nu/2} * K_nu(sqrt(2qc)) * exp(linear)
        where nu=alpha-d/2, c=beta+½γᵀΛγ, q=(x-μ)ᵀΛ(x-μ).
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

    def _m_step_subordinator(self, mu_new, gamma_new, L_new, gig_eta, **kwargs):
        from normix.distributions.gamma import Gamma
        gamma_dist = Gamma.from_expectation(jnp.array([gig_eta[0], gig_eta[2]]))
        joint_new = JointVarianceGamma(
            mu=mu_new, gamma=gamma_new, L_Sigma=L_new,
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

    def fit(self, X, *, verbose=0, max_iter=200, tol=1e-3,
            regularization='none',
            e_step_backend='cpu', m_step_backend='cpu',
            m_step_method='newton'):
        """Fit VG using EM.  Defaults to CPU E-step (faster than JAX vmap for
        the degenerate-GIG posterior arising from the Gamma subordinator)."""
        return super().fit(
            X, verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0)
