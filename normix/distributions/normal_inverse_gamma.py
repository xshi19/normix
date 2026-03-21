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
        d = self.d
        z_mu = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.mu, lower=True)
        z_gamma = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)
        Lambda_mu = jax.scipy.linalg.solve_triangular(self.L_Sigma.T, z_mu, lower=False)
        Lambda_gamma = jax.scipy.linalg.solve_triangular(self.L_Sigma.T, z_gamma, lower=False)

        mu_quad = 0.5 * jnp.dot(self.mu, Lambda_mu)
        gamma_quad = 0.5 * jnp.dot(self.gamma, Lambda_gamma)

        theta_1 = -(self.alpha + 1.0) - d / 2.0
        theta_2 = -(self.beta + mu_quad)
        theta_3 = -gamma_quad

        L_inv = jax.scipy.linalg.solve_triangular(
            self.L_Sigma, jnp.eye(d, dtype=jnp.float64), lower=True)
        Lambda = L_inv.T @ L_inv

        return jnp.concatenate([
            jnp.array([theta_1, theta_2, theta_3]),
            Lambda_gamma,
            Lambda_mu,
            (-0.5 * Lambda).ravel(),
        ])

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        """
        ψ(θ) = ½ log|Σ| + log Γ(α) − α log β + μᵀΛγ
        Analytical — no Bessel function needed (InverseGamma subordinator).
        """
        n = theta.shape[0]
        d = int(-1 + (1 + 4 * (n - 3)) ** 0.5) // 2

        theta_1 = theta[0]
        theta_2 = theta[1]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        theta_6 = theta[3 + 2 * d:].reshape(d, d)

        Lambda = -2.0 * theta_6
        Lambda = 0.5 * (Lambda + Lambda.T)

        _sign, log_det_Lambda = jnp.linalg.slogdet(Lambda)
        log_det_Sigma = -log_det_Lambda

        mu = jnp.linalg.solve(Lambda, theta_5)

        mu_quad = 0.5 * jnp.dot(mu, theta_5)
        alpha = -(theta_1 + d / 2.0) - 1.0
        beta = -theta_2 - mu_quad

        mu_Lambda_gamma = jnp.dot(mu, theta_4)

        psi = (0.5 * log_det_Sigma
               + jax.scipy.special.gammaln(alpha)
               - alpha * jnp.log(beta)
               + mu_Lambda_gamma)
        return psi

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

        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L_Sigma)))

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
            mu=mu_new, gamma=gamma_new, L_Sigma=L_new,
            alpha=ig_new.alpha, beta=ig_new.beta,
        )
        return NormalInverseGamma(joint_new)

    def regularize_det_sigma_one(self) -> "NormalInverseGamma":
        j = self._joint
        d = j.d
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L_Sigma)))
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)
        L_new = j.L_Sigma / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        beta_new = j.beta * scale
        joint_new = JointNormalInverseGamma(
            mu=j.mu, gamma=gamma_new, L_Sigma=L_new,
            alpha=j.alpha, beta=beta_new,
        )
        return NormalInverseGamma(joint_new)

    def marginal_log_likelihood(self, X):
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))

    @classmethod
    def fit(
        cls,
        X: jax.Array,
        *,
        key: jax.Array,
        max_iter: int = 200,
        tol: float = 1e-6,
        regularization: str = 'det_sigma_one',
        n_init: int = 1,
    ) -> "NormalInverseGamma":
        """Fit NInvG distribution to data using EM."""
        from normix.fitting.em import BatchEMFitter
        X = jnp.asarray(X, dtype=jnp.float64)
        fitter = BatchEMFitter(max_iter=max_iter, tol=tol,
                               regularization=regularization)
        best_model = None
        best_ll = -jnp.inf
        keys = jax.random.split(key, n_init)
        for k in keys:
            model = cls._initialize(X, k)
            fitted = fitter.fit(model, X)
            ll = fitted.marginal_log_likelihood(X)
            if best_model is None or float(ll) > float(best_ll):
                best_model = fitted
                best_ll = ll
        return best_model

    @classmethod
    def _initialize(cls, X: jax.Array, key: jax.Array) -> "NormalInverseGamma":
        """Moment-based initialization with random perturbation."""
        X = jnp.asarray(X, dtype=jnp.float64)
        n, d = X.shape
        mu = jnp.mean(X, axis=0)
        X_centered = X - mu
        sigma_emp = (X_centered.T @ X_centered) / n + 1e-4 * jnp.eye(d)
        key1, _key2 = jax.random.split(key)
        gamma = 0.01 * jax.random.normal(key1, (d,), dtype=jnp.float64)
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma_emp,
            alpha=3.0, beta=1.0,
        )
