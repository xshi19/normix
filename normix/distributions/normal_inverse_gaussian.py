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
        """
        θ = [-3/2-d/2, -(b+½μᵀΛμ), -(a+½γᵀΛγ), Λγ, Λμ, -½vec(Λ)]
        where p=-½, a=λ/μ_IG², b=λ.
        """
        d = self.d
        a_ig = self.lam / (self.mu_ig ** 2)

        z_mu = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.mu, lower=True)
        z_gamma = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)
        Lambda_mu = jax.scipy.linalg.solve_triangular(self.L_Sigma.T, z_mu, lower=False)
        Lambda_gamma = jax.scipy.linalg.solve_triangular(self.L_Sigma.T, z_gamma, lower=False)

        mu_quad = 0.5 * jnp.dot(self.mu, Lambda_mu)
        gamma_quad = 0.5 * jnp.dot(self.gamma, Lambda_gamma)

        theta_1 = -1.5 - d / 2.0
        theta_2 = -(self.lam + mu_quad)
        theta_3 = -(a_ig + gamma_quad)

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
        ψ(θ) for Joint NIG.  Uses K_{-1/2}(z) = √(π/(2z)) e^{-z} —
        no Bessel function evaluation needed.
        """
        n = theta.shape[0]
        d = int(-1 + (1 + 4 * (n - 3)) ** 0.5) // 2

        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        theta_6 = theta[3 + 2 * d:].reshape(d, d)

        Lambda = -2.0 * theta_6
        Lambda = 0.5 * (Lambda + Lambda.T)

        _sign, log_det_Lambda = jnp.linalg.slogdet(Lambda)
        log_det_Sigma = -log_det_Lambda

        mu = jnp.linalg.solve(Lambda, theta_5)
        gamma = jnp.linalg.solve(Lambda, theta_4)

        mu_quad = 0.5 * jnp.dot(mu, theta_5)
        gamma_quad = 0.5 * jnp.dot(gamma, theta_4)
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        mu_Lambda_gamma = jnp.dot(mu, theta_4)

        p = -0.5
        sqrt_ab = jnp.sqrt(a * b)
        log_K = 0.5 * jnp.log(jnp.pi / (2.0 * sqrt_ab + LOG_EPS)) - sqrt_ab

        psi = (0.5 * log_det_Sigma + jnp.log(2.0) + log_K
               + 0.5 * p * jnp.log((b + LOG_EPS) / (a + LOG_EPS))
               + mu_Lambda_gamma)
        return psi

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
        """
        Marginal NIG log-density.

        Uses K_{-1/2}(z) = sqrt(pi/(2z)) exp(-z) for the normalisation,
        leaving only one log_kv call at order nu = -1/2 - d/2.
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

        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L_Sigma)))

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
    ) -> "NormalInverseGaussian":
        """Fit NIG distribution to data using EM."""
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
    def _initialize(cls, X: jax.Array, key: jax.Array) -> "NormalInverseGaussian":
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
            mu_ig=1.0, lam=1.0,
        )
