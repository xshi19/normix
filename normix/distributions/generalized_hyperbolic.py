"""
Generalized Hyperbolic (GH) distribution.

Joint: X|Y ~ N(μ+γy, Σy), Y ~ GIG(p, a, b)
Marginal: GH(μ, γ, Σ, p, a, b)

Marginal log-density (closed form via Bessel functions):
  Let Q(x) = (x-μ)ᵀΣ⁻¹(x-μ),  A = a + γᵀΣ⁻¹γ
  log f(x) = const
    + (p - d/2) log(Q(x) + b) / 2 - p/2 log A
    + log K_{p-d/2}(√(A(Q(x)+b)))
    - log K_p(√(ab))
    + (x-μ)ᵀΣ⁻¹γ · ... (skewness term)

Full formula:
  f(x) = C(p,a,b,Σ) · (A/(Q(x)+b))^{(p-d/2)/2}
          · K_{p-d/2}(√(A(Q(x)+b)))
          · exp(γᵀΣ⁻¹(x-μ))

  C = (2π)^{-d/2} |Σ|^{-1/2} (a/b)^{p/2} / (2 K_p(√(ab))) · A^{(d/2-p)/2} · ... 
      (see below for precise formula)

Posterior Y|X = x ~ GIG(p - d/2, a + γᵀΣ⁻¹γ, b + (x-μ)ᵀΣ⁻¹(x-μ))
→ conditional expectations computed from GIG.
"""
from __future__ import annotations

from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from normix._bessel import log_kv
from normix.exponential_family import ExponentialFamily
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture

jax.config.update("jax_enable_x64", True)

_EPS = 1e-30


# ============================================================================
# Joint GH distribution
# ============================================================================

class JointGeneralizedHyperbolic(JointNormalMixture):
    """
    Joint f(x,y): X|Y~N(μ+γy, Σy), Y~GIG(p,a,b).

    Stored: mu, gamma, L (from JointNormalMixture) + p, a, b (GIG parameters).
    """

    p: jax.Array
    a: jax.Array
    b: jax.Array

    def __init__(self, mu, gamma, L, p, a, b):
        # eqx.Module fields must be set via object.__setattr__ before freeze
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L', jnp.asarray(L, dtype=jnp.float64))
        object.__setattr__(self, 'p', jnp.asarray(p, dtype=jnp.float64))
        object.__setattr__(self, 'a', jnp.asarray(a, dtype=jnp.float64))
        object.__setattr__(self, 'b', jnp.asarray(b, dtype=jnp.float64))

    # ------------------------------------------------------------------
    # JointNormalMixture abstract methods
    # ------------------------------------------------------------------

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.gig import GIG
        return GIG(p=self.p, a=self.a, b=self.b)

    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        from normix.distributions.gig import GIG
        dummy = GIG(p=jnp.ones(()), a=jnp.ones(()), b=jnp.ones(()))
        theta = jnp.array([p_eff - 1.0, -b_eff / 2.0, -a_eff / 2.0])
        return dummy._log_partition_from_theta(theta)

    def _conditional_expectations_impl(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """
        Posterior Y|X=x ~ GIG(p-d/2, a+γᵀΣ⁻¹γ, b+(x-μ)ᵀΣ⁻¹(x-μ)).
        """
        from normix.distributions.gig import GIG

        d = self.d
        z, w, z2, w2, zw = self._quad_forms(x)
        p_post = self.p - d / 2.0
        a_post = self.a + w2
        b_post = self.b + z2

        gig = GIG(p=p_post, a=a_post, b=b_post)
        eta = gig.expectation_params()
        return {
            'E_log_Y': eta[0],
            'E_inv_Y': eta[1],
            'E_Y': eta[2],
        }

    # ------------------------------------------------------------------
    # ExponentialFamily: natural params / log partition
    # ------------------------------------------------------------------

    def natural_params(self) -> jax.Array:
        """
        θ = [p-1-d/2, -(b+½μᵀΣ⁻¹μ), -(a+½γᵀΣ⁻¹γ), Σ⁻¹γ, Σ⁻¹μ, -½vec(Σ⁻¹)]
        """
        d = self.d
        # Σ⁻¹ = (LLᵀ)⁻¹ = L⁻ᵀ L⁻¹
        # L⁻¹ μ = z_mu, L⁻¹ γ = z_gamma
        z_mu = jax.scipy.linalg.solve_triangular(self.L, self.mu, lower=True)
        z_gamma = jax.scipy.linalg.solve_triangular(self.L, self.gamma, lower=True)

        # Σ⁻¹μ = Lᵀ⁻¹ L⁻¹ μ
        Lambda_mu = jax.scipy.linalg.solve_triangular(self.L.T, z_mu, lower=False)
        Lambda_gamma = jax.scipy.linalg.solve_triangular(self.L.T, z_gamma, lower=False)

        mu_quad = 0.5 * jnp.dot(self.mu, Lambda_mu)
        gamma_quad = 0.5 * jnp.dot(self.gamma, Lambda_gamma)

        theta_1 = self.p - 1.0 - d / 2.0
        theta_2 = -(self.b + mu_quad)
        theta_3 = -(self.a + gamma_quad)

        # Σ⁻¹: LLᵀ → Σ⁻¹ = L⁻ᵀ L⁻¹
        L_inv = jax.scipy.linalg.solve_triangular(
            self.L, jnp.eye(d, dtype=jnp.float64), lower=True)
        Lambda = L_inv.T @ L_inv   # Σ⁻¹

        return jnp.concatenate([
            jnp.array([theta_1, theta_2, theta_3]),
            Lambda_gamma,           # θ₄ = Σ⁻¹γ
            Lambda_mu,              # θ₅ = Σ⁻¹μ
            (-0.5 * Lambda).ravel(), # θ₆ = -½vec(Σ⁻¹)
        ])

    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        """
        ψ = ψ_GIG(p, a, b) + ½log|Σ| + μᵀΣ⁻¹γ

        Recover p,a,b,μ,γ,Σ from theta.
        """
        from normix.distributions.gig import GIG

        d = self.d
        theta_1 = theta[0]
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]           # Σ⁻¹γ
        theta_5 = theta[3 + d:3 + 2 * d]   # Σ⁻¹μ
        theta_6 = theta[3 + 2 * d:].reshape(d, d)  # -½Σ⁻¹

        # Σ⁻¹ = -2 θ₆
        Lambda = -2.0 * theta_6
        # Symmetrize for numerical stability
        Lambda = 0.5 * (Lambda + Lambda.T)

        # μ = Σ θ₅ = Λ⁻¹ θ₅ ... but we need det(Σ) = det(Λ⁻¹)
        # log|Σ| = -log|Λ|
        sign, log_det_Lambda = jnp.linalg.slogdet(Lambda)
        log_det_Sigma = -log_det_Lambda

        # μ and γ from Λμ = θ₅, Λγ = θ₄
        # For log partition we only need μᵀΛγ = μᵀ θ₄
        # Recover μ from Λμ = θ₅: μ = Λ⁻¹ θ₅
        mu = jnp.linalg.solve(Lambda, theta_5)
        gamma = jnp.linalg.solve(Lambda, theta_4)

        # p, a, b
        mu_quad = 0.5 * jnp.dot(mu, theta_5)
        gamma_quad = 0.5 * jnp.dot(gamma, theta_4)
        p = theta_1 + 1.0 + d / 2.0
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        # μᵀΛγ = μᵀ θ₄
        mu_Lambda_gamma = jnp.dot(mu, theta_4)

        gig_theta = jnp.array([p - 1.0, -b / 2.0, -a / 2.0])
        dummy = GIG(p=jnp.ones(()), a=jnp.ones(()), b=jnp.ones(()))
        psi_gig = dummy._log_partition_from_theta(gig_theta)

        return psi_gig + 0.5 * log_det_Sigma + mu_Lambda_gamma

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, p, a, b):
        """Construct from classical parameters."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L=L, p=p, a=a, b=b)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointGeneralizedHyperbolic":
        raise NotImplementedError(
            "JointGeneralizedHyperbolic.from_natural: not implemented — "
            "use from_classical or m_step."
        )

    @classmethod
    def _dummy_instance(cls) -> "JointGeneralizedHyperbolic":
        d = 1
        return cls(
            mu=jnp.zeros(d), gamma=jnp.zeros(d),
            L=jnp.eye(d), p=jnp.ones(()), a=jnp.ones(()), b=jnp.ones(())
        )


# ============================================================================
# Marginal GH distribution
# ============================================================================

class GeneralizedHyperbolic(NormalMixture):
    """
    Marginal Generalized Hyperbolic distribution f(x).

    Stores a JointGeneralizedHyperbolic. Provides:
      log_prob(x) — closed-form Bessel expression
      e_step, m_step — for EM fitting
      fit(X, ...) — convenience class method
    """

    def __init__(self, joint: JointGeneralizedHyperbolic):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, p, a, b) -> "GeneralizedHyperbolic":
        joint = JointGeneralizedHyperbolic.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, p=p, a=a, b=b)
        return cls(joint)

    # ------------------------------------------------------------------
    # Marginal log-density (closed form)
    # ------------------------------------------------------------------

    def log_prob(self, x: jax.Array) -> jax.Array:
        """
        Marginal log f(x).

        f(x) ∝ ((Q(x)+b)/A)^{(p-d/2)/2} · K_{p-d/2}(√(A(Q(x)+b))) · exp(γᵀΣ⁻¹(x-μ))

        where Q(x) = (x-μ)ᵀΣ⁻¹(x-μ),  A = a + γᵀΣ⁻¹γ.

        Full normalizing constant:
          C = (2π)^{-d/2} |Σ|^{-1/2} · (A/b)^{p/2} · A^{-d/4} · b^{d/4}
              / K_p(√(ab)) · (some power)
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        j = self._joint
        d = j.d

        # Solve L z = x - μ
        z, w, z2, w2, zw = j._quad_forms(x)
        # Q(x) = ‖z‖² = (x-μ)ᵀΣ⁻¹(x-μ)
        Q = z2
        # A = a + γᵀΣ⁻¹γ = a + w2
        A = j.a + w2
        # Bessel order in posterior: p_post = p - d/2
        p_post = j.p - d / 2.0

        # log f(x) = log C + (p-d/2)/2 * log((Q+b)/A) + ... (skewness) + log K_{p_post}(...)
        #
        # Precise formula (see e.g. Protassov 2004):
        #   log f(x) = ½ log(A) - ½ log(Q+b) ... wait, let me derive carefully.
        #
        # From Barndorff-Nielsen (1978) the marginal density is:
        #
        # f(x) = (2π)^{-d/2} |Σ|^{-1/2} · (a/b)^{p/2} / K_p(√(ab))
        #        · (A / (Q(x)+b))^{(d/2-p)/2}
        #        · K_{p-d/2}(√(A(Q(x)+b)))
        #        · exp((x-μ)ᵀΣ⁻¹γ)
        #
        # log f(x) = -d/2 log(2π) - ½ log|Σ|
        #            + p/2 (log a - log b)
        #            - log K_p(√(ab))
        #            + (d/2-p)/2 · log(A/(Q+b))
        #            + log K_{p-d/2}(√(A(Q+b)))
        #            + γᵀΣ⁻¹(x-μ)                ← = wᵀz (inner product)

        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L)))
        sqrt_ab = jnp.sqrt(j.a * j.b)
        sqrt_A_Qb = jnp.sqrt(A * (Q + j.b))

        log_f = (
            -0.5 * d * jnp.log(2.0 * jnp.pi)
            - 0.5 * log_det_sigma
            + 0.5 * j.p * (jnp.log(j.a + _EPS) - jnp.log(j.b + _EPS))
            - log_kv(j.p, sqrt_ab)
            + 0.5 * (d / 2.0 - j.p) * jnp.log(A / (Q + j.b + _EPS))
            + log_kv(p_post, sqrt_A_Qb)
            + zw                # γᵀΣ⁻¹(x-μ) = wᵀz
        )
        return log_f

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def m_step(
        self,
        X: jax.Array,
        expectations: Dict[str, jax.Array],
        solver: str = "newton",
    ) -> "GeneralizedHyperbolic":
        """
        M-step: update all parameters from E-step expectations.

        expectations: dict with keys E_log_Y (n,), E_inv_Y (n,), E_Y (n,)
        X: (n, d)
        solver: GIG η→θ solver ('newton', 'newton_analytical', 'lbfgs')
        """
        from normix.distributions.gig import GIG

        X = jnp.asarray(X, dtype=jnp.float64)
        n = X.shape[0]
        j = self._joint

        E_log_Y = expectations['E_log_Y']   # (n,)
        E_inv_Y = expectations['E_inv_Y']   # (n,)
        E_Y = expectations['E_Y']           # (n,)

        # Mean sufficient statistics for the normal parameters
        mean_E_inv_Y = jnp.mean(E_inv_Y)
        mean_E_Y = jnp.mean(E_Y)
        E_X = jnp.mean(X, axis=0)                          # (d,)
        E_X_inv_Y = jnp.mean(X * E_inv_Y[:, None], axis=0) # (d,)
        # E[XXᵀ/Y] = mean_i( xᵢxᵢᵀ · E[1/Yᵢ|Xᵢ] )
        E_XXT_inv_Y = jnp.mean(
            jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y), axis=0)  # (d,d)

        mu_new, gamma_new, L_new = JointNormalMixture._mstep_normal_params(
            E_X, E_X_inv_Y, E_XXT_inv_Y, mean_E_inv_Y, mean_E_Y
        )

        # GIG update: fit GIG to mean sufficient stats of Y
        # Warm-start from current GIG natural parameters
        gig_eta = jnp.array([
            jnp.mean(E_log_Y),
            mean_E_inv_Y,
            mean_E_Y,
        ])
        current_gig = GIG(p=j.p, a=j.a, b=j.b)
        gig_new = GIG.from_expectation(gig_eta,
                                        theta0=current_gig.natural_params(),
                                        solver=solver)

        joint_new = JointGeneralizedHyperbolic(
            mu=mu_new, gamma=gamma_new, L=L_new,
            p=gig_new.p, a=gig_new.a, b=gig_new.b,
        )
        return GeneralizedHyperbolic(joint_new)

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------

    def regularize_det_sigma_one(self) -> "GeneralizedHyperbolic":
        """
        Enforce |Σ| = 1 by rescaling (γ, Σ, a, b):
          Σ → Σ/s,  γ → γ/s,  a → a/s,  b → b*s,  where s = det(Σ)^{1/d}
        """
        j = self._joint
        d = j.d
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(j.L)))
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)

        L_new = j.L / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        a_new = j.a / scale
        b_new = j.b * scale

        joint_new = JointGeneralizedHyperbolic(
            mu=j.mu, gamma=gamma_new, L=L_new,
            p=j.p, a=a_new, b=b_new,
        )
        return GeneralizedHyperbolic(joint_new)

    # ------------------------------------------------------------------
    # Convenience fit classmethod
    # ------------------------------------------------------------------

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
        method: str = 'batch',
    ) -> "GeneralizedHyperbolic":
        """Fit GH distribution to data using EM."""
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
    def _initialize(cls, X: jax.Array, key: jax.Array) -> "GeneralizedHyperbolic":
        """Simple initialization from empirical moments + random perturbation."""
        X = jnp.asarray(X, dtype=jnp.float64)
        n, d = X.shape
        mu = jnp.mean(X, axis=0)
        X_centered = X - mu
        sigma_emp = (X_centered.T @ X_centered) / n
        # Add regularization
        sigma_emp = sigma_emp + 1e-4 * jnp.eye(d)
        L = jnp.linalg.cholesky(sigma_emp)
        gamma = jnp.zeros(d)
        # Random perturbation for multi-start
        key1, key2 = jax.random.split(key)
        gamma = gamma + 0.01 * jax.random.normal(key1, (d,), dtype=jnp.float64)
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma_emp,
            p=1.0, a=1.0, b=1.0,
        )
