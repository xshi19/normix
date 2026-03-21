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

from normix.utils.bessel import log_kv
from normix.exponential_family import ExponentialFamily
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture

jax.config.update("jax_enable_x64", True)

from normix.utils.constants import LOG_EPS, SIGMA_INIT_REG


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

    def __init__(self, mu, gamma, L_Sigma, p, a, b):
        # eqx.Module fields must be set via object.__setattr__ before freeze
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'p', jnp.asarray(p, dtype=jnp.float64))
        object.__setattr__(self, 'a', jnp.asarray(a, dtype=jnp.float64))
        object.__setattr__(self, 'b', jnp.asarray(b, dtype=jnp.float64))

    # ------------------------------------------------------------------
    # JointNormalMixture abstract methods
    # ------------------------------------------------------------------

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.generalized_inverse_gaussian import GIG
        return GIG(p=self.p, a=self.a, b=self.b)

    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        from normix.distributions.generalized_inverse_gaussian import GIG
        theta = jnp.array([p_eff - 1.0, -b_eff / 2.0, -a_eff / 2.0])
        return GIG._log_partition_from_theta(theta)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """
        Posterior Y|X=x ~ GIG(p-d/2, a+γᵀΣ⁻¹γ, b+(x-μ)ᵀΣ⁻¹(x-μ)).
        """
        from normix.distributions.generalized_inverse_gaussian import GIG

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

    def _posterior_gig_params(
        self, z2: jax.Array, w2: jax.Array
    ):
        """Posterior GIG (p, a, b) given quad-form scalars z2=‖L⁻¹(x-μ)‖², w2=‖L⁻¹γ‖²."""
        return (self.p - self.d / 2.0,
                self.a + w2,
                self.b + z2)

    # ------------------------------------------------------------------
    # ExponentialFamily: natural params / log partition
    # ------------------------------------------------------------------

    def natural_params(self) -> jax.Array:
        """
        θ = [p-1-d/2, -(b+½μᵀΣ⁻¹μ), -(a+½γᵀΣ⁻¹γ), Σ⁻¹γ, Σ⁻¹μ, -½vec(Σ⁻¹)]
        """
        _, _, mu_quad, gamma_quad, _ = self._precision_quantities()
        return self._assemble_natural_params(
            self.p - 1.0 - self.d / 2.0,
            -(self.b + mu_quad),
            -(self.a + gamma_quad),
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        """
        ψ = ψ_GIG(p, a, b) + ½log|Σ| + μᵀΣ⁻¹γ

        Recover p,a,b,μ,γ,Σ from theta.
        Dimension d is inferred from len(θ) = 3 + 2d + d².
        """
        from normix.distributions.generalized_inverse_gaussian import GIG
        from normix.mixtures.joint import JointNormalMixture

        (d, theta_1, theta_2, theta_3, *_, log_det_Sigma, _, _,
         mu_quad, gamma_quad, mu_Lambda_gamma) = JointNormalMixture._parse_joint_theta(theta)

        p = theta_1 + 1.0 + d / 2.0
        b = -theta_2 - mu_quad
        a = -theta_3 - gamma_quad

        gig_theta = jnp.array([p - 1.0, -b / 2.0, -a / 2.0])
        psi_gig = GIG._log_partition_from_theta(gig_theta)

        return psi_gig + 0.5 * log_det_Sigma + mu_Lambda_gamma

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, p, a, b):
        """Construct from classical parameters."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, p=p, a=a, b=b)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointGeneralizedHyperbolic":
        raise NotImplementedError(
            "JointGeneralizedHyperbolic.from_natural: not implemented — "
            "use from_classical or m_step."
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

        log_det_sigma = j.log_det_sigma()
        sqrt_ab = jnp.sqrt(j.a * j.b)
        sqrt_A_Qb = jnp.sqrt(A * (Q + j.b))

        log_f = (
            -0.5 * d * jnp.log(2.0 * jnp.pi)
            - 0.5 * log_det_sigma
            + 0.5 * j.p * (jnp.log(j.a + LOG_EPS) - jnp.log(j.b + LOG_EPS))
            - log_kv(j.p, sqrt_ab)
            + 0.5 * (d / 2.0 - j.p) * jnp.log(A / (Q + j.b + LOG_EPS))
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
        backend: str = "jax",
        method: str = "newton",
        maxiter: int = 20,
    ) -> "GeneralizedHyperbolic":
        """
        M-step: update all parameters from E-step expectations.

        Parameters
        ----------
        expectations : dict with keys E_log_Y (n,), E_inv_Y (n,), E_Y (n,)
        X : (n, d)
        backend : 'jax' (default) or 'cpu' — passed to :meth:`GIG.from_expectation`
        method : 'newton', 'lbfgs', 'bfgs' — passed to :meth:`GIG.from_expectation`
        maxiter : iteration budget for the GIG η→θ solver
        """
        from normix.distributions.generalized_inverse_gaussian import GIG

        X = jnp.asarray(X, dtype=jnp.float64)
        j = self._joint

        E_log_Y = expectations['E_log_Y']   # (n,)
        E_inv_Y = expectations['E_inv_Y']   # (n,)
        E_Y = expectations['E_Y']           # (n,)

        mean_E_inv_Y = jnp.mean(E_inv_Y)
        mean_E_Y = jnp.mean(E_Y)
        E_X = jnp.mean(X, axis=0)                          # (d,)
        E_X_inv_Y = jnp.mean(X * E_inv_Y[:, None], axis=0) # (d,)
        E_XXT_inv_Y = jnp.mean(
            jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y), axis=0)  # (d,d)

        mu_new, gamma_new, L_new = JointNormalMixture._mstep_normal_params(
            E_X, E_X_inv_Y, E_XXT_inv_Y, mean_E_inv_Y, mean_E_Y
        )

        gig_eta = jnp.array([
            jnp.mean(E_log_Y),
            mean_E_inv_Y,
            mean_E_Y,
        ])
        current_gig = GIG(p=j.p, a=j.a, b=j.b)
        gig_new = GIG.from_expectation(
            gig_eta,
            theta0=current_gig.natural_params(),
            backend=backend, method=method, maxiter=maxiter,
        )

        joint_new = JointGeneralizedHyperbolic(
            mu=mu_new, gamma=gamma_new, L_Sigma=L_new,
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
        log_det_sigma = j.log_det_sigma()
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)

        L_new = j.L_Sigma / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        a_new = j.a / scale
        b_new = j.b * scale

        joint_new = JointGeneralizedHyperbolic(
            mu=j.mu, gamma=gamma_new, L_Sigma=L_new,
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
                               regularization=regularization,
                               e_step_backend='cpu')
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
        sigma_emp = sigma_emp + SIGMA_INIT_REG * jnp.eye(d)
        L = jnp.linalg.cholesky(sigma_emp)
        gamma = jnp.zeros(d)
        # Random perturbation for multi-start
        key1, key2 = jax.random.split(key)
        gamma = gamma + 0.01 * jax.random.normal(key1, (d,), dtype=jnp.float64)
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma_emp,
            p=1.0, a=1.0, b=1.0,
        )
