"""
JointNormalMixture — abstract exponential family for normal variance-mean mixtures.

Joint distribution f(x,y):
  X|Y ~ N(μ + γy, Σy)
  Y   ~ subordinator (GIG, Gamma, InverseGamma, InverseGaussian)

Sufficient statistics:
  t(x,y) = [log y, 1/y, y, x, x/y, vec(xxᵀ/y)]

Natural parameters:
  θ₁ = p_sub - 1 - d/2          (GIG p; scalar, depends on subordinator)
  θ₂ = -(b_sub + ½μᵀΣ⁻¹μ)       < 0
  θ₃ = -(a_sub + ½γᵀΣ⁻¹γ)       < 0
  θ₄ = Σ⁻¹γ                      (d-vector)
  θ₅ = Σ⁻¹μ                      (d-vector)
  θ₆ = -½vec(Σ⁻¹)                (d²-vector)

Log partition:
  ψ = ψ_sub(p, a, b) + ½log|Σ| + μᵀΣ⁻¹γ

Expectation parameters (EM E-step quantities):
  η₁ = E[log Y]
  η₂ = E[1/Y]
  η₃ = E[Y]
  η₄ = E[X]      = μ + γ E[Y]
  η₅ = E[X/Y]    = μ E[1/Y] + γ
  η₆ = E[XXᵀ/Y]  = Σ + μμᵀ E[1/Y] + γγᵀ E[Y] + μγᵀ + γμᵀ

EM M-step closed-form (from η):
  Let D = 1 - E[1/Y]·E[Y]
  μ = (E[X] - E[Y]·E[X/Y]) / D
  γ = (E[X/Y] - E[1/Y]·E[X]) / D
  Σ = E[XXᵀ/Y] - E[X/Y]μᵀ - μE[X/Y]ᵀ + E[1/Y]μμᵀ - E[Y]γγᵀ
"""
from __future__ import annotations

import abc
from typing import Dict, Tuple

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily

jax.config.update("jax_enable_x64", True)

from normix.utils.constants import LOG_EPS, SAFE_DENOMINATOR, SIGMA_REG


class JointNormalMixture(ExponentialFamily):
    """
    Abstract joint distribution f(x, y) for normal variance-mean mixtures.

    Stored: mu (d,), gamma (d,), L_Sigma (d,d lower Cholesky of Σ)
    Subordinator parameters defined by concrete subclasses.
    """

    mu: jax.Array         # (d,) location
    gamma: jax.Array      # (d,) skewness
    L_Sigma: jax.Array    # (d,d) lower Cholesky of Σ

    # ------------------------------------------------------------------
    # Abstract: subordinator
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def subordinator(self) -> ExponentialFamily:
        """Return the fitted subordinator distribution."""

    @abc.abstractmethod
    def _subordinator_log_partition(self, p_eff, a_eff, b_eff) -> jax.Array:
        """
        Log partition of the subordinator given effective (p, a, b) parameters.
        For Gamma: p_eff=α, a_eff=β, b_eff=0 (ignored).
        For InverseGamma: p_eff=-α, a_eff=0 (ignored), b_eff=β.
        For InverseGaussian: uses μ_ig, λ_ig.
        For GIG: uses all three.
        """

    # ------------------------------------------------------------------
    # Derived from subclass
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        return int(self.mu.shape[0])

    def sigma(self) -> jax.Array:
        """Covariance matrix Σ = L_Sigma L_Sigmaᵀ."""
        return self.L_Sigma @ self.L_Sigma.T

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rvs(self, n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (X, Y) from the joint distribution.

        Returns
        -------
        X : (n, d) array
        Y : (n,) array
        """
        Y = self.subordinator().rvs(n, seed)
        rng = np.random.default_rng(seed + 1)
        mu = np.asarray(self.mu)
        gamma = np.asarray(self.gamma)
        L_np = np.asarray(self.L_Sigma)
        d = mu.shape[0]
        Z = rng.standard_normal((n, d))
        X = mu[None, :] + gamma[None, :] * Y[:, None] + np.sqrt(Y[:, None]) * (Z @ L_np.T)
        return X, Y

    # ------------------------------------------------------------------
    # Log-prob for joint (x, y)
    # ------------------------------------------------------------------

    def log_prob_joint(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        log f(x, y) = log f(x|y) + log f_Y(y).

        log f(x|y) = -d/2 log(2π) - ½ log|Σy| - ½(x-μ-γy)ᵀ(Σy)⁻¹(x-μ-γy)
                   = -d/2 log(2π) - ½ log|Σ| - d/2 log y
                     - 1/(2y) ‖L⁻¹(x-μ)‖² + γᵀΣ⁻¹(x-μ) - y/2 γᵀΣ⁻¹γ

        log f_Y(y) from subordinator.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        d = self.d

        # Residual r = x - μ, solve L_Sigma z = r
        r = x - self.mu
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, r, lower=True)
        # Solve L_Sigma w = γ
        w = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)

        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(self.L_Sigma)))

        log_fx_given_y = (
            -0.5 * d * jnp.log(2.0 * jnp.pi)
            - 0.5 * log_det_sigma
            - 0.5 * d * jnp.log(y)
            - 0.5 * jnp.dot(z, z) / y
            + jnp.dot(w, z)          # γᵀΣ⁻¹(x-μ) = wᵀz
            - 0.5 * y * jnp.dot(w, w)
        )

        log_fy = self.subordinator().log_prob(y)
        return log_fx_given_y + log_fy

    # ------------------------------------------------------------------
    # Conditional expectations for EM E-step
    # ------------------------------------------------------------------

    def conditional_expectations(self, x: jax.Array) -> Dict[str, jax.Array]:
        """
        Compute E[g(Y)|X=x] for EM E-step.

        The posterior Y|X follows a GIG-like distribution with parameters:
          p_post = p_eff - d/2
          a_post = a_eff + γᵀΣ⁻¹γ
          b_post = b_eff + (x-μ)ᵀΣ⁻¹(x-μ)

        Returns dict with keys: E_log_Y, E_inv_Y, E_Y.
        These are then used in the M-step.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return self._compute_posterior_expectations(x)

    @abc.abstractmethod
    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        """Implemented by each concrete subclass."""

    # ------------------------------------------------------------------
    # Helper: solve Cholesky-based quantities
    # ------------------------------------------------------------------

    def _quad_forms(self, x: jax.Array):
        """
        Compute z = L_Sigma⁻¹(x-μ), w = L_Sigma⁻¹γ.
        Returns z, w, ‖z‖², ‖w‖², zᵀw.
        """
        r = x - self.mu
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, r, lower=True)
        w = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)
        return z, w, jnp.dot(z, z), jnp.dot(w, w), jnp.dot(z, w)

    # ------------------------------------------------------------------
    # ExponentialFamily abstract methods
    # ------------------------------------------------------------------

    @staticmethod
    def sufficient_statistics(xy: jax.Array) -> jax.Array:
        """
        t(x,y) = [log y, 1/y, y, x, x/y, vec(xxᵀ/y)]
        Input: flat vector [x..., y] where x is d-dimensional.
        """
        d = xy.shape[0] - 1
        x = xy[:d]
        y = xy[d]
        return jnp.concatenate([
            jnp.array([jnp.log(y), 1.0 / y, y]),
            x,
            x / y,
            jnp.outer(x, x).ravel() / y,
        ])

    @staticmethod
    def log_base_measure(xy: jax.Array) -> jax.Array:
        d = xy.shape[0] - 1
        y = xy[d]
        return jnp.where(
            y > 0,
            -0.5 * d * jnp.log(2.0 * jnp.pi),
            -jnp.inf,
        )

    # ------------------------------------------------------------------
    # M-step: closed-form normal parameter update
    # ------------------------------------------------------------------

    @staticmethod
    def _mstep_normal_params(
        E_X: jax.Array,
        E_X_inv_Y: jax.Array,
        E_XXT_inv_Y: jax.Array,
        E_inv_Y: jax.Array,
        E_Y: jax.Array,
    ):
        """
        Closed-form M-step for μ, γ, Σ from expectation parameters.

        Returns (mu_new, gamma_new, L_new).
        """
        D = 1.0 - E_inv_Y * E_Y

        # Handle near-singular denominator
        safe_D = jnp.where(jnp.abs(D) > SAFE_DENOMINATOR, D, SAFE_DENOMINATOR)

        mu_new = (E_X - E_Y * E_X_inv_Y) / safe_D
        gamma_new = (E_X_inv_Y - E_inv_Y * E_X) / safe_D

        # Σ = E[XXᵀ/Y] - E[X/Y]μᵀ - μE[X/Y]ᵀ + E[1/Y]μμᵀ - E[Y]γγᵀ
        Sigma = (E_XXT_inv_Y
                 - jnp.outer(E_X_inv_Y, mu_new)
                 - jnp.outer(mu_new, E_X_inv_Y)
                 + E_inv_Y * jnp.outer(mu_new, mu_new)
                 - E_Y * jnp.outer(gamma_new, gamma_new))

        # Symmetrize and ensure positive definite
        Sigma = 0.5 * (Sigma + Sigma.T)
        d = Sigma.shape[0]
        # Add small regularization for numerical stability
        Sigma = Sigma + SIGMA_REG * jnp.eye(d)
        L_new = jnp.linalg.cholesky(Sigma)

        return mu_new, gamma_new, L_new
