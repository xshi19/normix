"""
NormalMixture — marginal f(x) = ∫ f(x,y) dy.

Owns a JointNormalMixture. Provides:
  log_prob(x)  — closed-form marginal log-density
  e_step(X)    — jax.vmap over conditional_expectations
  m_step(X, expectations) — returns new NormalMixture
  fit(X, ...)  — convenience EM fitting with multi-start
"""
from __future__ import annotations

import abc
from typing import Dict

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from normix.utils.constants import SIGMA_INIT_REG


class NormalMixture(eqx.Module):
    """
    Marginal f(x) = ∫₀^∞ f(x,y) dy.

    Not an exponential family. Owns a JointNormalMixture (which is).
    """

    _joint: "JointNormalMixture"  # type: ignore[name-defined]

    def __init__(self, joint):
        from normix.mixtures.joint import JointNormalMixture
        self._joint = joint

    @property
    def joint(self):
        return self._joint

    @property
    def d(self) -> int:
        return self._joint.d

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Marginal log f(x). Subclasses provide closed-form; default raises."""
        raise NotImplementedError(
            f"{type(self).__name__}.log_prob: implement closed-form or numerical integration"
        )

    def pdf(self, x: jax.Array) -> jax.Array:
        """Marginal f(x), single observation."""
        return jnp.exp(self.log_prob(x))

    def mean(self) -> jax.Array:
        """E[X] = μ + γ E[Y]."""
        j = self._joint
        E_Y = j.subordinator().mean()
        return j.mu + j.gamma * E_Y

    def cov(self) -> jax.Array:
        """Cov[X] = E[Y] Σ + Var[Y] γγᵀ."""
        j = self._joint
        E_Y = j.subordinator().mean()
        Var_Y = j.subordinator().var()
        return E_Y * j.sigma() + Var_Y * jnp.outer(j.gamma, j.gamma)

    def rvs(self, n: int, seed: int = 42) -> np.ndarray:
        """Sample X from the marginal distribution."""
        X, _ = self._joint.rvs(n, seed)
        return X

    def e_step(self, X: jax.Array, backend: str = 'jax') -> Dict[str, jax.Array]:
        """
        E-step: compute conditional expectations E[g(Y)|X=xᵢ] for each i.

        Returns dict of arrays with shape (n, ...) for each expectation.

        backend='jax' (default): jax.vmap over conditional_expectations.
            JIT-able, differentiable.
        backend='cpu': quad forms in JAX (vmapped) + GIG Bessel on CPU.
            Faster for large N; not JIT-able.
        """
        if backend == 'cpu':
            return self._e_step_cpu(X)
        return jax.vmap(self._joint.conditional_expectations)(X)

    def _e_step_cpu(self, X: jax.Array) -> Dict[str, jax.Array]:
        """
        E-step with quad forms in JAX (vmapped) and Bessel computation on CPU.

        Phase 1: Quad forms (d-dimensional matrix ops) stay in JAX — GPU-friendly.
        Phase 2: GIG expectations via vectorized scipy.kve — C-level, fast.

        Requires self._joint to implement _posterior_gig_params(z2, w2).
        """
        from normix.distributions.generalized_inverse_gaussian import GIG

        j = self._joint
        if not hasattr(j, '_posterior_gig_params'):
            raise NotImplementedError(
                f"{type(j).__name__} does not implement _posterior_gig_params. "
                "backend='cpu' requires this method. Use backend='jax' instead."
            )

        X = jnp.asarray(X, dtype=jnp.float64)

        def _quad_scalars(x):
            z, w, z2, w2, zw = j._quad_forms(x)
            return z2, w2

        z2_all, w2_all = jax.vmap(_quad_scalars)(X)

        p_post, a_post, b_post = j._posterior_gig_params(z2_all, w2_all)

        eta = GIG.expectation_params_batch(p_post, a_post, b_post, backend='cpu')

        return {
            'E_log_Y': eta[:, 0],
            'E_inv_Y': eta[:, 1],
            'E_Y':     eta[:, 2],
        }

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def m_step(
        self,
        X: jax.Array,
        expectations: Dict[str, jax.Array],
        **kwargs,
    ) -> "NormalMixture":
        """
        M-step: update model parameters from sufficient statistics.

        Returns a NEW NormalMixture with updated parameters.
        Subclasses must override _m_step_subordinator.
        """
        from normix.mixtures.joint import JointNormalMixture

        X = jnp.asarray(X, dtype=jnp.float64)

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

        return self._m_step_subordinator(
            mu_new, gamma_new, L_new, gig_eta, **kwargs)

    @abc.abstractmethod
    def _m_step_subordinator(
        self,
        mu_new: jax.Array,
        gamma_new: jax.Array,
        L_new: jax.Array,
        gig_eta: jax.Array,
        **kwargs,
    ) -> "NormalMixture":
        """Update subordinator parameters and construct a new marginal model."""

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def regularize_det_sigma_one(self) -> "NormalMixture":
        """
        Enforce |Σ| = 1 by rescaling.

        Σ → Σ/s, γ → γ/s, subordinator params scaled via _scale_subordinator.
        s = det(Σ)^{1/d}.
        """
        j = self._joint
        d = j.d
        log_det_sigma = j.log_det_sigma()
        log_scale = log_det_sigma / d
        scale = jnp.exp(log_scale)

        L_new = j.L_Sigma / jnp.sqrt(scale)
        gamma_new = j.gamma / scale

        return self._build_rescaled(j.mu, gamma_new, L_new, scale)

    @abc.abstractmethod
    def _build_rescaled(
        self,
        mu: jax.Array,
        gamma_new: jax.Array,
        L_new: jax.Array,
        scale: jax.Array,
    ) -> "NormalMixture":
        """Construct a new model with rescaled subordinator parameters."""

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def marginal_log_likelihood(self, X: jax.Array) -> jax.Array:
        """Mean log-likelihood over dataset."""
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))

    def fit(
        self,
        X: jax.Array,
        *,
        verbose: int = 0,
        max_iter: int = 200,
        tol: float = 1e-3,
        regularization: str = 'none',
        e_step_backend: str = 'jax',
        m_step_backend: str = 'cpu',
        m_step_method: str = 'newton',
    ) -> "EMResult":
        """Fit using self as initialization. Returns EMResult.

        Parameters
        ----------
        X : (n, d) data array
        verbose : 0 = silent, 1 = summary, 2 = per-iteration table
        max_iter, tol : EM convergence parameters
        regularization : 'det_sigma_one' or 'none'
        e_step_backend, m_step_backend : 'jax' or 'cpu'
        m_step_method : 'newton', 'lbfgs', or 'bfgs'

        Returns
        -------
        EMResult with .model, .log_likelihoods, .param_changes, etc.
        """
        from normix.fitting.em import BatchEMFitter
        fitter = BatchEMFitter(
            verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method,
        )
        return fitter.fit(self, X)

    @classmethod
    def default_init(cls, X: jax.Array) -> "NormalMixture":
        """Moment-based initialisation from data.

        Returns a model with:
          mu    = sample mean
          gamma = zeros
          Sigma = empirical covariance (regularized)
          subordinator = distribution-specific defaults

        Useful as a starting point for ``model.fit(X)``.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        n, d = X.shape
        mu = jnp.mean(X, axis=0)
        X_centered = X - mu
        sigma_emp = (X_centered.T @ X_centered) / n + SIGMA_INIT_REG * jnp.eye(d)
        gamma = jnp.zeros(d, dtype=jnp.float64)
        return cls._from_init_params(mu, gamma, sigma_emp)

    @classmethod
    @abc.abstractmethod
    def _from_init_params(
        cls, mu: jax.Array, gamma: jax.Array, sigma: jax.Array,
    ) -> "NormalMixture":
        """Construct a model with default subordinator parameters for initialisation."""
