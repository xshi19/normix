"""
NormalMixture — marginal f(x) = ∫ f(x,y) dy.

Owns a JointNormalMixture. Provides:
  log_prob(x)  — closed-form marginal log-density
  e_step(X)    — jax.vmap over conditional_expectations
  m_step(X, expectations) — returns new NormalMixture
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


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

        # Phase 1: Quad forms in JAX (vmapped — benefits from GPU for large d)
        def _quad_scalars(x):
            z, w, z2, w2, zw = j._quad_forms(x)
            return z2, w2

        z2_all, w2_all = jax.vmap(_quad_scalars)(X)  # (N,), (N,)

        # Posterior GIG params (JAX arithmetic, stays on device)
        p_post, a_post, b_post = j._posterior_gig_params(z2_all, w2_all)

        # Phase 2: GIG expectations via CPU Bessel
        # Transfers (N,) arrays to numpy, calls 6 × scipy.kve, returns (N,3)
        eta = GIG.expectation_params_batch(p_post, a_post, b_post, backend='cpu')

        return {
            'E_log_Y': eta[:, 0],
            'E_inv_Y': eta[:, 1],
            'E_Y':     eta[:, 2],
        }

    def m_step(
        self,
        X: jax.Array,
        expectations: Dict[str, jax.Array],
    ) -> "NormalMixture":
        """
        M-step: update model parameters from sufficient statistics.

        Returns a NEW NormalMixture with updated parameters.
        """
        raise NotImplementedError(f"{type(self).__name__}.m_step not implemented")

    def marginal_log_likelihood(self, X: jax.Array) -> jax.Array:
        """Mean log-likelihood over dataset."""
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))
