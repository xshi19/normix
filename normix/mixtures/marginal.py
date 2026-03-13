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

    def e_step(self, X: jax.Array) -> Dict[str, jax.Array]:
        """
        E-step: compute conditional expectations E[g(Y)|X=xᵢ] for each i.

        Returns dict of arrays with shape (n, ...) for each expectation.
        """
        return jax.vmap(self._joint.conditional_expectations)(X)

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
