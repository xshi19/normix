r"""
Mean-risk portfolio optimization for normal-mixture models.

For a multivariate normal mixture :math:`X \stackrel{d}{=} \mu + \gamma Y
+ \sqrt{Y} Z` with :math:`Z \sim \mathcal{N}(0, \Sigma)`, the mean-risk
problem

.. math::

    \min_w \rho(w^\top X) \quad \text{s.t.} \quad
    w^\top e = 1, \quad E[w^\top X] \ge m

reduces — for any coherent risk measure :math:`\rho` — to a two-dimensional
problem in the *reduced coordinates* :math:`\tilde\mu = w^\top\mu` and
:math:`\tilde\gamma = w^\top\gamma`. The minimum-dispersion weights that
realise a given :math:`(\tilde\mu, \tilde\gamma)` are

.. math::

    w^*(\tilde\mu, \tilde\gamma) = \Sigma^{-1}[\mu\;\gamma\;e]\,A^{-1}
    [\tilde\mu\;\tilde\gamma\;1]^\top, \qquad
    A = [\mu\;\gamma\;e]^\top \Sigma^{-1} [\mu\;\gamma\;e],

and the realised dispersion is
:math:`g(\tilde\mu, \tilde\gamma) = [\tilde\mu\;\tilde\gamma\;1] A^{-1}
[\tilde\mu\;\tilde\gamma\;1]^\top`. The map
:math:`(\tilde\mu, \tilde\gamma) \mapsto \rho` is the **efficient surface**;
its lower envelope under the return constraint is the **efficient frontier**.

See :doc:`../theory/mean_risk_optimization` for the derivation.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_factor, cho_solve

from normix.finance.risk import RiskMeasure
from normix.mixtures.marginal import NormalMixture

# Reciprocal golden ratio for golden-section search.
_INV_PHI = (5.0 ** 0.5 - 1.0) / 2.0


class EfficientSurface(eqx.Module):
    r"""Efficient surface :math:`(\tilde\mu, \tilde\gamma) \mapsto \rho` on a grid.

    ``risk[i, j]`` is the risk of the minimum-dispersion portfolio with
    reduced coordinates ``(mu_tilde[i], gamma_tilde[j])``;
    ``expected_return[i, j] = mu_tilde[i] + gamma_tilde[j] * E[Y]``.
    """

    mu_tilde: Array
    gamma_tilde: Array
    risk: Array
    expected_return: Array


class EfficientFrontier(eqx.Module):
    r"""Mean-risk efficient frontier: minimum risk per target expected return.

    Each entry is the solution of the reduced problem on the constraint line
    :math:`\tilde\mu + \tilde\gamma E[Y] = m`, including the realised
    portfolio ``weights`` of shape ``(K, d)``.
    """

    expected_return: Array
    risk: Array
    mu_tilde: Array
    gamma_tilde: Array
    weights: Array


def _golden_section_min(f, lo: Array, hi: Array, n_iter: int) -> Array:
    r"""Vectorized golden-section minimiser of a unimodal ``f: (K,) -> (K,)``.

    Shrinks each bracket ``[lo_k, hi_k]`` independently; returns the
    per-component minimiser. Used to minimise risk along return-constraint
    lines (the risk is convex in the reduced coordinates).
    """
    def body(_, bracket):
        a, b = bracket
        width = (b - a) * _INV_PHI
        c = b - width
        d = a + width
        take_left = f(c) < f(d)
        return jnp.where(take_left, a, c), jnp.where(take_left, d, b)

    a, b = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    return 0.5 * (a + b)


class MeanRiskProblem(eqx.Module):
    r"""Mean-risk optimization in reduced :math:`(\tilde\mu, \tilde\gamma)` coordinates.

    Bundles a fitted :class:`~normix.mixtures.marginal.NormalMixture` and a
    :class:`~normix.finance.risk.RiskMeasure`. All heavy evaluations share a
    fixed subordinator sample ``Y`` (common random numbers); draw it once via
    ``model.joint.subordinator().rvs(n, seed)``.
    """

    model: NormalMixture
    risk: RiskMeasure

    # ------------------------------------------------------------------
    # Reduced-coordinate algebra
    # ------------------------------------------------------------------

    def _reduced(self) -> tuple[Array, Array, Array]:
        r"""Return ``(A, A_inv, Sinv_M)`` with :math:`M = [\mu\;\gamma\;e]`.

        :math:`A = M^\top \Sigma^{-1} M` (3×3 SPD),
        :math:`\Sigma^{-1} M` (d×3), via Cholesky solves on
        :math:`\Sigma = L L^\top`.
        """
        mu = self.model.mu
        gamma = self.model.gamma
        e = jnp.ones_like(mu)
        M = jnp.stack([mu, gamma, e], axis=1)                  # (d, 3)
        Sinv_M = cho_solve((self.model.L_Sigma, True), M)      # (d, 3)
        A = M.T @ Sinv_M                                       # (3, 3)
        A_inv = cho_solve(cho_factor(A), jnp.eye(3, dtype=A.dtype))
        return A, A_inv, Sinv_M

    def E_Y(self) -> Array:
        r""":math:`E[Y]` — subordinator mean."""
        return self.model.joint.subordinator().mean()

    def weights(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        r"""Minimum-dispersion weights realising :math:`(\tilde\mu, \tilde\gamma)`."""
        _, A_inv, Sinv_M = self._reduced()
        c = jnp.stack([jnp.asarray(mu_tilde, dtype=jnp.float64),
                       jnp.asarray(gamma_tilde, dtype=jnp.float64),
                       jnp.ones((), dtype=jnp.float64)])
        return Sinv_M @ (A_inv @ c)

    def dispersion(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        r"""Realised dispersion :math:`g(\tilde\mu, \tilde\gamma) = w^{*\top}\Sigma w^*`."""
        _, A_inv, _ = self._reduced()
        c = jnp.stack([jnp.asarray(mu_tilde, dtype=jnp.float64),
                       jnp.asarray(gamma_tilde, dtype=jnp.float64),
                       jnp.ones((), dtype=jnp.float64)])
        return c @ A_inv @ c

    def expected_return(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        r""":math:`m = \tilde\mu + \tilde\gamma\,E[Y]`."""
        return mu_tilde + gamma_tilde * self.E_Y()

    def min_variance_point(self) -> tuple[Array, Array]:
        r"""Reduced coordinates :math:`(\tilde\mu, \tilde\gamma)` of the global
        minimum-variance portfolio :math:`w = \Sigma^{-1}e / (e^\top\Sigma^{-1}e)`.

        A convenient anchor for choosing efficient-surface grid ranges.
        """
        A, _, _ = self._reduced()
        denom = A[2, 2]
        return A[0, 2] / denom, A[1, 2] / denom

    def projection_at(self, mu_tilde: Array, gamma_tilde: Array):
        r"""Univariate portfolio return at :math:`(\tilde\mu, \tilde\gamma)`.

        Projects the minimum-dispersion :meth:`weights`; the result is a
        ``Univariate*`` instance with location :math:`\tilde\mu`, skewness
        :math:`\tilde\gamma`, variance :math:`g(\tilde\mu, \tilde\gamma)`.
        """
        return self.model.project(self.weights(mu_tilde, gamma_tilde))

    # ------------------------------------------------------------------
    # Risk on the efficient surface
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def _surface_risk(self, mu_flat: Array, gamma_flat: Array, Y: Array) -> Array:
        r"""Risk over flattened reduced coordinates (vectorized, JIT-able)."""
        _, A_inv, _ = self._reduced()
        C = jnp.stack([mu_flat, gamma_flat, jnp.ones_like(mu_flat)], axis=-1)
        g = jnp.einsum('pi,ij,pj->p', C, A_inv, C)
        sigma = jnp.sqrt(g)
        return jax.vmap(self.risk.value_reduced, in_axes=(0, 0, 0, None))(
            mu_flat, gamma_flat, sigma, Y)

    def risk_at(self, mu_tilde: Array, gamma_tilde: Array, Y: Array) -> Array:
        r"""Efficient-surface risk at a single :math:`(\tilde\mu, \tilde\gamma)`."""
        sigma = jnp.sqrt(self.dispersion(mu_tilde, gamma_tilde))
        return self.risk.value_reduced(
            jnp.asarray(mu_tilde, dtype=jnp.float64),
            jnp.asarray(gamma_tilde, dtype=jnp.float64), sigma, Y)

    def efficient_surface(
        self, mu_tilde: Array, gamma_tilde: Array, Y: Array,
    ) -> EfficientSurface:
        r"""Evaluate the efficient surface over the grid ``mu_tilde × gamma_tilde``.

        ``mu_tilde`` and ``gamma_tilde`` are 1-D arrays; the returned
        ``risk`` has shape ``(len(mu_tilde), len(gamma_tilde))``. Memory
        scales as ``len(mu_tilde) * len(gamma_tilde) * len(Y)``.
        """
        mu_tilde = jnp.asarray(mu_tilde, dtype=jnp.float64)
        gamma_tilde = jnp.asarray(gamma_tilde, dtype=jnp.float64)
        MU, GA = jnp.meshgrid(mu_tilde, gamma_tilde, indexing='ij')
        risk = self._surface_risk(MU.ravel(), GA.ravel(), Y).reshape(MU.shape)
        return EfficientSurface(
            mu_tilde=mu_tilde, gamma_tilde=gamma_tilde,
            risk=risk, expected_return=MU + GA * self.E_Y(),
        )

    def efficient_frontier(
        self,
        returns: Array,
        Y: Array,
        gamma_bounds: tuple[float, float],
        n_iter: int = 48,
    ) -> EfficientFrontier:
        r"""Minimum risk for each target expected return in ``returns``.

        For every target :math:`m`, minimises the risk along the constraint
        line :math:`\tilde\mu = m - \tilde\gamma E[Y]` over
        :math:`\tilde\gamma \in` ``gamma_bounds`` by golden-section search
        (the surface is convex, so the restriction is unimodal).
        """
        returns = jnp.asarray(returns, dtype=jnp.float64)
        E_Y = self.E_Y()
        lo = jnp.full_like(returns, gamma_bounds[0])
        hi = jnp.full_like(returns, gamma_bounds[1])

        def risk_of_gamma(gamma_vec: Array) -> Array:
            return self._surface_risk(returns - gamma_vec * E_Y, gamma_vec, Y)

        gamma_star = _golden_section_min(risk_of_gamma, lo, hi, n_iter)
        mu_star = returns - gamma_star * E_Y
        risk_star = self._surface_risk(mu_star, gamma_star, Y)
        weights = jax.vmap(self.weights)(mu_star, gamma_star)
        return EfficientFrontier(
            expected_return=returns, risk=risk_star,
            mu_tilde=mu_star, gamma_tilde=gamma_star, weights=weights,
        )
