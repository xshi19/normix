r"""
Coherent risk measures for portfolio projections.

The current implementation provides Conditional Value at Risk (CVaR), with
value, first derivatives, and second derivatives in both the projected
parameter space :math:`(\tilde\mu, \tilde\gamma, \tilde\sigma)` and the
portfolio-weight space :math:`w \in \mathbb{R}^d`. Formulas follow
``docs/theory/cvar_derivatives.rst``.

All Monte Carlo is Rao-Blackwellized over the subordinator :math:`Y`; the
caller passes a pre-sampled array ``Y`` so that ``value``, ``gradient_*``,
and ``hessian_*`` share common random numbers.
"""
from __future__ import annotations

import abc
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.finance._mc import cdf_rb, sample_Y, solve_var
from normix.finance.projection import PortfolioProjection


def _phi(z: jax.Array) -> jax.Array:
    return jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)


class RiskMeasure(eqx.Module):
    """Abstract base for portfolio risk measures."""

    @abc.abstractmethod
    def value(self, projection: PortfolioProjection, Y: jax.Array) -> jax.Array:
        """Risk of the portfolio represented by ``projection``."""


class CVaR(RiskMeasure):
    r"""Conditional Value at Risk at confidence :math:`\alpha \in (0, 1)`.

    For a univariate normal mixture :math:`X = \tilde\mu + \tilde\gamma Y
    + \tilde\sigma \sqrt{Y} Z`, the value is computed in closed form
    conditional on :math:`Y`:

    .. math::

        \operatorname{CVaR}_\alpha(X) = \frac{1}{\alpha} E_Y\!\left[
        \tilde\sigma \sqrt{Y} \, \varphi(z_Y)
        - (\tilde\mu + \tilde\gamma Y) \Phi(z_Y) \right],

    where :math:`z_Y = (x_\alpha - \tilde\mu - \tilde\gamma Y)
    / (\tilde\sigma \sqrt{Y})` and :math:`x_\alpha = -\operatorname{VaR}_\alpha`
    is found by bisection on the Rao-Blackwellized CDF.
    """

    alpha: float = eqx.field(static=True)

    def __init__(self, alpha: float):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        object.__setattr__(self, 'alpha', float(alpha))

    # ------------------------------------------------------------------
    # Sample helper (for callers that want CRN with default size)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_Y(projection: PortfolioProjection, n_mc: int, seed: int = 0) -> jax.Array:
        return sample_Y(projection.subordinator, n_mc, seed)

    # ------------------------------------------------------------------
    # VaR
    # ------------------------------------------------------------------

    def var(self, projection: PortfolioProjection, Y: jax.Array) -> jax.Array:
        r""":math:`\operatorname{VaR}_\alpha` via Rao-Blackwellised bisection."""
        return solve_var(projection.mu_p, projection.gamma_p, projection.sigma_p,
                         Y, self.alpha)

    # ------------------------------------------------------------------
    # Scalar-space value and derivatives
    # ------------------------------------------------------------------

    def value(self, projection: PortfolioProjection, Y: jax.Array) -> jax.Array:
        mu_p, gamma_p, sigma_p = projection.mu_p, projection.gamma_p, projection.sigma_p
        v = solve_var(mu_p, gamma_p, sigma_p, Y, self.alpha)
        sqY = jnp.sqrt(Y)
        z = (-v - mu_p - gamma_p * Y) / (sigma_p * sqY)
        Phi = jax.scipy.stats.norm.cdf(z)
        phi = _phi(z)
        # E[X · 1{X ≤ -v}] = E_Y[(μ + γY) Φ - σ√Y φ]
        integral = jnp.mean((mu_p + gamma_p * Y) * Phi - sigma_p * sqY * phi)
        return -integral / self.alpha

    def gradient_scalar(self, projection: PortfolioProjection,
                        Y: jax.Array) -> jax.Array:
        r"""Return :math:`(\partial r / \partial \tilde\mu,
        \partial r / \partial \tilde\gamma, \partial r / \partial \tilde\sigma)`."""
        mu_p, gamma_p, sigma_p = projection.mu_p, projection.gamma_p, projection.sigma_p
        v = solve_var(mu_p, gamma_p, sigma_p, Y, self.alpha)
        sqY = jnp.sqrt(Y)
        z = (-v - mu_p - gamma_p * Y) / (sigma_p * sqY)
        Phi = jax.scipy.stats.norm.cdf(z)
        phi = _phi(z)
        # value (recomputed cheap, reuse z for consistency)
        rcvar = -jnp.mean((mu_p + gamma_p * Y) * Phi - sigma_p * sqY * phi) / self.alpha
        d_mu = -1.0
        d_gamma = -jnp.mean(Y * Phi) / self.alpha
        d_sigma = (rcvar + mu_p - gamma_p * d_gamma) / sigma_p
        return jnp.stack([jnp.asarray(d_mu, dtype=jnp.float64), d_gamma, d_sigma])

    def hessian_scalar(self, projection: PortfolioProjection,
                       Y: jax.Array) -> jax.Array:
        r"""Return the :math:`3 \times 3` Hessian in :math:`(\tilde\mu, \tilde\gamma, \tilde\sigma)`.

        :math:`\partial^2/\partial \tilde\mu \, \cdot = 0` exactly; the other
        non-trivial blocks follow ``docs/theory/cvar_derivatives.rst``
        :eq:`cvar-nm-hessian`.
        """
        mu_p, gamma_p, sigma_p = projection.mu_p, projection.gamma_p, projection.sigma_p
        v = solve_var(mu_p, gamma_p, sigma_p, Y, self.alpha)
        sqY = jnp.sqrt(Y)
        z = (-v - mu_p - gamma_p * Y) / (sigma_p * sqY)
        phi = _phi(z)
        # ∂rVaR/∂γ = -E[√Y φ] / E[φ/√Y]
        num = jnp.mean(sqY * phi)
        den = jnp.mean(phi / sqY)
        dvar_dgamma = -num / den
        # ∂²rCVaR/∂γ² = (1/(ασ)) E[√Y φ (∂rVaR/∂γ + Y)]
        d2_gg = jnp.mean(sqY * phi * (dvar_dgamma + Y)) / (self.alpha * sigma_p)
        d2_gs = -(gamma_p / sigma_p) * d2_gg
        d2_ss = -(gamma_p / sigma_p) * d2_gs
        H = jnp.zeros((3, 3), dtype=jnp.float64)
        H = H.at[1, 1].set(d2_gg)
        H = H.at[1, 2].set(d2_gs)
        H = H.at[2, 1].set(d2_gs)
        H = H.at[2, 2].set(d2_ss)
        return H

    # ------------------------------------------------------------------
    # Portfolio-space value, gradient, Hessian (chain rule)
    # ------------------------------------------------------------------

    def value_w(self, model, w: jax.Array, Y: jax.Array) -> jax.Array:
        return self.value(PortfolioProjection.from_model(model, w), Y)

    def gradient_w(self, model, w: jax.Array, Y: jax.Array) -> jax.Array:
        r"""Gradient :math:`\nabla_w r_{\operatorname{CVaR}_\alpha}(w)`."""
        proj = PortfolioProjection.from_model(model, w)
        g = self.gradient_scalar(proj, Y)
        j = model._joint
        w = jnp.asarray(w, dtype=jnp.float64)
        Sigma_w = j.sigma() @ w
        # g[0] = -1 (∂/∂μ̃); ∂(wᵀμ)/∂w = μ ⇒ contribution is g[0] · μ = -μ.
        return g[0] * j.mu + g[1] * j.gamma + g[2] * Sigma_w / proj.sigma_p

    def hessian_w(self, model, w: jax.Array, Y: jax.Array) -> jax.Array:
        r"""Hessian :math:`H_{r_{\operatorname{CVaR}_\alpha}}(w)`."""
        proj = PortfolioProjection.from_model(model, w)
        g = self.gradient_scalar(proj, Y)
        H_s = self.hessian_scalar(proj, Y)
        j = model._joint
        w = jnp.asarray(w, dtype=jnp.float64)
        Sigma = j.sigma()
        Sigma_w = Sigma @ w
        sigma_p = proj.sigma_p
        sigma_p2 = sigma_p ** 2
        sigma_p3 = sigma_p ** 3

        H = (
            jnp.outer(j.gamma, j.gamma) * H_s[1, 1]
            + (jnp.outer(j.gamma, Sigma_w) + jnp.outer(Sigma_w, j.gamma))
              * H_s[1, 2] / sigma_p
            + jnp.outer(Sigma_w, Sigma_w) * H_s[2, 2] / sigma_p2
            + (sigma_p2 * Sigma - jnp.outer(Sigma_w, Sigma_w))
              * g[2] / sigma_p3
        )
        return H
