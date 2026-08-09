r"""
Coherent risk measures for univariate normal-mixture portfolio returns.

The current implementation provides Conditional Value at Risk (CVaR), with
value, first derivatives, and second derivatives in both the projected
parameter space :math:`(\tilde\mu, \tilde\gamma, \tilde\sigma)` and the
portfolio-weight space :math:`w \in \mathbb{R}^d`. Formulas follow
``docs/theory/cvar_derivatives.md``.

Monte Carlo for CVaR value and derivatives is conditional over the
subordinator :math:`Y` (common random numbers). Deterministic VaR uses
the ``Univariate*`` PINV :meth:`~normix.mixtures.marginal._UnivariateNormalMixtureMixin.ppf`.
"""
from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.finance._mc import quantile_cmc, quantile_cmc_raw
from normix.mixtures.marginal import NormalMixture, _UnivariateNormalMixtureMixin

# Bracket half-width (in mixture std units) for the reduced-coordinate CMC
# quantile. Generous because bisection cost is independent of the width.
_BRACKET_STD = 20.0


def _phi(z: jax.Array) -> jax.Array:
    return jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)


class RiskMeasure(eqx.Module):
    """Abstract base for portfolio risk measures."""

    @abc.abstractmethod
    def value(self, univariate: _UnivariateNormalMixtureMixin, Y: jax.Array) -> jax.Array:
        """Risk of the univariate normal mixture represented by ``univariate``."""

    @abc.abstractmethod
    def value_reduced(
        self, mu: jax.Array, gamma: jax.Array, sigma: jax.Array, Y: jax.Array,
    ) -> jax.Array:
        r"""Risk from raw scalar parameters :math:`(\tilde\mu, \tilde\gamma, \tilde\sigma)`.

        The univariate normal mixture is
        :math:`\tilde\mu + \tilde\gamma Y + \tilde\sigma\sqrt{Y}Z`. Unlike
        :meth:`value`, this signature takes plain scalars (no distribution
        object) and must be Bessel-/PINV-free so it can be
        :func:`jax.vmap`-ed across an efficient-surface grid that shares the
        subordinator draws ``Y``.
        """

    def value_grad_hess_w(
        self, model: NormalMixture, w: jax.Array, Y: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Fused :math:`(r(w), \nabla_w r, H_r(w))` under one CRN sample ``Y``.

        Consumed by :mod:`~normix.finance.transaction_costs` and
        :mod:`~normix.finance.diversification`. Concrete measures override
        with an analytically fused implementation (CVaR shares one CMC VaR
        solve); the base raises — a value-only measure cannot participate in
        the derivative-consuming layers.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide the fused derivative bundle"
        )

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
    is found by bisection on the conditional Monte Carlo CDF (same ``Y`` as
    the integral, for common random numbers).
    """

    alpha: float = eqx.field(static=True)

    def __init__(self, alpha: float):
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        object.__setattr__(self, 'alpha', float(alpha))

    # ------------------------------------------------------------------
    # VaR
    # ------------------------------------------------------------------

    def var(self, univariate: _UnivariateNormalMixtureMixin) -> jax.Array:
        r""":math:`\operatorname{VaR}_\alpha` via deterministic PINV :meth:`ppf`."""
        return -univariate.ppf(self.alpha)

    def _var_cmc(
        self, univariate: _UnivariateNormalMixtureMixin, Y: jax.Array,
    ) -> jax.Array:
        r""":math:`\operatorname{VaR}_\alpha` consistent with CMC ``Y`` samples."""
        return -quantile_cmc(univariate, self.alpha, Y)

    # ------------------------------------------------------------------
    # Scalar-space value and derivatives
    # ------------------------------------------------------------------

    def _cvar_from_quantile(
        self, mu: jax.Array, gamma: jax.Array, sigma: jax.Array,
        x_alpha: jax.Array, Y: jax.Array,
    ) -> jax.Array:
        r"""CVaR integral given the conditional-MC quantile :math:`x_\alpha`.

        :math:`\operatorname{CVaR}_\alpha = -\frac{1}{\alpha} E_Y[(\tilde\mu
        + \tilde\gamma Y)\Phi(z_Y) - \tilde\sigma\sqrt{Y}\varphi(z_Y)]` with
        :math:`z_Y = (x_\alpha - \tilde\mu - \tilde\gamma Y)/(\tilde\sigma
        \sqrt{Y})`. Shared by :meth:`value` and :meth:`value_reduced`.
        """
        sqY = jnp.sqrt(Y)
        z = (x_alpha - mu - gamma * Y) / (sigma * sqY)
        Phi = jax.scipy.stats.norm.cdf(z)
        phi = _phi(z)
        integral = jnp.mean((mu + gamma * Y) * Phi - sigma * sqY * phi)
        return -integral / self.alpha

    def _scalar_bundle(
        self, univariate: _UnivariateNormalMixtureMixin, Y: jax.Array,
        v: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""CVaR value, 3-vector gradient, and 3×3 Hessian at a fixed CMC VaR.

        Shares the single quantile solve ``v = VaR_α`` across value and both
        derivative orders (see :meth:`value_grad_hess_w`).
        """
        mu = univariate._mu_scalar
        gamma = univariate._gamma_scalar
        sigma = univariate._sigma_scalar
        sqY = jnp.sqrt(Y)
        z = (-v - mu - gamma * Y) / (sigma * sqY)
        Phi = jax.scipy.stats.norm.cdf(z)
        phi = _phi(z)

        rcvar = -jnp.mean((mu + gamma * Y) * Phi - sigma * sqY * phi) / self.alpha
        d_mu = jnp.asarray(-1.0, dtype=jnp.float64)
        d_gamma = -jnp.mean(Y * Phi) / self.alpha
        d_sigma = (rcvar + mu - gamma * d_gamma) / sigma
        g = jnp.stack([d_mu, d_gamma, d_sigma])

        num = jnp.mean(sqY * phi)
        den = jnp.mean(phi / sqY)
        dvar_dgamma = -num / den
        d2_gg = jnp.mean(sqY * phi * (dvar_dgamma + Y)) / (self.alpha * sigma)
        d2_gs = -(gamma / sigma) * d2_gg
        d2_ss = -(gamma / sigma) * d2_gs
        H = jnp.zeros((3, 3), dtype=jnp.float64)
        H = H.at[1, 1].set(d2_gg)
        H = H.at[1, 2].set(d2_gs)
        H = H.at[2, 1].set(d2_gs)
        H = H.at[2, 2].set(d2_ss)
        return rcvar, g, H

    def value(self, univariate: _UnivariateNormalMixtureMixin, Y: jax.Array) -> jax.Array:
        x_alpha = -self._var_cmc(univariate, Y)
        return self._cvar_from_quantile(
            univariate._mu_scalar, univariate._gamma_scalar,
            univariate._sigma_scalar, x_alpha, Y,
        )

    def value_reduced(
        self, mu: jax.Array, gamma: jax.Array, sigma: jax.Array, Y: jax.Array,
    ) -> jax.Array:
        r"""CVaR from raw scalar parameters, vectorizable over a surface grid.

        Inverts the conditional-MC CDF for :math:`x_\alpha` within an
        analytic bracket :math:`E[X] \pm 20\,\mathrm{std}[X]` derived from
        the subordinator sample moments of ``Y`` — no PINV table, no Bessel
        evaluation — then reuses :meth:`_cvar_from_quantile`.
        """
        E_Y = jnp.mean(Y)
        Var_Y = jnp.var(Y)
        mean = mu + gamma * E_Y
        half = _BRACKET_STD * jnp.sqrt(E_Y * sigma ** 2 + Var_Y * gamma ** 2)
        x_alpha = quantile_cmc_raw(
            self.alpha, mu, gamma, sigma, Y, mean - half, mean + half,
        )
        return self._cvar_from_quantile(mu, gamma, sigma, x_alpha, Y)

    def gradient_scalar(
        self, univariate: _UnivariateNormalMixtureMixin, Y: jax.Array,
    ) -> jax.Array:
        r"""Return :math:`(\partial r / \partial \tilde\mu,
        \partial r / \partial \tilde\gamma, \partial r / \partial \tilde\sigma)`."""
        v = self._var_cmc(univariate, Y)
        _, g, _ = self._scalar_bundle(univariate, Y, v)
        return g

    def hessian_scalar(
        self, univariate: _UnivariateNormalMixtureMixin, Y: jax.Array,
    ) -> jax.Array:
        r"""Return the :math:`3 \times 3` Hessian in :math:`(\tilde\mu, \tilde\gamma, \tilde\sigma)`.

        :math:`\partial^2/\partial \tilde\mu \, \cdot = 0` exactly; the other
        non-trivial blocks follow ``docs/theory/cvar_derivatives.md``
        :eq:`cvar-nm-hessian`.
        """
        v = self._var_cmc(univariate, Y)
        _, _, H = self._scalar_bundle(univariate, Y, v)
        return H

    # ------------------------------------------------------------------
    # Portfolio-space value, gradient, Hessian (chain rule)
    # ------------------------------------------------------------------

    def _grad_w_from_scalar(
        self,
        model: NormalMixture,
        w: jax.Array,
        uni: _UnivariateNormalMixtureMixin,
        g: jax.Array,
    ) -> jax.Array:
        j = model._joint
        w = jnp.asarray(w, dtype=jnp.float64)
        Sigma_w = j.sigma() @ w
        return g[0] * j.mu + g[1] * j.gamma + g[2] * Sigma_w / uni._sigma_scalar

    def _hess_w_from_scalar(
        self,
        model: NormalMixture,
        w: jax.Array,
        uni: _UnivariateNormalMixtureMixin,
        g: jax.Array,
        H_s: jax.Array,
    ) -> jax.Array:
        j = model._joint
        w = jnp.asarray(w, dtype=jnp.float64)
        Sigma = j.sigma()
        Sigma_w = Sigma @ w
        sigma = uni._sigma_scalar
        sigma2 = sigma ** 2
        sigma3 = sigma ** 3
        return (
            jnp.outer(j.gamma, j.gamma) * H_s[1, 1]
            + (jnp.outer(j.gamma, Sigma_w) + jnp.outer(Sigma_w, j.gamma))
              * H_s[1, 2] / sigma
            + jnp.outer(Sigma_w, Sigma_w) * H_s[2, 2] / sigma2
            + (sigma2 * Sigma - jnp.outer(Sigma_w, Sigma_w))
              * g[2] / sigma3
        )

    @eqx.filter_jit
    def value_w(self, model: NormalMixture, w: jax.Array, Y: jax.Array) -> jax.Array:
        return self.value(model.project(w), Y)

    @eqx.filter_jit
    def gradient_w(self, model: NormalMixture, w: jax.Array, Y: jax.Array) -> jax.Array:
        r"""Gradient :math:`\nabla_w r_{\operatorname{CVaR}_\alpha}(w)`."""
        uni = model.project(w)
        v = self._var_cmc(uni, Y)
        _, g, _ = self._scalar_bundle(uni, Y, v)
        return self._grad_w_from_scalar(model, w, uni, g)

    @eqx.filter_jit
    def hessian_w(self, model: NormalMixture, w: jax.Array, Y: jax.Array) -> jax.Array:
        r"""Hessian :math:`H_{r_{\operatorname{CVaR}_\alpha}}(w)`."""
        uni = model.project(w)
        v = self._var_cmc(uni, Y)
        _, g, H_s = self._scalar_bundle(uni, Y, v)
        return self._hess_w_from_scalar(model, w, uni, g, H_s)

    @eqx.filter_jit
    def value_grad_hess_w(
        self, model: NormalMixture, w: jax.Array, Y: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Fused :math:`(r, \nabla_w r, H_r)` with one CMC VaR solve.

        Equivalent to calling :meth:`value_w`, :meth:`gradient_w`, and
        :meth:`hessian_w` separately, but projects once and bisects the
        conditional-MC CDF once.
        """
        uni = model.project(w)
        v = self._var_cmc(uni, Y)
        value, g, H_s = self._scalar_bundle(uni, Y, v)
        grad_w = self._grad_w_from_scalar(model, w, uni, g)
        hess_w = self._hess_w_from_scalar(model, w, uni, g, H_s)
        return value, grad_w, hess_w
