"""
Factor-analysis mixture base class: :math:`\\Sigma = F F^\\top + \\mathrm{diag}(D)` storage.

A factor-analysis Generalized-Hyperbolic random vector

.. math::

    X \\stackrel{d}{=} \\mu + \\gamma Y + \\sqrt{Y}\\,(F Z + \\varepsilon),
    \\qquad Z \\sim N(0, I_r),\\;
    \\varepsilon \\sim N(0, \\mathrm{diag}(D)),

where :math:`Y` is a GH subordinator (Gamma, InverseGamma,
InverseGaussian or GIG). The marginal :math:`X` is a normal
variance-mean mixture with dispersion :math:`\\Sigma = F F^\\top +
\\mathrm{diag}(D)`. Storing :math:`(F, D)` instead of a full Cholesky of
:math:`\\Sigma` keeps quadratic forms and log-determinants in
:math:`\\mathcal{O}(d r^2 + r^3)` via Woodbury and makes the rotation
gauge of :math:`F` irrelevant for convergence (we measure on
:math:`\\Sigma`).

This module hosts only the abstract :class:`FactorNormalMixture` base.
The four concrete subordinator families
(:class:`~normix.distributions.variance_gamma.FactorVarianceGamma`,
:class:`~normix.distributions.normal_inverse_gamma.FactorNormalInverseGamma`,
:class:`~normix.distributions.normal_inverse_gaussian.FactorNormalInverseGaussian`,
:class:`~normix.distributions.generalized_hyperbolic.FactorGeneralizedHyperbolic`)
live next to their full-:math:`\\Sigma` siblings in
``normix/distributions/``.

The FA complete-data structure is over :math:`(X, Y, Z)` with ten
sufficient statistics (see ``docs/theory/factor_analysis.rst``), so this
family does not share the joint's six-statistic exponential-family
signature — that is why :class:`FactorNormalMixture` is a sibling of
:class:`~normix.mixtures.marginal.NormalMixture`, not a subclass.
"""
from __future__ import annotations

import abc
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.fitting.eta import FactorMixtureStats
from normix.mixtures.marginal import MarginalMixture
from normix.utils.constants import B_POST_FLOOR, D_FLOOR, SIGMA_INIT_REG


# ============================================================================
# Abstract base
# ============================================================================


class FactorNormalMixture(MarginalMixture):
    r"""
    Abstract marginal :math:`f(x)` for a factor-analysis normal
    variance-mean mixture with dispersion
    :math:`\Sigma = F F^\top + \mathrm{diag}(D)`.

    Stored fields
    -------------
    mu : (d,)
        Location :math:`\mu`.
    gamma : (d,)
        Skewness :math:`\gamma`.
    F : (d, r)
        Factor loadings; ``r`` is the latent factor dimension.
    D : (d,)
        Diagonal entries of the residual covariance (positive).
    subordinator : ExponentialFamily
        Fitted subordinator (Gamma / InverseGamma / InverseGaussian /
        GIG).

    Subclasses define the subordinator family (via the instance stored in
    ``subordinator``, which supplies the shared posterior map through
    ``to_gig()``) and implement :meth:`log_prob`,
    :meth:`_subordinator_from_eta`, and a few forwarders / initialisers.
    The prior-to-posterior GIG conjugacy (:meth:`_posterior_gig_params`)
    is uniform across families and lives on the base. The linear algebra for
    :math:`\Sigma^{-1}` and :math:`\log|\Sigma|` is shared via
    Woodbury helpers (:meth:`_M`, :meth:`_solve`, :meth:`_quad_form`,
    :meth:`_log_det_sigma`, :meth:`_beta`).

    Notes
    -----
    ``F`` is identifiable only up to a right :math:`r \times r`
    orthogonal rotation, so convergence is measured on
    :math:`\Sigma = F F^\top + \mathrm{diag}(D)`
    (:meth:`em_convergence_params`) rather than on ``F`` directly.
    """

    mu: jax.Array          # (d,)
    gamma: jax.Array       # (d,)
    F: jax.Array           # (d, r)
    D: jax.Array           # (d,) diagonal entries (positive)
    subordinator: ExponentialFamily

    # ------------------------------------------------------------------
    # Shape accessors
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        return int(self.mu.shape[0])

    @property
    def r(self) -> int:
        return int(self.F.shape[1])

    # ------------------------------------------------------------------
    # Shared subclass helper
    # ------------------------------------------------------------------

    @staticmethod
    def _check_init_args(
        mu: jax.Array, gamma: jax.Array, F: jax.Array, D: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Coerce ``(mu, gamma, F, D)`` to float64 and floor ``D`` for positivity."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        F = jnp.asarray(F, dtype=jnp.float64)
        D = jnp.maximum(jnp.asarray(D, dtype=jnp.float64), D_FLOOR)
        return mu, gamma, F, D

    # ------------------------------------------------------------------
    # Woodbury helpers (Σ = F Fᵀ + diag(D), all in O(d r² + r³))
    # ------------------------------------------------------------------

    def _M(self) -> jax.Array:
        r""":math:`M = I_r + F^\top D^{-1} F`, shape ``(r, r)``."""
        return (jnp.eye(self.r, dtype=jnp.float64)
                + (self.F.T / self.D) @ self.F)

    def _solve(self, x: jax.Array) -> jax.Array:
        r""":math:`\Sigma^{-1} x` via Woodbury, ``x`` shape ``(d,)``."""
        Dinv_x = x / self.D
        inner = jax.scipy.linalg.solve(
            self._M(), self.F.T @ Dinv_x, assume_a='pos')
        return Dinv_x - (self.F @ inner) / self.D

    def _solve_matrix(self, X: jax.Array) -> jax.Array:
        r""":math:`\Sigma^{-1} X` via Woodbury, ``X`` shape ``(d, k)``."""
        Dinv_X = X / self.D[:, None]
        inner = jax.scipy.linalg.solve(
            self._M(), self.F.T @ Dinv_X, assume_a='pos')
        return Dinv_X - (self.F @ inner) / self.D[:, None]

    def _quad_form(self, x: jax.Array) -> jax.Array:
        r""":math:`x^\top \Sigma^{-1} x`, scalar."""
        return jnp.dot(x, self._solve(x))

    def _log_det_sigma(self) -> jax.Array:
        r""":math:`\log|\Sigma| = \log|D| + \log|M|`."""
        sign, logdet_M = jnp.linalg.slogdet(self._M())
        return jnp.sum(jnp.log(self.D)) + logdet_M

    def _beta(self) -> jax.Array:
        r""":math:`\beta = F^\top \Sigma^{-1} = M^{-1} F^\top D^{-1}`,
        shape ``(r, d)``."""
        FtDinv = self.F.T / self.D[None, :]
        return jax.scipy.linalg.solve(self._M(), FtDinv, assume_a='pos')

    def _quad_forms(
        self, x: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        r"""Return :math:`(z_2, w_2, zw)` =
        :math:`((x-\mu)^\top \Sigma^{-1} (x-\mu),\;
        \gamma^\top \Sigma^{-1} \gamma,\;
        \gamma^\top \Sigma^{-1} (x-\mu))`.
        """
        r_vec = x - self.mu
        Sigma_inv_r = self._solve(r_vec)
        Sigma_inv_g = self._solve(self.gamma)
        z2 = jnp.dot(r_vec, Sigma_inv_r)
        w2 = jnp.dot(self.gamma, Sigma_inv_g)
        zw = jnp.dot(self.gamma, Sigma_inv_r)
        return z2, w2, zw

    # ------------------------------------------------------------------
    # Marginal moments (mean / cov / sigma)
    # ------------------------------------------------------------------

    def sigma(self) -> jax.Array:
        r""":math:`\Sigma = F F^\top + \mathrm{diag}(D)` (dense)."""
        return self.F @ self.F.T + jnp.diag(self.D)

    def log_det_sigma(self) -> jax.Array:
        r""":math:`\log|\Sigma|`, computed via Woodbury."""
        return self._log_det_sigma()

    def mean(self) -> jax.Array:
        r""":math:`E[X] = \mu + \gamma\,E[Y]`."""
        return self.mu + self.gamma * self.subordinator.mean()

    def cov(self) -> jax.Array:
        r""":math:`\mathrm{Cov}[X] = E[Y]\,\Sigma + \mathrm{Var}[Y]\,\gamma\gamma^\top`."""
        E_Y = self.subordinator.mean()
        Var_Y = self.subordinator.var()
        return E_Y * self.sigma() + Var_Y * jnp.outer(self.gamma, self.gamma)

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        r"""Sample ``n`` observations from the marginal :math:`f(x)`.

        Uses :math:`X = \mu + \gamma Y + \sqrt{Y}(F Z + \varepsilon)`.
        """
        Y = self.subordinator.rvs(n, seed)
        key = jax.random.PRNGKey(seed + 1)
        kZ, kE = jax.random.split(key)
        Z = jax.random.normal(kZ, shape=(n, self.r), dtype=jnp.float64)
        E = jax.random.normal(kE, shape=(n, self.d), dtype=jnp.float64)
        eps = E * jnp.sqrt(self.D)[None, :]
        sqrtY = jnp.sqrt(Y)[:, None]
        X = (self.mu[None, :]
             + self.gamma[None, :] * Y[:, None]
             + sqrtY * (Z @ self.F.T + eps))
        return X

    # ------------------------------------------------------------------
    # Subordinator hooks (subclass-specific)
    # ------------------------------------------------------------------

    def _posterior_gig_params(
        self, z2: jax.Array, w2: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        r"""Prior-to-posterior GIG conjugacy map, uniform across families.

        Posterior :math:`Y \mid X = x \sim \mathrm{GIG}(p_{\mathrm{post}},
        a_{\mathrm{post}}, b_{\mathrm{post}})` given quad-form scalars
        :math:`z_2 = (x-\mu)^\top \Sigma^{-1} (x-\mu)`,
        :math:`w_2 = \gamma^\top \Sigma^{-1} \gamma`:

        .. math::

            p_{\mathrm{post}} = p_{\mathrm{gig}} - \tfrac{d}{2}, \quad
            a_{\mathrm{post}} = a_{\mathrm{gig}} + w_2, \quad
            b_{\mathrm{post}} = b_{\mathrm{gig}} + z_2,

        where :math:`(p_{\mathrm{gig}}, a_{\mathrm{gig}}, b_{\mathrm{gig}})`
        are the subordinator's exact GIG coordinates (``subordinator.to_gig()``).
        See :meth:`~normix.mixtures.joint.JointNormalMixture._posterior_gig_params`
        for the per-family table. Stays **pure**; the :math:`b_{\mathrm{post}}`
        floor lives in :meth:`_floored_posterior_gig_params`.
        """
        gig = self.subordinator.to_gig()
        return gig.p - self.d / 2.0, gig.a + w2, gig.b + z2

    def _floored_posterior_gig_params(
        self, z2: jax.Array, w2: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        r"""E-step entry point: :meth:`_posterior_gig_params` with
        :math:`b_{\mathrm{post}}` floored at
        :data:`~normix.utils.constants.B_POST_FLOOR` (single chokepoint;
        only binds for VG, prior :math:`b=0`)."""
        p_post, a_post, b_post = self._posterior_gig_params(z2, w2)
        return p_post, a_post, jnp.maximum(b_post, B_POST_FLOOR)

    @classmethod
    @abc.abstractmethod
    def _subordinator_from_eta(
        cls,
        eta: FactorMixtureStats,
        *,
        theta0=None,
        **kwargs,
    ) -> ExponentialFamily:
        """Fit the subordinator from the relevant subset of
        :math:`(s_1, s_2, s_3) = (E[1/Y], E[Y], E[\\log Y])`.

        ``theta0`` is forwarded to the subordinator solver as a
        warm-start; closed-form subordinators ignore it.
        """

    @abc.abstractmethod
    def _subordinator_expectations(
        self,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        r"""Return :math:`(E[\log Y], E[1/Y], E[Y])` under the
        subordinator prior (used by :meth:`compute_eta_from_model`)."""

    # ------------------------------------------------------------------
    # E-step: posterior expectations + factor-stat reduction
    # ------------------------------------------------------------------

    def _conditional_expectations(self, x: jax.Array):
        r"""Return :math:`(E[\log Y \mid x],\, E[1/Y \mid x],\,
        E[Y \mid x])` for a single observation."""
        from normix.distributions.generalized_inverse_gaussian import GIG
        z2, w2, _zw = self._quad_forms(x)
        p_post, a_post, b_post = self._floored_posterior_gig_params(z2, w2)
        gig = GIG(p=p_post, a=a_post, b=b_post)
        eta = gig.expectation_params()
        return eta[0], eta[1], eta[2]

    def e_step(self, X: jax.Array, *, backend: str = 'jax') -> FactorMixtureStats:
        r"""Full E-step: posterior :math:`Y` expectations + deterministic
        :math:`Z` reductions.

        Returns a :class:`~normix.fitting.eta.FactorMixtureStats`
        whose first six fields are batch averages of the standard normal
        variance-mean mixture sufficient statistics, and whose four
        :math:`Z`-fields are computed from the first six via the
        deterministic relations in
        ``docs/theory/factor_analysis.rst`` §E-Step (no extra Bessel
        evaluations).
        """
        if backend == 'cpu':
            sub_exp = self._e_step_subordinator_cpu(X)
        elif backend == 'jax':
            sub_exp = self._e_step_subordinator_jax(X)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        return self._aggregate_stats(X, sub_exp)

    def _e_step_subordinator_jax(self, X: jax.Array):
        E_log_Y, E_inv_Y, E_Y = jax.vmap(
            self._conditional_expectations)(X)
        return E_log_Y, E_inv_Y, E_Y

    def _e_step_subordinator_cpu(self, X: jax.Array):
        """CPU path: per-observation quad forms in JAX (vmapped) plus
        GIG Bessel via scipy."""
        from normix.distributions.generalized_inverse_gaussian import GIG

        X = jnp.asarray(X, dtype=jnp.float64)

        def _z2(x):
            return self._quad_forms(x)[0]

        z2_all = jax.vmap(_z2)(X)
        w2 = self._quad_form(self.gamma)
        # The posterior map returns scalars or arrays matching z2_all;
        # broadcasting handles both.
        p_post, a_post, b_post = self._floored_posterior_gig_params(z2_all, w2)
        n = X.shape[0]
        p_post = jnp.broadcast_to(p_post, (n,))
        a_post = jnp.broadcast_to(a_post, (n,))
        b_post = jnp.broadcast_to(b_post, (n,))
        eta = GIG.expectation_params_batch(
            p_post, a_post, b_post, backend='cpu')
        return eta[:, 0], eta[:, 1], eta[:, 2]

    def _aggregate_stats(
        self,
        X: jax.Array,
        sub_exp: Tuple[jax.Array, jax.Array, jax.Array],
    ) -> FactorMixtureStats:
        """Average per-obs expectations into the six classical sums and
        compute the four factor-related statistics deterministically."""
        X = jnp.asarray(X, dtype=jnp.float64)
        E_log_Y, E_inv_Y, E_Y = sub_exp

        s1 = jnp.mean(E_inv_Y)
        s2 = jnp.mean(E_Y)
        s3 = jnp.mean(E_log_Y)
        s4 = jnp.mean(X, axis=0)
        s5 = jnp.mean(X * E_inv_Y[:, None], axis=0)
        s6 = jnp.mean(jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y), axis=0)

        s7, s8, s9, s10 = self._z_stats_from_six(s1, s2, s4, s5, s6)
        return FactorMixtureStats(
            E_inv_Y=s1, E_Y=s2, E_log_Y=s3,
            E_X=s4, E_X_inv_Y=s5, E_XXT_inv_Y=s6,
            E_XZT_inv_sqrtY=s7, E_Z_inv_sqrtY=s8,
            E_Z_sqrtY=s9, E_ZZT=s10,
        )

    def _z_stats_from_six(
        self,
        s1: jax.Array, s2: jax.Array,
        s4: jax.Array, s5: jax.Array, s6: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Deterministic relations from
        ``docs/theory/factor_analysis.rst`` §E-Step. Inputs are the
        already-aggregated first six statistics; outputs are
        :math:`(s_7, s_8, s_9, s_{10})`.

        Computed at the *current* model parameters (μ, γ, F, β); β is
        evaluated once via :meth:`_beta`.
        """
        mu = self.mu
        gamma = self.gamma
        beta = self._beta()                              # (r, d)

        # s_7 = (s_6 - s_5 μᵀ - s_4 γᵀ) βᵀ                shape (d, r)
        s7 = (s6 - jnp.outer(s5, mu) - jnp.outer(s4, gamma)) @ beta.T

        # s_8 = β (s_5 - μ s_1 - γ)                       shape (r,)
        s8 = beta @ (s5 - mu * s1 - gamma)

        # s_9 = β (s_4 - μ - γ s_2)                       shape (r,)
        s9 = beta @ (s4 - mu - gamma * s2)

        # s_10 = I_r - β F + β · core · βᵀ                shape (r, r)
        # core = s_6 - s_5 μᵀ - μ s_5ᵀ + μ μᵀ s_1
        #        - (s_4 - μ) γᵀ - γ (s_4 - μ)ᵀ + γ γᵀ s_2
        core = (s6
                - jnp.outer(s5, mu) - jnp.outer(mu, s5)
                + jnp.outer(mu, mu) * s1
                - jnp.outer(s4 - mu, gamma)
                - jnp.outer(gamma, s4 - mu)
                + jnp.outer(gamma, gamma) * s2)
        s10 = (jnp.eye(self.r, dtype=jnp.float64)
               - beta @ self.F
               + beta @ core @ beta.T)
        # Symmetrise to remove float64 roundoff (s10 is symmetric in math).
        s10 = 0.5 * (s10 + s10.T)
        return s7, s8, s9, s10

    # ------------------------------------------------------------------
    # M-step: closed-form FA update on (μ, γ, F, D)
    # ------------------------------------------------------------------

    @staticmethod
    def _mstep_factor_params(
        eta: FactorMixtureStats,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Closed-form M-step for :math:`(\mu, \gamma, F, D)` from
        :class:`FactorMixtureStats`. Implements equations
        :eq:`fa-aux` and :eq:`fa-mstep` of
        ``docs/theory/factor_analysis.rst``.
        """
        s1 = eta.E_inv_Y
        s2 = eta.E_Y
        s4 = eta.E_X
        s5 = eta.E_X_inv_Y
        s6 = eta.E_XXT_inv_Y
        s7 = eta.E_XZT_inv_sqrtY
        s8 = eta.E_Z_inv_sqrtY
        s9 = eta.E_Z_sqrtY
        s10 = eta.E_ZZT
        d = s4.shape[0]

        # s_10⁻¹ applied to the small column blocks we need.
        s10_inv_s8 = jax.scipy.linalg.solve(s10, s8, assume_a='pos')
        s10_inv_s9 = jax.scipy.linalg.solve(s10, s9, assume_a='pos')
        s10_inv_s7T = jax.scipy.linalg.solve(s10, s7.T, assume_a='pos')

        q1 = jnp.dot(s8, s10_inv_s8) - s1
        q2 = jnp.dot(s9, s10_inv_s8) - 1.0
        q3 = jnp.dot(s9, s10_inv_s9) - s2
        q4 = s10_inv_s7T.T @ s8 - s5     # (d,)
        q5 = s10_inv_s7T.T @ s9 - s4     # (d,)

        denom = q2 ** 2 - q1 * q3
        mu = (q2 * q5 - q3 * q4) / denom
        gamma = (q2 * q4 - q1 * q5) / denom

        # F = (s_7 - μ s_8ᵀ - γ s_9ᵀ) s_10⁻¹              shape (d, r)
        F = jax.scipy.linalg.solve(
            s10, (s7 - jnp.outer(mu, s8) - jnp.outer(gamma, s9)).T,
            assume_a='pos',
        ).T

        # D = diag(R) where R is the long expression of §M-Step.
        # We only need diag(R); compute term by term.
        F_s8 = F @ s8                # (d,)
        F_s9 = F @ s9                # (d,)
        F_s10_FT_diag = jnp.einsum('ij,jk,ik->i', F, s10, F)

        diag_R = (s1 * mu * mu
                  + s2 * gamma * gamma
                  - 2.0 * s4 * gamma
                  - 2.0 * s5 * mu
                  + jnp.diag(s6)
                  - 2.0 * jnp.einsum('ij,ij->i', s7, F)
                  + 2.0 * F_s8 * mu
                  + 2.0 * F_s9 * gamma
                  + F_s10_FT_diag
                  + 2.0 * mu * gamma)

        D = jnp.maximum(diag_R, D_FLOOR)
        return mu, gamma, F, D

    def m_step(self, eta: FactorMixtureStats, **kwargs) -> "FactorNormalMixture":
        r"""Full M-step: update :math:`(\mu, \gamma, F, D)` and the
        subordinator from :class:`FactorMixtureStats`.

        Equivalent to ``self.m_step_normal(eta).m_step_subordinator(eta,
        **kwargs)``. Subclasses with iterative subordinator solvers (e.g.
        :class:`~normix.distributions.generalized_hyperbolic.FactorGeneralizedHyperbolic`)
        may override to inject a warm-start.
        """
        return self.m_step_normal(eta).m_step_subordinator(eta, **kwargs)

    def m_step_normal(self, eta: FactorMixtureStats) -> "FactorNormalMixture":
        r"""M-step for :math:`(\mu, \gamma, F, D)` only (MCECM cycle 1).

        Subordinator unchanged.
        """
        mu_new, gamma_new, F_new, D_new = self._mstep_factor_params(eta)
        return self._replace_factor_params(mu_new, gamma_new, F_new, D_new)

    def m_step_subordinator(
        self, eta: FactorMixtureStats, **kwargs,
    ) -> "FactorNormalMixture":
        r"""M-step for the subordinator only (MCECM cycle 2).

        Reads the subordinator-relevant fields of ``eta``;
        :math:`(\mu, \gamma, F, D)` are read from ``self`` and copied
        unchanged. Subclasses with iterative solvers may override to
        add a warm-start or sanity-check fallback.
        """
        sub = type(self)._subordinator_from_eta(
            eta, theta0=self.subordinator.natural_params(), **kwargs)
        return self._with_subordinator(sub)

    # ------------------------------------------------------------------
    # Immutable update helpers
    # ------------------------------------------------------------------

    def _replace_factor_params(
        self,
        mu: jax.Array, gamma: jax.Array, F: jax.Array, D: jax.Array,
    ) -> "FactorNormalMixture":
        """Return a copy with (mu, gamma, F, D) replaced; subordinator unchanged."""
        return eqx.tree_at(
            lambda m: (m.mu, m.gamma, m.F, m.D),
            self,
            (mu, gamma, F, D),
        )

    def _with_subordinator(
        self, subordinator: ExponentialFamily,
    ) -> "FactorNormalMixture":
        """Return a copy with a new subordinator, all else unchanged."""
        return eqx.tree_at(lambda m: m.subordinator, self, subordinator)

    # ------------------------------------------------------------------
    # Eta from model (incremental EM warm-start)
    # ------------------------------------------------------------------

    def compute_eta_from_model(self) -> FactorMixtureStats:
        r"""Reconstruct :class:`FactorMixtureStats` from the model's own
        parameters.

        The first six fields use the marginal expectations of the joint
        sufficient statistics under the current :math:`\Sigma = F F^\top
        + \mathrm{diag}(D)`; the four :math:`Z`-fields use the same
        deterministic relations as :meth:`e_step` (§E-Step of the
        theory doc).
        """
        E_log_Y, E_inv_Y, E_Y = self._subordinator_expectations()
        mu = self.mu
        gamma = self.gamma
        Sigma = self.sigma()

        s4 = mu + gamma * E_Y
        s5 = mu * E_inv_Y + gamma
        s6 = (Sigma
              + jnp.outer(mu, mu) * E_inv_Y
              + jnp.outer(gamma, gamma) * E_Y
              + jnp.outer(mu, gamma) + jnp.outer(gamma, mu))
        s7, s8, s9, s10 = self._z_stats_from_six(E_inv_Y, E_Y, s4, s5, s6)
        return FactorMixtureStats(
            E_inv_Y=E_inv_Y, E_Y=E_Y, E_log_Y=E_log_Y,
            E_X=s4, E_X_inv_Y=s5, E_XXT_inv_Y=s6,
            E_XZT_inv_sqrtY=s7, E_Z_inv_sqrtY=s8,
            E_Z_sqrtY=s9, E_ZZT=s10,
        )

    # ------------------------------------------------------------------
    # Convergence hook
    # ------------------------------------------------------------------

    def em_convergence_params(self):
        r"""Return :math:`(\mu, \gamma, \Sigma)` for the convergence
        check.

        :math:`\Sigma = F F^\top + \mathrm{diag}(D)` rather than ``F``
        directly, because ``F`` is identifiable only up to an
        :math:`r \times r` orthogonal rotation and would never converge
        in norm.
        """
        return (self.mu, self.gamma, self.sigma())

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def _rescale(self, scale: jax.Array) -> "FactorNormalMixture":
        r"""Apply :math:`Y \to s\,Y` reparameterisation.

        Pushes the scale through :math:`(\gamma, F, D)` and the
        subordinator: :math:`\gamma \to \gamma/s`,
        :math:`F \to F/\sqrt{s}`, :math:`D \to D/s` (so
        :math:`\Sigma \to \Sigma/s`); the subordinator-side rescale is
        delegated to :meth:`_build_rescaled`.
        """
        F_new = self.F / jnp.sqrt(scale)
        D_new = self.D / scale
        gamma_new = self.gamma / scale
        return self._build_rescaled(self.mu, gamma_new, F_new, D_new, scale)

    def regularize_det_sigma(
        self, target_log_det: float = 0.0,
    ) -> "FactorNormalMixture":
        r"""Rescale to enforce :math:`\log|\Sigma| = \mathrm{target\_log\_det}`.

        Same family as the full-Σ implementation but the
        log-determinant is computed via the Woodbury identity
        :math:`\log|\Sigma| = \log|D| + \log|I_r + F^\top D^{-1} F|`.
        """
        log_scale = (self._log_det_sigma() - target_log_det) / self.d
        return self._rescale(jnp.exp(log_scale))

    def regularize_det_sigma_one(self) -> "FactorNormalMixture":
        r"""Enforce :math:`|\Sigma| = 1`. Alias for
        :meth:`regularize_det_sigma` with ``target_log_det = 0``.
        """
        return self.regularize_det_sigma(0.0)

    def regularize_a_eq_b(self) -> "FactorNormalMixture":
        r"""Rescale subordinator so that :math:`a = b = \sqrt{ab}`.

        Default no-op; overridden in
        :class:`~normix.distributions.generalized_hyperbolic.FactorGeneralizedHyperbolic`
        and
        :class:`~normix.distributions.normal_inverse_gaussian.FactorNormalInverseGaussian`.
        """
        return self

    @abc.abstractmethod
    def _build_rescaled(
        self,
        mu: jax.Array,
        gamma_new: jax.Array,
        F_new: jax.Array,
        D_new: jax.Array,
        scale: jax.Array,
    ) -> "FactorNormalMixture":
        """Construct a new model with rescaled :math:`(F, D, \\gamma)`
        and a subordinator scaled to compensate (so :math:`Y\\Sigma`
        keeps the correct dispersion)."""

    # ------------------------------------------------------------------
    # Default initialisation
    # ------------------------------------------------------------------

    @classmethod
    def default_init(
        cls, X: jax.Array, *, r: int = 1,
    ) -> "FactorNormalMixture":
        """Moment-based initialisation from data with ``r`` factors.

        Sets :math:`\\mu = \\bar X`, :math:`\\gamma = 0`, and splits the
        empirical covariance into a rank-``r`` factor block (top-``r``
        eigenvectors scaled by :math:`\\sqrt{\\lambda - \\bar\\lambda}`)
        plus a positive diagonal residual.

        Useful as a starting point for ``model.fit(X)``.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        n, d = X.shape
        mu = jnp.mean(X, axis=0)
        Xc = X - mu
        S = (Xc.T @ Xc) / n + SIGMA_INIT_REG * jnp.eye(d)

        eigvals, eigvecs = jnp.linalg.eigh(S)            # ascending
        idx = jnp.argsort(eigvals)[::-1]                 # descending
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        top = eigvecs[:, :r]
        top_vals = eigvals[:r]
        rest_mean = jnp.mean(eigvals[r:]) if r < d else jnp.float64(D_FLOOR)
        scaled_vals = jnp.maximum(top_vals - rest_mean, D_FLOOR)
        F = top * jnp.sqrt(scaled_vals)[None, :]         # (d, r)
        D = jnp.maximum(jnp.diag(S) - jnp.sum(F ** 2, axis=1), D_FLOOR)

        gamma = jnp.zeros(d, dtype=jnp.float64)
        return cls._from_init_params(mu, gamma, F, D)

    @classmethod
    @abc.abstractmethod
    def _from_init_params(
        cls,
        mu: jax.Array, gamma: jax.Array, F: jax.Array, D: jax.Array,
    ) -> "FactorNormalMixture":
        """Construct a model with default subordinator parameters."""

    # ------------------------------------------------------------------
    # replace(**) — immutable update on top-level fields
    # ------------------------------------------------------------------

    _NORMAL_KEYS = ('mu', 'gamma', 'F', 'D', 'subordinator')

    def replace(self, **updates) -> "FactorNormalMixture":
        r"""Return a new model with selected top-level fields replaced.

        Accepts any subset of :attr:`_NORMAL_KEYS`. Subclass-specific
        subordinator parameter shortcuts (e.g. ``alpha=...``) are not
        supported on the FA family — use ``replace(subordinator=
        Gamma(alpha=..., beta=...))`` instead. This keeps the contract
        narrow (one storage form per family).
        """
        unknown = set(updates) - set(self._NORMAL_KEYS)
        if unknown:
            raise ValueError(
                f"unknown field(s) {sorted(unknown)} for "
                f"{type(self).__name__}; valid keys are "
                f"{sorted(self._NORMAL_KEYS)}")
        if not updates:
            return self
        keys = tuple(updates.keys())
        new_values = tuple(updates[k] for k in keys)
        return eqx.tree_at(
            lambda m: tuple(getattr(m, k) for k in keys),
            self, new_values,
        )
