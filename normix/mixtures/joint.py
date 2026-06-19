"""
JointNormalMixture — abstract exponential family for normal variance-mean mixtures.

Joint distribution :math:`f(x, y)`:

.. math::

    X \\mid Y \\sim \\mathcal{N}(\\mu + \\gamma y,\\; \\Sigma y), \\quad
    Y \\sim \\text{subordinator (GIG, Gamma, InvGamma, InvGaussian)}

**Sufficient statistics:**

.. math::

    t(x, y) = [\\log y,\\; 1/y,\\; y,\\; x,\\; x/y,\\; \\mathrm{vec}(xx^\\top/y)]

**Natural parameters:**

.. math::

    \\theta_1 = p_{\\mathrm{sub}} - 1 - d/2, \\quad
    \\theta_2 = -(b_{\\mathrm{sub}}/2 + \\tfrac{1}{2}\\mu^\\top\\Sigma^{-1}\\mu) < 0

.. math::

    \\theta_3 = -(a_{\\mathrm{sub}}/2 + \\tfrac{1}{2}\\gamma^\\top\\Sigma^{-1}\\gamma) < 0, \\quad
    \\theta_4 = \\Sigma^{-1}\\gamma, \\quad
    \\theta_5 = \\Sigma^{-1}\\mu, \\quad
    \\theta_6 = -\\tfrac{1}{2}\\mathrm{vec}(\\Sigma^{-1})

For a GIG subordinator, :math:`\\theta_2,\\theta_3` combine :math:`-b/2,-a/2` with the
normal quadratic forms so that :math:`\\theta^{\\top} t` matches
:math:`-(a y + b/y)/2` from :math:`f_Y` plus the :math:`y`-dependent terms from
:math:`f_{X\\mid Y}`.

**Log-partition:**

.. math::

    \\psi = \\psi_{\\mathrm{sub}}(p, a, b) + \\tfrac{1}{2}\\log|\\Sigma| + \\mu^\\top\\Sigma^{-1}\\gamma

**Expectation parameters (EM E-step quantities):**

.. math::

    \\eta_1 = E[\\log Y], \\quad \\eta_2 = E[1/Y], \\quad \\eta_3 = E[Y]

.. math::

    \\eta_4 = E[X] = \\mu + \\gamma E[Y], \\quad
    \\eta_5 = E[X/Y] = \\mu E[1/Y] + \\gamma

.. math::

    \\eta_6 = E[XX^\\top/Y] = \\Sigma + \\mu\\mu^\\top E[1/Y]
    + \\gamma\\gamma^\\top E[Y] + \\mu\\gamma^\\top + \\gamma\\mu^\\top

**EM M-step closed-form** (let :math:`D = 1 - E[1/Y] \\cdot E[Y]`):

.. math::

    \\mu = \\frac{E[X] - E[Y] E[X/Y]}{D}, \\quad
    \\gamma = \\frac{E[X/Y] - E[1/Y] E[X]}{D}

.. math::

    \\Sigma = E[XX^\\top/Y] - E[X/Y]\\mu^\\top - \\mu E[X/Y]^\\top
    + E[1/Y]\\mu\\mu^\\top - E[Y]\\gamma\\gamma^\\top
"""
from __future__ import annotations

import abc
from typing import Dict, Tuple

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily


from normix.utils.constants import B_POST_FLOOR, LOG_EPS, SAFE_DENOMINATOR, SIGMA_REG


class JointNormalMixture(ExponentialFamily):
    r"""
    Abstract joint distribution :math:`f(x, y)` for normal variance-mean mixtures.

    Stored: ``mu`` (d,), ``gamma`` (d,), ``L_Sigma`` (d×d lower Cholesky of :math:`\Sigma`).
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

    # ------------------------------------------------------------------
    # Derived from subclass
    # ------------------------------------------------------------------

    @property
    def d(self) -> int:
        return int(self.mu.shape[0])

    def sigma(self) -> jax.Array:
        r"""Covariance matrix :math:`\Sigma = L_\Sigma L_\Sigma^\top`."""
        return self.L_Sigma @ self.L_Sigma.T

    def log_det_sigma(self) -> jax.Array:
        r""":math:`\log|\Sigma| = 2\sum_i \log L_{ii}`, via Cholesky diagonal."""
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.L_Sigma)))

    def _mu_Lambda_gamma(self) -> jax.Array:
        r""":math:`\mu^\top \Sigma^{-1} \gamma` via Cholesky."""
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.mu, lower=True)
        w = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)
        return jnp.dot(z, w)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rvs(self, n: int, seed: int = 42) -> Tuple[jax.Array, jax.Array]:
        """
        Sample :math:`(X, Y)` from the joint distribution via JAX PRNG.

        Returns
        -------
        X : jax.Array
            Shape ``(n, d)``.
        Y : jax.Array
            Shape ``(n,)``.
        """
        Y = self.subordinator().rvs(n, seed)
        key = jax.random.PRNGKey(seed + 1)
        d = self.d
        Z = jax.random.normal(key, shape=(n, d), dtype=jnp.float64)
        X = (self.mu[None, :]
             + self.gamma[None, :] * Y[:, None]
             + jnp.sqrt(Y[:, None]) * (Z @ self.L_Sigma.T))
        return X, Y

    # ------------------------------------------------------------------
    # Log-prob for joint (x, y)
    # ------------------------------------------------------------------

    def log_prob_joint(self, x: jax.Array, y: jax.Array) -> jax.Array:
        r"""
        :math:`\log f(x, y) = \log f(x\mid y) + \log f_Y(y)`.

        .. math::

            \log f(x\mid y) = -\tfrac{d}{2}\log(2\pi) - \tfrac{1}{2}\log|\Sigma|
            - \tfrac{d}{2}\log y - \frac{1}{2y}\|L^{-1}(x-\mu)\|^2
            + \gamma^\top\Sigma^{-1}(x-\mu) - \tfrac{y}{2}\gamma^\top\Sigma^{-1}\gamma

        :math:`\log f_Y(y)` from subordinator.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        d = self.d

        # Residual r = x - μ, solve L_Sigma z = r
        r = x - self.mu
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, r, lower=True)
        # Solve L_Sigma w = γ
        w = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)

        log_det_sigma = self.log_det_sigma()

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
        r"""
        Compute :math:`E[g(Y)\mid X=x]` for the EM E-step.

        The posterior :math:`Y\mid X=x` is

        .. math::

            \mathrm{GIG}\!\left(p - \tfrac{d}{2},\;
            a + \gamma^\top\Sigma^{-1}\gamma,\;
            b + (x-\mu)^\top\Sigma^{-1}(x-\mu)\right),

        with the family-specific prior :math:`(p, a, b)` resolved by
        :meth:`_posterior_gig_params`. Returns a dict with keys
        ``E_log_Y``, ``E_inv_Y``, ``E_Y``.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return self._compute_posterior_expectations(x)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        r"""Posterior :math:`(E[\log Y], E[1/Y], E[Y]\mid x)` via the GIG moments.

        The posterior scale :math:`b_{\mathrm{post}} = b +
        (x-\mu)^\top\Sigma^{-1}(x-\mu)` is floored at
        :data:`~normix.utils.constants.B_POST_FLOOR`, which bounds
        :math:`E[1/Y\mid x]` for observations near the mode. The floor only
        binds for VG (prior :math:`b=0`); see :doc:`../docs/theory/em_algorithm`.
        """
        from normix.distributions.generalized_inverse_gaussian import GIG
        _z, _w, z2, w2, _zw = self._quad_forms(x)
        p_post, a_post, b_post = self._floored_posterior_gig_params(z2, w2)
        eta = GIG(p=p_post, a=a_post, b=b_post).expectation_params()
        return {'E_log_Y': eta[0], 'E_inv_Y': eta[1], 'E_Y': eta[2]}

    def _posterior_gig_params(self, z2: jax.Array, w2: jax.Array):
        r"""Prior-to-posterior GIG conjugacy map, uniform across families.

        Returns :math:`(p_{\mathrm{post}}, a_{\mathrm{post}},
        b_{\mathrm{post}})` from quad-form scalars
        :math:`z_2 = (x-\mu)^\top\Sigma^{-1}(x-\mu)` and
        :math:`w_2 = \gamma^\top\Sigma^{-1}\gamma`:

        .. math::

            p_{\mathrm{post}} = p_{\mathrm{gig}} - \tfrac{d}{2}, \quad
            a_{\mathrm{post}} = a_{\mathrm{gig}} + w_2, \quad
            b_{\mathrm{post}} = b_{\mathrm{gig}} + z_2,

        where :math:`(p_{\mathrm{gig}}, a_{\mathrm{gig}}, b_{\mathrm{gig}})`
        are the subordinator's exact GIG coordinates (``subordinator().to_gig()``).
        Per family:

        ============= ============================================ ===================================
        Subordinator  :math:`(p_{\mathrm{gig}}, a, b)`              :math:`(p_{\mathrm{post}}, a, b)`
        ============= ============================================ ===================================
        Gamma (VG)    :math:`(\alpha,\ 2\beta,\ 0)`                :math:`(\alpha-\tfrac d2,\ 2\beta+w_2,\ z_2)`
        InvGamma      :math:`(-\alpha,\ 0,\ 2\beta)`               :math:`(-\alpha-\tfrac d2,\ w_2,\ 2\beta+z_2)`
        InvGaussian   :math:`(-\tfrac12,\ \lambda/\mu_{IG}^2,\ \lambda)` :math:`(-\tfrac12-\tfrac d2,\ \lambda/\mu_{IG}^2+w_2,\ \lambda+z_2)`
        GIG (GH)      :math:`(p,\ a,\ b)`                          :math:`(p-\tfrac d2,\ a+w_2,\ b+z_2)`
        ============= ============================================ ===================================

        This map stays **pure** (no floor); the :math:`b_{\mathrm{post}}` floor
        is applied by :meth:`_floored_posterior_gig_params`.
        """
        gig = self.subordinator().to_gig()
        return gig.p - self.d / 2.0, gig.a + w2, gig.b + z2

    def _floored_posterior_gig_params(self, z2: jax.Array, w2: jax.Array):
        r"""E-step entry point: :meth:`_posterior_gig_params` with
        :math:`b_{\mathrm{post}}` floored at
        :data:`~normix.utils.constants.B_POST_FLOOR`.

        The floor bounds :math:`E[1/Y\mid x]` for observations near the mode;
        it only binds for VG (prior :math:`b=0`). This is the single
        chokepoint through which all E-step paths obtain the posterior GIG.
        """
        p_post, a_post, b_post = self._posterior_gig_params(z2, w2)
        return p_post, a_post, jnp.maximum(b_post, B_POST_FLOOR)

    # ------------------------------------------------------------------
    # Helper: solve Cholesky-based quantities
    # ------------------------------------------------------------------

    def _quad_forms(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""
        Compute :math:`z = L_\Sigma^{-1}(x-\mu)`, :math:`w = L_\Sigma^{-1}\gamma`.
        Returns :math:`z, w, \|z\|^2, \|w\|^2, z^\top w`.
        """
        r = x - self.mu
        z = jax.scipy.linalg.solve_triangular(self.L_Sigma, r, lower=True)
        w = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)
        return z, w, jnp.dot(z, z), jnp.dot(w, w), jnp.dot(z, w)

    # ------------------------------------------------------------------
    # Shared natural_params computation
    # ------------------------------------------------------------------

    def _precision_quantities(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""
        Cholesky-based precision decomposition shared by all Joint subclasses.

        Returns :math:`(\Lambda\mu, \Lambda\gamma, \mu_{\mathrm{quad}}, \gamma_{\mathrm{quad}}, \Lambda)`
        where :math:`\Lambda = \Sigma^{-1}` and :math:`x_{\mathrm{quad}} = \tfrac{1}{2}x^\top\Sigma^{-1}x`.
        """
        d = self.d
        z_mu = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.mu, lower=True)
        z_gamma = jax.scipy.linalg.solve_triangular(self.L_Sigma, self.gamma, lower=True)
        Lambda_mu = jax.scipy.linalg.solve_triangular(self.L_Sigma.T, z_mu, lower=False)
        Lambda_gamma = jax.scipy.linalg.solve_triangular(self.L_Sigma.T, z_gamma, lower=False)
        mu_quad = 0.5 * jnp.dot(self.mu, Lambda_mu)
        gamma_quad = 0.5 * jnp.dot(self.gamma, Lambda_gamma)
        L_inv = jax.scipy.linalg.solve_triangular(
            self.L_Sigma, jnp.eye(d, dtype=jnp.float64), lower=True)
        Lambda = L_inv.T @ L_inv
        return Lambda_mu, Lambda_gamma, mu_quad, gamma_quad, Lambda

    def _assemble_natural_params(
          self,
        theta_1: jax.Array, theta_2: jax.Array, theta_3: jax.Array,
        Lambda_mu: jax.Array, Lambda_gamma: jax.Array, Lambda: jax.Array,
    ) -> jax.Array:
        r"""
        Assemble the full :math:`\theta` vector from subordinator scalars
        and precomputed precision components (from :meth:`_precision_quantities`).
        """
        return jnp.concatenate([
            jnp.array([theta_1, theta_2, theta_3]),
            Lambda_gamma,
            Lambda_mu,
            (-0.5 * Lambda).ravel(),
        ])

    @staticmethod
    def _dim_from_theta_length(n: int) -> int:
        """Recover dimensionality d from |θ| = 3 + 2d + d²."""
        return int(-1 + (1 + 4 * (n - 3)) ** 0.5) // 2

    @staticmethod
    def _recover_normal_params(theta: jax.Array):
        """
        Recover (d, mu, gamma, L_Sigma) from a joint theta vector.

        Delegates Λ → (μ, L_Σ) to :meth:`MultivariateNormal.from_natural`,
        then γ = Σ θ₄ = L_Σ L_Σᵀ θ₄.
        """
        from normix.distributions.normal import MultivariateNormal

        n = theta.shape[0]
        d = JointNormalMixture._dim_from_theta_length(n)

        mvn = MultivariateNormal.from_natural(
            jnp.concatenate([theta[3 + d:3 + 2 * d], theta[3 + 2 * d:]]))
        gamma = mvn.L_Sigma @ (mvn.L_Sigma.T @ theta[3:3 + d])
        return d, mvn.mu, gamma, mvn.L_Sigma

    # ------------------------------------------------------------------
    # ExponentialFamily abstract methods
    # ------------------------------------------------------------------

    @staticmethod
    def sufficient_statistics(xy: jax.Array) -> jax.Array:
        r"""
        :math:`t(x,y) = [\log y,\; 1/y,\; y,\; x,\; x/y,\; \mathrm{vec}(xx^\top/y)]`.

        Input: flat vector :math:`[x_1,\ldots,x_d, y]`.
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
    # η → model: closed-form M-step as exponential-family inversion
    # ------------------------------------------------------------------

    @classmethod
    def from_expectation(cls, eta, **kwargs) -> "JointNormalMixture":
        r"""Construct from expectation parameters :math:`\eta`.

        Two input forms are supported:

        - :class:`~normix.fitting.eta.NormalMixtureEta` — the natural η
          pytree of the joint normal-variance-mean mixture; uses the
          closed-form M-step (:math:`\mu, \gamma, \Sigma` analytical;
          subordinator via :meth:`_subordinator_from_eta`).
        - flat ``jax.Array`` — the generic Bregman solver inherited from
          :class:`~normix.exponential_family.ExponentialFamily`.

        The pytree path is the canonical η→θ map for these distributions:
        it is exact (no Bregman iterations on the normal block) and uses
        the subordinator's own ``from_expectation`` (closed-form for
        Gamma / InverseGamma / InverseGaussian; numerical for GIG).

        Parameters
        ----------
        eta : NormalMixtureEta or jax.Array
            Expectation parameters.
        **kwargs
            For the pytree path: forwarded to
            :meth:`_subordinator_from_eta` (e.g. ``backend``, ``method``,
            ``maxiter``, ``theta0`` for warm-starting GIG).
            For the flat-array path: forwarded to the parent solver.
        """
        from normix.fitting.eta import NormalMixtureEta

        if isinstance(eta, NormalMixtureEta):
            mu, gamma, L_Sigma = cls._mstep_normal_params(eta)
            sub = cls._subordinator_from_eta(eta, **kwargs)
            return cls._from_normal_and_subordinator(mu, gamma, L_Sigma, sub)
        return super().from_expectation(eta, **kwargs)

    @classmethod
    @abc.abstractmethod
    def _subordinator_from_eta(
        cls, eta: "NormalMixtureEta", *, theta0=None, **kwargs,
    ) -> ExponentialFamily:
        r"""Fit the subordinator from the marginal expectation pytree.

        Reads the subordinator-relevant fields of ``eta`` (a subset of
        :math:`E[\log Y], E[1/Y], E[Y]`) and returns a fitted instance of
        the appropriate subordinator family. ``theta0`` is forwarded to
        the subordinator's solver as a warm-start; subordinators with
        closed-form ``from_expectation`` may ignore it.
        """

    @classmethod
    @abc.abstractmethod
    def _from_normal_and_subordinator(
        cls,
        mu: jax.Array,
        gamma: jax.Array,
        L_Sigma: jax.Array,
        subordinator: ExponentialFamily,
    ) -> "JointNormalMixture":
        """Construct from normal parameters and a fitted subordinator."""

    # ------------------------------------------------------------------
    # M-step: closed-form normal parameter update
    # ------------------------------------------------------------------

    @staticmethod
    def _mstep_normal_params(eta: "NormalMixtureEta"):
        r"""
        Closed-form M-step for :math:`\mu, \gamma, \Sigma` from expectation parameters.

        Parameters
        ----------
        eta : NormalMixtureEta
            Aggregated expectation parameters.

        Returns ``(mu_new, gamma_new, L_new)``.
        """
        from normix.fitting.eta import NormalMixtureEta  # noqa: F811

        D = 1.0 - eta.E_inv_Y * eta.E_Y

        # Cauchy–Schwarz gives D <= 0 always; floor at -SAFE_DENOMINATOR so a
        # tiny |D| (or roundoff D > 0) never flips the sign of mu, gamma.
        safe_D = -jnp.maximum(jnp.abs(D), SAFE_DENOMINATOR)

        mu_new = (eta.E_X - eta.E_Y * eta.E_X_inv_Y) / safe_D
        gamma_new = (eta.E_X_inv_Y - eta.E_inv_Y * eta.E_X) / safe_D

        Sigma = (eta.E_XXT_inv_Y
                 - jnp.outer(eta.E_X_inv_Y, mu_new)
                 - jnp.outer(mu_new, eta.E_X_inv_Y)
                 + eta.E_inv_Y * jnp.outer(mu_new, mu_new)
                 - eta.E_Y * jnp.outer(gamma_new, gamma_new))

        Sigma = 0.5 * (Sigma + Sigma.T)
        d = Sigma.shape[0]
        Sigma = Sigma + SIGMA_REG * jnp.eye(d)
        L_new = jnp.linalg.cholesky(Sigma)

        return mu_new, gamma_new, L_new
