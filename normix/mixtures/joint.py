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

jax.config.update("jax_enable_x64", True)

from normix.utils.constants import LOG_EPS, SAFE_DENOMINATOR, SIGMA_REG


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
        Compute :math:`E[g(Y)\mid X=x]` for EM E-step.

        The posterior :math:`Y\mid X` follows a GIG-like distribution:

        .. math::

            p_{\mathrm{post}} = p_{\mathrm{eff}} - d/2, \quad
            a_{\mathrm{post}} = a_{\mathrm{eff}} + \gamma^\top\Sigma^{-1}\gamma, \quad
            b_{\mathrm{post}} = b_{\mathrm{eff}} + (x-\mu)^\top\Sigma^{-1}(x-\mu)

        Returns dict with keys: ``E_log_Y``, ``E_inv_Y``, ``E_Y``.
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

    def _precision_quantities(self):
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
        self, theta_1: jax.Array, theta_2: jax.Array, theta_3: jax.Array,
    ) -> jax.Array:
        r"""
        Assemble the full :math:`\theta` vector from scalars :math:`\theta_1, \theta_2, \theta_3`
        and shared Cholesky quantities.

        :math:`\theta = [\theta_1, \theta_2, \theta_3, \Sigma^{-1}\gamma, \Sigma^{-1}\mu, -\tfrac{1}{2}\mathrm{vec}(\Sigma^{-1})]`
        """
        Lambda_mu, Lambda_gamma, _, _, Lambda = self._precision_quantities()
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
    def _parse_joint_theta(theta: jax.Array):
        """
        Parse a joint theta vector into its components.

        Returns (d, theta_1..3, theta_4..6, Lambda, log_det_Sigma, mu, gamma,
                 mu_quad, gamma_quad, mu_Lambda_gamma).
        """
        n = theta.shape[0]
        d = JointNormalMixture._dim_from_theta_length(n)

        theta_1 = theta[0]
        theta_2 = theta[1]
        theta_3 = theta[2]
        theta_4 = theta[3:3 + d]
        theta_5 = theta[3 + d:3 + 2 * d]
        theta_6 = theta[3 + 2 * d:].reshape(d, d)

        Lambda = -2.0 * theta_6
        Lambda = 0.5 * (Lambda + Lambda.T)

        _sign, log_det_Lambda = jnp.linalg.slogdet(Lambda)
        log_det_Sigma = -log_det_Lambda

        mu = jnp.linalg.solve(Lambda, theta_5)
        gamma = jnp.linalg.solve(Lambda, theta_4)

        mu_quad = 0.5 * jnp.dot(mu, theta_5)
        gamma_quad = 0.5 * jnp.dot(gamma, theta_4)
        mu_Lambda_gamma = jnp.dot(mu, theta_4)

        return (d, theta_1, theta_2, theta_3, theta_4, theta_5,
                Lambda, log_det_Sigma, mu, gamma,
                mu_quad, gamma_quad, mu_Lambda_gamma)

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

        safe_D = jnp.where(jnp.abs(D) > SAFE_DENOMINATOR, D, SAFE_DENOMINATOR)

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
