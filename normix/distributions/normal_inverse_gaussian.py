"""
Normal-Inverse Gaussian (NIG) distribution.

Special case of GH with GIG → InverseGaussian subordinator (:math:`p = -1/2`).
:math:`Y \\sim \\mathrm{InvGaussian}(\\mu_{IG}, \\lambda)`,
i.e. GIG(:math:`p = -1/2`, :math:`a = \\lambda/\\mu_{IG}^2`, :math:`b = \\lambda`).

Stored: :math:`\\mu`, :math:`\\gamma`, :math:`L_\\Sigma` (Cholesky of :math:`\\Sigma`),
:math:`\\mu_{IG}` (IG mean), :math:`\\lambda` (IG shape).
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from normix.exponential_family import ExponentialFamily
from normix.mixtures.factor import FactorNormalMixture
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture, _UnivariateNormalMixtureMixin
from normix.utils.bessel import log_kv
from normix.utils.constants import LOG_EPS


class JointNormalInverseGaussian(JointNormalMixture):
    r"""
    Joint :math:`f(x,y)`: :math:`X\mid Y \sim \mathcal{N}(\mu+\gamma y, \Sigma y)`,
    :math:`Y \sim \mathrm{InvGaussian}(\mu_{IG}, \lambda)`.

    Stored: :math:`\mu_{IG}` (IG mean) and :math:`\lambda` (IG shape) directly.
    GIG params: :math:`p = -1/2`, :math:`a = \lambda/\mu_{IG}^2`, :math:`b = \lambda`.
    """

    mu_ig: jax.Array   # IG mean parameter
    lam: jax.Array     # IG shape parameter

    def __init__(self, mu, gamma, L_Sigma, mu_ig, lam):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'mu_ig', jnp.asarray(mu_ig, dtype=jnp.float64))
        object.__setattr__(self, 'lam', jnp.asarray(lam, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.inverse_gaussian import InverseGaussian
        return InverseGaussian(mu=self.mu_ig, lam=self.lam)

    def _posterior_gig_params(
        self, z2: jax.Array, w2: jax.Array
    ):
        r"""Posterior :math:`Y\mid X=x \sim \mathrm{GIG}(-1/2-d/2, a_{\mathrm{post}}, b_{\mathrm{post}})`:

        .. math::

            a_{\mathrm{post}} = \lambda/\mu_{IG}^2 + \gamma^\top\Sigma^{-1}\gamma, \quad
            b_{\mathrm{post}} = \lambda + (x-\mu)^\top\Sigma^{-1}(x-\mu)

        from quad-form scalars :math:`z_2 = (x-\mu)^\top\Sigma^{-1}(x-\mu)`,
        :math:`w_2 = \gamma^\top\Sigma^{-1}\gamma`. The IG limit keeps
        :math:`b_{\mathrm{post}} \ge \lambda > 0`, so the E-step's
        :data:`~normix.utils.constants.B_POST_FLOOR` is dormant here.
        """
        a_ig = self.lam / (self.mu_ig ** 2)
        return (-0.5 - self.d / 2.0,
                a_ig + w2,
                self.lam + z2)

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [-3/2-d/2,\; -(\lambda/2+\tfrac{1}{2}\mu^\top\Lambda\mu),\;
        -(\lambda/(2\mu_{IG}^2)+\tfrac{1}{2}\gamma^\top\Lambda\gamma),\;
        \Lambda\gamma,\; \Lambda\mu,\; -\tfrac{1}{2}\mathrm{vec}(\Lambda)]`

        where :math:`p=-1/2`, :math:`a=\lambda/\mu_{IG}^2`, :math:`b=\lambda`, aligned
        with GIG natural parameters on :math:`[\log y,\,1/y,\,y]`.
        """
        a_ig = self.lam / (self.mu_ig ** 2)
        Lambda_mu, Lambda_gamma, mu_quad, gamma_quad, Lambda = self._precision_quantities()
        return self._assemble_natural_params(
            -1.5 - self.d / 2.0,
            -(self.lam / 2.0 + mu_quad),
            -(a_ig / 2.0 + gamma_quad),
            Lambda_mu, Lambda_gamma, Lambda,
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta)` for Joint NIG. Uses :math:`K_{-1/2}(z) = \sqrt{\pi/(2z)}\,e^{-z}` —
        no Bessel function evaluation needed.
        """
        j = JointNormalInverseGaussian.from_natural(theta)
        a = j.lam / (j.mu_ig ** 2)
        b = j.lam
        sqrt_ab = jnp.sqrt(a * b)
        log_K = 0.5 * jnp.log(jnp.pi / (2.0 * sqrt_ab + LOG_EPS)) - sqrt_ab

        return (0.5 * j.log_det_sigma() + jnp.log(2.0) + log_K
                + 0.5 * (-0.5) * jnp.log((b + LOG_EPS) / (a + LOG_EPS))
                + j._mu_Lambda_gamma())

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, mu_ig, lam):
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, mu_ig=mu_ig, lam=lam)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointNormalInverseGaussian":
        r"""Recover classical parameters from :math:`\theta`.

        From :math:`b = -2\theta_2 - 2\mu_{\mathrm{quad}} = \lambda` and
        :math:`a = -2\theta_3 - 2\gamma_{\mathrm{quad}} = \lambda/\mu_{IG}^2`,
        so :math:`\mu_{IG} = \sqrt{b/a}`.
        """
        from normix.mixtures.joint import JointNormalMixture
        theta = jnp.asarray(theta, dtype=jnp.float64)
        d, mu, gamma, L_Sigma = JointNormalMixture._recover_normal_params(theta)
        mu_quad = 0.5 * jnp.dot(mu, theta[3 + d:3 + 2 * d])
        gamma_quad = 0.5 * jnp.dot(gamma, theta[3:3 + d])
        lam = 2.0 * (-theta[1] - mu_quad)           # b = λ
        a_ig = 2.0 * (-theta[2] - gamma_quad)       # a = λ/μ_IG²
        mu_ig = jnp.sqrt(lam / jnp.maximum(a_ig, 1e-30))
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma, mu_ig=mu_ig, lam=lam)

    @classmethod
    def _subordinator_from_eta(cls, eta, *, theta0=None, **kwargs):
        r"""Fit InverseGaussian subordinator from :math:`(E[Y], E[1/Y])`.

        ``theta0`` is accepted for API uniformity and ignored —
        InverseGaussian's ``from_expectation`` is closed-form
        (:math:`\mu = E[Y]`, :math:`\lambda = 1/(E[1/Y] - 1/E[Y])`).
        """
        from normix.distributions.inverse_gaussian import InverseGaussian
        return InverseGaussian.from_expectation(
            jnp.array([eta.E_Y, eta.E_inv_Y]))

    @classmethod
    def _from_normal_and_subordinator(cls, mu, gamma, L_Sigma, subordinator):
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma,
                   mu_ig=subordinator.mu, lam=subordinator.lam)

    def to_joint_generalized_hyperbolic(self):
        r"""Exact embedding into :class:`JointGeneralizedHyperbolic`.

        Lifts the InverseGaussian subordinator to GIG via
        :meth:`InverseGaussian.to_gig` (no boundary approximation) and
        keeps the Normal block unchanged.
        """
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        gig = self.subordinator().to_gig()
        return JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L_Sigma=self.L_Sigma,
            p=gig.p, a=gig.a, b=gig.b,
        )



class NormalInverseGaussian(NormalMixture):
    """Marginal Normal-Inverse Gaussian distribution f(x)."""

    def __init__(self, joint: JointNormalInverseGaussian):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, mu_ig, lam):
        joint = JointNormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=mu_ig, lam=lam)
        return cls(joint)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        Marginal NIG log-density.

        Uses :math:`K_{-1/2}(z) = \sqrt{\pi/(2z)}\,e^{-z}` for the normalisation,
        leaving only one :math:`\log K_\nu` call at order :math:`\nu = -1/2 - d/2`.
        """
        j = self._joint
        d = j.d
        p = -0.5
        a = j.lam / (j.mu_ig ** 2)
        b = j.lam

        z, w, z2, w2, zw = j._quad_forms(x)
        Q = z2
        A = a + w2
        nu = p - d / 2.0

        log_det_sigma = j.log_det_sigma()

        sqrt_ab = jnp.sqrt(a * b)
        log_K_p = 0.5 * jnp.log(jnp.pi / (2.0 * sqrt_ab + LOG_EPS)) - sqrt_ab

        sqrt_A_Qb = jnp.sqrt(A * (Q + b))

        log_f = (
            -0.5 * d * jnp.log(2.0 * jnp.pi)
            - 0.5 * log_det_sigma
            + 0.5 * p * (jnp.log(a + LOG_EPS) - jnp.log(b + LOG_EPS))
            - log_K_p
            + 0.5 * (d / 2.0 - p) * jnp.log(A / (Q + b + LOG_EPS))
            + log_kv(nu, sqrt_A_Qb)
            + zw
        )
        return log_f

    def _subordinator_expectations(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        from normix.distributions.generalized_inverse_gaussian import GIG
        j = self._joint
        gig = GIG(p=jnp.float64(-0.5), a=j.lam / j.mu_ig**2, b=j.lam)
        eta = gig.expectation_params()
        return eta[0], eta[1], eta[2]

    @classmethod
    def _joint_class(cls):
        return JointNormalInverseGaussian

    @classmethod
    def _subordinator_keys(cls):
        return ('mu_ig', 'lam')

    @classmethod
    def _univariate_class(cls):
        return UnivariateNormalInverseGaussian

    @property
    def mu_ig(self) -> jax.Array:
        r""":math:`\mu_{IG}` — InverseGaussian mean (forwarded from the joint)."""
        return self._joint.mu_ig

    @property
    def lam(self) -> jax.Array:
        r""":math:`\lambda` — InverseGaussian shape (forwarded from the joint)."""
        return self._joint.lam

    def _build_rescaled(self, mu, gamma_new, L_new, scale) -> "NormalInverseGaussian":
        # Σ → Σ/s pairs with Y → s·Y. For IG(μ_IG, λ):
        # s·Y ~ IG(s·μ_IG, s·λ).
        j = self._joint
        joint_new = JointNormalInverseGaussian(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            mu_ig=j.mu_ig * scale, lam=j.lam * scale,
        )
        return NormalInverseGaussian(joint_new)

    def regularize_a_eq_b(self) -> "NormalInverseGaussian":
        r"""Rescale so :math:`a = b = \sqrt{ab}`.

        For NIG, :math:`a = \lambda/\mu_{IG}^2,\;b = \lambda`, so
        :math:`s = \sqrt{a/b} = 1/\mu_{IG}`. After rescaling
        :math:`\mu_{IG}' = 1`, i.e. the InverseGaussian has unit mean.
        """
        scale = 1.0 / self._joint.mu_ig
        return self._rescale(scale)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.0)

    def to_generalized_hyperbolic(self):
        r"""Exact embedding into the :class:`GeneralizedHyperbolic` family.

        No boundary approximation: NIG sits in the strict interior of GH's
        parameter space (:math:`p = -1/2,\; a = \lambda/\mu_{IG}^2,\; b = \lambda`).
        """
        from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
        return GeneralizedHyperbolic(self._joint.to_joint_generalized_hyperbolic())


# ============================================================================
# Univariate Normal-Inverse Gaussian (scalar API + cdf/ppf)
# ============================================================================


class UnivariateNormalInverseGaussian(_UnivariateNormalMixtureMixin, NormalInverseGaussian):
    r"""Univariate (d=1) Normal-Inverse-Gaussian distribution.

    Sibling of :class:`NormalInverseGaussian` for 1-D problems; see
    :class:`~normix.distributions.variance_gamma.UnivariateVarianceGamma`
    for the contract.
    """

    @classmethod
    def from_classical(
        cls, *, mu, gamma, sigma, mu_ig, lam,
    ) -> "UnivariateNormalInverseGaussian":
        joint = JointNormalInverseGaussian.from_classical(
            mu=jnp.atleast_1d(jnp.asarray(mu, dtype=jnp.float64)),
            gamma=jnp.atleast_1d(jnp.asarray(gamma, dtype=jnp.float64)),
            sigma=jnp.atleast_2d(jnp.asarray(sigma, dtype=jnp.float64)),
            mu_ig=mu_ig, lam=lam,
        )
        return cls(joint)


# ============================================================================
# Factor-analysis Normal-Inverse Gaussian (Σ = F Fᵀ + diag(D))
# ============================================================================


class FactorNormalInverseGaussian(FactorNormalMixture):
    r"""Factor-analysis Normal-Inverse-Gaussian:
    :math:`Y \sim \mathrm{InvGaussian}(\mu_{IG}, \lambda)`,
    :math:`\Sigma = F F^\top + \mathrm{diag}(D)`.

    GIG params: :math:`p = -1/2`, :math:`a = \lambda/\mu_{IG}^2`,
    :math:`b = \lambda`.
    """

    def __init__(self, mu, gamma, F, D, *, mu_ig, lam):
        from normix.distributions.inverse_gaussian import InverseGaussian
        mu, gamma, F, D = FactorNormalMixture._check_init_args(mu, gamma, F, D)
        sub = InverseGaussian(
            mu=jnp.asarray(mu_ig, dtype=jnp.float64),
            lam=jnp.asarray(lam, dtype=jnp.float64),
        )
        object.__setattr__(self, 'mu', mu)
        object.__setattr__(self, 'gamma', gamma)
        object.__setattr__(self, 'F', F)
        object.__setattr__(self, 'D', D)
        object.__setattr__(self, 'subordinator', sub)

    @classmethod
    def from_classical(
        cls, *, mu, gamma, F, D, mu_ig, lam,
    ) -> "FactorNormalInverseGaussian":
        return cls(mu=mu, gamma=gamma, F=F, D=D, mu_ig=mu_ig, lam=lam)

    @property
    def mu_ig(self) -> jax.Array:
        return self.subordinator.mu

    @property
    def lam(self) -> jax.Array:
        return self.subordinator.lam

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        d = self.d
        p = -0.5
        a = self.lam / (self.mu_ig ** 2)
        b = self.lam

        z2, w2, zw = self._quad_forms(x)
        Q = z2
        A = a + w2
        nu = p - d / 2.0
        log_det_sigma = self._log_det_sigma()

        sqrt_ab = jnp.sqrt(a * b)
        log_K_p = 0.5 * jnp.log(jnp.pi / (2.0 * sqrt_ab + LOG_EPS)) - sqrt_ab
        sqrt_A_Qb = jnp.sqrt(A * (Q + b))

        return (-0.5 * d * jnp.log(2.0 * jnp.pi)
                - 0.5 * log_det_sigma
                + 0.5 * p * (jnp.log(a + LOG_EPS) - jnp.log(b + LOG_EPS))
                - log_K_p
                + 0.5 * (d / 2.0 - p) * jnp.log(A / (Q + b + LOG_EPS))
                + log_kv(nu, sqrt_A_Qb)
                + zw)

    def _posterior_gig_params(self, z2, w2):
        a_ig = self.lam / (self.mu_ig ** 2)
        return (-0.5 - self.d / 2.0,
                a_ig + w2,
                self.lam + z2)

    def _subordinator_expectations(self):
        from normix.distributions.generalized_inverse_gaussian import GIG
        gig = GIG(
            p=jnp.float64(-0.5),
            a=self.lam / (self.mu_ig ** 2), b=self.lam,
        )
        eta = gig.expectation_params()
        return eta[0], eta[1], eta[2]

    @classmethod
    def _subordinator_from_eta(cls, eta, *, theta0=None, **kwargs):
        from normix.distributions.inverse_gaussian import InverseGaussian
        return InverseGaussian.from_expectation(
            jnp.array([eta.E_Y, eta.E_inv_Y]))

    def _build_rescaled(self, mu, gamma_new, F_new, D_new, scale):
        # Σ → Σ/s pairs with Y → s·Y so that Y·Σ keeps its dispersion.
        # IG(μ_IG, λ): Y → s·Y ⇒ μ_IG → s·μ_IG, λ → s·λ.
        return FactorNormalInverseGaussian(
            mu=mu, gamma=gamma_new, F=F_new, D=D_new,
            mu_ig=self.mu_ig * scale, lam=self.lam * scale,
        )

    def regularize_a_eq_b(self) -> "FactorNormalInverseGaussian":
        r"""Rescale so :math:`a = b`. For NIG (:math:`a = \lambda/\mu_{IG}^2`,
        :math:`b = \lambda`), this means :math:`\mu_{IG} \to 1`.
        """
        scale = 1.0 / self.mu_ig
        return self._rescale(scale)

    @classmethod
    def _from_init_params(cls, mu, gamma, F, D):
        return cls.from_classical(
            mu=mu, gamma=gamma, F=F, D=D, mu_ig=1.0, lam=1.0)
