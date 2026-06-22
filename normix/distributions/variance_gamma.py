"""
Variance Gamma (VG) distribution.

Special case of GH with GIG → Gamma subordinator
(:math:`b \\to 0`, :math:`p > 0`).
:math:`Y \\sim \\mathrm{Gamma}(\\alpha, \\beta)`,
i.e. GIG(:math:`p = \\alpha`, :math:`a = 2\\beta`, :math:`b \\to 0`).

Stored: :math:`\\mu`, :math:`\\gamma`, :math:`L_\\Sigma` (Cholesky of :math:`\\Sigma`),
:math:`\\alpha` (shape), :math:`\\beta` (rate) of Gamma.
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
from normix.utils.constants import ALPHA_MOMENT_MARGIN, LOG_EPS


class JointVarianceGamma(JointNormalMixture):
    r"""
    Joint :math:`f(x,y)`: :math:`X\mid Y \sim \mathcal{N}(\mu+\gamma y, \Sigma y)`,
    :math:`Y \sim \mathrm{Gamma}(\alpha, \beta)`.

    GIG limit: :math:`p = \alpha`, :math:`a = 2\beta`, :math:`b \to 0`.
    """

    alpha: jax.Array   # Gamma shape
    beta: jax.Array    # Gamma rate

    def __init__(self, mu, gamma, L_Sigma, alpha, beta):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'alpha', jnp.asarray(alpha, dtype=jnp.float64))
        object.__setattr__(self, 'beta', jnp.asarray(beta, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.gamma import Gamma
        return Gamma(alpha=self.alpha, beta=self.beta)

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [\alpha-1-d/2,\; -\tfrac{1}{2}\mu^\top\Lambda\mu,\;
        -(\beta+\tfrac{1}{2}\gamma^\top\Lambda\gamma),\;
        \Lambda\gamma,\; \Lambda\mu,\; -\tfrac{1}{2}\mathrm{vec}(\Lambda)]`

        (Gamma subordinator: :math:`p=\alpha`, :math:`a=2\beta`, :math:`b\to 0`).
        """
        Lambda_mu, Lambda_gamma, mu_quad, gamma_quad, Lambda = self._precision_quantities()
        return self._assemble_natural_params(
            self.alpha - 1.0 - self.d / 2.0,
            -mu_quad,
            -(self.beta + gamma_quad),
            Lambda_mu, Lambda_gamma, Lambda,
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta) = \tfrac{1}{2}\log|\Sigma| + \log\Gamma(\alpha) - \alpha\log\beta + \mu^\top\Lambda\gamma`.

        Analytical — no Bessel function needed (Gamma subordinator).
        """
        j = JointVarianceGamma.from_natural(theta)
        return (0.5 * j.log_det_sigma()
                + jax.scipy.special.gammaln(j.alpha)
                - j.alpha * jnp.log(j.beta)
                + j._mu_Lambda_gamma())

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta):
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, alpha=alpha, beta=beta)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointVarianceGamma":
        r"""Recover classical parameters from :math:`\theta`.

        :math:`\alpha = \theta_1 + 1 + d/2`, :math:`\beta = -\theta_3 - \gamma_{\mathrm{quad}}`.
        """
        from normix.mixtures.joint import JointNormalMixture
        theta = jnp.asarray(theta, dtype=jnp.float64)
        d, mu, gamma, L_Sigma = JointNormalMixture._recover_normal_params(theta)
        gamma_quad = 0.5 * jnp.dot(gamma, theta[3:3 + d])
        alpha = theta[0] + 1.0 + d / 2.0
        beta = -theta[2] - gamma_quad
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma, alpha=alpha, beta=beta)

    @classmethod
    def _subordinator_from_eta(cls, eta, *, theta0=None, **kwargs):
        r"""Fit Gamma subordinator from :math:`(E[\log Y], E[Y])`.

        ``theta0`` is accepted for API uniformity and ignored — Gamma's
        :meth:`~normix.distributions.gamma.Gamma.from_expectation` is
        closed-form (Newton on digamma). An optional ``alpha_min`` (the VG
        likelihood-boundedness control) is forwarded to clamp the shape.
        """
        from normix.distributions.gamma import Gamma
        backend = kwargs.get('backend', 'jax')
        return Gamma.from_expectation(
            jnp.array([eta.E_log_Y, eta.E_Y]),
            backend=backend, alpha_min=kwargs.get('alpha_min'))

    @classmethod
    def _from_normal_and_subordinator(cls, mu, gamma, L_Sigma, subordinator):
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma,
                   alpha=subordinator.alpha, beta=subordinator.beta)

    def to_joint_generalized_hyperbolic(self, *, boundary_eps: float = 0.0):
        r"""Exact embedding into :class:`JointGeneralizedHyperbolic`.

        Lifts the Gamma subordinator to GIG via :meth:`Gamma.to_gig` and
        keeps the Normal block (:math:`\mu, \gamma, L_\Sigma`) unchanged.
        """
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        gig = self.subordinator().to_gig(boundary_eps=boundary_eps)
        return JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L_Sigma=self.L_Sigma,
            p=gig.p, a=gig.a, b=gig.b,
        )



class VarianceGamma(NormalMixture):
    """Marginal Variance Gamma distribution f(x)."""

    def __init__(self, joint: JointVarianceGamma):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta) -> "VarianceGamma":
        joint = JointVarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=alpha, beta=beta)
        return cls(joint)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        Marginal VG log-density (own formula, no GH delegation).

        .. math::

            f(x) \propto \left(\frac{q}{2c}\right)^{\nu/2}
            K_\nu\!\left(\sqrt{2qc}\right) \exp(\gamma^\top\Sigma^{-1}(x-\mu))

        where :math:`\nu = \alpha - d/2`,
        :math:`c = \beta + \tfrac{1}{2}\gamma^\top\Lambda\gamma`,
        :math:`q = (x-\mu)^\top\Lambda(x-\mu)`.
        """
        j = self._joint
        d = j.d
        alpha = j.alpha
        beta = j.beta

        z, w, z2, w2, zw = j._quad_forms(x)
        q = z2
        gamma_quad = w2
        linear = zw

        c = beta + 0.5 * gamma_quad
        nu = alpha - d / 2.0

        log_det_sigma = j.log_det_sigma()

        log_C = (jnp.log(2.0)
                 - 0.5 * d * jnp.log(2.0 * jnp.pi)
                 - 0.5 * log_det_sigma
                 - jax.scipy.special.gammaln(alpha)
                 + alpha * jnp.log(beta))

        z_arg = jnp.sqrt(2.0 * q * c)
        log_K = log_kv(nu, z_arg)

        log_f = (log_C
                 + 0.5 * nu * jnp.log(q / (2.0 * c + LOG_EPS) + LOG_EPS)
                 + log_K
                 + linear)
        return log_f

    def _subordinator_expectations(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        r"""Prior Gamma moments :math:`(E[\log Y], E[1/Y], E[Y])`.

        :math:`E[\log Y] = \psi(\alpha) - \log\beta` and :math:`E[Y] = \alpha/\beta`
        are exact and finite for all :math:`\alpha > 0`. The inverse moment
        :math:`E[1/Y] = \beta/(\alpha-1)` diverges as :math:`\alpha \downarrow 1`
        (and is *negative* for :math:`\alpha < 1`), so its denominator is floored
        at :data:`~normix.utils.constants.ALPHA_MOMENT_MARGIN` — a
        distribution-specific regularisation that keeps it finite, positive, and
        continuous without routing the well-conditioned moments through the GIG
        Bessel path.
        """
        j = self._joint
        E_log_Y = jax.scipy.special.digamma(j.alpha) - jnp.log(j.beta)
        E_inv_Y = j.beta / jnp.maximum(j.alpha - 1.0, ALPHA_MOMENT_MARGIN)
        E_Y = j.alpha / j.beta
        return E_log_Y, E_inv_Y, E_Y

    @classmethod
    def _joint_class(cls):
        return JointVarianceGamma

    @classmethod
    def _subordinator_keys(cls):
        return ('alpha', 'beta')

    @classmethod
    def _univariate_class(cls):
        return UnivariateVarianceGamma

    @property
    def alpha(self) -> jax.Array:
        r""":math:`\alpha` — Gamma shape (forwarded from the joint)."""
        return self._joint.alpha

    @property
    def beta(self) -> jax.Array:
        r""":math:`\beta` — Gamma rate (forwarded from the joint)."""
        return self._joint.beta

    def _build_rescaled(self, mu, gamma_new, L_new, scale) -> "VarianceGamma":
        j = self._joint
        joint_new = JointVarianceGamma(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            alpha=j.alpha, beta=j.beta / scale,
        )
        return VarianceGamma(joint_new)

    def fit(self, X, *, algorithm='em', verbose=0, max_iter=200, tol=1e-3,
            regularization='none',
            e_step_backend='cpu', m_step_backend='cpu',
            m_step_method='newton', alpha_min=None):
        r"""Fit VG using EM or MCECM.  Defaults to CPU E-step (faster than JAX
        vmap for the degenerate-GIG posterior arising from the Gamma subordinator).

        ``alpha_min`` (float or ``'density'`` / ``'inverse_moment'``) is the
        opt-in lower bound on the Gamma shape :math:`\alpha` that keeps the VG
        marginal likelihood bounded; see
        :meth:`~normix.mixtures.marginal.MarginalMixture.fit`.
        """
        return super().fit(
            X, algorithm=algorithm,
            verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method, alpha_min=alpha_min)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0)

    def to_generalized_hyperbolic(self, *, boundary_eps: float = 0.0):
        r"""Exact embedding into the :class:`GeneralizedHyperbolic` family."""
        from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
        return GeneralizedHyperbolic(
            self._joint.to_joint_generalized_hyperbolic(boundary_eps=boundary_eps))


# ============================================================================
# Univariate Variance Gamma (scalar API + cdf/ppf)
# ============================================================================


class UnivariateVarianceGamma(_UnivariateNormalMixtureMixin, VarianceGamma):
    r"""Univariate (d=1) Variance Gamma distribution.

    Sibling of :class:`VarianceGamma` for 1-D problems: exposes scalar
    ``mean``/``var``/``std``, ``(n,)``-shaped ``rvs``, and ``cdf``/``ppf``
    backed by a PINV table over the marginal log-density. EM, ``fit``,
    ``replace``, and regularisation are inherited from
    :class:`VarianceGamma`.
    """

    @classmethod
    def from_classical(
        cls, *, mu, gamma, sigma, alpha, beta,
    ) -> "UnivariateVarianceGamma":
        r"""Build from scalar or 1-D classical parameters.

        ``mu``, ``gamma`` may be scalars or ``(1,)``; ``sigma`` may be a
        scalar variance, ``(1,)``, or ``(1, 1)``.
        """
        joint = JointVarianceGamma.from_classical(
            mu=jnp.atleast_1d(jnp.asarray(mu, dtype=jnp.float64)),
            gamma=jnp.atleast_1d(jnp.asarray(gamma, dtype=jnp.float64)),
            sigma=jnp.atleast_2d(jnp.asarray(sigma, dtype=jnp.float64)),
            alpha=alpha, beta=beta,
        )
        return cls(joint)


# ============================================================================
# Factor-analysis Variance Gamma (Σ = F Fᵀ + diag(D))
# ============================================================================


class FactorVarianceGamma(FactorNormalMixture):
    r"""Factor-analysis Variance Gamma:
    :math:`Y \sim \mathrm{Gamma}(\alpha, \beta)`,
    :math:`\Sigma = F F^\top + \mathrm{diag}(D)`.

    GIG limit of the subordinator: :math:`p = \alpha`,
    :math:`a = 2\beta`, :math:`b \to 0`.
    """

    def __init__(self, mu, gamma, F, D, *, alpha, beta):
        from normix.distributions.gamma import Gamma
        mu, gamma, F, D = FactorNormalMixture._check_init_args(mu, gamma, F, D)
        sub = Gamma(
            alpha=jnp.asarray(alpha, dtype=jnp.float64),
            beta=jnp.asarray(beta, dtype=jnp.float64),
        )
        object.__setattr__(self, 'mu', mu)
        object.__setattr__(self, 'gamma', gamma)
        object.__setattr__(self, 'F', F)
        object.__setattr__(self, 'D', D)
        object.__setattr__(self, 'subordinator', sub)

    @classmethod
    def from_classical(
        cls, *, mu, gamma, F, D, alpha, beta,
    ) -> "FactorVarianceGamma":
        return cls(mu=mu, gamma=gamma, F=F, D=D, alpha=alpha, beta=beta)

    @property
    def alpha(self) -> jax.Array:
        return self.subordinator.alpha

    @property
    def beta(self) -> jax.Array:
        return self.subordinator.beta

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""Marginal VG log-density evaluated with Woodbury Σ-solve."""
        x = jnp.asarray(x, dtype=jnp.float64)
        d = self.d
        alpha = self.alpha
        beta = self.beta

        z2, w2, zw = self._quad_forms(x)
        c = beta + 0.5 * w2
        nu = alpha - d / 2.0
        log_det_sigma = self._log_det_sigma()

        log_C = (jnp.log(2.0)
                 - 0.5 * d * jnp.log(2.0 * jnp.pi)
                 - 0.5 * log_det_sigma
                 - jax.scipy.special.gammaln(alpha)
                 + alpha * jnp.log(beta))

        z_arg = jnp.sqrt(2.0 * z2 * c)
        log_K = log_kv(nu, z_arg)
        return (log_C
                + 0.5 * nu * jnp.log(z2 / (2.0 * c + LOG_EPS) + LOG_EPS)
                + log_K + zw)

    def _subordinator_expectations(self):
        r"""Prior Gamma moments; see :meth:`VarianceGamma._subordinator_expectations`."""
        E_log_Y = jax.scipy.special.digamma(self.alpha) - jnp.log(self.beta)
        E_inv_Y = self.beta / jnp.maximum(self.alpha - 1.0, ALPHA_MOMENT_MARGIN)
        E_Y = self.alpha / self.beta
        return E_log_Y, E_inv_Y, E_Y

    @classmethod
    def _subordinator_from_eta(cls, eta, *, theta0=None, **kwargs):
        from normix.distributions.gamma import Gamma
        backend = kwargs.get('backend', 'jax')
        return Gamma.from_expectation(
            jnp.array([eta.E_log_Y, eta.E_Y]),
            backend=backend, alpha_min=kwargs.get('alpha_min'))

    def _build_rescaled(self, mu, gamma_new, F_new, D_new, scale):
        # Σ → Σ/s pairs with subordinator Y → s·Y so that Y·Σ keeps the
        # dispersion. For Gamma(α, β), Y/s ⇒ β → β·s, α unchanged.
        return FactorVarianceGamma(
            mu=mu, gamma=gamma_new, F=F_new, D=D_new,
            alpha=self.alpha, beta=self.beta / scale,
        )

    @classmethod
    def _from_init_params(cls, mu, gamma, F, D):
        return cls.from_classical(
            mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
