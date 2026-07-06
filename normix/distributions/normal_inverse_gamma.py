"""
Normal-Inverse Gamma (NInvG) distribution.

Special case of GH with GIG → InverseGamma subordinator
(:math:`a \\to 0`, :math:`p < 0`).
:math:`Y \\sim \\mathrm{InvGamma}(\\alpha, \\beta)`,
i.e. GIG(:math:`p = -\\alpha`, :math:`a \\to 0`, :math:`b = 2\\beta`).

Stored: :math:`\\mu`, :math:`\\gamma`, :math:`L_\\Sigma` (Cholesky of :math:`\\Sigma`),
:math:`\\alpha` (shape), :math:`\\beta` (rate) of InverseGamma.
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


class JointNormalInverseGamma(JointNormalMixture):
    r"""
    Joint :math:`f(x,y)`: :math:`X\mid Y \sim \mathcal{N}(\mu+\gamma y, \Sigma y)`,
    :math:`Y \sim \mathrm{InvGamma}(\alpha, \beta)`.

    GIG limit: :math:`p = -\alpha`, :math:`a \to 0`, :math:`b = 2\beta`.

    The :math:`\gamma y` mean term is the *skewness* that generalizes this
    beyond the textbook Normal-inverse-gamma conjugate prior — see
    :class:`NormalInverseGamma` for its role in the marginal density.
    """

    alpha: jax.Array
    beta: jax.Array

    def __init__(self, mu, gamma, L_Sigma, alpha, beta):
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'alpha', jnp.asarray(alpha, dtype=jnp.float64))
        object.__setattr__(self, 'beta', jnp.asarray(beta, dtype=jnp.float64))

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.inverse_gamma import InverseGamma
        return InverseGamma(alpha=self.alpha, beta=self.beta)

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [-(\alpha+1)-d/2,\; -(\beta+\tfrac{1}{2}\mu^\top\Lambda\mu),\;
        -\tfrac{1}{2}\gamma^\top\Lambda\gamma,\;
        \Lambda\gamma,\; \Lambda\mu,\; -\tfrac{1}{2}\mathrm{vec}(\Lambda)]`

        (InverseGamma subordinator: :math:`p=-\alpha`, :math:`a\to 0`, :math:`b=2\beta`).
        """
        Lambda_mu, Lambda_gamma, mu_quad, gamma_quad, Lambda = self._precision_quantities()
        return self._assemble_natural_params(
            -(self.alpha + 1.0) - self.d / 2.0,
            -(self.beta + mu_quad),
            -gamma_quad,
            Lambda_mu, Lambda_gamma, Lambda,
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta) = \tfrac{1}{2}\log|\Sigma| + \log\Gamma(\alpha) - \alpha\log\beta + \mu^\top\Lambda\gamma`.

        Analytical — no Bessel function needed (InverseGamma subordinator).
        """
        j = JointNormalInverseGamma.from_natural(theta)
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
    def from_natural(cls, theta: jax.Array) -> "JointNormalInverseGamma":
        r"""Recover classical parameters from :math:`\theta`.

        :math:`\alpha = -(\theta_1 + d/2) - 1`, :math:`\beta = -\theta_2 - \mu_{\mathrm{quad}}`.
        """
        from normix.mixtures.joint import JointNormalMixture
        theta = jnp.asarray(theta, dtype=jnp.float64)
        d, mu, gamma, L_Sigma = JointNormalMixture._recover_normal_params(theta)
        mu_quad = 0.5 * jnp.dot(mu, theta[3 + d:3 + 2 * d])
        alpha = -(theta[0] + d / 2.0) - 1.0
        beta = -theta[1] - mu_quad
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma, alpha=alpha, beta=beta)

    @classmethod
    def _subordinator_from_eta(cls, eta, *, theta0=None, **kwargs):
        r"""Fit InverseGamma subordinator from :math:`(-E[1/Y], E[\log Y])`.

        ``theta0`` is accepted for API uniformity and ignored —
        InverseGamma's ``from_expectation`` is closed-form (Newton on
        digamma).
        """
        from normix.distributions.inverse_gamma import InverseGamma
        backend = kwargs.get('backend', 'jax')
        return InverseGamma.from_expectation(
            jnp.array([-eta.E_inv_Y, eta.E_log_Y]), backend=backend)

    @classmethod
    def _from_normal_and_subordinator(cls, mu, gamma, L_Sigma, subordinator):
        return cls(mu=mu, gamma=gamma, L_Sigma=L_Sigma,
                   alpha=subordinator.alpha, beta=subordinator.beta)

    def to_joint_generalized_hyperbolic(self, *, boundary_eps: float = 0.0):
        r"""Exact embedding into :class:`JointGeneralizedHyperbolic`.

        Lifts the InverseGamma subordinator to GIG via
        :meth:`InverseGamma.to_gig` and keeps the Normal block unchanged.
        """
        from normix.distributions.generalized_hyperbolic import JointGeneralizedHyperbolic
        gig = self.subordinator().to_gig(boundary_eps=boundary_eps)
        return JointGeneralizedHyperbolic(
            mu=self.mu, gamma=self.gamma, L_Sigma=self.L_Sigma,
            p=gig.p, a=gig.a, b=gig.b,
        )



class NormalInverseGamma(NormalMixture):
    r"""Marginal Normal-Inverse Gamma distribution :math:`f(x)`.

    Not the textbook Normal-inverse-gamma conjugate prior — that
    distribution is a pure *variance* mixture,
    :math:`X\mid Y\sim\mathcal N(\mu,\Sigma Y)`, whose marginal is a plain
    Student-t with no special functions. This class is the GH-family
    variance-*mean* mixture (see :class:`JointNormalInverseGamma`):
    :math:`X\mid Y\sim\mathcal N(\mu+\gamma Y,\Sigma Y)`. The extra
    :math:`\gamma Y` term is why :meth:`log_prob` needs
    :func:`~normix.utils.bessel.log_kv`: expanding the quadratic form adds a
    term linear in :math:`Y`, turning the :math:`\int_0^\infty dY` integral
    from a Gamma integral into a Generalized Inverse Gaussian one,
    :math:`2(b/a)^{p/2}K_p(\sqrt{ab})` with :math:`a=\gamma^\top\Sigma^{-1}\gamma`.
    At :math:`\gamma=0` this reduces exactly to the textbook result
    :math:`f(x)=t\big(x\mid\nu=2\alpha,\ \hat\mu=\mu,\
    \hat\Sigma=\tfrac{\beta}{\alpha}\Sigma\big)` — see
    :doc:`/theory/gh` "Special Cases".
    """

    def __init__(self, joint: JointNormalInverseGamma):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, alpha, beta):
        joint = JointNormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=alpha, beta=beta)
        return cls(joint)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        Marginal NInvG log-density (own formula, no GH delegation).

        GIG params: :math:`p=-\alpha`, :math:`a=\gamma^\top\Lambda\gamma`,
        :math:`b=2\beta+Q(x)`.
        The normalising integral is :math:`2(b/a)^{p/2} K_p(\sqrt{ab})`.

        :math:`a,b` are floored at :data:`~normix.utils.constants.LOG_EPS`
        *before* forming :math:`\sqrt{ab}`, matching the floor used in the
        :math:`(b/a)^{p/2}` ratio. Without this, :math:`\gamma=0` (the
        common no-skew case) sends the true :math:`a=0` into ``log_kv``
        unfloored while the ratio uses the floored value, breaking the
        cancellation between the two that is needed for a finite limit.
        """
        j = self._joint
        d = j.d
        alpha = j.alpha
        beta = j.beta

        z, w, z2, w2, zw = j._quad_forms(x)
        q = z2
        a_gig = w2
        b_gig = 2.0 * beta + q
        p_gig = -(alpha + d / 2.0)
        linear = zw

        log_det_sigma = j.log_det_sigma()

        log_C = (-0.5 * d * jnp.log(2.0 * jnp.pi)
                 - 0.5 * log_det_sigma
                 - jax.scipy.special.gammaln(alpha)
                 + alpha * jnp.log(beta))

        a_eff = a_gig + LOG_EPS
        b_eff = b_gig + LOG_EPS
        sqrt_ab = jnp.sqrt(a_eff * b_eff)
        log_bessel = log_kv(p_gig, sqrt_ab)
        log_integral = (jnp.log(2.0)
                        + 0.5 * p_gig * jnp.log(b_eff / a_eff)
                        + log_bessel)

        log_f = log_C + linear + log_integral
        return log_f

    def _subordinator_expectations(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        r"""Prior InverseGamma moments :math:`(E[\log Y], E[1/Y], E[Y])`.

        :math:`E[\log Y] = \log\beta - \psi(\alpha)` and
        :math:`E[1/Y] = \alpha/\beta` are exact and finite for all
        :math:`\alpha > 0`. The forward moment :math:`E[Y] = \beta/(\alpha-1)`
        diverges as :math:`\alpha \downarrow 1` (and is *negative* for
        :math:`\alpha < 1`), so its denominator is floored at
        :data:`~normix.utils.constants.ALPHA_MOMENT_MARGIN` — the symmetric
        analogue of the VG inverse-moment floor.
        """
        j = self._joint
        E_log_Y = jnp.log(j.beta) - jax.scipy.special.digamma(j.alpha)
        E_inv_Y = j.alpha / j.beta
        E_Y = j.beta / jnp.maximum(j.alpha - 1.0, ALPHA_MOMENT_MARGIN)
        return E_log_Y, E_inv_Y, E_Y

    @classmethod
    def _joint_class(cls):
        return JointNormalInverseGamma

    @classmethod
    def _subordinator_keys(cls):
        return ('alpha', 'beta')

    @classmethod
    def _univariate_class(cls):
        return UnivariateNormalInverseGamma

    @property
    def alpha(self) -> jax.Array:
        r""":math:`\alpha` — InverseGamma shape (forwarded from the joint)."""
        return self._joint.alpha

    @property
    def beta(self) -> jax.Array:
        r""":math:`\beta` — InverseGamma rate (forwarded from the joint)."""
        return self._joint.beta

    def _build_rescaled(self, mu, gamma_new, L_new, scale) -> "NormalInverseGamma":
        j = self._joint
        joint_new = JointNormalInverseGamma(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            alpha=j.alpha, beta=j.beta * scale,
        )
        return NormalInverseGamma(joint_new)

    def fit(self, X, *, algorithm='em', verbose=0, max_iter=200, tol=1e-3,
            regularization='none',
            e_step_backend='cpu', m_step_backend='cpu',
            m_step_method='newton'):
        """Fit NInvG using EM or MCECM.  Defaults to CPU E-step (faster than
        JAX vmap for the degenerate-GIG posterior arising from the InverseGamma
        subordinator)."""
        return super().fit(
            X, algorithm=algorithm,
            verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=3.0, beta=1.0)

    def to_generalized_hyperbolic(self, *, boundary_eps: float = 0.0):
        r"""Exact embedding into the :class:`GeneralizedHyperbolic` family."""
        from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
        return GeneralizedHyperbolic(
            self._joint.to_joint_generalized_hyperbolic(boundary_eps=boundary_eps))


# ============================================================================
# Univariate Normal-Inverse Gamma (scalar API + cdf/ppf)
# ============================================================================


class UnivariateNormalInverseGamma(_UnivariateNormalMixtureMixin, NormalInverseGamma):
    r"""Univariate (d=1) Normal-Inverse-Gamma distribution.

    Sibling of :class:`NormalInverseGamma` for 1-D problems; see
    :class:`~normix.distributions.variance_gamma.UnivariateVarianceGamma`
    for the contract.
    """

    @classmethod
    def from_classical(
        cls, *, mu, gamma, sigma, alpha, beta,
    ) -> "UnivariateNormalInverseGamma":
        joint = JointNormalInverseGamma.from_classical(
            mu=jnp.atleast_1d(jnp.asarray(mu, dtype=jnp.float64)),
            gamma=jnp.atleast_1d(jnp.asarray(gamma, dtype=jnp.float64)),
            sigma=jnp.atleast_2d(jnp.asarray(sigma, dtype=jnp.float64)),
            alpha=alpha, beta=beta,
        )
        return cls(joint)


# ============================================================================
# Factor-analysis Normal-Inverse Gamma (Σ = F Fᵀ + diag(D))
# ============================================================================


class FactorNormalInverseGamma(FactorNormalMixture):
    r"""Factor-analysis Normal-Inverse-Gamma:
    :math:`Y \sim \mathrm{InvGamma}(\alpha, \beta)`,
    :math:`\Sigma = F F^\top + \mathrm{diag}(D)`.

    GIG limit: :math:`p = -\alpha`, :math:`a \to 0`, :math:`b = 2\beta`.
    """

    def __init__(self, mu, gamma, F, D, *, alpha, beta):
        from normix.distributions.inverse_gamma import InverseGamma
        mu, gamma, F, D = FactorNormalMixture._check_init_args(mu, gamma, F, D)
        sub = InverseGamma(
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
    ) -> "FactorNormalInverseGamma":
        return cls(mu=mu, gamma=gamma, F=F, D=D, alpha=alpha, beta=beta)

    @property
    def alpha(self) -> jax.Array:
        return self.subordinator.alpha

    @property
    def beta(self) -> jax.Array:
        return self.subordinator.beta

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        d = self.d
        alpha = self.alpha
        beta = self.beta

        z2, w2, zw = self._quad_forms(x)
        a_gig = w2
        b_gig = 2.0 * beta + z2
        p_gig = -(alpha + d / 2.0)
        log_det_sigma = self._log_det_sigma()

        log_C = (-0.5 * d * jnp.log(2.0 * jnp.pi)
                 - 0.5 * log_det_sigma
                 - jax.scipy.special.gammaln(alpha)
                 + alpha * jnp.log(beta))

        # a, b floored consistently before sqrt(ab) — see NormalInverseGamma.log_prob.
        a_eff = a_gig + LOG_EPS
        b_eff = b_gig + LOG_EPS
        sqrt_ab = jnp.sqrt(a_eff * b_eff)
        log_bessel = log_kv(p_gig, sqrt_ab)
        log_integral = (jnp.log(2.0)
                        + 0.5 * p_gig * jnp.log(b_eff / a_eff)
                        + log_bessel)
        return log_C + zw + log_integral

    def _subordinator_expectations(self):
        r"""Prior InverseGamma moments; see :meth:`NormalInverseGamma._subordinator_expectations`."""
        E_log_Y = jnp.log(self.beta) - jax.scipy.special.digamma(self.alpha)
        E_inv_Y = self.alpha / self.beta
        E_Y = self.beta / jnp.maximum(self.alpha - 1.0, ALPHA_MOMENT_MARGIN)
        return E_log_Y, E_inv_Y, E_Y

    @classmethod
    def _subordinator_from_eta(cls, eta, *, theta0=None, **kwargs):
        from normix.distributions.inverse_gamma import InverseGamma
        backend = kwargs.get('backend', 'jax')
        return InverseGamma.from_expectation(
            jnp.array([-eta.E_inv_Y, eta.E_log_Y]), backend=backend)

    def _build_rescaled(self, mu, gamma_new, F_new, D_new, scale):
        # InverseGamma(α, β): Y → Y/s ⇒ β → β/s, α unchanged.
        return FactorNormalInverseGamma(
            mu=mu, gamma=gamma_new, F=F_new, D=D_new,
            alpha=self.alpha, beta=self.beta * scale,
        )

    @classmethod
    def _from_init_params(cls, mu, gamma, F, D):
        return cls.from_classical(
            mu=mu, gamma=gamma, F=F, D=D, alpha=3.0, beta=1.0)
