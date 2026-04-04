"""
Generalized Hyperbolic (GH) distribution.

Joint: :math:`X \\mid Y \\sim \\mathcal{N}(\\mu + \\gamma y, \\Sigma y)`,
:math:`Y \\sim \\mathrm{GIG}(p, a, b)`.

Marginal: :math:`\\mathrm{GH}(\\mu, \\gamma, \\Sigma, p, a, b)`.

**Marginal log-density** (closed form via Bessel functions):

Let :math:`Q(x) = (x-\\mu)^\\top \\Sigma^{-1}(x-\\mu)`,
:math:`A = a + \\gamma^\\top \\Sigma^{-1} \\gamma`.

.. math::

    \\log f(x) = -\\tfrac{d}{2}\\log(2\\pi) - \\tfrac{1}{2}\\log|\\Sigma|
    + \\tfrac{p}{2}(\\log a - \\log b) - \\log K_p(\\sqrt{ab})
    + \\tfrac{d/2-p}{2}\\log\\frac{A}{Q(x)+b}
    + \\log K_{p-d/2}\\!\\left(\\sqrt{A(Q(x)+b)}\\right)
    + \\gamma^\\top \\Sigma^{-1}(x - \\mu)

**Posterior:** :math:`Y \\mid X = x \\sim
\\mathrm{GIG}(p - d/2,\\; a + \\gamma^\\top\\Sigma^{-1}\\gamma,\\;
b + (x-\\mu)^\\top\\Sigma^{-1}(x-\\mu))`.
"""
from __future__ import annotations

from typing import Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.utils.bessel import log_kv
from normix.exponential_family import ExponentialFamily
from normix.mixtures.joint import JointNormalMixture
from normix.mixtures.marginal import NormalMixture

jax.config.update("jax_enable_x64", True)

from normix.utils.constants import LOG_EPS, GIG_CLAMP_LO, GIG_CLAMP_HI, GIG_P_MAX


# ============================================================================
# Joint GH distribution
# ============================================================================

class JointGeneralizedHyperbolic(JointNormalMixture):
    r"""
    Joint :math:`f(x,y)`: :math:`X\mid Y \sim \mathcal{N}(\mu+\gamma y, \Sigma y)`,
    :math:`Y \sim \mathrm{GIG}(p, a, b)`.

    Stored: ``mu``, ``gamma``, ``L_Sigma`` (from :class:`JointNormalMixture`) +
    ``p``, ``a``, ``b`` (GIG parameters).
    """

    p: jax.Array
    a: jax.Array
    b: jax.Array

    def __init__(self, mu, gamma, L_Sigma, p, a, b):
        # eqx.Module fields must be set via object.__setattr__ before freeze
        object.__setattr__(self, 'mu', jnp.asarray(mu, dtype=jnp.float64))
        object.__setattr__(self, 'gamma', jnp.asarray(gamma, dtype=jnp.float64))
        object.__setattr__(self, 'L_Sigma', jnp.asarray(L_Sigma, dtype=jnp.float64))
        object.__setattr__(self, 'p', jnp.asarray(p, dtype=jnp.float64))
        object.__setattr__(self, 'a', jnp.asarray(a, dtype=jnp.float64))
        object.__setattr__(self, 'b', jnp.asarray(b, dtype=jnp.float64))

    # ------------------------------------------------------------------
    # JointNormalMixture abstract methods
    # ------------------------------------------------------------------

    def subordinator(self) -> ExponentialFamily:
        from normix.distributions.generalized_inverse_gaussian import GIG
        return GIG(p=self.p, a=self.a, b=self.b)

    def _compute_posterior_expectations(
        self, x: jax.Array
    ) -> Dict[str, jax.Array]:
        r"""
        Posterior :math:`Y\mid X=x \sim \mathrm{GIG}(p-d/2,\;
        a+\gamma^\top\Sigma^{-1}\gamma,\;
        b+(x-\mu)^\top\Sigma^{-1}(x-\mu))`.
        """
        from normix.distributions.generalized_inverse_gaussian import GIG

        d = self.d
        z, w, z2, w2, zw = self._quad_forms(x)
        p_post = self.p - d / 2.0
        a_post = self.a + w2
        b_post = self.b + z2

        gig = GIG(p=p_post, a=a_post, b=b_post)
        eta = gig.expectation_params()
        return {
            'E_log_Y': eta[0],
            'E_inv_Y': eta[1],
            'E_Y': eta[2],
        }

    def _posterior_gig_params(
        self, z2: jax.Array, w2: jax.Array
    ):
        r"""Posterior GIG :math:`(p, a, b)` given quad-form scalars :math:`z_2=\|L^{-1}(x-\mu)\|^2`, :math:`w_2=\|L^{-1}\gamma\|^2`."""
        return (self.p - self.d / 2.0,
                self.a + w2,
                self.b + z2)

    # ------------------------------------------------------------------
    # ExponentialFamily: natural params / log partition
    # ------------------------------------------------------------------

    def natural_params(self) -> jax.Array:
        r"""
        :math:`\theta = [p-1-d/2,\; -(b/2+\tfrac{1}{2}\mu^\top\Sigma^{-1}\mu),\;
        -(a/2+\tfrac{1}{2}\gamma^\top\Sigma^{-1}\gamma),\;
        \Sigma^{-1}\gamma,\; \Sigma^{-1}\mu,\; -\tfrac{1}{2}\mathrm{vec}(\Sigma^{-1})]`

        The scalar coefficients on sufficient statistics :math:`1/y` and :math:`y`
        match the GIG convention :math:`\theta_{\mathrm{GIG}} = [p-1,\,-b/2,\,-a/2]`
        on :math:`t_Y = [\log y,\,1/y,\,y]`.
        """
        _, _, mu_quad, gamma_quad, _ = self._precision_quantities()
        return self._assemble_natural_params(
            self.p - 1.0 - self.d / 2.0,
            -(self.b / 2.0 + mu_quad),
            -(self.a / 2.0 + gamma_quad),
        )

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi = \psi_{\mathrm{GIG}}(p, a, b) + \tfrac{1}{2}\log|\Sigma| + \mu^\top\Sigma^{-1}\gamma`.

        Recovers :math:`p, a, b, \mu, \gamma, \Sigma` from :math:`\theta`.
        Dimension :math:`d` is inferred from :math:`|\theta| = 3 + 2d + d^2`.
        """
        from normix.distributions.generalized_inverse_gaussian import GIG
        from normix.mixtures.joint import JointNormalMixture

        (d, theta_1, theta_2, theta_3, *_, log_det_Sigma, _, _,
         mu_quad, gamma_quad, mu_Lambda_gamma) = JointNormalMixture._parse_joint_theta(theta)

        p = theta_1 + 1.0 + d / 2.0
        b = 2.0 * (-theta_2 - mu_quad)
        a = 2.0 * (-theta_3 - gamma_quad)

        gig_theta = jnp.array([p - 1.0, -b / 2.0, -a / 2.0])
        psi_gig = GIG._log_partition_from_theta(gig_theta)

        return psi_gig + 0.5 * log_det_Sigma + mu_Lambda_gamma

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, p, a, b):
        """Construct from classical parameters."""
        mu = jnp.asarray(mu, dtype=jnp.float64)
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        L = jnp.linalg.cholesky(sigma)
        return cls(mu=mu, gamma=gamma, L_Sigma=L, p=p, a=a, b=b)

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "JointGeneralizedHyperbolic":
        raise NotImplementedError(
            "JointGeneralizedHyperbolic.from_natural: not implemented — "
            "use from_classical or m_step."
        )



# ============================================================================
# Marginal GH distribution
# ============================================================================

class GeneralizedHyperbolic(NormalMixture):
    """
    Marginal Generalized Hyperbolic distribution :math:`f(x)`.

    Stores a :class:`JointGeneralizedHyperbolic`. Provides:

    - ``log_prob(x)`` — closed-form Bessel expression
    - ``e_step``, ``m_step`` — for EM fitting
    - ``fit(X, ...)`` — convenience fitting method
    """

    def __init__(self, joint: JointGeneralizedHyperbolic):
        object.__setattr__(self, '_joint', joint)

    @classmethod
    def from_classical(cls, *, mu, gamma, sigma, p, a, b) -> "GeneralizedHyperbolic":
        joint = JointGeneralizedHyperbolic.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, p=p, a=a, b=b)
        return cls(joint)

    # ------------------------------------------------------------------
    # Marginal log-density (closed form)
    # ------------------------------------------------------------------

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""
        Marginal :math:`\log f(x)`.

        .. math::

            f(x) \propto \left(\frac{A}{Q(x)+b}\right)^{(d/2-p)/2}
            K_{p-d/2}\!\left(\sqrt{A(Q(x)+b)}\right)
            \exp\!\left(\gamma^\top\Sigma^{-1}(x-\mu)\right)

        where :math:`Q(x) = (x-\mu)^\top\Sigma^{-1}(x-\mu)`,
        :math:`A = a + \gamma^\top\Sigma^{-1}\gamma`.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        j = self._joint
        d = j.d

        # Solve L z = x - μ
        z, w, z2, w2, zw = j._quad_forms(x)
        # Q(x) = ‖z‖² = (x-μ)ᵀΣ⁻¹(x-μ)
        Q = z2
        # A = a + γᵀΣ⁻¹γ = a + w2
        A = j.a + w2
        # Bessel order in posterior: p_post = p - d/2
        p_post = j.p - d / 2.0

        # log f(x) = log C + (p-d/2)/2 * log((Q+b)/A) + ... (skewness) + log K_{p_post}(...)
        #
        # Precise formula (see e.g. Protassov 2004):
        #   log f(x) = ½ log(A) - ½ log(Q+b) ... wait, let me derive carefully.
        #
        # From Barndorff-Nielsen (1978) the marginal density is:
        #
        # f(x) = (2π)^{-d/2} |Σ|^{-1/2} · (a/b)^{p/2} / K_p(√(ab))
        #        · (A / (Q(x)+b))^{(d/2-p)/2}
        #        · K_{p-d/2}(√(A(Q(x)+b)))
        #        · exp((x-μ)ᵀΣ⁻¹γ)
        #
        # log f(x) = -d/2 log(2π) - ½ log|Σ|
        #            + p/2 (log a - log b)
        #            - log K_p(√(ab))
        #            + (d/2-p)/2 · log(A/(Q+b))
        #            + log K_{p-d/2}(√(A(Q+b)))
        #            + γᵀΣ⁻¹(x-μ)                ← = wᵀz (inner product)

        log_det_sigma = j.log_det_sigma()
        sqrt_ab = jnp.sqrt(j.a * j.b)
        sqrt_A_Qb = jnp.sqrt(A * (Q + j.b))

        log_f = (
            -0.5 * d * jnp.log(2.0 * jnp.pi)
            - 0.5 * log_det_sigma
            + 0.5 * j.p * (jnp.log(j.a + LOG_EPS) - jnp.log(j.b + LOG_EPS))
            - log_kv(j.p, sqrt_ab)
            + 0.5 * (d / 2.0 - j.p) * jnp.log(A / (Q + j.b + LOG_EPS))
            + log_kv(p_post, sqrt_A_Qb)
            + zw                # γᵀΣ⁻¹(x-μ) = wᵀz
        )
        return log_f

    # ------------------------------------------------------------------
    # M-step subordinator (GIG requires backend/method/maxiter)
    # ------------------------------------------------------------------

    def _subordinator_expectations(self):
        j = self._joint
        eta = j.subordinator().expectation_params()
        return eta[0], eta[1], eta[2]

    def m_step_subordinator(self, eta, **kwargs):
        from normix.distributions.generalized_inverse_gaussian import GIG
        j = self._joint
        backend = kwargs.get('backend', 'jax')
        method = kwargs.get('method', 'newton')
        maxiter = kwargs.get('maxiter', 20)

        gig_eta = jnp.array([eta.E_log_Y, eta.E_inv_Y, eta.E_Y])
        current_gig = GIG(p=j.p, a=j.a, b=j.b)

        if backend == 'cpu':
            try:
                gig_new = GIG.from_expectation(
                    gig_eta,
                    theta0=current_gig.natural_params(),
                    backend='cpu', method=method, maxiter=maxiter,
                )
                p_new = float(gig_new.p)
                a_new = float(gig_new.a)
                b_new = float(gig_new.b)
                if (abs(p_new) > GIG_P_MAX
                        or a_new > GIG_CLAMP_HI or b_new > GIG_CLAMP_HI
                        or a_new < GIG_CLAMP_LO or b_new < GIG_CLAMP_LO):
                    gig_new = current_gig
            except Exception:
                gig_new = current_gig
        else:
            gig_new = GIG.from_expectation(
                gig_eta,
                theta0=current_gig.natural_params(),
                backend=backend, method=method, maxiter=maxiter,
            )
            sane = (
                (jnp.abs(gig_new.p) <= GIG_P_MAX)
                & (gig_new.a >= GIG_CLAMP_LO) & (gig_new.a <= GIG_CLAMP_HI)
                & (gig_new.b >= GIG_CLAMP_LO) & (gig_new.b <= GIG_CLAMP_HI)
            )
            gig_new = jax.tree.map(
                lambda new, old: jnp.where(sane, new, old),
                gig_new, current_gig,
            )

        joint_new = JointGeneralizedHyperbolic(
            mu=j.mu, gamma=j.gamma, L_Sigma=j.L_Sigma,
            p=gig_new.p, a=gig_new.a, b=gig_new.b,
        )
        return GeneralizedHyperbolic(joint_new)

    def _build_rescaled(self, mu, gamma_new, L_new, scale):
        j = self._joint
        a_new = jnp.clip(j.a / scale, GIG_CLAMP_LO, GIG_CLAMP_HI)
        b_new = jnp.clip(j.b * scale, GIG_CLAMP_LO, GIG_CLAMP_HI)
        joint_new = JointGeneralizedHyperbolic(
            mu=mu, gamma=gamma_new, L_Sigma=L_new,
            p=j.p, a=a_new, b=b_new,
        )
        return GeneralizedHyperbolic(joint_new)

    @classmethod
    def _from_init_params(cls, mu, gamma, sigma):
        return cls.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, p=1.0, a=1.0, b=1.0)

    @classmethod
    def default_init(cls, X: jax.Array) -> "GeneralizedHyperbolic":
        r"""Warm-start from the best of NIG / VG / NInvG sub-model fits.

        Runs 5 EM iterations (JAX backend, Newton method) for each special
        case, converts each to GH parametrisation, and selects the candidate
        with the highest marginal log-likelihood.  A moment-based fallback
        (:math:`p=1, a=1, b=1`) is included as a fourth candidate.

        Fully JAX-native: no try/except, no Python branching on data values.
        """
        from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
        from normix.distributions.variance_gamma import VarianceGamma
        from normix.distributions.normal_inverse_gamma import NormalInverseGamma
        from normix.fitting.em import BatchEMFitter

        X = jnp.asarray(X, dtype=jnp.float64)

        fitter = BatchEMFitter(
            max_iter=5, tol=1e-6, verbose=0,
            e_step_backend='jax', m_step_backend='jax',
            m_step_method='newton',
        )

        # --- NIG: p=-0.5, a=λ/μ², b=λ ---
        nig = fitter.fit(NormalInverseGaussian.default_init(X), X).model
        j_nig = nig._joint
        p_nig = jnp.float64(-0.4999)
        a_nig = jnp.clip(j_nig.lam / (j_nig.mu_ig ** 2), GIG_CLAMP_LO, GIG_CLAMP_HI)
        b_nig = jnp.clip(j_nig.lam, GIG_CLAMP_LO, GIG_CLAMP_HI)
        ll_nig = nig.marginal_log_likelihood(X)

        # --- VG: p=α, a=2β, b≈0 ---
        vg = fitter.fit(VarianceGamma.default_init(X), X).model
        j_vg = vg._joint
        p_vg = j_vg.alpha
        a_vg = jnp.clip(2.0 * j_vg.beta, GIG_CLAMP_LO, GIG_CLAMP_HI)
        b_vg = jnp.float64(1e-4)
        ll_vg = vg.marginal_log_likelihood(X)

        # --- NInvG: p=-α, a≈0, b=2β ---
        ninvg = fitter.fit(NormalInverseGamma.default_init(X), X).model
        j_ninvg = ninvg._joint
        p_ninvg = -j_ninvg.alpha
        a_ninvg = jnp.float64(1e-4)
        b_ninvg = jnp.clip(2.0 * j_ninvg.beta, GIG_CLAMP_LO, GIG_CLAMP_HI)
        ll_ninvg = ninvg.marginal_log_likelihood(X)

        # --- Fallback: moment-based default (p=1, a=1, b=1) ---
        fallback = super().default_init(X)
        j_fb = fallback._joint
        ll_fb = fallback.marginal_log_likelihood(X)

        # Stack candidates and select best by log-likelihood
        lls = jnp.array([ll_nig, ll_vg, ll_ninvg, ll_fb])
        lls = jnp.where(jnp.isfinite(lls), lls, -jnp.inf)
        best = jnp.argmax(lls)

        mus = jnp.stack([j_nig.mu, j_vg.mu, j_ninvg.mu, j_fb.mu])
        gammas = jnp.stack([j_nig.gamma, j_vg.gamma, j_ninvg.gamma, j_fb.gamma])
        Ls = jnp.stack([j_nig.L_Sigma, j_vg.L_Sigma, j_ninvg.L_Sigma, j_fb.L_Sigma])
        ps = jnp.array([p_nig, p_vg, p_ninvg, j_fb.p])
        a_s = jnp.array([a_nig, a_vg, a_ninvg, j_fb.a])
        bs = jnp.array([b_nig, b_vg, b_ninvg, j_fb.b])

        return cls.from_classical(
            mu=mus[best], gamma=gammas[best],
            sigma=Ls[best] @ Ls[best].T,
            p=ps[best], a=a_s[best], b=bs[best],
        )

    def fit(self, X, *, algorithm='em', verbose=0, max_iter=200, tol=1e-3,
            regularization='det_sigma_one',
            e_step_backend='cpu', m_step_backend='cpu',
            m_step_method='newton'):
        """Fit GH distribution using EM or MCECM.

        Defaults to CPU backends and det_sigma_one regularization
        (GH has scale non-identifiability requiring |Sigma| = 1).
        """
        return super().fit(
            X, algorithm=algorithm,
            verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method)

