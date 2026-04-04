"""
Tests for JAX normix distributions.

Validates:
  - Exponential family structure (log_partition, expectation_params, fisher_info)
  - from_natural / from_expectation roundtrips
  - log_prob vs scipy reference
  - fit_mle convergence
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import (
    Gamma, InverseGamma, InverseGaussian, GIG,
    GeneralizedHyperbolic, MultivariateNormal,
    JointGeneralizedHyperbolic,
    JointNormalInverseGamma,
    JointNormalInverseGaussian,
    JointVarianceGamma,
)


# ===========================================================================
# Gamma
# ===========================================================================

class TestGamma:

    def test_natural_params(self):
        g = Gamma(alpha=2.0, beta=3.0)
        theta = g.natural_params()
        np.testing.assert_allclose(theta, [1.0, -3.0], rtol=1e-12)

    def test_from_natural_roundtrip(self):
        theta = jnp.array([1.0, -3.0])
        g = Gamma.from_natural(theta)
        np.testing.assert_allclose(float(g.alpha), 2.0, rtol=1e-10)
        np.testing.assert_allclose(float(g.beta), 3.0, rtol=1e-10)

    def test_log_partition(self):
        g = Gamma(alpha=2.0, beta=1.0)
        from scipy.special import gammaln
        psi = float(g.log_partition())
        # ψ(θ) = log Γ(2) - 2*log(1) = 0
        assert abs(psi) < 1e-10

    def test_expectation_params(self):
        alpha, beta = 3.0, 2.0
        g = Gamma(alpha=alpha, beta=beta)
        eta = g.expectation_params()
        from scipy.special import digamma
        np.testing.assert_allclose(float(eta[0]),
                                   digamma(alpha) - np.log(beta),
                                   rtol=1e-8)
        np.testing.assert_allclose(float(eta[1]), alpha / beta, rtol=1e-8)

    def test_from_expectation_roundtrip(self):
        alpha, beta = 3.0, 2.0
        g = Gamma(alpha=alpha, beta=beta)
        eta = g.expectation_params()
        g2 = Gamma.from_expectation(eta)
        np.testing.assert_allclose(float(g2.alpha), alpha, rtol=1e-6)
        np.testing.assert_allclose(float(g2.beta), beta, rtol=1e-6)

    def test_log_prob_vs_scipy(self):
        from scipy.stats import gamma as sp_gamma
        g = Gamma(alpha=2.0, beta=3.0)
        xs = [0.1, 0.5, 1.0, 2.0, 5.0]
        for x in xs:
            our = float(g.log_prob(jnp.array(x)))
            ref = float(sp_gamma.logpdf(x, a=2.0, scale=1.0/3.0))
            assert abs(our - ref) < 1e-9, f"x={x}: ours={our}, scipy={ref}"

    def test_fit_mle(self):
        key = jax.random.PRNGKey(1)
        alpha_true, beta_true = 3.0, 2.0
        # Draw samples using a fixed method
        np.random.seed(42)
        X = np.random.gamma(shape=alpha_true, scale=1.0/beta_true, size=5000)
        X = jnp.array(X)
        g = Gamma.fit_mle(X)
        np.testing.assert_allclose(float(g.alpha), alpha_true, rtol=0.05)
        np.testing.assert_allclose(float(g.beta), beta_true, rtol=0.05)

    def test_fisher_information_positive_definite(self):
        g = Gamma(alpha=2.0, beta=1.5)
        FI = g.fisher_information()
        assert FI.shape == (2, 2)
        eigvals = jnp.linalg.eigvalsh(FI)
        assert jnp.all(eigvals > 0), f"Fisher info not PD: {eigvals}"


# ===========================================================================
# InverseGamma
# ===========================================================================

class TestInverseGamma:

    def test_natural_params(self):
        ig = InverseGamma(alpha=2.0, beta=3.0)
        theta = ig.natural_params()
        # θ = [β, -(α+1)] = [3, -3]
        np.testing.assert_allclose(theta, [3.0, -3.0], rtol=1e-12)

    def test_expectation_params(self):
        alpha, beta = 3.0, 2.0
        ig = InverseGamma(alpha=alpha, beta=beta)
        eta = ig.expectation_params()
        from scipy.special import digamma
        np.testing.assert_allclose(float(eta[0]), -alpha / beta, rtol=1e-8)
        np.testing.assert_allclose(float(eta[1]),
                                   np.log(beta) - digamma(alpha), rtol=1e-8)

    def test_from_expectation_roundtrip(self):
        alpha, beta = 3.0, 2.0
        ig = InverseGamma(alpha=alpha, beta=beta)
        eta = ig.expectation_params()
        ig2 = InverseGamma.from_expectation(eta)
        np.testing.assert_allclose(float(ig2.alpha), alpha, rtol=1e-6)
        np.testing.assert_allclose(float(ig2.beta), beta, rtol=1e-6)

    def test_log_prob_vs_scipy(self):
        from scipy.stats import invgamma
        alpha, beta = 2.0, 3.0
        ig = InverseGamma(alpha=alpha, beta=beta)
        xs = [0.1, 0.5, 1.0, 2.0, 5.0]
        for x in xs:
            our = float(ig.log_prob(jnp.array(x)))
            # scipy invgamma: shape=alpha, scale=beta (rate=1/scale=1/beta)
            ref = float(invgamma.logpdf(x, a=alpha, scale=beta))
            assert abs(our - ref) < 1e-9, f"x={x}: ours={our}, scipy={ref}"


# ===========================================================================
# InverseGaussian
# ===========================================================================

class TestInverseGaussian:

    def test_natural_params(self):
        mu, lam = 2.0, 4.0
        ig = InverseGaussian(mu=mu, lam=lam)
        theta = ig.natural_params()
        # θ = [-λ/(2μ²), -λ/2] = [-4/8, -2] = [-0.5, -2]
        np.testing.assert_allclose(theta, [-0.5, -2.0], rtol=1e-12)

    def test_from_natural_roundtrip(self):
        mu, lam = 2.0, 4.0
        ig = InverseGaussian(mu=mu, lam=lam)
        theta = ig.natural_params()
        ig2 = InverseGaussian.from_natural(theta)
        np.testing.assert_allclose(float(ig2.mu), mu, rtol=1e-10)
        np.testing.assert_allclose(float(ig2.lam), lam, rtol=1e-10)

    def test_expectation_params(self):
        mu, lam = 2.0, 4.0
        ig = InverseGaussian(mu=mu, lam=lam)
        eta = ig.expectation_params()
        np.testing.assert_allclose(float(eta[0]), mu, rtol=1e-10)
        np.testing.assert_allclose(float(eta[1]), 1.0/mu + 1.0/lam, rtol=1e-10)

    def test_from_expectation_roundtrip(self):
        mu, lam = 2.0, 4.0
        ig = InverseGaussian(mu=mu, lam=lam)
        eta = ig.expectation_params()
        ig2 = InverseGaussian.from_expectation(eta)
        np.testing.assert_allclose(float(ig2.mu), mu, rtol=1e-8)
        np.testing.assert_allclose(float(ig2.lam), lam, rtol=1e-8)

    def test_log_prob_vs_scipy(self):
        from scipy.stats import invgauss as sp_invgauss
        mu, lam = 2.0, 4.0
        ig = InverseGaussian(mu=mu, lam=lam)
        xs = [0.5, 1.0, 2.0, 3.0]
        for x in xs:
            our = float(ig.log_prob(jnp.array(x)))
            # scipy invgauss(mu=mu/lam, scale=lam): mean=mu, shape=lam
            # scipy parameterization: mu_sp = mu/scale = mu/lam,  scale = lam
            ref = float(sp_invgauss.logpdf(x, mu=mu / lam, scale=lam))
            assert abs(our - ref) < 1e-9, f"x={x}: ours={our}, scipy={ref}"

    def test_cdf_moderate(self):
        from scipy.stats import invgauss as sp_invgauss
        mu, lam = 2.0, 4.0
        ig = InverseGaussian(mu=mu, lam=lam)
        for x in [0.5, 1.0, 2.0, 3.0, 5.0]:
            our = float(ig.cdf(jnp.array(x)))
            ref = float(sp_invgauss.cdf(x, mu=mu / lam, scale=lam))
            np.testing.assert_allclose(our, ref, rtol=1e-8,
                                       err_msg=f"CDF mismatch at x={x}")

    @pytest.mark.parametrize("mu,lam,x", [
        (1.0, 1000.0, 1.0),
        (1.0, 1e4, 1.0),
        (1.0, 1e6, 1.0),
        (0.5, 1000.0, 0.5),
        (2.0, 500.0, 2.0),
    ])
    def test_cdf_extreme_shape(self, mu, lam, x):
        """CDF must be finite and match scipy in high-shape (lam/mu large) regimes."""
        from scipy.stats import invgauss as sp_invgauss
        ig = InverseGaussian(mu=mu, lam=lam)
        our = float(ig.cdf(jnp.array(x)))
        ref = float(sp_invgauss.cdf(x, mu=mu / lam, scale=lam))
        assert np.isfinite(our), f"CDF returned non-finite for mu={mu}, lam={lam}, x={x}"
        np.testing.assert_allclose(our, ref, rtol=1e-6,
                                   err_msg=f"mu={mu}, lam={lam}, x={x}")


# ===========================================================================
# GIG
# ===========================================================================

class TestGIG:

    def test_log_prob_vs_scipy(self):
        from scipy.stats import geninvgauss
        p, a, b = 1.0, 1.0, 1.0
        gig = GIG(p=p, a=a, b=b)
        # scipy params: b_scipy=sqrt(ab)=1, scale=sqrt(b/a)=1
        xs = [0.5, 1.0, 2.0, 3.0]
        for x in xs:
            our = float(gig.log_prob(jnp.array(x)))
            ref = float(geninvgauss.logpdf(x, p=p, b=np.sqrt(a*b),
                                           scale=np.sqrt(b/a)))
            assert abs(our - ref) < 1e-8, f"x={x}: ours={our}, scipy={ref}"

    def test_natural_params(self):
        p, a, b = 2.0, 3.0, 4.0
        gig = GIG(p=p, a=a, b=b)
        theta = gig.natural_params()
        # θ = [p-1, -b/2, -a/2] = [1, -2, -1.5]
        np.testing.assert_allclose(theta, [1.0, -2.0, -1.5], rtol=1e-12)

    def test_from_natural_roundtrip(self):
        p, a, b = 2.0, 3.0, 4.0
        gig = GIG(p=p, a=a, b=b)
        theta = gig.natural_params()
        gig2 = GIG.from_natural(theta)
        np.testing.assert_allclose(float(gig2.p), p, rtol=1e-10)
        np.testing.assert_allclose(float(gig2.a), a, rtol=1e-10)
        np.testing.assert_allclose(float(gig2.b), b, rtol=1e-10)

    def test_expectation_params(self):
        """E[X], E[1/X] from Bessel ratios."""
        from scipy.special import kv as scipy_kv
        p, a, b = 1.0, 1.0, 1.0
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        sqrt_ab = np.sqrt(a * b)
        E_inv = float(np.exp(np.log(scipy_kv(p-1, sqrt_ab)) - np.log(scipy_kv(p, sqrt_ab))))
        E_x = float(np.exp(np.log(scipy_kv(p+1, sqrt_ab)) - np.log(scipy_kv(p, sqrt_ab))))
        np.testing.assert_allclose(float(eta[1]), E_inv, rtol=1e-6)
        np.testing.assert_allclose(float(eta[2]), E_x, rtol=1e-6)

    def test_from_expectation_roundtrip(self):
        """η → θ → η roundtrip."""
        p, a, b = 1.5, 2.0, 0.5
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        gig2 = GIG.from_expectation(eta)
        np.testing.assert_allclose(float(gig2.p), p, rtol=1e-4)
        np.testing.assert_allclose(float(gig2.a), a, rtol=1e-4)
        np.testing.assert_allclose(float(gig2.b), b, rtol=1e-4)

    @pytest.mark.parametrize("p,a,b", [
        (2.0, 1.0, 1e-12),   # Near Gamma limit
        (-2.0, 1e-12, 1.0),  # Near InverseGamma limit
        (-0.5, 1.0, 1.0),    # InverseGaussian special case
    ])
    def test_degenerate_limits(self, p, a, b):
        """GIG degenerate cases should have finite log_prob."""
        gig = GIG(p=p, a=a, b=b)
        lp = float(gig.log_partition())
        assert np.isfinite(lp), f"Log partition not finite for p={p},a={a},b={b}: {lp}"

    def test_fisher_information_finite(self):
        gig = GIG(p=1.0, a=1.0, b=1.0)
        FI = gig.fisher_information()
        assert FI.shape == (3, 3)
        assert jnp.all(jnp.isfinite(FI))


# ===========================================================================
# GeneralizedHyperbolic (marginal)
# ===========================================================================

class TestGeneralizedHyperbolic:

    @pytest.fixture
    def gh_1d(self):
        return GeneralizedHyperbolic.from_classical(
            mu=np.array([0.0]),
            gamma=np.array([0.0]),
            sigma=np.array([[1.0]]),
            p=1.0, a=1.0, b=1.0,
        )

    @pytest.fixture
    def gh_2d(self):
        return GeneralizedHyperbolic.from_classical(
            mu=np.array([0.5, -0.3]),
            gamma=np.array([0.2, 0.1]),
            sigma=np.array([[1.0, 0.3], [0.3, 1.5]]),
            p=1.0, a=1.0, b=1.0,
        )

    def test_log_prob_finite(self, gh_1d):
        xs = jnp.array([[0.0], [1.0], [-1.0], [2.0]])
        lps = jax.vmap(gh_1d.log_prob)(xs)
        assert jnp.all(jnp.isfinite(lps))

    def test_log_prob_decreases_away_from_mode(self, gh_1d):
        """For symmetric GH, log_prob should decrease as |x| increases."""
        x0 = float(gh_1d.log_prob(jnp.array([0.0])))
        x1 = float(gh_1d.log_prob(jnp.array([1.0])))
        x2 = float(gh_1d.log_prob(jnp.array([2.0])))
        assert x0 > x1 > x2

    def test_log_prob_2d_finite(self, gh_2d):
        X = jax.random.normal(jax.random.PRNGKey(0), (20, 2), dtype=jnp.float64)
        lps = jax.vmap(gh_2d.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_conditional_expectations_finite(self, gh_2d):
        x = jnp.array([0.5, -0.3])
        cond = gh_2d._joint.conditional_expectations(x)
        for k, v in cond.items():
            assert np.isfinite(float(v)), f"{k} is not finite: {v}"
        assert float(cond['E_Y']) > 0
        assert float(cond['E_inv_Y']) > 0

    def test_e_step_shapes(self, gh_2d):
        from normix.fitting.eta import NormalMixtureEta
        X = jax.random.normal(jax.random.PRNGKey(1), (30, 2), dtype=jnp.float64)
        eta = gh_2d.e_step(X)
        assert isinstance(eta, NormalMixtureEta)
        assert eta.E_Y.shape == ()
        assert eta.E_inv_Y.shape == ()
        assert eta.E_log_Y.shape == ()
        assert eta.E_X.shape == (2,)
        assert eta.E_X_inv_Y.shape == (2,)
        assert eta.E_XXT_inv_Y.shape == (2, 2)

    def test_m_step_increases_ll(self, gh_2d):
        X = jax.random.normal(jax.random.PRNGKey(2), (100, 2), dtype=jnp.float64)
        ll0 = float(gh_2d.marginal_log_likelihood(X))
        eta = gh_2d.e_step(X)
        gh_new = gh_2d.m_step(eta)
        ll1 = float(gh_new.marginal_log_likelihood(X))
        assert ll1 >= ll0 - 1e-6, f"LL decreased: {ll0:.4f} → {ll1:.4f}"

    def test_em_convergence(self):
        """EM should monotonically increase log-likelihood."""
        from normix.fitting.em import BatchEMFitter
        np.random.seed(0)
        X = jnp.array(np.random.standard_normal((200, 2)))
        gh0 = GeneralizedHyperbolic.from_classical(
            mu=np.zeros(2), gamma=np.zeros(2),
            sigma=np.eye(2), p=1.0, a=1.0, b=1.0,
        )
        ll_prev = float(gh0.marginal_log_likelihood(X))
        model = gh0
        for _ in range(5):
            eta = model.e_step(X)
            model = model.m_step(eta)
            ll = float(model.marginal_log_likelihood(X))
            assert ll >= ll_prev - 1e-4, f"LL decreased: {ll_prev:.4f} → {ll:.4f}"
            ll_prev = ll


# ===========================================================================
# Joint normal mixtures — exponential-family log_prob vs explicit joint density
# ===========================================================================

class TestJointExponentialFamilyLogProb:
    """``log_prob(concat(x,[y]))`` must agree with ``log_prob_joint(x, y)``."""

    def test_joint_variance_gamma(self):
        mu = jnp.array([0.0, 0.0])
        gamma = jnp.array([0.1, -0.1])
        L = jnp.eye(2)
        j = JointVarianceGamma(mu, gamma, L, alpha=2.0, beta=1.0)
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-12)

    def test_joint_generalized_hyperbolic(self):
        mu = jnp.array([0.0, 0.0])
        gamma = jnp.array([0.1, -0.1])
        L = jnp.eye(2)
        j = JointGeneralizedHyperbolic(mu, gamma, L, p=1.0, a=1.0, b=1.0)
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-12)

    def test_joint_normal_inverse_gaussian(self):
        mu = jnp.array([0.0, 0.0])
        gamma = jnp.array([0.1, -0.1])
        L = jnp.eye(2)
        j = JointNormalInverseGaussian(mu, gamma, L, mu_ig=1.0, lam=2.0)
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-12)

    def test_joint_normal_inverse_gamma(self):
        mu = jnp.array([0.0, 0.0])
        gamma = jnp.array([0.1, -0.1])
        L = jnp.eye(2)
        j = JointNormalInverseGamma(mu, gamma, L, alpha=3.0, beta=1.5)
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-12)


# ===========================================================================
# Joint normal mixtures — exponential-family round-trip tests (T3)
# ===========================================================================

class TestJointExponentialFamilyRoundTrip:
    """
    Verify the EF contract for each joint class:
      1. log_partition() agrees with _log_partition_from_theta(natural_params()).
      2. expectation_params() = jax.grad(_log_partition_from_theta)(natural_params()).
      3. rvs produces finite (X, Y) samples.
      4. conditional_expectations returns finite, sign-correct moments.
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_ef_contract(joint):
        """Shared checks: log_partition self-consistency and grad=eta."""
        theta = joint.natural_params()
        psi_direct = float(joint.log_partition())
        psi_from_theta = float(type(joint)._log_partition_from_theta(theta))
        np.testing.assert_allclose(psi_direct, psi_from_theta, rtol=1e-10,
                                   err_msg="log_partition() != _log_partition_from_theta(theta)")

        eta = joint.expectation_params()
        grad_eta = jax.grad(type(joint)._log_partition_from_theta)(theta)
        np.testing.assert_allclose(np.array(eta), np.array(grad_eta), rtol=1e-6,
                                   err_msg="expectation_params() != grad log_partition")

    @staticmethod
    def _check_rvs(joint, n=30):
        X, Y = joint.rvs(n, seed=0)
        assert X.shape == (n, joint.d)
        assert Y.shape == (n,)
        assert jnp.all(jnp.isfinite(X)), "rvs X contains non-finite values"
        assert jnp.all(jnp.isfinite(Y)), "rvs Y contains non-finite values"
        assert jnp.all(Y > 0), "rvs Y contains non-positive values"

    @staticmethod
    def _check_cond_expectations(joint):
        x = jnp.zeros(joint.d, dtype=jnp.float64)
        cond = joint.conditional_expectations(x)
        for k, v in cond.items():
            assert np.isfinite(float(v)), f"conditional_expectations[{k}] is not finite"
        assert float(cond['E_Y']) > 0
        assert float(cond['E_inv_Y']) > 0

    # -----------------------------------------------------------------------
    # JointVarianceGamma
    # -----------------------------------------------------------------------

    def test_vg_ef_contract(self):
        j = JointVarianceGamma(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.2, 0.1]),
            L_Sigma=jnp.array([[1.0, 0.0], [0.3, 0.9]]),
            alpha=2.0, beta=1.0,
        )
        self._check_ef_contract(j)

    def test_vg_rvs(self):
        j = JointVarianceGamma(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), alpha=2.0, beta=1.0,
        )
        self._check_rvs(j)

    def test_vg_cond_expectations(self):
        j = JointVarianceGamma(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), alpha=2.0, beta=1.0,
        )
        self._check_cond_expectations(j)

    # -----------------------------------------------------------------------
    # JointNormalInverseGamma
    # -----------------------------------------------------------------------

    def test_ninvg_ef_contract(self):
        j = JointNormalInverseGamma(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.2, 0.1]),
            L_Sigma=jnp.array([[1.0, 0.0], [0.3, 0.9]]),
            alpha=3.0, beta=1.5,
        )
        self._check_ef_contract(j)

    def test_ninvg_rvs(self):
        j = JointNormalInverseGamma(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), alpha=3.0, beta=1.5,
        )
        self._check_rvs(j)

    def test_ninvg_cond_expectations(self):
        j = JointNormalInverseGamma(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), alpha=3.0, beta=1.5,
        )
        self._check_cond_expectations(j)

    # -----------------------------------------------------------------------
    # JointNormalInverseGaussian
    # -----------------------------------------------------------------------

    def test_nig_ef_contract(self):
        j = JointNormalInverseGaussian(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.2, 0.1]),
            L_Sigma=jnp.array([[1.0, 0.0], [0.3, 0.9]]),
            mu_ig=1.0, lam=2.0,
        )
        self._check_ef_contract(j)

    def test_nig_rvs(self):
        j = JointNormalInverseGaussian(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), mu_ig=1.0, lam=2.0,
        )
        self._check_rvs(j)

    def test_nig_cond_expectations(self):
        j = JointNormalInverseGaussian(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), mu_ig=1.0, lam=2.0,
        )
        self._check_cond_expectations(j)

    # -----------------------------------------------------------------------
    # JointGeneralizedHyperbolic
    # -----------------------------------------------------------------------

    def test_gh_ef_contract(self):
        j = JointGeneralizedHyperbolic(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.2, 0.1]),
            L_Sigma=jnp.array([[1.0, 0.0], [0.3, 0.9]]),
            p=1.0, a=1.0, b=1.0,
        )
        self._check_ef_contract(j)

    def test_gh_rvs(self):
        j = JointGeneralizedHyperbolic(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), p=1.0, a=1.0, b=1.0,
        )
        self._check_rvs(j)

    def test_gh_cond_expectations(self):
        j = JointGeneralizedHyperbolic(
            mu=jnp.zeros(2), gamma=jnp.array([0.1, -0.1]),
            L_Sigma=jnp.eye(2), p=1.0, a=1.0, b=1.0,
        )
        self._check_cond_expectations(j)


# ===========================================================================
# MultivariateNormal — ExponentialFamily round-trip tests (D3)
# ===========================================================================

class TestMultivariateNormal:
    """
    Validate the full ExponentialFamily contract for MultivariateNormal:
      1. log_partition() agrees with _log_partition_from_theta(natural_params()).
      2. natural_params() → from_natural() recovers original parameters.
      3. expectation_params() = jax.grad(_log_partition_from_theta)(natural_params()).
      4. log_prob agrees with scipy.stats.multivariate_normal.
      5. mean(), cov() return correct values.
      6. rvs produces finite samples with correct shape.
      7. fit_mle recovers parameters from large sample.
    """

    @pytest.fixture
    def mvn_1d(self):
        return MultivariateNormal.from_classical(
            mu=jnp.array([1.5]),
            sigma=jnp.array([[2.0]]),
        )

    @pytest.fixture
    def mvn_2d(self):
        return MultivariateNormal.from_classical(
            mu=jnp.array([1.0, -0.5]),
            sigma=jnp.array([[2.0, 0.6], [0.6, 1.5]]),
        )

    def test_log_partition_consistency(self, mvn_2d):
        theta = mvn_2d.natural_params()
        psi_direct = float(mvn_2d.log_partition())
        psi_from_theta = float(MultivariateNormal._log_partition_from_theta(theta))
        np.testing.assert_allclose(psi_direct, psi_from_theta, rtol=1e-10)

    def test_from_natural_roundtrip(self, mvn_2d):
        theta = mvn_2d.natural_params()
        mvn2 = MultivariateNormal.from_natural(theta)
        np.testing.assert_allclose(
            np.array(mvn2.mu), np.array(mvn_2d.mu), rtol=1e-9)
        np.testing.assert_allclose(
            np.array(mvn2.cov()), np.array(mvn_2d.cov()), rtol=1e-9)

    def test_expectation_params_eq_grad(self, mvn_2d):
        theta = mvn_2d.natural_params()
        eta = mvn_2d.expectation_params()
        grad_eta = jax.grad(MultivariateNormal._log_partition_from_theta)(theta)
        np.testing.assert_allclose(
            np.array(eta), np.array(grad_eta), rtol=1e-6)

    def test_log_prob_vs_scipy(self, mvn_2d):
        from scipy.stats import multivariate_normal
        mu = np.array([1.0, -0.5])
        cov = np.array([[2.0, 0.6], [0.6, 1.5]])
        xs = [np.array([0.0, 0.0]), np.array([1.0, -0.5]), np.array([2.0, 1.0])]
        for x in xs:
            our = float(mvn_2d.log_prob(jnp.array(x)))
            ref = float(multivariate_normal.logpdf(x, mean=mu, cov=cov))
            np.testing.assert_allclose(our, ref, rtol=1e-9,
                                       err_msg=f"log_prob mismatch at x={x}")

    def test_log_prob_1d_vs_scipy(self, mvn_1d):
        from scipy.stats import norm
        for x in [-1.0, 0.0, 1.5, 3.0]:
            our = float(mvn_1d.log_prob(jnp.array([x])))
            ref = float(norm.logpdf(x, loc=1.5, scale=np.sqrt(2.0)))
            np.testing.assert_allclose(our, ref, rtol=1e-9)

    def test_mean(self, mvn_2d):
        mu = np.array([1.0, -0.5])
        np.testing.assert_allclose(np.array(mvn_2d.mean()), mu, rtol=1e-12)

    def test_cov(self, mvn_2d):
        cov = np.array([[2.0, 0.6], [0.6, 1.5]])
        np.testing.assert_allclose(np.array(mvn_2d.cov()), cov, rtol=1e-9)

    def test_rvs_shape_and_finite(self, mvn_2d):
        X = mvn_2d.rvs(50, seed=7)
        assert X.shape == (50, 2)
        assert jnp.all(jnp.isfinite(X))

    def test_rvs_empirical_mean(self, mvn_2d):
        X = mvn_2d.rvs(10000, seed=0)
        emp_mean = jnp.mean(X, axis=0)
        np.testing.assert_allclose(
            np.array(emp_mean), np.array(mvn_2d.mean()), atol=0.05)

    def test_fit_mle_recovers_params(self):
        """fit_mle should recover μ and Σ from a large sample."""
        mu_true = jnp.array([1.0, -0.5])
        cov_true = jnp.array([[2.0, 0.6], [0.6, 1.5]])
        mvn_true = MultivariateNormal.from_classical(mu=mu_true, sigma=cov_true)
        X = mvn_true.rvs(5000, seed=42)
        mvn_fit = MultivariateNormal.fit_mle(X)
        np.testing.assert_allclose(
            np.array(mvn_fit.mean()), np.array(mu_true), atol=0.05)
        np.testing.assert_allclose(
            np.array(mvn_fit.cov()), np.array(cov_true), atol=0.1)

    def test_sample_backward_compat(self, mvn_2d):
        """Legacy sample(key, shape) API still works."""
        key = jax.random.PRNGKey(0)
        X = mvn_2d.sample(key, shape=(20,))
        assert X.shape == (20, 2)
        assert jnp.all(jnp.isfinite(X))
