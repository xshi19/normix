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
    GeneralizedHyperbolic,
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
        X = jax.random.normal(jax.random.PRNGKey(1), (30, 2), dtype=jnp.float64)
        exp = gh_2d.e_step(X)
        assert exp['E_Y'].shape == (30,)
        assert exp['E_inv_Y'].shape == (30,)
        assert exp['E_log_Y'].shape == (30,)

    def test_m_step_increases_ll(self, gh_2d):
        X = jax.random.normal(jax.random.PRNGKey(2), (100, 2), dtype=jnp.float64)
        ll0 = float(gh_2d.marginal_log_likelihood(X))
        exp = gh_2d.e_step(X)
        gh_new = gh_2d.m_step(X, exp)
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
            exp = model.e_step(X)
            model = model.m_step(X, exp)
            ll = float(model.marginal_log_likelihood(X))
            assert ll >= ll_prev - 1e-4, f"LL decreased: {ll_prev:.4f} → {ll:.4f}"
            ll_prev = ll
