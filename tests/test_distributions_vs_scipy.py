"""
Tests comparing normix univariate distributions against SciPy references.

Covers:
  - PDF / CDF comparison
  - Natural ↔ Expectation parameter roundtrips
  - Gradient of log-partition = expectation parameters (T2 invariant)
  - Hessian SPD (T2 invariant)
  - Random sampling moment checks
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from scipy.special import digamma

jax.config.update("jax_enable_x64", True)

from normix.distributions.gamma import Gamma
from normix.distributions.inverse_gamma import InverseGamma
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.distributions.generalized_inverse_gaussian import GIG


# ============================================================
# Gamma
# ============================================================

class TestGammaVsScipy:

    @pytest.mark.parametrize("alpha,beta", [
        (2.0, 1.0), (2.0, 2.0), (3.0, 1.5), (5.0, 2.0),
    ])
    def test_pdf_comparison(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        xs = jnp.linspace(0.01, 10.0, 50)
        for x in xs:
            our = float(g.pdf(x))
            ref = float(stats.gamma.pdf(float(x), a=alpha, scale=1.0 / beta))
            np.testing.assert_allclose(our, ref, rtol=1e-6,
                                       err_msg=f"alpha={alpha}, beta={beta}, x={float(x)}")

    @pytest.mark.parametrize("alpha,beta", [
        (2.0, 1.0), (3.0, 1.5), (5.0, 2.0),
    ])
    def test_cdf_comparison(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        xs = jnp.linspace(0.01, 10.0, 50)
        for x in xs:
            our = float(g.cdf(x))
            ref = float(stats.gamma.cdf(float(x), a=alpha, scale=1.0 / beta))
            np.testing.assert_allclose(our, ref, rtol=1e-6,
                                       err_msg=f"alpha={alpha}, beta={beta}, x={float(x)}")

    @pytest.mark.parametrize("alpha,beta", [
        (2.0, 1.0), (3.0, 2.0), (5.0, 0.5),
    ])
    def test_natural_expectation_roundtrip(self, alpha, beta):
        g = Gamma(alpha=alpha, beta=beta)
        theta = g.natural_params()
        eta = g.expectation_params()
        g2 = Gamma.from_expectation(eta)
        theta2 = g2.natural_params()
        np.testing.assert_allclose(np.array(theta), np.array(theta2), rtol=1e-5)

    @pytest.mark.parametrize("alpha,beta", [
        (2.0, 1.0), (3.0, 2.0), (5.0, 0.5),
    ])
    def test_gradient_equals_expectation(self, alpha, beta):
        """∇ψ(θ) = η: gradient of log-partition equals expectation parameters."""
        g = Gamma(alpha=alpha, beta=beta)
        theta = g.natural_params()
        grad_psi = jax.grad(Gamma._log_partition_from_theta)(theta)
        eta = g.expectation_params()
        np.testing.assert_allclose(np.array(grad_psi), np.array(eta), rtol=1e-6)

    @pytest.mark.parametrize("alpha,beta", [
        (2.0, 1.0), (3.0, 2.0), (5.0, 0.5),
    ])
    def test_hessian_spd(self, alpha, beta):
        """Hessian of log-partition (Fisher info) must be symmetric positive definite."""
        g = Gamma(alpha=alpha, beta=beta)
        FI = np.array(g.fisher_information())
        np.testing.assert_allclose(FI, FI.T, rtol=1e-10)
        eigvals = np.linalg.eigvalsh(FI)
        assert np.all(eigvals > 0), f"Fisher info not PD: eigvals={eigvals}"

    def test_rvs_moments(self):
        alpha, beta = 3.0, 2.0
        g = Gamma(alpha=alpha, beta=beta)
        samples = np.array(g.rvs(10000, seed=42))
        np.testing.assert_allclose(samples.mean(), alpha / beta, rtol=0.05)
        np.testing.assert_allclose(samples.var(), alpha / beta**2, rtol=0.1)


# ============================================================
# Inverse Gamma
# ============================================================

class TestInverseGammaVsScipy:

    @pytest.mark.parametrize("alpha,beta", [
        (5.0, 2.0), (6.0, 2.0), (7.0, 1.0), (8.0, 1.5),
    ])
    def test_pdf_comparison(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        xs = jnp.linspace(0.01, 5.0, 50)
        for x in xs:
            our = float(ig.pdf(x))
            ref = float(stats.invgamma.pdf(float(x), a=alpha, scale=beta))
            np.testing.assert_allclose(our, ref, rtol=1e-6,
                                       err_msg=f"alpha={alpha}, beta={beta}, x={float(x)}")

    @pytest.mark.parametrize("alpha,beta", [
        (5.0, 2.0), (6.0, 2.0), (7.0, 1.0),
    ])
    def test_cdf_comparison(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        xs = jnp.linspace(0.05, 5.0, 50)
        for x in xs:
            our = float(ig.cdf(x))
            ref = float(stats.invgamma.cdf(float(x), a=alpha, scale=beta))
            np.testing.assert_allclose(our, ref, rtol=1e-5, atol=1e-15,
                                       err_msg=f"alpha={alpha}, beta={beta}, x={float(x)}")

    @pytest.mark.parametrize("alpha,beta", [
        (5.0, 2.0), (6.0, 2.0), (8.0, 1.5),
    ])
    def test_natural_expectation_roundtrip(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        theta = ig.natural_params()
        eta = ig.expectation_params()
        ig2 = InverseGamma.from_expectation(eta)
        theta2 = ig2.natural_params()
        np.testing.assert_allclose(np.array(theta), np.array(theta2), rtol=1e-5)

    @pytest.mark.parametrize("alpha,beta", [
        (5.0, 2.0), (6.0, 2.0), (8.0, 1.5),
    ])
    def test_gradient_equals_expectation(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        theta = ig.natural_params()
        grad_psi = jax.grad(InverseGamma._log_partition_from_theta)(theta)
        eta = ig.expectation_params()
        np.testing.assert_allclose(np.array(grad_psi), np.array(eta), rtol=1e-6)

    @pytest.mark.parametrize("alpha,beta", [
        (5.0, 2.0), (6.0, 2.0), (8.0, 1.5),
    ])
    def test_hessian_spd(self, alpha, beta):
        ig = InverseGamma(alpha=alpha, beta=beta)
        FI = np.array(ig.fisher_information())
        np.testing.assert_allclose(FI, FI.T, rtol=1e-10)
        eigvals = np.linalg.eigvalsh(FI)
        assert np.all(eigvals > 0), f"Fisher info not PD: eigvals={eigvals}"

    def test_rvs_moments(self):
        alpha, beta = 5.0, 2.0
        ig = InverseGamma(alpha=alpha, beta=beta)
        samples = np.array(ig.rvs(10000, seed=42))
        np.testing.assert_allclose(samples.mean(), beta / (alpha - 1), rtol=0.05)


# ============================================================
# Inverse Gaussian
# ============================================================

class TestInverseGaussianVsScipy:

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (1.0, 5.0),
    ])
    def test_pdf_comparison(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        xs = jnp.linspace(0.01, 5.0, 50)
        for x in xs:
            our = float(ig.pdf(x))
            ref = float(stats.invgauss.pdf(float(x), mu=mu / lam, scale=lam))
            np.testing.assert_allclose(our, ref, rtol=1e-6,
                                       err_msg=f"mu={mu}, lam={lam}, x={float(x)}")

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 0.5), (0.5, 2.0),
    ])
    def test_cdf_comparison(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        xs = jnp.linspace(0.01, 5.0, 50)
        for x in xs:
            our = float(ig.cdf(x))
            ref = float(stats.invgauss.cdf(float(x), mu=mu / lam, scale=lam))
            np.testing.assert_allclose(our, ref, rtol=1e-5,
                                       err_msg=f"mu={mu}, lam={lam}, x={float(x)}")

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 0.5), (0.5, 2.0),
    ])
    def test_natural_expectation_roundtrip(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        theta = ig.natural_params()
        eta = ig.expectation_params()
        ig2 = InverseGaussian.from_expectation(eta)
        theta2 = ig2.natural_params()
        np.testing.assert_allclose(np.array(theta), np.array(theta2), rtol=1e-4)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 0.5), (0.5, 2.0),
    ])
    def test_gradient_equals_expectation(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        theta = ig.natural_params()
        grad_psi = jax.grad(InverseGaussian._log_partition_from_theta)(theta)
        eta = ig.expectation_params()
        np.testing.assert_allclose(np.array(grad_psi), np.array(eta), rtol=1e-5)

    @pytest.mark.parametrize("mu,lam", [
        (1.0, 1.0), (2.0, 0.5), (0.5, 2.0),
    ])
    def test_hessian_spd(self, mu, lam):
        ig = InverseGaussian(mu=mu, lam=lam)
        FI = np.array(ig.fisher_information())
        np.testing.assert_allclose(FI, FI.T, rtol=1e-10)
        eigvals = np.linalg.eigvalsh(FI)
        assert np.all(eigvals > 0), f"Fisher info not PD: eigvals={eigvals}"

    def test_rvs_moments(self):
        mu, lam = 2.0, 3.0
        ig = InverseGaussian(mu=mu, lam=lam)
        samples = np.array(ig.rvs(10000, seed=42))
        np.testing.assert_allclose(samples.mean(), mu, rtol=0.05)
        np.testing.assert_allclose(samples.var(), mu**3 / lam, rtol=0.15)


# ============================================================
# GIG
# ============================================================

class TestGIGVsScipy:

    @pytest.mark.parametrize("p,a,b", [
        (1.0, 1.0, 1.0), (2.0, 2.5, 1.5), (-0.5, 2.0, 1.0), (0.5, 1.0, 2.0),
    ])
    def test_pdf_comparison(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        xs = jnp.linspace(0.1, 5.0, 50)
        for x in xs:
            our = float(gig.pdf(x))
            ref = float(stats.geninvgauss.pdf(
                float(x), p=p, b=np.sqrt(a * b), scale=np.sqrt(b / a)))
            np.testing.assert_allclose(our, ref, rtol=1e-5,
                                       err_msg=f"p={p}, a={a}, b={b}, x={float(x)}")

    @pytest.mark.parametrize("p,a,b", [
        (1.0, 1.0, 1.0), (2.0, 2.5, 1.5),
    ])
    def test_moments_vs_scipy(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        sp_mean, sp_var = stats.geninvgauss.stats(
            p=p, b=np.sqrt(a * b), scale=np.sqrt(b / a), moments='mv')
        np.testing.assert_allclose(float(gig.mean()), float(sp_mean), rtol=1e-6)
        np.testing.assert_allclose(float(gig.var()), float(sp_var), rtol=1e-5)

    @pytest.mark.parametrize("p,a,b", [
        (1.5, 2.0, 0.5), (2.0, 2.5, 1.5),
    ])
    def test_natural_expectation_roundtrip(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        theta = gig.natural_params()
        eta = gig.expectation_params()
        gig2 = GIG.from_expectation(eta)
        theta2 = gig2.natural_params()
        np.testing.assert_allclose(np.array(theta), np.array(theta2), rtol=0.05)

    @pytest.mark.parametrize("p,a,b", [
        (1.0, 1.0, 1.0), (2.0, 2.5, 1.5), (-0.5, 2.0, 1.0),
    ])
    def test_gradient_equals_expectation(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        theta = gig.natural_params()
        grad_psi = jax.grad(GIG._log_partition_from_theta)(theta)
        eta = gig.expectation_params()
        np.testing.assert_allclose(np.array(grad_psi), np.array(eta), rtol=1e-4)

    @pytest.mark.parametrize("p,a,b", [
        (1.0, 1.0, 1.0), (2.0, 2.5, 1.5), (-0.5, 2.0, 1.0),
    ])
    def test_hessian_finite(self, p, a, b):
        """GIG Hessian (via jax.hessian) should be finite and symmetric."""
        gig = GIG(p=p, a=a, b=b)
        FI = np.array(gig.fisher_information())
        assert np.all(np.isfinite(FI)), f"Fisher info not finite"
        np.testing.assert_allclose(FI, FI.T, rtol=1e-6)

    def test_rvs_moments(self):
        p, a, b = 1.0, 1.0, 1.0
        gig = GIG(p=p, a=a, b=b)
        samples = np.array(gig.rvs(10000, seed=42))
        np.testing.assert_allclose(samples.mean(), float(gig.mean()), rtol=0.05)
