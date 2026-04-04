"""
Tests for the ExponentialFamily base class (current API).

Uses Gamma distribution as a concrete test case to verify the
exponential-family framework: log_prob decomposition, natural/expectation
parameter conversions, fit_mle convergence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import ExponentialFamily, Gamma, InverseGamma, InverseGaussian, GIG


class TestExponentialFamilyContract:
    """Verify the EF contract using Gamma as a concrete subclass."""

    def test_log_prob_decomposition(self):
        """log p(x) = log h(x) + θ·t(x) - ψ(θ)."""
        g = Gamma(alpha=2.0, beta=3.0)
        x = jnp.array(1.5)
        theta = g.natural_params()
        t = Gamma.sufficient_statistics(x)
        psi = g.log_partition()
        h = Gamma.log_base_measure(x)
        manual = float(h) + float(jnp.dot(theta, t)) - float(psi)
        np.testing.assert_allclose(float(g.log_prob(x)), manual, rtol=1e-10)

    def test_pdf_equals_exp_log_prob(self):
        g = Gamma(alpha=2.0, beta=3.0)
        x = jnp.array(1.5)
        np.testing.assert_allclose(float(g.pdf(x)),
                                   float(jnp.exp(g.log_prob(x))), rtol=1e-12)

    def test_from_natural_invertible(self):
        g = Gamma(alpha=3.0, beta=2.0)
        theta = g.natural_params()
        g2 = Gamma.from_natural(theta)
        np.testing.assert_allclose(float(g2.alpha), 3.0, rtol=1e-10)
        np.testing.assert_allclose(float(g2.beta), 2.0, rtol=1e-10)

    def test_from_expectation_invertible(self):
        g = Gamma(alpha=3.0, beta=2.0)
        eta = g.expectation_params()
        g2 = Gamma.from_expectation(eta)
        np.testing.assert_allclose(float(g2.alpha), 3.0, rtol=1e-5)
        np.testing.assert_allclose(float(g2.beta), 2.0, rtol=1e-5)

    def test_fisher_information_shape(self):
        g = Gamma(alpha=2.0, beta=3.0)
        FI = g.fisher_information()
        assert FI.shape == (2, 2)
        assert jnp.all(jnp.isfinite(FI))

    def test_fit_mle_recovers_params(self):
        key = jax.random.PRNGKey(0)
        alpha_true, beta_true = 3.0, 2.0
        g_true = Gamma(alpha=alpha_true, beta=beta_true)
        data = g_true.rvs(5000, seed=42)
        g_init = Gamma(alpha=1.0, beta=1.0)
        g_fit = g_init.fit(data)
        np.testing.assert_allclose(float(g_fit.alpha), alpha_true, rtol=0.1)
        np.testing.assert_allclose(float(g_fit.beta), beta_true, rtol=0.1)


class TestExponentialFamilyInvariants:
    """T2: Mathematical invariants that should hold for all EF distributions."""

    @pytest.mark.parametrize("dist", [
        Gamma(alpha=2.0, beta=3.0),
        Gamma(alpha=5.0, beta=0.5),
        InverseGamma(alpha=5.0, beta=2.0),
        InverseGamma(alpha=8.0, beta=1.5),
        InverseGaussian(mu=1.0, lam=1.0),
        InverseGaussian(mu=2.0, lam=0.5),
        GIG(p=1.0, a=1.0, b=1.0),
        GIG(p=2.0, a=2.5, b=1.5),
        GIG(p=-0.5, a=2.0, b=1.0),
    ], ids=lambda d: f"{type(d).__name__}({','.join(f'{float(v):.1f}' for v in jax.tree.leaves(d))})")
    def test_gradient_log_partition_equals_eta(self, dist):
        """∇ψ(θ) must equal η = E[t(X)]."""
        theta = dist.natural_params()
        grad_psi = jax.grad(type(dist)._log_partition_from_theta)(theta)
        eta = dist.expectation_params()
        np.testing.assert_allclose(np.array(grad_psi), np.array(eta), rtol=1e-4)

    @pytest.mark.parametrize("dist", [
        Gamma(alpha=2.0, beta=3.0),
        Gamma(alpha=5.0, beta=0.5),
        InverseGamma(alpha=5.0, beta=2.0),
        InverseGamma(alpha=8.0, beta=1.5),
        InverseGaussian(mu=1.0, lam=1.0),
        InverseGaussian(mu=2.0, lam=0.5),
    ], ids=lambda d: f"{type(d).__name__}({','.join(f'{float(v):.1f}' for v in jax.tree.leaves(d))})")
    def test_hessian_log_partition_spd(self, dist):
        """∇²ψ(θ) (Fisher information) must be symmetric positive definite."""
        FI = np.array(dist.fisher_information())
        np.testing.assert_allclose(FI, FI.T, rtol=1e-8)
        eigvals = np.linalg.eigvalsh(FI)
        assert np.all(eigvals > 0), f"Not PD: eigvals={eigvals}"

    @pytest.mark.parametrize("dist", [
        GIG(p=1.0, a=1.0, b=1.0),
        GIG(p=2.0, a=2.5, b=1.5),
        GIG(p=-0.5, a=2.0, b=1.0),
    ], ids=lambda d: f"GIG({','.join(f'{float(v):.1f}' for v in jax.tree.leaves(d))})")
    def test_hessian_log_partition_finite_gig(self, dist):
        """GIG Hessian (numerical) should be finite and symmetric."""
        FI = np.array(dist.fisher_information())
        assert np.all(np.isfinite(FI)), f"GIG Fisher info not finite"
        np.testing.assert_allclose(FI, FI.T, rtol=1e-6)

    @pytest.mark.parametrize("dist,rtol", [
        (Gamma(alpha=2.0, beta=3.0), 0.01),
        (InverseGamma(alpha=5.0, beta=2.0), 0.01),
        (InverseGaussian(mu=1.0, lam=1.0), 0.01),
        (GIG(p=1.5, a=2.0, b=0.5), 0.05),
    ], ids=["Gamma", "InverseGamma", "InverseGaussian", "GIG"])
    def test_natural_expectation_roundtrip(self, dist, rtol):
        """θ → η → θ roundtrip must recover original parameters."""
        theta = dist.natural_params()
        eta = dist.expectation_params()
        dist2 = type(dist).from_expectation(eta)
        theta2 = dist2.natural_params()
        np.testing.assert_allclose(np.array(theta), np.array(theta2), rtol=rtol)
