"""
Tests for normix.divergences — Hellinger distance and KL divergence.

Validates:
  - Tier 1 (functional core) correctness against analytical formulas
  - Tier 2 (instance methods) consistency with Tier 1
  - Tier 3 (module convenience) dispatch for ExponentialFamily and NormalMixture
  - JAX transformations: jit, vmap, grad
  - Mathematical properties: non-negativity, identity, symmetry (Hellinger)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import (
    Gamma, InverseGamma, InverseGaussian, GIG,
    JointVarianceGamma, VarianceGamma,
    JointGeneralizedHyperbolic, GeneralizedHyperbolic,
    squared_hellinger, kl_divergence,
    squared_hellinger_from_psi, kl_divergence_from_psi,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _gig_hellinger_analytical(p1, a1, b1, p2, a2, b2):
    """GIG squared Hellinger from gig.rst proposition (for cross-checking)."""
    from normix.utils.bessel import log_kv
    pb = (p1 + p2) / 2.0
    ab = (a1 + a2) / 2.0
    bb = (b1 + b2) / 2.0
    log_num = float(log_kv(pb, jnp.sqrt(ab * bb)))
    log_den1 = float(log_kv(p1, jnp.sqrt(a1 * b1)))
    log_den2 = float(log_kv(p2, jnp.sqrt(a2 * b2)))
    log_affinity = (
        0.25 * p1 * np.log(a1 / b1)
        + 0.25 * p2 * np.log(a2 / b2)
        - 0.5 * (log_den1 + log_den2)
        + log_num
        - 0.5 * pb * np.log(ab / bb)
    )
    return 1.0 - np.exp(log_affinity)


# ===========================================================================
# Tier 1: functional core
# ===========================================================================

class TestTier1Gamma:

    def test_hellinger_identical(self):
        g = Gamma(alpha=3.0, beta=2.0)
        theta = g.natural_params()
        h2 = squared_hellinger_from_psi(
            Gamma._log_partition_from_theta, theta, theta)
        np.testing.assert_allclose(float(h2), 0.0, atol=1e-12)

    def test_kl_identical(self):
        g = Gamma(alpha=3.0, beta=2.0)
        theta = g.natural_params()
        kl = kl_divergence_from_psi(
            Gamma._log_partition_from_theta,
            Gamma._grad_log_partition,
            theta, theta)
        np.testing.assert_allclose(float(kl), 0.0, atol=1e-12)

    def test_hellinger_nonneg(self):
        t1 = Gamma(alpha=2.0, beta=1.0).natural_params()
        t2 = Gamma(alpha=5.0, beta=3.0).natural_params()
        h2 = squared_hellinger_from_psi(
            Gamma._log_partition_from_theta, t1, t2)
        assert float(h2) >= 0.0

    def test_kl_nonneg(self):
        t1 = Gamma(alpha=2.0, beta=1.0).natural_params()
        t2 = Gamma(alpha=5.0, beta=3.0).natural_params()
        kl = kl_divergence_from_psi(
            Gamma._log_partition_from_theta,
            Gamma._grad_log_partition, t1, t2)
        assert float(kl) >= -1e-12

    def test_hellinger_symmetric(self):
        t1 = Gamma(alpha=2.0, beta=1.0).natural_params()
        t2 = Gamma(alpha=5.0, beta=3.0).natural_params()
        psi = Gamma._log_partition_from_theta
        h_12 = squared_hellinger_from_psi(psi, t1, t2)
        h_21 = squared_hellinger_from_psi(psi, t2, t1)
        np.testing.assert_allclose(float(h_12), float(h_21), rtol=1e-12)

    def test_kl_asymmetric(self):
        t1 = Gamma(alpha=2.0, beta=1.0).natural_params()
        t2 = Gamma(alpha=5.0, beta=3.0).natural_params()
        psi = Gamma._log_partition_from_theta
        grad = Gamma._grad_log_partition
        kl_12 = kl_divergence_from_psi(psi, grad, t1, t2)
        kl_21 = kl_divergence_from_psi(psi, grad, t2, t1)
        assert abs(float(kl_12) - float(kl_21)) > 1e-3


class TestTier1GIG:

    def test_hellinger_vs_analytical(self):
        """General exp-family formula matches the GIG-specific proposition."""
        p1, a1, b1 = 2.0, 3.0, 1.0
        p2, a2, b2 = -1.5, 2.0, 4.0
        g1 = GIG(p=p1, a=a1, b=b1)
        g2 = GIG(p=p2, a=a2, b=b2)
        h2_general = float(squared_hellinger_from_psi(
            GIG._log_partition_from_theta,
            g1.natural_params(), g2.natural_params()))
        h2_analytical = _gig_hellinger_analytical(p1, a1, b1, p2, a2, b2)
        np.testing.assert_allclose(h2_general, h2_analytical, rtol=1e-8)


# ===========================================================================
# Tier 2: instance methods
# ===========================================================================

class TestTier2:

    def test_gamma_instance_matches_tier1(self):
        g1 = Gamma(alpha=2.0, beta=1.0)
        g2 = Gamma(alpha=5.0, beta=3.0)
        h2_t1 = squared_hellinger_from_psi(
            Gamma._log_partition_from_theta,
            g1.natural_params(), g2.natural_params())
        h2_t2 = g1.squared_hellinger(g2)
        np.testing.assert_allclose(float(h2_t2), float(h2_t1), rtol=1e-12)

    def test_kl_instance_matches_tier1(self):
        g1 = Gamma(alpha=2.0, beta=1.0)
        g2 = Gamma(alpha=5.0, beta=3.0)
        kl_t1 = kl_divergence_from_psi(
            Gamma._log_partition_from_theta,
            Gamma._grad_log_partition,
            g1.natural_params(), g2.natural_params())
        kl_t2 = g1.kl_divergence(g2)
        np.testing.assert_allclose(float(kl_t2), float(kl_t1), rtol=1e-12)

    def test_gig_instance(self):
        g1 = GIG(p=2.0, a=3.0, b=1.0)
        g2 = GIG(p=-1.0, a=2.0, b=4.0)
        h2 = float(g1.squared_hellinger(g2))
        assert 0 < h2 < 1

    def test_inverse_gaussian(self):
        ig1 = InverseGaussian(mu=1.0, lam=2.0)
        ig2 = InverseGaussian(mu=2.0, lam=3.0)
        h2 = float(ig1.squared_hellinger(ig2))
        assert 0 < h2 < 1

    def test_inverse_gamma(self):
        ig1 = InverseGamma(alpha=3.0, beta=2.0)
        ig2 = InverseGamma(alpha=5.0, beta=1.0)
        h2 = float(ig1.squared_hellinger(ig2))
        assert 0 < h2 < 1


# ===========================================================================
# Tier 3: module convenience
# ===========================================================================

class TestTier3:

    def test_convenience_gamma(self):
        g1 = Gamma(alpha=2.0, beta=1.0)
        g2 = Gamma(alpha=5.0, beta=3.0)
        h2 = float(squared_hellinger(g1, g2))
        assert 0 < h2 < 1

    def test_convenience_kl(self):
        g1 = Gamma(alpha=2.0, beta=1.0)
        g2 = Gamma(alpha=5.0, beta=3.0)
        kl = float(kl_divergence(g1, g2))
        assert kl > 0

    def test_convenience_normal_mixture(self):
        d = 2
        mu = jnp.zeros(d)
        gamma = jnp.array([0.1, -0.1])
        L = jnp.eye(d)
        vg1 = VarianceGamma(
            JointVarianceGamma(mu=mu, gamma=gamma, L_Sigma=L,
                               alpha=3.0, beta=2.0))
        vg2 = VarianceGamma(
            JointVarianceGamma(mu=mu, gamma=gamma, L_Sigma=L,
                               alpha=4.0, beta=3.0))
        h2 = float(squared_hellinger(vg1, vg2))
        assert 0 < h2 < 1


# ===========================================================================
# JAX transformations
# ===========================================================================

class TestJit:

    def test_tier1_jit(self):
        t1 = Gamma(alpha=2.0, beta=1.0).natural_params()
        t2 = Gamma(alpha=5.0, beta=3.0).natural_params()

        @jax.jit
        def h(tp, tq):
            return squared_hellinger_from_psi(
                Gamma._log_partition_from_theta, tp, tq)

        result = float(h(t1, t2))
        assert 0 < result < 1

    def test_tier2_jit(self):
        import equinox as eqx
        g1 = Gamma(alpha=2.0, beta=1.0)
        g2 = Gamma(alpha=5.0, beta=3.0)
        h2 = float(eqx.filter_jit(g1.squared_hellinger)(g2))
        assert 0 < h2 < 1

    def test_tier1_kl_jit(self):
        t1 = Gamma(alpha=2.0, beta=1.0).natural_params()
        t2 = Gamma(alpha=5.0, beta=3.0).natural_params()

        @jax.jit
        def kl(tp, tq):
            return kl_divergence_from_psi(
                Gamma._log_partition_from_theta,
                Gamma._grad_log_partition, tp, tq)

        result = float(kl(t1, t2))
        assert result > 0


class TestVmap:

    def test_tier1_vmap_over_targets(self):
        """vmap Hellinger from one source to a batch of targets."""
        t_src = Gamma(alpha=3.0, beta=2.0).natural_params()
        alphas = jnp.array([1.0, 2.0, 3.0, 5.0, 10.0])
        t_batch = jax.vmap(lambda a: Gamma(a, 2.0).natural_params())(alphas)

        h2_batch = jax.vmap(lambda tq: squared_hellinger_from_psi(
            Gamma._log_partition_from_theta, t_src, tq))(t_batch)

        assert h2_batch.shape == (5,)
        np.testing.assert_allclose(float(h2_batch[2]), 0.0, atol=1e-12)
        assert all(float(h) >= 0 for h in h2_batch)

    def test_tier1_vmap_kl(self):
        t_src = Gamma(alpha=3.0, beta=2.0).natural_params()
        alphas = jnp.array([1.0, 2.0, 5.0])
        t_batch = jax.vmap(lambda a: Gamma(a, 2.0).natural_params())(alphas)

        kl_batch = jax.vmap(lambda tq: kl_divergence_from_psi(
            Gamma._log_partition_from_theta,
            Gamma._grad_log_partition,
            t_src, tq))(t_batch)

        assert kl_batch.shape == (3,)
        assert all(float(k) >= -1e-12 for k in kl_batch)

    def test_gig_vmap(self):
        t_src = GIG(p=1.0, a=2.0, b=3.0).natural_params()
        ps = jnp.array([-2.0, -1.0, 0.5, 1.0, 3.0])
        t_batch = jax.vmap(lambda p: GIG(p, 2.0, 3.0).natural_params())(ps)

        h2_batch = jax.vmap(lambda tq: squared_hellinger_from_psi(
            GIG._log_partition_from_theta, t_src, tq))(t_batch)

        assert h2_batch.shape == (5,)
        np.testing.assert_allclose(float(h2_batch[3]), 0.0, atol=1e-10)


class TestGrad:

    def test_grad_hellinger_wrt_alpha(self):
        """Gradient of Hellinger w.r.t. Gamma shape parameter."""
        target = Gamma(alpha=5.0, beta=2.0)

        def loss(alpha):
            p = Gamma(alpha, 2.0)
            return p.squared_hellinger(target)

        grad_fn = jax.grad(loss)
        g = float(grad_fn(jnp.float64(5.0)))
        np.testing.assert_allclose(g, 0.0, atol=1e-8)

        g_away = float(grad_fn(jnp.float64(3.0)))
        assert g_away != 0.0

    def test_grad_kl_wrt_alpha(self):
        target = Gamma(alpha=5.0, beta=2.0)

        def loss(alpha):
            p = Gamma(alpha, 2.0)
            return p.kl_divergence(target)

        grad_fn = jax.grad(loss)
        g = float(grad_fn(jnp.float64(5.0)))
        np.testing.assert_allclose(g, 0.0, atol=1e-8)

    def test_grad_hellinger_theta_space(self):
        """Gradient of Tier 1 Hellinger in theta-space."""
        tq = Gamma(alpha=5.0, beta=2.0).natural_params()
        grad_fn = jax.grad(lambda tp: squared_hellinger_from_psi(
            Gamma._log_partition_from_theta, tp, tq))

        g_at_same = grad_fn(tq)
        np.testing.assert_allclose(g_at_same, jnp.zeros(2), atol=1e-8)

    def test_grad_gig_hellinger(self):
        """Gradient flows through GIG Bessel functions."""
        target = GIG(p=2.0, a=3.0, b=1.0)

        def loss(p_val):
            g = GIG(p_val, 3.0, 1.0)
            return g.squared_hellinger(target)

        grad_fn = jax.grad(loss)
        g = float(grad_fn(jnp.float64(2.0)))
        np.testing.assert_allclose(g, 0.0, atol=1e-6)


# ===========================================================================
# Combined jit + vmap + grad
# ===========================================================================

class TestCombinedTransforms:

    def test_jit_vmap_hellinger(self):
        t_src = Gamma(alpha=3.0, beta=2.0).natural_params()
        alphas = jnp.array([1.0, 2.0, 5.0])
        t_batch = jax.vmap(lambda a: Gamma(a, 2.0).natural_params())(alphas)

        @jax.jit
        def batch_h2(t_batch):
            return jax.vmap(lambda tq: squared_hellinger_from_psi(
                Gamma._log_partition_from_theta, t_src, tq))(t_batch)

        result = batch_h2(t_batch)
        assert result.shape == (3,)

    def test_jit_grad_hellinger(self):
        target = Gamma(alpha=5.0, beta=2.0)

        @jax.jit
        @jax.grad
        def grad_loss(alpha):
            p = Gamma(alpha, 2.0)
            return p.squared_hellinger(target)

        g = float(grad_loss(jnp.float64(3.0)))
        assert g != 0.0
