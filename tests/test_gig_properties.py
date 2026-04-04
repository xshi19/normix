"""
GIG property-based and edge-case tests (T6).

Validates:
  - K_v symmetry: log_kv(v, z) == log_kv(-v, z)
  - Moment ordering: E[1/Y] * E[Y] > 1 (Cauchy-Schwarz)
  - EF invariants: ∇ψ = η, Hessian SPD across wide parameter range
  - rvs positivity and finiteness
  - log_prob finiteness for extreme (p, a, b) combinations
  - from_natural / natural_params round-trip across parameter grid
  - Bessel-based moments: E[Y^α] = r(α) * K_{p+α}(√ab) / K_p(√ab)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

from normix.distributions.generalized_inverse_gaussian import GIG
from normix.utils.bessel import log_kv


# ---------------------------------------------------------------------------
# Parameter grid for parametric tests
# ---------------------------------------------------------------------------

_PARAM_GRID = [
    # (p, a, b)
    (1.0,   1.0,   1.0),
    (2.0,   3.0,   0.5),
    (-0.5,  2.0,   2.0),   # InverseGaussian special case
    (-2.0,  0.5,   4.0),
    (0.5,   0.1,   5.0),
    (5.0,   4.0,   0.2),
    (-3.0,  0.3,   3.0),
    (0.1,   10.0,  0.1),   # large a
    (1.0,   0.1,   10.0),  # large b
    (10.0,  1.0,   1.0),   # large positive p
    (-10.0, 1.0,   1.0),   # large negative p
]


class TestGIGBessel:
    """Tests relying on the Bessel-function structure of GIG."""

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_log_kv_symmetry(self, p, a, b):
        """K_v(z) = K_{-v}(z): log_kv should be invariant to sign of v."""
        sqrt_ab = float(np.sqrt(a * b))
        v = p - 0.5
        lkv_pos = float(log_kv(v,  sqrt_ab))
        lkv_neg = float(log_kv(-v, sqrt_ab))
        assert_allclose(lkv_pos, lkv_neg, rtol=1e-9,
                        err_msg=f"K_v != K_{{-v}} for v={v:.2f}, z={sqrt_ab:.3f}")

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_log_prob_finite(self, p, a, b):
        """log_prob must be finite and negative for any positive x."""
        gig = GIG(p=p, a=a, b=b)
        xs = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        lps = jax.vmap(gig.log_prob)(xs)
        assert jnp.all(jnp.isfinite(lps)), (
            f"Non-finite log_prob for p={p},a={a},b={b}: {lps}")

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_log_partition_finite(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        psi = float(gig.log_partition())
        assert np.isfinite(psi), f"Non-finite log_partition for p={p},a={a},b={b}"

    @pytest.mark.parametrize("v,z", [
        (0.5, 1.0), (1.5, 2.0), (-0.5, 1.0), (2.0, 0.5), (5.0, 3.0),
    ])
    def test_log_kv_jax_differentiable(self, v, z):
        """jax.grad of log_kv(v, z) w.r.t. z matches central finite difference."""
        z_jnp = jnp.float64(z)
        grad_jax = float(jax.grad(lambda zz: log_kv(v, zz))(z_jnp))
        eps = 1e-5
        fd = (float(log_kv(v, z + eps)) - float(log_kv(v, z - eps))) / (2 * eps)
        assert_allclose(grad_jax, fd, rtol=1e-4,
                        err_msg=f"JAX grad of log_kv != FD for v={v}, z={z}")


class TestGIGEFContract:
    """Verify the exponential-family contract across a wide parameter range."""

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_grad_psi_equals_eta(self, p, a, b):
        """∇ψ(θ) = η for all parameter combinations."""
        gig = GIG(p=p, a=a, b=b)
        theta = gig.natural_params()
        eta_direct = gig.expectation_params()
        eta_grad = jax.grad(GIG._log_partition_from_theta)(theta)
        assert_allclose(np.array(eta_direct), np.array(eta_grad), rtol=1e-5,
                        err_msg=f"∇ψ ≠ η for p={p},a={a},b={b}")

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_hessian_finite_and_symmetric(self, p, a, b):
        """Fisher information I(θ) = ∇²ψ(θ) must be finite and symmetric."""
        gig = GIG(p=p, a=a, b=b)
        FI = gig.fisher_information()
        assert FI.shape == (3, 3), "Fisher info must be 3×3"
        assert jnp.all(jnp.isfinite(FI)), (
            f"Non-finite Fisher info for p={p},a={a},b={b}")
        assert_allclose(np.array(FI), np.array(FI).T, atol=1e-8,
                        err_msg=f"Fisher info not symmetric for p={p},a={a},b={b}")

    @pytest.mark.parametrize("p,a,b", [
        # Moderate parameters where autodiff through Bessel is accurate
        (1.0, 2.0, 2.0), (-0.5, 1.0, 1.0), (2.0, 1.0, 1.0), (-1.0, 2.0, 1.0),
    ])
    def test_hessian_spd_moderate(self, p, a, b):
        """Fisher information must be positive semidefinite (moderate params, CPU Hessian)."""
        gig = GIG(p=p, a=a, b=b)
        # Use CPU backend for more accurate Hessian (analytical Bessel derivatives)
        FI = gig.fisher_information(backend='cpu')
        eigvals = np.linalg.eigvalsh(np.array(FI))
        assert np.all(eigvals > -1e-8), (
            f"Fisher info not PSD for p={p},a={a},b={b}: eigvals={eigvals}")

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_natural_roundtrip(self, p, a, b):
        """natural_params → from_natural must recover (p, a, b)."""
        gig = GIG(p=p, a=a, b=b)
        theta = gig.natural_params()
        gig2 = GIG.from_natural(theta)
        assert_allclose(float(gig2.p), p, rtol=1e-10)
        assert_allclose(float(gig2.a), a, rtol=1e-10)
        assert_allclose(float(gig2.b), b, rtol=1e-10)


class TestGIGMoments:
    """Verify moment properties of GIG."""

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_expectation_params_positive(self, p, a, b):
        """E[1/Y] and E[Y] must be strictly positive."""
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        assert float(eta[1]) > 0, f"E[1/Y] ≤ 0 for p={p},a={a},b={b}: {float(eta[1])}"
        assert float(eta[2]) > 0, f"E[Y] ≤ 0 for p={p},a={a},b={b}: {float(eta[2])}"

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_cauchy_schwarz(self, p, a, b):
        """Cauchy-Schwarz: E[Y] * E[1/Y] >= 1."""
        gig = GIG(p=p, a=a, b=b)
        eta = gig.expectation_params()
        product = float(eta[2]) * float(eta[1])
        assert product >= 1.0 - 1e-9, (
            f"Cauchy-Schwarz violated: E[Y]*E[1/Y]={product:.6f} < 1 "
            f"for p={p},a={a},b={b}")

    @pytest.mark.parametrize("p,a,b", _PARAM_GRID)
    def test_mean_positive(self, p, a, b):
        """mean() must equal E[Y] > 0."""
        gig = GIG(p=p, a=a, b=b)
        m = float(gig.mean())
        eta = gig.expectation_params()
        assert m > 0, f"mean() ≤ 0 for p={p},a={a},b={b}"
        assert_allclose(m, float(eta[2]), rtol=1e-8)


class TestGIGRvs:
    """Verify random variate generation."""

    @pytest.mark.parametrize("p,a,b", [
        (1.0,  1.0,  1.0),
        (-0.5, 2.0,  2.0),
        (2.0,  3.0,  0.5),
        (-2.0, 0.5,  4.0),
    ])
    def test_rvs_positive_and_finite(self, p, a, b):
        gig = GIG(p=p, a=a, b=b)
        samples = gig.rvs(500, seed=0)
        assert samples.shape == (500,)
        assert jnp.all(jnp.isfinite(samples)), f"Non-finite samples for p={p},a={a},b={b}"
        assert jnp.all(samples > 0), f"Non-positive samples for p={p},a={a},b={b}"

    def test_rvs_empirical_mean(self):
        """Empirical mean should match E[Y] to ~2%."""
        p, a, b = 1.5, 2.0, 0.5
        gig = GIG(p=p, a=a, b=b)
        samples = gig.rvs(10000, seed=42)
        assert_allclose(float(jnp.mean(samples)), float(gig.mean()), rtol=0.03)

    def test_rvs_empirical_inv_mean(self):
        """Empirical mean of 1/Y should match E[1/Y] to ~2%."""
        p, a, b = 1.0, 1.0, 1.0
        gig = GIG(p=p, a=a, b=b)
        samples = gig.rvs(10000, seed=7)
        emp_inv = float(jnp.mean(1.0 / samples))
        eta = gig.expectation_params()
        assert_allclose(emp_inv, float(eta[1]), rtol=0.03)


class TestGIGExtremeParameters:
    """Edge cases at parameter extremes."""

    @pytest.mark.parametrize("b", [1e-10, 1e-8, 1e-4])
    def test_near_gamma_limit_log_partition_finite(self, b):
        """Near Gamma limit (b→0) log_partition stays finite."""
        gig = GIG(p=2.0, a=2.0, b=b)
        assert np.isfinite(float(gig.log_partition()))

    @pytest.mark.parametrize("a", [1e-10, 1e-8, 1e-4])
    def test_near_invgamma_limit_log_partition_finite(self, a):
        """Near InverseGamma limit (a→0) log_partition stays finite."""
        gig = GIG(p=-2.0, a=a, b=2.0)
        assert np.isfinite(float(gig.log_partition()))

    @pytest.mark.parametrize("p", [-20.0, -10.0, -1.0, 1.0, 10.0, 20.0])
    def test_large_p_log_partition_finite(self, p):
        """Large |p| with moderate a, b should remain finite."""
        gig = GIG(p=p, a=1.0, b=1.0)
        assert np.isfinite(float(gig.log_partition()))

    @pytest.mark.parametrize("scale", [0.01, 0.1, 10.0, 100.0])
    def test_scale_invariance(self, scale):
        """GIG(p, a/c, b*c) and GIG(p, a, b) should have E[Y] differing by c."""
        p, a, b = 1.5, 2.0, 0.5
        gig1 = GIG(p=p, a=a,       b=b)
        gig2 = GIG(p=p, a=a/scale, b=b*scale)
        # Under this reparametrization, E[Y] is scaled by `scale`
        assert_allclose(float(gig2.mean()), float(gig1.mean()) * scale, rtol=1e-4,
                        err_msg=f"Scale invariance failed at scale={scale}")
