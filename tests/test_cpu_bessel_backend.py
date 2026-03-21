"""
Tests for the CPU backend implementation (design doc: docs/design/cpu_bessel_design.md).

Covers:
  Phase 1: log_kv(v, z, backend='cpu') accuracy vs JAX path
  Phase 2: GIG.expectation_params(backend='cpu'), expectation_params_batch
           GIG.from_expectation(backend='cpu', method='lbfgs')
  Phase 3: NormalMixture.e_step(X, backend='cpu')
  Phase 4: BatchEMFitter(e_step_backend='cpu', m_step_backend='cpu', m_step_method='lbfgs')
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.utils.bessel import log_kv
from normix.distributions.generalized_inverse_gaussian import GIG
from normix.distributions.generalized_hyperbolic import (
    JointGeneralizedHyperbolic, GeneralizedHyperbolic,
)
from normix.distributions.variance_gamma import JointVarianceGamma, VarianceGamma
from normix.distributions.normal_inverse_gamma import (
    JointNormalInverseGamma, NormalInverseGamma,
)
from normix.distributions.normal_inverse_gaussian import (
    JointNormalInverseGaussian, NormalInverseGaussian,
)
from normix.fitting.em import BatchEMFitter


# ---------------------------------------------------------------------------
# Phase 1: log_kv backend='cpu'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v,z", [
    (0.5, 1.0),
    (1.0, 2.0),
    (1.5, 0.5),
    (2.0, 5.0),
    (5.0, 3.0),
    (10.0, 20.0),
    (0.5, 30.0),
    (1.0, 50.0),
    (30.0, 10.0),
    (50.0, 20.0),
    (0.0, 1.0),
    (1.0, 1e-8),
])
def test_log_kv_cpu_vs_jax(v, z):
    """CPU and JAX paths agree to ~12 digits for typical EM parameter ranges."""
    result_jax = float(log_kv(jnp.array(v), jnp.array(z), backend='jax'))
    result_cpu = float(log_kv(v, z, backend='cpu'))
    abs_err = abs(result_jax - result_cpu)
    rel_err = abs_err / (abs(result_jax) + 1e-15)
    assert rel_err < 1e-9 or abs_err < 1e-9, (
        f"log_kv({v},{z}): jax={result_jax}, cpu={result_cpu}, "
        f"rel_err={rel_err:.2e}"
    )


def test_log_kv_cpu_vectorized():
    """CPU backend handles array inputs (broadcasting)."""
    vs = np.array([0.5, 1.0, 1.5, 2.0])
    zs = np.array([1.0, 2.0, 3.0, 4.0])
    result_cpu = log_kv(vs, zs, backend='cpu')
    assert result_cpu.shape == (4,)
    for i, (v, z) in enumerate(zip(vs, zs)):
        expected = float(log_kv(v, z, backend='jax'))
        assert abs(float(result_cpu[i]) - expected) < 1e-9, (
            f"CPU vectorized log_kv({v},{z}): got {result_cpu[i]}, expected {expected}"
        )


def test_log_kv_cpu_broadcast():
    """CPU backend broadcasts v scalar over z array."""
    v = np.float64(1.0)
    zs = np.array([0.5, 1.0, 2.0, 5.0])
    result_cpu = log_kv(v, zs, backend='cpu')
    assert result_cpu.shape == (4,)
    assert np.all(np.isfinite(result_cpu))


def test_log_kv_default_backend_unchanged():
    """Calling log_kv without backend still uses JAX path (backward compat)."""
    v, z = jnp.array(1.0), jnp.array(2.0)
    result_default = float(log_kv(v, z))
    result_jax = float(log_kv(v, z, backend='jax'))
    assert result_default == result_jax


def test_log_kv_jax_path_still_jit_able():
    """The JAX path (default) remains JIT-able after the refactor."""
    jitted = jax.jit(lambda v, z: log_kv(v, z))
    result = float(jitted(jnp.array(1.0), jnp.array(2.0)))
    assert np.isfinite(result)


def test_log_kv_jax_path_still_differentiable():
    """The JAX path (default) still has custom JVP — gradients work."""
    grad_z = jax.grad(lambda z: log_kv(jnp.array(1.0), z))(jnp.array(2.0))
    assert jnp.isfinite(grad_z)
    grad_v = jax.grad(lambda v: log_kv(v, jnp.array(2.0)))(jnp.array(1.0))
    assert jnp.isfinite(grad_v)


def test_log_kv_small_z_cpu():
    """CPU backend handles z near zero (inf_mask fix)."""
    for v in [0.5, 1.0, 2.0]:
        for z in [1e-10, 1e-20, 1e-50]:
            result = float(log_kv(v, z, backend='cpu'))
            assert np.isfinite(result), f"log_kv({v},{z},cpu) not finite: {result}"
            assert result > 0, f"log_kv({v},{z},cpu) should be large positive"


# ---------------------------------------------------------------------------
# Phase 2: GIG.expectation_params(backend='cpu')
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,a,b", [
    (1.0, 2.0, 3.0),
    (0.5, 1.0, 1.0),
    (-0.5, 2.0, 4.0),
    (2.0, 0.5, 0.5),
    (-1.0, 3.0, 1.0),
    (0.1, 5.0, 0.1),
])
def test_gig_expectation_params_cpu_vs_jax(p, a, b):
    """GIG.expectation_params(backend='cpu') agrees with JAX path."""
    gig = GIG(p=p, a=a, b=b)
    eta_jax = np.array(gig.expectation_params(backend='jax'))
    eta_cpu = np.array(gig.expectation_params(backend='cpu'))
    np.testing.assert_allclose(eta_cpu, eta_jax, rtol=1e-6, atol=1e-8,
                                err_msg=f"GIG({p},{a},{b}) expectation_params mismatch")


def test_gig_expectation_params_default_unchanged():
    """Default (no backend arg) still uses JAX path."""
    gig = GIG(p=1.0, a=2.0, b=3.0)
    eta_default = np.array(gig.expectation_params())
    eta_jax = np.array(gig.expectation_params(backend='jax'))
    np.testing.assert_array_equal(eta_default, eta_jax)


def test_gig_expectation_params_batch_cpu():
    """GIG.expectation_params_batch(backend='cpu') agrees with JAX path."""
    p = jnp.array([1.0, 0.5, -0.5, 2.0])
    a = jnp.array([2.0, 1.0, 2.0,  0.5])
    b = jnp.array([3.0, 1.0, 4.0,  0.5])

    eta_jax = np.array(GIG.expectation_params_batch(p, a, b, backend='jax'))
    eta_cpu = np.array(GIG.expectation_params_batch(p, a, b, backend='cpu'))

    assert eta_jax.shape == (4, 3)
    assert eta_cpu.shape == (4, 3)
    np.testing.assert_allclose(eta_cpu, eta_jax, rtol=1e-6, atol=1e-8,
                                err_msg="expectation_params_batch cpu vs jax mismatch")


def test_gig_expectation_params_batch_default_is_jax():
    """Default backend for batch is JAX (backward compat)."""
    p = jnp.array([1.0, 0.5])
    a = jnp.array([2.0, 1.0])
    b = jnp.array([3.0, 1.0])
    eta_default = np.array(GIG.expectation_params_batch(p, a, b))
    eta_jax = np.array(GIG.expectation_params_batch(p, a, b, backend='jax'))
    np.testing.assert_array_equal(eta_default, eta_jax)


@pytest.mark.parametrize("p,a,b", [
    (1.0, 2.0, 3.0),
    (0.5, 1.0, 1.0),
    (-0.5, 2.0, 4.0),
    (-1.0, 3.0, 1.0),
])
def test_gig_from_expectation_solver_cpu(p, a, b):
    """GIG.from_expectation(backend='cpu') recovers correct η."""
    gig_true = GIG(p=p, a=a, b=b)
    eta = np.array(gig_true.expectation_params(backend='cpu'))

    theta0 = gig_true.natural_params()
    gig_recovered = GIG.from_expectation(
        jnp.array(eta), theta0=theta0, backend='cpu', method='lbfgs', tol=1e-8,
    )

    eta_recovered = np.array(gig_recovered.expectation_params(backend='cpu'))
    np.testing.assert_allclose(eta_recovered, eta, rtol=1e-5, atol=1e-7,
                                err_msg=f"GIG({p},{a},{b}) cpu solver roundtrip failed")


# ---------------------------------------------------------------------------
# Phase 3: NormalMixture.e_step(backend='cpu')
# ---------------------------------------------------------------------------

def _make_gh_model(d=2, seed=42):
    """Create a simple GH model for testing."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d)
    gamma = 0.1 * rng.standard_normal(d)
    sigma = np.eye(d) + 0.1 * rng.standard_normal((d, d))
    sigma = sigma @ sigma.T + 0.01 * np.eye(d)
    return GeneralizedHyperbolic.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, p=1.0, a=2.0, b=3.0
    )


@pytest.mark.parametrize("d", [1, 2, 4])
def test_e_step_cpu_vs_jax_gh(d):
    """GH e_step(backend='cpu') agrees with JAX path."""
    model = _make_gh_model(d=d)
    rng = np.random.default_rng(0)
    X = jnp.array(rng.standard_normal((20, d)))

    exp_jax = model.e_step(X, backend='jax')
    exp_cpu = model.e_step(X, backend='cpu')

    for key in ['E_log_Y', 'E_inv_Y', 'E_Y']:
        np.testing.assert_allclose(
            np.array(exp_cpu[key]), np.array(exp_jax[key]),
            rtol=1e-5, atol=1e-7,
            err_msg=f"e_step cpu vs jax mismatch for {key} (d={d})"
        )


def test_e_step_default_backend_unchanged():
    """Default e_step (no backend arg) still uses JAX path."""
    model = _make_gh_model(d=2)
    X = jnp.array(np.random.default_rng(1).standard_normal((10, 2)))
    exp_default = model.e_step(X)
    exp_jax = model.e_step(X, backend='jax')
    for key in ['E_log_Y', 'E_inv_Y', 'E_Y']:
        np.testing.assert_array_equal(
            np.array(exp_default[key]), np.array(exp_jax[key])
        )


def _make_vg_model(d=2, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d)
    gamma = 0.1 * rng.standard_normal(d)
    sigma = np.eye(d) + 0.1 * rng.standard_normal((d, d))
    sigma = sigma @ sigma.T + 0.01 * np.eye(d)
    return VarianceGamma.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0
    )


def _make_nig_model(d=2, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d)
    gamma = 0.1 * rng.standard_normal(d)
    sigma = np.eye(d) + 0.1 * rng.standard_normal((d, d))
    sigma = sigma @ sigma.T + 0.01 * np.eye(d)
    return NormalInverseGamma.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0
    )


def _make_niig_model(d=2, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d)
    gamma = 0.1 * rng.standard_normal(d)
    sigma = np.eye(d) + 0.1 * rng.standard_normal((d, d))
    sigma = sigma @ sigma.T + 0.01 * np.eye(d)
    return NormalInverseGaussian.from_classical(
        mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=2.0
    )


@pytest.mark.parametrize("model_fn,name", [
    (_make_vg_model, "VG"),
    (_make_nig_model, "NIG"),
    (_make_niig_model, "NIIG"),
])
def test_e_step_cpu_vs_jax_all_distributions(model_fn, name):
    """CPU e_step agrees with JAX path for all joint distributions."""
    model = model_fn(d=2)
    X = jnp.array(np.random.default_rng(7).standard_normal((15, 2)))

    exp_jax = model.e_step(X, backend='jax')
    exp_cpu = model.e_step(X, backend='cpu')

    for key in ['E_log_Y', 'E_inv_Y', 'E_Y']:
        np.testing.assert_allclose(
            np.array(exp_cpu[key]), np.array(exp_jax[key]),
            rtol=1e-5, atol=1e-7,
            err_msg=f"{name} e_step cpu vs jax mismatch for {key}"
        )


# ---------------------------------------------------------------------------
# Phase 4: BatchEMFitter with CPU backend
# ---------------------------------------------------------------------------

def test_batch_em_fitter_cpu_backend_fields():
    """BatchEMFitter accepts e_step_backend and m_step_backend/method fields."""
    fitter = BatchEMFitter(
        max_iter=5, tol=1e-4,
        e_step_backend='cpu', m_step_backend='cpu', m_step_method='lbfgs',
    )
    assert fitter.e_step_backend == 'cpu'
    assert fitter.m_step_backend == 'cpu'
    assert fitter.m_step_method == 'lbfgs'


def test_batch_em_fitter_defaults_unchanged():
    """BatchEMFitter default fields."""
    fitter = BatchEMFitter()
    assert fitter.e_step_backend == 'jax'
    assert fitter.m_step_backend == 'cpu'
    assert fitter.m_step_method == 'newton'
    assert fitter.max_iter == 200
    assert fitter.tol == 1e-6


def test_batch_em_fitter_cpu_converges_gh():
    """BatchEMFitter with CPU backend converges on a small GH problem."""
    rng = np.random.default_rng(42)
    d = 2
    n = 50
    true_model = GeneralizedHyperbolic.from_classical(
        mu=np.zeros(d), gamma=np.zeros(d),
        sigma=np.eye(d), p=1.0, a=1.0, b=1.0,
    )
    X = jnp.array(true_model.rvs(n, seed=42))

    fitter = BatchEMFitter(
        max_iter=10, tol=1e-3,
        e_step_backend='cpu', m_step_backend='cpu', m_step_method='lbfgs',
    )
    init_model = GeneralizedHyperbolic.from_classical(
        mu=np.zeros(d), gamma=np.zeros(d),
        sigma=np.eye(d), p=1.0, a=1.0, b=1.0,
    )
    fitted = fitter.fit(init_model, X)
    ll = float(fitted.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"Log-likelihood not finite: {ll}"


def test_batch_em_fitter_cpu_same_result_as_default():
    """CPU backend gives same log-likelihood as JAX backend after convergence."""
    rng = np.random.default_rng(0)
    d = 2
    n = 40
    true_model = GeneralizedHyperbolic.from_classical(
        mu=np.zeros(d), gamma=np.zeros(d),
        sigma=np.eye(d), p=1.0, a=1.0, b=1.0,
    )
    X = jnp.array(true_model.rvs(n, seed=123))

    init_model = GeneralizedHyperbolic.from_classical(
        mu=np.zeros(d), gamma=np.zeros(d),
        sigma=np.eye(d), p=1.0, a=1.0, b=1.0,
    )

    fitter_jax = BatchEMFitter(max_iter=5, tol=1e-4,
                                e_step_backend='jax', m_step_backend='jax', m_step_method='newton')
    fitter_cpu = BatchEMFitter(max_iter=5, tol=1e-4,
                                e_step_backend='cpu', m_step_backend='cpu', m_step_method='lbfgs')

    fitted_jax = fitter_jax.fit(init_model, X)
    fitted_cpu = fitter_cpu.fit(init_model, X)

    ll_jax = float(fitted_jax.marginal_log_likelihood(X))
    ll_cpu = float(fitted_cpu.marginal_log_likelihood(X))

    # Both should converge to approximately the same log-likelihood
    assert abs(ll_cpu - ll_jax) < 0.05, (
        f"CPU vs JAX EM log-likelihood too different: {ll_cpu:.4f} vs {ll_jax:.4f}"
    )


# ---------------------------------------------------------------------------
# Posterior GIG params
# ---------------------------------------------------------------------------

def test_posterior_gig_params_gh():
    """JointGeneralizedHyperbolic._posterior_gig_params is consistent with
    _conditional_expectations_impl."""
    d = 2
    j = JointGeneralizedHyperbolic(
        mu=jnp.zeros(d), gamma=jnp.zeros(d),
        L_Sigma=jnp.eye(d), p=jnp.array(1.0),
        a=jnp.array(2.0), b=jnp.array(3.0),
    )
    x = jnp.array([1.0, 0.5])
    z, w, z2, w2, zw = j._quad_forms(x)

    p_post, a_post, b_post = j._posterior_gig_params(z2, w2)
    assert float(p_post) == pytest.approx(1.0 - d / 2.0)
    assert float(a_post) == pytest.approx(2.0 + float(w2))
    assert float(b_post) == pytest.approx(3.0 + float(z2))


def test_posterior_gig_params_vg():
    d = 2
    j = JointVarianceGamma(
        mu=jnp.zeros(d), gamma=jnp.zeros(d),
        L_Sigma=jnp.eye(d),
        alpha=jnp.array(2.0), beta=jnp.array(1.0),
    )
    x = jnp.array([1.0, 0.5])
    z, w, z2, w2, zw = j._quad_forms(x)
    p_post, a_post, b_post = j._posterior_gig_params(z2, w2)
    assert float(p_post) == pytest.approx(2.0 - d / 2.0)
    assert float(a_post) == pytest.approx(2.0 * 1.0 + float(w2))
    assert float(b_post) == pytest.approx(float(z2))


def test_posterior_gig_params_nig():
    d = 2
    j = JointNormalInverseGamma(
        mu=jnp.zeros(d), gamma=jnp.zeros(d),
        L_Sigma=jnp.eye(d),
        alpha=jnp.array(2.0), beta=jnp.array(1.0),
    )
    x = jnp.array([1.0, 0.5])
    z, w, z2, w2, zw = j._quad_forms(x)
    p_post, a_post, b_post = j._posterior_gig_params(z2, w2)
    assert float(p_post) == pytest.approx(-2.0 - d / 2.0)
    assert float(a_post) == pytest.approx(float(w2))
    assert float(b_post) == pytest.approx(2.0 * 1.0 + float(z2))


def test_posterior_gig_params_niig():
    d = 2
    j = JointNormalInverseGaussian(
        mu=jnp.zeros(d), gamma=jnp.zeros(d),
        L_Sigma=jnp.eye(d),
        mu_ig=jnp.array(1.0), lam=jnp.array(2.0),
    )
    x = jnp.array([1.0, 0.5])
    z, w, z2, w2, zw = j._quad_forms(x)
    p_post, a_post, b_post = j._posterior_gig_params(z2, w2)
    a_ig = float(j.lam) / float(j.mu_ig) ** 2
    assert float(p_post) == pytest.approx(-0.5 - d / 2.0)
    assert float(a_post) == pytest.approx(a_ig + float(w2))
    assert float(b_post) == pytest.approx(float(j.lam) + float(z2))
