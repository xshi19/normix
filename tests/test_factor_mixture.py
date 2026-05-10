"""
Tests for the factor-analysis mixture family.

Covers the four guarantees required by Phase 4 of the EM-extension plan
(``docs/archive/design/em_covariance_extensions.md`` §9):

1. Density agreement — :class:`FactorXxx` and :class:`Xxx` produce the
   same ``log_prob`` when constructed with matched
   :math:`\\Sigma = F F^\\top + \\mathrm{diag}(D)`. Validates the
   Woodbury solve and log-determinant.
2. Recovery on synthetic data — fitting recovers
   :math:`\\Sigma = F F^\\top + \\mathrm{diag}(D)` (modulo the rotation
   gauge of :math:`F`).
3. Match against full covariance — with ``r = d - 1`` and large ``n``,
   the FA marginal log-likelihood is within tolerance of the
   full-covariance fit's marginal log-likelihood.
4. Convergence on :math:`\\Sigma` — :meth:`em_convergence_params`
   returns :math:`\\Sigma`, which is invariant to rotations of
   :math:`F`.
5. Shrinkage combinator on factor stats — ``Shrinkage(rule, eta0,
   tau=0)`` agrees with the base rule when applied to a
   :class:`FactorMixtureStats` pytree.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import (
    BatchEMFitter,
    FactorMixtureStats,
    FactorGeneralizedHyperbolic,
    FactorNormalInverseGamma,
    FactorNormalInverseGaussian,
    FactorVarianceGamma,
    GeneralizedHyperbolic,
    IdentityUpdate,
    NormalInverseGamma,
    NormalInverseGaussian,
    Shrinkage,
    VarianceGamma,
)


KEY = jax.random.PRNGKey(7)


# ============================================================================
# Helpers
# ============================================================================


def _matched_factor_full(factor_cls, full_cls, *, mu, gamma, F, D, **sub):
    """Build a (factor model, full-cov model) pair sharing Σ = FFᵀ+diag(D)."""
    Sigma = F @ F.T + jnp.diag(D)
    factor_model = factor_cls.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, **sub)
    full_model = full_cls.from_classical(
        mu=mu, gamma=gamma, sigma=Sigma, **sub)
    return factor_model, full_model


@pytest.fixture(scope='module')
def small_factor_setup():
    """Fixed (mu, gamma, F, D) for d=4, r=2."""
    F = jnp.array(
        [[1.0, 0.0],
         [0.7, 0.3],
         [-0.2, 0.5],
         [0.0, 0.8]],
    )
    D = jnp.array([0.4, 0.5, 0.3, 0.6])
    mu = jnp.array([0.1, -0.2, 0.3, 0.0])
    gamma = jnp.array([0.2, 0.0, -0.1, 0.1])
    return mu, gamma, F, D


# ============================================================================
# 1. Density / Woodbury agreement
# ============================================================================


@pytest.mark.parametrize(
    "factor_cls, full_cls, sub",
    [
        (FactorVarianceGamma, VarianceGamma, dict(alpha=2.0, beta=1.5)),
        (FactorNormalInverseGamma, NormalInverseGamma,
         dict(alpha=3.0, beta=1.0)),
        (FactorNormalInverseGaussian, NormalInverseGaussian,
         dict(mu_ig=1.0, lam=2.0)),
        (FactorGeneralizedHyperbolic, GeneralizedHyperbolic,
         dict(p=1.0, a=1.5, b=1.0)),
    ],
    ids=["VG", "NInvG", "NIG", "GH"],
)
def test_factor_log_prob_matches_full_cov(
    factor_cls, full_cls, sub, small_factor_setup,
):
    """Factor log_prob matches full-cov log_prob when Σ = FFᵀ+diag(D)."""
    mu, gamma, F, D = small_factor_setup
    fa, full = _matched_factor_full(
        factor_cls, full_cls, mu=mu, gamma=gamma, F=F, D=D, **sub)

    # Σ and log|Σ| match.
    np.testing.assert_allclose(fa.sigma(), full.sigma(), atol=1e-12)
    np.testing.assert_allclose(
        float(fa.log_det_sigma()), float(full.log_det_sigma()),
        rtol=1e-10)

    # Per-observation log-density matches.
    key = jax.random.PRNGKey(11)
    X = jax.random.normal(key, (16, mu.shape[0]), dtype=jnp.float64)
    lp_fa = jax.vmap(fa.log_prob)(X)
    lp_full = jax.vmap(full.log_prob)(X)
    np.testing.assert_allclose(np.asarray(lp_fa), np.asarray(lp_full),
                               rtol=1e-10, atol=1e-12)


def test_factor_quad_form_agrees_with_dense_solve(small_factor_setup):
    """_quad_form(x) = xᵀΣ⁻¹x where Σ is the dense Woodbury reconstruction."""
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
    Sigma = fa.sigma()
    Sigma_inv = jnp.linalg.inv(Sigma)
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (5, 4), dtype=jnp.float64)
    for x in X:
        q_fa = float(fa._quad_form(x))
        q_dense = float(x @ Sigma_inv @ x)
        np.testing.assert_allclose(q_fa, q_dense, rtol=1e-10)


def test_factor_beta_matches_FT_Sigma_inv(small_factor_setup):
    """β = M⁻¹FᵀD⁻¹ equals FᵀΣ⁻¹ from a dense solve."""
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
    beta_fast = fa._beta()
    beta_dense = F.T @ jnp.linalg.inv(fa.sigma())
    np.testing.assert_allclose(np.asarray(beta_fast),
                               np.asarray(beta_dense),
                               rtol=1e-10, atol=1e-12)


# ============================================================================
# 2. E-step / M-step round-trip
# ============================================================================


def test_e_step_returns_factor_stats(small_factor_setup):
    """e_step returns a FactorMixtureStats with the right shapes."""
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
    X = fa.rvs(64, seed=3)

    eta = fa.e_step(X, backend='jax')
    assert isinstance(eta, FactorMixtureStats)
    assert eta.E_X.shape == (mu.shape[0],)
    assert eta.E_XXT_inv_Y.shape == (mu.shape[0], mu.shape[0])
    assert eta.E_XZT_inv_sqrtY.shape == (mu.shape[0], F.shape[1])
    assert eta.E_Z_inv_sqrtY.shape == (F.shape[1],)
    assert eta.E_ZZT.shape == (F.shape[1], F.shape[1])

    # All finite.
    leaves = jax.tree.leaves(eta)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf))


def test_m_step_normal_then_subordinator_recovers_eta(small_factor_setup):
    """m_step on the model's own η returns the same model up to tiny FP drift.

    This is a fixed-point sanity check: η = compute_eta_from_model() and
    m_step(η) should give back the same parameters (the closed-form
    inversion is injective).
    """
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)

    eta = fa.compute_eta_from_model()
    fa2 = fa.m_step(eta)

    np.testing.assert_allclose(fa2.mu, fa.mu, atol=1e-8)
    np.testing.assert_allclose(fa2.gamma, fa.gamma, atol=1e-8)
    # Σ must match (F is only identifiable up to rotation).
    np.testing.assert_allclose(
        np.asarray(fa2.sigma()), np.asarray(fa.sigma()),
        atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(float(fa2.alpha), float(fa.alpha), rtol=1e-6)
    np.testing.assert_allclose(float(fa2.beta), float(fa.beta), rtol=1e-6)


# ============================================================================
# 3. Recovery on synthetic data: Σ recovered up to rotation gauge of F
# ============================================================================


def test_em_recovers_sigma_and_subordinator():
    """Generate from FactorVG with known params, fit, recover Σ and α, β."""
    F_true = jnp.array(
        [[1.0, 0.0],
         [0.8, 0.2],
         [0.5, -0.5],
         [0.0, 1.0],
         [-0.3, 0.7]],
    )
    D_true = jnp.array([0.3, 0.4, 0.2, 0.5, 0.3])
    mu_true = jnp.array([0.1, 0.0, -0.2, 0.3, 0.5])
    gamma_true = jnp.array([0.2, -0.1, 0.0, 0.1, 0.0])
    true_model = FactorVarianceGamma.from_classical(
        mu=mu_true, gamma=gamma_true, F=F_true, D=D_true,
        alpha=3.0, beta=2.0,
    )
    n = 4000
    X = true_model.rvs(n, seed=0)

    init = FactorVarianceGamma.default_init(X, r=2)
    fitter = BatchEMFitter(max_iter=120, tol=1e-4, verbose=0,
                           e_step_backend='jax', m_step_backend='jax')
    result = fitter.fit(init, X)
    fitted = result.model

    # μ recovered.
    np.testing.assert_allclose(
        np.asarray(fitted.mu), np.asarray(mu_true),
        atol=0.15, rtol=0.5)

    # Σ recovered up to FP / sample noise (Σ is rotation-invariant for F).
    Sigma_fit = np.asarray(fitted.sigma())
    Sigma_true = np.asarray(true_model.sigma())
    rel_err = np.linalg.norm(Sigma_fit - Sigma_true) / np.linalg.norm(Sigma_true)
    assert rel_err < 0.25, f"Σ relative error too large: {rel_err:.4f}"

    # Subordinator within typical EM tolerance for n=4000.
    assert abs(float(fitted.alpha) - 3.0) < 0.5
    assert abs(float(fitted.beta) - 2.0) < 0.5


# ============================================================================
# 4. Match against full-cov fit when r = d - 1
# ============================================================================


def test_factor_fit_matches_full_cov_when_r_equals_d_minus_1():
    """With r = d - 1 and large n, FA marginal LL matches full-cov LL."""
    d, r, n = 4, 3, 3000
    Sigma_true = jnp.array(
        [[1.2, 0.3, 0.0, -0.1],
         [0.3, 0.9, 0.2, 0.0],
         [0.0, 0.2, 1.0, 0.1],
         [-0.1, 0.0, 0.1, 0.8]],
    )
    mu_true = jnp.zeros(d)
    gamma_true = jnp.array([0.2, 0.0, -0.1, 0.0])
    truth = VarianceGamma.from_classical(
        mu=mu_true, gamma=gamma_true, sigma=Sigma_true,
        alpha=3.0, beta=2.0,
    )
    X = truth.rvs(n, seed=1)

    fitter = BatchEMFitter(max_iter=80, tol=1e-4, verbose=0,
                           e_step_backend='jax', m_step_backend='jax')
    full_result = fitter.fit(VarianceGamma.default_init(X), X)
    fa_result = fitter.fit(FactorVarianceGamma.default_init(X, r=r), X)

    ll_full = float(full_result.model.marginal_log_likelihood(X))
    ll_fa = float(fa_result.model.marginal_log_likelihood(X))
    # FA can match full-cov to within a small absolute tolerance because
    # rank-(d-1) plus diagonal can express any symmetric PD matrix.
    assert abs(ll_fa - ll_full) < 0.05, (
        f"FA LL={ll_fa:.4f} vs full-cov LL={ll_full:.4f}")


# ============================================================================
# 5. Convergence on Σ ignores rotational gauge of F
# ============================================================================


def test_em_convergence_params_invariant_to_F_rotation(small_factor_setup):
    """em_convergence_params returns Σ which is rotation-invariant in F."""
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)

    # Apply an orthogonal rotation Q to F: F' = F Q.  Σ = FFᵀ + D is
    # invariant.  em_convergence_params should also be invariant.
    key = jax.random.PRNGKey(0)
    Q_raw = jax.random.normal(key, (F.shape[1], F.shape[1]),
                              dtype=jnp.float64)
    Q, _ = jnp.linalg.qr(Q_raw)
    F_rot = F @ Q
    fa_rot = fa.replace(F=F_rot)

    p1 = fa.em_convergence_params()
    p2 = fa_rot.em_convergence_params()
    for a, b in zip(jax.tree.leaves(p1), jax.tree.leaves(p2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   atol=1e-10)


# ============================================================================
# 6. Shrinkage combinator on FactorMixtureStats
# ============================================================================


def test_shrinkage_tau_zero_equals_base_on_factor_stats(small_factor_setup):
    """Shrinkage(IdentityUpdate(), eta0, tau=0) is a no-op for the factor
    stats type, just like for NormalMixtureEta."""
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
    X = fa.rvs(128, seed=2)

    eta_hat = fa.e_step(X, backend='jax')
    eta_prev = fa.compute_eta_from_model()
    eta0 = fa.compute_eta_from_model()

    base = IdentityUpdate()
    shrunk = Shrinkage(base, eta0, tau=jnp.float64(0.0))

    out_base, _ = base(eta_prev, eta_hat,
                       jnp.int32(0), jnp.int32(X.shape[0]),
                       base.initial_state())
    out_shrunk, _ = shrunk(eta_prev, eta_hat,
                           jnp.int32(0), jnp.int32(X.shape[0]),
                           shrunk.initial_state())

    for lb, ls in zip(jax.tree.leaves(out_base),
                      jax.tree.leaves(out_shrunk)):
        np.testing.assert_allclose(np.asarray(lb), np.asarray(ls),
                                   rtol=1e-10, atol=1e-12)


def test_shrinkage_per_field_tau_only_sigma_on_factor_stats(small_factor_setup):
    """Per-field τ with non-zero entry only on E_XXT_inv_Y leaves the
    other 9 statistics untouched."""
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
    X = fa.rvs(128, seed=2)

    eta_hat = fa.e_step(X, backend='jax')
    eta_prev = fa.compute_eta_from_model()
    eta0 = fa.compute_eta_from_model()

    d = mu.shape[0]
    r = F.shape[1]
    tau = FactorMixtureStats(
        E_inv_Y=jnp.float64(0.0),
        E_Y=jnp.float64(0.0),
        E_log_Y=jnp.float64(0.0),
        E_X=jnp.zeros(d),
        E_X_inv_Y=jnp.zeros(d),
        E_XXT_inv_Y=jnp.full((d, d), 0.5),
        E_XZT_inv_sqrtY=jnp.zeros((d, r)),
        E_Z_inv_sqrtY=jnp.zeros(r),
        E_Z_sqrtY=jnp.zeros(r),
        E_ZZT=jnp.zeros((r, r)),
    )

    base = IdentityUpdate()
    shrunk = Shrinkage(base, eta0, tau=tau)
    out_shrunk, _ = shrunk(eta_prev, eta_hat,
                           jnp.int32(0), jnp.int32(X.shape[0]),
                           shrunk.initial_state())

    # Fields whose τ leaf is zero should equal η̂ exactly.
    for fld in ('E_inv_Y', 'E_Y', 'E_log_Y', 'E_X', 'E_X_inv_Y',
                'E_XZT_inv_sqrtY', 'E_Z_inv_sqrtY', 'E_Z_sqrtY', 'E_ZZT'):
        np.testing.assert_allclose(
            np.asarray(getattr(out_shrunk, fld)),
            np.asarray(getattr(eta_hat, fld)),
            rtol=1e-10, atol=1e-12,
            err_msg=f"field {fld} unexpectedly modified",
        )

    # E_XXT_inv_Y should be a strict convex combination of η̂ and η₀.
    out_sigma = out_shrunk.E_XXT_inv_Y
    expected = (1.0 / 1.5) * eta_hat.E_XXT_inv_Y \
        + (0.5 / 1.5) * eta0.E_XXT_inv_Y
    np.testing.assert_allclose(np.asarray(out_sigma), np.asarray(expected),
                               rtol=1e-10, atol=1e-12)


# ============================================================================
# 7. Mean / cov / rvs sanity checks
# ============================================================================


def test_factor_mean_and_cov_match_sample_estimates():
    """Sample mean/cov from rvs should be close to the analytic mean/cov."""
    F = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    D = jnp.array([0.5, 0.5, 0.5])
    fa = FactorVarianceGamma.from_classical(
        mu=jnp.array([0.0, 0.0, 0.0]),
        gamma=jnp.array([0.1, 0.0, -0.1]),
        F=F, D=D, alpha=4.0, beta=2.0,
    )
    X = fa.rvs(20000, seed=42)

    sample_mean = jnp.mean(X, axis=0)
    sample_cov = jnp.cov(X.T)

    np.testing.assert_allclose(np.asarray(sample_mean),
                               np.asarray(fa.mean()),
                               atol=0.05)
    rel = jnp.linalg.norm(sample_cov - fa.cov()) / jnp.linalg.norm(fa.cov())
    assert float(rel) < 0.1


# ============================================================================
# 8. Replace / immutability
# ============================================================================


def test_replace_updates_top_level_fields(small_factor_setup):
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)

    new_F = F * 2.0
    fa2 = fa.replace(F=new_F)
    np.testing.assert_allclose(np.asarray(fa2.F), 2.0 * np.asarray(F))
    # Immutability: original unchanged.
    np.testing.assert_allclose(np.asarray(fa.F), np.asarray(F))


def test_replace_unknown_key_raises(small_factor_setup):
    mu, gamma, F, D = small_factor_setup
    fa = FactorVarianceGamma.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, alpha=2.0, beta=1.0)
    with pytest.raises(ValueError, match="unknown field"):
        fa.replace(alpha=99.0)


# ============================================================================
# 9. det(Σ)=1 regularisation round-trips through Y'·Σ' invariance
# ============================================================================


@pytest.mark.parametrize(
    "factor_cls, sub",
    [
        (FactorVarianceGamma, dict(alpha=2.5, beta=1.5)),
        (FactorNormalInverseGamma, dict(alpha=3.0, beta=1.5)),
        (FactorNormalInverseGaussian, dict(mu_ig=1.0, lam=2.0)),
        (FactorGeneralizedHyperbolic, dict(p=1.0, a=1.5, b=1.0)),
    ],
    ids=["VG", "NInvG", "NIG", "GH"],
)
def test_regularize_det_sigma_one_preserves_log_prob(factor_cls, sub):
    """The det(Σ)=1 rescaling is a reparametrisation; log_prob is invariant."""
    F = jnp.array([[1.0, 0.0], [0.5, 0.3], [0.0, 0.8]])
    D = jnp.array([0.5, 0.7, 0.4])
    mu = jnp.array([0.1, 0.2, -0.1])
    gamma = jnp.array([0.1, 0.0, -0.05])

    fa = factor_cls.from_classical(
        mu=mu, gamma=gamma, F=F, D=D, **sub)
    fa_reg = fa.regularize_det_sigma_one()

    # |Σ| = 1 after regularisation.
    np.testing.assert_allclose(float(fa_reg.log_det_sigma()), 0.0, atol=1e-10)

    # log_prob unchanged on a fresh sample.
    key = jax.random.PRNGKey(2)
    X = jax.random.normal(key, (8, mu.shape[0]), dtype=jnp.float64)
    lp_before = jax.vmap(fa.log_prob)(X)
    lp_after = jax.vmap(fa_reg.log_prob)(X)
    np.testing.assert_allclose(np.asarray(lp_before), np.asarray(lp_after),
                               rtol=1e-9, atol=1e-10)
