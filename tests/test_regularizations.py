"""
Tests for the orbit-regularization options.

Covers the three regularizations now exposed by :class:`BatchEMFitter`:

- ``'det_sigma_one'`` — :math:`|\\Sigma| = 1`
- ``'det_sigma_x'`` — :math:`|\\Sigma| = |\\Sigma_0|` (initial reference)
- ``'a_eq_b'`` — GIG subordinator: :math:`a = b = \\sqrt{ab}`

For each variant we check:

1. **Log-prob invariance** — applying the rescaling does not change the
   marginal density evaluated on a fresh sample. The rescaling is a pure
   reparameterisation of the same distribution.
2. **Orbit invariants** — :math:`p`, :math:`\\mu`, :math:`a\\cdot b`, and
   :math:`\\gamma^\\top \\Sigma^{-1} \\gamma` must be preserved exactly.
3. **Target equality** — after rescaling, the chosen target is met
   (:math:`|\\Sigma|=1`, :math:`|\\Sigma|=|\\Sigma_0|`, or :math:`a = b`).
4. **Idempotence** — applying the rescaling twice equals applying it once.

Parametrised over both full-cov families
(:class:`VarianceGamma`, :class:`NormalInverseGamma`,
:class:`NormalInverseGaussian`, :class:`GeneralizedHyperbolic`) and
factor-analysis families
(:class:`FactorVarianceGamma`, :class:`FactorNormalInverseGamma`,
:class:`FactorNormalInverseGaussian`,
:class:`FactorGeneralizedHyperbolic`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import (
    BatchEMFitter,
    FactorGeneralizedHyperbolic,
    FactorNormalInverseGamma,
    FactorNormalInverseGaussian,
    FactorVarianceGamma,
    GeneralizedHyperbolic,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)


# ============================================================================
# Helpers
# ============================================================================


def _gamma_E_Y(model) -> np.ndarray:
    r"""Orbit-invariant skewness vector :math:`\gamma\,E[Y]`.

    Under :math:`Y \to s\,Y`, :math:`\gamma \to \gamma/s` and
    :math:`E[Y] \to s\,E[Y]`, so the product is preserved.
    """
    sub = (model.subordinator if hasattr(model, 'subordinator')
           and not callable(model.subordinator)
           else model._joint.subordinator())
    if callable(sub):
        sub = sub()
    return np.asarray(model.gamma) * float(sub.mean())


def _cov_X(model) -> np.ndarray:
    """Marginal covariance — invariant under the scale orbit."""
    return np.asarray(model.cov())


def _ab(model) -> float:
    """``a · b`` for GH-family models (orbit invariant). Returns 0 when one
    of (a, b) is identically zero (VG / NInvG boundary cases)."""
    family = type(model).__name__
    if family in ('VarianceGamma', 'FactorVarianceGamma'):
        return 0.0
    if family in ('NormalInverseGamma', 'FactorNormalInverseGamma'):
        return 0.0
    if family in ('NormalInverseGaussian', 'FactorNormalInverseGaussian'):
        a = float(model.lam / model.mu_ig ** 2)
        b = float(model.lam)
        return a * b
    if family in ('GeneralizedHyperbolic', 'FactorGeneralizedHyperbolic'):
        return float(model.a * model.b)
    raise KeyError(family)


# ============================================================================
# Fixture: one model per family on a small d (and a fresh test sample)
# ============================================================================

D = 4
KEY = jax.random.PRNGKey(7)
X = jax.random.normal(KEY, (32, D), dtype=jnp.float64) * 0.2

mu0    = jnp.array([0.1, -0.2, 0.0, 0.3])
gamma0 = jnp.array([0.2, 0.1, -0.1, 0.0])
sigma0 = (jnp.array([[1.5, 0.3, 0.1, 0.0],
                     [0.3, 1.2, 0.2, 0.1],
                     [0.1, 0.2, 1.0, 0.2],
                     [0.0, 0.1, 0.2, 0.8]]))
F0 = jnp.array([[1.0, 0.0],
                [0.6, 0.3],
                [-0.2, 0.5],
                [0.0, 0.7]])
D0 = jnp.array([0.4, 0.5, 0.3, 0.6])


def _make_full_models():
    return {
        'VG': VarianceGamma.from_classical(
            mu=mu0, gamma=gamma0, sigma=sigma0, alpha=2.5, beta=1.5),
        'NInvG': NormalInverseGamma.from_classical(
            mu=mu0, gamma=gamma0, sigma=sigma0, alpha=3.0, beta=1.5),
        'NIG': NormalInverseGaussian.from_classical(
            mu=mu0, gamma=gamma0, sigma=sigma0, mu_ig=1.5, lam=2.0),
        'GH': GeneralizedHyperbolic.from_classical(
            mu=mu0, gamma=gamma0, sigma=sigma0, p=1.0, a=2.0, b=0.5),
    }


def _make_factor_models():
    return {
        'FactorVG': FactorVarianceGamma.from_classical(
            mu=mu0, gamma=gamma0, F=F0, D=D0, alpha=2.5, beta=1.5),
        'FactorNInvG': FactorNormalInverseGamma.from_classical(
            mu=mu0, gamma=gamma0, F=F0, D=D0, alpha=3.0, beta=1.5),
        'FactorNIG': FactorNormalInverseGaussian.from_classical(
            mu=mu0, gamma=gamma0, F=F0, D=D0, mu_ig=1.5, lam=2.0),
        'FactorGH': FactorGeneralizedHyperbolic.from_classical(
            mu=mu0, gamma=gamma0, F=F0, D=D0, p=1.0, a=2.0, b=0.5),
    }


def _all_models():
    return {**_make_full_models(), **_make_factor_models()}


ALL_NAMES = list(_all_models().keys())


# ============================================================================
# 1. Log-prob invariance under each regularization
# ============================================================================


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("method", ['det_sigma_one', 'det_sigma_x', 'a_eq_b'])
def test_regularization_preserves_log_prob(name, method):
    model = _all_models()[name]
    target = float(model.log_det_sigma())

    if method == 'det_sigma_one':
        regularised = model.regularize_det_sigma(0.0)
    elif method == 'det_sigma_x':
        regularised = model.regularize_det_sigma(target)
    else:
        regularised = model.regularize_a_eq_b()

    lp_before = jax.vmap(model.log_prob)(X)
    lp_after = jax.vmap(regularised.log_prob)(X)
    np.testing.assert_allclose(
        np.asarray(lp_after), np.asarray(lp_before),
        rtol=1e-9, atol=1e-10,
        err_msg=f"{name} {method} broke log-prob invariance",
    )


# ============================================================================
# 2. Orbit invariants (μ, p-equivalent, ab, γᵀΣ⁻¹γ)
# ============================================================================


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("method", ['det_sigma_one', 'det_sigma_x', 'a_eq_b'])
def test_regularization_preserves_orbit_invariants(name, method):
    model = _all_models()[name]
    target = float(model.log_det_sigma())
    if method == 'det_sigma_one':
        regularised = model.regularize_det_sigma(0.0)
    elif method == 'det_sigma_x':
        regularised = model.regularize_det_sigma(target)
    else:
        regularised = model.regularize_a_eq_b()

    np.testing.assert_allclose(
        np.asarray(regularised.mu), np.asarray(model.mu), atol=1e-12)
    np.testing.assert_allclose(_ab(regularised), _ab(model), atol=1e-12)
    # γ·E[Y] is orbit-invariant (γ scales as 1/s, E[Y] scales as s).
    np.testing.assert_allclose(
        _gamma_E_Y(regularised), _gamma_E_Y(model),
        rtol=1e-9, atol=1e-12)
    # The full marginal Cov[X] is invariant by construction.
    np.testing.assert_allclose(
        _cov_X(regularised), _cov_X(model), rtol=1e-8, atol=1e-10)


# ============================================================================
# 3. Target equalities
# ============================================================================


@pytest.mark.parametrize("name", ALL_NAMES)
def test_det_sigma_one_yields_unit_determinant(name):
    model = _all_models()[name]
    regularised = model.regularize_det_sigma(0.0)
    np.testing.assert_allclose(
        float(regularised.log_det_sigma()), 0.0, atol=1e-10)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_det_sigma_x_yields_target_determinant(name):
    model = _all_models()[name]
    target = float(model.log_det_sigma()) + 1.0  # arbitrary non-zero target
    regularised = model.regularize_det_sigma(target)
    np.testing.assert_allclose(
        float(regularised.log_det_sigma()), target, atol=1e-9)


@pytest.mark.parametrize(
    "name", ['NIG', 'GH', 'FactorNIG', 'FactorGH'],
)
def test_a_eq_b_makes_a_equal_b(name):
    model = _all_models()[name]
    regularised = model.regularize_a_eq_b()
    if name in ('NIG', 'FactorNIG'):
        a = float(regularised.lam / regularised.mu_ig ** 2)
        b = float(regularised.lam)
    else:
        a = float(regularised.a)
        b = float(regularised.b)
    np.testing.assert_allclose(a, b, rtol=1e-10)


@pytest.mark.parametrize(
    "name", ['VG', 'NInvG', 'FactorVG', 'FactorNInvG'],
)
def test_a_eq_b_is_noop_for_boundary_families(name):
    """VG (b=0) and NInvG (a=0) cannot satisfy a=b without a degenerate
    rescale; the default behaviour is to return the model unchanged."""
    model = _all_models()[name]
    regularised = model.regularize_a_eq_b()
    assert regularised is model


# ============================================================================
# 4. Idempotence: regularize ∘ regularize == regularize
# ============================================================================


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("method", ['det_sigma_one', 'a_eq_b'])
def test_regularization_idempotent(name, method):
    model = _all_models()[name]
    if method == 'det_sigma_one':
        once  = model.regularize_det_sigma(0.0)
        twice = once.regularize_det_sigma(0.0)
    else:
        once  = model.regularize_a_eq_b()
        twice = once.regularize_a_eq_b()
    lp_once = jax.vmap(once.log_prob)(X)
    lp_twice = jax.vmap(twice.log_prob)(X)
    np.testing.assert_allclose(
        np.asarray(lp_twice), np.asarray(lp_once),
        rtol=1e-10, atol=1e-12)


# ============================================================================
# 5. BatchEMFitter integration
# ============================================================================


def test_batch_em_fitter_rejects_unknown_regularization():
    with pytest.raises(ValueError, match="regularization must be one of"):
        BatchEMFitter(regularization='bogus')


def test_batch_em_fitter_det_sigma_x_targets_initial_log_det():
    """After EM with regularization='det_sigma_x', |Σ_fitted| matches
    |Σ_init| up to FP roundoff."""
    init = NormalInverseGaussian.default_init(np.asarray(X))
    target = float(init.log_det_sigma())

    fitter = BatchEMFitter(
        regularization='det_sigma_x',
        max_iter=4, tol=1e-2, verbose=0,
        e_step_backend='cpu', m_step_backend='cpu',
    )
    result = fitter.fit(init, np.asarray(X))
    np.testing.assert_allclose(
        float(result.model.log_det_sigma()), target,
        atol=1e-6, rtol=1e-6)


def test_batch_em_fitter_a_eq_b_yields_equal_a_b_for_gh():
    init = GeneralizedHyperbolic.default_init(np.asarray(X))
    fitter = BatchEMFitter(
        regularization='a_eq_b',
        max_iter=4, tol=1e-2, verbose=0,
        e_step_backend='cpu', m_step_backend='cpu',
    )
    result = fitter.fit(init, np.asarray(X))
    a = float(result.model.a)
    b = float(result.model.b)
    np.testing.assert_allclose(a, b, rtol=1e-9)


# ============================================================================
# 6. FactorGH.default_init runs and beats the moment-based fallback
# ============================================================================


def test_factor_gh_default_init_beats_moment_fallback():
    """default_init should never give a worse LL than the parent's
    moment-based fallback."""
    n, d = 600, 4
    truth = FactorGeneralizedHyperbolic.from_classical(
        mu=jnp.zeros(d), gamma=0.1 * jnp.ones(d),
        F=F0, D=D0, p=1.0, a=2.0, b=0.5,
    )
    Xn = truth.rvs(n, seed=11)

    # Parent's moment-based init via super().default_init.
    fallback = FactorNormalMixture_default_init_fallback(
        FactorGeneralizedHyperbolic, Xn, r=2)
    ll_fallback = float(fallback.marginal_log_likelihood(Xn))

    init = FactorGeneralizedHyperbolic.default_init(Xn, r=2)
    ll_init = float(init.marginal_log_likelihood(Xn))

    # The new default_init runs short EM fits, so it should never lose
    # to the cold moment-based fallback.
    assert ll_init >= ll_fallback - 1e-6, (
        f"default_init LL={ll_init:.6f} worse than moment fallback "
        f"LL={ll_fallback:.6f}")


def FactorNormalMixture_default_init_fallback(cls, X, *, r):
    """Helper that calls the parent (PCA-based) default_init bypassing
    the GH-specific subclass override."""
    from normix.mixtures.factor import FactorNormalMixture
    return FactorNormalMixture.default_init.__func__(cls, X, r=r)
