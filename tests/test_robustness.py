"""Translation, support, and fit_mle robustness.

These pin the three failures verified on master 85f835e: the normal M-step
loses Σ when moments are stored uncentered, positive-support densities are
NaN off the support, and fit_mle returns a Newton cap or a clamp instead of
raising on data with no finite MLE.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import GIG, Gamma, InverseGamma, InverseGaussian, VarianceGamma
from normix.fitting.em import BatchEMFitter
from normix.fitting.eta_rules import IdentityUpdate, Shrinkage
from normix.fitting.shrinkage_targets import eta0_from_model
from normix.mixtures.joint import JointNormalMixture
from normix.utils.constants import SIGMA_REG

pytestmark = pytest.mark.contract

_FAMILIES = [Gamma, InverseGamma, InverseGaussian, GIG]


def _package_sigma(model) -> np.ndarray:
    _, _, L = JointNormalMixture._mstep_normal_params(model.compute_eta_from_model())
    return np.asarray(L @ L.T)


@pytest.mark.parametrize("shift", [0.0, 1e8, 1e10])
@pytest.mark.parametrize("gamma", [0.0, 0.3])
def test_exact_vg_sigma_is_translation_invariant(shift, gamma):
    """Σ = 1 survives the M-step at a large location, including γ = 0.

    γ = 0 is the silent failure: the uncentered second moment rounds to
    c², the M-step returns only SIGMA_REG, and the fit still reports success.
    """
    model = VarianceGamma.from_classical(
        mu=jnp.array([shift]),
        gamma=jnp.array([gamma]),
        sigma=jnp.eye(1),
        alpha=3.0,
        beta=2.0,
    )
    sigma = _package_sigma(model)
    assert np.isfinite(sigma).all()
    np.testing.assert_allclose(sigma, np.eye(1), atol=1e-6, rtol=0.0)
    assert float(sigma[0, 0]) > 100.0 * SIGMA_REG


@pytest.mark.parametrize("shift", [1e8, 1e10])
def test_fit_gamma_zero_does_not_collapse_sigma(shift):
    """A shifted γ = 0 sample must not fit a jitter covariance."""
    base = VarianceGamma.from_classical(
        mu=jnp.zeros(1),
        gamma=jnp.zeros(1),
        sigma=jnp.eye(1),
        alpha=3.0,
        beta=2.0,
    )
    X = jnp.asarray(base.rvs(400, seed=1)) + shift
    result = VarianceGamma.default_init(X).fit(
        X, max_iter=15, tol=1e-8,
        e_step_backend="jax", m_step_backend="jax",
    )
    sigma = np.asarray(result.model.sigma())
    assert result.diverged is False
    assert np.isfinite(sigma).all()
    assert float(sigma[0, 0]) > 0.1


def test_shrinkage_from_model_keeps_sigma_at_large_location():
    """A self-prior must not be translated as if it still contained μ.

    ``compute_eta_from_model`` stores (s4, s5, s6) about μ at this
    location. Subtracting the sample mean from those moments puts
    s1 c c^T back into s6, and the fit reports success with Σ ~ 1e15.
    """
    shift = 1e8
    base = VarianceGamma.from_classical(
        mu=jnp.zeros(1),
        gamma=jnp.zeros(1),
        sigma=jnp.eye(1),
        alpha=3.0,
        beta=2.0,
    )
    X = jnp.asarray(base.rvs(200, seed=0)) + shift
    init = VarianceGamma.default_init(X)
    result = BatchEMFitter(
        max_iter=8,
        tol=1e-6,
        eta_update=Shrinkage(IdentityUpdate(), eta0_from_model(init), tau=0.3),
        e_step_backend="jax",
        m_step_backend="jax",
    ).fit(init, X)
    sigma = float(np.asarray(result.model.sigma())[0, 0])
    assert result.diverged is False
    assert math.isfinite(sigma)
    assert 0.1 < sigma < 100.0


def test_log_prob_pdf_cdf_off_support_and_at_zero():
    """x < 0 is off support; x = 0 is the one-sided limit. Grads stay non-NaN."""
    cases = [
        (Gamma(0.5, 2.0), math.inf, lambda t: Gamma(t, 2.0), 0.5),
        (Gamma(1.0, 2.0), math.log(2.0), lambda t: Gamma(t, 2.0), 1.0),
        (Gamma(3.0, 2.0), -math.inf, lambda t: Gamma(t, 2.0), 3.0),
        (InverseGamma(2.0, 1.0), -math.inf, lambda t: InverseGamma(t, 1.0), 2.0),
        (InverseGaussian(1.0, 2.0), -math.inf, lambda t: InverseGaussian(t, 2.0), 1.0),
        (GIG(0.5, 1.0, 1.0), -math.inf, lambda t: GIG(t, 1.0, 1.0), 0.5),
        (GIG(1.0, 1.0, 1.0), -math.inf, lambda t: GIG(t, 1.0, 1.0), 1.0),
        (GIG(2.0, 1.0, 1.0), -math.inf, lambda t: GIG(t, 1.0, 1.0), 2.0),
    ]
    for dist, log_at_zero, build, t0 in cases:
        for x in (-1.0, -1e-300):
            lp = float(dist.log_prob(x))
            assert lp == -math.inf
            assert float(dist.pdf(x)) == 0.0
            assert float(dist.cdf(x)) == 0.0
            gx = float(jax.grad(lambda z, dist=dist: dist.log_prob(z))(jnp.float64(x)))
            gt = float(jax.grad(
                lambda t, build=build, x=x: build(t).log_prob(jnp.float64(x)),
            )(jnp.float64(t0)))
            assert not math.isnan(gx)
            assert not math.isnan(gt)
        lp0 = float(dist.log_prob(0.0))
        assert not math.isnan(lp0)
        if math.isinf(log_at_zero):
            assert lp0 == log_at_zero
        else:
            assert lp0 == pytest.approx(log_at_zero, rel=1e-12, abs=1e-12)
        assert float(dist.cdf(0.0)) == 0.0
        pdf0 = float(dist.pdf(0.0))
        assert not math.isnan(pdf0)


@pytest.mark.parametrize("cls", _FAMILIES)
@pytest.mark.parametrize("label,x", [
    ("empty", np.array([])),
    ("n=1", np.array([2.0])),
    ("constant", np.full(50, 2.0)),
    ("nonpositive", np.array([0.0, 1.0, 2.0])),
    ("negative", np.array([-1.0, 1.0, 2.0])),
    ("nan", np.array([np.nan, 1.0, 2.0])),
    ("inf", np.array([np.inf, 1.0, 2.0])),
])
def test_fit_mle_rejects_invalid_data(cls, label, x):
    with pytest.raises(ValueError):
        cls.fit_mle(jnp.asarray(x))


def test_fit_mle_constant_message_names_jensen_gap():
    with pytest.raises(ValueError, match="Jensen"):
        Gamma.fit_mle(jnp.full(20, 3.7))


def test_fit_mle_vmap_on_valid_samples():
    """The host-side check must not block a traced fit of a valid sample."""
    x = jnp.array([0.5, 1.0, 2.0, 3.0, 0.7, 1.4])
    fitted = jax.vmap(Gamma.fit_mle)(jnp.stack([x, x * 1.1]))
    assert np.isfinite(np.asarray(fitted.alpha)).all()
    assert np.isfinite(np.asarray(fitted.beta)).all()


def test_fit_mle_valid_sample_unchanged():
    x = jnp.array([0.5, 1.0, 2.0, 3.0, 0.7, 1.4])
    g = Gamma.fit_mle(x)
    assert math.isfinite(float(g.alpha)) and math.isfinite(float(g.beta))
    ig = InverseGaussian.fit_mle(x)
    mu = float(np.mean(np.asarray(x)))
    lam = 1.0 / float(np.mean(1.0 / np.asarray(x) - 1.0 / mu))
    np.testing.assert_allclose(float(ig.mu), mu, rtol=1e-9)
    np.testing.assert_allclose(float(ig.lam), lam, rtol=1e-9)
