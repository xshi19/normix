"""
EM algorithm regression tests (current API).

Verify that EM fitting produces valid, finite results with reasonable
log-likelihoods for all mixture distribution families.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix.distributions.variance_gamma import VarianceGamma
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.fitting.em import BatchEMFitter
from normix.fitting.eta import NormalMixtureEta
from normix.mixtures.joint import JointNormalMixture

class TestVGEMRegression:

    def test_vg_em_1d(self):
        true = VarianceGamma.from_classical(
            mu=jnp.array([0.5]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), atol=0.3)
        assert result.n_iter <= 50

    def test_vg_em_2d(self):
        true = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 1.0]),
            gamma=jnp.array([0.2, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=3.0, beta=2.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)

class TestNInvGEMRegression:

    def test_ninvg_em_1d(self):
        true = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=3.0, beta=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        assert float(fitted._joint.alpha) > 1.0
        assert result.n_iter <= 50

    def test_ninvg_em_2d(self):
        true = NormalInverseGamma.from_classical(
            mu=jnp.array([0.0, 0.5]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
            alpha=4.0, beta=2.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)

class TestNIGEMRegression:

    def test_nig_em_1d(self):
        true = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), mu_ig=1.0, lam=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        assert float(fitted._joint.mu_ig) > 0
        assert float(fitted._joint.lam) > 0
        assert result.n_iter <= 50

    def test_nig_em_2d(self):
        true = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0, 0.5]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.2], [0.2, 1.0]]),
            mu_ig=1.0, lam=2.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=50, tol=1e-8, verbose=0,
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)

    def test_nig_docs_quick_usage_converges(self):
        """Docs gallery example: centred μ=0 must converge within 50 iters.

        Pure relative ‖Δμ‖/‖μ‖ inflated near-zero μ drifts along the (μ, γ)
        ridge; hybrid-scale RMS rms(Δ)/(1+rms(θ)) is the intended criterion.
        """
        true = NormalInverseGaussian.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.4]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            mu_ig=1.0, lam=1.5,
        )
        X = true.rvs(2_000, seed=0)
        result = NormalInverseGaussian.default_init(X).fit(
            X, max_iter=50, tol=1e-3)
        assert result.converged
        assert int(result.n_iter) <= 50
        assert int(result.n_iter) > 1
        np.testing.assert_allclose(
            np.asarray(result.model.gamma), np.array([0.3, -0.4]), atol=0.15)

class TestParamChangeMetric:
    """Hybrid-scale RMS change is roughly dimension-free."""

    def test_same_per_coord_drift_independent_of_d(self):
        from normix.fitting.em import _param_change

        scores = []
        for d in (1, 2, 10, 50):
            old = (jnp.zeros(d), 0.3 * jnp.ones(d), jnp.eye(d))
            new = (
                1e-3 * jnp.ones(d),
                0.3 * jnp.ones(d) + 1e-3 * jnp.ones(d),
                jnp.eye(d),
            )
            scores.append(float(_param_change(new, old)))
        np.testing.assert_allclose(scores, scores[0], rtol=1e-12)

class TestGHEMRegression:

    def test_gh_em_1d_det_sigma_one(self):
        true = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), p=1.0, a=1.0, b=1.0,
        )
        X = true.rvs(2000, seed=42)
        result = true.fit(X, max_iter=30, tol=1e-4, verbose=0,
                          regularization='det_sigma_one',
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        assert float(fitted._joint.a) > 0
        assert float(fitted._joint.b) > 0
        ll = float(fitted.marginal_log_likelihood(X))
        assert np.isfinite(ll)
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), atol=0.25)

    def test_gh_em_2d_det_sigma_one(self):
        true = GeneralizedHyperbolic.from_classical(
            mu=jnp.array([0.0, 0.5]),
            gamma=jnp.array([0.2, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            p=-0.5, a=1.0, b=1.0,
        )
        X = true.rvs(3000, seed=42)
        result = true.fit(X, max_iter=30, tol=1e-4, verbose=0,
                          regularization='det_sigma_one',
                          e_step_backend='cpu', m_step_backend='cpu')
        fitted = result.model
        L = np.array(fitted._joint.L_Sigma)
        Sigma = L @ L.T
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)
        ll = float(fitted.marginal_log_likelihood(X))
        assert np.isfinite(ll)
        np.testing.assert_allclose(
            np.array(fitted.mean()), np.array(true.mean()), atol=0.25)

def _consistent_eta(mu_t, gam_t, eta2, D):
    """Build eta so the exact M-step returns (mu_t, gam_t) for this D."""
    eta3 = (1.0 - D) / eta2
    eta5 = gam_t + eta2 * mu_t
    eta4 = mu_t * D + eta3 * eta5
    d = mu_t.shape[0]
    return NormalMixtureEta(
        E_inv_Y=jnp.asarray(eta2),
        E_Y=jnp.asarray(eta3),
        E_log_Y=jnp.asarray(0.0),
        E_X=eta4,
        E_X_inv_Y=eta5,
        E_XXT_inv_Y=jnp.eye(d),
    )

class TestMstepDenominatorSign:
    """B1: sign-preserving M-step denominator D = 1 - eta2*eta3."""

    @pytest.mark.parametrize("D", [-1e-12, -5e-11])
    def test_tiny_D_preserves_mu_gamma_sign(self, D):
        mu_t = jnp.array([0.5, -1.0])
        gam_t = jnp.array([2.0, 0.3])
        eta2 = 1.3
        ref = JointNormalMixture._mstep_normal_params(
            _consistent_eta(mu_t, gam_t, eta2, -1e-8))
        got = JointNormalMixture._mstep_normal_params(
            _consistent_eta(mu_t, gam_t, eta2, D))
        for a, b in zip(ref[:2], got[:2]):
            assert np.all(np.sign(np.array(a)) == np.sign(np.array(b)))

    def test_positive_D_roundoff_uses_negative_floor(self):
        """Near-Gaussian VG e-step: M-step μ keeps the data sign (B1).

        Correct Bessel moments make D ≤ 0 here (Cauchy–Schwarz). The
        synthetic D > 0 roundoff case lives in
        ``TestMStepDenominatorSign`` in ``test_jax_distributions.py``.
        """
        key = jax.random.PRNGKey(7)
        d, n = 2, 5000
        mu_true = jnp.array([1.5, -0.7])
        A = jax.random.normal(key, (d, d)) * 0.5
        Sigma = A @ A.T + jnp.eye(d)
        X = jax.random.multivariate_normal(
            jax.random.PRNGKey(8), mu_true, Sigma, (n,))
        model = VarianceGamma.from_classical(
            mu=mu_true, gamma=jnp.array([0.01, -0.01]),
            sigma=Sigma, alpha=1e9, beta=1e9)
        eta = model.e_step(X, backend="jax").eta
        D = float(1.0 - eta.E_inv_Y * eta.E_Y)
        assert D < 1e-6
        mu, _, _ = JointNormalMixture._mstep_normal_params(eta)
        assert float(mu[0]) * float(mu_true[0]) > 0

    @pytest.mark.parametrize(
        "cls",
        [VarianceGamma, NormalInverseGamma, NormalInverseGaussian,
         GeneralizedHyperbolic],
    )
    def test_e_step_batch_D_nonpositive(self, cls):
        key = jax.random.PRNGKey(0)
        d, n = 3, 1500
        X = jax.random.multivariate_normal(
            key, jnp.zeros(d), jnp.eye(d), (n,)) * 1.2 + 0.3
        model = cls.default_init(X)
        eta = model.e_step(X, backend="jax").eta
        D = float(1.0 - eta.E_inv_Y * eta.E_Y)
        assert D <= 1e-9

def _heavy_peaked_vg_1d():
    """Heavy-peaked d=1 VG data with a near-mode observation (T5 setup)."""
    vg_true = VarianceGamma.from_classical(
        mu=jnp.array([0.0]), gamma=jnp.array([0.2]),
        sigma=jnp.array([[1.0]]), alpha=0.7, beta=1.0,
    )
    X = vg_true.rvs(3000, seed=0).reshape(-1, 1)
    return jnp.concatenate([jnp.mean(X, axis=0, keepdims=True), X], axis=0)

def _assert_model_finite(model) -> None:
    for leaf in jax.tree.leaves(model):
        assert jnp.all(jnp.isfinite(leaf))

class TestEMMonotoneLL:
    """T5: per-iteration LL is non-decreasing (EM invariant) with track_ll."""

    @pytest.mark.parametrize("loop", ["scan", "python"])
    def test_heavy_peaked_vg_monotone_ll(self, loop):
        X = _heavy_peaked_vg_1d()
        if loop == "scan":
            fitter = BatchEMFitter(
                max_iter=100, tol=1e-4, verbose=0, track_ll=True,
                e_step_backend="jax", m_step_backend="jax",
            )
        else:
            fitter = BatchEMFitter(
                max_iter=100, tol=1e-4, verbose=0, track_ll=True,
                e_step_backend="cpu", m_step_backend="cpu",
            )
        result = fitter.fit(VarianceGamma.default_init(X), X)
        assert result.log_likelihoods is not None
        assert not result.diverged
        changes = result.param_changes
        dll = jnp.diff(result.log_likelihoods)
        # Only check iterations that actually updated the model (scan freezes
        # post-convergence with change=0). Allow float64 slack in floored VG EM.
        active = changes[:-1] > 0
        assert jnp.all(dll[active] >= -1e-5)

class TestEMAitkenEarliestStop:
    """Aitken remaining gap needs three consecutive ℓ; EMResult uses Python types."""

    @staticmethod
    def _near_mle_setup():
        """True VG parameters as init: Aitken remaining is small by step 3."""
        true = VarianceGamma.from_classical(
            mu=jnp.array([0.5]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = true.rvs(2000, seed=42)
        return true, X, 1.0

    @pytest.mark.contract
    @pytest.mark.parametrize("loop", ["scan", "python"])
    def test_cannot_stop_before_three_lls(self, loop):
        init, X, _ = self._near_mle_setup()
        kwargs = dict(max_iter=2, tol=1e6, verbose=0)
        if loop == "scan":
            kwargs.update(e_step_backend="jax", m_step_backend="jax")
        else:
            kwargs.update(e_step_backend="cpu", m_step_backend="cpu")
        result = BatchEMFitter(**kwargs).fit(init, X)
        assert result.converged is False
        assert result.diverged is False
        assert result.n_iter == 2

    @pytest.mark.contract
    @pytest.mark.parametrize("loop", ["scan", "python"])
    def test_earliest_stop_is_three_and_python_types(self, loop):
        init, X, tol = self._near_mle_setup()
        kwargs = dict(max_iter=20, tol=tol, verbose=0)
        if loop == "scan":
            kwargs.update(e_step_backend="jax", m_step_backend="jax")
        else:
            kwargs.update(e_step_backend="cpu", m_step_backend="cpu")
        result = BatchEMFitter(**kwargs).fit(init, X)
        assert result.converged is True
        assert result.diverged is False
        assert result.n_iter == 3
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iter, int)
        assert isinstance(result.diverged, bool)

    @pytest.mark.contract
    def test_scan_and_loop_agree_on_earliest_stop(self):
        init, X, tol = self._near_mle_setup()
        scan = BatchEMFitter(
            max_iter=20, tol=tol, verbose=0,
            e_step_backend="jax", m_step_backend="jax",
        ).fit(init, X)
        loop = BatchEMFitter(
            max_iter=20, tol=tol, verbose=0,
            e_step_backend="cpu", m_step_backend="cpu",
        ).fit(init, X)
        assert scan.converged is True and loop.converged is True
        assert scan.n_iter == loop.n_iter == 3

    @pytest.mark.contract
    def test_scan_does_not_diverge_after_convergence(self):
        """Post-convergence scan steps must not set diverged on an accepted model."""
        init, X, tol = self._near_mle_setup()
        fitter = BatchEMFitter(
            max_iter=20, tol=tol, verbose=0,
            e_step_backend="jax", m_step_backend="jax",
        )
        fitter._force_nonfinite_at_step = 4
        result = fitter.fit(init, X)
        assert result.converged is True
        assert result.diverged is False
        assert result.n_iter == 3

    @pytest.mark.contract
    def test_scan_does_not_converge_after_divergence(self):
        """Post-divergence scan padding must not set converged=True."""
        init, X, _ = self._near_mle_setup()
        fitter = BatchEMFitter(
            max_iter=5, tol=1e6, verbose=0,
            e_step_backend="jax", m_step_backend="jax",
        )
        fitter._force_nonfinite_at_step = 1
        result = fitter.fit(init, X)
        assert result.diverged is True
        assert result.converged is False
        assert result.n_iter == 1

class TestEMDivergenceGuard:
    """T7: non-finite iterate triggers diverged=True and keep-last-finite."""

    @staticmethod
    def _fit_with_forced_divergence(fitter, model, X, *, at_step: int):
        fitter._force_nonfinite_at_step = at_step
        return fitter.fit(model, X)

    @pytest.mark.parametrize("loop", ["scan", "python"])
    def test_diverged_keeps_last_finite(self, loop):
        X = _heavy_peaked_vg_1d()
        init = VarianceGamma.default_init(X)
        if loop == "scan":
            fitter = BatchEMFitter(
                max_iter=20, tol=1e-8, verbose=0,
                e_step_backend="jax", m_step_backend="jax",
            )
        else:
            fitter = BatchEMFitter(
                max_iter=20, tol=1e-8, verbose=0,
                e_step_backend="cpu", m_step_backend="cpu",
            )
        result = self._fit_with_forced_divergence(
            fitter, init, X, at_step=2)
        assert result.diverged is True
        assert not result.converged
        _assert_model_finite(result.model)
        assert int(result.n_iter) >= 1
        assert jnp.any(~jnp.isfinite(result.param_changes))


def _review_vg_setup(scale: float = 1.0):
    """2026-09-05 review VG: Gamma(0.7, rate=0.7), n=2000, seed 13, init α=β=2."""
    rng = np.random.default_rng(13)
    y = rng.gamma(0.7, 1.0 / 0.7, 2000)
    x = (np.sqrt(y) * rng.normal(size=2000))[:, None] * scale
    X = jnp.asarray(x, dtype=jnp.float64)
    init = VarianceGamma.from_classical(
        mu=X.mean(axis=0), gamma=jnp.zeros(1),
        sigma=jnp.atleast_2d(X.var()), alpha=2.0, beta=2.0,
    )
    return init, X


class TestAitkenRemaining:
    """Aitken remaining gap formula (geometric sequence)."""

    def test_geometric_half(self):
        from normix.fitting.em import _aitken_remaining
        # ℓ = 0, 1, 1.5 → Δ1=1, Δ2=0.5, a=0.5 → remaining = 0.5
        rem = _aitken_remaining(
            jnp.asarray(1.5), jnp.asarray(1.0), jnp.asarray(0.0))
        np.testing.assert_allclose(float(rem), 0.5, rtol=1e-12)

    def test_stalled_is_zero(self):
        from normix.fitting.em import _aitken_remaining
        rem = _aitken_remaining(
            jnp.asarray(-1.2), jnp.asarray(-1.2), jnp.asarray(-1.2))
        np.testing.assert_allclose(float(rem), 0.0, atol=1e-15)

    def test_negative_delta_does_not_stop(self):
        from normix.fitting.em import _aitken_remaining
        rem = _aitken_remaining(
            jnp.asarray(0.9), jnp.asarray(1.0), jnp.asarray(0.0))
        assert float(rem) == np.inf


class TestReviewVGAitken:
    """Review VG 2000-obs case: default tol must not stop at α≈1.59."""

    def test_default_tol_and_continuation(self):
        init, X = _review_vg_setup()
        result = init.fit(
            X, max_iter=80, verbose=0,
            e_step_backend="cpu", m_step_backend="cpu",
        )
        alpha = float(result.model.joint.subordinator().alpha)
        assert result.n_iter > 3
        assert not (result.converged and abs(alpha - 1.59) < 0.1), (
            f"stopped at α={alpha:.4f} with converged={result.converged} "
            f"after {result.n_iter} iters (review false stop was α≈1.59)"
        )
        continued = result.model.fit(
            X, max_iter=50, tol=1e-8, verbose=0,
            e_step_backend="cpu", m_step_backend="cpu",
        )
        alpha2 = float(continued.model.joint.subordinator().alpha)
        np.testing.assert_allclose(alpha2, 0.7, atol=0.15)

    def test_unit_scale_does_not_stop_after_one_iter(self):
        """Data × 0.001 must not stop after one iteration from the hybrid-RMS floor."""
        init, X = _review_vg_setup(scale=0.001)
        result = init.fit(
            X, max_iter=80, verbose=0,
            e_step_backend="cpu", m_step_backend="cpu",
        )
        assert result.n_iter > 1
        if result.converged:
            assert result.n_iter >= 3


class TestConjugacyMeanLogLik:
    """E-step conjugacy ℓ_n matches mean log_prob (GIG-convention ψ)."""

    _MU = jnp.array([0.0, 0.2])
    _GAMMA = jnp.array([0.3, -0.1])
    _SIGMA = jnp.array([[1.0, 0.2], [0.2, 1.0]])
    _F = jnp.array([[0.8], [0.4]])
    _D = jnp.array([0.5, 0.6])

    @staticmethod
    def _models():
        from normix.distributions.normal_inverse_gamma import (
            FactorNormalInverseGamma, NormalInverseGamma,
        )
        from normix.distributions.normal_inverse_gaussian import (
            FactorNormalInverseGaussian, NormalInverseGaussian,
        )
        from normix.distributions.variance_gamma import FactorVarianceGamma
        from normix.distributions.generalized_hyperbolic import (
            FactorGeneralizedHyperbolic,
        )
        mu, g, S = (
            TestConjugacyMeanLogLik._MU,
            TestConjugacyMeanLogLik._GAMMA,
            TestConjugacyMeanLogLik._SIGMA,
        )
        F, D = TestConjugacyMeanLogLik._F, TestConjugacyMeanLogLik._D
        return [
            GeneralizedHyperbolic.from_classical(
                mu=mu, gamma=g, sigma=S, p=-0.5, a=1.5, b=1.0),
            VarianceGamma.from_classical(
                mu=mu, gamma=g, sigma=S, alpha=2.0, beta=1.5),
            NormalInverseGamma.from_classical(
                mu=mu, gamma=g, sigma=S, alpha=3.0, beta=1.0),
            NormalInverseGaussian.from_classical(
                mu=mu, gamma=g, sigma=S, mu_ig=1.0, lam=1.5),
            FactorGeneralizedHyperbolic.from_classical(
                mu=mu, gamma=g, F=F, D=D, p=-0.5, a=1.5, b=1.0),
            FactorVarianceGamma.from_classical(
                mu=mu, gamma=g, F=F, D=D, alpha=2.0, beta=1.5),
            FactorNormalInverseGamma.from_classical(
                mu=mu, gamma=g, F=F, D=D, alpha=3.0, beta=1.0),
            FactorNormalInverseGaussian.from_classical(
                mu=mu, gamma=g, F=F, D=D, mu_ig=1.0, lam=1.5),
        ]

    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    def test_estep_log_lik_matches_marginal(self, backend):
        rtol = 1e-8 if backend == "jax" else 1e-7
        atol = 1e-8 if backend == "jax" else 1e-7
        for model in self._models():
            X = model.rvs(80, seed=4)
            estep = model.e_step(X, backend=backend)
            mll = float(model.marginal_log_likelihood(X))
            np.testing.assert_allclose(
                float(estep.log_lik), mll, rtol=rtol, atol=atol,
                err_msg=f"{type(model).__name__} backend={backend}",
            )

    @pytest.mark.parametrize("loop", ["scan", "python"])
    def test_track_ll_length_equals_n_iter(self, loop):
        true = VarianceGamma.from_classical(
            mu=jnp.array([0.5]), gamma=jnp.array([0.3]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = true.rvs(400, seed=1)
        kwargs = dict(max_iter=20, tol=1e-3, verbose=0, track_ll=True)
        if loop == "scan":
            kwargs.update(e_step_backend="jax", m_step_backend="jax")
        else:
            kwargs.update(e_step_backend="cpu", m_step_backend="cpu")
        result = BatchEMFitter(**kwargs).fit(true, X)
        assert result.log_likelihoods is not None
        assert result.log_likelihoods.shape[0] == int(result.n_iter)
        assert result.param_changes.shape[0] == int(result.n_iter)

