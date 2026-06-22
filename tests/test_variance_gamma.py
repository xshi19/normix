"""
Tests for Variance Gamma distributions (current API).

Covers joint and marginal construction, rvs, log_prob, mean/cov,
conditional expectations, and EM fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions import JointVarianceGamma, VarianceGamma


def _joint_1d(mu=0.0, gamma=0.5, sigma=1.0, alpha=2.0, beta=1.0):
    return JointVarianceGamma.from_classical(
        mu=jnp.array([mu]), gamma=jnp.array([gamma]),
        sigma=jnp.array([[sigma]]), alpha=alpha, beta=beta,
    )


def _joint_2d():
    return JointVarianceGamma.from_classical(
        mu=jnp.array([0.0, 1.0]),
        gamma=jnp.array([0.5, -0.3]),
        sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        alpha=2.0, beta=1.0,
    )


# ============================================================
# Joint VG Tests
# ============================================================

class TestJointVarianceGamma:

    def test_construction_attributes(self):
        j = _joint_1d()
        assert j.d == 1
        np.testing.assert_allclose(np.array(j.mu), [0.0])
        np.testing.assert_allclose(np.array(j.gamma), [0.5])
        np.testing.assert_allclose(float(j.alpha), 2.0)
        np.testing.assert_allclose(float(j.beta), 1.0)

    def test_natural_params_finite(self):
        j = _joint_1d()
        theta = j.natural_params()
        assert jnp.all(jnp.isfinite(theta))

    def test_log_prob_joint_finite(self):
        j = _joint_1d()
        x = jnp.array([0.5])
        y = jnp.array(1.0)
        lp = float(j.log_prob_joint(x, y))
        assert np.isfinite(lp)

    def test_rvs_shapes(self):
        j = _joint_1d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 1)
        assert Y.shape == (100,)

    def test_rvs_2d_shapes(self):
        j = _joint_2d()
        X, Y = j.rvs(100, seed=42)
        assert X.shape == (100, 2)
        assert Y.shape == (100,)

    @pytest.mark.parametrize("alpha,beta", [(2.0, 1.0), (3.0, 0.5)])
    def test_rvs_y_mean(self, alpha, beta):
        """Y ~ Gamma(alpha, beta), so E[Y] = alpha/beta."""
        j = JointVarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), alpha=alpha, beta=beta,
        )
        _, Y = j.rvs(20000, seed=42)
        np.testing.assert_allclose(float(Y.mean()), alpha / beta, rtol=0.05)

    def test_log_prob_joint_positive_density(self):
        j = _joint_2d()
        X, Y = j.rvs(50, seed=42)
        lps = jax.vmap(lambda xy: j.log_prob_joint(xy[:2], xy[2]))(
            jnp.column_stack([X, Y[:, None]]))
        assert jnp.all(jnp.isfinite(lps))

    def test_sufficient_statistics_shape(self):
        j = _joint_2d()
        x = jnp.array([0.5, -0.3])
        y = jnp.array(0.8)
        t = j.sufficient_statistics(jnp.concatenate([x, jnp.array([y])]))
        assert t.shape[0] > 0
        assert jnp.all(jnp.isfinite(t))

    def test_log_prob_vs_log_prob_joint(self):
        """EF log_prob(concat(x,[y])) must agree with log_prob_joint(x,y)."""
        j = _joint_2d()
        x = jnp.array([0.3, -0.2])
        y = jnp.array(0.8)
        xy = jnp.concatenate([x, jnp.array([y])])
        np.testing.assert_allclose(
            float(j.log_prob(xy)), float(j.log_prob_joint(x, y)), rtol=1e-10)


# ============================================================
# Marginal VG Tests
# ============================================================

class TestVarianceGamma:

    def test_from_classical(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        assert vg._joint.d == 1

    def test_rvs_shape_1d(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = vg.rvs(100, seed=42)
        assert X.shape == (100, 1)

    def test_rvs_shape_2d(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(100, seed=42)
        assert X.shape == (100, 2)

    def test_log_prob_finite(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_mean_formula(self):
        """E[X] = μ + γ·E[Y] = μ + γ·α/β."""
        alpha, beta = 2.0, 1.0
        mu = jnp.array([0.5])
        gamma = jnp.array([0.3])
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=jnp.array([[1.0]]),
            alpha=alpha, beta=beta,
        )
        expected = mu + gamma * (alpha / beta)
        np.testing.assert_allclose(np.array(vg.mean()), np.array(expected), rtol=1e-10)

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sample_mean_matches_analytical(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.5, -0.3]),
            gamma=jnp.array([0.3, 0.1]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(30000, seed=42)
        sample_mean = np.array(X.mean(axis=0))
        analytical_mean = np.array(vg.mean())
        np.testing.assert_allclose(sample_mean, analytical_mean, rtol=0.05, atol=0.05)

    @pytest.mark.slow
    @pytest.mark.stress
    def test_sample_cov_matches_analytical(self):
        """Cov[X] = E[Y]·Σ + Var[Y]·γγ^T."""
        alpha, beta = 2.0, 1.0
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.5, -0.3]),
            sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
            alpha=alpha, beta=beta,
        )
        X = vg.rvs(50000, seed=42)
        sample_cov = np.cov(np.array(X), rowvar=False)
        analytical_cov = np.array(vg.cov())
        np.testing.assert_allclose(sample_cov, analytical_cov, rtol=0.1)


# ============================================================
# Conditional Expectations
# ============================================================

class TestConditionalExpectationsVG:

    def test_conditional_expectations_finite(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=2.0, beta=1.0,
        )
        X = vg.rvs(50, seed=42)
        for i in range(5):
            cond = vg._joint.conditional_expectations(X[i])
            assert np.isfinite(float(cond['E_Y']))
            assert np.isfinite(float(cond['E_inv_Y']))
            assert np.isfinite(float(cond['E_log_Y']))

    def test_conditional_expectations_positive(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.3, -0.2]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            alpha=2.0, beta=1.0,
        )
        X = vg.rvs(50, seed=42)
        for i in range(5):
            cond = vg._joint.conditional_expectations(X[i])
            assert float(cond['E_Y']) > 0
            assert float(cond['E_inv_Y']) > 0


# ============================================================
# Inverse-moment singularity / overflow fix
# ============================================================

class TestInverseMomentSingularityVG:
    r"""Regression tests for the E[1/Y|x] overflow.

    For VG the posterior scale is b_post = (x-mu)^T Sigma^{-1} (x-mu), which
    -> 0 for observations at the mode. When alpha <= d/2 + 1 the conditional
    inverse moment E[1/Y|x] diverges there and the covariance M-step overflows
    to nan. The E-step floors b_post at B_POST_FLOOR to keep it finite.
    See dev-notes/tech_notes/vg_em_inverse_moment_singularity.md.
    """

    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    def test_estep_finite_at_mode_small_alpha(self, backend):
        # alpha = 0.7 < d/2 + 1 = 1.5 (d=1): without the floor, the exact-mode
        # observation x = mu gives b_post = 0 and E[1/Y|x] = inf.
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.3]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[2.0]]), alpha=0.7, beta=1.0,
        )
        X = jnp.array([[0.3], [0.3 + 1e-9], [1.0], [-0.5]])  # first two near/at mode
        eta = vg.e_step(X, backend=backend)
        for leaf in jax.tree_util.tree_leaves(eta):
            assert jnp.all(jnp.isfinite(leaf))

    def test_conditional_expectations_finite_at_mode(self):
        # x = mu exactly, symmetric, small alpha.
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), alpha=0.6, beta=1.0,
        )
        cond = vg._joint.conditional_expectations(jnp.array([0.0]))
        assert np.isfinite(float(cond['E_inv_Y'])) and float(cond['E_inv_Y']) > 0
        assert np.isfinite(float(cond['E_Y'])) and float(cond['E_Y']) > 0
        assert np.isfinite(float(cond['E_log_Y']))

    @pytest.mark.parametrize(
        "alpha, nu_branch, expected",
        [
            # 0 < nu < 1 branch: nu = alpha - d/2 = 0.2 at alpha=0.7, d=1.
            # Asymptotic Gamma(1-nu)/Gamma(nu) * 2^(1-2nu) * a_post^nu * b_min^(nu-1).
            (0.7, "0<nu<1", 2.99e4),
            # nu < 0 branch: nu = -0.3 at alpha=0.2, d=1.
            # Asymptotic 2|nu|/b_min = (d - 2 alpha)/b_min.
            (0.2, "nu<0", 6.11e5),
        ],
    )
    def test_inverse_moment_cap_value(self, alpha, nu_branch, expected):
        r"""T2: the *value* of the capped E[1/Y|x=mu], not just isfinite.

        With gamma=0, Sigma=1, beta=1 (d=1) the posterior at x=mu is
        GIG(p_post = alpha - 1/2, a_post = 2, b_post = B_POST_FLOOR), so the
        floored inverse moment must match the exact Bessel value at b_min=1e-6
        and the section-2 small-omega asymptotics. An isfinite-only test cannot
        catch a regression in the floor constant; this one can.
        """
        from scipy.special import kve, gamma as gamma_fn

        b_min = 1e-6  # hardcoded: a floor regression must break this test
        a_post = 2.0
        p_post = alpha - 0.5

        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.3]), gamma=jnp.array([0.0]),
            sigma=jnp.array([[1.0]]), alpha=alpha, beta=1.0,
        )
        cond = vg._joint.conditional_expectations(jnp.array([0.3]))
        e_inv_y = float(cond['E_inv_Y'])

        # Exact GIG inverse moment at the floor (scipy reference).
        omega = np.sqrt(a_post * b_min)
        exact = np.sqrt(a_post / b_min) * kve(p_post - 1, omega) / kve(p_post, omega)
        assert e_inv_y == pytest.approx(float(exact), rel=1e-3)
        assert e_inv_y == pytest.approx(expected, rel=0.05)

        # Section-2 asymptotics: the cap is uniform in a_post / linear in d.
        if nu_branch == "0<nu<1":
            nu = p_post
            asymptotic = (gamma_fn(1 - nu) / gamma_fn(nu)
                          * 2.0 ** (1 - 2 * nu)
                          * a_post ** nu * b_min ** (nu - 1))
            assert e_inv_y == pytest.approx(asymptotic, rel=0.1)
        else:
            asymptotic = (1.0 - 2.0 * alpha) / b_min  # (d - 2 alpha)/b_min, d=1
            assert e_inv_y == pytest.approx(asymptotic, rel=0.05)

    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    def test_em_no_overflow_heavy_peaked(self, backend):
        # Heavy-peaked data from a small-alpha VG drives the EM into the
        # alpha < 1.5 regime where E[1/Y|x] previously overflowed to nan.
        vg_true = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.2]),
            sigma=jnp.array([[1.0]]), alpha=0.7, beta=1.0,
        )
        X = vg_true.rvs(3000, seed=0).reshape(-1, 1)
        # Guarantee a near-mode observation is present throughout EM.
        X = jnp.concatenate([jnp.mean(X, axis=0, keepdims=True), X], axis=0)
        res = VarianceGamma.default_init(X).fit(
            X, max_iter=100, tol=1e-4, e_step_backend=backend, verbose=1)
        assert res.log_likelihoods is not None
        assert jnp.all(jnp.isfinite(res.log_likelihoods))
        j = res.model._joint
        for leaf in (j.mu, j.gamma, j.L_Sigma, j.alpha, j.beta):
            assert jnp.all(jnp.isfinite(leaf))

    @staticmethod
    def _high_dim_vg_data(d, n=1500, seed=0):
        """VG(α=1) data in d dimensions with the sample mean appended as a
        near-mode observation (drives EM into the α ≤ d/2+1 divergent regime)."""
        rng = np.random.default_rng(seed)
        mu = jnp.asarray(rng.normal(size=d))
        gamma = jnp.asarray(rng.normal(size=d) * 0.2)
        A = rng.normal(size=(d, d))
        sigma = jnp.asarray(A @ A.T + d * np.eye(d))
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=1.0, beta=1.0)
        X = vg.rvs(n, seed=seed)
        return jnp.concatenate([jnp.mean(X, axis=0, keepdims=True), X], axis=0)

    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    @pytest.mark.parametrize("d", [5, 10])
    def test_em_no_overflow_high_dim(self, d, backend):
        r"""T1: high-d VG EM stays finite and non-decreasing in LL.

        With the default init α≈2 and threshold α ≤ d/2+1, every d ≥ 2 starts
        in the divergent regime, so the near-mode observation would overflow
        E[1/Y|x] without the b_post floor. Existing coverage was d=1 only.
        """
        X = self._high_dim_vg_data(d, seed=d)
        init = VarianceGamma.default_init(X)
        ll_init = float(init.marginal_log_likelihood(X))
        res = init.fit(X, max_iter=80, tol=1e-4,
                       e_step_backend=backend, verbose=1)
        assert res.log_likelihoods is not None
        assert jnp.all(jnp.isfinite(res.log_likelihoods))
        j = res.model._joint
        for leaf in (j.mu, j.gamma, j.L_Sigma, j.alpha, j.beta):
            assert jnp.all(jnp.isfinite(leaf))
        ll_final = float(res.model.marginal_log_likelihood(X))
        assert ll_final >= ll_init - 1e-6

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    def test_em_no_overflow_high_dim_d50(self, backend):
        """T1 (slow): the same guarantee at d=50, deep in the divergent regime."""
        X = self._high_dim_vg_data(50, n=2000, seed=50)
        init = VarianceGamma.default_init(X)
        ll_init = float(init.marginal_log_likelihood(X))
        res = init.fit(X, max_iter=60, tol=1e-4,
                       e_step_backend=backend, verbose=1)
        assert res.log_likelihoods is not None
        assert jnp.all(jnp.isfinite(res.log_likelihoods))
        j = res.model._joint
        for leaf in (j.mu, j.gamma, j.L_Sigma, j.alpha, j.beta):
            assert jnp.all(jnp.isfinite(leaf))
        assert float(res.model.marginal_log_likelihood(X)) >= ll_init - 1e-6

    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    def test_mcecm_no_overflow_heavy_peaked(self, backend):
        r"""T4: MCECM on heavy-peaked small-alpha VG stays finite.

        MCECM (E -> M_normal -> E -> M_sub) runs a *second* E-step between the
        normal and subordinator M-steps, so it re-evaluates E[1/Y|x] on the
        updated normal block. Combined with the b_post floor (E-step) and the
        B2 prior-moment floor (subordinator warm start), every iterate must
        stay finite on alpha_true = 0.7 data with a near-mode observation.
        """
        vg_true = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.2]),
            sigma=jnp.array([[1.0]]), alpha=0.7, beta=1.0,
        )
        X = vg_true.rvs(2000, seed=1).reshape(-1, 1)
        X = jnp.concatenate([jnp.mean(X, axis=0, keepdims=True), X], axis=0)
        res = VarianceGamma.default_init(X).fit(
            X, algorithm='mcecm', max_iter=50, tol=1e-4,
            e_step_backend=backend, m_step_backend=backend,
            regularization='none', verbose=1)
        assert res.log_likelihoods is not None
        assert jnp.all(jnp.isfinite(res.log_likelihoods))
        j = res.model._joint
        for leaf in (j.mu, j.gamma, j.L_Sigma, j.alpha, j.beta):
            assert jnp.all(jnp.isfinite(leaf))


class TestAlphaMinBound:
    r"""F4: the opt-in ``alpha_min`` shape bound (VG likelihood-boundedness).

    The b_post floor keeps EM finite but does not stop α drifting below the
    degeneracy threshold (density unbounded at α ≤ d/2). ``alpha_min`` is the
    ghyp "fix-λ" analogue: it restricts the estimand so that cannot happen.
    """

    @staticmethod
    def _heavy_peaked_data(d, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        mu = jnp.asarray(rng.normal(size=d))
        gamma = jnp.asarray(rng.normal(size=d) * 0.2)
        A = rng.normal(size=(d, d))
        sigma = jnp.asarray(A @ A.T + d * np.eye(d))
        vg = VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=0.7, beta=1.0)
        X = vg.rvs(n, seed=seed)
        return jnp.concatenate([jnp.mean(X, axis=0, keepdims=True), X], axis=0)

    def test_default_none_is_no_op(self):
        """alpha_min=None must leave the fitted α bit-identical to the
        unspecified default (no behavior change)."""
        X = self._heavy_peaked_data(2, seed=2)
        res_default = VarianceGamma.default_init(X).fit(X, max_iter=40, tol=1e-4)
        res_none = VarianceGamma.default_init(X).fit(
            X, max_iter=40, tol=1e-4, alpha_min=None)
        assert float(res_default.model.alpha) == float(res_none.model.alpha)

    @pytest.mark.parametrize("backend", ["jax", "cpu"])
    def test_density_sentinel_keeps_alpha_above_half_d(self, backend):
        r"""``alpha_min='density'`` clamps α ≥ d/2 + ε, so the fitted VG
        density is bounded at x=μ (α > d/2)."""
        from normix.utils.constants import ALPHA_MIN_MARGIN
        d = 2
        X = self._heavy_peaked_data(d, seed=3)
        # Unconstrained: α drifts into the unbounded-density regime α ≤ d/2.
        res_free = VarianceGamma.default_init(X).fit(
            X, max_iter=60, tol=1e-4, e_step_backend=backend)
        assert float(res_free.model.alpha) < d / 2.0
        # Constrained: α pinned at the density-boundedness floor.
        res_bnd = VarianceGamma.default_init(X).fit(
            X, max_iter=60, tol=1e-4, e_step_backend=backend,
            alpha_min='density')
        floor = d / 2.0 + ALPHA_MIN_MARGIN
        assert float(res_bnd.model.alpha) >= floor - 1e-9
        # Density finite at the fitted mode.
        f_mode = float(res_bnd.model.pdf(res_bnd.model._joint.mu))
        assert np.isfinite(f_mode)

    def test_absolute_float_bound(self):
        """An absolute float alpha_min is honored directly."""
        X = self._heavy_peaked_data(2, seed=4)
        res = VarianceGamma.default_init(X).fit(
            X, max_iter=60, tol=1e-4, alpha_min=3.0)
        assert float(res.model.alpha) >= 3.0 - 1e-9

    def test_inverse_moment_sentinel(self):
        r"""``alpha_min='inverse_moment'`` clamps α ≥ d/2 + 1 + ε."""
        from normix.utils.constants import ALPHA_MIN_MARGIN
        d = 2
        X = self._heavy_peaked_data(d, seed=5)
        res = VarianceGamma.default_init(X).fit(
            X, max_iter=60, tol=1e-4, alpha_min='inverse_moment')
        assert float(res.model.alpha) >= d / 2.0 + 1.0 + ALPHA_MIN_MARGIN - 1e-9

    def test_invalid_sentinel_raises(self):
        X = self._heavy_peaked_data(1, seed=6)
        with pytest.raises(ValueError, match="alpha_min"):
            VarianceGamma.default_init(X).fit(X, max_iter=2, alpha_min='nonsense')


# ============================================================
# Edge Cases
# ============================================================

class TestVarianceGammaEdgeCases:

    def test_symmetric_case(self):
        """gamma=0 (symmetric) should produce finite log_prob."""
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0, 0.0]),
            gamma=jnp.array([0.0, 0.0]),
            sigma=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            alpha=2.5, beta=1.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))

    def test_large_alpha(self):
        vg = VarianceGamma.from_classical(
            mu=jnp.array([0.0]), gamma=jnp.array([0.5]),
            sigma=jnp.array([[1.0]]), alpha=20.0, beta=10.0,
        )
        X = vg.rvs(20, seed=42)
        lps = jax.vmap(vg.log_prob)(X)
        assert jnp.all(jnp.isfinite(lps))
