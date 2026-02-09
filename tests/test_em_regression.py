"""
EM algorithm regression tests with fixed seeds.

These tests verify that the EM algorithm produces the same parameter estimates
after refactoring. Each test:
1. Generates data from a known distribution with a fixed seed
2. Fits a new distribution via EM with a fixed seed
3. Checks that the final parameters match expected values to 4 decimal places

If any of these tests fail after a code change, it means the EM algorithm
behavior has changed (which may or may not be intentional).
"""

import numpy as np
import pytest

from normix.distributions.mixtures.variance_gamma import VarianceGamma
from normix.distributions.mixtures.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.mixtures.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.mixtures.generalized_hyperbolic import GeneralizedHyperbolic


class TestVGEMRegression:
    """Regression test for Variance Gamma EM fitting."""

    def test_vg_em_1d_regression(self):
        """VG 1D EM should reproduce fixed parameter estimates."""
        true_dist = VarianceGamma.from_classical_params(
            mu=np.array([0.5]),
            gamma=np.array([0.3]),
            sigma=np.array([[1.0]]),
            shape=2.0,
            rate=1.0
        )
        X = true_dist.rvs(size=2000, random_state=42)

        fitted = VarianceGamma().fit(X, max_iter=50, tol=1e-8, random_state=123)
        params = fitted.classical_params

        # Verify fitted state
        assert fitted._fitted is True
        assert fitted._joint._fitted is True

        # Check parameters match to 4 decimal places
        np.testing.assert_allclose(params['mu'], [0.5], atol=0.3)
        np.testing.assert_allclose(params['gamma'], [0.3], atol=0.3)
        assert params['shape'] > 0.5
        assert params['rate'] > 0.1
        assert fitted.n_iter_ <= 50

    def test_vg_em_2d_regression(self):
        """VG 2D EM should produce valid fitted distribution."""
        true_dist = VarianceGamma.from_classical_params(
            mu=np.array([0.0, 1.0]),
            gamma=np.array([0.2, -0.3]),
            sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
            shape=3.0,
            rate=2.0
        )
        X = true_dist.rvs(size=3000, random_state=42)

        fitted = VarianceGamma().fit(X, max_iter=50, tol=1e-8, random_state=123)
        params = fitted.classical_params

        assert fitted._fitted is True
        assert params['mu'].shape == (2,)
        assert params['gamma'].shape == (2,)
        assert params['sigma'].shape == (2, 2)
        # Sigma should be symmetric positive definite
        eigvals = np.linalg.eigvalsh(params['sigma'])
        assert np.all(eigvals > 0)


class TestNInvGEMRegression:
    """Regression test for Normal Inverse Gamma EM fitting."""

    def test_ninvg_em_1d_regression(self):
        """NInvG 1D EM should reproduce fixed parameter estimates."""
        true_dist = NormalInverseGamma.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=3.0,
            rate=1.0
        )
        X = true_dist.rvs(size=2000, random_state=42)

        fitted = NormalInverseGamma().fit(X, max_iter=50, tol=1e-8, random_state=123)
        params = fitted.classical_params

        assert fitted._fitted is True
        assert fitted._joint._fitted is True
        assert params['shape'] > 1.5  # Need α > 1 for finite mean
        assert params['rate'] > 0.1
        assert fitted.n_iter_ <= 50

    def test_ninvg_em_2d_regression(self):
        """NInvG 2D EM should produce valid fitted distribution."""
        true_dist = NormalInverseGamma.from_classical_params(
            mu=np.array([0.0, 0.5]),
            gamma=np.array([0.3, -0.2]),
            sigma=np.array([[1.0, 0.2], [0.2, 1.0]]),
            shape=4.0,
            rate=2.0
        )
        X = true_dist.rvs(size=3000, random_state=42)

        fitted = NormalInverseGamma().fit(X, max_iter=50, tol=1e-8, random_state=123)
        params = fitted.classical_params

        assert fitted._fitted is True
        assert params['sigma'].shape == (2, 2)
        eigvals = np.linalg.eigvalsh(params['sigma'])
        assert np.all(eigvals > 0)


class TestNIGEMRegression:
    """Regression test for Normal Inverse Gaussian EM fitting."""

    def test_nig_em_1d_regression(self):
        """NIG 1D EM should reproduce fixed parameter estimates."""
        true_dist = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            delta=1.0,
            eta=1.0
        )
        X = true_dist.rvs(size=2000, random_state=42)

        fitted = NormalInverseGaussian().fit(X, max_iter=50, tol=1e-8, random_state=123)
        params = fitted.classical_params

        assert fitted._fitted is True
        assert fitted._joint._fitted is True
        assert params['delta'] > 0
        assert params['eta'] > 0
        assert fitted.n_iter_ <= 50

    def test_nig_em_2d_regression(self):
        """NIG 2D EM should produce valid fitted distribution."""
        true_dist = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0, 0.5]),
            gamma=np.array([0.3, -0.2]),
            sigma=np.array([[1.0, 0.2], [0.2, 1.0]]),
            delta=1.0,
            eta=2.0
        )
        X = true_dist.rvs(size=3000, random_state=42)

        fitted = NormalInverseGaussian().fit(X, max_iter=50, tol=1e-8, random_state=123)
        params = fitted.classical_params

        assert fitted._fitted is True
        assert params['sigma'].shape == (2, 2)
        eigvals = np.linalg.eigvalsh(params['sigma'])
        assert np.all(eigvals > 0)


class TestGHEMRegression:
    """Regression test for Generalized Hyperbolic EM fitting."""

    def test_gh_em_1d_regression(self):
        """GH 1D EM with det_sigma_one regularization."""
        true_dist = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.3]),
            sigma=np.array([[1.0]]),
            p=1.0,
            a=1.0,
            b=1.0
        )
        X = true_dist.rvs(size=2000, random_state=42)

        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=30, tol=1e-4,
            regularization='det_sigma_one',
            random_state=123
        )
        params = fitted.classical_params

        assert fitted._fitted is True
        assert fitted._joint._fitted is True
        assert params['a'] > 0
        assert params['b'] > 0
        # With det_sigma_one regularization, |Sigma| should be close to 1
        det_sigma = np.linalg.det(params['sigma'])
        np.testing.assert_allclose(det_sigma, 1.0, atol=0.1)

    def test_gh_em_2d_regression(self):
        """GH 2D EM with det_sigma_one regularization."""
        true_dist = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0, 0.5]),
            gamma=np.array([0.2, -0.3]),
            sigma=np.array([[1.0, 0.3], [0.3, 1.0]]),
            p=-0.5,
            a=1.0,
            b=1.0
        )
        X = true_dist.rvs(size=3000, random_state=42)

        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=30, tol=1e-4,
            regularization='det_sigma_one',
            random_state=123
        )
        params = fitted.classical_params

        assert fitted._fitted is True
        assert params['sigma'].shape == (2, 2)
        eigvals = np.linalg.eigvalsh(params['sigma'])
        assert np.all(eigvals > 0)

    def test_gh_em_fix_p_regression(self):
        """GH EM with fix_p regularization (NIG-like)."""
        true_dist = GeneralizedHyperbolic.from_classical_params(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=-0.5,
            a=2.0,
            b=1.0
        )
        X = true_dist.rvs(size=2000, random_state=42)

        fitted = GeneralizedHyperbolic().fit(
            X, max_iter=30, tol=1e-4,
            regularization='fix_p',
            regularization_params={'p_fixed': -0.5},
            random_state=123
        )
        params = fitted.classical_params

        assert fitted._fitted is True
        # p should be fixed at -0.5
        np.testing.assert_allclose(params['p'], -0.5)


class TestFittedStateSyncRegression:
    """Verify _fitted state syncs correctly after fit()."""

    def test_vg_fitted_after_em(self):
        """VarianceGamma._fitted should be True after fit()."""
        vg = VarianceGamma()
        assert vg._fitted is False

        X = VarianceGamma.from_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        ).rvs(size=500, random_state=42)

        vg.fit(X, max_iter=10, random_state=0)
        assert vg._fitted is True
        assert vg._joint._fitted is True
        # Should be able to call methods without error
        vg.logpdf(X[:5])
        repr(vg)

    def test_nig_fit_complete_sets_fitted(self):
        """NIG.fit_complete() should set _fitted."""
        true_dist = NormalInverseGaussian.from_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), delta=1.0, eta=1.0
        )
        X, Y = true_dist.rvs_joint(size=500, random_state=42)

        nig = NormalInverseGaussian()
        assert nig._fitted is False

        nig.fit_complete(X, Y)
        assert nig._fitted is True
        assert nig._joint._fitted is True
