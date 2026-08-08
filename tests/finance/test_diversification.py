r"""
Tests for ``normix.finance.diversification`` (Phase F: ENB).

Cover Meucci variance ENB and generalized (squared-risk) ENB:

- constrained minimum torsion yields :math:`T H T^\top = I` and
  :math:`\sum p = 1`;
- isotropic covariance + equal weights gives :math:`N = n`;
- single-asset weight gives :math:`N = 1`;
- variance path matches generalized path on a symmetric model
  (:math:`\gamma = 0`) to MC tolerance;
- heavy-tail / skew model splits CVaR-ENB below variance-ENB;
- PCA torsion differs from minimum torsion;
- zero weights and material indefiniteness yield NaN.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import NormalInverseGaussian, VarianceGamma
from normix.finance import (
    CVaR,
    GeneralizedENB,
    MinimumTorsion,
    PCATorsion,
    VarianceENB,
)


def _isotropic_nig(d: int = 4):
    return NormalInverseGaussian.from_classical(
        mu=jnp.zeros(d),
        gamma=jnp.zeros(d),
        sigma=jnp.eye(d),
        mu_ig=1.0,
        lam=1.5,
    )


def test_minimum_torsion_identity_and_simplex():
    Sigma = jnp.array(
        [[1.0, 0.3, 0.1],
         [0.3, 1.2, 0.2],
         [0.1, 0.2, 0.8]],
        dtype=jnp.float64,
    )
    dec = MinimumTorsion().decompose(Sigma)
    I = dec.T @ Sigma @ dec.T.T
    np.testing.assert_allclose(I, np.eye(3), atol=1e-10)
    assert bool(dec.valid)

    w = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float64)
    res = VarianceENB().evaluate_covariance(Sigma, w)
    np.testing.assert_allclose(float(jnp.sum(res.p)), 1.0, atol=1e-12)
    assert 1.0 - 1e-8 <= float(res.enb) <= 3.0 + 1e-8
    np.testing.assert_allclose(
        float(res.risk) ** 2, float(w @ Sigma @ w), rtol=1e-12,
    )


def test_isotropic_equal_weight_enb_is_n():
    d = 5
    model = _isotropic_nig(d)
    w = jnp.full(d, 1.0 / d)
    res = VarianceENB().evaluate(model, w)
    np.testing.assert_allclose(float(res.enb), d, rtol=1e-10)


def test_single_asset_enb_is_one():
    d = 4
    model = _isotropic_nig(d)
    w = jnp.zeros(d).at[0].set(1.0)
    res = VarianceENB().evaluate(model, w)
    np.testing.assert_allclose(float(res.enb), 1.0, rtol=1e-10)


def test_symmetric_model_cvar_matches_variance():
    """With gamma = 0 and mu = 0, CVaR is proportional to sigma-tilde."""
    d = 3
    model = NormalInverseGaussian.from_classical(
        mu=jnp.zeros(d),
        gamma=jnp.zeros(d),
        sigma=jnp.array(
            [[1.0, 0.4, 0.1],
             [0.4, 1.5, 0.2],
             [0.1, 0.2, 0.9]],
            dtype=jnp.float64,
        ),
        mu_ig=1.0,
        lam=2.0,
    )
    w = jnp.array([0.4, 0.35, 0.25], dtype=jnp.float64)
    Y = model.joint.subordinator().rvs(30_000, seed=0)
    res_var = VarianceENB().evaluate(model, w)
    res_cvar = GeneralizedENB(CVaR(0.05)).evaluate(model, w, Y)
    np.testing.assert_allclose(float(res_cvar.enb), float(res_var.enb), rtol=5e-2)
    np.testing.assert_allclose(np.asarray(res_cvar.p), np.asarray(res_var.p), atol=5e-2)


def test_heavy_tail_cvar_enb_below_variance():
    """Cov prop. I => variance ENB = n; CVaR loads on the skewed name."""
    E_Y, Var_Y = 1.0, 0.2
    g = -1.2
    beta = E_Y / Var_Y
    alpha = E_Y * beta
    s2 = jnp.ones(4).at[0].set(1.0 - Var_Y * g * g / E_Y)
    model = VarianceGamma.from_classical(
        mu=jnp.zeros(4),
        gamma=jnp.zeros(4).at[0].set(g),
        sigma=jnp.diag(s2),
        alpha=alpha,
        beta=beta,
    )
    cov = np.asarray(model.cov())
    # Off-diagonals ~0; diagonals equal.
    np.testing.assert_allclose(cov, np.eye(4) * cov[0, 0], atol=1e-12)

    w = jnp.full(4, 0.25)
    Y = model.joint.subordinator().rvs(40_000, seed=1)
    res_var = VarianceENB().evaluate(model, w)
    res_cvar = GeneralizedENB(CVaR(0.01)).evaluate(model, w, Y)

    np.testing.assert_allclose(float(res_var.enb), 4.0, rtol=1e-8)
    assert float(res_cvar.enb) < float(res_var.enb) - 0.15
    assert float(res_cvar.p[0]) > 0.25 + 0.1


def test_pca_differs_from_minimum_torsion():
    Sigma = jnp.array(
        [[1.0, 0.7, 0.5],
         [0.7, 1.0, 0.6],
         [0.5, 0.6, 1.0]],
        dtype=jnp.float64,
    )
    w = jnp.array([0.5, 0.3, 0.2], dtype=jnp.float64)
    mt = VarianceENB(torsion=MinimumTorsion()).evaluate_covariance(Sigma, w)
    pca = VarianceENB(torsion=PCATorsion()).evaluate_covariance(Sigma, w)
    assert abs(float(mt.enb) - float(pca.enb)) > 1e-6
    assert not np.allclose(np.asarray(mt.T), np.asarray(pca.T), atol=1e-6)


def test_zero_weights_nan():
    model = _isotropic_nig(3)
    res = VarianceENB().evaluate(model, jnp.zeros(3))
    assert np.isnan(float(res.enb))


def test_indefinite_matrix_nan():
    H = jnp.array(
        [[1.0, 0.0],
         [0.0, -1.0]],
        dtype=jnp.float64,
    )
    res = VarianceENB().evaluate_covariance(H, jnp.array([0.5, 0.5]))
    assert np.isnan(float(res.enb))
    assert not bool(MinimumTorsion().decompose(H).valid)


def test_variance_vmap_over_weights():
    model = _isotropic_nig(3)
    W = jnp.array(
        [[1.0, 0.0, 0.0],
         [1 / 3, 1 / 3, 1 / 3]],
        dtype=jnp.float64,
    )
    enb = VarianceENB()
    Ns = jax.vmap(lambda w: enb.evaluate(model, w).enb)(W)
    np.testing.assert_allclose(float(Ns[0]), 1.0, rtol=1e-10)
    np.testing.assert_allclose(float(Ns[1]), 3.0, rtol=1e-10)
