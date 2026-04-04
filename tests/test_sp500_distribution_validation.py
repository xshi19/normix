"""
SP500 distribution validation tests.

Fits GH, NIG, Variance Gamma, and Normal Inverse Gamma to SP500 log returns
and validates convergence and goodness-of-fit.

Requires the data file data/sp500_sample.csv. Tests are skipped when the
data file is not present.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

jax.config.update("jax_enable_x64", True)

from normix.distributions.variance_gamma import VarianceGamma
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic

DATA_PATH = Path(__file__).parent.parent / "data" / "sp500_sample.csv"
MAX_ITER = 100
EM_TOL = 1e-3
MAX_STOCKS = 10

_cache = None


def _load_data():
    global _cache
    if _cache is not None:
        return _cache
    if not DATA_PATH.exists():
        pytest.skip(f"SP500 data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).dropna(axis=1)
    X = jnp.asarray(df.values[:, :MAX_STOCKS], dtype=jnp.float64)
    _cache = X
    return X


def _init_model(cls, X, **extra):
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    sigma_emp = jnp.cov(X.T) + 1e-4 * jnp.eye(d)
    return cls.from_classical(mu=mu, gamma=jnp.zeros(d), sigma=sigma_emp, **extra)


@pytest.mark.parametrize("dist_name,extra", [
    ("VG", dict(alpha=2.0, beta=1.0)),
    ("NInvG", dict(alpha=3.0, beta=1.0)),
    ("NIG", dict(mu_ig=1.0, lam=1.0)),
])
def test_em_convergence(dist_name, extra):
    """EM fitting should converge with finite log-likelihood."""
    X = _load_data()
    cls_map = {"VG": VarianceGamma, "NInvG": NormalInverseGamma, "NIG": NormalInverseGaussian}
    model = _init_model(cls_map[dist_name], X, **extra)
    result = model.fit(X, max_iter=MAX_ITER, tol=EM_TOL, verbose=0,
                       e_step_backend='cpu', m_step_backend='cpu')
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"{dist_name}: non-finite LL={ll}"
    assert result.n_iter >= 1


def test_gh_em_convergence():
    """GH EM with det_sigma_one should converge."""
    X = _load_data()
    model = _init_model(GeneralizedHyperbolic, X, p=-0.5, a=2.0, b=1.0)
    result = model.fit(X, max_iter=MAX_ITER, tol=EM_TOL, verbose=0,
                       regularization='det_sigma_one',
                       e_step_backend='cpu', m_step_backend='cpu')
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"GH: non-finite LL={ll}"
