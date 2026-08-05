"""
SP500 distribution validation tests.

Fits GH, NIG, Variance Gamma, and Normal Inverse Gamma to SP500 log returns
and validates convergence and goodness-of-fit.

Requires the data file data/sp500_sample.csv. Tests are skipped when the
data file is not present.
"""
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from normix.distributions.variance_gamma import VarianceGamma
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic

MAX_ITER = 100
EM_TOL = 1e-3


def _init_model(cls, X, **extra):
    d = X.shape[1]
    mu = jnp.mean(X, axis=0)
    sigma_emp = jnp.cov(X.T) + 1e-4 * jnp.eye(d)
    return cls.from_classical(mu=mu, gamma=jnp.zeros(d), sigma=sigma_emp, **extra)


@pytest.mark.parametrize("dist_name,extra", [
    ("VG", dict(alpha=2.0, beta=1.0)),
    ("NInvG", dict(alpha=3.0, beta=1.0)),
    ("NIG", dict(mu_ig=1.0, lam=1.0)),
])
def test_em_convergence(dist_name, extra, sp500_sample):
    """EM fitting should converge, improve LL, and stay finite."""
    X = sp500_sample
    cls_map = {"VG": VarianceGamma, "NInvG": NormalInverseGamma, "NIG": NormalInverseGaussian}
    model = _init_model(cls_map[dist_name], X, **extra)
    ll0 = float(model.marginal_log_likelihood(X))
    result = model.fit(X, max_iter=MAX_ITER, tol=EM_TOL, verbose=0,
                       e_step_backend='cpu', m_step_backend='cpu')
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"{dist_name}: non-finite LL={ll}"
    assert result.n_iter >= 1
    assert ll >= ll0 - 1e-6, (
        f"{dist_name}: LL did not improve ({ll0:.4f} → {ll:.4f})")
    assert jnp.all(jnp.isfinite(result.model.mu))
    assert jnp.all(jnp.isfinite(result.model.gamma))


def test_gh_em_convergence(sp500_sample):
    """GH EM with det_sigma_one should converge and improve LL."""
    X = sp500_sample
    model = _init_model(GeneralizedHyperbolic, X, p=-0.5, a=2.0, b=1.0)
    ll0 = float(model.marginal_log_likelihood(X))
    result = model.fit(X, max_iter=MAX_ITER, tol=EM_TOL, verbose=0,
                       regularization='det_sigma_one',
                       e_step_backend='cpu', m_step_backend='cpu')
    ll = float(result.model.marginal_log_likelihood(X))
    assert np.isfinite(ll), f"GH: non-finite LL={ll}"
    assert ll >= ll0 - 1e-6, f"GH: LL did not improve ({ll0:.4f} → {ll:.4f})"
    assert float(result.model._joint.a) > 0
    assert float(result.model._joint.b) > 0
