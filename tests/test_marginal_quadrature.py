"""
Contract: multivariate marginal ``log_prob`` vs trapezoidal joint integration.

For each GH-family mixture, the closed-form marginal density must match
``log ∫ f(x, y) dy`` on a log-``y`` grid (Jacobian ``e^w``), including at
``x = μ`` where the joint concentrates as ``y → 0``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.distributions.normal_inverse_gamma import NormalInverseGamma
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.variance_gamma import VarianceGamma

pytestmark = pytest.mark.contract

_N_GRID = 5000
_LOG_Y_LO = -20.0
_LOG_Y_HI = 12.0
_ATOL = 1e-8


def _models_d2():
    mu = jnp.array([0.1, -0.2])
    gamma = jnp.array([0.3, -0.1])
    sigma = jnp.array([[1.0, 0.3], [0.3, 0.8]])
    return {
        "VG": VarianceGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=2.0, beta=1.0),
        "NInvG": NormalInverseGamma.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, alpha=3.0, beta=1.0),
        "NIG": NormalInverseGaussian.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, mu_ig=1.0, lam=1.0),
        "GH": GeneralizedHyperbolic.from_classical(
            mu=mu, gamma=gamma, sigma=sigma, p=-0.5, a=2.0, b=1.0),
    }


def _log_marginal_quad(joint, x: jax.Array) -> float:
    """``log ∫ f(x,y) dy`` via trapezoid on ``w = log y``."""
    w = jnp.linspace(_LOG_Y_LO, _LOG_Y_HI, _N_GRID)
    y = jnp.exp(w)
    log_f = jax.vmap(lambda yi: joint.log_prob_joint(x, yi))(y)
    log_integrand = log_f + w
    m = jnp.max(log_integrand)
    f = jnp.exp(log_integrand - m)
    dw = w[1] - w[0]
    integ = dw * (0.5 * f[0] + jnp.sum(f[1:-1]) + 0.5 * f[-1])
    return float(m + jnp.log(integ))


@pytest.mark.parametrize("dist_name", ["VG", "NInvG", "NIG", "GH"])
@pytest.mark.parametrize("x_kind", ["mu", "offset"])
def test_marginal_log_prob_matches_joint_quadrature(dist_name, x_kind):
    model = _models_d2()[dist_name]
    x = model.mu if x_kind == "mu" else model.mu + jnp.array([0.5, -0.3])
    lp = float(model.log_prob(x))
    lq = _log_marginal_quad(model.joint, x)
    assert abs(lp - lq) <= _ATOL, (
        f"{dist_name}/{x_kind}: |log_prob − quad| = {abs(lp - lq):.3e} > {_ATOL}"
    )
