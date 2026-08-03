"""
Unit tests for previously untested utilities (roadmap T4).

Covers ``gammaincinv``, ``build_pinv_table`` / ``rvs_pinv``, and a basic
``quantile_cmc`` consistency check.  The Rényi-gradient edge case (B2) and
the CMC bracket regression (B7) live with their fixes in
``test_varentropy.py`` and ``finance/test_cvar.py``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special, stats

from normix import Gamma, UnivariateVarianceGamma
from normix.finance._mc import cdf_cmc_raw, quantile_cmc
from normix.utils.gammainc import gammaincinv
from normix.utils.rvs import build_pinv_table, rvs_pinv


# ---------------------------------------------------------------------------
# gammaincinv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 20.0])
@pytest.mark.parametrize("q", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_gammaincinv_matches_scipy(a, q):
    ours = float(gammaincinv(jnp.asarray(a), jnp.asarray(q)))
    ref = float(special.gammaincinv(a, q))
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10)


def test_gammaincinv_vmap_and_jit():
    a = jnp.array([1.0, 2.0, 5.0])
    q = jnp.array([0.1, 0.5, 0.9])
    fn = jax.jit(jax.vmap(gammaincinv))
    ours = np.asarray(fn(a, q))
    ref = special.gammaincinv(np.asarray(a), np.asarray(q))
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# build_pinv_table / rvs_pinv
# ---------------------------------------------------------------------------


def test_build_pinv_table_gamma_ppf_matches_scipy():
    """PINV table for Gamma(2,1) recovers scipy ppf on common quantiles."""
    g = Gamma(alpha=2.0, beta=1.0)

    def log_kernel(w):
        x = jnp.exp(w)
        return g.log_prob(x) + w

    mode_w = jnp.log(jnp.asarray(1.0))  # mode of Gamma(2,1) is 1
    u_grid, x_grid = build_pinv_table(log_kernel, mode_w, x_of_w=jnp.exp)
    qs = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
    ours = jnp.interp(qs, u_grid, x_grid)
    ref = stats.gamma.ppf(np.asarray(qs), a=2.0, scale=1.0)
    np.testing.assert_allclose(np.asarray(ours), ref, rtol=1e-3, atol=1e-4)


def test_rvs_pinv_sample_moments():
    """Samples from a PINV table of N(0,1) have plausible mean/var."""
    def log_kernel(w):
        return -0.5 * w * w - 0.5 * jnp.log(2.0 * jnp.pi)

    u_grid, x_grid = build_pinv_table(log_kernel, jnp.asarray(0.0), n_grid=2000)
    key = jax.random.PRNGKey(0)
    samples = np.asarray(rvs_pinv(key, u_grid, x_grid, 50_000))
    np.testing.assert_allclose(samples.mean(), 0.0, atol=0.02)
    np.testing.assert_allclose(samples.var(), 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# quantile_cmc (basic consistency; bracket edge case is B7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
def test_quantile_cmc_inverts_cmc_cdf(q):
    uv = UnivariateVarianceGamma.from_classical(
        mu=0.0, gamma=0.2, sigma=1.0, alpha=2.0, beta=1.0,
    )
    Y = uv.subordinator.rvs(2000, seed=1)
    x = float(quantile_cmc(uv, q, Y))
    F = float(cdf_cmc_raw(
        x, uv._mu_scalar, uv._gamma_scalar, uv._sigma_scalar, Y))
    np.testing.assert_allclose(F, q, atol=1e-6)
