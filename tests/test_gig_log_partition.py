"""N2: GIG log-partition small-z predicate.

The small-z form is used iff ``z < GIG_DEGEN_THRESHOLD`` and
``|p| log(2/z) > BESSEL_QUAD_LOG_DROP``, with Gamma vs InverseGamma by
``sign(p)``. Off-domain ``θ`` returns ``+∞``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gammaln, kve

from normix.distributions.generalized_inverse_gaussian import GIG
from normix.utils.constants import (
    BESSEL_QUAD_LOG_DROP,
    GIG_DEGEN_THRESHOLD,
    LOG_EPS,
    TINY,
)

pytestmark = pytest.mark.contract

RTOL = 1e-10
ATOL = 1e-8

# Review cases plus the symmetric invalid InverseGamma point and a valid
# Gamma-limit control. Pass bar: JAX and CPU ψ match the interior formula.
_REVIEW_CASES = [
    (-2.0, 1.0, 1e-22),
    (0.0, 1.0, 1e-22),
    (2.0, 1e-22, 1.0),
    (2.0, 1.0, 1e-22),
]


def _interior_psi(p: float, a: float, b: float) -> float:
    """Independent interior GIG ψ via SciPy ``kve``."""
    z = math.sqrt(a * b)
    log_k = math.log(float(kve(p, z))) - z
    return math.log(2.0) + log_k + 0.5 * p * (math.log(b) - math.log(a))


def _jax_cpu_psi(p: float, a: float, b: float) -> tuple[float, float]:
    gig = GIG(p, a, b)
    theta = np.asarray(gig.natural_params(), dtype=np.float64)
    psi_jax = float(gig.log_partition())
    psi_cpu = float(type(gig)._log_partition_cpu(theta))
    return psi_jax, psi_cpu


class TestGIGReviewNormalization:
    """The four verification-script cases against the interior formula."""

    @pytest.mark.parametrize("p,a,b", _REVIEW_CASES)
    def test_jax_cpu_match_interior(self, p, a, b):
        psi_ref = _interior_psi(p, a, b)
        psi_jax, psi_cpu = _jax_cpu_psi(p, a, b)
        assert_allclose(psi_jax, psi_ref, rtol=RTOL, atol=ATOL)
        assert_allclose(psi_cpu, psi_ref, rtol=RTOL, atol=ATOL)

    def test_wrong_sign_is_not_log_eps_floor(self):
        """GIG(-2, 1, 1e-22) must not return the LOG_EPS / TINY Gamma floors."""
        psi_jax, psi_cpu = _jax_cpu_psi(-2.0, 1.0, 1e-22)
        floor_jax = float(gammaln(LOG_EPS) - LOG_EPS * math.log(0.5))
        floor_cpu = float(gammaln(TINY) - TINY * math.log(0.5))
        assert abs(psi_jax - floor_jax) > 1.0
        assert abs(psi_cpu - floor_cpu) > 1.0


class TestGIGLimitTruncationGrid:
    """(p, z) points on both sides of ``|p| log(2/z) = C``."""

    @pytest.mark.parametrize("p,z", [
        # Kernel: |p| log(2/z) < C at z = 1e-10
        (0.02, 1e-10),
        (0.05, 1e-10),
        (0.5, 1e-10),
        (0.0, 1e-10),
        (-0.5, 1e-10),
        # Limit: z < 1e-10 and |p| log(2/z) > 40
        (2.0, 1e-11),
        (-2.0, 1e-11),
        (1.0, 1e-20),
        (-1.0, 1e-20),
    ])
    def test_matches_interior(self, p, z):
        psi_ref = _interior_psi(p, z, z)
        psi_jax, psi_cpu = _jax_cpu_psi(p, z, z)
        assert_allclose(psi_jax, psi_ref, rtol=RTOL, atol=ATOL)
        assert_allclose(psi_cpu, psi_ref, rtol=RTOL, atol=ATOL)

    def test_small_p_does_not_use_gamma_form(self):
        """At p=0.02, z=1e-10 the Gamma one-term form is O(1) wrong."""
        p, z = 0.02, 1e-10
        psi_jax, _ = _jax_cpu_psi(p, z, z)
        psi_gamma = float(gammaln(p) - p * math.log(z / 2.0))
        psi_ref = _interior_psi(p, z, z)
        assert abs(psi_gamma - psi_ref) > 0.1
        assert abs(psi_jax - psi_ref) < 1e-8

    def test_limit_boundary_uses_threshold_constants(self):
        """The predicate is z < z_max and |p| log(2/z) > C, not z alone."""
        z = GIG_DEGEN_THRESHOLD
        log_2z = math.log(2.0 / z)
        p_star = BESSEL_QUAD_LOG_DROP / log_2z
        assert p_star > 1.0
        # Just below C: kernel (small |p|)
        p_lo = 0.5 * p_star
        assert abs(p_lo) * log_2z < BESSEL_QUAD_LOG_DROP
        psi_jax, psi_cpu = _jax_cpu_psi(p_lo, z, z)
        psi_ref = _interior_psi(p_lo, z, z)
        assert_allclose(psi_jax, psi_ref, rtol=RTOL, atol=ATOL)
        assert_allclose(psi_cpu, psi_ref, rtol=RTOL, atol=ATOL)


class TestGIGOffDomain:
    """θ ∉ Θ ⇒ ψ = +∞ (Bregman line search must not accept a clamp)."""

    @pytest.mark.parametrize("p,a,b", [
        (-2.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    ])
    def test_log_partition_is_inf(self, p, a, b):
        gig = GIG(p, a, b)
        psi_jax = float(gig.log_partition())
        psi_cpu = float(type(gig)._log_partition_cpu(
            np.asarray(gig.natural_params(), dtype=np.float64),
        ))
        assert np.isinf(psi_jax) and psi_jax > 0.0
        assert np.isinf(psi_cpu) and psi_cpu > 0.0

    def test_valid_gamma_boundary_is_finite(self):
        gig = GIG(2.0, 3.0, 0.0)
        psi = float(gig.log_partition())
        psi_gamma = float(gammaln(2.0) - 2.0 * math.log(1.5))
        assert np.isfinite(psi)
        assert_allclose(psi, psi_gamma, rtol=RTOL, atol=ATOL)

    def test_valid_invgamma_boundary_is_finite(self):
        gig = GIG(-2.0, 0.0, 3.0)
        psi = float(gig.log_partition())
        psi_ig = float(gammaln(2.0) - 2.0 * math.log(1.5))
        assert np.isfinite(psi)
        assert_allclose(psi, psi_ig, rtol=RTOL, atol=ATOL)
