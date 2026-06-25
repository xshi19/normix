"""Posterior-GIG consolidation regression tests (R1 + R2; plan T3).

After R2 the eight per-family ``_posterior_gig_params`` overrides were
replaced by one base implementation per hierarchy
(``JointNormalMixture``, ``FactorNormalMixture``) routed through
``subordinator().to_gig()``; R1 collapsed the ``B_POST_FLOOR`` application
into a single ``_floored_posterior_gig_params`` chokepoint per hierarchy.

These tests pin the refactor:

1. **Map equivalence** — the consolidated ``_posterior_gig_params`` returns
   exactly the analytic per-family formula the deleted overrides encoded,
   for all four families × {joint, factor} (``rtol=1e-12``).
2. **Dormancy** — for GH / NIG / NInvG (prior ``b > 0``) the floor does not
   bind, so ``_floored_posterior_gig_params`` is identical to the unfloored
   map and ``conditional_expectations(x=mu)`` equals the unfloored GIG
   moments exactly.
3. **Backend parity** — ``e_step`` outputs match between the ``jax`` and
   ``cpu`` backends for every family and both hierarchies on fixed seeds.

See ``dev-notes/archive/design/em_robustness_followups.md`` §3.3.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from normix import (
    FactorGeneralizedHyperbolic,
    FactorNormalInverseGamma,
    FactorNormalInverseGaussian,
    FactorVarianceGamma,
    GeneralizedHyperbolic,
    NormalInverseGamma,
    NormalInverseGaussian,
    VarianceGamma,
)
from normix.distributions.generalized_inverse_gaussian import GIG
from normix.utils.constants import B_POST_FLOOR


# ---------------------------------------------------------------------------
# Builders: matched joint (via marginal) and factor models per family.
# Σ = F Fᵀ + diag(D) so the joint and factor instances share the same prior.
# ---------------------------------------------------------------------------

D_DIM = 3
_MU = jnp.array([0.3, -0.2, 0.5])
_GAMMA = jnp.array([0.4, -0.1, 0.2])
_F = jnp.array([[0.6], [0.2], [-0.3]])
_D = jnp.array([0.5, 0.8, 0.4])
_SIGMA = _F @ _F.T + jnp.diag(_D)

# (marginal_cls, factor_cls, subordinator kwargs, expected GIG embedding (p,a,b))
_FAMILIES = {
    "VG": (
        VarianceGamma, FactorVarianceGamma,
        dict(alpha=2.3, beta=1.4),
        lambda k: (k["alpha"], 2.0 * k["beta"], 0.0),
    ),
    "NInvG": (
        NormalInverseGamma, FactorNormalInverseGamma,
        dict(alpha=2.1, beta=1.7),
        lambda k: (-k["alpha"], 0.0, 2.0 * k["beta"]),
    ),
    "NIG": (
        NormalInverseGaussian, FactorNormalInverseGaussian,
        dict(mu_ig=1.2, lam=2.5),
        lambda k: (-0.5, k["lam"] / k["mu_ig"] ** 2, k["lam"]),
    ),
    "GH": (
        GeneralizedHyperbolic, FactorGeneralizedHyperbolic,
        dict(p=0.7, a=1.6, b=2.2),
        lambda k: (k["p"], k["a"], k["b"]),
    ),
}


def _make_joint(name):
    marg_cls, _, sub, _ = _FAMILIES[name]
    return marg_cls.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, **sub)._joint


def _make_factor(name):
    _, fac_cls, sub, _ = _FAMILIES[name]
    return fac_cls.from_classical(mu=_MU, gamma=_GAMMA, F=_F, D=_D, **sub)


@pytest.mark.parametrize("name", list(_FAMILIES))
@pytest.mark.parametrize("hierarchy", ["joint", "factor"])
def test_posterior_gig_map_matches_analytic_formula(name, hierarchy):
    """R2 parity: the consolidated map equals the per-family formula
    ``(p_gig - d/2, a_gig + w2, b_gig + z2)`` the deleted overrides encoded."""
    model = _make_joint(name) if hierarchy == "joint" else _make_factor(name)
    _, _, sub, embed = _FAMILIES[name]
    p_gig, a_gig, b_gig = embed(sub)

    # The map is pure in (z2, w2): feed arbitrary positive scalars.
    z2, w2 = 1.37, 0.83
    p_post, a_post, b_post = model._posterior_gig_params(z2, w2)

    np.testing.assert_allclose(float(p_post), p_gig - D_DIM / 2.0, rtol=1e-12)
    np.testing.assert_allclose(float(a_post), a_gig + w2, rtol=1e-12)
    np.testing.assert_allclose(float(b_post), b_gig + z2, rtol=1e-12)


@pytest.mark.parametrize("name", list(_FAMILIES))
@pytest.mark.parametrize("hierarchy", ["joint", "factor"])
def test_floor_binds_only_for_vg(name, hierarchy):
    """R1 dormancy: at the mode (z2=0) the floor changes b_post only for VG
    (prior b=0); for GH/NIG/NInvG the prior b>0 leaves it untouched."""
    model = _make_joint(name) if hierarchy == "joint" else _make_factor(name)
    w2 = 0.83
    _, _, b_unfloored = model._posterior_gig_params(0.0, w2)
    _, _, b_floored = model._floored_posterior_gig_params(0.0, w2)

    if name == "VG":
        assert float(b_unfloored) == pytest.approx(0.0, abs=0.0)
        assert float(b_floored) == pytest.approx(B_POST_FLOOR)
    else:
        assert float(b_unfloored) > B_POST_FLOOR
        np.testing.assert_allclose(
            float(b_floored), float(b_unfloored), rtol=1e-12)


@pytest.mark.parametrize("name", ["NInvG", "NIG", "GH"])
def test_conditional_expectations_at_mode_unfloored(name):
    """Dormancy: ``conditional_expectations(x=mu)`` equals the GIG moments at
    the *unfloored* posterior params for the families whose prior b>0."""
    joint = _make_joint(name)
    cond = joint.conditional_expectations(_MU)

    # At x=mu, z2=0; floor is dormant, so the reference uses the raw map.
    z, w, z2, w2, zw = joint._quad_forms(_MU)
    p_post, a_post, b_post = joint._posterior_gig_params(z2, w2)
    ref = GIG(p=p_post, a=a_post, b=b_post).expectation_params()

    np.testing.assert_allclose(float(cond["E_log_Y"]), float(ref[0]), rtol=1e-12)
    np.testing.assert_allclose(float(cond["E_inv_Y"]), float(ref[1]), rtol=1e-12)
    np.testing.assert_allclose(float(cond["E_Y"]), float(ref[2]), rtol=1e-12)


def _data_with_mode(seed):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(12, D_DIM)) * 0.7 + np.asarray(_MU))
    # Prepend an exact-mode observation so the VG floor is exercised.
    return jnp.concatenate([_MU[None, :], X], axis=0)


@pytest.mark.parametrize("name", list(_FAMILIES))
def test_estep_marginal_backend_parity(name):
    """Both E-step backends route through the same floored chokepoint, so
    marginal e_step must agree between jax and cpu."""
    marg_cls, _, sub, _ = _FAMILIES[name]
    model = marg_cls.from_classical(mu=_MU, gamma=_GAMMA, sigma=_SIGMA, **sub)
    X = _data_with_mode(seed=0)
    eta_jax = model.e_step(X, backend="jax")
    eta_cpu = model.e_step(X, backend="cpu")
    # The jax and cpu GIG-Bessel paths differ by O(1e-9) near the VG floor
    # (large E[1/Y|x]); cross-backend agreement to rtol=1e-7 is the guarantee.
    for a, b in zip(jax.tree_util.tree_leaves(eta_jax),
                    jax.tree_util.tree_leaves(eta_cpu)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("name", list(_FAMILIES))
def test_estep_factor_backend_parity(name):
    """Same parity guarantee for the factor hierarchy."""
    model = _make_factor(name)
    X = _data_with_mode(seed=1)
    eta_jax = model.e_step(X, backend="jax")
    eta_cpu = model.e_step(X, backend="cpu")
    for a, b in zip(jax.tree_util.tree_leaves(eta_jax),
                    jax.tree_util.tree_leaves(eta_cpu)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-7, atol=1e-8)
