"""
NormalMixtureEta — expectation parametrization for normal variance-mean mixtures.

The six statistic fields are the batch-averaged sufficient statistics
:math:`\\hat\\eta = \\frac{1}{n}\\sum_i E[t(X_i, Y_i) \\mid X_i]`,
in the *theory order* used in :doc:`/theory/shrinkage` and
:doc:`/theory/factor_analysis`:

.. math::

    s_1 = E[Y^{-1}], \\;\\;
    s_2 = E[Y], \\;\\;
    s_3 = E[\\log Y], \\;\\;
    s_4 = E[X-o], \\;\\;
    s_5 = E[(X-o) / Y], \\;\\;
    s_6 = E[(X-o)(X-o)^\\top / Y].

Here :math:`o` is the explicit ``origin`` (zero by default).
This is the expectation parametrization in those coordinates of
:class:`~normix.mixtures.joint.JointNormalMixture`.
GIG arrays use the last-axis order ``(E_log_Y, E_inv_Y, E_Y)``.
The private packing helpers convert between these layouts by name.
"""
from __future__ import annotations

from typing import Any, Callable, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.utils.constants import MOMENT_CANCEL_ATOL


def _moment_origin(mu: jax.Array, E_inv_Y: jax.Array) -> jax.Array:
    """Choose an origin before forming location moments in float64."""
    finite = jnp.isfinite(E_inv_Y)
    cancel = jnp.finfo(jnp.float64).eps * jnp.where(finite, E_inv_Y, 0.0) * jnp.sum(mu ** 2)
    return jnp.where(~finite | (cancel > MOMENT_CANCEL_ATOL), mu, jnp.zeros_like(mu))


def _combine_split(eta_fin: jax.Array, m: jax.Array, v: jax.Array) -> jax.Array:
    r""":math:`\eta_{\mathrm{fin}} + m v`, with :math:`0\cdot\infty` masked off."""
    finite = jnp.isfinite(m)
    value = eta_fin + jnp.where(finite, m, 0.0) * v
    return jnp.where(finite | (v == 0), value, jnp.copysign(jnp.inf, v))

def _half_moments_exist(subordinator):
    r"""Existence of :math:`E[Y^{-1/2}]` and :math:`E[Y^{1/2}]` at the two GIG tails."""
    gig = subordinator.to_gig()
    return (gig.b > 0) | (gig.p > 0.5), (gig.a > 0) | (gig.p < -0.5)


def _prior_existence(mu, gamma, noise, E_inv_Y, E_Y, half_inv, half_y):
    r"""Absolute integrability of :math:`X, X/Y, XX^\top/Y`.

    At each tail a drift term needs the first moment and a noise term
    needs the half moment. The latter drops out on deterministic coordinates.
    """
    inv_ok, y_ok = jnp.isfinite(E_inv_Y), jnp.isfinite(E_Y)
    x = (y_ok | (gamma == 0)) & (half_y | ~noise)
    x_inv = (inv_ok | (mu == 0)) & (half_inv | ~noise)

    def scatter_exists(c, first, half):
        drift = c != 0
        drift_pair = jnp.outer(drift, drift)
        drift_noise = jnp.outer(drift, noise)
        return (first | ~drift_pair) & (half | ~(drift_noise | drift_noise.T))

    scatter = scatter_exists(mu, inv_ok, half_inv) & scatter_exists(gamma, y_ok, half_y)
    return x, x_inv, scatter


def _prior_location_moments(mu, gamma, sigma, E_inv_Y, E_Y, half_inv, half_y, *, group_cross):
    r"""Moments of :math:`X, X/Y, XX^\top/Y` for location relative to the given origin.

    ``group_cross`` preserves the summation order of full-covariance reconstruction.
    """
    x_ok, x_inv_ok, scatter_ok = _prior_existence(
        mu, gamma, jnp.diag(sigma) > 0, E_inv_Y, E_Y, half_inv, half_y)
    inv = jnp.where(jnp.isfinite(E_inv_Y), E_inv_Y, 0.0)
    y = jnp.where(jnp.isfinite(E_Y), E_Y, 0.0)
    scatter = sigma + jnp.outer(mu, mu) * inv + jnp.outer(gamma, gamma) * y
    cross = jnp.outer(mu, gamma)
    scatter = scatter + (cross + cross.T) if group_cross else scatter + cross + cross.T
    return (jnp.where(x_ok, mu + gamma * y, jnp.inf),
            jnp.where(x_inv_ok, mu * inv + gamma, jnp.inf),
            jnp.where(scatter_ok, scatter, jnp.inf))


def _checked_normal_params(eta, params, denom, *, schur=None):
    r"""Validate the input to the normal M-step and return its parameters.

    Allow a positive determinant deficit of :math:`8\epsilon\max(b^2,|ac|)`
    in :math:`b^2-ac`, where :math:`a,c` are the diagonal entries and
    :math:`b` is the off-diagonal. On a violation the larger term is
    :math:`b^2`, which equals one for the full normal block. Factor models
    also require a nonsingular Schur complement. Invalid concrete inputs
    raise ValueError and invalid traced inputs give NaN parameters.
    """
    D = denom if schur is None else 1.0 - eta.E_inv_Y * eta.E_Y
    rtol = 8 * jnp.finfo(jnp.asarray(D).dtype).eps
    valid = (eta.E_inv_Y > 0) & (eta.E_Y > 0) & jnp.isfinite(D) & (D <= rtol)
    if schur is not None:
        a, b, c = schur
        valid &= ((a > 0) & (c > 0) & jnp.isfinite(denom)
                  & (denom != 0) & (denom <= rtol * b ** 2))
    for leaf in jax.tree.leaves(eta):
        valid &= jnp.all(jnp.isfinite(leaf))
    if isinstance(valid, jax.core.Tracer):
        return jax.tree.map(lambda p: jnp.where(valid, p, jnp.nan), params)
    if not bool(valid):
        raise ValueError(
            "Infeasible normal M-step statistics. Expected finite moments, positive "
            "E[Y] and E[1/Y] with product >= 1, and a nonsingular factor block for FA."
        )
    return params


def _pack_gig_eta(*, E_log_Y: jax.Array, E_inv_Y: jax.Array, E_Y: jax.Array) -> jax.Array:
    """Pack named moments in GIG order along the last axis."""
    return jnp.stack((E_log_Y, E_inv_Y, E_Y), axis=-1)


def _unpack_gig_eta(eta: jax.Array) -> dict[str, jax.Array]:
    """Name the moments of a scalar or batched GIG expectation array."""
    return {"E_log_Y": eta[..., 0], "E_inv_Y": eta[..., 1], "E_Y": eta[..., 2]}


class EStepResult(eqx.Module):
    r"""E-step output: aggregated :math:`\eta` and mean log-likelihood.

    :math:`\ell_n` is *not* a field of :class:`NormalMixtureEta` —
    ``affine_combine`` / shrinkage would mix it. It is computed from the
    E-step's posterior GIG log-partition (already in
    ``BesselMoments.log_k``) via conjugacy, with no extra Bessel calls
    and no cache on the model.
    """

    eta: Any
    #: mean log-likelihood :math:`\ell_n(\theta)` at the E-step parameters,
    #: nats per observation
    log_lik: jax.Array


class NormalMixtureEta(eqx.Module):
    r"""Aggregated expectation parameters for normal variance-mean mixtures.

    Fields are stored in **theory order** ``(s_1, …, s_6)``: the first six
    statistics are shared with :class:`FactorMixtureStats` so that
    shrinkage targets, weights, and tests written for the standard family
    transfer unchanged.

    The location fields contain moments of :math:`X-\mathrm{origin}`.
    Here :math:`o=\mathrm{origin}`. The default origin is zero, so existing six-field constructors store
    raw moments. Use :meth:`rebase` to change coordinates and
    :func:`affine_combine` to combine statistics with different origins.
    """

    #: scalar; :math:`s_1 = \frac{1}{n}\sum_i E[1/Y_i \mid X_i]`
    E_inv_Y: jax.Array
    #: scalar; :math:`s_2 = \frac{1}{n}\sum_i E[Y_i \mid X_i]`
    E_Y: jax.Array
    #: scalar; :math:`s_3 = \frac{1}{n}\sum_i E[\log Y_i \mid X_i]`
    E_log_Y: jax.Array
    #: shape :math:`(d,)`; :math:`s_4 = \frac{1}{n}\sum_i (X_i-o)`
    E_X: jax.Array
    #: shape :math:`(d,)`; :math:`s_5 = \frac{1}{n}\sum_i (X_i-o) \, E[1/Y_i \mid X_i]`
    E_X_inv_Y: jax.Array
    #: shape :math:`(d, d)`; :math:`s_6 = \frac{1}{n}\sum_i (X_i-o)(X_i-o)^\top E[1/Y_i \mid X_i]`
    E_XXT_inv_Y: jax.Array

    #: Coordinate origin; shape :math:`(d,)`. It is not a sufficient statistic.
    origin: jax.Array = eqx.field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        self.origin = jnp.broadcast_to(jnp.asarray(self.origin, dtype=jnp.float64),
                                       jnp.shape(self.E_X))

    def rebase(self, origin: jax.Array) -> "NormalMixtureEta":
        r"""Express the same moments about ``origin``.

        For :math:`r=\mathrm{origin}_{new}-\mathrm{origin}_{old}`,
        :math:`s_4'=s_4-r`, :math:`s_5'=s_5-s_1r`, and
        :math:`s_6'=s_6-s_5r^\top-rs_5^\top+s_1rr^\top`.
        """
        origin = jnp.broadcast_to(jnp.asarray(origin, dtype=jnp.float64), self.origin.shape)
        r = origin - self.origin
        return eqx.tree_at(
            lambda e: (e.E_X, e.E_X_inv_Y, e.E_XXT_inv_Y, e.origin), self,
            (self.E_X - r, self.E_X_inv_Y - self.E_inv_Y * r,
             self.E_XXT_inv_Y - jnp.outer(self.E_X_inv_Y, r)
             - jnp.outer(r, self.E_X_inv_Y) + self.E_inv_Y * jnp.outer(r, r),
             origin),
        )


class FactorMixtureStats(NormalMixtureEta):
    r"""Aggregated expectation parameters for factor-analysis mixtures.

    Fields are stored in theory order ``(s_1, …, s_{10})`` from
    :doc:`/theory/factor_analysis`. The first six are identical to
    :class:`NormalMixtureEta` (so shrinkage targets, η-update rules, and
    weight pytrees designed for the standard family broadcast onto the
    factor family without modification). The four extra fields involve
    the latent factor :math:`Z`.
    """

    #: shape :math:`(d, r)`; :math:`s_7 = \frac{1}{n}\sum_i E[(X_i-o) Z_i^\top Y_i^{-1/2} \mid X_i]`
    E_XZT_inv_sqrtY: jax.Array
    #: shape :math:`(r,)`; :math:`s_8 = \frac{1}{n}\sum_i E[Z_i Y_i^{-1/2} \mid X_i]`
    E_Z_inv_sqrtY: jax.Array
    #: shape :math:`(r,)`; :math:`s_9 = \frac{1}{n}\sum_i E[Z_i Y_i^{1/2} \mid X_i]`
    E_Z_sqrtY: jax.Array
    #: shape :math:`(r, r)`; :math:`s_{10} = \frac{1}{n}\sum_i E[Z_i Z_i^\top \mid X_i]`
    E_ZZT: jax.Array

    def rebase(self, origin: jax.Array) -> "FactorMixtureStats":
        r"""Translate the location moments and the X–Z cross moment."""
        rebased = super().rebase(origin)
        return eqx.tree_at(
            lambda e: e.E_XZT_inv_sqrtY, rebased,
            self.E_XZT_inv_sqrtY
            - jnp.outer(rebased.origin - self.origin, self.E_Z_inv_sqrtY),
        )


# ---------------------------------------------------------------------------
# Generalised affine combination
# ---------------------------------------------------------------------------

Weight = Union[float, jax.Array, "NormalMixtureEta", Callable[..., "NormalMixtureEta"]]


def _apply(weight: Weight, eta):
    r"""Apply a weight (scalar / pytree / callable) to an eta pytree.

    See :func:`affine_combine` for the contract on ``weight``.
    """
    if callable(weight):
        return weight(eta)
    if isinstance(weight, type(eta)):
        return jax.tree.map(jnp.multiply, weight, eta)
    return jax.tree.map(lambda x: weight * x, eta)


def affine_combine(
    eta_prev,
    eta_new,
    b: Weight,
    c: Weight,
    a=None,
):
    r"""Affine combination :math:`\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta`.

    The weights ``b`` and ``c`` may be:

    - **scalar** (Python number or 0-d ``jax.Array``) — broadcast to every
      leaf of ``eta``;
    - **stats-shape pytree** (same type as ``eta_prev`` / ``eta_new``) —
      block-diagonal weighting; leaf-wise multiply;
    - **callable** ``η → η`` — arbitrary linear operator on η (e.g. an
      ``eqx.nn.Linear`` wrapped to operate on a flattened pytree).

    The shift ``a`` is either ``None`` (zero) or a stats-shape pytree.
    ``eta_prev`` is rebased to ``eta_new.origin`` before combining, and the
    result keeps that origin. Weights and the shift are not coordinates:
    their ``origin`` leaves are ignored, and ``a`` is added field by field
    in ``eta_new``'s coordinates.

    Parameters
    ----------
    eta_prev :
        Running state :math:`\eta_{t-1}`.
    eta_new :
        New batch estimate :math:`\hat\eta`.
    b :
        Weight on previous state.
    c :
        Weight on new estimate.
    a :
        Additive shift (e.g. shrinkage prior). ``None`` means zero.
    """
    if isinstance(eta_new, NormalMixtureEta):
        eta_prev = eta_prev.rebase(eta_new.origin)
    out = jax.tree.map(jnp.add, _apply(b, eta_prev), _apply(c, eta_new))
    if a is not None:
        out = jax.tree.map(jnp.add, out, a)
    if isinstance(eta_new, NormalMixtureEta):
        out = eqx.tree_at(lambda e: e.origin, out, eta_new.origin)
    return out
