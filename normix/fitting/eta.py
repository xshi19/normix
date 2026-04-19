"""
NormalMixtureEta — expectation parametrization for normal variance-mean mixtures.

The six fields are the batch-averaged sufficient statistics
:math:`\\hat\\eta = \\frac{1}{n}\\sum_i E[t(X_i, Y_i) \\mid X_i]`,
in the *theory order* used in :doc:`../docs/theory/shrinkage` and
:doc:`../docs/theory/factor_analysis`:

.. math::

    s_1 = E[Y^{-1}], \\;\\;
    s_2 = E[Y], \\;\\;
    s_3 = E[\\log Y], \\;\\;
    s_4 = E[X], \\;\\;
    s_5 = E[X / Y], \\;\\;
    s_6 = E[X X^\\top / Y].

This is the expectation parametrization of
:class:`~normix.mixtures.joint.JointNormalMixture`.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp


class NormalMixtureEta(eqx.Module):
    r"""Aggregated expectation parameters for normal variance-mean mixtures.

    Fields are stored in **theory order** ``(s_1, …, s_6)``: the first six
    statistics are shared with :class:`FactorMixtureStats` so that
    shrinkage targets, weights, and tests written for the standard family
    transfer unchanged.

    Attributes
    ----------
    E_inv_Y : scalar
        :math:`s_1 = \frac{1}{n}\sum_i E[1/Y_i \mid X_i]`
    E_Y : scalar
        :math:`s_2 = \frac{1}{n}\sum_i E[Y_i \mid X_i]`
    E_log_Y : scalar
        :math:`s_3 = \frac{1}{n}\sum_i E[\log Y_i \mid X_i]`
    E_X : (d,)
        :math:`s_4 = \frac{1}{n}\sum_i X_i`
    E_X_inv_Y : (d,)
        :math:`s_5 = \frac{1}{n}\sum_i X_i \, E[1/Y_i \mid X_i]`
    E_XXT_inv_Y : (d, d)
        :math:`s_6 = \frac{1}{n}\sum_i X_i X_i^\top E[1/Y_i \mid X_i]`
    """

    E_inv_Y: jax.Array
    E_Y: jax.Array
    E_log_Y: jax.Array
    E_X: jax.Array
    E_X_inv_Y: jax.Array
    E_XXT_inv_Y: jax.Array


# ---------------------------------------------------------------------------
# Generalised affine combination
# ---------------------------------------------------------------------------

# A weight on a stats pytree may be:
#   - a scalar (Python float / int / 0-d jax.Array) — broadcast to every leaf;
#   - a stats-shape pytree (e.g. NormalMixtureEta) — leaf-wise multiply;
#   - a callable ``η → η`` — arbitrary linear map.
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
    out = jax.tree.map(jnp.add, _apply(b, eta_prev), _apply(c, eta_new))
    if a is not None:
        out = jax.tree.map(jnp.add, out, a)
    return out
