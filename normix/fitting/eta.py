"""
NormalMixtureEta — expectation parametrization for normal variance-mean mixtures.

The six fields are the batch-averaged sufficient statistics
:math:`\\hat\\eta = \\frac{1}{n}\\sum_i E[t(X_i, Y_i) \\mid X_i]`:

.. math::

    t(x, y) = [\\log y,\\; 1/y,\\; y,\\; x,\\; x/y,\\; \\mathrm{vec}(xx^\\top/y)]

This is the expectation parametrization of
:class:`~normix.mixtures.joint.JointNormalMixture`.
"""
from __future__ import annotations

import equinox as eqx
import jax


class NormalMixtureEta(eqx.Module):
    r"""Aggregated expectation parameters for normal variance-mean mixtures.

    Attributes
    ----------
    E_log_Y : scalar
        :math:`\frac{1}{n}\sum_i E[\log Y_i \mid X_i]`
    E_inv_Y : scalar
        :math:`\frac{1}{n}\sum_i E[1/Y_i \mid X_i]`
    E_Y : scalar
        :math:`\frac{1}{n}\sum_i E[Y_i \mid X_i]`
    E_X : (d,)
        :math:`\frac{1}{n}\sum_i X_i`
    E_X_inv_Y : (d,)
        :math:`\frac{1}{n}\sum_i X_i \, E[1/Y_i \mid X_i]`
    E_XXT_inv_Y : (d, d)
        :math:`\frac{1}{n}\sum_i X_i X_i^\top E[1/Y_i \mid X_i]`
    """

    E_log_Y: jax.Array
    E_inv_Y: jax.Array
    E_Y: jax.Array
    E_X: jax.Array
    E_X_inv_Y: jax.Array
    E_XXT_inv_Y: jax.Array
