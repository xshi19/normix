r"""
Shrinkage target builders for penalised EM.

The :class:`~normix.fitting.eta_rules.Shrinkage` combinator pulls the
running expectation parameters toward a prior :math:`\eta_0`. The helpers
below construct that prior from a fitted (or moment-initialised)
:class:`~normix.mixtures.marginal.NormalMixture`, optionally substituting
a custom dispersion :math:`\Sigma_0`.

Per :doc:`/theory/shrinkage` (Shrunk Sufficient Statistics), the
prior expectation parameters are

.. math::

    s_1 &= E[Y^{-1}\mid\theta_0],\quad
    s_2  = E[Y\mid\theta_0],\quad
    s_3  = E[\log Y\mid\theta_0] \\
    s_4 &= \mu_0 + \gamma_0\,E[Y\mid\theta_0] \\
    s_5 &= \mu_0\,E[Y^{-1}\mid\theta_0] + \gamma_0 \\
    s_6 &= \Sigma_0
           + \mu_0\mu_0^\top E[Y^{-1}\mid\theta_0]
           + \gamma_0\gamma_0^\top E[Y\mid\theta_0]
           + \mu_0\gamma_0^\top + \gamma_0\mu_0^\top.

The constructors return exact moments, including :math:`+\infty` for
nonexistent moments. Full-covariance models return a six-field
:class:`~normix.fitting.eta.NormalMixtureEta`. The Σ-only variants reuse
the model's own :math:`(\mu, \gamma, p, a, b)` to fill the other five
fields, keeping the public contract simple ("``eta0`` is always a full
prior") while the user's per-field ``tau`` selects which fields are
actually shrunk.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.distributions.gamma import Gamma
from normix.distributions.inverse_gamma import InverseGamma
from normix.fitting.eta import NormalMixtureEta
from normix.utils.constants import ALPHA_REFERENCE_MARGIN

if TYPE_CHECKING:  # pragma: no cover
    from normix.mixtures.marginal import MarginalMixture, NormalMixture


__all__ = [
    "eta0_from_model",
    "eta0_isotropic",
    "eta0_diagonal",
    "eta0_with_sigma",
]


def _em_reference_eta(model: "MarginalMixture") -> NormalMixtureEta:
    """Finite fitting statistics from the same family and normal parameters."""
    if isinstance(model.subordinator(), (Gamma, InverseGamma)):
        alpha = jnp.maximum(model.alpha, 1.0 + ALPHA_REFERENCE_MARGIN)
        model = eqx.tree_at(lambda m: m.alpha, model, alpha)
    return model.compute_eta_from_model()


def eta0_from_model(model: "MarginalMixture") -> NormalMixtureEta:
    r"""Prior :math:`\eta_0` equal to the model's current expectation parameters.

    Equivalent to ``model.compute_eta_from_model()``, including :math:`+\infty`
    for nonexistent moments.

    Parameters
    ----------
    model : MarginalMixture
        Source model; the prior reuses its :math:`(\mu, \gamma, \Sigma)`
        and subordinator parameters.

    Returns
    -------
    NormalMixtureEta
        Expectation pytree with an explicit origin and the model's statistic layout.
    """
    return model.compute_eta_from_model()


def eta0_with_sigma(
    model: "NormalMixture",
    Sigma0: jax.Array,
) -> NormalMixtureEta:
    r"""Prior :math:`\eta_0` reusing model parameters with a custom :math:`\Sigma_0`.

    Substitutes ``Sigma0`` for the model's covariance in the
    :math:`s_6` term while keeping :math:`\mu, \gamma` and the
    subordinator expectations from ``model``. This is the building block
    for "shrink Σ only" workflows: combine with a per-field ``tau`` that
    is non-zero only on ``E_XXT_inv_Y``.

    Parameters
    ----------
    model : NormalMixture
        Source model; provides :math:`(\mu, \gamma, p, a, b)`.
    Sigma0 : jax.Array
        Prior dispersion to embed in :math:`s_6`, shape :math:`(d, d)`.
        Must be positive semi-definite (not checked).

    Returns
    -------
    NormalMixtureEta
        Six-statistic expectation pytree with an explicit origin with ``E_XXT_inv_Y`` rebuilt from
        ``Sigma0``.
    """
    from normix.mixtures.marginal import NormalMixture
    if not isinstance(model, NormalMixture):
        raise TypeError("Dispersion-substitution targets require a full or univariate mixture")
    Sigma0 = jnp.asarray(Sigma0, dtype=jnp.float64)
    return model._prior_eta(Sigma0)


def eta0_isotropic(
    model: "NormalMixture",
    sigma2: float,
) -> NormalMixtureEta:
    r"""Prior with an isotropic dispersion :math:`\Sigma_0 = \sigma^2 I_d`.

    Parameters
    ----------
    model : NormalMixture
        Source model; provides :math:`(\mu, \gamma, p, a, b)`.
    sigma2 : float
        Common variance for the isotropic prior. Must be positive.

    Returns
    -------
    NormalMixtureEta
        Six-statistic expectation pytree with an explicit origin.
    """
    d = model.d
    Sigma0 = jnp.asarray(sigma2, dtype=jnp.float64) * jnp.eye(d, dtype=jnp.float64)
    return eta0_with_sigma(model, Sigma0)


def eta0_diagonal(
    model: "NormalMixture",
    diag: jax.Array,
) -> NormalMixtureEta:
    r"""Prior with a diagonal dispersion :math:`\Sigma_0 = \mathrm{diag}(\text{diag})`.

    Parameters
    ----------
    model : NormalMixture
        Source model; provides :math:`(\mu, \gamma, p, a, b)`.
    diag : jax.Array
        Diagonal entries of :math:`\Sigma_0`, shape :math:`(d,)`. Must be
        positive.

    Returns
    -------
    NormalMixtureEta
        Six-statistic expectation pytree with an explicit origin.
    """
    diag = jnp.asarray(diag, dtype=jnp.float64)
    Sigma0 = jnp.diag(diag)
    return eta0_with_sigma(model, Sigma0)
