r"""
Shrinkage target builders for penalised EM.

The :class:`~normix.fitting.eta_rules.Shrinkage` combinator pulls the
running expectation parameters toward a prior :math:`\eta_0`. The helpers
below construct that prior from a fitted (or moment-initialised)
:class:`~normix.mixtures.marginal.NormalMixture`, optionally substituting
a custom dispersion :math:`\Sigma_0`.

Per :doc:`../docs/theory/shrinkage` (Shrunk Sufficient Statistics), the
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

All four constructors return a complete six-field
:class:`~normix.fitting.eta.NormalMixtureEta`. The Σ-only variants reuse
the model's own :math:`(\mu, \gamma, p, a, b)` to fill the other five
fields, keeping the public contract simple ("``eta0`` is always a full
prior") while the user's per-field ``tau`` selects which fields are
actually shrunk.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from normix.fitting.eta import NormalMixtureEta

if TYPE_CHECKING:  # pragma: no cover
    from normix.mixtures.marginal import NormalMixture


__all__ = [
    "eta0_from_model",
    "eta0_isotropic",
    "eta0_diagonal",
    "eta0_with_sigma",
]


def eta0_from_model(model: "NormalMixture") -> NormalMixtureEta:
    r"""Prior :math:`\eta_0` equal to the model's current expectation parameters.

    Equivalent to ``model.compute_eta_from_model()``.

    Parameters
    ----------
    model : NormalMixture
        Source model; the prior reuses its :math:`(\mu, \gamma, \Sigma)`
        and subordinator parameters.

    Returns
    -------
    NormalMixtureEta
        Six-field expectation pytree.
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
    Sigma0 : (d, d) jax.Array
        Prior dispersion to embed in :math:`s_6`. Must be positive
        semi-definite (not checked).

    Returns
    -------
    NormalMixtureEta
        Six-field expectation pytree with ``E_XXT_inv_Y`` rebuilt from
        ``Sigma0``.
    """
    Sigma0 = jnp.asarray(Sigma0, dtype=jnp.float64)
    j = model._joint
    mu, gamma = j.mu, j.gamma

    E_log_Y, E_inv_Y, E_Y = model._subordinator_expectations()

    return NormalMixtureEta(
        E_inv_Y=E_inv_Y,
        E_Y=E_Y,
        E_log_Y=E_log_Y,
        E_X=mu + gamma * E_Y,
        E_X_inv_Y=mu * E_inv_Y + gamma,
        E_XXT_inv_Y=(Sigma0
                     + jnp.outer(mu, mu) * E_inv_Y
                     + jnp.outer(gamma, gamma) * E_Y
                     + jnp.outer(mu, gamma)
                     + jnp.outer(gamma, mu)),
    )


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
        Six-field expectation pytree.
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
    diag : (d,) jax.Array
        Diagonal entries of :math:`\Sigma_0`. Must be positive.

    Returns
    -------
    NormalMixtureEta
        Six-field expectation pytree.
    """
    diag = jnp.asarray(diag, dtype=jnp.float64)
    Sigma0 = jnp.diag(diag)
    return eta0_with_sigma(model, Sigma0)
