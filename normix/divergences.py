"""
Statistical divergences for exponential family distributions.

**Tier 1 — functional core** (pure JAX, maximum composability):

.. math::

    H^2(p, q) = 1 - \\exp\\!\\left(\\psi\\!\\left(\\frac{\\theta_p + \\theta_q}{2}\\right)
    - \\frac{\\psi(\\theta_p) + \\psi(\\theta_q)}{2}\\right)

.. math::

    D_{\\mathrm{KL}}(p \\| q) = \\psi(\\theta_q) - \\psi(\\theta_p)
    - (\\theta_q - \\theta_p)^\\top \\nabla\\psi(\\theta_p)

**Tier 3 — module convenience** (delegates to Tier 2 instance methods):

``squared_hellinger(p, q)`` and ``kl_divergence(p, q)`` accept
:class:`~normix.exponential_family.ExponentialFamily` or
:class:`~normix.mixtures.marginal.NormalMixture` objects.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp



# ------------------------------------------------------------------
# Tier 1: Functional core — pure functions of (psi, theta_p, theta_q)
# ------------------------------------------------------------------

def squared_hellinger_from_psi(
    psi: Callable[[jax.Array], jax.Array],
    theta_p: jax.Array,
    theta_q: jax.Array,
) -> jax.Array:
    r"""Squared Hellinger distance from the log-partition function alone.

    .. math::

        H^2(p, q) = 1 - \exp\!\left(\psi\!\left(\frac{\theta_p + \theta_q}{2}\right)
        - \frac{\psi(\theta_p) + \psi(\theta_q)}{2}\right)

    Parameters
    ----------
    psi : callable
        Log-partition function :math:`\psi(\theta) \to \mathbb{R}`.
    theta_p, theta_q : jax.Array
        Natural parameter vectors.
    """
    theta_mid = 0.5 * (theta_p + theta_q)
    log_affinity = psi(theta_mid) - 0.5 * (psi(theta_p) + psi(theta_q))
    return jnp.clip(1.0 - jnp.exp(log_affinity), 0.0, 1.0)


def kl_divergence_from_psi(
    psi: Callable[[jax.Array], jax.Array],
    grad_psi: Callable[[jax.Array], jax.Array],
    theta_p: jax.Array,
    theta_q: jax.Array,
) -> jax.Array:
    r"""KL divergence :math:`D_{\mathrm{KL}}(p \| q)` as a Bregman divergence of :math:`\psi`.

    .. math::

        D_{\mathrm{KL}}(p \| q) = \psi(\theta_q) - \psi(\theta_p)
        - (\theta_q - \theta_p)^\top \nabla\psi(\theta_p)

    Parameters
    ----------
    psi : callable
        Log-partition function.
    grad_psi : callable
        Gradient :math:`\nabla\psi(\theta)`.
    theta_p, theta_q : jax.Array
        Natural parameter vectors.
    """
    eta_p = grad_psi(theta_p)
    return psi(theta_q) - psi(theta_p) - jnp.dot(theta_q - theta_p, eta_p)


# ------------------------------------------------------------------
# Tier 3: Module convenience — delegates to Tier 2 instance methods
# ------------------------------------------------------------------

def squared_hellinger(p, q) -> jax.Array:
    r"""Squared Hellinger distance between two distributions.

    For :class:`~normix.exponential_family.ExponentialFamily` objects, calls
    ``p.squared_hellinger(q)`` (Tier 2), which defaults to the general
    :math:`\psi`-based formula and can be overridden by subclasses.

    For :class:`~normix.mixtures.marginal.NormalMixture` objects, delegates to
    the joint distributions as an upper-bound approximation.

    Parameters
    ----------
    p, q : ExponentialFamily or NormalMixture
        Must be the same type (or at least share the same log-partition).
    """
    from normix.mixtures.marginal import NormalMixture

    if isinstance(p, NormalMixture):
        return p.joint.squared_hellinger(q.joint)
    return p.squared_hellinger(q)


def kl_divergence(p, q) -> jax.Array:
    r"""KL divergence :math:`D_{\mathrm{KL}}(p \| q)`.

    For :class:`~normix.exponential_family.ExponentialFamily` objects, calls
    ``p.kl_divergence(q)`` (Tier 2).

    For :class:`~normix.mixtures.marginal.NormalMixture` objects, delegates to
    the joint distributions.

    Parameters
    ----------
    p, q : ExponentialFamily or NormalMixture
        Must be the same type.
    """
    from normix.mixtures.marginal import NormalMixture

    if isinstance(p, NormalMixture):
        return p.joint.kl_divergence(q.joint)
    return p.kl_divergence(q)
