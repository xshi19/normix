"""
Eta update rules for incremental and penalised EM.

Two-layer abstraction
---------------------

The most general rule is

.. math::

    \\eta_t = \\mathrm{rule}(\\eta_{t-1},\\, \\hat\\eta_{\\text{batch}}),

implemented by :class:`EtaUpdateRule` via ``__call__``. This leaves room
for non-affine predictors (e.g. an MLP that maps
``(η_{t-1}, η̂) → η_t``) without another API revision.

Most rules in this module are **affine**:

.. math::

    \\eta_t = a + b\\,\\eta_{t-1} + c\\,\\hat\\eta_{\\text{batch}},

implemented by :class:`AffineRule`, which delegates to
:meth:`AffineRule.weights` returning ``(a, b, c, state)`` and combines
via :func:`~normix.fitting.eta.affine_combine`.

All rules are :class:`equinox.Module` pytrees so their hyperparameters
(e.g. ``tau0``, ``w``, ``tau``) are JAX array leaves — JIT-compatible
and differentiable for future meta-learning of step-size schedules.
"""
from __future__ import annotations

import abc
from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.fitting.eta import NormalMixtureEta, affine_combine


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class EtaUpdateRule(eqx.Module):
    r"""Abstract base for eta update rules.

    The fitter only knows :meth:`__call__`; whether a concrete rule is
    affine, a combinator, or an ML-style predictor is invisible at the
    call site.

    Subclasses override :meth:`__call__`. Rules with no per-step memory
    inherit :meth:`initial_state` returning an empty ``dict``.
    """

    def initial_state(self) -> Dict:
        return {}

    @abc.abstractmethod
    def __call__(
        self,
        eta_prev,
        eta_new,
        step,
        batch_size,
        state: Dict,
    ):
        """Return ``(eta_t, new_state)``. Pure / JIT-friendly."""


class AffineRule(EtaUpdateRule):
    r"""Specialisation: :math:`\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta`.

    Subclasses implement :meth:`weights` instead of :meth:`__call__`. The
    base class provides a single :meth:`__call__` that delegates to
    :meth:`weights` and runs the combination through
    :func:`~normix.fitting.eta.affine_combine`.
    """

    @abc.abstractmethod
    def weights(
        self, step, batch_size, state: Dict,
    ) -> Tuple[Optional[NormalMixtureEta], jax.Array, jax.Array, Dict]:
        """Return ``(a, b, c, updated_state)``."""

    def __call__(self, eta_prev, eta_new, step, batch_size, state):
        a, b, c, state = self.weights(step, batch_size, state)
        return affine_combine(eta_prev, eta_new, b, c, a), state


# ---------------------------------------------------------------------------
# Concrete affine rules
# ---------------------------------------------------------------------------

class IdentityUpdate(AffineRule):
    """Pass-through: :math:`\\eta_t = \\hat\\eta` (standard batch EM)."""

    def weights(self, step, batch_size, state):
        return None, jnp.float64(0.0), jnp.float64(1.0), state


class RobbinsMonroUpdate(AffineRule):
    r"""Robbins–Monro: :math:`c = 1/(\tau_0 + t)`, :math:`b = 1 - c`.

    Parameters
    ----------
    tau0 : float
        Initial step-size denominator (higher → slower adaptation).
    """

    tau0: jax.Array

    def __init__(self, tau0: float = 10.0):
        self.tau0 = jnp.asarray(tau0, dtype=jnp.float64)

    def weights(self, step, batch_size, state):
        tau_t = self.tau0 + step
        c = 1.0 / tau_t
        b = 1.0 - c
        return None, b, c, state


class SampleWeightedUpdate(AffineRule):
    r"""Incremental mean: :math:`b = n/(n+m)`, :math:`c = m/(n+m)`.

    Tracks cumulative sample count *n*; each batch contributes *m*.
    """

    def weights(self, step, batch_size, state):
        n = state.get('cumulative_n', jnp.float64(0.0))
        m = jnp.float64(batch_size)
        total = n + m
        b = n / total
        c = m / total
        return None, b, c, {**state, 'cumulative_n': total}

    def initial_state(self):
        return {'cumulative_n': jnp.float64(0.0)}


class EWMAUpdate(AffineRule):
    r"""Exponentially weighted moving average: :math:`b = 1-w`, :math:`c = w`.

    Parameters
    ----------
    w : float
        Weight on the new batch (0 < w ≤ 1).
    """

    w: jax.Array

    def __init__(self, w: float = 0.1):
        self.w = jnp.asarray(w, dtype=jnp.float64)

    def weights(self, step, batch_size, state):
        return None, 1.0 - self.w, self.w, state


class ShrinkageUpdate(AffineRule):
    r"""Shrinkage toward a prior: :math:`a = \frac{\tau}{1+\tau}\eta_0`,
    :math:`b = 0`, :math:`c = \frac{1}{1+\tau}`.

    Parameters
    ----------
    eta0 : NormalMixtureEta
        Prior expectation parameters (shrinkage target).
    tau : float
        Shrinkage strength (higher → more regularisation).

    .. note::
       This is the legacy uniform-shrinkage rule. Phase 2 of the
       EM-extension plan replaces it with a ``Shrinkage(base, eta0, tau)``
       combinator that supports per-field ``tau`` and composes with any
       base rule.
    """

    eta0: NormalMixtureEta
    tau: jax.Array

    def __init__(self, eta0: NormalMixtureEta, tau: float = 0.5):
        self.eta0 = eta0
        self.tau = jnp.asarray(tau, dtype=jnp.float64)

    def weights(self, step, batch_size, state):
        factor = self.tau / (1.0 + self.tau)
        a = jax.tree.map(lambda x: factor * x, self.eta0)
        c = 1.0 / (1.0 + self.tau)
        return a, jnp.float64(0.0), c, state


class AffineUpdate(AffineRule):
    r"""User-defined constant :math:`(a, b, c)`.

    All three coefficients are pytree values — ``b`` and ``c`` are scalar
    ``jax.Array`` leaves, ``a`` is an optional :class:`NormalMixtureEta`.
    For time-varying schedules, subclass :class:`EtaUpdateRule` directly.

    Parameters
    ----------
    a : NormalMixtureEta or None
        Additive shift (e.g. scaled prior).
    b : float
        Weight on previous state.
    c : float
        Weight on new batch.
    """

    a: Optional[NormalMixtureEta]
    b: jax.Array
    c: jax.Array

    def __init__(
        self,
        a: Optional[NormalMixtureEta] = None,
        b: float = 0.0,
        c: float = 1.0,
    ):
        self.a = a
        self.b = jnp.asarray(b, dtype=jnp.float64)
        self.c = jnp.asarray(c, dtype=jnp.float64)

    def weights(self, step, batch_size, state):
        return self.a, self.b, self.c, state
