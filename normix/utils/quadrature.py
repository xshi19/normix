r"""Discrete probability measures for conditional expectations.

See :doc:`/theory/quadrature_scenarios` for the quadrature construction
and the finance identities preserved by positive weights.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from normix.exponential_family import ExponentialFamily

__all__ = ["DiscreteMeasure"]


class DiscreteMeasure(eqx.Module):
    r"""Probability measure :math:`Q=\sum_j p_j\delta_{Y_j}` on positive nodes.

    Parameters
    ----------
    Y : Array
        Nonempty, finite, positive one-dimensional nodes.
    log_p : Array
        Log weights with the same shape as ``Y``. Normalized at construction;
        :math:`-\infty` denotes a zero weight. At least one must be finite.
    tail_index : Array
        Positive-moment threshold of the source law. Defaults to infinity
        for empirical measures. Quadrature retains the source threshold so
        truncation cannot hide a divergent risk or derivative.
    """

    Y: Array
    log_p: Array
    tail_index: Array

    def __init__(
        self, Y: Array, log_p: Array, tail_index: Array | float = jnp.inf,
    ) -> None:
        Y = jnp.asarray(Y, dtype=jnp.float64)
        log_p = jnp.asarray(log_p, dtype=jnp.float64)
        tail_index = jnp.asarray(tail_index, dtype=jnp.float64)
        if Y.ndim != 1 or Y.size == 0 or log_p.shape != Y.shape:
            raise ValueError("nodes and log weights must be nonempty matching 1-D arrays")
        if tail_index.ndim != 0:
            raise ValueError("tail_index must be scalar")
        self.Y = eqx.error_if(
            Y, jnp.any(~jnp.isfinite(Y) | (Y <= 0)),
            "nodes must be finite and positive",
        )
        log_p = eqx.error_if(
            log_p, jnp.any(jnp.isnan(log_p) | jnp.isposinf(log_p))
            | ~jnp.any(jnp.isfinite(log_p)),
            "log weights must be finite or -inf with positive total mass",
        )
        self.log_p = log_p - logsumexp(log_p)
        self.tail_index = eqx.error_if(
            tail_index, jnp.isnan(tail_index) | (tail_index <= 0),
            "tail_index must be positive",
        )

    @property
    def p(self) -> Array:
        """Normalized, nonnegative probabilities."""
        return jnp.exp(self.log_p)

    def mean(self) -> Array:
        r"""Finite discrete mean :math:`E_Q[Y]`, irrespective of the source law."""
        return self.p @ self.Y

    @classmethod
    def from_sample(cls, Y: Array) -> DiscreteMeasure:
        """Assign equal probabilities to a sample, preserving its order."""
        return cls(Y, jnp.zeros_like(jnp.asarray(Y, dtype=jnp.float64)))

    @classmethod
    def from_weights(cls, Y: Array, w: Array) -> DiscreteMeasure:
        """Normalize finite nonnegative weights with positive total mass."""
        w = jnp.asarray(w, dtype=jnp.float64)
        w = eqx.error_if(
            w, jnp.any(~jnp.isfinite(w) | (w < 0)) | ~jnp.any(w > 0),
            "weights must be finite and nonnegative with positive total mass",
        )
        positive = w > 0
        return cls(Y, jnp.where(positive, jnp.log(jnp.where(positive, w, 1.0)), -jnp.inf))

    @classmethod
    def coerce(
        cls, scenarios: DiscreteMeasure | Array | None,
        subordinator: ExponentialFamily | None = None,
    ) -> DiscreteMeasure:
        """Use a measure, assign sample weights, or build subordinator quadrature.

        ``None`` requires a subordinator. Explicit measures retain their
        source law metadata and are returned unchanged.
        """
        if scenarios is None:
            if subordinator is None:
                raise ValueError("scenarios are required without a subordinator")
            return subordinator.quadrature()
        if isinstance(scenarios, cls):
            return scenarios
        return cls.from_sample(scenarios)
