"""Numerical evidence for the claims in formal/lean/coverage.toml."""

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from normix.distributions.gamma import Gamma


@settings(max_examples=20, deadline=None)
@given(
    alpha=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
    beta=st.floats(
        min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False
    ),
)
@example(alpha=2.0, beta=3.0)
@example(alpha=1e-3, beta=1e3)
@example(alpha=1e3, beta=1e-3)
def test_gamma_var_formula(alpha: float, beta: float) -> None:
    """Check EF-04-var and the moment coordinates of public JAX derivatives."""
    dist = Gamma(
        jnp.asarray(alpha, dtype=jnp.float64), jnp.asarray(beta, dtype=jnp.float64)
    )
    theta = dist.natural_params()
    mean, var = dist.mean(), dist.var()

    def log_partition(theta: jax.Array) -> jax.Array:
        return Gamma.from_natural(theta).log_partition()

    np.testing.assert_allclose(theta, [alpha - 1.0, -beta], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        [mean, var], [alpha / beta, alpha / beta**2], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(var, mean / beta, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        mean, jax.grad(log_partition)(theta)[1], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        var, jax.hessian(log_partition)(theta)[1, 1], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        mean, dist.expectation_params(backend="jax")[1], rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        var, dist.fisher_information(backend="jax")[1, 1], rtol=1e-12, atol=0
    )
