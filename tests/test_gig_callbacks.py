"""CPU-executable checks for the accelerator GIG callback path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from normix import GIG


@pytest.fixture
def offload_calls(monkeypatch):
    """Route traced solves through the callback and record operand shapes."""
    pure_callback = jax.pure_callback
    calls = []

    def record(callback, *args, **kwargs):
        def wrapped(*operands):
            calls.append(jax.tree.map(np.shape, operands))
            return callback(*operands)

        return pure_callback(wrapped, *args, **kwargs)

    monkeypatch.setattr(jax, "pure_callback", record)
    monkeypatch.setattr(
        jax.lax, "platform_dependent",
        lambda *args, default, **kwargs: default(*args),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        yield calls


@pytest.fixture(scope="module")
@jax.default_device(jax.devices("cpu")[0])
def targets_and_starts():
    models = [
        GIG(0.5, 1.0, 1.0),
        GIG(0.5, 10.0, 0.1),
        GIG(-1.0, 0.1, 10.0),
        GIG(-0.5, 2.0, 1.0),
    ]
    targets = jnp.stack([model.expectation_params() for model in models])
    starts = jnp.stack([
        GIG(model.p + 0.25, model.a * 1.5, model.b * 0.5).natural_params()
        for model in models
    ])
    return targets, starts


def _invert(eta, theta0):
    return GIG.from_expectation(
        eta, theta0=theta0, maxiter=100, tol=1e-10,
    ).natural_params()


@pytest.mark.contract
@pytest.mark.parametrize("start_mode", ["cold", "mapped", "shared"])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_gig_callback_matches_scalar(
    offload_calls, targets_and_starts, start_mode, batch_shape,
):
    """Allow adaptive-solve rounding for distant shared warm starts."""
    targets, starts = targets_and_starts
    indices = jnp.arange(int(np.prod(batch_shape))) % len(targets)
    targets = targets[indices].reshape(batch_shape + (3,))
    starts = starts[indices].reshape(batch_shape + (3,))
    scalar = jax.jit(
        lambda eta, theta0: GIG._solve_expectation(
            eta, theta0, maxiter=100, tol=1e-10,
        )
    )
    expected = np.stack([
        scalar(eta, None if start_mode == "cold" else (
            starts.reshape(-1, 3)[0] if start_mode == "shared" else start
        ))
        for eta, start in zip(targets.reshape(-1, 3), starts.reshape(-1, 3))
    ]).reshape(targets.shape)
    if start_mode == "cold":
        starts = None
    elif start_mode == "shared":
        starts = starts.reshape(-1, 3)[0]

    invert = lambda eta, theta0: _invert(eta, theta0)
    for _ in batch_shape:
        invert = jax.vmap(invert, in_axes=(0, 0 if start_mode == "mapped" else None))
    actual = jax.jit(invert)(targets, starts).block_until_ready()
    assert np.isfinite(expected).all()
    rtol = 1e-8 if start_mode == "shared" else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1e-12)
    assert len(offload_calls) == 1
    assert offload_calls[0][0] == targets.shape


@pytest.mark.contract
def test_gig_callback_nested_broadcast(offload_calls, targets_and_starts):
    targets, starts = targets_and_starts
    targets = targets[:2]
    starts = starts[0] * jnp.array([0.9, 1.0, 1.1])[:, None]
    scalar = jax.jit(_invert)
    expected = np.array([[scalar(eta, start) for start in starts] for eta in targets])
    offload_calls.clear()
    invert = jax.vmap(jax.vmap(_invert, in_axes=(None, 0)), in_axes=(0, None))
    actual = jax.jit(invert)(targets, starts).block_until_ready()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    assert offload_calls == [((2, 3, 3), (2, 3, 3))]


@pytest.mark.contract
def test_gig_callback_preserves_implicit_derivative(offload_calls, targets_and_starts):
    targets, starts = targets_and_starts
    targets, starts = targets[:2], starts[:2]
    theta = jax.jit(jax.vmap(_invert))(targets, starts)
    fisher = np.asarray(jax.vmap(GIG._hessian_log_partition)(theta))
    for differentiate in (jax.jacfwd, jax.jacrev):
        derivative = jax.jit(jax.vmap(differentiate(_invert)))(targets, starts)
        np.testing.assert_allclose(
            fisher @ derivative, np.broadcast_to(np.eye(3), fisher.shape),
            rtol=1e-9, atol=1e-9,
        )
    start_derivative = jax.jit(jax.vmap(jax.jacrev(_invert, argnums=1)))(targets, starts)
    np.testing.assert_array_equal(start_derivative, np.zeros_like(fisher))


@pytest.mark.contract
def test_gig_callback_invalid_target_does_not_affect_batch(offload_calls, targets_and_starts):
    targets, starts = targets_and_starts
    expected = jax.jit(_invert)(targets[0], starts[0])
    targets = targets[:2].at[1, 1].set(-1.0)
    actual = jax.jit(jax.vmap(_invert))(targets, starts[:2])
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)
    assert np.isnan(actual[1]).all()
