"""Parameter and order derivatives of information measures."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import digamma, polygamma

from normix import Gamma, InverseGamma, NormalInverseGaussian, VarianceGamma


def _joint(beta):
    return VarianceGamma.from_classical(
        mu=jnp.array([0.2]), gamma=jnp.array([0.3]), sigma=jnp.array([[1.4]]),
        alpha=3.0, beta=beta,
    ).joint


def _nig(lam):
    return NormalInverseGaussian.from_classical(
        mu=jnp.array([0.2]), gamma=jnp.array([0.3]), sigma=jnp.array([[1.4]]),
        mu_ig=1.2, lam=lam,
    ).joint


@pytest.mark.contract
@pytest.mark.parametrize('factory', [lambda b: Gamma(3.0, b), lambda b: InverseGamma(3.0, b), _joint, _nig])
@pytest.mark.parametrize('order', [0.7, 1.0, 2.0])
def test_renyi_parameter_and_order_gradients(factory, order):
    def measure(x):
        return factory(x[0]).renyi(x[1])

    point = jnp.array([2.0, order])
    step = 1e-4
    differences = np.array([
        (measure(point + step * direction) - measure(point - step * direction)) / (2 * step)
        for direction in np.eye(2)
    ])
    actual = jax.jit(jax.grad(measure))(point)
    np.testing.assert_allclose(actual, differences, rtol=2e-5, atol=2e-7)


@pytest.mark.contract
@pytest.mark.parametrize('order', [0.7, 1.0, 2.0])
def test_gamma_renyi_parameter_closed_form(order):
    actual = jax.grad(lambda x: Gamma(x[0], x[1]).renyi(order))(jnp.array([3.0, 2.0]))
    shape_derivative = (1 + (1 - 3.0) * polygamma(1, 3.0) if order == 1
                        else order / (1 - order)
                        * (digamma(order * 2 + 1) - digamma(3.0) - np.log(order)))
    np.testing.assert_allclose(actual, [shape_derivative, -0.5], atol=1e-9)


@pytest.mark.contract
@pytest.mark.parametrize('factory', [lambda a: Gamma(a, 2.0), lambda a: InverseGamma(a, 2.0), _joint, _nig])
@pytest.mark.parametrize('name', ['entropy', 'varentropy'])
def test_entropy_varentropy_parameter_gradients(factory, name):
    measure = lambda value: getattr(factory(value), name)()
    step = 1e-4
    expected = (measure(3.0 + step) - measure(3.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(measure))(3.0), expected, rtol=2e-5, atol=2e-7)


@pytest.mark.contract
def test_renyi_parameter_gradients_vmap_at_one():
    derivative = jax.jit(jax.vmap(jax.grad(lambda b, q: Gamma(3.0, b).renyi(q)), in_axes=(None, 0)))
    np.testing.assert_allclose(derivative(2.0, jnp.array([0.7, 1.0, 2.0])), -0.5, atol=1e-9)
