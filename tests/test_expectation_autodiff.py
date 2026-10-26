"""Implicit expectation inversion against independent Hessians and differences."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import polygamma

from normix import GIG, Gamma, InverseGamma, InverseGaussian, VarianceGamma, NormalInverseGamma
from normix.exponential_family import ExponentialFamily


_GIG_REF_PATH = Path(__file__).resolve().parent / "data" / "gig_hessian_reference.json"
_GIG_HESSIANS = {
    (row["p"], row["a"], row["b"]): np.array(row["hessian"])
    for row in json.loads(_GIG_REF_PATH.read_text())["points"]
}


def _reference_hessian(dist):
    if isinstance(dist, (Gamma, InverseGamma)):
        a, b = float(dist.alpha), float(dist.beta)
        h = np.array([[polygamma(1, a), 1 / b], [1 / b, a / b**2]])
        return h if isinstance(dist, Gamma) else h[::-1, ::-1]
    key = (float(dist.p), float(dist.a), float(dist.b))
    return _GIG_HESSIANS[key]


_INVERSE_JACOBIAN_DISTS = [
    Gamma(3.0, 2.0), Gamma(0.7, 0.8), Gamma(20.0, 5.0),
    InverseGamma(3.0, 2.0), InverseGamma(0.7, 0.8), InverseGamma(20.0, 5.0),
    GIG(0.5, 1.0, 1.0), GIG(-1.5, 2.0, 0.7), GIG(2.5, 0.5, 3.0),
    GIG(0.5, 1e-4, 1e4),
]
_GIG_COLD_START_DISTS = [
    GIG(0.5, 1.0, 1.0), GIG(-1.5, 2.0, 0.7), GIG(2.5, 0.5, 3.0),
]


@pytest.mark.contract
@pytest.mark.timeout(300)
@pytest.mark.parametrize('dist', _INVERSE_JACOBIAN_DISTS)
def test_expectation_inverse_jacobians(dist):
    eta = dist.expectation_params()
    theta0 = dist.natural_params() * 0.95

    def invert(target):
        return type(dist).from_expectation(target, theta0=theta0, tol=1e-12).natural_params()

    reference = np.linalg.inv(_reference_hessian(dist))
    forward = jax.jit(jax.jacfwd(invert))(eta)
    reverse = jax.jit(jax.jacrev(invert))(eta)
    invert_jit = jax.jit(invert)
    differences = []
    for i in range(len(eta)):
        step = 1e-5 * (abs(float(eta[i])) if i > 0 else 1.0)
        delta = jnp.eye(len(eta))[i] * step
        differences.append((invert_jit(eta + delta) - invert_jit(eta - delta)) / (2 * step))
    for actual in (forward, reverse):
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(actual, np.array(differences).T, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda target: jnp.sum(invert(target))))(eta),
                               np.sum(reference, axis=0), rtol=2e-6, atol=2e-7)
    jax.clear_caches()


@pytest.mark.contract
@pytest.mark.timeout(300)
@pytest.mark.parametrize('dist', _GIG_COLD_START_DISTS)
def test_gig_cold_start_grad_and_jit(dist):
    def invert(eta):
        return GIG.from_expectation(eta, tol=1e-12).natural_params()

    eta = dist.expectation_params()
    np.testing.assert_allclose(jax.jit(invert)(eta), dist.natural_params(), rtol=2e-6, atol=2e-7)
    reference = np.linalg.inv(_reference_hessian(dist))
    np.testing.assert_allclose(jax.grad(lambda target: jnp.sum(invert(target)))(eta),
                               reference.sum(axis=0), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(jax.jit(jax.jacrev(invert))(eta), reference,
                               rtol=2e-6, atol=2e-7)
    jax.clear_caches()


@pytest.mark.contract
@pytest.mark.parametrize('compiled', [False, True])
def test_failed_expectation_solve_has_nan_derivative(compiled):
    eta = GIG(0.5, 1.0, 1.0).expectation_params()

    def invert(target):
        return GIG.from_expectation(target, theta0=GIG(0.0, 10.0, 10.0).natural_params(),
                                    maxiter=1).natural_params()

    derivative = jax.jacrev(invert)
    if compiled:
        derivative = jax.jit(derivative)
    assert np.isnan(derivative(eta)).all()


@pytest.mark.contract
def test_inverse_gaussian_closed_form_derivative_unchanged():
    dist = InverseGaussian(1.5, 2.0)
    inverse = lambda eta: InverseGaussian.from_expectation(eta).natural_params()
    np.testing.assert_allclose(jax.jacrev(inverse)(dist.expectation_params()),
                               np.linalg.inv(dist.fisher_information()), rtol=1e-9)


@pytest.mark.contract
@pytest.mark.parametrize('family', [VarianceGamma, NormalInverseGamma])
def test_joint_mstep_parameter_derivative(family):
    def roundtrip(shape):
        model = family.from_classical(mu=jnp.array([0.2]), gamma=jnp.array([0.3]),
                                      sigma=jnp.array([[1.4]]), alpha=shape, beta=2.0)
        return type(model.joint).from_expectation(model.compute_eta_from_model()).alpha

    np.testing.assert_allclose(jax.jit(jax.grad(roundtrip))(3.0), 1.0, atol=1e-7)


@pytest.mark.contract
def test_gamma_shape_projection_derivative():
    eta = Gamma(0.7, 2.0).expectation_params()
    invert = lambda target: Gamma.from_expectation(target, alpha_min=2.0).natural_params()
    expected = np.array([[0.0, 0.0], [0.0, 2.0 / float(eta[1])**2]])
    np.testing.assert_allclose(jax.jacrev(invert)(eta), expected, atol=1e-9)


class _RedundantNormal(ExponentialFamily):
    theta: jax.Array

    @classmethod
    def _log_partition_from_theta(cls, theta: jax.Array) -> jax.Array:
        return (theta[0] + theta[1])**2 / 2

    def natural_params(self) -> jax.Array:
        return self.theta

    @staticmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        return jnp.array([x, x])

    @staticmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        return -(x**2 + jnp.log(2 * jnp.pi)) / 2

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "_RedundantNormal":
        return cls(theta)


@pytest.mark.contract
def test_singular_expectation_inverse_has_nan_derivative():
    invert = lambda eta: _RedundantNormal.from_expectation(
        eta, theta0=jnp.array([0.5, 0.5]), method='newton', maxiter=2,
    ).natural_params()
    eta = jnp.ones(2)
    np.testing.assert_allclose(invert(eta), [0.5, 0.5])
    assert np.isnan(jax.jit(jax.jacfwd(invert))(eta)).all()
    assert np.isnan(jax.jit(jax.jacrev(invert))(eta)).all()


@pytest.mark.contract
@pytest.mark.parametrize('method', ['lbfgs', 'bfgs'])
def test_generic_expectation_solver_derivative_under_jit(method):
    dist = Gamma(3.0, 2.0)

    def invert(eta):
        return ExponentialFamily.from_expectation.__func__(
            Gamma, eta, theta0=Gamma(2.8, 1.8).natural_params(), method=method,
            maxiter=50, tol=1e-12,
        ).natural_params()

    expected = np.linalg.inv(_reference_hessian(dist))
    np.testing.assert_allclose(jax.jit(jax.jacrev(invert))(dist.expectation_params()),
                               expected, rtol=1e-6, atol=1e-7)


@pytest.mark.contract
@pytest.mark.parametrize('method', ['lbfgs', 'bfgs'])
def test_gig_cold_start_quasi_newton_derivative_under_jit(method):
    dist = GIG(0.5, 1.0, 1.0)

    def invert(eta):
        return GIG.from_expectation(eta, method=method, maxiter=100).natural_params()

    eta = dist.expectation_params()
    expected = np.linalg.inv(_reference_hessian(dist))
    np.testing.assert_allclose(jax.jit(invert)(eta), dist.natural_params(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(jax.jit(jax.jacrev(invert))(eta), expected, rtol=1e-5, atol=1e-6)
    jax.clear_caches()
