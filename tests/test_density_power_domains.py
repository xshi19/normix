"""Powered joint densities against independent one-dimensional quadrature."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, kv

from normix import GeneralizedHyperbolic, NormalInverseGamma, NormalInverseGaussian, VarianceGamma


def _model(family, shape, d):
    return family.from_classical(
        mu=jnp.arange(d) * 0.2, gamma=jnp.full(d, 0.3),
        sigma=jnp.eye(d) * 1.7, alpha=shape, beta=1.3,
    )


def _power_integrand(family, shape, d, q):
    normal = (1 - q) * d / 2 * np.log(2 * np.pi * 1.7) - d / 2 * np.log(q)
    constant = normal + q * (shape * np.log(1.3) - gammaln(shape))
    power = (1 + q * (shape - 1) + (1 - q) * d / 2
             if family is VarianceGamma else q * (shape + 1) - (1 - q) * d / 2 - 1)

    def integrand(t):
        if t > 700:
            return 0.0
        return np.exp(constant + power * t - q * 1.3 * np.exp(t))

    return integrand


@pytest.mark.contract
@pytest.mark.parametrize('family,q', [(VarianceGamma, 2.0), (NormalInverseGamma, 0.5)])
@pytest.mark.parametrize('d', [1, 3])
@pytest.mark.parametrize('gap', [-0.04, 0.0, 0.04])
def test_joint_power_integrability_boundary(family, q, d, gap):
    boundary = (1 + d / 2 - (1 + d / 2) / q if family is VarianceGamma
                else (1 + d / 2) / q - 1 - d / 2)
    shape = boundary + gap / q
    model = _model(family, shape, d)
    integrand = _power_integrand(family, shape, d, q)
    if gap > 0:
        reference = np.log(quad(integrand, -np.inf, 0, epsabs=1e-10)[0]
                           + quad(integrand, 0, np.inf, epsabs=1e-10)[0])
        np.testing.assert_allclose(model.joint.log_density_power(q), reference, atol=1e-9)
        np.testing.assert_allclose(model.joint_renyi(q), reference / (1 - q), atol=1e-9)
    else:
        tail1 = quad(integrand, -60, -40)[0]
        tail2 = quad(integrand, -80, -60)[0]
        assert tail2 >= tail1 * (1 - 1e-12)
        assert np.isposinf(model.joint.log_density_power(q))
        assert np.isinf(model.joint_renyi(q))
        assert np.sign(model.joint_renyi(q)) == np.sign(1 - q)
    lifted = model.joint.to_joint_generalized_hyperbolic()
    np.testing.assert_allclose(lifted.log_density_power(q), model.joint.log_density_power(q), atol=1e-9)


@pytest.mark.contract
@pytest.mark.parametrize('family', [VarianceGamma, NormalInverseGamma])
@pytest.mark.parametrize('q', [0.8, 1.2])
def test_joint_finite_power_quadrature(family, q):
    model = _model(family, 3.0, 2)
    integrand = _power_integrand(family, 3.0, 2, q)
    reference = np.log(quad(integrand, -np.inf, 0)[0] + quad(integrand, 0, np.inf)[0])
    np.testing.assert_allclose(model.joint.log_density_power(q), reference, atol=1e-9)


@pytest.mark.contract
@pytest.mark.parametrize('family,shape,orders', [
    (VarianceGamma, 0.2, [0.5, 1.0, 2.0]),
    (NormalInverseGamma, 0.5, [0.5, 1.0, 2.0]),
])
def test_joint_power_domain_jit_vmap(family, shape, orders):
    joint = _model(family, shape, 1).joint
    actual = jax.jit(jax.vmap(joint.log_density_power))(jnp.array(orders))
    expected = [joint.log_density_power(q) for q in orders]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert np.isposinf(actual[2 if family is VarianceGamma else 0])


@pytest.mark.contract
@pytest.mark.parametrize('p,a,b', [(0.4, 1.3, 2.1), (-0.5, 2.0, 2.0)])
@pytest.mark.parametrize('q', [0.5, 2.0])
def test_interior_gh_nig_power_quadrature(p, a, b, q):
    kwargs = dict(mu=jnp.array([0.2]), gamma=jnp.array([0.3]), sigma=jnp.array([[1.7]]))
    model = GeneralizedHyperbolic.from_classical(**kwargs, p=p, a=a, b=b)
    log_z = np.log(2 * kv(p, np.sqrt(a * b))) + p / 2 * np.log(b / a)
    normal = (1 - q) / 2 * np.log(2 * np.pi * 1.7) - 0.5 * np.log(q)

    def integrand(t):
        if abs(t) > 700:
            return 0.0
        return np.exp(normal - q * log_z + (q * (p - 1) + (1 - q) / 2 + 1) * t
                      - q / 2 * (a * np.exp(t) + b * np.exp(-t)))

    reference = np.log(quad(integrand, -np.inf, np.inf, epsabs=1e-10)[0])
    np.testing.assert_allclose(model.joint.log_density_power(q), reference, atol=1e-9)
    if p == -0.5:
        nig = NormalInverseGaussian.from_classical(**kwargs, mu_ig=np.sqrt(b / a), lam=b)
        np.testing.assert_allclose(nig.joint.log_density_power(q), reference, atol=1e-9)


@pytest.mark.contract
@pytest.mark.parametrize('family,q', [(VarianceGamma, 1.0000005), (NormalInverseGamma, 0.9999995)])
def test_joint_power_domain_inside_renyi_taylor_window(family, q):
    joint = _model(family, 1e-8, 1).joint
    assert np.isposinf(joint.log_density_power(q))
    assert np.isinf(joint.renyi(q))
    assert np.sign(joint.renyi(q)) == np.sign(1 - q)
