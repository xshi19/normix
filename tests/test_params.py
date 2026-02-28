"""
Tests for frozen dataclass parameter containers.

Tests that each parameter dataclass:
- Can be constructed with valid values
- Is frozen (raises FrozenInstanceError on attribute assignment)
- Supports dataclasses.asdict() for dict conversion
- Has correct field names and types
"""

import dataclasses
import pytest
import numpy as np

from normix.params import (
    ExponentialParams,
    GammaParams,
    InverseGammaParams,
    InverseGaussianParams,
    GIGParams,
    MultivariateNormalParams,
    VarianceGammaParams,
    NormalInverseGammaParams,
    NormalInverseGaussianParams,
    GHParams,
)


# ============================================================================
# Univariate parameter dataclasses
# ============================================================================

class TestExponentialParams:
    def test_construction(self):
        p = ExponentialParams(rate=2.0)
        assert p.rate == 2.0

    def test_frozen(self):
        p = ExponentialParams(rate=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.rate = 3.0

    def test_asdict(self):
        p = ExponentialParams(rate=2.0)
        d = dataclasses.asdict(p)
        assert d == {"rate": 2.0}

    def test_slots(self):
        p = ExponentialParams(rate=2.0)
        assert not hasattr(p, "__dict__")


class TestGammaParams:
    def test_construction(self):
        p = GammaParams(shape=2.0, rate=1.5)
        assert p.shape == 2.0
        assert p.rate == 1.5

    def test_frozen(self):
        p = GammaParams(shape=2.0, rate=1.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.shape = 3.0

    def test_asdict(self):
        p = GammaParams(shape=2.0, rate=1.5)
        d = dataclasses.asdict(p)
        assert d == {"shape": 2.0, "rate": 1.5}


class TestInverseGammaParams:
    def test_construction(self):
        p = InverseGammaParams(shape=3.0, rate=2.0)
        assert p.shape == 3.0
        assert p.rate == 2.0

    def test_frozen(self):
        p = InverseGammaParams(shape=3.0, rate=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.rate = 5.0


class TestInverseGaussianParams:
    def test_construction(self):
        p = InverseGaussianParams(delta=1.0, eta=2.0)
        assert p.delta == 1.0
        assert p.eta == 2.0

    def test_frozen(self):
        p = InverseGaussianParams(delta=1.0, eta=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.delta = 3.0


class TestGIGParams:
    def test_construction(self):
        p = GIGParams(p=-0.5, a=1.0, b=2.0)
        assert p.p == -0.5
        assert p.a == 1.0
        assert p.b == 2.0

    def test_frozen(self):
        p = GIGParams(p=-0.5, a=1.0, b=2.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.a = 5.0

    def test_asdict(self):
        p = GIGParams(p=-0.5, a=1.0, b=2.0)
        d = dataclasses.asdict(p)
        assert d == {"p": -0.5, "a": 1.0, "b": 2.0}


# ============================================================================
# Multivariate parameter dataclasses
# ============================================================================

class TestMultivariateNormalParams:
    def test_construction(self):
        mu = np.array([1.0, 2.0])
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        p = MultivariateNormalParams(mu=mu, sigma=sigma)
        np.testing.assert_array_equal(p.mu, mu)
        np.testing.assert_array_equal(p.sigma, sigma)

    def test_frozen_mu(self):
        mu = np.array([1.0, 2.0])
        sigma = np.eye(2)
        p = MultivariateNormalParams(mu=mu, sigma=sigma)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.mu = np.array([3.0, 4.0])

    def test_frozen_sigma(self):
        mu = np.array([1.0, 2.0])
        sigma = np.eye(2)
        p = MultivariateNormalParams(mu=mu, sigma=sigma)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.sigma = np.eye(2) * 2

    def test_asdict(self):
        mu = np.array([1.0])
        sigma = np.array([[2.0]])
        p = MultivariateNormalParams(mu=mu, sigma=sigma)
        d = dataclasses.asdict(p)
        assert "mu" in d
        assert "sigma" in d


# ============================================================================
# Normal mixture parameter dataclasses
# ============================================================================

class TestVarianceGammaParams:
    def test_construction(self):
        p = VarianceGammaParams(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=2.0,
            rate=1.0,
        )
        assert p.shape == 2.0
        assert p.rate == 1.0
        np.testing.assert_array_equal(p.mu, [0.0])

    def test_frozen(self):
        p = VarianceGammaParams(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=2.0,
            rate=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.shape = 3.0

    def test_field_names(self):
        fields = {f.name for f in dataclasses.fields(VarianceGammaParams)}
        assert fields == {"mu", "gamma", "sigma", "shape", "rate"}


class TestNormalInverseGammaParams:
    def test_construction(self):
        p = NormalInverseGammaParams(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            shape=2.0,
            rate=1.0,
        )
        assert p.shape == 2.0
        assert p.rate == 1.0

    def test_field_names(self):
        fields = {f.name for f in dataclasses.fields(NormalInverseGammaParams)}
        assert fields == {"mu", "gamma", "sigma", "shape", "rate"}


class TestNormalInverseGaussianParams:
    def test_construction(self):
        p = NormalInverseGaussianParams(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            delta=1.0,
            eta=2.0,
        )
        assert p.delta == 1.0
        assert p.eta == 2.0

    def test_field_names(self):
        fields = {f.name for f in dataclasses.fields(NormalInverseGaussianParams)}
        assert fields == {"mu", "gamma", "sigma", "delta", "eta"}


class TestGHParams:
    def test_construction(self):
        p = GHParams(
            mu=np.array([0.0, 1.0]),
            gamma=np.array([0.5, -0.5]),
            sigma=np.eye(2),
            p=-0.5,
            a=1.0,
            b=2.0,
        )
        assert p.p == -0.5
        assert p.a == 1.0
        assert p.b == 2.0
        np.testing.assert_array_equal(p.mu, [0.0, 1.0])

    def test_frozen(self):
        p = GHParams(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=-0.5,
            a=1.0,
            b=2.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.p = 1.0

    def test_field_names(self):
        fields = {f.name for f in dataclasses.fields(GHParams)}
        assert fields == {"mu", "gamma", "sigma", "p", "a", "b"}

    def test_asdict(self):
        p = GHParams(
            mu=np.array([0.0]),
            gamma=np.array([0.5]),
            sigma=np.array([[1.0]]),
            p=-0.5,
            a=1.0,
            b=2.0,
        )
        d = dataclasses.asdict(p)
        assert set(d.keys()) == {"mu", "gamma", "sigma", "p", "a", "b"}
        assert d["p"] == -0.5
