"""
Tests for the cache infrastructure on Distribution base class.

Tests that:
- _fitted flag works correctly
- _check_fitted() raises before fitting, passes after
- _invalidate_cache() clears cached_property values
- _invalidate_cache() is idempotent (safe when no cache)
- _cached_attrs inheritance works for subclasses
"""

from functools import cached_property

import numpy as np
import pytest

from normix.base.distribution import Distribution


# ============================================================================
# Minimal concrete subclass for testing
# ============================================================================

class _MockDistribution(Distribution):
    """Minimal concrete Distribution for testing cache infrastructure."""

    _cached_attrs = ('expensive_value',)

    def __init__(self):
        super().__init__()
        self._data = None
        self._compute_count = 0  # Track how many times expensive_value is computed

    @cached_property
    def expensive_value(self) -> float:
        """Simulates an expensive derived computation."""
        self._compute_count += 1
        return self._data * 2.0

    def set_data(self, value: float):
        self._data = value
        self._fitted = True
        self._invalidate_cache()

    # Required abstract methods (minimal stubs)
    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        return np.zeros_like(np.asarray(x))

    def rvs(self, size=None, random_state=None):
        return np.zeros(size or 1)

    def fit(self, data, *args, **kwargs):
        self.set_data(float(np.mean(data)))
        return self


class _MockChild(_MockDistribution):
    """Child class that extends _cached_attrs."""

    _cached_attrs = _MockDistribution._cached_attrs + ('another_value',)

    @cached_property
    def another_value(self) -> float:
        return self._data ** 2


# ============================================================================
# Tests
# ============================================================================

class TestFittedFlag:
    def test_initially_not_fitted(self):
        dist = _MockDistribution()
        assert dist._fitted is False

    def test_fitted_after_set_data(self):
        dist = _MockDistribution()
        dist.set_data(5.0)
        assert dist._fitted is True

    def test_fitted_after_fit(self):
        dist = _MockDistribution()
        result = dist.fit(np.array([1.0, 2.0, 3.0]))
        assert dist._fitted is True
        assert result is dist  # fit returns self


class TestCheckFitted:
    def test_raises_when_not_fitted(self):
        dist = _MockDistribution()
        with pytest.raises(ValueError, match="parameters not set"):
            dist._check_fitted()

    def test_passes_when_fitted(self):
        dist = _MockDistribution()
        dist.set_data(5.0)
        dist._check_fitted()  # Should not raise

    def test_error_includes_class_name(self):
        dist = _MockDistribution()
        with pytest.raises(ValueError, match="_MockDistribution"):
            dist._check_fitted()


class TestInvalidateCache:
    def test_cached_property_computed_once(self):
        dist = _MockDistribution()
        dist.set_data(5.0)
        assert dist._compute_count == 0

        # First access computes
        val1 = dist.expensive_value
        assert val1 == 10.0
        assert dist._compute_count == 1

        # Second access uses cache
        val2 = dist.expensive_value
        assert val2 == 10.0
        assert dist._compute_count == 1

    def test_invalidate_clears_cache(self):
        dist = _MockDistribution()
        dist.set_data(5.0)

        val1 = dist.expensive_value
        assert dist._compute_count == 1

        # Invalidate and recompute
        dist._invalidate_cache()
        val2 = dist.expensive_value
        assert dist._compute_count == 2
        assert val2 == 10.0  # Same data, same result

    def test_set_data_invalidates_cache(self):
        dist = _MockDistribution()
        dist.set_data(5.0)

        val1 = dist.expensive_value
        assert val1 == 10.0
        assert dist._compute_count == 1

        # Change data — cache should be invalidated
        dist.set_data(7.0)
        val2 = dist.expensive_value
        assert val2 == 14.0
        assert dist._compute_count == 2

    def test_invalidate_idempotent_no_cache(self):
        """_invalidate_cache is safe to call when no cached values exist."""
        dist = _MockDistribution()
        dist._invalidate_cache()  # No error
        dist._invalidate_cache()  # Still no error

    def test_invalidate_idempotent_after_clear(self):
        dist = _MockDistribution()
        dist.set_data(5.0)
        _ = dist.expensive_value
        dist._invalidate_cache()
        dist._invalidate_cache()  # Second call is safe


class TestCachedAttrsInheritance:
    def test_parent_cached_attrs(self):
        assert 'expensive_value' in _MockDistribution._cached_attrs

    def test_child_extends_cached_attrs(self):
        assert 'expensive_value' in _MockChild._cached_attrs
        assert 'another_value' in _MockChild._cached_attrs

    def test_child_invalidates_both(self):
        dist = _MockChild()
        dist.set_data(5.0)

        # Access both cached properties
        assert dist.expensive_value == 10.0
        assert dist.another_value == 25.0

        # Change data
        dist.set_data(3.0)

        # Both should be recomputed
        assert dist.expensive_value == 6.0
        assert dist.another_value == 9.0

    def test_base_distribution_has_empty_cached_attrs(self):
        assert Distribution._cached_attrs == ()


# ============================================================================
# Tests for NormalMixture cache behavior
# ============================================================================

class TestNormalMixtureCacheInfrastructure:
    """Test that NormalMixture correctly syncs _fitted with its joint."""

    def test_normal_mixture_not_fitted_initially(self):
        """NormalMixture should not be fitted at creation."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma()
        assert vg._fitted is False

    def test_normal_mixture_fitted_after_from_classical_params(self):
        """NormalMixture should be fitted after from_classical_params."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        )
        assert vg._fitted is True
        assert vg._joint._fitted is True

    def test_normal_mixture_check_fitted_syncs_from_joint(self):
        """_check_fitted should sync _fitted from joint when joint is fitted."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma()
        vg._joint = vg._create_joint_distribution()
        # Set params directly on joint (simulates EM behavior)
        vg._joint.set_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        )
        # Marginal _fitted is False, but joint is fitted
        assert vg._fitted is False
        assert vg._joint._fitted is True
        # _check_fitted should sync and not raise
        vg._check_fitted()
        assert vg._fitted is True

    def test_normal_mixture_check_fitted_raises_when_no_joint(self):
        """_check_fitted should raise when joint is not initialized."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma()
        with pytest.raises(ValueError, match="parameters not set"):
            vg._check_fitted()

    def test_normal_mixture_classical_params_cached_property(self):
        """classical_params cached property should work on NormalMixture."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        )
        params = vg.classical_params
        assert 'mu' in params
        assert 'gamma' in params
        assert 'sigma' in params

    def test_normal_mixture_invalidate_cache_on_param_change(self):
        """classical_params should be invalidated when params change."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        )
        # Access classical_params to populate cache
        params1 = vg.classical_params

        # Change params via setter (which calls _invalidate_cache)
        vg.set_classical_params(
            mu=np.array([1.0]), gamma=np.array([0.2]),
            sigma=np.array([[2.0]]), shape=3.0, rate=2.0
        )
        params2 = vg.classical_params
        assert not np.allclose(params1['mu'], params2['mu'])

    def test_normal_mixture_repr_uses_fitted(self):
        """__repr__ should use _fitted flag."""
        from normix.distributions.mixtures.variance_gamma import VarianceGamma
        vg = VarianceGamma()
        assert "not fitted" in repr(vg)
        vg = VarianceGamma.from_classical_params(
            mu=np.array([0.0]), gamma=np.array([0.1]),
            sigma=np.array([[1.0]]), shape=2.0, rate=1.0
        )
        assert "not fitted" not in repr(vg)

    def test_normal_mixture_params_are_properties(self):
        """NormalMixture parameter accessors should be plain properties (not cached)."""
        from normix.base.mixture import NormalMixture
        # NormalMixture uses @property (not @cached_property) for parameter
        # accessors because the EM loop may update self._joint directly.
        # The joint distribution's own cached_property handles caching.
        for attr in ('natural_params', 'classical_params', 'expectation_params'):
            assert isinstance(
                getattr(NormalMixture, attr), property
            ), f"{attr} should be a plain @property on NormalMixture"
