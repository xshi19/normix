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
