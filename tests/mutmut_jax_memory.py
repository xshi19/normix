"""Pytest plugin used only by the mutmut baseline (see [tool.mutmut]).

The selected GIG/Bessel tests compile many JAX programs in one process and
peak near 6 GB RSS. mutmut runs them in long-lived children, so this plugin
drops JAX's in-memory compilation caches after each test module to bound
memory. Load it with ``-p tests.mutmut_jax_memory``.
"""

import jax


def pytest_runtest_teardown(item, nextitem):
    if nextitem is None or nextitem.module is not item.module:
        jax.clear_caches()
