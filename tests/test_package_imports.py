"""Release smoke tests for package imports and optional dependencies."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax


jax.config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_normix_without_matplotlib():
    """Importing normix should not require the optional plotting extra."""
    code = """
import builtins
import sys

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "matplotlib" or name.startswith("matplotlib."):
        raise ImportError("matplotlib blocked for smoke test")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import normix

assert "normix.utils.plotting" not in sys.modules
assert callable(normix.log_kv)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_plotting_exports_remain_available():
    """The public plotting re-exports remain available lazily."""
    from normix.utils import FIG_H, FIG_W, PHI, plot_pdf_cdf_comparison

    assert PHI > 1.0
    assert FIG_W > 0
    assert FIG_H > 0
    assert callable(plot_pdf_cdf_comparison)


def test_plotting_module_remains_importable():
    """Notebook-style submodule imports still work."""
    from normix.utils import plotting

    assert hasattr(plotting, "plot_pdf_cdf_comparison")
