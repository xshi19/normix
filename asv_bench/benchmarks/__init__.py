"""ASV micro-suite for normix.

Enable float64 before any JAX work. Each benchmark module also sets this
because ASV may import files as top-level modules rather than as a package.

``JAX_PLATFORMS=cuda`` with a CPU-only jax fails at first device placement
(import of ``normix.utils.bessel``), so we drop back to cpu *before*
importing jax when the CUDA plugin is missing. ``setup`` then skips the
cuda series via ``NotImplementedError``.
"""

from __future__ import annotations

import importlib.util
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REQUESTED_PLATFORM = os.environ.get("JAX_PLATFORMS", "")
_CUDA_PLUGIN_MISSING = False
if _REQUESTED_PLATFORM == "cuda":
    if importlib.util.find_spec("jax_cuda12_plugin") is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
        _CUDA_PLUGIN_MISSING = True

import jax

jax.config.update("jax_enable_x64", True)


def require_requested_device() -> None:
    """Skip the cuda series when this interpreter cannot see a GPU."""
    if _REQUESTED_PLATFORM != "cuda":
        return
    if _CUDA_PLUGIN_MISSING:
        raise NotImplementedError(
            "JAX_PLATFORMS=cuda but jax_cuda12_plugin is not installed"
        )
    platforms = {d.platform for d in jax.devices()}
    if "gpu" not in platforms:
        raise NotImplementedError(
            f"JAX_PLATFORMS=cuda but jax.devices() has no gpu: {jax.devices()!r}"
        )


def block_pytree(tree) -> None:
    """``block_until_ready`` on every array leaf (η pytrees, fitted models)."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
