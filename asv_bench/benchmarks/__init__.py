"""ASV micro-suite for normix.

Enable float64 before any JAX work. Each benchmark module also sets this
because ASV may import files as top-level modules rather than as a package.

``JAX_PLATFORMS=cuda`` fails at first device placement (import of
``normix.utils.bessel``) when there is no GPU, so we drop back to cpu
*before* importing jax when the CUDA plugin is missing **or** no NVIDIA
device is visible (GitHub-hosted Linux runners can have the plugin
wheels without a GPU). ``setup`` then skips the cuda series via
``NotImplementedError``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _cuda_unavailable() -> bool:
    if importlib.util.find_spec("jax_cuda12_plugin") is None:
        return True
    return not os.path.exists("/dev/nvidia0") and shutil.which("nvidia-smi") is None


_REQUESTED_PLATFORM = os.environ.get("JAX_PLATFORMS", "")
_CUDA_UNAVAILABLE = False
if _REQUESTED_PLATFORM == "cuda" and _cuda_unavailable():
    os.environ["JAX_PLATFORMS"] = "cpu"
    _CUDA_UNAVAILABLE = True

import jax

jax.config.update("jax_enable_x64", True)


def require_requested_device() -> None:
    """Skip the cuda series when this interpreter cannot see a GPU."""
    if _REQUESTED_PLATFORM != "cuda":
        return
    if _CUDA_UNAVAILABLE:
        raise NotImplementedError(
            "JAX_PLATFORMS=cuda but CUDA jax / NVIDIA device is not available"
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
