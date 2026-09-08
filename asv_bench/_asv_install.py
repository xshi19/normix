"""ASV ``install_command``: project wheel plus CUDA jax on Linux with a GPU.

ASV's uv plugin pip-installs ``jax==0.9.1`` from ``matrix.req`` *without*
extras. The project's ``cuda12`` extra is Linux-only (``pyproject.toml``
marker), but an already-installed jax will not grow extras unless we ask
for ``jax[cuda12]`` explicitly. Pin both here, lockstep with ``matrix.req``
and ``[tool.uv] constraint-dependencies``.

GitHub-hosted runners are Linux, so ``{wheel}[cuda12]`` would otherwise
pull CUDA jax on a machine with no GPU. Skip that extra when
``/dev/nvidia0`` and ``nvidia-smi`` are both absent; the cuda
``env_nobuild`` series then skips in ``benchmarks/__init__.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_JAX_PIN = "0.9.1"


def nvidia_gpu_present() -> bool:
    return os.path.exists("/dev/nvidia0") or shutil.which("nvidia-smi") is not None


def main(wheel: str) -> None:
    extra = (
        "[cuda12]"
        if sys.platform.startswith("linux") and nvidia_gpu_present()
        else ""
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{wheel}{extra}",
            "--force-reinstall",
        ]
    )
    if extra:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"jax[cuda12]=={_JAX_PIN}",
                f"jaxlib=={_JAX_PIN}",
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} WHEEL")
    main(sys.argv[1])
