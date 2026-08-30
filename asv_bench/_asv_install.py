"""ASV ``install_command``: project wheel plus CUDA jax on Linux.

ASV's uv plugin pip-installs ``jax==0.9.1`` from ``matrix.req`` *without*
extras. The project's ``cuda12`` extra is Linux-only (``pyproject.toml``
marker), but an already-installed jax will not grow extras unless we ask
for ``jax[cuda12]`` explicitly. Pin both here, lockstep with ``matrix.req``
and ``[tool.uv] constraint-dependencies``.
"""

from __future__ import annotations

import subprocess
import sys

_JAX_PIN = "0.9.1"


def main(wheel: str) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{wheel}[cuda12]",
            "--force-reinstall",
        ]
    )
    if sys.platform.startswith("linux"):
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
