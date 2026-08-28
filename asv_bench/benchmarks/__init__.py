"""ASV micro-suite for normix.

Enable float64 before any JAX work. Each benchmark module also sets this
because ASV may import files as top-level modules rather than as a package.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
