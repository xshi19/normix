# Installation

normix requires **Python ≥ 3.12** and runs on CPU or GPU through JAX.

## From PyPI

```bash
pip install normix
```

Optional plotting helpers:

```bash
pip install "normix[plotting]"
```

The docs site header shows the version of the package used to *build* these
pages. PyPI releases can lag the docs branch — if `pip` installs an older
wheel than the header, prefer the from-source path below for the matching
code.

## From source (development)

```bash
git clone https://github.com/xshi19/normix
cd normix
uv sync
```

`uv sync` installs the locked dependency set into a project virtual environment.
Run anything in that environment with `uv run`, e.g. `uv run python`,
`uv run pytest`, `uv run jupyter lab`.

Editable install with pip instead of uv:

```bash
pip install -e .
```

## Float64 precision

normix relies on double precision throughout — the Bessel evaluations, the GIG
$\eta \mapsto \theta$ solve, and log-density arithmetic all lose accuracy in
float32. Enable it **before importing** normix:

```python
import jax
jax.config.update("jax_enable_x64", True)
import normix
```

If float64 is not enabled, normix emits a warning on import.

## Dependencies

| Package | Role |
|---|---|
| `jax` | Array computation, autodiff, JIT, `vmap` |
| `equinox` | Immutable pytree modules |
| `jaxopt` | L-BFGS/BFGS for the GIG $\eta \mapsto \theta$ solve |
| `tensorflow-probability` | Reference Bessel functions |
| `scipy` | CPU Bessel evaluation on the EM hot path |
| `optax` *(optional)* | Gradient-based fitting experiments |

## Optional extras

```bash
uv sync --extra docs --extra plotting   # Sphinx site + matplotlib helpers
```

Once installed, head to the {doc}`quickstart` for a 30-second example, or
{doc}`first_model` for a guided walkthrough.
