---
name: running-tests
description: Run and profile the normix test suite. Use when running pytest, choosing marker expressions, investigating slow tests, checking JAX compilation, or deciding whether to include slow, stress, integration, or GPU tests.
---

# Running Tests

## Default Command

Use the fast default suite unless the user asks for exhaustive validation:

```bash
uv run pytest tests/
```

The default pytest config excludes tests marked `slow`, `stress`,
`integration`, or `gpu`.

## Marker Suites

Use marker expressions for explicit test intent:

```bash
uv run pytest tests/ -m "smoke or contract"
uv run pytest tests/ -m "slow or stress or integration"
uv run pytest tests/ -m gpu
uv run pytest tests/ -m "not gpu"
```

Marker meanings:

- `smoke`: small API-level tests that should always be fast.
- `contract`: essential mathematical/API invariants.
- `slow`: tests with noticeably high runtime.
- `stress`: high-iteration, large-sample, or broad numerical stress tests.
- `integration`: larger workflow or real-data validation tests.
- `gpu`: tests that require or specifically exercise GPU behavior.

Importance and cost are orthogonal. A test may be both `contract` and `slow`.

## Profiling Recipe

Start with pytest durations:

```bash
uv run pytest tests/ --durations=50
```

For JAX compilation diagnostics:

```bash
JAX_LOG_COMPILES=1 uv run pytest tests/ --durations=50
```

For GPU allocator diagnostics:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests/ --durations=50
```

`--durations=50` prints the 50 slowest test calls. Use `--durations=0` only
when a complete timing report is needed.

## JAX Batching

Distribution `log_prob` and `pdf` methods are single-observation APIs. Batch
them with `jax.vmap`, for example:

```python
pdfs = jax.vmap(dist.pdf)(xs)
```

Do not pass a vector of univariate observations directly to `pdf`; vector-valued
single observations already mean something for multivariate distributions.
