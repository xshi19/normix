# EM Algorithm GPU Profiling: CPU vs GPU vs NumPy

**Date**: March 2026  
**Status**: Investigated, warm-start + pure-JAX Newton applied  
**Related files**: `scripts/profile_cpu_vs_gpu.py`, `scripts/profile_numpy_vs_jax.py`, `scripts/profile_numpy_quick.py`

## Summary

The GH (Generalized Hyperbolic) EM algorithm was observed to be very slow on GPU.
Profiling revealed that the bottleneck is the GIG η→θ optimization in the M-step.
Three solver variants were benchmarked:

| GH M-step variant | 468-stock CPU | 468-stock GPU | Status |
|-------------------|---------------|---------------|--------|
| Original (scipy multi-start, no warm-start) | 43.9s | 28.2s | baseline |
| Warm-start scipy (single start + `jax.grad`) | 1.78s | 1.28s | previous |
| **Pure-JAX Newton** (current) | **2.50s** | **3.52s** | **current** |
| NumPy reference | 0.42s | — | reference |

The pure-JAX Newton solver eliminates the scipy dependency on the EM hot path.
It is slightly slower than warm-start scipy because `jax.hessian` triggers many
`pure_callback` round-trips for the Bessel function. This overhead will vanish
once `log_kv` is reimplemented in pure JAX (see `jax_native_bessel_feasibility.md`).

## Experimental Setup

- **Data**: S&P 500 daily returns, 2552 observations × 468 stocks
- **GPU**: NVIDIA RTX 4090, JAX 0.9.1
- **Distributions tested**: GH, VG, NIG, NormalInverseGamma
- **Metric**: Average time per EM iteration (after warmup), broken down into E-step and M-step

## Results: Before Any Fix (scipy multi-start, no warm-start)

### Per-iteration timing (468 stocks, seconds)

| Dist | Sub | M-step type | CPU E | CPU M | CPU iter | GPU E | GPU M | GPU iter |
|------|-----|-------------|-------|-------|----------|-------|-------|----------|
| NIG | InvGaussian | closed-form | 0.12 | 1.24 | 1.42 | 0.07 | 0.02 | 0.10 |
| VG | Gamma | jaxopt LBFGS | 0.11 | 1.28 | 1.45 | 0.07 | 0.23 | 0.31 |
| NIG-α | InvGamma | jaxopt LBFGS | 0.12 | 1.30 | 1.47 | 0.07 | 0.23 | 0.32 |
| **GH** | **GIG** | **scipy L-BFGS-B** | **0.11** | **43.9** | **44.1** | **0.08** | **28.2** | **28.3** |

### Per-iteration timing (10 stocks, seconds)

| Dist | CPU iter | GPU iter | GPU/CPU |
|------|----------|----------|---------|
| NIG | 0.13 | 0.10 | 0.73× (GPU faster) |
| VG | 0.21 | 0.32 | 1.58× (GPU slower!) |
| NIG-α | 0.21 | 0.32 | 1.49× (GPU slower!) |
| **GH** | **73.1** | **36.1** | 0.49× |

### Subordinator `from_expectation` cost (isolated)

| Subordinator | Time/call | Ratio vs GIG |
|-------------|-----------|-------------|
| GIG (scipy L-BFGS-B, multi-start) | 61,487 ms | 1× |
| Gamma (jaxopt LBFGS) | 222 ms | 278× faster |
| InverseGamma (jaxopt LBFGS) | 229 ms | 268× faster |
| InverseGaussian (closed-form) | 3.3 ms | 18,867× faster |

## Results: Warm-Start Scipy (intermediate)

Changes: single-start scipy L-BFGS-B from current θ, explicit `jax.grad` for Jacobian.

### GH M-step only (seconds)

| Dim | CPU M (before) | CPU M (after) | GPU M (after) | Speedup (CPU) |
|-----|----------------|---------------|---------------|---------------|
| 10 | 72.9 | 1.38 | 2.42 | 53× |
| 468 | 43.9 | 1.78 | 1.28 | 25× |

## Results: Pure-JAX Newton (current implementation)

Changes: replaced scipy warm-start path with a damped Newton solver implemented
entirely in JAX (`jax.grad`, `jax.hessian`, `jax.lax.scan`, `jax.lax.while_loop`).
Uses exp-reparametrisation θ₂ = −exp(φ₂), θ₃ = −exp(φ₃) for unconstrained optimization.

### GH iteration comparison across all three versions (seconds)

| Variant | 10-stock CPU | 10-stock GPU | 468-stock CPU | 468-stock GPU |
|---------|-------------|-------------|---------------|---------------|
| Original (multi-start scipy) | 73.1 | 36.1 | 44.1 | 28.3 |
| Warm-start scipy | 1.49 | 2.52 | 2.03 | 1.38 |
| **Pure-JAX Newton** | **1.18** | **3.89** | **2.66** | **3.62** |
| NumPy reference | 0.01 | — | 0.66 | — |

### Full comparison — pure-JAX Newton (468 stocks)

| Dist | NumPy iter | JAX CPU iter | JAX GPU iter |
|------|-----------|-------------|-------------|
| NIG | 0.60 | 1.92 | 0.12 |
| VG | 0.54 | 1.65 | 0.36 |
| NIG-α | 0.56 | 1.71 | 0.37 |
| **GH** | **0.66** | **2.66** | **3.62** |

### Why pure-JAX Newton is slower than warm-start scipy

The Newton method computes the 3×3 Hessian via `jax.hessian(_log_partition)` at
every step. Each Hessian evaluation triggers ~9 calls to `log_kv` (second-order
chain rule through the 3D objective), each going through `pure_callback`. Combined
with the backtracking line search (`jax.lax.while_loop`) and scan loop, this creates
**more** callback round-trips than scipy's L-BFGS-B (which only needs function + gradient).

On **GPU** this is especially bad: each `pure_callback` forces a GPU→CPU→GPU sync,
and the Newton method does this ~15× per step (Hessian + gradient + function evals)
vs ~2× for L-BFGS-B (function + gradient). The `lax.while_loop` in backtracking
adds further per-iteration kernel launch overhead.

On **CPU**, the pure-JAX Newton is only ~30% slower than warm-start scipy and
actually faster at 10 dimensions (1.18s vs 1.49s), since the overhead per step
is lower with tiny matrices.

### Why this design is still correct

Despite being slower today, the pure-JAX Newton is the right architecture because:

1. **No scipy on the hot path**: The EM iteration loop has zero scipy/numpy dependency
2. **JIT-ready**: Once `log_kv` is pure JAX, the entire Newton solver can be JIT-compiled
3. **GPU-ready**: With pure-JAX Bessel, the Newton solver runs entirely on-device
4. **The performance gap will invert**: The current bottleneck is `pure_callback`, not
   the Newton method itself. A pure-JAX `log_kv` would make the Hessian evaluation
   nearly free (XLA can fuse the computation graph), turning the 3.5s into ~0.01s.

## Root Cause Analysis

### 1. GIG `from_expectation` was the dominant bottleneck

Before the fix, the GIG M-step consumed **99.7%** of each EM iteration:
- Multi-start optimization: ~15 starting points × L-BFGS-B convergence
- Each function evaluation triggered `pure_callback` to scipy for `log_kv`
- No warm-starting: each EM iteration solved from scratch

### 2. E-step uses `pure_callback` for Bessel function

All distributions share the same E-step cost (~0.07–0.12s) because:
- `GIG.expectation_params()` calls `jax.grad(_log_partition_from_theta)`
- The log partition involves `log_kv` which uses `jax.pure_callback` to scipy
- Each JVP evaluation triggers ~5 `log_kv` calls (primal + recurrence + FD for ∂/∂v)
- Under `jax.vmap`, this creates n_obs separate callback invocations

The numpy version's E-step is faster because it uses vectorized `scipy.special.kve`
calls directly on entire arrays (one call per batch, not per observation).

### 3. GPU helps with matrix operations, not with callbacks

At 468 dimensions, the GPU provides significant speedup for:
- NIG: 14× faster (M-step is pure matrix ops: Cholesky, solves, einsum)
- VG/NIG-α: 4.5× faster (mixed matrix ops + jaxopt iteration)
- GH: limited benefit (callback overhead dominates)

At 10 dimensions, the GPU is actually **slower** for VG/NIG-α because small 10×10
matrix operations don't amortize GPU kernel launch overhead.

### 4. NumPy remains faster for several reasons

| Aspect | NumPy | JAX |
|--------|-------|-----|
| E-step Bessel | Vectorized `kve` on full array | `vmap` over `pure_callback` per obs |
| M-step GIG gradient | Semi-analytical (exact ∂/∂z, FD ∂/∂v) | `jax.grad`/`jax.hessian` through `pure_callback` |
| M-step GIG warm-start | Always warm-starts from current θ | Now warm-starts (was cold multi-start) |
| M-step GIG starts | Single start | Single start (was 15+) |
| M-step matrix ops | NumPy/LAPACK | JAX XLA |

## What Remains Slow — The Bessel Callback Wall

After the warm-start fix, the remaining bottleneck is the **Bessel function callback**.
Every `log_kv` call goes through `jax.pure_callback` → numpy → scipy, creating
synchronization overhead especially on GPU. This affects:

- **E-step** (via `expectation_params` → `jax.grad` → `log_kv`): ~0.07–0.12s
- **M-step Newton** (via `jax.hessian` → many `log_kv` calls): ~1–4s
- **Log-likelihood** (via `vmap(log_prob)` → `log_kv`): ~0.02–0.04s

A pure-JAX implementation of `log_kv` would eliminate this bottleneck entirely,
making the Newton solver essentially free and enabling true GPU acceleration.
See `docs/tech_notes/jax_native_bessel_feasibility.md` for investigation.
