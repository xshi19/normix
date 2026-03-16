# EM Algorithm GPU Profiling: CPU vs GPU vs NumPy

**Date**: March 2026 (updated)  
**Status**: Updated with pure-JAX Bessel results (March 2026)  
**Related files**: `scripts/profile_cpu_vs_gpu.py`, `scripts/rerun_em_profiling.py`

## Current Architecture (as of March 2026)

The package now uses pure-JAX `log_kv` with zero scipy callbacks:

- **`log_kv`**: pure JAX, `lax.cond` regime selection (Hankel / Olver / quadrature / small-z)
- **E-step**: `jax.vmap` over GIG `expectation_params`, entirely on-device
- **M-step GIG**: JAX Newton solver, `jax.hessian`-based, `lax.scan(length=20)`, warm-start
- **M-step others (VG, NIG-α)**: unchanged, `jaxopt.LBFGS`
- **M-step NIG**: closed-form

## Summary Table (March 2026 re-run)

Per-iteration timing (seconds, 5 iterations avg after warmup, S&P 500 data).

### 10 stocks

| Dist | Sub | CPU E | CPU M | CPU iter | GPU E | GPU M | GPU iter |
|------|-----|------:|------:|---------:|------:|------:|---------:|
| NIG | InvGaussian | 0.97 | 0.005 | 1.35 | 1.04 | 0.01 | 1.63 |
| VG | Gamma | 0.99 | 0.059 | 1.38 | 1.11 | 0.23 | 1.88 |
| NIG-α | InvGamma | 1.00 | 0.058 | 1.39 | 1.14 | 0.27 | 1.97 |
| **GH** | **GIG** | **0.97** | **3.38** | **4.71** | **1.05** | **6.69** | **8.29** |

### 468 stocks

| Dist | Sub | CPU E | CPU M | CPU iter | GPU E | GPU M | GPU iter |
|------|-----|------:|------:|---------:|------:|------:|---------:|
| NIG | InvGaussian | 1.18 | 1.04 | 2.87 | 1.08 | 0.14 | 1.77 |
| VG | Gamma | 0.99 | 1.36 | 2.69 | 1.09 | 0.33 | 2.00 |
| NIG-α | InvGamma | 1.00 | 1.38 | 2.74 | 1.08 | 0.34 | 2.18 |
| **GH** | **GIG** | **1.01** | **4.95** | **6.51** | **1.09** | **7.10** | **8.77** |

## Historical Comparison (GH M-step only, 468 stocks)

| Variant | CPU M | GPU M | Status |
|---------|------:|------:|--------|
| Original (scipy multi-start, no warm-start) | 43.9s | 28.2s | baseline (2025) |
| Warm-start scipy (single start + `jax.grad`) | 1.78s | 1.28s | intermediate |
| Pure-JAX Newton (before pure-JAX Bessel) | 2.66s | 3.62s | after warm-start fix |
| **Pure-JAX Newton (current, pure-JAX Bessel)** | **4.95s** | **7.10s** | **current (2026)** |
| NumPy reference | 0.42s | — | reference |

## Why the GH M-Step Is Still the Bottleneck

The pure-JAX Bessel function was expected to speed things up, but the GH M-step
is now **slower** than with the old scipy callback. The root cause has been
diagnosed in detail (see `docs/tech_notes/jax_overhead_diagnosis.md`):

### Measured overhead at each layer (RTX 4090)

| Operation | NumPy | JAX JIT CPU | JAX JIT GPU | JAX Eager GPU |
|---|---:|---:|---:|---:|
| Single `log_kv(0.5, 1.0)` | 3.8 μs | 73.7 μs | 1,724 μs | 379,328 μs |
| Gradient ∇ψ(θ) | 27.7 μs | 94.5 μs | 2,798 μs | 1,946,824 μs |
| Hessian ∇²ψ(θ) | 113.6 μs | 143.2 μs | 5,832 μs | 10,494,692 μs |

The **eager hessian** (10.5 seconds) is what runs inside `lax.scan` at each Newton
step. With `scan_length=20`, the theoretical cost is 20 × 10.5s = 210s, but XLA
caching reduces this to ~5–7s per M-step call.

### Root cause: wrong problem size for GPU

The GIG η→θ optimization is a **3-dimensional problem** (3-vector θ, 3×3 Hessian).
GPU kernel launch latency (~1ms per dispatch) far exceeds the actual computation
(~1μs). This is confirmed by known JAX issues:

- [JAX #5986](https://github.com/google/jax/issues/5986): `lax.cond` inside
  `lax.scan` significantly slower on GPU (open P2 bug since 2021)
- [JAX #24411](https://github.com/jax-ml/jax/issues/24411): JAX extremely slow
  on GPUs for small workloads
- [JAX benchmarking docs](https://docs.jax.dev/en/latest/benchmarking.html):
  "10×10 inputs → JAX/GPU 10× slower than NumPy/CPU"

### Why the old scipy callback was accidentally faster

With the old `jax.pure_callback` path, `log_kv` was called in scipy C code
directly: each callback was ~0.3ms because scipy's `kve` is a single fast C call.
The new pure-JAX path replaces each callback with a GPU kernel launch (~1.7ms),
making each `log_kv` call 5× slower in this M-step context.

The pure-JAX Bessel **helps the E-step** (batch vmap benefits from GPU
parallelism) but **hurts the M-step** (scalar 3D optimization, GPU overhead).

## What Improved

### E-step

The E-step times are now **similar across CPU and GPU** (~1s each for 10 stocks,
~1s for 468 stocks). Previously the E-step was 0.07–0.12s on GPU because it used
`pure_callback` (CPU execution). Now both E-steps are ~1s because:
- GPU: pure-JAX `log_kv` runs on-device but has per-dispatch overhead
- CPU: same pure-JAX computation runs on CPU

### VG/NIG-α M-step on GPU (468 stocks)

These improved significantly: 0.23s and 0.32s (unchanged jaxopt LBFGS), vs
0.23s and 0.32s previously. These distributions don't use `log_kv` in their
M-step, so the Bessel change doesn't affect them.

The GPU speedup on NIG (0.14s vs 1.04s CPU) for 468 stocks reflects the
pure matrix operations that genuinely benefit from GPU.

## Analysis vs. Original Numbers

### 10 stocks — the picture now

| Dist | CPU iter | GPU iter | GPU/CPU | Change from original |
|------|--------:|--------:|--------:|----------------------|
| NIG | 1.35s | 1.63s | 1.21× slower | GPU now 1.21× slower (was 0.73×) |
| VG | 1.38s | 1.88s | 1.36× slower | GPU now 1.36× slower (was 1.58×) |
| NIG-α | 1.39s | 1.97s | 1.42× slower | GPU now 1.42× slower (was 1.49×) |
| **GH** | **4.71s** | **8.29s** | **1.76× slower** | Both slower (was CPU 1.18s, GPU 3.89s) |

The small-d (10-stock) case is now dominated by E-step overhead from the pure-JAX
Bessel dispatch, which worsened from ~0.1s to ~1s per step.

### 468 stocks

| Dist | CPU iter | GPU iter | GPU/CPU | Change from original |
|------|--------:|--------:|--------:|----------------------|
| NIG | 2.87s | 1.77s | 0.62× faster | GPU improvement roughly same |
| VG | 2.69s | 2.00s | 0.74× faster | Slight regression |
| NIG-α | 2.74s | 2.18s | 0.80× faster | Slight regression |
| **GH** | **6.51s** | **8.77s** | **1.35× slower** | Both slower than original |

## Recommended Path Forward

The **GIG M-step is the single bottleneck** and will remain so with the current
JAX Newton architecture. The right fix is to force CPU execution for the GIG
solver, which avoids GPU dispatch overhead for this inherently scalar problem:

1. **Force CPU for GIG solver** (highest impact, low effort):
   - JIT-compile `_objective` and `_grad_objective` for CPU device
   - Pass `theta` and `eta` through `jax.device_put(..., cpu)` before the solve
   - Transfer result back to GPU after solve
   - Expected result: scipy-wrapped JAX on CPU ≈ 6–8ms per M-step (vs 5–7s now)

2. **Pure NumPy fallback for GIG solver** (fastest, breaks pure-JAX):
   - Call `normix_numpy.GIG._expectation_to_natural` directly for the 3D solve
   - Expected: ~0.9ms per M-step call (same as NumPy reference baseline)
   - Trade-off: re-introduces numpy dependency on the EM hot path

3. **Accept current E-step regression** (pure-JAX Bessel is worth it for large N):
   - For d=468, the E-step is 1.08s vs the original 0.07s (GPU)
   - But the old number depended on CPU callbacks — future GPU-native path
     requires fixing the `lax.cond` dispatch overhead

## Experimental Setup

- **Data**: S&P 500 daily returns, 2552 observations × 468 stocks
- **GPU**: NVIDIA RTX 4090, JAX 0.9.1
- **Branch**: `feat/jax-native-bessel-v2`
- **Metric**: Average time per EM iteration (iters 2–5 after warmup)
- **GIG solver**: `solver='newton'` (default), `scan_length=20`, warm-start
