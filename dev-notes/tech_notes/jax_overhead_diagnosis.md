# JAX vs NumPy Overhead Diagnosis: GIG Solver

**Date**: March 2026  
**Status**: Investigated, root cause identified  
**Related files**: `scripts/benchmark_log_partition.py`, `normix/_bessel.py`, `normix/distributions/gig.py`

## Summary

The GIG η→θ optimization (M-step of GH EM) runs **0.9 ms in pure NumPy** but
**1.8–6.3 seconds in pure JAX** — a 2000–7000× slowdown. This is not a bug in
our code but a fundamental mismatch between the problem size and JAX's execution
model.

## Benchmark Results

Micro-benchmark on NVIDIA RTX 4090, JAX 0.9.1. Median times after warmup.

### Layer 1: Single `log_kv(v=0.5, z=1.0)` call

| Implementation | Time | Ratio vs NumPy |
|---|---:|---:|
| NumPy `scipy.special.kve` | **3.8 μs** | 1× |
| JAX JIT (CPU) | 73.7 μs | 19× |
| JAX JIT (GPU) | 1724 μs | 454× |
| JAX eager (GPU) | 379,328 μs | 99,824× |

The GPU overhead comes from kernel launch latency: even a trivial scalar
computation requires host→device dispatch + synchronization. CPU-JAX is only
19× slower than NumPy (dispatch overhead without GPU latency).

### Layer 2: Gradient ∇ψ(θ)

| Implementation | Time | Ratio vs NumPy |
|---|---:|---:|
| NumPy analytical (3 `log_kv` calls + FD) | **27.7 μs** | 1× |
| JAX JIT grad (CPU) | 94.5 μs | 3.4× |
| JAX JIT grad (GPU) | 2,798 μs | 101× |
| JAX eager grad (GPU) | 1,946,824 μs | 70,282× |

NumPy's analytical gradient calls `scipy.special.kve` 5 times in C. JAX JIT on
CPU is only 3.4× slower — the overhead is just XLA dispatch. But GPU dispatch
adds 30× on top. **Eager mode** (no JIT) is catastrophic: 1.9 seconds for a
single gradient evaluation because every JAX operation issues a separate GPU kernel.

### Layer 3: Hessian ∇²ψ(θ)

| Implementation | Time | Ratio vs NumPy |
|---|---:|---:|
| NumPy FD (9 `_log_partition` evals) | **113.6 μs** | 1× |
| JAX autodiff hess JIT (CPU) | 143.2 μs | 1.3× |
| JAX analytical hess JIT (CPU) | 160.5 μs | 1.4× |
| JAX analytical hess JIT (GPU) | 1,419 μs | 12× |
| JAX autodiff hess JIT (GPU) | 5,832 μs | 51× |
| JAX eager hess (GPU) | 10,494,692 μs | 92,427× |

**Critical finding**: the eager hessian (what actually executes inside `lax.scan`)
takes **10.5 seconds per evaluation**. With `scan_length=5`, that's 5 × 10.5s ≈ 52s
of computation, mostly wasted on tracing. The measured 1.8s for the full solver
suggests XLA does cache some intermediate traces, but the overhead is still enormous.

### Layer 4: Full η→θ solve

| Implementation | Time | Ratio vs NumPy |
|---|---:|---:|
| Legacy NumPy/SciPy | **0.91 ms** | 1× |
| scipy-wrapped JAX | 6.43 ms | 7× |
| JAX Newton analytical (scan=5, GPU) | 1,795 ms | 1,972× |
| JAX Newton autodiff (scan=5, GPU) | 6,309 ms | 6,932× |

## Root Cause: Three Layers of Overhead

### 1. GPU dispatch latency for tiny operations

The GIG problem is 3-dimensional: θ is a 3-vector, the Hessian is 3×3. GPU kernel
launch overhead (~1ms per dispatch) far exceeds the actual computation (~1μs).
This is confirmed by JAX's own documentation:

> "For a 1000×1000 matrix operation, once compiled, JAX/GPU runs ~30× faster than
> NumPy/CPU, but with 10×10 inputs, JAX/GPU runs 10× slower due to overhead."
> — [JAX Benchmarking Guide](https://docs.jax.dev/en/latest/benchmarking.html)

### 2. `lax.cond` inside `lax.scan` — known performance bug

Our `log_kv` uses `lax.cond` for branch selection (Hankel / Olver / Quadrature / Small-z).
[JAX issue #5986](https://github.com/google/jax/issues/5986) documents that
`lax.cond` inside `lax.scan` is **significantly slower** on GPU compared to
equivalent arithmetic. This is an open P2 bug since 2021, still unfixed.

### 3. No JIT boundary around the Newton solver

The Newton solvers (`_solve_jaxopt`, `_solve_newton_analytical`) are plain Python
functions, not wrapped in `jax.jit`. Each call to `GIG.from_expectation` re-traces
the entire `lax.scan` body. The module-level JIT wrappers (`_objective_jit`,
`_grad_objective`) are only used by the scipy-wrapped path.

The `lax.scan` does trace the body once per call and reuses it across iterations,
but each `from_expectation` invocation pays the full tracing cost again.

### CPU vs GPU: the GPU makes it worse

| Operation | NumPy | JAX CPU | JAX GPU |
|---|---:|---:|---:|
| `log_kv` | 3.8 μs | 73.7 μs | 1,724 μs |
| ∇ψ | 27.7 μs | 94.5 μs | 2,798 μs |
| ∇²ψ (autodiff) | 113.6 μs | 143.2 μs | 5,832 μs |

JAX on **CPU** is only 1.3–3.4× slower than NumPy for JIT'd operations — this is
just XLA dispatch overhead and is tolerable. The GPU adds 10–50× on top because
every tiny operation requires a separate GPU kernel launch.

## Implications for the EM Architecture

### The M-step (GIG solver)

The GIG η→θ optimization is inherently a tiny scalar problem. It will **never**
benefit from GPU acceleration. The two viable paths are:

1. **Keep scipy-wrapped JAX** (current): 6.4 ms per M-step, 7× slower than NumPy.
   Acceptable since the M-step runs once per EM iteration.
2. **Force CPU for the GIG solver**: Use `jax.device_put(theta, cpu_device)` before
   the solve, then transfer back. The JIT'd gradient on CPU takes only 94.5 μs,
   close to NumPy's 27.7 μs.

### The E-step (vmap over observations)

The E-step computes `GIG.expectation_params` for N=2552 observations via `jax.vmap`.
This is a **large batch** that should benefit from GPU parallelism — unlike the
scalar M-step. The key bottleneck there is `log_kv` dispatch overhead per
observation (1.7 ms × 2552 obs would be 4.3s sequentially, but `vmap` can
parallelize).

### Where GPU actually helps

GPU acceleration benefits the matrix operations in the E-step and M-step
(Cholesky, solves, einsum on d×d matrices where d=10–468). These are already
fast. The Bessel function is the only non-matrix-op bottleneck.

## Resolution: stable JIT-cached Newton (May 2026)

Root cause #3 above ("No JIT boundary around the Newton solver") has been
addressed. `normix/fitting/solvers.py::make_jit_newton_solver` returns a
`@jax.jit`-decorated Newton solve specialised to a fixed
`(f, grad_fn, hess_fn, bounds)` 4-tuple, and
`normix/distributions/generalized_inverse_gaussian.py` builds a module-level
instance (`_gig_jax_newton_jit`) that all warm-started `backend='jax',
method='newton'` calls share. The compiled XLA executable now caches across
EM iterations instead of being re-traced on every call.

The same fix was applied to `_newton_digamma` (Gamma / InverseGamma / VG /
NInvG) by adding `@jax.jit`.

Measured impact on RTX 4090, JAX 0.9.1
(`benchmarks/bench_jit_solvers.py`):

| Path | First call (compile + run) | Cached call | Per-call speedup |
|---|---:|---:|---:|
| GIG warm-start, symmetric | `3.05 s` | `24 ms` | `126×` |
| GIG warm-start, asymmetric | `41 ms` (cache hit) | `39 ms` | `1.05×` |
| Gamma `_newton_digamma`, target=0 | `253 ms` | `4.5 ms` | `56×` |

End-to-end EM (`benchmarks/bench_em_mixture.py --large`):

| Config | Old per-iter | New per-iter | Δ |
|---|---:|---:|---:|
| GH `cpu/jax/newton` | `3.29 s` | `1.73 s` | `-47.5 %` |
| GH `cpu/cpu/lbfgs`  | `383 ms` | `299 ms` | `-21.9 %` |

The 2-iteration GH JAX/JAX run still pays the one-time compile in iteration 1;
longer EM loops should approach the cached `~24 ms/call` floor and erase the
2000–7000× headline reported above. Root causes #1 (GPU dispatch latency for
3D problems) and #2 (`lax.cond`-in-`lax.scan` bug) remain — they affect the
absolute floor, not the per-call cache miss penalty.

## References

1. [JAX #5986](https://github.com/google/jax/issues/5986): `lax.cond` inside
   `lax.scan` significantly slower on GPU (open P2, 2021)
2. [JAX #24411](https://github.com/jax-ml/jax/issues/24411): JAX extremely slow
   on GPUs for small workloads
3. [JAX #20326](https://github.com/google/jax/issues/20326): JAX slower than NumPy
   for small-scale linear algebra in loops
4. [JAX Benchmarking Guide](https://docs.jax.dev/en/latest/benchmarking.html):
   per-operation overhead documentation
