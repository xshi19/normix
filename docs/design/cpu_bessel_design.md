# Design Report: CPU-Based Bessel Functions for the EM Hot Path

**Date**: March 2026  
**Branch**: `feat/jax-native-bessel-v2`  
**Status**: Design proposal  

## 1. Problem Statement

The EM algorithm for Generalized Hyperbolic (GH) and related distributions (NIG, VG, NormalInverseGamma) has two performance bottlenecks, both rooted in the Bessel function `log_kv`:

| Step | What calls `log_kv` | Call pattern | Current time (468 stocks) |
|------|---------------------|-------------|--------------------------|
| **E-step** | `GIG.expectation_params()` via `jax.grad(_log_partition_from_theta)` | N=2552 obs × vmap | ~1.1s (GPU), ~1.0s (CPU) |
| **M-step** (GH only) | `GIG.from_expectation()` — Newton/scipy optimizer | 1 scalar 3D problem | ~5–7s (GPU), ~5s (CPU) |

The E-step calls `log_kv` inside `jax.grad` of the GIG log-partition. The gradient requires 5 `log_kv` evaluations per observation (the primal + 4 for the JVP: `K_{v-1}`, `K_{v+1}`, `K_{v+ε}`, `K_{v-ε}`). For N=2552, that is ~12,760 scalar `log_kv` calls, currently dispatched via `jax.vmap`.

The normix_numpy reference implementation completes the E-step in **0.07s** (vectorized scipy.kve on CPU) vs 1.1s in JAX. The M-step completes in **0.42s** (scipy L-BFGS-B + scipy.kve) vs 5–7s in JAX.

### Root cause summary

1. **E-step**: The pure-JAX `log_kv` uses `lax.cond` for regime selection. When vmapped on GPU, each observation triggers separate kernel launches for the condition checks. The old `pure_callback` to scipy was faster because `kve` is a single C call per element.

2. **M-step**: The GIG η→θ problem is 3-dimensional. GPU kernel launch overhead (~1ms) dwarfs the actual computation (~1μs). This is a fundamental mismatch: JAX/GPU is designed for large parallel workloads, not tiny scalar optimizations.

## 2. Proposed Architecture: Dual `log_kv` Implementations

### 2.1 Two versions of `log_kv`

| Version | Location | Backend | Vectorization | Use case |
|---------|----------|---------|---------------|----------|
| `log_kv` (JAX) | `normix/_bessel.py` | Pure JAX, `lax.cond` + custom JVP | `jax.vmap` | `log_prob`, `pdf`, `cdf`, anything needing JAX autodiff/JIT |
| `log_kv_cpu` | `normix/_bessel_cpu.py` | NumPy + `scipy.special.kve` | NumPy broadcasting | EM E-step, GIG M-step, anything on the EM hot path |

#### `log_kv_cpu` specification

```python
# normix/_bessel_cpu.py

import numpy as np
from scipy.special import kve, gammaln

def log_kv_cpu(v, z):
    """
    log K_v(z) via scipy.special.kve. Fully vectorized over both v and z.

    Parameters
    ----------
    v : float or ndarray — order (any real)
    z : float or ndarray — argument (must be > 0)

    Returns
    -------
    result : float or ndarray
    """
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    v, z = np.broadcast_arrays(v, z)
    z = np.maximum(z, np.finfo(np.float64).tiny)

    result = np.log(kve(v, z)) - z

    # Handle underflow for small z
    inf_mask = np.isinf(result)
    if np.any(inf_mask):
        v_abs = np.abs(v[inf_mask])
        z_inf = z[inf_mask]
        large_v = v_abs > 0.5
        result[inf_mask] = np.where(
            large_v,
            gammaln(v_abs) - np.log(2.0) + v_abs * (np.log(2.0) - np.log(z_inf)),
            np.log(np.maximum(-np.log(z_inf / 2.0) - np.euler_gamma, 1e-300)),
        )
    return result
```

This is essentially `normix_numpy.utils.bessel.log_kv_vectorized` — the implementation already exists, just needs to be lifted into the `normix` package.

### 2.2 CPU-based GIG expectation parameters

The E-step requires `η = ∇ψ(θ)` for N posterior GIG distributions. Currently this is done via `jax.grad(_log_partition_from_theta)` + `jax.vmap`. The CPU alternative computes the gradient analytically using Bessel ratios:

```python
# normix/_gig_cpu.py

import numpy as np
from normix._bessel_cpu import log_kv_cpu

def gig_expectation_params_cpu(p, a, b):
    """
    Vectorized GIG expectation parameters η = [E[log X], E[1/X], E[X]].

    All inputs are arrays of shape (N,). Returns (N, 3) array.
    Uses Bessel ratios and the derivative formula from normix_numpy.
    """
    sqrt_ab = np.sqrt(a * b)
    log_sqrt_ba = 0.5 * (np.log(b) - np.log(a))

    log_kp = log_kv_cpu(p, sqrt_ab)
    log_kp_m1 = log_kv_cpu(p - 1.0, sqrt_ab)
    log_kp_p1 = log_kv_cpu(p + 1.0, sqrt_ab)

    E_inv_X = np.exp(log_kp_m1 - log_kp - log_sqrt_ba)
    E_X = np.exp(log_kp_p1 - log_kp + log_sqrt_ba)

    # E[log X] = ∂/∂p log K_p(√(ab)) + ½ log(b/a)
    eps = 1e-5
    log_kp_pe = log_kv_cpu(p + eps, sqrt_ab)
    log_kp_me = log_kv_cpu(p - eps, sqrt_ab)
    dlog_kv_dp = (log_kp_pe - log_kp_me) / (2.0 * eps)
    E_log_X = dlog_kv_dp + log_sqrt_ba

    return np.column_stack([E_log_X, E_inv_X, E_X])
```

Key point: **6 calls to `scipy.kve`** over arrays of shape (N,) — each call is a single C-level loop over the array. No Python per-element overhead. No JVP tracing. Expected time: ~0.05–0.1s for N=2552 (matching normix_numpy's 0.07s).

### 2.3 CPU-based GIG log-partition gradient and Hessian

For the M-step (GH distribution only), we need the gradient and Hessian of the GIG log-partition in the η→θ solver. The CPU version uses the same analytical formulas that `_gig_bessel_quantities` already implements, but with numpy/scipy:

```python
def gig_log_partition_grad_hess_cpu(theta):
    """
    Gradient and Hessian of ψ_GIG(θ) using scipy Bessel, for scalar θ.
    Returns (grad, hess) where grad is (3,) and hess is (3,3).
    """
    # Same logic as _gig_bessel_quantities + _analytical_grad_hess_phi,
    # but using log_kv_cpu instead of JAX log_kv.
    ...
```

This replaces the `_solve_cpu_legacy` approach (which delegates to `normix_numpy`) with a self-contained CPU solver inside `normix`. The benefit: no cross-package dependency on `normix_numpy` for the hot path.

## 3. Integration Points: Where CPU Versions Are Used

### 3.1 E-step: `conditional_expectations` → `GIG.expectation_params()`

The call chain is:

```
NormalMixture.e_step(X)
  → jax.vmap(joint.conditional_expectations)(X)    # vmapped over N obs
    → GIG(p_post, a_post, b_post).expectation_params()
      → jax.grad(_log_partition_from_theta)(theta)
        → log_kv(p, sqrt_ab)  [inside grad, triggers JVP]
```

The CPU version would replace the entire chain from `GIG.expectation_params()` down:

```
NormalMixture.e_step(X)
  → joint._conditional_expectations_batch_cpu(X)    # NEW: batch CPU path
    → compute p_post, a_post, b_post for all N obs  (vectorized numpy)
    → gig_expectation_params_cpu(p_post, a_post, b_post)
      → log_kv_cpu(...)  [6 scipy.kve calls on arrays of shape (N,)]
```

### 3.2 M-step: `GIG.from_expectation()` → η→θ solver

Already handled by `solver='cpu_legacy'`, but we would make a self-contained version:

```
GIG.from_expectation(eta, theta0=..., solver='cpu')    # NEW solver name
  → _solve_cpu(eta, theta0, tol)
    → scipy.optimize.minimize with gig_log_partition_grad_hess_cpu
```

### 3.3 Distributions affected

| Distribution | E-step uses `log_kv` via | M-step uses `log_kv` via |
|---|---|---|
| **GeneralizedHyperbolic** | `GIG.expectation_params()` | `GIG.from_expectation()` |
| **NormalInverseGaussian** | `GIG.expectation_params()` | No (closed-form IG) |
| **VarianceGamma** | `GIG.expectation_params()` | No (closed-form Gamma) |
| **NormalInverseGamma** | `GIG.expectation_params()` | No (closed-form InvGamma) |

All four distributions call `GIG.expectation_params()` in the E-step. Only GH calls `GIG.from_expectation()` in the M-step.

## 4. Impact on Distribution Classes

### 4.1 The core question: JIT-ability and JAX gradients

The distribution classes (`GIG`, `GeneralizedHyperbolic`, etc.) are `eqx.Module` pytrees. Their core exponential-family methods (`_log_partition_from_theta`, `natural_params`, `log_prob`) are designed to be JIT-compatible and support `jax.grad`.

**The CPU versions do NOT touch any of these methods.** The architecture is:

```
Distribution class (eqx.Module)
├── _log_partition_from_theta(theta)   ← pure JAX, JIT-able, autodiff-able
├── natural_params()                    ← pure JAX
├── log_prob(x)                         ← pure JAX, uses log_kv (JAX)
├── expectation_params()                ← jax.grad(_log_partition_from_theta)
│                                         [JAX path: still available, still JIT-able]
├── pdf(x), cdf(x)                     ← pure JAX
│
│   EM-specific methods (NOT JIT-able, Python-level):
├── e_step(X)                           ← currently uses vmap; NEW: batch CPU path
├── m_step(X, expectations)             ← currently Python loop; NEW: CPU solver
└── fit(X, ...)                         ← Python loop, calls e_step/m_step
```

**Recommendation**: The CPU path is an **alternative execution strategy** for the EM algorithm, not a replacement for the core distribution API. The classes remain fully JIT-able and autodiff-able for all other uses.

### 4.2 Design options for integrating the CPU E-step

#### Option A: CPU path inside `conditional_expectations` (recommended)

Add a `_conditional_expectations_batch_cpu` method to `JointNormalMixture` that accepts the full data matrix X (shape `(N, d)`) and returns expectations as numpy arrays, then converts to JAX:

```python
class JointNormalMixture(ExponentialFamily):

    def conditional_expectations_batch_cpu(self, X):
        """
        Batch CPU path for E-step. Vectorized numpy/scipy, no vmap.

        Returns dict of jax.Arrays (transferred from CPU numpy arrays).
        """
        X_np = np.asarray(X)
        # Vectorized quad forms
        r = X_np - np.asarray(self.mu)[None, :]
        L_np = np.asarray(self.L)
        z = scipy.linalg.solve_triangular(L_np, r.T, lower=True).T  # (N, d)
        w = scipy.linalg.solve_triangular(L_np, np.asarray(self.gamma), lower=True)
        z2 = np.sum(z**2, axis=1)  # (N,)
        w2 = np.dot(w, w)          # scalar

        p_post, a_post, b_post = self._posterior_gig_params_cpu(z2, w2)
        eta = gig_expectation_params_cpu(p_post, a_post, b_post)  # (N, 3)

        return {
            'E_log_Y': jnp.asarray(eta[:, 0]),
            'E_inv_Y': jnp.asarray(eta[:, 1]),
            'E_Y': jnp.asarray(eta[:, 2]),
        }
```

Then in `NormalMixture.e_step`:

```python
def e_step(self, X, *, backend='cpu'):
    if backend == 'cpu':
        return self._joint.conditional_expectations_batch_cpu(X)
    else:
        return jax.vmap(self._joint.conditional_expectations)(X)
```

**Advantages**:
- Clean separation: the `backend` parameter controls execution strategy
- Distribution classes remain untouched for non-EM use
- The `conditional_expectations` (single-obs, JAX) method still exists for JIT/grad use
- The batch CPU path is self-contained, easy to test

**Disadvantages**:
- Some code duplication (quad forms computed in both numpy and JAX)
- The CPU path returns `jnp.Array` from numpy data (one device transfer per E-step)

#### Option B: `jax.pure_callback` wrapper

Wrap the CPU `log_kv` as a `jax.pure_callback` inside `GIG.expectation_params()`, so the vmap path transparently dispatches to scipy:

```python
@jax.custom_jvp
def log_kv_callback(v, z):
    return jax.pure_callback(
        lambda v, z: log_kv_cpu(np.asarray(v), np.asarray(z)),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        v, z,
    )
```

**Advantages**:
- No change to the class hierarchy; `expectation_params()` just works
- The vmap still works (JAX handles batching of callbacks)

**Disadvantages**:
- `pure_callback` is NOT differentiable by default — would need a custom JVP that also uses callbacks. This is the old architecture we moved away from.
- Per-element callback overhead: ~0.3ms per call (measured in the old architecture). For N=2552, that's 0.3ms × 2552 × 5 ≈ 3.8s — worse than the current pure-JAX path.
- Fundamentally incompatible with the design philosophy of being "pure JAX, zero scipy callbacks."

**Verdict**: Option B is rejected. The callback overhead per element is too high, and it reintroduces the exact dependency we eliminated.

#### Option C: Hybrid — CPU for E-step, JAX for everything else

Same as Option A, but formalized as a pattern: the EM fitter chooses the backend, not the distribution:

```python
class BatchEMFitter(eqx.Module):
    e_step_backend: str = 'cpu'  # 'cpu' or 'jax'
    m_step_solver: str = 'cpu'   # 'cpu', 'newton', 'newton_analytical'

    def fit(self, model, X):
        for i in range(self.max_iter):
            expectations = model.e_step(X, backend=self.e_step_backend)
            model = model.m_step(X, expectations, solver=self.m_step_solver)
            ...
```

**Recommendation: Option C** (which subsumes Option A). The fitter owns the execution strategy. The distribution classes expose both paths but default to the one that makes sense.

### 4.3 What stays JAX-only

These methods remain pure JAX, fully JIT-able, and autodiff-compatible:

- `_log_partition_from_theta(theta)` — the single source of truth
- `natural_params()` — parameter conversion
- `log_prob(x)` / `pdf(x)` — density evaluation (uses JAX `log_kv`)
- `expectation_params()` — `jax.grad` of log partition (available but not used in EM hot path)
- `fisher_information()` — `jax.hessian` of log partition
- `log_prob_joint(x, y)` — joint density

These are the methods users might want to JIT or differentiate through for applications beyond EM (e.g., variational inference, score matching, custom losses).

### 4.4 What gets a CPU alternative

Only the EM-specific methods get CPU alternatives:

- `e_step(X, backend=...)` — CPU path via `gig_expectation_params_cpu`
- `m_step(X, expectations, solver=...)` — CPU path via `gig_log_partition_grad_hess_cpu` (GH only)

## 5. Expected Performance

### E-step

| Implementation | Mechanism | Expected time (N=2552, d=468) |
|---|---|---|
| **normix_numpy** (reference) | 6 × `scipy.kve` on (N,) arrays | **0.07s** |
| **Proposed CPU path** | Same as above | **~0.07–0.10s** |
| Current JAX (vmap, GPU) | `vmap(_log_kv_scalar)` × 5 per obs | ~1.1s |
| Current JAX (vmap, CPU) | Same, CPU device | ~1.0s |

The CPU path should achieve near-parity with normix_numpy because it uses the same scipy.kve calls. The small overhead comes from the `jnp.asarray` conversion back to JAX arrays.

### M-step (GH only)

| Implementation | Mechanism | Expected time (468 stocks) |
|---|---|---|
| **normix_numpy** (reference) | scipy L-BFGS-B + scipy.kve | **0.42s** |
| **Proposed CPU solver** (warm-start) | scipy L-BFGS-B + log_kv_cpu, 0–2 iters | **~0.5–1.0ms** |
| `cpu_legacy` (current) | delegates to normix_numpy | ~0.12s |
| JAX Newton (GPU) | `lax.scan` + `jax.hessian` | ~7.1s |

With warm-start from the previous EM iteration, the optimizer typically converges in 0–1 iterations, making each M-step call ~1ms.

### Per-iteration total (GH, 468 stocks)

| Component | Current (GPU) | Proposed (CPU E + CPU M) | normix_numpy |
|---|---|---|---|
| E-step | 1.09s | ~0.10s | 0.07s |
| M-step | 7.10s | ~0.01s | 0.42s |
| Other (regularize, LL check) | ~0.58s | ~0.30s | 0.07s |
| **Total per iteration** | **8.77s** | **~0.41s** | **0.56s** |

The proposed architecture would bring JAX normix to near-parity with the numpy reference for the EM algorithm, while retaining full JAX capabilities for all other uses.

## 6. Implementation Plan

### Phase 1: CPU Bessel module (`normix/_bessel_cpu.py`)

- Lift `log_kv_vectorized` from `normix_numpy.utils.bessel` into `normix/_bessel_cpu.py`
- Add `log_kv_derivative_v_cpu` and `log_kv_derivative_z_cpu` (from normix_numpy)
- Unit tests comparing CPU vs JAX versions

### Phase 2: CPU GIG expectations (`normix/_gig_cpu.py`)

- Implement `gig_expectation_params_cpu(p, a, b)` — vectorized over arrays
- Implement `gig_log_partition_grad_hess_cpu(theta)` — scalar, for M-step solver
- Implement `_solve_cpu(eta, theta0, tol)` — self-contained scipy L-BFGS-B solver
- Unit tests: roundtrip η → θ → η

### Phase 3: Batch CPU E-step

- Add `conditional_expectations_batch_cpu(X)` to `JointNormalMixture` (base)
- Implement `_posterior_gig_params_cpu(z2, w2)` in each joint subclass
- Add `backend` parameter to `NormalMixture.e_step`
- Profile and validate against normix_numpy

### Phase 4: Integration

- Add `e_step_backend` and `m_step_solver` to `BatchEMFitter`
- Default to `backend='cpu'` for EM, `solver='cpu'` for GH M-step
- Update `fit()` classmethods to pass through the backend parameter
- Full regression tests: EM convergence, log-likelihood matches

## 7. Risks and Mitigations

### Risk 1: Device transfer overhead

Converting between JAX arrays (possibly on GPU) and numpy arrays incurs a transfer cost. For the E-step, we transfer X (shape N×d) to CPU and results (shape N×3) back.

**Mitigation**: The transfer is O(Nd) floats, which is ~10MB for N=2552, d=468. At PCIe 4.0 bandwidth (~25 GB/s), this is ~0.4ms — negligible compared to the 1s+ savings.

### Risk 2: Breaking JIT-ability of the EM loop

The CPU path uses numpy/scipy, which cannot be JIT-compiled. The EM loop is already a Python `for` loop (not `jax.lax.while_loop`) precisely because the GIG solver uses scipy.

**Mitigation**: No regression. The EM loop was never JIT-able. The CPU path simply makes the non-JIT part faster.

### Risk 3: Accuracy differences between scipy.kve and pure-JAX log_kv

The two implementations use different algorithms. Small numerical differences could cause EM convergence differences.

**Mitigation**: Both implementations are accurate to ~12 digits for the parameter ranges encountered in EM. The EM algorithm is robust to such differences. We should add regression tests comparing the two paths.

### Risk 4: Maintenance burden of two code paths

Having both JAX and CPU versions of the Bessel function and GIG expectations means maintaining two implementations.

**Mitigation**: The CPU version is simple (6 calls to `scipy.kve` + basic numpy arithmetic). It's 50 lines of code, stable, and unlikely to need changes. The JAX version is the complex one (4 regimes, custom JVP, etc.).

## 8. Recommendations

1. **Adopt Option C**: The EM fitter controls the execution strategy via `e_step_backend` and `m_step_solver` parameters. Distribution classes expose both paths.

2. **Default to CPU for EM**: The `BatchEMFitter` should default to `e_step_backend='cpu'` and `m_step_solver='cpu'`. Users who want pure-JAX EM (e.g., for GPU-heavy workloads with very large N) can set `backend='jax'`.

3. **Keep the pure-JAX `log_kv`**: It remains the default for all non-EM uses (`log_prob`, `pdf`, gradient-based inference). The pure-JAX path is the right choice for density evaluation over large batches (where GPU parallelism helps) and for any use case requiring `jax.grad` or `jax.jit`.

4. **Self-contained CPU module**: Put the CPU implementations in `normix/_bessel_cpu.py` and `normix/_gig_cpu.py`, not in normix_numpy. This removes the cross-package dependency for the hot path.

5. **Phase the implementation**: Start with the E-step CPU path (Phase 1–3), which gives the biggest speedup for all four distributions. The M-step CPU solver (Phase 2) is only needed for GH and is less urgent since `cpu_legacy` already works.

## 9. Summary

The proposed dual-implementation architecture cleanly separates concerns:

- **Pure JAX path**: for `log_prob`, `pdf`, JIT-able operations, gradient-based inference — everything that benefits from JAX's tracing and GPU acceleration.
- **CPU path**: for EM-specific bulk computations (`expectation_params` over N observations, GIG η→θ solver) where scipy's C-level Bessel is 10–100× faster than JAX dispatch.

The distribution classes remain `eqx.Module` pytrees, fully JIT-able and autodiff-compatible. The CPU path is an optimization for the EM algorithm, controlled by the fitter, not baked into the distribution API.

Expected outcome: **~20× speedup** for GH EM iterations (8.77s → ~0.41s), bringing JAX normix to near-parity with the numpy reference.
