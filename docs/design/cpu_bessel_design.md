# Design Report: CPU-Based Bessel Functions for the EM Hot Path

**Date**: March 2026 (v2 — revised after review)  
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

## 2. Proposed Architecture

### 2.1 Design principles

Three principles guide this revision:

1. **`backend` parameter, not separate functions.** All functionality lives on existing classes/functions. A `backend='cpu'|'jax'` parameter selects the implementation. No standalone CPU functions in the public API.

2. **Only Bessel-related computation goes to CPU.** The quad-form calculations (`L⁻¹(x-μ)`, `‖z‖²`, etc.) remain in JAX — they are d-dimensional matrix operations that benefit from GPU when d is large. Only the `log_kv` calls (and the gradient/Hessian that depend on them) switch to CPU.

3. **Adding `backend` does not break JIT-ability.** The `backend` parameter is a Python-level string resolved before JAX tracing. When `backend='jax'` (default), all code paths remain fully traceable. When `backend='cpu'`, the code runs eagerly with scipy — appropriate since the EM loop is already a Python `for` loop.

### 2.2 `log_kv(v, z, backend='jax')` — unified entry point

The `log_kv` function gets a `backend` parameter. The internal pure-JAX implementation (with `@jax.custom_jvp`) is renamed to a private function:

```python
# normix/_bessel.py

def log_kv(v, z, backend='jax'):
    """
    log K_v(z) with backend selection.

    backend='jax' : pure-JAX, lax.cond regime selection, custom JVP.
                    JIT-able, differentiable. Default for log_prob, pdf, etc.
    backend='cpu' : scipy.special.kve, fully vectorized numpy.
                    Not JIT-able. Fast for EM hot path.
    """
    if backend == 'cpu':
        return _log_kv_cpu(v, z)
    return _log_kv_jax(v, z)


@functools.partial(jax.custom_jvp, nondiff_argnums=())
def _log_kv_jax(v, z):
    # ... existing pure-JAX implementation (unchanged) ...

@_log_kv_jax.defjvp
def _log_kv_jax_jvp(primals, tangents):
    # ... existing JVP (unchanged, but calls _log_kv_jax internally) ...


def _log_kv_cpu(v, z):
    """log K_v(z) via scipy.special.kve. Fully vectorized numpy."""
    import numpy as np
    from scipy.special import kve, gammaln

    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    v, z = np.broadcast_arrays(v, z)
    z = np.maximum(z, np.finfo(np.float64).tiny)

    result = np.log(kve(v, z)) - z

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

**Why this doesn't break JIT:** `log_kv(v, z)` (no `backend` argument) calls `_log_kv_jax`, which is unchanged — same `@jax.custom_jvp`, same JIT-ability. The `backend` keyword is resolved at Python time. Any existing code that calls `log_kv(v, z)` without `backend` is unaffected.

**Why not `nondiff_argnums` for `backend`:** The `@jax.custom_jvp` decorator requires fixed function signatures. Rather than fight with JAX's tracing for a keyword parameter, we keep the `@custom_jvp` on the private `_log_kv_jax` and use the public `log_kv` as a thin dispatcher.

### 2.3 `GIG.expectation_params(backend='jax')` — class method with backend

The `GIG` class overrides `expectation_params` from `ExponentialFamily` to accept a `backend` parameter:

```python
class GIG(ExponentialFamily):

    def expectation_params(self, backend='jax'):
        """
        η = [E[log X], E[1/X], E[X]].

        backend='jax' : jax.grad of log partition (JIT-able, default)
        backend='cpu' : analytical Bessel ratios via scipy.kve (fast)
        """
        if backend == 'cpu':
            return self._expectation_params_cpu()
        return jax.grad(self._log_partition_from_theta)(self.natural_params())

    def _expectation_params_cpu(self):
        """Analytical GIG expectations using scipy Bessel."""
        p = float(self.p)
        a = float(self.a)
        b = float(self.b)
        sqrt_ab = np.sqrt(a * b)
        log_sqrt_ba = 0.5 * (np.log(b) - np.log(a))

        log_kp     = log_kv(p, sqrt_ab, backend='cpu')
        log_kp_m1  = log_kv(p - 1.0, sqrt_ab, backend='cpu')
        log_kp_p1  = log_kv(p + 1.0, sqrt_ab, backend='cpu')

        E_inv_X = np.exp(log_kp_m1 - log_kp - log_sqrt_ba)
        E_X     = np.exp(log_kp_p1 - log_kp + log_sqrt_ba)

        eps = 1e-5
        log_kp_pe = log_kv(p + eps, sqrt_ab, backend='cpu')
        log_kp_me = log_kv(p - eps, sqrt_ab, backend='cpu')
        E_log_X = (log_kp_pe - log_kp_me) / (2.0 * eps) + log_sqrt_ba

        return jnp.array([E_log_X, E_inv_X, E_X])
```

**Why this doesn't break JIT:**
- `expectation_params()` (no `backend` argument) → `backend='jax'` → `jax.grad(...)` → fully JIT-able, same as before.
- `expectation_params(backend='cpu')` → runs scipy eagerly → not JIT-able, but the caller (EM E-step) is a Python loop.
- The `backend` parameter is a Python string, not a traced value. It's resolved before JAX sees the function. From JAX's perspective, calling `gig.expectation_params()` traces exactly the same code as before.
- The `GIG` class attributes (`p`, `a`, `b`) are unchanged — still scalar `jax.Array` fields, still a valid pytree.

**Batch version for the E-step** (classmethod on GIG):

```python
class GIG(ExponentialFamily):

    @staticmethod
    def expectation_params_batch(p, a, b, backend='jax'):
        """
        Vectorized η for arrays of (p, a, b), each shape (N,).
        Returns (N, 3) array.

        backend='jax' : vmap over scalar JAX grad
        backend='cpu' : vectorized scipy.kve (6 C-level array calls)
        """
        if backend == 'cpu':
            return GIG._expectation_params_batch_cpu(p, a, b)
        def _single(pi, ai, bi):
            return GIG(p=pi, a=ai, b=bi).expectation_params()
        return jax.vmap(_single)(p, a, b)

    @staticmethod
    def _expectation_params_batch_cpu(p, a, b):
        """Vectorized CPU path — 6 scipy.kve calls on (N,) arrays."""
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        sqrt_ab = np.sqrt(a * b)
        log_sqrt_ba = 0.5 * (np.log(b) - np.log(a))

        log_kp    = log_kv(p, sqrt_ab, backend='cpu')
        log_kp_m1 = log_kv(p - 1.0, sqrt_ab, backend='cpu')
        log_kp_p1 = log_kv(p + 1.0, sqrt_ab, backend='cpu')

        E_inv_X = np.exp(log_kp_m1 - log_kp - log_sqrt_ba)
        E_X     = np.exp(log_kp_p1 - log_kp + log_sqrt_ba)

        eps = 1e-5
        log_kp_pe = log_kv(p + eps, sqrt_ab, backend='cpu')
        log_kp_me = log_kv(p - eps, sqrt_ab, backend='cpu')
        E_log_X = (log_kp_pe - log_kp_me) / (2.0 * eps) + log_sqrt_ba

        return jnp.column_stack([
            jnp.asarray(E_log_X),
            jnp.asarray(E_inv_X),
            jnp.asarray(E_X),
        ])
```

### 2.4 `GIG.from_expectation(..., solver='cpu')` — M-step with backend

The existing `from_expectation` already has a `solver` parameter. We replace `'cpu_legacy'` (which depends on normix_numpy) with a self-contained `'cpu'` solver that uses `log_kv(..., backend='cpu')`:

```python
class GIG(ExponentialFamily):

    @classmethod
    def from_expectation(cls, eta, *, theta0=None, solver='newton', ...):
        ...
        if solver == 'cpu':
            theta_scaled = cls._solve_cpu(eta_scaled, theta0_scaled, tol)
        ...

    @classmethod
    def _solve_cpu(cls, eta, theta0, tol):
        """scipy L-BFGS-B with log_kv(backend='cpu') for gradient."""
        # Uses log_kv(..., backend='cpu') for objective and gradient.
        # Same analytical gradient as _analytical_grad_hess_phi,
        # but with scipy Bessel instead of JAX Bessel.
        ...
```

## 3. E-step Integration: Only Bessel Goes to CPU

### 3.1 The key insight

The E-step has two phases with different computational profiles:

| Phase | Operation | Dimension | GPU benefit? |
|-------|-----------|-----------|-------------|
| **Quad forms** | `z = L⁻¹(x-μ)`, `‖z‖²`, `‖w‖²` | d-dimensional matrix ops | Yes (large d) |
| **Bessel** | `GIG.expectation_params()` → `log_kv(p, √ab)` | Scalar per observation | No |

Only the Bessel phase should go to CPU. The quad forms stay in JAX.

### 3.2 E-step call flow with `backend`

```
NormalMixture.e_step(X, backend='cpu')
│
├── Phase 1: Quad forms in JAX (vmapped, GPU-accelerated)
│   jax.vmap(joint._quad_forms)(X)
│   → z2 (N,), w2 scalar   [JAX arrays, possibly on GPU]
│
├── Posterior GIG params (simple JAX arithmetic, stays on device)
│   p_post = p - d/2          (N,)
│   a_post = a + w2           (N,)
│   b_post = b + z2           (N,)
│
└── Phase 2: GIG expectations via CPU Bessel
    GIG.expectation_params_batch(p_post, a_post, b_post, backend='cpu')
    → transfers (N,) arrays to numpy
    → 6 × scipy.kve on (N,) arrays  [C-level vectorized]
    → transfers (N, 3) result back to JAX
```

Compare with the current `backend='jax'` flow:
```
NormalMixture.e_step(X, backend='jax')
│
└── jax.vmap(joint.conditional_expectations)(X)
    → Per observation (vmapped):
      ├── _quad_forms(x)           [JAX, single obs]
      ├── GIG(p_post, a_post, b_post)
      └── gig.expectation_params() [jax.grad → 5 log_kv calls]
```

### 3.3 Implementation in `NormalMixture.e_step`

```python
class NormalMixture(eqx.Module):

    def e_step(self, X, backend='jax'):
        if backend == 'cpu':
            return self._e_step_cpu(X)
        return jax.vmap(self._joint.conditional_expectations)(X)

    def _e_step_cpu(self, X):
        """
        E-step with Bessel computation on CPU, quad forms in JAX.
        """
        from normix.distributions.gig import GIG

        X = jnp.asarray(X, dtype=jnp.float64)
        j = self._joint

        # Phase 1: Quad forms in JAX (vmapped — benefits from GPU for large d)
        def _quad_scalars(x):
            z, w, z2, w2, zw = j._quad_forms(x)
            return z2, w2
        z2_all, w2_all = jax.vmap(_quad_scalars)(X)  # (N,), (N,)

        # Posterior GIG params (JAX arithmetic, stays on device)
        p_post, a_post, b_post = j._posterior_gig_params(z2_all, w2_all)

        # Phase 2: GIG expectations via CPU Bessel
        eta = GIG.expectation_params_batch(p_post, a_post, b_post, backend='cpu')

        return {
            'E_log_Y': eta[:, 0],
            'E_inv_Y': eta[:, 1],
            'E_Y': eta[:, 2],
        }
```

Each `JointNormalMixture` subclass implements `_posterior_gig_params`:

```python
# In JointGeneralizedHyperbolic:
def _posterior_gig_params(self, z2, w2):
    return (self.p - self.d / 2.0,
            self.a + w2,
            self.b + z2)

# In JointVarianceGamma:
def _posterior_gig_params(self, z2, w2):
    return (self.alpha - self.d / 2.0,
            2.0 * self.beta + w2,
            z2)

# In JointNormalInverseGamma:
def _posterior_gig_params(self, z2, w2):
    return (-self.alpha - self.d / 2.0,
            w2,
            2.0 * self.beta + z2)

# In JointNormalInverseGaussian:
def _posterior_gig_params(self, z2, w2):
    a_ig = self.lam / (self.mu_ig ** 2)
    return (-0.5 - self.d / 2.0,
            a_ig + w2,
            self.lam + z2)
```

Note: `w2` is the same for all observations (it's `‖L⁻¹γ‖²`, which doesn't depend on x). It comes out as shape `(N,)` from vmap but with identical values. We could optimize this by computing it once, but vmap handles it transparently.

### 3.4 What stays in JAX vs what goes to CPU

| Operation | Backend | Reason |
|-----------|---------|--------|
| `L⁻¹(x-μ)`, `‖z‖²`, `‖w‖²` | JAX (vmap) | d-dimensional matrix ops, GPU-friendly |
| `p_post`, `a_post`, `b_post` | JAX | Simple arithmetic on JAX arrays |
| `log_kv(p, √ab)` (6 calls) | CPU (scipy) | C-level vectorized, no per-element dispatch overhead |
| `log_prob(x)` | JAX | Uses `log_kv` with default `backend='jax'` |
| `_log_partition_from_theta` | JAX | Single source of truth, `jax.grad` target |

## 4. Impact on JIT-ability

### 4.1 Why `backend` does not break JIT

The `backend` parameter is a **Python-level string**, not a JAX-traced value. It controls which Python code path executes, identical to how `dtype` parameters or `if isinstance(...)` checks work in JAX code:

```python
# This is fine — Python-level dispatch before tracing:
def log_kv(v, z, backend='jax'):
    if backend == 'cpu':    # Python bool, resolved before trace
        return _log_kv_cpu(v, z)
    return _log_kv_jax(v, z)
```

When `backend='jax'` (the default), JAX traces `_log_kv_jax` — the exact same function it traces today. The `if backend == 'cpu'` branch is never entered, never traced, never compiled. It's dead code from JAX's perspective.

When `backend='cpu'`, the scipy code runs eagerly in Python. JAX never sees it. This is only used inside the EM loop, which is already a Python `for` loop.

### 4.2 What remains JIT-able

All existing methods work exactly as before when called without `backend` or with `backend='jax'`:

```python
gig = GIG(p=1.0, a=2.0, b=3.0)

# All JIT-able (same as before):
jax.jit(gig.log_partition)()
jax.jit(gig.expectation_params)()              # default backend='jax'
jax.grad(gig._log_partition_from_theta)(theta)
jax.hessian(gig._log_partition_from_theta)(theta)
jax.vmap(gh.log_prob)(X)

# NOT JIT-able (EM hot path — never was JIT-able):
gig.expectation_params(backend='cpu')
GIG.expectation_params_batch(p_arr, a_arr, b_arr, backend='cpu')
```

### 4.3 Pytree structure unchanged

The `GIG` class stores `(p, a, b)` — all `jax.Array` scalars. Adding a `backend` parameter to *methods* does not change the pytree structure. The class is still a valid `eqx.Module`:

```python
# This still works:
jax.tree.leaves(gig)     # [p, a, b]
jax.tree.map(f, gig)     # applies f to each leaf
eqx.tree_serialise_leaves("model.eqx", gig)
```

If we ever wanted the GIG instance to *remember* its preferred backend, we could add a static field:

```python
class GIG(ExponentialFamily):
    p: jax.Array
    a: jax.Array
    b: jax.Array
    backend: str = eqx.field(static=True, default='jax')
```

`eqx.field(static=True)` means `backend` is not a pytree leaf — it's metadata. JIT recompiles when `backend` changes (like `dtype` or `shape`), which is the correct behavior. But this is optional; passing `backend` as a method argument is simpler and more explicit.

## 5. M-step: GIG.from_expectation with `solver='cpu'`

For the GH M-step, `GIG.from_expectation` needs a CPU solver. The existing `solver='cpu_legacy'` delegates to normix_numpy. The new `solver='cpu'` is self-contained, using `log_kv(..., backend='cpu')` for the gradient and Hessian:

```python
class GIG(ExponentialFamily):

    @classmethod
    def _solve_cpu(cls, eta, theta0, tol):
        """
        scipy L-BFGS-B with analytical gradient via log_kv(backend='cpu').
        """
        import numpy as np
        from scipy.optimize import minimize

        eta_np = np.asarray(eta)
        theta0_np = np.asarray(theta0)

        def objective_and_grad(theta_np):
            # Compute ψ(θ) and ∇ψ(θ) using log_kv(backend='cpu')
            p = theta_np[0] + 1.0
            b = max(-2.0 * theta_np[1], 0.0)
            a = max(-2.0 * theta_np[2], 0.0)
            sqrt_ab = np.sqrt(a * b)

            psi = np.log(2) + float(log_kv(p, sqrt_ab, backend='cpu')) \
                  + 0.5 * p * (np.log(b) - np.log(a))
            obj = psi - np.dot(theta_np, eta_np)

            # Analytical gradient via Bessel ratios
            gig_tmp = cls(p=p, a=a, b=b)
            eta_hat = np.asarray(gig_tmp._expectation_params_cpu())
            grad = eta_hat - eta_np
            return obj, grad

        result = minimize(
            lambda t: objective_and_grad(t),
            theta0_np,
            jac=True,
            method='L-BFGS-B',
            bounds=[(-np.inf, np.inf), (-np.inf, 0), (-np.inf, 0)],
            options={'maxiter': 500, 'ftol': tol**2, 'gtol': tol},
        )
        return jnp.asarray(result.x, dtype=jnp.float64)
```

This replaces `cpu_legacy` (which imports normix_numpy) with a self-contained solver.

## 6. Summary of API Changes

### `log_kv`

```python
# Before:
log_kv(v, z)            # pure JAX, custom_jvp

# After:
log_kv(v, z)            # same as before (backend='jax' default)
log_kv(v, z, backend='cpu')  # scipy.kve, fast, not JIT-able
```

### `GIG`

```python
# Before:
gig.expectation_params()                          # jax.grad
GIG.from_expectation(eta, solver='cpu_legacy')     # normix_numpy dependency

# After:
gig.expectation_params()                           # same (backend='jax' default)
gig.expectation_params(backend='cpu')              # analytical scipy Bessel
GIG.expectation_params_batch(p, a, b)              # vmap (backend='jax' default)
GIG.expectation_params_batch(p, a, b, backend='cpu')  # vectorized scipy
GIG.from_expectation(eta, solver='cpu')            # self-contained scipy solver
```

### `NormalMixture.e_step`

```python
# Before:
model.e_step(X)                # jax.vmap over conditional_expectations

# After:
model.e_step(X)                # same (backend='jax' default)
model.e_step(X, backend='cpu') # quad forms in JAX + Bessel on CPU
```

### `BatchEMFitter`

```python
# Fitter controls execution strategy:
fitter = BatchEMFitter(e_step_backend='cpu', m_step_solver='cpu')
```

## 7. Expected Performance

### E-step (backend='cpu' — only Bessel on CPU)

| Phase | Operation | Backend | Expected time |
|-------|-----------|---------|---------------|
| Quad forms | `vmap(_quad_forms)` over N=2552 obs | JAX (GPU) | ~0.02s |
| Posterior params | arithmetic on (N,) arrays | JAX | ~0.001s |
| **Bessel** | 6 × `scipy.kve` on (N,) arrays | **CPU** | **~0.05s** |
| Transfer | (N,) to numpy + (N,3) back to JAX | PCIe | ~0.001s |
| **Total E-step** | | | **~0.07s** |

vs current: ~1.1s (GPU), ~1.0s (CPU).

The matrix operations (quad forms) benefit from GPU parallelism for large d, while the Bessel calls benefit from scipy's C-level vectorization. The hybrid approach gets the best of both worlds.

### M-step (GH, solver='cpu')

| Implementation | Expected time (warm-start) |
|---|---|
| **Proposed CPU solver** | ~0.5–1.0ms per call |
| `cpu_legacy` (current) | ~0.12s per call |
| JAX Newton (GPU) | ~7.1s per call |

### Per-iteration total (GH, 468 stocks)

| Component | Current (GPU) | Proposed (hybrid) | normix_numpy |
|---|---|---|---|
| E-step | 1.09s | ~0.07s | 0.07s |
| M-step | 7.10s | ~0.01s | 0.42s |
| Other | ~0.58s | ~0.30s | 0.07s |
| **Total** | **8.77s** | **~0.38s** | **0.56s** |

## 8. Implementation Plan

### Phase 1: `log_kv` with `backend` parameter

- Rename existing pure-JAX implementation to `_log_kv_jax` (private)
- Add `_log_kv_cpu` (lifted from `normix_numpy.utils.bessel.log_kv_vectorized`)
- Public `log_kv(v, z, backend='jax')` dispatches between them
- Unit tests: CPU vs JAX accuracy comparison

### Phase 2: `GIG` methods with `backend`

- `GIG.expectation_params(backend='jax')` — override base class
- `GIG._expectation_params_cpu()` — analytical Bessel ratios via `log_kv(backend='cpu')`
- `GIG.expectation_params_batch(p, a, b, backend='jax')` — batch version
- `GIG._solve_cpu(eta, theta0, tol)` — self-contained scipy L-BFGS-B
- Add `solver='cpu'` to `GIG.from_expectation`
- Unit tests: roundtrip η → θ → η, CPU vs JAX agreement

### Phase 3: E-step with `backend`

- Add `_posterior_gig_params(z2, w2)` to each `JointNormalMixture` subclass
- Add `NormalMixture.e_step(X, backend='jax')` with CPU path
- `_e_step_cpu`: quad forms in JAX vmap + `GIG.expectation_params_batch(backend='cpu')`
- Profile and validate against normix_numpy

### Phase 4: Fitter integration

- Add `e_step_backend` and `m_step_solver` to `BatchEMFitter`
- Default to `backend='cpu'` for EM, `solver='cpu'` for GH M-step
- Full regression tests: EM convergence, log-likelihood matches

## 9. Risks and Mitigations

### Risk 1: Device transfer overhead

The CPU E-step transfers `p_post, a_post, b_post` (3 arrays of shape (N,)) to numpy and the result (N, 3) back. For N=2552, this is ~80KB — negligible at PCIe 4.0 bandwidth (~25 GB/s): ~0.003ms.

The quad forms stay on GPU. Only the Bessel-related data crosses the device boundary.

### Risk 2: Breaking JIT-ability

`backend` is a Python-level string, not a traced value. When omitted (defaulting to `'jax'`), all code paths are identical to today. No regression.

### Risk 3: Accuracy differences

scipy.kve and pure-JAX `log_kv` use different algorithms. Both are accurate to ~12 digits for EM parameter ranges. Regression tests will verify.

### Risk 4: Maintenance of two paths

The CPU path is ~50 lines (6 scipy.kve calls + numpy arithmetic). The JAX path is ~200 lines (4 regimes, custom JVP). The CPU path is simple and stable.

## 10. Recommendations

1. **Use `backend` parameter everywhere.** `log_kv(v, z, backend=...)`, `GIG.expectation_params(backend=...)`, `e_step(X, backend=...)`. No standalone CPU functions in the public API.

2. **Only Bessel on CPU.** Quad forms (`L⁻¹(x-μ)`, norms) stay in JAX — they benefit from GPU. Only `log_kv` calls and the gradient/Hessian that depend on them switch to CPU.

3. **Default to `backend='jax'` for all methods.** The CPU path is opt-in, controlled by the caller. For EM, the `BatchEMFitter` opts in via `e_step_backend='cpu'`.

4. **Phase 1 first.** Adding `backend` to `log_kv` is the foundation. Everything else builds on it.
