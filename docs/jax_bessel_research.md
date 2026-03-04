# JAX Bessel Function & Optimization Ecosystem Research

**Date:** 2026-03-04  
**Context:** Evaluating feasibility of porting `normix` (currently NumPy/SciPy) to JAX for GPU acceleration and automatic differentiation.

---

## 1. Does JAX support $K_\nu(z)$ (modified Bessel function of the second kind)?

**No.** JAX's `jax.scipy.special` does **not** include `kv` or `kve`.

JAX's built-in Bessel function coverage is limited to the modified Bessel functions of the **first** kind:

| Function | Description |
|----------|-------------|
| `jax.scipy.special.i0` | $I_0(z)$ |
| `jax.scipy.special.i1` | $I_1(z)$ |
| `jax.scipy.special.i0e` | $I_0(z) e^{-\|z\|}$ (exponentially scaled) |
| `jax.scipy.special.i1e` | $I_1(z) e^{-\|z\|}$ (exponentially scaled) |
| `jax.lax.bessel_i0e` | Lower-level $I_0^e$ |
| `jax.lax.bessel_i1e` | Lower-level $I_1^e$ |
| `jax.scipy.special.bessel_jn` | $J_n(z)$ (Bessel of the first kind) |

There is an open feature request ([jax-ml/jax#9956](https://github.com/jax-ml/jax/issues/9956)) for adding `kv` to `jax.scipy.special`, and a related PR ([#17038](https://github.com/jax-ml/jax/pull/17038)) adding `j0`/`j1`/`jn`, but **neither `kv` nor `kve` has been merged**.

---

## 2. Can JAX differentiate through Bessel functions?

**Partially.** For the functions that *are* built in (`i0`, `i1`, `i0e`, `i1e`), JAX provides automatic differentiation. However, since `kv`/`kve` are not built in, there is nothing to differentiate through natively.

JAX provides two mechanisms for defining custom derivative rules:

- **`jax.custom_jvp`** — forward-mode (JVP) custom rules
- **`jax.custom_vjp`** — reverse-mode (VJP) custom rules

These would be required to make any custom `kv` implementation differentiable.

---

## 3. What about $\log K_\nu(z)$?

**Not available natively in JAX.** No `log_kv` function exists in `jax.scipy.special`.

The current `normix` implementation (`normix/utils/bessel.py`) uses `scipy.special.kve` with the identity:

$$\log K_\nu(z) = \log K_\nu^e(z) - z$$

where $K_\nu^e(z) = K_\nu(z) e^z$ is the exponentially scaled version. This cannot be directly ported to JAX without a JAX-compatible `kve`.

---

## 4. JAX-compatible libraries with Bessel $K_\nu$ support

### 4a. `logbesselk` (v3.4.0) — **Best option**

**Package:** [PyPI](https://pypi.org/project/logbesselk/) | [GitHub (tk2lab/logbesselk)](https://github.com/tk2lab/logbesselk)  
**Requirements:** Python ≥ 3.10, JAX ≥ 0.4  
**License:** Apache-2.0  

**Functions provided (JAX module: `logbesselk.jax`):**

| Function | Description |
|----------|-------------|
| `log_bessel_k(v, x)` | $\log K_\nu(x)$ |
| `bessel_ke(v, x)` | $K_\nu^e(x) = K_\nu(x) e^x$ |
| `bessel_kratio(v, x, d)` | $K_{\nu+d}(x) / K_\nu(x)$ |
| `log_abs_deriv_bessel_k(v, x, m, n)` | $\log\left|\frac{\partial^m}{\partial\nu^m}\frac{\partial^n}{\partial x^n} K_\nu(x)\right|$ |

**Key capability — gradients with respect to BOTH $\nu$ and $x$:**

```python
import jax
from logbesselk.jax import log_bessel_k as logk

v, x = 1.0, 1.0

# Gradient w.r.t. order v
dlogk_dv = jax.grad(logk, 0)(v, x)

# Gradient w.r.t. argument x
dlogk_dx = jax.grad(logk, 1)(v, x)

# JIT + vmap compatible
logk_vec_jit = jax.jit(jax.vmap(logk))
```

**Algorithm:** Uses numerical integration based on the integral representation of $K_\nu(x)$, with a fixed number of integration intervals for stable computation time. Published in *SoftwareX* (2022): "Fast parallel calculation of modified Bessel function of the second kind and its derivatives" (Takekawa).

**This is the only library that supports $\partial/\partial\nu$ differentiation of $K_\nu$ in JAX.**

### 4b. TensorFlow Probability — JAX Substrate

**Package:** `tensorflow-probability` (with JAX substrate)  
**Module:** `tfp.substrates.jax.math`

**Functions:**

| Function | Description | $\nabla_z$ | $\nabla_\nu$ |
|----------|-------------|:---:|:---:|
| `bessel_kve(v, z)` | $K_\nu^e(z)$ | ✅ | ❌ |
| `log_bessel_kve(v, z)` | $\log K_\nu^e(z)$ | ✅ | ❌ |
| `bessel_ive(v, z)` | $I_\nu^e(z)$ | ✅ | ❌ |
| `log_bessel_ive(v, z)` | $\log I_\nu^e(z)$ | ✅ | ❌ |
| `bessel_iv_ratio(v, z)` | $I_\nu(z) / I_{\nu-1}(z)$ | ✅ | ❌ |

**Critical limitation:** Gradients with respect to the order parameter $\nu$ are **not defined**. The TFP documentation explicitly states this. This means TFP is insufficient for GIG/GH distributions where $p$ (the order parameter) must be optimized.

**Distributions available in `tfp.substrates.jax.distributions`:**
- `NormalInverseGaussian` — with location, scale, tailweight, skewness
- `InverseGamma` — with concentration and scale
- No `GeneralizedInverseGaussian` distribution

### 4c. `spbessax`

**Package:** [PyPI](https://pypi.org/project/spbessax/)  
**Provides:** Spherical Bessel functions (not modified Bessel functions of the second kind)  
**Relevance:** Not applicable to our use case (GIG requires $K_\nu$, not $j_n$)

### 4d. `distrax` (DeepMind)

**Package:** `distrax` v0.1.7  
**Provides:** Probability distributions built on JAX  
**Bessel support:** None directly; depends on JAX/TFP for special functions  
**Relevance:** Could be useful as API inspiration but doesn't solve the Bessel function gap

---

## 5. Numerical stability challenges with Bessel functions in JAX

The core challenges are the same as in NumPy/SciPy, but amplified by JAX's constraints:

### 5a. Overflow/underflow
- $K_\nu(z) \to \infty$ as $z \to 0^+$ and $K_\nu(z) \to 0$ exponentially as $z \to \infty$
- Working in log-space ($\log K_\nu(z)$) is essential
- The exponentially scaled form $K_\nu^e(z) = K_\nu(z) e^z$ helps for large $z$

### 5b. Large order $\nu$
- For large $|\nu|$, $K_\nu(z)$ can be astronomically large or small
- Asymptotic expansions are needed: $K_\nu(z) \approx \frac{\Gamma(|\nu|)}{2}\left(\frac{2}{z}\right)^{|\nu|}$ for small $z$

### 5c. JAX-specific issues
- **No branching in JIT:** JAX's `jit` requires static control flow. The current NumPy implementation uses `if np.any(inf_mask):` which cannot be JIT-compiled. Must use `jnp.where` for conditional logic.
- **No in-place mutation:** `result[inf_mask] = approx` is not allowed in JAX. Must use functional updates (`result.at[inf_mask].set(approx)` or `jnp.where`).
- **Float64 precision:** JAX defaults to float32. Must explicitly enable float64 with `jax.config.update("jax_enable_x64", True)` for sufficient precision in Bessel function computation.
- **`pure_callback` limitations:** Using `jax.pure_callback` to wrap `scipy.special.kve` works but prevents GPU execution and breaks the autodiff graph.

### 5d. Research advances (2024)
A 2024 paper ("Accurate Computation of the Logarithm of Modified Bessel Functions on GPUs", ICS 2024) presents GPU-optimized algorithms for $\log K_\nu(x)$ and $\log I_\nu(x)$ achieving:
- Machine-precision accuracy across all parameter ranges
- 45–77× speedup over SciPy on GPU
- No underflow failures

This suggests that high-quality GPU implementations are feasible but require custom XLA kernels.

---

## 6. How does TensorFlow Probability handle Bessel functions in JAX mode?

TFP uses a **substrate architecture**: the same Python source code runs on TensorFlow, JAX, or NumPy backends by swapping the underlying array library.

For Bessel functions specifically:
- `tfp.substrates.jax.math.bessel_kve` computes $K_\nu^e(z)$ using a pure-JAX implementation (polynomial/rational approximations)
- Custom gradients are defined using `@tfp_custom_gradient` (TFP's wrapper around `jax.custom_vjp`)
- Gradient w.r.t. $z$: defined via the recurrence $K'_\nu(z) = -\frac{1}{2}(K_{\nu-1}(z) + K_{\nu+1}(z))$
- Gradient w.r.t. $\nu$: **not implemented** (explicitly documented as undefined)

The lack of $\nabla_\nu$ support is a fundamental limitation for our use case where the GIG order parameter $p$ needs to be fitted.

---

## 7. Custom derivative implementations for Bessel functions

### Derivative w.r.t. $x$ (argument)

Well-established recurrence relation:

$$\frac{d}{dx} K_\nu(x) = -\frac{1}{2}\left(K_{\nu-1}(x) + K_{\nu+1}(x)\right)$$

Therefore:

$$\frac{d}{dx} \log K_\nu(x) = -\frac{1}{2}\left(e^{\log K_{\nu-1}(x) - \log K_\nu(x)} + e^{\log K_{\nu+1}(x) - \log K_\nu(x)}\right)$$

This is already implemented in `normix/utils/bessel.py` as `log_kv_derivative_z`. Could be ported to JAX using `custom_vjp`.

### Derivative w.r.t. $\nu$ (order)

**No closed-form recurrence exists.** DLMF §10.38 provides integral representations but they are not simple. Approaches:

1. **Finite differences** (current `normix` approach):
   $$\frac{\partial}{\partial\nu}\log K_\nu(x) \approx \frac{\log K_{\nu+\varepsilon}(x) - \log K_{\nu-\varepsilon}(x)}{2\varepsilon}$$
   
2. **`logbesselk` approach:** Uses numerical integration of the integral representation with automatic differentiation through the integrand. This gives exact gradients (to numerical precision) via JAX's autodiff.

3. **Custom `jax.custom_vjp`** wrapping finite differences:
   ```python
   @jax.custom_vjp
   def log_kv(v, x):
       return _log_kv_impl(v, x)
   
   def log_kv_fwd(v, x):
       val = log_kv(v, x)
       return val, (v, x, val)
   
   def log_kv_bwd(res, g):
       v, x, val = res
       # d/dv via finite differences
       dv = (log_kv(v + eps, x) - log_kv(v - eps, x)) / (2 * eps)
       # d/dx via recurrence
       dx = _log_kv_deriv_x(v, x, val)
       return (g * dv, g * dx)
   
   log_kv.defvjp(log_kv_fwd, log_kv_bwd)
   ```

---

## 8. JAX constrained optimization

### `jax.scipy.optimize.minimize`

- **Only BFGS** is supported (no L-BFGS-B, no bounds, no constraints)
- Does not support differentiation through the solver
- Does not support multi-dimensional array parameters
- Computes gradients via JAX autodiff automatically

### JAXopt (v0.8) — **Recommended**

| Solver | Constraint type | Notes |
|--------|----------------|-------|
| `jaxopt.BFGS` | Unconstrained | Native JAX implementation |
| `jaxopt.LBFGS` | Unconstrained | Native, memory-efficient |
| `jaxopt.LBFGSB` | Box constraints | Native L-BFGS-B |
| `jaxopt.ScipyBoundedMinimize` | Box constraints | Wraps `scipy.optimize.minimize` |
| `jaxopt.ScipyMinimize` | Unconstrained | Wraps SciPy, more methods |
| `jaxopt.ProjectedGradient` | Convex constraints | Projected gradient descent |
| `jaxopt.MirrorDescent` | Convex constraints | Mirror descent |
| `jaxopt.OptaxSolver` | Unconstrained | Wraps Optax optimizers |

**Key advantage:** JAXopt solvers support implicit differentiation through the optimization solution (differentiating w.r.t. hyperparameters).

**For our use case (GIG/GH parameter fitting):** `jaxopt.LBFGSB` or `jaxopt.ProjectedGradient` with `projection_box` would replace `scipy.optimize.minimize(method='L-BFGS-B', bounds=...)`.

**Important:** L-BFGS-B in JAXopt requires `float64` (FORTRAN backend).

---

## 9. JAX optimization ecosystem summary

| Library | Focus | Key Use Case |
|---------|-------|-------------|
| **Optax** (v0.2.7) | Gradient transforms, SGD-style | Neural network training, composable optimizers |
| **JAXopt** (v0.8) | Classical optimization | L-BFGS, constrained optimization, differentiable solvers |
| `jax.scipy.optimize` | Minimal wrapper | Simple unconstrained BFGS |
| **Optimistix** | Root finding + optimization | Nonlinear least squares, Newton methods |

For `normix`, the relevant library is **JAXopt** for EM parameter updates (constrained optimization steps) and potentially **Optax** for gradient-based MLE.

---

## 10. Implementing $\frac{\partial}{\partial\nu}\log K_\nu(x)$ and $\frac{\partial}{\partial x}\log K_\nu(x)$ in JAX

### Option A: Use `logbesselk` directly (recommended for prototyping)

```python
import jax
from logbesselk.jax import log_bessel_k

# Both derivatives are automatic
grad_v = jax.grad(log_bessel_k, 0)   # ∂/∂v
grad_x = jax.grad(log_bessel_k, 1)   # ∂/∂x

# Works with jit and vmap
grad_v_jit = jax.jit(jax.vmap(grad_v))
```

### Option B: Custom implementation with `jax.custom_vjp`

For maximum control and to avoid the `logbesselk` dependency:

```python
import jax
import jax.numpy as jnp
from jax import custom_vjp

@custom_vjp
def log_kv(v, x):
    """Compute log K_v(x) using TFP's bessel_kve."""
    from tensorflow_probability.substrates.jax.math import bessel_kve
    return jnp.log(bessel_kve(v, x)) - x

def log_kv_fwd(v, x):
    val = log_kv(v, x)
    return val, (v, x, val)

def log_kv_bwd(res, g):
    v, x, log_kv_val = res
    
    # ∂/∂x: use recurrence K'_v(x) = -(K_{v-1}(x) + K_{v+1}(x))/2
    log_kv_m1 = log_kv(v - 1.0, x)
    log_kv_p1 = log_kv(v + 1.0, x)
    dx = -0.5 * (jnp.exp(log_kv_m1 - log_kv_val) + 
                 jnp.exp(log_kv_p1 - log_kv_val))
    
    # ∂/∂v: finite differences (no closed form)
    eps = 1e-4
    dv = (log_kv(v + eps, x) - log_kv(v - eps, x)) / (2 * eps)
    
    return (g * dv, g * dx)

log_kv.defvjp(log_kv_fwd, log_kv_bwd)
```

### Option C: Hybrid — TFP for forward, `logbesselk` for $\nabla_\nu$

Use TFP's numerically stable `bessel_kve` for evaluation but `logbesselk` for the order gradient, combining the strengths of both.

---

## Summary & Recommendations for `normix` JAX Port

### Bessel Function Strategy

| Requirement | Best Solution | Fallback |
|-------------|--------------|----------|
| $\log K_\nu(x)$ evaluation | `logbesselk.jax.log_bessel_k` | TFP `log_bessel_kve` (add $-x$) |
| $\partial/\partial x$ | `logbesselk` (automatic) | `custom_vjp` with recurrence |
| $\partial/\partial\nu$ | `logbesselk` (automatic) | `custom_vjp` with finite differences |
| $K_{\nu_1}(x)/K_{\nu_2}(x)$ | `logbesselk.jax.bessel_kratio` | Exp of log difference |
| JIT + vmap | ✅ All options | — |
| GPU support | ✅ `logbesselk` | TFP also GPU-compatible |

### Optimization Strategy

| Current (`scipy`) | JAX Replacement |
|-------------------|----------------|
| `scipy.optimize.minimize(method='L-BFGS-B', bounds=...)` | `jaxopt.LBFGSB` or `jaxopt.ScipyBoundedMinimize` |
| Unconstrained BFGS | `jaxopt.BFGS` or `jax.scipy.optimize.minimize` |
| EM parameter updates | Direct JAX (autodiff through M-step) |

### Key Dependencies for JAX Port

```
jax >= 0.4
jaxlib >= 0.4
logbesselk >= 3.4.0
jaxopt >= 0.8
tensorflow-probability[jax]  # optional, for distributions
```

### Critical JAX Porting Considerations

1. **Enable float64:** `jax.config.update("jax_enable_x64", True)` — required for numerical stability in Bessel functions and L-BFGS-B
2. **No in-place mutation:** Replace `result[mask] = value` with `jnp.where(mask, value, result)`
3. **Static control flow:** Replace `if np.any(mask):` with `jnp.where`-based logic
4. **Functional style:** All state updates must return new arrays; `cached_property` pattern needs rethinking for JIT
5. **`logbesselk` is the only library providing $\nabla_\nu \log K_\nu$**, which is essential for GIG/GH distributions where the order parameter $p$ is fitted
