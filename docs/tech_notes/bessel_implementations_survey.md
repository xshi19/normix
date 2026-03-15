# Survey of K_v(z) Implementations for Pure-JAX Reimplementation

**Date**: March 2026  
**Status**: Research  
**Related files**: `normix/_bessel.py`, `normix_numpy/utils/bessel.py`, `docs/tech_notes/tfp_bessel_crash_investigation.md`

## Motivation

The current `log_kv(v, z)` implementation in `normix/_bessel.py` uses `scipy.special.kve` via `jax.pure_callback` with `@jax.custom_jvp` for derivatives. This works correctly but has two drawbacks:

1. **Callback overhead**: Each evaluation exits the XLA computation graph, preventing full JIT optimization and adding per-call overhead.
2. **No GPU support**: `pure_callback` calls back to the CPU host, so the function cannot run on GPU/TPU accelerators.

This note surveys three existing implementations and the underlying mathematical methods to assess whether a pure-JAX `log_kv` is feasible.

---

## 1. scipy.special.kve (AMOS library)

### Algorithm

scipy wraps the Fortran AMOS library by Donald E. Amos, specifically the `zbesk` routine. The AMOS package implements a composite algorithm:

- **Miller's backward recurrence** for small arguments and moderate orders
- **Asymptotic expansion for large z** (Hankel's expansion, DLMF 10.40.2):
  $$K_\nu(z) \sim \sqrt{\frac{\pi}{2z}} e^{-z} \sum_{k=0}^{\infty} \frac{a_k(\nu)}{z^k}$$
  where $a_k(\nu) = \frac{\prod_{j=0}^{k-1}(4\nu^2 - (2j+1)^2)}{k! \, 8^k}$
- **Uniform asymptotic expansion for large order** (Olver's expansion, DLMF 10.41.4):
  $$K_\nu(\nu z) \sim \sqrt{\frac{\pi}{2\nu}} \frac{e^{-\nu\eta}}{(1+z^2)^{1/4}} \sum_{k=0}^{\infty} \frac{(-1)^k U_k(p)}{\nu^k}$$
  where $\eta = \sqrt{1+z^2} + \ln\frac{z}{1+\sqrt{1+z^2}}$ and $p = 1/\sqrt{1+z^2}$
- **Connection formulas** between I_v, K_v, and Wronskian identities
- **Overflow/underflow management** via the exponentially scaled form `kve(v,z) = K_v(z) * exp(z)`

### Differentiability

Not differentiable. AMOS is compiled Fortran; no autodiff graph is available. Derivatives must be supplied externally (as we do with `@jax.custom_jvp`).

### References

- D.E. Amos, "Algorithm 644: A portable package for Bessel functions of a complex argument and nonnegative order", ACM TOMS 12(3), 1986.
- AMOS library: http://netlib.org/amos/

### Pure-JAX feasibility

**Not directly portable.** The AMOS code is ~3000 lines of Fortran with complex branching logic, goto statements, and machine-dependent constants. However, the *mathematical methods* it uses (Hankel expansion, Olver expansion) are well-documented and can be reimplemented.

### Known issues

- Extremely robust — handles all edge cases gracefully
- The exponentially scaled form `kve` prevents overflow for large z
- No known numerical failures in our test suite (51/51 tests pass)

---

## 2. TensorFlow Probability `log_bessel_kve`

### Algorithm

TFP implements a **composite pure-TF/JAX algorithm** that combines two methods with a branching strategy based on the order v:

**For v < 50: Temme's method** (N. Temme, 1975)
1. Decompose v = n + u where n = round(v) and |u| < 0.5
2. For |z| ≤ 2: **Temme series expansion** — a power series based on the representation of K_u(z) in terms of modified gamma-related functions:
   - Computes coefficients from the Chebyshev expansion of 1/Γ(1±v)
   - Iteratively sums terms until convergence (up to 1000 iterations)
3. For |z| > 2: **Continued fraction** via Steed's algorithm — evaluates the confluent hypergeometric function U(v+1/2, 2v+1, 2z) as a continued fraction:
   - Based on the identity $K_v(z) = \sqrt{\pi}(2z)^v e^{-z} U(v+\tfrac{1}{2}, 2v+1, 2z)$
   - Uses modified Lentz's method for numerical stability
4. **Forward recurrence** from K_u, K_{u+1} up to K_v using:
   $$K_{v+1}(z) = \frac{2v}{z} K_v(z) + K_{v-1}(z)$$

**For v ≥ 50: Olver's uniform asymptotic expansion** (DLMF 10.41.4)
- Uses 10 precomputed polynomial coefficients U_k(p)
- Evaluated via Horner's method

### Differentiability

**Partial.** Gradients with respect to z are implemented via the recurrence:
$$\frac{d}{dz} K_v^e(z) = \frac{z-v}{z} K_v^e(z) - K_{v-1}^e(z)$$

**Gradients with respect to v are NOT implemented.** The TFP source contains explicit TODOs:
```
# TODO(b/169357627): Implement gradients of modified bessel functions with
# respect to parameters.
```

This is a significant limitation for our use case, since the GIG log-partition function requires ∂/∂v log K_v(z).

### References

- N. Temme, "On the Numerical Evaluation of the Modified Bessel Function of the Third Kind", J. Comput. Phys. 19, 1975.
- J. Campbell, "On Temme's Algorithm for the Modified Bessel Function of the Third Kind", ACM TOMS, 1980.
- Numerical Recipes in C, 2nd Edition, 1992 (§6.7).
- DLMF §10.41 (Olver expansion).

### Pure-JAX feasibility

**Already pure JAX** (via TFP's JAX substrate). However, there are critical issues:

### Known issues

1. **Hard crashes on extreme parameters.** The Temme series and continued fraction code segfaults for certain (v, z) combinations that GIG optimization probes. See `docs/tech_notes/tfp_bessel_crash_investigation.md` for details.
2. **No ∂/∂v gradient.** Would need to be added externally (as finite differences or analytical formula).
3. **The v < 50 / v ≥ 50 threshold is somewhat arbitrary** — the Temme method can be slow to converge for v near 50.
4. **While_loop iterations** (up to 1000 for series, up to 1000 for continued fraction) can be slow under JIT compilation.
5. **JAX version incompatibility**: TFP 0.25.0 (latest) is incompatible with JAX ≥ 0.9.1 due to removed internal APIs.

---

## 3. logbesselk (Takekawa, 2022)

### Algorithm

The `logbesselk` library uses a fundamentally different approach: **numerical integration of the integral representation**:

$$K_v(z) = \int_0^\infty e^{-z \cosh t} \cosh(vt) \, dt$$

The key innovation is:

1. **Pre-refined integration range**: Instead of adaptive quadrature, the algorithm pre-determines the integration bounds based on (v, z) so that the integrand is negligible outside the range.
2. **Fixed-interval quadrature**: Uses a fixed number of quadrature points regardless of (v, z), stabilizing computation time and making it GPU-friendly.
3. **Log-space computation**: Works in log space throughout to avoid overflow/underflow.

The motivation for this approach is that traditional methods (series expansion + continued fraction + asymptotic expansion) have parameter regions where they are accurate but slow to converge, and the branching between methods is problematic for GPU parallelism (warp divergence).

### Differentiability

**Fully differentiable with respect to both v and z.** The integral representation is smooth in both parameters, and the fixed-point quadrature is a weighted sum of smooth functions, so autodiff works naturally.

Supports:
- `jax.grad` (both parameters)
- `jax.vmap`
- `jax.jit`
- `tf.GradientTape` (TF version)

### References

- T. Takekawa, "Fast parallel calculation of modified Bessel function of the second kind and its derivatives", SoftwareX 17, 100923, 2022. [arXiv:2108.11560](https://arxiv.org/abs/2108.11560)

### Pure-JAX feasibility

**Already pure JAX.** The JAX backend is a first-class implementation. The algorithm uses only standard operations (exp, cosh, log, weighted sums) — no callbacks or external libraries needed.

### Known issues / limitations

1. **Accuracy**: The paper claims accuracy comparable to AMOS, but the fixed quadrature has a precision ceiling that depends on the number of points.
2. **Overhead for scalar evaluation**: The fixed-interval approach is optimized for batched GPU computation; for scalar CPU evaluation, the overhead of computing all quadrature points may be higher than adaptive methods.
3. **Small library**: Only 8 GitHub stars, single author — limited community validation.
4. **Requires Python ≥ 3.10, JAX ≥ 0.4.**

---

## 4. Mathematical Background: Asymptotic Forms

### Large-z asymptotic (Hankel's expansion, DLMF 10.40.2)

For fixed v and z → ∞:

$$K_\nu(z) \sim \sqrt{\frac{\pi}{2z}} e^{-z} \sum_{k=0}^{\infty} \frac{a_k(\nu)}{z^k}$$

where $a_0 = 1$ and:

$$a_k(\nu) = \frac{(4\nu^2 - 1^2)(4\nu^2 - 3^2)\cdots(4\nu^2 - (2k-1)^2)}{k! \, 8^k}$$

In the exponentially scaled form:
$$K_\nu^e(z) = K_\nu(z) e^z \sim \sqrt{\frac{\pi}{2z}} \sum_{k=0}^{\infty} \frac{a_k(\nu)}{z^k}$$

**Properties:**
- Divergent series, but asymptotically valid for large z
- Excellent accuracy with just a few terms when z ≫ v²
- Each $a_k$ is a polynomial in v² of degree k, so **differentiable in v**
- Pure arithmetic — trivially implementable in JAX
- Already used in `normix_numpy/utils/bessel.py:_log_kv_deriv_v_asymptotic`

### Large-order uniform asymptotic (Olver, DLMF 10.41.4)

For v → ∞ with z/v = w fixed:

$$K_\nu(\nu w) \sim \sqrt{\frac{\pi}{2\nu}} \frac{e^{-\nu\eta}}{(1+w^2)^{1/4}} \sum_{k=0}^{\infty} \frac{(-1)^k U_k(p)}{\nu^k}$$

where:
- $p = (1+w^2)^{-1/2}$
- $\eta = \sqrt{1+w^2} + \ln\frac{w}{1+\sqrt{1+w^2}}$
- $U_k(p)$ are polynomials given by the recurrence in DLMF 10.41.9

**Properties:**
- Valid **uniformly** for all z > 0 when v is large
- Converges rapidly for large v — the "double asymptotic" property means it also works when z is large
- The U_k(p) polynomials are fixed and can be hardcoded (TFP uses 10 terms)
- Pure arithmetic — implementable in JAX

### Small-z behavior

For z → 0+ with v > 0 fixed:

$$K_\nu(z) \sim \frac{\Gamma(\nu)}{2} \left(\frac{2}{z}\right)^\nu$$

For v = 0: $K_0(z) \sim -\ln(z/2) - \gamma$ where γ is Euler's constant.

This is already implemented as the fallback in both our numpy and JAX bessel modules.

---

## Comparison Matrix

| Feature | scipy/AMOS (current) | TFP | logbesselk | Pure-JAX (proposed) |
|---------|---------------------|-----|------------|-------------------|
| **Method** | Composite (Miller + Hankel + Olver) | Composite (Temme + CF + Olver) | Numerical integration | Composite (Temme + Hankel + Olver) |
| **Pure JAX** | No (callback) | Yes | Yes | Yes |
| **GPU support** | No | Yes | Yes | Yes |
| **∂/∂z** | Via custom_jvp | Yes (recurrence) | Yes (autodiff) | Yes (recurrence) |
| **∂/∂v** | Via FD in custom_jvp | **No** | Yes (autodiff) | Via FD or analytical |
| **Extreme params** | Robust | **Crashes** | Unknown | Needs careful design |
| **Accuracy (typical)** | ~15 digits (f64) | ~7 digits (f32), ~14 (f64) | ~14 digits (f64) | Target: ~14 digits |
| **JIT overhead** | Callback overhead | while_loop unrolling | Fixed computation | Minimal |
| **External deps** | scipy | tensorflow-probability | logbesselk | None |

---

## Recommendations for Pure-JAX Implementation

### Proposed Strategy: Composite Algorithm

A pure-JAX `log_kv(v, z)` could combine:

1. **Small z, small v (z ≤ 2, |v| < 0.5)**: Temme series (from TFP, with improved convergence checks)
2. **Moderate z, small v (z > 2, |v| < 0.5)**: Continued fraction / Hypergeometric U (from TFP)
3. **Forward recurrence** from K_u to K_v for non-half-integer orders
4. **Large z (z ≫ v²)**: Hankel's asymptotic expansion (DLMF 10.40.2) — simplest and most reliable
5. **Large v (v ≥ 50)**: Olver's uniform asymptotic expansion (DLMF 10.41.4) — already proven in TFP

### Alternative: Integration Approach (from logbesselk)

The `logbesselk` approach of numerical integration is appealing because:
- Single code path (no branching)
- Naturally differentiable in both v and z
- GPU-friendly (fixed computation graph)
- Avoids the numerical pitfalls of series/CF convergence

However, it requires careful tuning of quadrature parameters and may not achieve the same precision as the composite approach.

### Derivative Strategy for ∂/∂v

Three options, in order of preference:
1. **Analytical via asymptotic expansion** (large z): Already implemented in `normix_numpy/utils/bessel.py:_log_kv_deriv_v_asymptotic`. Extends naturally to log space.
2. **Central finite differences** on log_kv (moderate z): Current approach in `normix/_bessel.py`. Simple and works with `@jax.custom_jvp`.
3. **Autodiff through integration** (if using logbesselk approach): Free from the integral representation.

### Key Risks

1. **Numerical edge cases**: The GIG distribution probes extreme parameter ranges during optimization. Any pure-JAX implementation must be tested against the full suite of 51 tests, especially `test_gig.py` and `test_generalized_hyperbolic.py`.
2. **JAX's `jnp.where` semantics**: Both branches are always evaluated, so guards like `jnp.where(z > threshold, asymptotic_result, series_result)` do not prevent NaN/Inf from propagating. Must ensure each branch returns finite values for all inputs, even outside its domain of accuracy.
3. **Convergence of iterative methods under JIT**: `jax.lax.while_loop` is needed for variable-iteration methods. Fixed-iteration alternatives (like logbesselk) are simpler for JIT.

### Next Steps

1. **Prototype**: Implement the Hankel large-z expansion as a pure-JAX function (simplest case, covers z > ~20)
2. **Validate**: Compare against scipy on the full (v, z) grid used by GIG
3. **Extend**: Add Olver expansion for large v, then Temme/CF for small z
4. **Evaluate logbesselk**: Install and benchmark against scipy for our parameter ranges
5. **Test**: Run full test suite with each candidate implementation

---

## DLMF References

- §10.25: Definitions of I_v, K_v
- §10.40: Asymptotic expansions for large argument (Hankel)
- §10.41: Asymptotic expansions for large order (Olver uniform)
- §10.74: Methods of computation (overview of all numerical approaches)
