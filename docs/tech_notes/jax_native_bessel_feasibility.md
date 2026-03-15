# Feasibility: JAX-Native Modified Bessel Function $K_v(z)$

**Date**: March 2026  
**Status**: Investigation complete, implementation pending  
**Related files**: `normix/_bessel.py`, `docs/tech_notes/tfp_bessel_crash_investigation.md`

## Motivation

The current `log_kv(v, z)` uses `jax.pure_callback` to `scipy.special.kve`. This
forces CPU execution on every call, creating a major performance bottleneck:

- **E-step**: `jax.vmap(GIG.expectation_params)` triggers ~5 callbacks per observation
  via the JVP chain (primal + recurrence for ∂/∂z + FD for ∂/∂v). For 2552 observations,
  this means ~12,760 CPU round-trips per E-step.
- **M-step**: GIG's L-BFGS-B optimizer evaluates `_log_partition_from_theta` (which
  calls `log_kv`) on every function and gradient evaluation.
- **GPU**: Callbacks force GPU→CPU→GPU synchronization, preventing any parallelism benefit.

A pure-JAX `log_kv` would make the entire EM pipeline JIT-compilable and GPU-native.

## Existing Implementations Surveyed

### 1. scipy.special.kve (AMOS Fortran library)

- **Algorithm**: Composite strategy from Donald Amos's 1986 ACM TOMS paper:
  - Miller backward recurrence for small z
  - Hankel asymptotic expansion for large z (DLMF 10.40.2)
  - Olver uniform asymptotic expansion for large v (DLMF 10.41)
  - Connection formulas and normalization
- **Code**: ~3000 lines of Fortran 77 (`zbesk.f`, `zbknu.f`, `zacon.f`, etc.)
- **Differentiable**: No (compiled Fortran, no autodiff graph)
- **Numerical stability**: Excellent. Handles all edge cases gracefully.
- **Reference**: Amos, D.E. (1986). "Algorithm 644: A portable package for Bessel
  functions of a complex argument and nonnegative order." ACM TOMS 12(3), pp. 265–273.

### 2. TensorFlow Probability `log_bessel_kve`

- **Algorithm**: Composite strategy combining:
  - Temme series for small z (Temme 1975)
  - Continued fraction (CF1/CF2) for moderate z (Campbell 1980)
  - Olver uniform expansion for v ≥ 50 (DLMF 10.41.4)
  - Forward recurrence to shift order
- **Code**: Pure Python/JAX (~1200 lines in `bessel.py`)
- **Differentiable**: Partially — ∂/∂z works, but **∂/∂v is NOT implemented**
  (explicit `TODO` comments in source). This is insufficient for our GIG log partition
  which requires ∂/∂v for the `p` parameter.
- **Numerical stability**: **Crashes with segfault on extreme parameters** that GIG
  optimization probes (see `tfp_bessel_crash_investigation.md`). The crash occurs in
  `_temme_series` for very small z values that arise during L-BFGS-B exploration.
  `jnp.where` cannot prevent this because JAX evaluates both branches.
- **Compatibility**: TFP 0.25.0 is incompatible with JAX ≥ 0.9.1 (removed internal APIs).
- **References**:
  - Temme, N.M. (1975). "On the numerical evaluation of the modified Bessel function
    of the third kind." J. Comput. Phys. 19, pp. 324–337.
  - Campbell, J.B. (1980). "On Temme's algorithm for the modified Bessel function of
    the third kind." ACM TOMS 6(4), pp. 581–586.
  - DLMF §10.41: https://dlmf.nist.gov/10.41

### 3. logbesselk (Takekawa 2022)

- **Algorithm**: Numerical integration of the integral representation:
  $$K_v(z) = \int_0^\infty e^{-z\cosh t} \cosh(vt)\, dt$$
  Uses pre-refined integration bounds and fixed-interval quadrature (Gauss–Legendre
  or trapezoidal). No branching between different regimes.
- **Code**: Pure Python, supports both JAX and TensorFlow backends (~300 lines core)
- **Differentiable**: **Fully differentiable in both v and z** — autodiff works
  naturally through the quadrature sum. This is the key advantage.
- **Numerical stability**: Unknown for extreme GIG parameters. The integral
  representation is inherently well-conditioned, but fixed quadrature may lose
  precision for very large v or z where the integrand is sharply peaked.
- **Performance**: Claims "less than half the computation time" vs traditional methods.
  The fixed number of quadrature points makes it embarrassingly parallel — ideal for GPU.
- **Reference**: Takekawa, T. (2022). "Fast parallel calculation of modified Bessel
  function of the second kind and its derivatives." SoftwareX 17, 100923.
  arXiv:2108.11560.

### 4. Asymptotic Expansion (DLMF 10.40.2, Wikipedia)

For large $z$, the modified Bessel function has the well-known expansion:
$$K_v(z) \sim \sqrt{\frac{\pi}{2z}} e^{-z} \sum_{k=0}^{K} \frac{a_k(v)}{z^k}$$
where $a_k(v) = \frac{\prod_{j=1}^k [4v^2 - (2j-1)^2]}{k!\, 8^k}$.

In log-space (which is what we need):
$$\log K_v(z) \approx \frac{1}{2}\log\frac{\pi}{2z} - z + \log\left(\sum_{k=0}^K \frac{a_k(v)}{z^k}\right)$$

- **Trivially implementable in pure JAX**: ~20 lines of code.
- **Differentiable**: Fully (polynomial operations, standard JAX primitives).
- **Accuracy**: Excellent for $z > \max(20, v^2/(K-1))$ with $K=9$ terms. This is
  already used in `normix_numpy/utils/bessel.py` for `log_kv_derivative_v`.
- **Limitation**: Only valid for large z. Needs another method for small/moderate z.

### 5. Small-z Asymptotic (DLMF 10.30.2)

For $0 < z \ll \sqrt{v+1}$:
$$K_v(z) \sim \frac{\Gamma(v)}{2}\left(\frac{2}{z}\right)^v \quad (v > 0)$$
$$K_0(z) \sim -\ln(z/2) - \gamma$$

This is already used as the overflow fallback in the current `_bessel.py` (lines 35–53).

## Feasibility Assessment: Pure-JAX `log_kv`

### Proposed Composite Strategy

Combine regime-specific methods, all implementable in pure JAX:

| Regime | Method | Accuracy | Complexity |
|--------|--------|----------|-----------|
| $z > \max(20, v^2/8)$ | Hankel asymptotic (DLMF 10.40.2) | ~15 digits | ~20 lines |
| $v \gg z$, $v > 50$ | Olver uniform (DLMF 10.41.4) | ~12 digits | ~80 lines |
| $0 < z < 1$, small $v$ | Temme series or power series | ~15 digits | ~100 lines |
| moderate $z$, moderate $v$ | Numerical integration (logbesselk) | ~12 digits | ~50 lines |

### Key Challenge: Branching in JAX

JAX's `jnp.where` evaluates **both branches** before selecting output. If one branch
would produce NaN/Inf or crash, the entire computation fails. This is exactly what
caused the TFP crash.

**Solution**: Use "safe-value substitution" — evaluate every branch with clamped
inputs that prevent overflow/underflow, then select the correct result:

```python
# Safe values prevent overflow in non-selected branches
z_safe_large = jnp.maximum(z, 20.0)  # safe for large-z branch
z_safe_small = jnp.clip(z, 1e-300, 1e3)  # safe for small-z branch

result_large = _hankel_asymptotic(v, z_safe_large)
result_small = _temme_series(v, z_safe_small)

use_large = z > threshold
return jnp.where(use_large, result_large, result_small)
```

This is the pattern already used successfully in `GIG._log_partition_from_theta`
for the Gamma/InverseGamma/Bessel branch selection.

### Derivative with respect to order v

This is the hardest part. None of the standard expansions directly give ∂/∂v:
- The Hankel expansion has $a_k(v)$ depending polynomially on $v^2$ — differentiable.
- The Temme series requires careful handling of $\Gamma(v)$ derivatives.
- The integral representation $\int_0^\infty e^{-z\cosh t} \cosh(vt)\, dt$ is
  trivially differentiable: $\partial/\partial v = \int_0^\infty e^{-z\cosh t} t\sinh(vt)\, dt$.

The current approach (FD with $\varepsilon=10^{-5}$) works well and could be retained
initially, with the pure-JAX primal replacing only the callback overhead.

### Recommended Implementation Path

**Phase 1** (high impact, low effort):
- Implement the Hankel large-z asymptotic in pure JAX (~20 lines)
- Use it when $z > \max(20, v^2/8)$ — covers most GIG E-step evaluations
  where $\sqrt{ab}$ is typically moderate-to-large
- Fall back to `pure_callback` for remaining cases
- This alone could eliminate ~70% of callback overhead

**Phase 2** (medium effort):
- Add the small-z Gamma/log asymptotic (already in `_bessel.py` lines 35–53)
- Add the Olver uniform expansion for large v
- This covers ~95% of evaluations in pure JAX

**Phase 3** (higher effort):
- Port the logbesselk numerical integration for the remaining moderate regime
- Or implement Temme series with safe-value substitution
- Goal: 100% pure JAX, zero callbacks

### Expected Impact

With a pure-JAX `log_kv`:
- **E-step**: Fully JIT-compilable, `vmap` runs natively on GPU — estimated 10–50×
  speedup for the E-step portion
- **M-step**: GIG optimizer evaluations stay on-device — estimated 2–5× speedup
- **GPU benefit**: No more GPU↔CPU sync overhead — the full EM pipeline can run
  on GPU without any host round-trips
- **JIT compilation**: The EM loop could be wrapped in `jax.lax.while_loop` for
  additional overhead reduction

## References

1. Amos, D.E. (1986). Algorithm 644: A portable package for Bessel functions of a
   complex argument and nonnegative order. ACM TOMS 12(3), 265–273.
2. Temme, N.M. (1975). On the numerical evaluation of the modified Bessel function
   of the third kind. J. Comput. Phys. 19, 324–337.
3. Campbell, J.B. (1980). On Temme's algorithm for the modified Bessel function of
   the third kind. ACM TOMS 6(4), 581–586.
4. Takekawa, T. (2022). Fast parallel calculation of modified Bessel function of the
   second kind and its derivatives. SoftwareX 17, 100923. arXiv:2108.11560.
5. DLMF §10.40 (asymptotic expansions): https://dlmf.nist.gov/10.40
6. DLMF §10.41 (uniform asymptotic expansions): https://dlmf.nist.gov/10.41
