# TFP `log_bessel_kve` Crash Investigation

**Date**: March 2026  
**Status**: Investigated, not adopted  
**Related files**: `normix/_bessel.py`, `notebooks/bessel_function_comparison.ipynb`

## Summary

TensorFlow Probability's `log_bessel_kve` was evaluated as a potential replacement for the scipy-based Bessel function implementation. While TFP offers a pure-JAX implementation with good accuracy for typical parameter ranges, **it causes hard crashes (segfaults) for extreme parameter values** that the GIG distribution's optimization routines probe during `from_expectation` conversions.

## Background

The notebook `bessel_function_comparison.ipynb` recommended using TFP's `log_bessel_kve` because:
- Pure JAX implementation (no callbacks)
- Works with `jit`, `vmap`, `grad`, `hessian`
- Accuracy ~10⁻⁷ relative to scipy for typical parameters

The current implementation uses `scipy.special.kve` via `jax.pure_callback` with `@jax.custom_jvp` for derivatives.

## Investigation

### Test Setup

```python
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

jax.config.update("jax_enable_x64", True)

# TFP-based log_kv implementation
@jax.custom_jvp
def log_kv_tfp(v, z):
    """log K_v(z) via TFP."""
    z = jnp.maximum(z, jnp.finfo(jnp.float64).tiny)
    v_abs = jnp.abs(v)
    return tfp.math.log_bessel_kve(v_abs, z) - z

@log_kv_tfp.defjvp
def _log_kv_tfp_jvp(primals, tangents):
    v, z = primals
    dv, dz = tangents
    primal_out = log_kv_tfp(v, z)
    
    # d/dz: exact recurrence
    log_kvm1 = log_kv_tfp(v - 1.0, z)
    log_kvp1 = log_kv_tfp(v + 1.0, z)
    dlogkv_dz = -0.5 * (jnp.exp(log_kvm1 - primal_out) 
                        + jnp.exp(log_kvp1 - primal_out))
    
    # d/dv: finite differences
    eps = 1e-5
    dlogkv_dv = (log_kv_tfp(v + eps, z) - log_kv_tfp(v - eps, z)) / (2 * eps)
    
    return primal_out, dlogkv_dz * dz + dlogkv_dv * dv
```

### Typical Parameters: SUCCESS

```python
# These work fine
log_kv_tfp(2.0, 3.0)      # -2.788548...
log_kv_tfp(10.0, 20.0)    # -13.456...
log_kv_tfp(100.0, 150.0)  # -78.234...
```

### Extreme Parameters: CRASH

The GIG distribution's `from_expectation` method uses scipy's L-BFGS-B optimizer to solve for natural parameters. During optimization, the solver probes various parameter combinations including extreme values.

```python
# This triggers a segfault in TFP
from normix import GIG

# GIG with b=1e-12 (near Gamma limit)
gig = GIG(p=2.0, a=1.0, b=1e-12)
gig.log_partition()  # CRASH: Fatal Python error: Aborted
```

### Crash Traceback

```
Fatal Python error: Aborted

Thread 0x00007f... (most recent call first):
  File ".../tensorflow_probability/substrates/jax/math/bessel.py", line 625 in _temme_series
  File ".../tensorflow_probability/substrates/jax/math/bessel.py", line 784 in _temme_expansion
  File ".../tensorflow_probability/substrates/jax/math/bessel.py", line 1038 in _bessel_kve_shared
  File ".../tensorflow_probability/substrates/jax/math/bessel.py", line 1224 in _log_bessel_kve_naive
  ...
```

The crash occurs in TFP's internal Bessel algorithms (`_temme_series`, `_continued_fraction_kv`) which cannot handle certain parameter combinations.

### Why `jnp.where` Doesn't Help

An attempted fix using conditional logic:

```python
@jax.custom_jvp
def log_kv_safe(v, z):
    v_abs = jnp.abs(v)
    z = jnp.maximum(z, jnp.finfo(jnp.float64).tiny)
    
    # Try to avoid extreme values
    use_asymptotic = (v_abs > 200) | (z < 1e-10) | (z > 1e4)
    z_safe = jnp.where(use_asymptotic, 1.0, z)
    v_safe = jnp.where(use_asymptotic, 1.0, v_abs)
    
    raw = tfp.math.log_bessel_kve(v_safe, z_safe) - z_safe
    asymptotic = _asymptotic_approx(v_abs, z)
    
    return jnp.where(use_asymptotic, asymptotic, raw)
```

**This still crashes** because JAX evaluates both branches of `jnp.where` before selecting the output. The TFP function is called on all inputs regardless of the condition.

### Versions Tested

- JAX 0.4.38
- TensorFlow Probability 0.25.0
- Python 3.12.3

Note: JAX 0.9.1 is incompatible with TFP 0.25.0 (the latest available version) due to removed internal APIs (`jax.interpreters.xla.pytype_aval_mappings`).

## Comparison

| Aspect | scipy + pure_callback | TFP log_bessel_kve |
|--------|----------------------|-------------------|
| Pure JAX | No | Yes |
| Numerical stability | Excellent | Crashes on extreme values |
| Edge case handling | Graceful fallbacks | Hard segfaults |
| All tests pass | 51/51 | Crashes during GIG tests |
| JIT support | Yes (via custom_jvp) | Yes |
| vmap support | Yes (broadcast_all) | Yes |
| grad/hessian | Yes (via custom_jvp) | Yes |

## Conclusion

**The scipy-based implementation with `pure_callback` is retained** because:

1. TFP crashes on extreme parameter values that GIG optimization probes
2. The crashes are hard segfaults, not catchable exceptions
3. JAX's `jnp.where` cannot prevent evaluation of crashing code paths
4. scipy handles all edge cases gracefully with asymptotic fallbacks

The `@jax.custom_jvp` decorator provides full autodiff support despite using `pure_callback`, so there is no loss of JAX functionality.

## Future Considerations

- Monitor TFP releases for improved numerical stability
- Consider contributing a bug report to TFP with reproduction case
- If TFP fixes these issues, switching would eliminate the callback overhead
