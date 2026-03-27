# GIG Random Variate Generation: Algorithm Review

This document reviews algorithms for sampling from the **Generalized Inverse Gaussian (GIG)** distribution in a JAX-compatible setting. The GIG density is

$$
f(x \mid p, a, b) = \frac{(a/b)^{p/2}}{2\,K_p(\sqrt{ab})}\, x^{p-1}\,\exp\!\Bigl(-\tfrac{1}{2}(b/x + ax)\Bigr), \quad x > 0.
$$

## Context

All simpler subordinator distributions now use pure-JAX `rvs`:

| Distribution | JAX primitive | Notes |
|---|---|---|
| Gamma(α, β) | `jax.random.gamma` | Shape-rate: sample shape, divide by rate |
| InverseGamma(α, β) | `β / jax.random.gamma` | Reciprocal of Gamma |
| InverseGaussian(μ, λ) | Michael-Schucany-Haas (1976) | Normal + Uniform via `jnp.where` branching |

The GIG is the only remaining distribution that depends on `scipy.stats`. Replacing it with a JAX implementation would make the entire sampling chain JIT-able, enabling:

- **JIT compilation** of the full `rvs` path from subordinator through joint/marginal mixtures.
- **GPU-accelerated sampling** for large-scale Monte Carlo.
- **Differentiable sampling** (reparametrization gradients) if needed in the future.

---

## 1. What SciPy Uses

SciPy's `scipy.stats.geninvgauss._rvs` implements the algorithm from:

> W. Hörmann and J. Leydold, "Generating generalized inverse Gaussian random variates," *Statistics and Computing*, 24(4):547–557, 2014.

This is an **acceptance-rejection method** combining:

1. **Ratio-of-Uniforms (RoU)** variants due to Lehner (1989) — used when the log-density is T-concave (roughly when $p \geq 1$ or $\sqrt{ab} \geq 1/2$).
2. A **new rejection algorithm** for the non-T-concave regime — ensures a uniformly bounded rejection constant even when the GIG approaches a Gamma distribution.

The algorithm has **three parameter-dependent branches**:

| Regime | Condition (approx.) | Method |
|---|---|---|
| Region 1 | $p \geq 1$ or $ab \geq 1/4$ (T-concave) | RoU variant 1 (mode-shifted) |
| Region 2 | T-concave, alternative parametrization | RoU variant 2 (Lehner 1989) |
| Region 3 | Non-T-concave | Custom rejection with bounded constant |

### Pros
- Uniformly fast across all parameter ranges.
- Short setup time (good for varying-parameter use, e.g. Gibbs sampling).
- Well-tested in production (SciPy, R's `GIGrvg`).

### Cons
- Involves a **while loop** with data-dependent termination (acceptance-rejection).
- Multiple branches and mode computations.
- Not trivially portable to JAX's functional control flow (`lax.while_loop`).

---

## 2. Devroye (2014) Algorithm

> L. Devroye, "Random variate generation for the generalized inverse Gaussian distribution," *Statistics and Computing*, 24(2):239–246, 2014.

Devroye's method exploits the **log-concavity** of the GIG density. Since $\log f(x)$ is concave for all valid $(p, a, b)$ (when $p \geq 1$; for $p < 1$ with $ab > 0$ the density is still log-concave on its support), a universal envelope applies:

$$
f(x) \leq f(m)\,\min\bigl[1,\;\exp\bigl(1 - f(m)|x - m|\bigr)\bigr],
$$

where $m$ is the mode of $f$.

### Algorithm Sketch

1. Compute the mode $m$ of the GIG:
   $$m = \frac{(p-1) + \sqrt{(p-1)^2 + ab}}{a}$$
2. Compute $c = f(m)$ (the density at the mode).
3. **Rejection loop:**
   - Generate proposal from a uniform-exponential mixture envelope.
   - Accept/reject based on the density ratio.
   - Guaranteed acceptance rate $\geq 25\%$ for continuous log-concave densities.

### Pros
- Conceptually simple — a single algorithm for the entire parameter range.
- Guaranteed acceptance rate ≥ 25% (uniformly efficient).
- Only needs evaluation of $f(x)$ and its mode (no Bessel function *derivatives*).

### Cons
- Still uses a **while loop** (rejection sampling).
- The 25% acceptance rate, while guaranteed, is not as efficient as the Hörmann-Leydold method for typical parameters (which can achieve much higher acceptance rates in the T-concave regime).
- Requires evaluating the unnormalized density, which involves $x^{p-1} e^{-(b/x + ax)/2}$ — numerically delicate for extreme parameters.

---

## 3. Existing JAX Ecosystem

| Package | GIG support? | Notes |
|---|---|---|
| `jax.random` | **No** | Has `gamma`, `normal`, `uniform`, etc. |
| TensorFlow Probability (JAX substrate) | **No GIG** | Has `InverseGaussian`, `NormalInverseGaussian` |
| NumPyro | **No built-in GIG** | Has `InverseGamma`, etc. |

**No existing JAX package provides GIG sampling.** We must implement it ourselves.

---

## 4. JAX Implementation Challenges

### 4.1 Rejection Sampling in JAX

Acceptance-rejection algorithms require a **while loop** whose iteration count is random. In JAX:

- **`jax.lax.while_loop`**: Supports JIT compilation with dynamic termination. Each iteration must have fixed-shape state. This is the correct primitive for scalar rejection sampling.
- **Vectorized rejection**: For generating $n$ samples, a naive approach runs one `while_loop` per sample (via `vmap`). A more efficient approach uses **"rejection with padding"**: generate a batch of proposals, accept the valid ones, and loop only to fill remaining slots.

### 4.2 `jnp.where` vs `jax.lax.cond`

For the **parameter-dependent branching** (choosing which algorithm variant to use):

| | `jnp.where` | `jax.lax.cond` |
|---|---|---|
| Evaluates both branches | Yes (always) | No (only selected branch) |
| Vectorizable | Yes (elementwise) | No (scalar only) |
| Gradient-safe | Yes (via masking) | Yes |
| Best for | Array operations, sample-level branching | Scalar control flow, setup-phase branching |

**Recommendation**: Use `jax.lax.cond` for the **one-time setup** (choosing algorithm parameters based on $(p, a, b)$) and `jnp.where` for any **per-sample branching** within the rejection loop.

### 4.3 Bessel Function Evaluation

The unnormalized GIG log-density is:

$$\log \tilde{f}(x) = (p-1)\log x - \tfrac{1}{2}(b/x + ax)$$

This does **not** require Bessel function evaluation — only the normalization constant $K_p(\sqrt{ab})$ does, which is not needed for acceptance-rejection sampling (the ratio $f(x)/g(x)$ cancels the normalizing constant).

This is a significant advantage: **the rejection sampler only needs the unnormalized log-density**, avoiding Bessel evaluation entirely.

---

## 5. Recommended Approach

### Primary: Hörmann-Leydold (2014) via `lax.while_loop`

This is the same algorithm SciPy uses, and it is the most efficient for production use.

**Implementation plan:**

1. **Setup phase** (runs once per call to `rvs`):
   - Compute the mode $m$.
   - Based on $(p, a, b)$, select algorithm region (1, 2, or 3).
   - Compute the bounding rectangle parameters for the ratio-of-uniforms method.
   - Use `jax.lax.cond` or `jax.lax.switch` for region selection.

2. **Sampling phase** (runs $n$ times):
   - For each sample, run `lax.while_loop` with the RoU acceptance-rejection step.
   - Use `jax.vmap` over the sample dimension.

3. **Special-case routing**:
   - If $b \approx 0$ and $p > 0$: delegate to `Gamma(p, a/2).rvs` (already JAX).
   - If $a \approx 0$ and $p < 0$: delegate to `InverseGamma(-p, b/2).rvs` (already JAX).
   - If $p = -1/2$: delegate to `InverseGaussian.rvs` (already JAX).

### Alternative: Devroye (2014)

Simpler to implement as a first pass. Could serve as a fallback or reference implementation.

---

## 6. Mode of the GIG Distribution

The mode is needed by all rejection-based algorithms. For the GIG density:

$$
m = \frac{(p - 1) + \sqrt{(p-1)^2 + ab}}{a}
$$

Special cases:
- $a \to 0$: $m \to b / (2(1-p))$ (InverseGamma limit)
- $b \to 0$: $m \to (p-1)/a$ for $p > 1$, or $m \to 0^+$ for $p \leq 1$ (Gamma limit)

A numerically stable formulation for $p > 1$:

$$
m = \frac{b}{(1-p) + \sqrt{(p-1)^2 + ab}}
$$

This avoids catastrophic cancellation when $ab \ll (p-1)^2$.

---

## 7. Comparison Summary

| Algorithm | Efficiency | Simplicity | JAX-friendliness | Production-tested |
|---|---|---|---|---|
| Hörmann-Leydold (2014) | ★★★★★ | ★★★ | ★★★ | ★★★★★ (SciPy, R) |
| Devroye (2014) | ★★★★ | ★★★★★ | ★★★★ | ★★★ (MATLAB) |
| Inverse CDF (numerical) | ★★ | ★★★ | ★★ | ★★★ |

### Decision Factors

1. **Efficiency**: Hörmann-Leydold achieves higher acceptance rates in the T-concave regime (common in practice), but Devroye's guaranteed ≥25% is respectable.

2. **Implementation complexity**: Devroye requires ~50 lines; Hörmann-Leydold requires ~150 lines with three algorithm branches.

3. **JAX compatibility**: Both require `lax.while_loop`. Devroye's single-branch structure is slightly easier to JIT. Hörmann-Leydold's multi-branch structure needs `lax.switch` for the setup, which is straightforward.

4. **Numerical robustness**: Both avoid Bessel function evaluation in the inner loop. Hörmann-Leydold has more extensive testing across extreme parameters.

---

## 8. Recommendation

**Start with Devroye (2014)** for its simplicity and guaranteed efficiency, then optimize to Hörmann-Leydold if profiling shows the rejection rate is a bottleneck. Both algorithms:

- Avoid Bessel function evaluation in the rejection loop.
- Use only the unnormalized log-density $(p-1)\log x - (b/x + ax)/2$.
- Are compatible with `jax.lax.while_loop` + `jax.vmap`.

The special-case routing to Gamma / InverseGamma / InverseGaussian for degenerate parameters should be implemented regardless of which algorithm is chosen for the general case.

---

## References

1. Hörmann, W. and Leydold, J. (2014). "Generating generalized inverse Gaussian random variates." *Statistics and Computing*, 24(4):547–557.
2. Devroye, L. (2014). "Random variate generation for the generalized inverse Gaussian distribution." *Statistics and Computing*, 24(2):239–246.
3. Dagpunar, J. S. (1989). "An easily implemented generalized inverse Gaussian generator." *Communications in Statistics — Simulation and Computation*, 18(2):703–710.
4. Michael, J. R., Schucany, W. R., and Haas, R. W. (1976). "Generating random variates using transformations with multiple roots." *The American Statistician*, 30(2):88–90.
5. Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
6. Lambardi di San Miniato, M. and Kenne Pagui, E. C. (2022). "Scalable random number generation for truncated log-concave distributions." arXiv:2204.01364.
