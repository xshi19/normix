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

## 3. What the R `GeneralizedHyperbolic` Package Uses

The R package `GeneralizedHyperbolic` (Scott, Trendall & Luen) uses:

> Dagpunar, J. S. (1989). "An easily implemented generalised inverse Gaussian generator." *Commun. Statist. — Simula.*, 18:703–710.

This is the **original** ratio-of-uniforms algorithm for the GIG — the same method that Hörmann & Leydold (2014) later improved upon. Dagpunar's algorithm:

- Uses a single ratio-of-uniforms acceptance-rejection scheme.
- Is "easily implemented" (the paper's title), making it the historical default in many R and MATLAB packages.
- Has a **critical limitation**: the rejection constant becomes prohibitively large when the GIG approaches a Gamma distribution (i.e. $b \to 0$), making sampling extremely slow or even incorrect in the non-T-concave regime ($\lambda < 1$ and $\psi\chi < 1/4$).

The R `GIGrvg` package (also by Leydold) supersedes this with the Hörmann-Leydold (2014) three-region algorithm that fixes the non-T-concave problem.

The `GeneralizedHyperbolic` package's `qgig` (quantile function) is also noteworthy: it offers two methods:
- `method = "spline"` (default): builds a **cubic spline approximation** to $F^{-1}$ using 501 interpolation points, then uses `uniroot` on the spline. Falls back to integration for extreme tail probabilities ($< 10^{-7}$).
- `method = "integrate"`: always evaluates $F(x)$ via numerical integration of the PDF using the incomplete Bessel $K$ function.

This spline-based quantile function is essentially a simpler version of the PINV method discussed in §4 below.

---

## 4. Numerical Inverse CDF (PINV) Method

> Hörmann, W. and Leydold, J. (2011). "Generating generalized inverse Gaussian random variates by fast inversion." *Computational Statistics & Data Analysis*, 55(1):213–217.

This is **distinct** from their 2014 rejection paper. The idea is:

$$X = F^{-1}(U), \quad U \sim \mathrm{Uniform}(0,1)$$

Since the GIG has no closed-form quantile function $F^{-1}$, it must be approximated numerically. The **PINV** (Polynomial Interpolation based INVersion) method:

1. **Setup phase** (expensive, one-time):
   - Numerically integrate the PDF $f(x)$ using Gauss-Lobatto quadrature to build $F(x)$ on a grid.
   - Construct a **piecewise polynomial approximation** (Newton interpolation) of $F^{-1}(u)$ on $[0,1]$.
   - Adaptively refine intervals until the u-error $\varepsilon_u(u) = |u - F(\hat{F}^{-1}(u))|$ is below tolerance (e.g. $10^{-10}$).
   - Typical setup: 250–500 polynomial intervals, requiring thousands of PDF evaluations.

2. **Sampling phase** (very fast, per-sample):
   - Draw $U \sim \mathrm{Uniform}(0,1)$.
   - Locate the correct polynomial interval (binary search or direct indexing).
   - Evaluate the polynomial to get $X \approx F^{-1}(U)$.

SciPy provides this as `scipy.stats.sampling.NumericalInversePolynomial` (PINV) and the related `FastGeneratorInversion`.

### JAX-Friendliness Analysis

The PINV method has a fundamentally different computational profile from rejection sampling:

| Property | Rejection (Hörmann-Leydold / Devroye) | Inverse CDF (PINV) |
|---|---|---|
| **Setup cost** | Cheap (mode + a few constants) | Expensive (thousands of PDF evals, integration) |
| **Per-sample cost** | Variable (rejection loop) | Fixed (polynomial eval + lookup) |
| **Control flow** | `lax.while_loop` (dynamic) | Pure array ops (static) |
| **Vectorizability** | Requires `vmap` over `while_loop` | Fully vectorizable (`jnp.searchsorted` + polynomial eval) |
| **GPU parallelism** | Limited by `while_loop` serialization | Excellent — uniform memory access pattern |
| **JIT compilation** | Works but `while_loop` has overhead | Trivial — all static shapes |
| **Parameter changes** | Instant (just recompute setup constants) | Must rebuild entire interpolation table |

#### Why PINV Could Be Very Efficient in JAX

The key insight is that **after setup, PINV has no control flow at all**. The per-sample operation is:

```python
u = jax.random.uniform(key, shape=(n,))
idx = jnp.searchsorted(u_breakpoints, u)          # O(log K) per sample, vectorized
x = polynomial_eval(coefficients[idx], u)          # pure arithmetic, vectorized
```

This is a **purely data-parallel computation** — every sample is independent, there are no while loops, no branching, and the memory access pattern is predictable. On a GPU, all $n$ samples can be computed simultaneously in a single kernel launch.

By contrast, rejection sampling via `vmap(lax.while_loop)` compiles to $n$ independent while loops. Although JAX can execute them, **the worst-case sample determines the wall-clock time** for the entire batch (all threads must wait for the last rejection loop to finish). The expected number of iterations varies across samples, causing thread divergence on GPU.

#### Why PINV Has Drawbacks

1. **Setup cost**: Building the interpolation table requires integrating the GIG PDF, which involves the unnormalized density $(p-1)\log x - (b/x + ax)/2$. This must be done on a fine grid (hundreds of points) with adaptive refinement. In JAX, this setup is hard to JIT because:
   - The number of intervals is data-dependent.
   - Adaptive refinement is inherently sequential.
   - The setup would likely run on CPU (not JIT-compiled).

2. **Parameter changes**: In EM fitting, the GIG parameters change at every M-step. Rebuilding the PINV table at each iteration adds significant overhead. For the **fixed-parameter case** (sample $10^6$ variates from one GIG), PINV wins. For the **varying-parameter case** (sample 1 variate each from $10^3$ different GIGs, as in Gibbs sampling), rejection wins.

3. **Storage**: The polynomial coefficients for ~300 intervals × ~10 coefficients per polynomial = ~3000 floats. Modest, but more than the ~10 floats needed by rejection methods.

4. **Accuracy at tails**: Extreme quantiles ($u < 10^{-10}$ or $u > 1 - 10^{-10}$) may need special treatment or higher-degree polynomials. The R `GeneralizedHyperbolic` package falls back to direct integration for these.

#### Hybrid Approach: Precomputed Table + JAX Sampling

A pragmatic approach for normix:

1. **During `__init__` or first `rvs` call**: build the PINV table on CPU using scipy/numpy (not JIT-compiled). Store the breakpoints and polynomial coefficients as JAX arrays on the model (they are static for fixed parameters).
2. **During `rvs`**: use the precomputed table with `jnp.searchsorted` + polynomial evaluation — fully JIT-able and GPU-parallel.
3. **On parameter change** (e.g. after M-step): invalidate and rebuild the table.

This gives the best of both worlds: the setup is done once (or once per EM iteration), and sampling is maximally parallel. However, it introduces **statefulness** (the cached table) that conflicts with the immutable `eqx.Module` design.

---

## 5. Existing JAX Ecosystem (as of March 2026)

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

## 8. Comparison Summary

| Algorithm | Per-sample efficiency | Setup cost | JAX vectorization | Varying params | Production-tested |
|---|---|---|---|---|---|
| Hörmann-Leydold (2014) | ★★★★★ | ★★★★★ (trivial) | ★★★ (`while_loop`) | ★★★★★ | ★★★★★ (SciPy, R `GIGrvg`) |
| Devroye (2014) | ★★★★ | ★★★★★ (trivial) | ★★★ (`while_loop`) | ★★★★★ | ★★★ (MATLAB) |
| Dagpunar (1989) | ★★★ (fails near Gamma) | ★★★★★ (trivial) | ★★★ (`while_loop`) | ★★★★★ | ★★★★ (R `GeneralizedHyperbolic`) |
| PINV (Hörmann-Leydold 2011) | ★★★★★ | ★★ (expensive) | ★★★★★ (no loops!) | ★★ (rebuild table) | ★★★★ (SciPy) |

### Decision Factors

1. **Fixed-parameter, large $n$ case** (e.g. simulating $10^6$ GIG samples for a Monte Carlo study): **PINV wins.** The expensive setup is amortized, and sampling is maximally parallel on GPU — no thread divergence from `while_loop`.

2. **Varying-parameter case** (e.g. EM fitting where GIG parameters change every iteration, or Gibbs sampling with $n=1$ per parameter): **Rejection methods win.** Setup is instant, and the per-sample `while_loop` cost is modest.

3. **Implementation complexity**: Devroye ≈ 50 lines, Hörmann-Leydold ≈ 150 lines, PINV ≈ 200+ lines (plus CPU setup code).

4. **normix use case**: In EM fitting, `rvs` is called with changing parameters at each iteration — the varying-parameter case. But for final model sampling and Monte Carlo, parameters are fixed and $n$ is large — the fixed-parameter case. Both scenarios matter.

---

## 9. Recommendation

### Phase 1: Rejection-based (immediate)

Implement **Devroye (2014)** for its simplicity and guaranteed efficiency:

- Single algorithm for the entire parameter range.
- Guaranteed acceptance rate ≥ 25%.
- Avoid Bessel function evaluation in the rejection loop — use only the unnormalized log-density $(p-1)\log x - (b/x + ax)/2$.
- Compatible with `jax.lax.while_loop` + `jax.vmap`.
- ~50 lines of JAX code.

If profiling shows the ≥25% acceptance rate is a bottleneck, upgrade to Hörmann-Leydold (2014) with three-region branching via `lax.switch`.

### Phase 2: PINV for large-sample GPU path (future)

For GPU-accelerated large-sample generation, consider adding a PINV-based path:

- Build interpolation table on CPU during setup.
- Store coefficients as JAX arrays on the model.
- Sample via `jnp.searchsorted` + polynomial evaluation — fully JIT-able, no while loops, perfect GPU parallelism.
- This would be a separate method (e.g. `rvs_fast`) or activated by a flag, since the setup cost makes it unsuitable for the varying-parameter case.

### Always: Special-case routing

Regardless of algorithm choice, delegate to the already-implemented JAX samplers for degenerate GIG parameters:

- $b \approx 0$, $p > 0$ → `Gamma(p, a/2).rvs`
- $a \approx 0$, $p < 0$ → `InverseGamma(-p, b/2).rvs`
- $p = -1/2$ → `InverseGaussian.rvs`

---

## References

1. Hörmann, W. and Leydold, J. (2014). "Generating generalized inverse Gaussian random variates." *Statistics and Computing*, 24(4):547–557.
2. Hörmann, W. and Leydold, J. (2011). "Generating generalized inverse Gaussian random variates by fast inversion." *Computational Statistics & Data Analysis*, 55(1):213–217.
3. Devroye, L. (2014). "Random variate generation for the generalized inverse Gaussian distribution." *Statistics and Computing*, 24(2):239–246.
4. Dagpunar, J. S. (1989). "An easily implemented generalized inverse Gaussian generator." *Communications in Statistics — Simulation and Computation*, 18(2):703–710.
5. Michael, J. R., Schucany, W. R., and Haas, R. W. (1976). "Generating random variates using transformations with multiple roots." *The American Statistician*, 30(2):88–90.
6. Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag.
7. Lambardi di San Miniato, M. and Kenne Pagui, E. C. (2022). "Scalable random number generation for truncated log-concave distributions." arXiv:2204.01364.
8. Hörmann, W. and Leydold, J. (2003). "Continuous random variate generation by fast numerical inversion." *ACM Transactions on Modeling and Computer Simulation*, 13(4):347–362.
9. Scott, D., Trendall, R. and Luen, M. R package `GeneralizedHyperbolic`. CRAN.
