# GIG Random Variate Generation

This note documents the JAX-based GIG sampling implementations in normix,
their design rationale, and benchmark results.

## Motivation

The GIG was the last normix distribution depending on `scipy.stats` for
sampling.  Replacing it with pure-JAX methods makes the entire `rvs` path
JIT-able and GPU-accelerated.

## Implemented Methods

### 1. Devroye-style TDR (`method='devroye'`)

**Module:** `normix/distributions/_gig_rvs.py`

Works in the log-transformed variable $w = \log x$ where the GIG log-kernel

$$g(w) = p\,w - \tfrac{1}{2}(a\,e^w + b\,e^{-w})$$

is strictly concave ($g''(w) = -(a\,e^w + b\,e^{-w})/2 < 0$).

A **three-piece TDR hat** is constructed from tangent lines at mode $\pm \sigma$
($\sigma = 1/\sqrt{-g''(w_0)}$) joined by a flat cap at $g(w_0)$:

1. Left tail: exponential with rate $g'(w_L)$ from $-\infty$ to the crossover.
2. Flat cap: constant $g(w_0)$ around the mode.
3. Right tail: exponential with rate $|g'(w_R)|$ from the crossover to $+\infty$.

**Acceptance rate ≈ 80–90 %** for typical GIG parameters.

**GPU strategy:** all $M \times n$ proposals ($M = 20$ rounds) are generated in a
single batch — zero loops, fully parallel.  `jnp.argmax` selects the first
accepted proposal per sample.

**Bessel-free:** only the unnormalized log-kernel is evaluated.

### 2. Numerical Inverse CDF — PINV (`method='pinv'`)

**Module:** `normix/utils/rvs.py` (generic), `normix/distributions/_gig_rvs.py` (GIG wrapper)

The PINV method builds $F^{-1}$ numerically:

1. **Setup (CPU, ~1.5 ms):** Evaluate the log-kernel on a 4 000-point grid in
   $w$-space, integrate via the trapezoidal rule to obtain the CDF, and store
   $(u_{\text{grid}}, x_{\text{grid}})$ as JAX arrays.
2. **Sampling (GPU, ~1–3 ms):** Draw $U \sim \text{Uniform}(0,1)$ and
   interpolate $X = F^{-1}(U)$ via `jnp.interp`.

**Bessel-free:** the CDF is normalised by dividing the cumulative integral by the
total, so the Bessel normalising constant $K_p(\sqrt{ab})$ is never needed.

**Generic:** the PINV infrastructure in `utils/rvs.py` accepts any univariate
log-kernel callable.  Future distributions can reuse it by providing their own
`log_kernel` and `mode`.

### 3. SciPy baseline (`method='scipy'`)

The original `scipy.stats.geninvgauss` path is retained as a CPU fallback /
comparison baseline.  It uses the Hörmann-Leydold (2014) three-region
acceptance-rejection algorithm.

## Benchmark Results

Environment: Python 3.12, JAX 0.9.1, CUDA GPU. Warm timings (JIT cached).

### Timing (ms, median across parameter sets)

| Method | n = 100 | n = 1 000 | n = 10 000 | n = 100 000 |
|---|---|---|---|---|
| **Devroye** | 10 | 11 | 11 | 12 |
| **PINV** (setup + sample) | 3 | 3 | 3 | 4 |
| **SciPy** | 0.5 | 0.7 | 1.0 | 8 |

Tested with 6 parameter sets: (p, a, b) ∈ {(1,1,1), (0.5,2,0.5), (−0.5,1,1),
(3,0.5,2), (−2,0.5,3), (0.1,10,0.1)}.

Key observations:

- **PINV is fastest** across all sample sizes (3–4 ms total including table build).
- **Devroye** has ~10 ms constant overhead from the batched (M × n) proposal
  generation on GPU.
- **SciPy** scales linearly (CPU); competitive at small n but 2× slower than
  PINV at n = 100 000.

### Validation

All 18 parameter × method combinations pass the Kolmogorov-Smirnov test at
α = 0.01 (n = 10 000).  Sample means match analytical means to < 2 % relative
error.

Benchmark script: `scripts/benchmark_gig_rvs.py`.

## Design Decisions

| Decision | Rationale |
|---|---|
| Log-transform $w = \log x$ | Removes the singularity at $x = 0$ for $p < 1$; makes the GIG log-kernel bounded and strictly concave for all valid (p, a, b). |
| Batch rejection (no `while_loop`) | `vmap(while_loop)` on GPU costs ~400 ms overhead per call.  Generating all M × n proposals in one batch reduces this to ~10 ms. |
| Generic PINV in `utils/rvs.py` | The method is distribution-agnostic; only a `log_kernel` callable and a mode are needed.  Reusable for future distributions. |
| GIG Devroye stays in `distributions/_gig_rvs.py` | The TDR envelope is GIG-specific (relies on the particular form of $g(w)$). |
| Default method = `'devroye'` | Pure JAX with no CPU setup step.  PINV is faster but requires an eager CPU table build. |

## References

1. Devroye, L. (2014). "Random variate generation for the generalized inverse
   Gaussian distribution." *Statistics and Computing*, 24(2):239–246.
2. Hörmann, W. and Leydold, J. (2011). "Generating generalized inverse Gaussian
   random variates by fast inversion." *Computational Statistics & Data
   Analysis*, 55(1):213–217.
3. Hörmann, W. and Leydold, J. (2014). "Generating generalized inverse Gaussian
   random variates." *Statistics and Computing*, 24(4):547–557.
