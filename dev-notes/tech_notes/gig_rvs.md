# GIG Random Variate Generation

This note documents the JAX-based GIG sampling implementations in normix,
their design rationale, and benchmark results.

## Motivation

The GIG was the last normix distribution depending on `scipy.stats` for
sampling.  Replacing it with pure-JAX methods makes the entire `rvs` path
JIT-able and GPU-accelerated.

## Implemented Methods

### 1. Devroye-style TDR (`method='devroye'`)

**Location:** private helpers in `normix/distributions/generalized_inverse_gaussian.py`
(`_gig_rvs_devroye`, `_gig_tdr_setup`).

Works in $w=\log x$ with the centred kernel $\psi(u)=g(w_0+u)-g(w_0)$ in
$(p,z,s)$ coordinates ($z=\sqrt{ab}$, $s=\tfrac12\log(b/a)$,
$w_0=s+\operatorname{asinh}(p/z)$ for $a,b>0$). At an exact $a=0$ or $b=0$
boundary `_gig_log_mode` uses the rationalized $x$-space pair instead of
asinh, and default `rvs` samples Gamma / InverseGamma (NaN off $\Theta$).
Coefficients of $\operatorname{expm1}(\pm u)$
are $r\pm|p|$ with $r-|p|=z^2/(r+|p|)$. Tangent points solve $\psi(\pm t)=-1$
(fixed bisection, `BESSEL_WINDOW_ITERS`); the hat is flat on the $e^{-1}$
secants and exponential in the tails. Concavity gives acceptance
$\ge e^{-1}$ uniformly in $(p,a,b)$ with $a,b>0$.

`lax.while_loop` redraws only unaccepted columns — never emit a rejected
proposal via `argmax` on an all-False mask (exhausted columns are NaN).
`mode()` and the PINV seed share `_gig_log_mode`.

**Bessel-free:** only the unnormalized log-kernel is evaluated.

### 2. Numerical Inverse CDF — PINV (`method='pinv'`)

**Location:** `normix/utils/rvs.py` exposes the pure-JAX
`build_pinv_table` + `rvs_pinv`; `GIG.rvs(method='pinv')` (and the
shared `GIG.cdf` / `GIG.ppf`) go through `quantile_table()` with
`log_kernel(w) = self.log_prob(exp(w)) + w` seeded at
`_gig_log_mode(p-1, a, b)`. The remaining GIG-specific sampler helper is
`_gig_rvs_devroye` (TDR).

The PINV method builds $F^{-1}$ numerically:

1. **Setup (~1.5 ms):** Tail boundaries via JAX bisection (`lax.fori_loop`),
   then evaluate the log-kernel on a 4 000-point grid in $w$-space and
   integrate via the trapezoidal rule (`jnp.cumsum`). Returns
   $(u_{\text{grid}}, x_{\text{grid}})$ as JAX arrays.
2. **Sampling (GPU, ~1–3 ms):** Draw $U \sim \text{Uniform}(0,1)$ and
   interpolate $X = F^{-1}(U)$ via `jnp.interp`.

**Bessel-free:** the CDF is normalised by dividing the cumulative integral by the
total, so the Bessel normalising constant $K_p(\sqrt{ab})$ is never needed.

**Generic:** the PINV infrastructure in `utils/rvs.py` accepts any univariate
log-kernel callable. `InverseGaussian.ppf` and the univariate Bessel-mixture
marginals (`UnivariateVarianceGamma`, `UnivariateNormalInverseGamma`,
`UnivariateNormalInverseGaussian`, `UnivariateGeneralizedHyperbolic`)
reuse the same `build_pinv_table` call.

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

Numbers above are from a one-off Devroye vs PINV vs SciPy sweep. Devroye
trend tracking is `asv_bench/benchmarks/gig.py` (`Sampling`). Re-run a
method comparison from the deep-dive layer if these numbers go stale.

## Design Decisions

| Decision | Rationale |
|---|---|
| Log-transform $w = \log x$ | Strictly log-concave target for every $(p,a,b)$ with $a,b>0$. |
| $(p,z,s)$ envelope, $e^{-1}$ tangents | Curvature $\sigma=z^{-1/2}$ is the wrong width when $z\ll 1$; the $e^{-1}$ construction has acceptance $\ge e^{-1}$ uniformly. |
| `lax.while_loop` redraw | Exhausted rejection rounds are a failure, not a sample. `argmax` of an all-False mask emitted envelope draws. |
| Generic PINV in `utils/rvs.py` | The method is distribution-agnostic; only a `log_kernel` callable and a mode are needed.  Reusable for future distributions. |
| GIG Devroye lives inside `distributions/generalized_inverse_gaussian.py` | The TDR envelope is GIG-specific (relies on the particular form of $g(w)$). Co-locating with the `GIG` class keeps the implementation surface contiguous. |
| Default method = `'devroye'` | Pure JAX with no setup step.  PINV is faster end-to-end but requires an eager table build before the first sample. |

## References

1. Devroye, L. (2014). "Random variate generation for the generalized inverse
   Gaussian distribution." *Statistics and Computing*, 24(2):239–246.
2. Hörmann, W. and Leydold, J. (2011). "Generating generalized inverse Gaussian
   random variates by fast inversion." *Computational Statistics & Data
   Analysis*, 55(1):213–217.
3. Hörmann, W. and Leydold, J. (2014). "Generating generalized inverse Gaussian
   random variates." *Statistics and Computing*, 24(4):547–557.
