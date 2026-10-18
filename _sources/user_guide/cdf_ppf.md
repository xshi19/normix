# CDF and PPF accuracy

Gamma and inverse-gamma quantiles use the analytical distribution function.
Inverse-Gaussian and interior GIG quantiles use a trapezoidal PINV table
({ref}`HormannLeydold2011 <hormannleydold2011>`).
A degenerate GIG delegates to the Gamma or inverse-gamma CDF.

## Tolerances

These are the agreements the test suite pins against SciPy.

| Family | Method | Tolerance |
|---|---|---|
| Gamma, inverse-gamma | analytical | `rtol` $10^{-6}$, `atol` $10^{-8}$ |
| Inverse-Gaussian, interior GIG | 4000-point table | `rtol` $10^{-3}$, `atol` $10^{-4}$ |
| GIG round trip `cdf(ppf(q))` | same table | `rtol` $10^{-4}$, `atol` $10^{-5}$ |
| Degenerate GIG | Gamma or inverse-gamma limit | `rtol` $10^{-8}$ |

The grid has 4000 points.
Tail bisection stops at mass $10^{-14}$.
Left of the grid, `cdf` returns 0.
Right of the grid, `cdf` returns 1.
`ppf` interpolates on the stored quantile nodes.

The interpolant is differentiable in $x$.
A derivative in the parameters differentiates the grid too.
That derivative belongs to this approximation, not to a separate exact CDF.
Call `quantile_table()` once when many quantiles share the parameters.
`cdf` and `ppf` are JIT-safe.
The degeneracy test uses `lax.cond`.

The CPU and JAX paths both evaluate `log_kv` inside normix.
SciPy `kve` is a test oracle, not the CDF.
