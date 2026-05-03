# Test Suite Performance Investigation

Date: 2026-04-28

## Scope

This note summarizes a full static and runtime review of the current `normix`
test suite. It covers:

- slowest individual tests and slowest test files
- potentially redundant or duplicated coverage
- GPU memory behavior during tests
- the `IncrementalEMFitter` / Robbins-Monro GH slowdown
- JAX compilation, dispatch, and Python-loop overhead
- recommended improvements and an execution plan

No library behavior was changed as part of this investigation.

## Measurement Setup

The full test suite was run with per-test timing enabled:

```bash
uv run pytest tests/ --durations=0 --tb=no -q
```

Observed result:

- `792 passed`
- `1 warning`
- total wall time: `948.18 s` (`0:15:48`)
- JAX backend: GPU
- visible GPU: NVIDIA GeForce RTX 4090

Resource monitoring was sampled every 3 seconds during most of the run. The
per-process CPU metric was not reliable because the monitor initially attached
to a wrapper process, so only system-level CPU should be treated as usable.

Observed resources:

| Metric | Value |
|---|---:|
| Samples collected | 308 |
| Monitor elapsed | 935.5 s |
| Peak system CPU | 24.1% |
| Peak RAM RSS | 8001 MB |
| Average RAM RSS | 5899 MB |
| Peak GPU utilization | 71% |
| Average GPU utilization | 9.1% |
| Peak GPU memory | 23806 MB |
| Average GPU memory | 20910 MB |

## Slowest Test Files

The following totals are sums of individual pytest call durations from the
`--durations=0` report.

| Rank | File | Tests | Total |
|---:|---|---:|---:|
| 1 | `tests/test_incremental_em.py` | 38 | 177.7 s |
| 2 | `tests/test_solvers.py` | 48 | 149.0 s |
| 3 | `tests/test_distributions_vs_scipy.py` | 66 | 99.8 s |
| 4 | `tests/test_gig_properties.py` | 126 | 72.6 s |
| 5 | `tests/test_jax_distributions.py` | 59 | 60.2 s |
| 6 | `tests/test_jax_bessel.py` | 62 | 53.0 s |
| 7 | `tests/test_generalized_hyperbolic.py` | 20 | 52.3 s |
| 8 | `tests/test_cpu_bessel_backend.py` | 42 | 50.9 s |
| 9 | `tests/test_em_cpu_vs_jax.py` | 20 | 38.4 s |
| 10 | `tests/test_gig_special_cases.py` | 15 | 23.8 s |
| 11 | `tests/test_em_regression.py` | 8 | 20.7 s |
| 12 | `tests/test_marginal_api.py` | 39 | 17.3 s |
| 13 | `tests/test_extreme_parameters.py` | 44 | 17.2 s |
| 14 | `tests/test_mcecm.py` | 19 | 15.5 s |
| 15 | `tests/test_normal_inverse_gamma.py` | 17 | 14.5 s |
| 16 | `tests/test_variance_gamma.py` | 20 | 13.9 s |
| 17 | `tests/test_multivariate_rvs.py` | 15 | 13.6 s |
| 18 | `tests/test_divergences.py` | 27 | 13.4 s |
| 19 | `tests/test_exponential_family.py` | 24 | 10.4 s |
| 20 | `tests/test_sp500_distribution_validation.py` | 4 | 8.3 s |
| 21 | `tests/test_normal_inverse_gaussian.py` | 18 | 8.0 s |
| 22 | `tests/test_high_dimensional.py` | 9 | 5.7 s |
| 23 | `tests/test_package_imports.py` | 2 | 5.3 s |

## Slowest Individual Tests

| Rank | Test | Time | Main cause |
|---:|---|---:|---|
| 1 | `test_incremental_em_robbins_monro[GH]` | 60.06 s | 20 fixed GH incremental steps, each with JAX/JAX GIG M-step solve |
| 2 | `TestSolveBregmanGIG::test_solve_bregman_cpu_hybrid` | 38.08 s | CPU hybrid Bregman solve |
| 3 | `TestGeneralizedHyperbolicFitting::test_em_monotone_ll` | 26.31 s | manual GH EM loop using default JAX/JAX paths |
| 4 | `test_incremental_em_ll_improves` | 23.35 s | 30 incremental EM steps |
| 5 | `TestGeneralizedHyperbolic::test_em_convergence` | 19.75 s | manual GH EM loop using default JAX/JAX paths |
| 6 | `TestGIGVsScipy::test_pdf_comparison[-0.5-2.0-1.0]` | 18.74 s | 50 sequential GIG PDF evaluations |
| 7 | `TestSolveBregmanGIG::test_from_expectation_warm_start[jax-lbfgs]` | 18.71 s | JAX LBFGS GIG solve |
| 8 | `TestGIGVsScipy::test_pdf_comparison[2.0-2.5-1.5]` | 18.68 s | 50 sequential GIG PDF evaluations |
| 9 | `TestGIGVsScipy::test_pdf_comparison[0.5-1.0-2.0]` | 18.65 s | 50 sequential GIG PDF evaluations |
| 10 | `TestGIGVsScipy::test_pdf_comparison[1.0-1.0-1.0]` | 18.37 s | 50 sequential GIG PDF evaluations |
| 11 | `TestSolveBregmanGIG::test_from_expectation_warm_start[jax-bfgs]` | 17.00 s | JAX BFGS GIG solve |
| 12 | `TestExponentialFamilyFromExpectation::test_gig_backend_and_method_args` | 16.89 s | GIG Bregman dispatch |
| 13 | `TestSolveBregmanGIG::test_result_converged` | 16.56 s | JAX LBFGS with high iteration budget |
| 14 | `test_incremental_em_robbins_monro[VG]` | 15.25 s | fixed JAX/JAX incremental EM loop |
| 15 | `test_incremental_em_fine_tuning` | 12.83 s | mini-batch loop with inner iterations |
| 16 | `test_batch_em_fitter_cpu_same_result_as_default` | 10.63 s | JAX/JAX vs CPU/CPU comparison |
| 17 | `test_incremental_em_robbins_monro[NInvG]` | 9.86 s | fixed JAX/JAX incremental EM loop |
| 18 | `test_log_kv_hessian_wrt_z` | 9.61 s | second-order autodiff through Bessel function |
| 19 | `TestGeneralizedHyperbolic::test_m_step_increases_ll` | 8.42 s | one GH E+M step on default JAX/JAX path |
| 20 | `test_incremental_em_various_rules[identity-rule0]` | 8.18 s | fixed JAX/JAX incremental EM loop |

## Main Runtime Findings

### 1. GPU memory is reserved, not necessarily actively used

The run showed GPU memory near full (`~21 GB` average on a `24 GB` RTX 4090)
while average GPU utilization was only `9.1%`.

This is expected JAX behavior. By default, JAX/XLA preallocates a large GPU
memory pool and keeps it for reuse. Tools such as `nvidia-smi` report the
reserved pool as used memory even when the current active arrays are small.

This does not by itself mean the algorithms require 20+ GB of live working
memory. It mostly means JAX has reserved a device allocator arena.

Useful environment variables for tests:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests/
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 uv run pytest tests/
JAX_PLATFORM_NAME=cpu uv run pytest tests/
```

The low average utilization is also expected for this suite. Many tests run
tiny scalar or low-dimensional computations: small GIG solves, `d=1..4`
matrix work, and batches of only `20..200`. These workloads do not saturate a
large GPU. Python dispatch, XLA compilation, CPU SciPy work, and host-device
synchronization can dominate the actual arithmetic.

### 2. Robbins-Monro GH is warm-started but still slow

`test_incremental_em_robbins_monro[GH]` was the slowest test at `60.06 s`.

The GH GIG M-step is warm-started. `GeneralizedHyperbolic.m_step_subordinator`
passes the current GIG natural parameters to `GIG.from_expectation`:

```python
gig_new = GIG.from_expectation(
    gig_eta,
    theta0=current_gig.natural_params(),
    backend=backend,
    method=method,
    maxiter=maxiter,
)
```

So the slowdown is not because the GIG optimizer lacks a numerical warm start.
The slowdown comes from the surrounding execution model:

- `IncrementalEMFitter` uses a Python `for` loop over `max_steps`.
- The Robbins-Monro test uses `max_steps=20`.
- The test uses `e_step_backend='jax', m_step_backend='jax'`.
- GH `m_step` performs a GIG `eta -> theta` solve every step.
- The problem is only 3-dimensional, so GPU kernel launch and JAX dispatch
  overhead are large relative to the math.

Regular batch EM tests are often faster because many of them use CPU/CPU
backends and can also stop early by convergence tolerance. Incremental EM
currently always runs exactly `max_steps`.

### 3. JAX warm-start is not the same as compilation caching

Numerical warm-starting means the optimizer starts near the previous solution.
That reduces mathematical iteration work.

Compilation caching is different. JAX reuses compiled XLA executables when the
same jitted function, static arguments, shapes, and dtypes are seen again.

The current Bregman solver path creates fresh Python closures inside each
solve. For JAX quasi-Newton:

```python
def _jax_quasi_newton(f, eta, theta0, bounds, max_steps, tol, method):
    phi0, to_theta, _ = _setup_reparam(theta0, bounds)

    def obj_phi(phi):
        theta = to_theta(phi)
        return f(theta) - jnp.dot(theta, eta)

    solver = jaxopt.LBFGS(fun=obj_phi, maxiter=max_steps, tol=tol, jit=True)
    result = solver.run(phi0)
```

The numerical values are warm-started, but a fresh `obj_phi` closure and a
fresh solver object are created inside the Python call. JAX/JAX workloads in
small loops can therefore spend a lot of time tracing, dispatching, launching
tiny kernels, and synchronizing.

For the custom Newton path, the inner solve uses `jax.lax.scan`, but the
incremental EM loop around it is still a Python loop. A more JAX-native design
would make the repeated EM or GIG solve one stable `@jax.jit` function that
receives changing `eta` and `theta0` as array arguments, not as freshly closed
over Python values.

### 4. GIG SciPy PDF tests are slow because they are scalar loops

The four slow `TestGIGVsScipy::test_pdf_comparison[...]` tests each loop over
50 x-points and call `gig.pdf(x)` sequentially. The four tests together cost
about `74 s`.

This is not a large-data stress test. It is a scalar Python loop repeatedly
entering JAX/Bessel code. It should be batched with `jax.vmap` or reduced to a
smaller representative grid.

### 5. Bessel autodiff tests are intentionally expensive

`test_log_kv_hessian_wrt_z` costs `9.61 s`, and the first-order Bessel gradient
parameterizations are around `1.9..2.0 s` each. These tests exercise difficult
custom-JVP and higher-order autodiff behavior. They are valuable, but they are
not cheap unit tests.

## Potentially Redundant or Duplicated Coverage

### EM fitting is tested in many places

Full or near-full EM is exercised in:

- `tests/test_em_regression.py`
- `tests/test_variance_gamma.py`
- `tests/test_normal_inverse_gamma.py`
- `tests/test_normal_inverse_gaussian.py`
- `tests/test_generalized_hyperbolic.py`
- `tests/test_sp500_distribution_validation.py`
- `tests/test_em_cpu_vs_jax.py`
- `tests/test_mcecm.py`
- `tests/test_cpu_bessel_backend.py`
- manual loops in `tests/test_jax_distributions.py`

The per-distribution `.fit(...)` tests overlap strongly with
`test_em_regression.py`, which already covers VG, NInvG, NIG, and GH in 1D
and 2D.

Recommendation: keep one canonical synthetic EM regression file, then reduce
per-distribution files to construction, formulas, edge cases, and light smoke
tests.

### Sample mean/covariance checks are duplicated

Large Monte Carlo moment checks exist both in per-distribution files and in
`tests/test_multivariate_rvs.py`.

Examples:

- `test_variance_gamma.py`
- `test_normal_inverse_gamma.py`
- `test_normal_inverse_gaussian.py`
- `test_generalized_hyperbolic.py`
- `test_multivariate_rvs.py`

`test_multivariate_rvs.py` is the better canonical home for joint sample mean
and covariance validation because it tests all four mixture families together
against a shared theoretical covariance formula.

### Conditional expectations are repeated

Finite/positive conditional expectation checks appear in:

- per-distribution files for VG, NInvG, NIG, GH
- `tests/test_jax_distributions.py`
- CPU/JAX backend comparison tests

Recommendation: keep cross-family conditional expectation checks in one or two
canonical files and remove repetitive per-distribution loops unless they cover
a distribution-specific edge case.

### Univariate EF invariants are triple-covered

Gamma, InverseGamma, InverseGaussian, and GIG invariants appear across:

- `tests/test_jax_distributions.py`
- `tests/test_distributions_vs_scipy.py`
- `tests/test_exponential_family.py`
- `tests/test_gig_properties.py`

Recommendation: make `test_distributions_vs_scipy.py` the canonical SciPy
reference file, `test_exponential_family.py` the base-class contract file,
and `test_gig_properties.py` the GIG-specific stress file. Avoid repeating the
same invariant grid in all three.

### SP500 / real-data EM is split across three files

Real-data EM-style tests appear in:

- `tests/test_sp500_distribution_validation.py`
- `tests/test_em_cpu_vs_jax.py`
- `tests/test_mcecm.py`

Recommendation: keep one short default real-data smoke test. Mark larger
real-data validation as slow/integration.

## Recommendations

### Test-suite changes

1. Add pytest markers:
   - `smoke`
   - `contract`
   - `slow`
   - `integration`
   - `gpu`
   - `stress`

2. Define a default fast suite:
   - unit tests
   - small mathematical invariant checks
   - short smoke tests
   - no large Monte Carlo
   - no full SP500 validation
   - no repeated JAX/JAX GH incremental stress test

3. Move stress tests behind explicit marks:
   - large Monte Carlo sample mean/covariance checks
   - SP500 full validation
   - high-iteration GIG solver tests
   - Bessel Hessian / extensive autodiff grids

4. Reduce obvious scalar-loop overhead:
   - replace GIG PDF loops with `jax.vmap`
   - reduce `jnp.linspace(..., 50)` grids to around 10 representative points
   - keep one high-resolution grid only under `slow`

5. Consolidate EM tests:
   - keep `test_em_regression.py` as canonical synthetic EM coverage
   - remove or shrink redundant per-distribution `.fit(...)` tests
   - keep `test_em_cpu_vs_jax.py` focused on backend agreement, not broad
     convergence coverage

6. Reduce default incremental EM smoke tests:
   - for GH, use CPU backend or fewer `max_steps`
   - keep JAX/JAX GH incremental EM as a marked `slow` or `gpu` test
   - consider a targeted test that verifies Robbins-Monro weight arithmetic
     separately from full GH fitting

7. Make large Monte Carlo tests deterministic but smaller:
   - reduce `50000` samples to `10000..15000`
   - loosen covariance tolerances where appropriate
   - keep a larger sample stress version under `slow`

### Implementation-performance changes

1. Build a stable jitted GIG Newton solve:
   - define one top-level function receiving `eta_scaled` and `theta0_scaled`
   - avoid fresh Python closures per solve
   - pass changing values as arrays

2. Consider a JAX-native `IncrementalEMFitter` scan path:
   - pre-generate mini-batch indices or keys
   - use `jax.lax.scan` over steps
   - make the full step function trace once
   - support only JAX-compatible eta update rules in this path

3. Prefer CPU backend for tiny scalar solves in default tests:
   - especially GH/GIG M-steps with `d <= 3` and small `n`
   - leave JAX/GPU performance testing to explicit GPU benchmarks

4. Separate benchmarking from correctness tests:
   - correctness tests should be small and focused
   - stress tests should live under `benchmarks/` or `pytest.mark.slow`
   - performance regressions should use a dedicated benchmark command rather
     than the default `uv run pytest tests/`

5. Investigate compilation cache effectiveness:
   - run with `JAX_LOG_COMPILES=1`
   - compare current `GIG.from_expectation(... backend='jax')` with a stable
     top-level jitted solver
   - measure first-call vs second-call time separately

## Step-by-Step Plan

### Phase 1 — Make Cost Visible

Status: implemented. Marker definitions and the default fast marker expression
live in `pyproject.toml`; the contributor note lives in `README.md`.

1. Add pytest marker definitions in `pyproject.toml`.
2. Add a short `docs/investigations/` note or README section explaining:
   - default test command
   - slow test command
   - GPU test command
3. Add a profiling recipe:
   - `uv run pytest tests/ --durations=50`
   - optional `JAX_LOG_COMPILES=1`
   - optional `XLA_PYTHON_CLIENT_PREALLOCATE=false`

Exit criteria:

- the default command and slow command are clearly separated
- contributors can reproduce a per-test duration report

### Phase 2 — Quick Test-Suite Wins

Status: partially implemented. The slowest stress-style tests are marked, and
the SciPy PDF comparison grids are batched with `jax.vmap` without reducing
grid size. Duplicate EM and Monte Carlo coverage can still be consolidated in
later cleanup.

1. Mark the slowest stress-style tests:
   - GH JAX/JAX incremental EM
   - high-iteration GIG solver tests
   - Bessel Hessian test
   - SP500 validation
   - large Monte Carlo covariance tests
2. Reduce GIG SciPy PDF grids or vectorize them.
3. Reduce duplicated per-distribution sample mean/covariance tests.
4. Keep `test_multivariate_rvs.py` as canonical moment coverage.

Exit criteria:

- default suite time drops substantially without losing core coverage
- slow/stress tests remain available explicitly

### Phase 3 — Consolidate EM Coverage

Status: implemented. `test_em_regression.py` is the canonical synthetic batch
EM suite. Redundant full-fit tests were removed from the per-distribution
files, the extra GH manual EM loop was removed from `test_jax_distributions.py`,
and backend/MCECM real-data tests now focus on targeted step-level behavior
rather than repeated full EM convergence comparisons.

1. Keep `test_em_regression.py` as canonical synthetic batch EM coverage.
2. Trim per-distribution EM tests to one light smoke test or remove where
   fully redundant.
3. Keep `test_em_cpu_vs_jax.py` focused on backend agreement.
4. Keep `test_mcecm.py` focused on MCECM-specific behavior.
5. Move expensive real-data convergence validation behind `integration` or
   `slow`.

Exit criteria:

- no distribution loses EM coverage
- duplicate full-fit tests are removed or marked slow
- real-data tests do not dominate the default suite

### Phase 4 — Improve JAX Solver Structure

Status: implemented (2026-05-03). A `make_jit_newton_solver` helper in
`normix/fitting/solvers.py` builds a `@jax.jit`-decorated Newton solve that
bakes the distribution-level `(f, grad_fn, hess_fn, bounds)` into one stable
closure. `GeneralizedInverseGaussian.from_expectation` routes the warm-started
`backend='jax', method='newton'` path through a module-level instance
(`_gig_jax_newton_jit`), so all GH M-step solves share one XLA executable.
The same JIT-cache fix was applied to `_newton_digamma` (used by Gamma /
InverseGamma / VG / NInvG M-steps) by adding `@jax.jit`.

A new benchmark `benchmarks/bench_jit_solvers.py` measures first-call vs
cached-call latency and is wired into `run_all.py`.

1. Prototype a stable jitted GIG Newton function outside the classmethod path.
2. Compare:
   - current JAX/JAX `GIG.from_expectation`
   - stable jitted Newton solve
   - CPU SciPy solve
3. Measure first-call and second-call latency separately.
4. If successful, route the GH JAX M-step through the stable solver.

Exit criteria:

- repeated GH/GIG JAX M-steps avoid repeated tracing/dispatch where possible
- Robbins-Monro GH JAX/JAX test improves without changing math

Result (`bench_jit_solvers`, RTX 4090, JAX 0.9.1):

| Case | First call | Cached call | Ratio |
|---|---:|---:|---:|
| GIG warm-start, symmetric (p=0.5, a=b=1) | `3.05 s` | `24 ms` | `126×` |
| GIG warm-start, asymmetric (a≫b, p=0.5)  | `41 ms`  | `39 ms` | `1.05×` |
| GIG warm-start, InvGauss limit (p=-½)    | `41 ms`  | `44 ms` | `0.9×`  |
| Gamma `_newton_digamma`, target=0.0      | `253 ms` | `4.5 ms`| `56×`   |

The asymmetric / InvGauss cases land at ~40 ms cached because their first
call hits the cache primed by the symmetric case (same shapes/dtypes). End
to end, `bench_em_mixture --large` reports the GH `cpu/jax/newton` row
dropping from `3.29 s/iter` to `1.73 s/iter` (-47.5 %) on a 2-iteration run
that still pays the one-time compile in iteration 1; longer EM loops should
amortise the compile and approach the cached `~24 ms/call` floor.

Other distributions: cached time matches CPU/LBFGS within `~1.4×` for GIG
and `~10–20×` slower than `_newton_digamma_cpu` (still small, sub-millisecond
for both — Gamma rarely dominates EM time anyway).

### Phase 5 — JAX-Native Incremental EM Path

1. Add an optional scan-based path for `IncrementalEMFitter` when:
   - e-step backend is JAX
   - m-step backend is JAX
   - eta update rule is JAX-compatible
   - no verbose Python reporting is required
2. Generate all batch keys or indices outside the scan.
3. Keep the current Python-loop path for CPU backends and debug verbosity.

Exit criteria:

- JAX/JAX incremental EM compiles the repeated step once
- Python-loop overhead no longer dominates tiny mini-batch tests

### Phase 6 — Benchmark Policy

1. Move heavy stress validation into `benchmarks/` or marked tests.
2. Add a benchmark comparing:
   - CPU/CPU GH EM
   - JAX/JAX GH EM
   - incremental GH with CPU M-step
   - incremental GH with JAX M-step
3. Record expected performance regimes:
   - small `n`, small `d`: CPU often wins
   - large batch E-step: GPU may win if the path is fully JAX-native
   - tiny GIG solves: CPU or cached JAX solve likely wins

Exit criteria:

- correctness tests stay fast
- stress/performance questions are answered by explicit benchmark commands

## Practical Short-Term Recommendation

For the current codebase, the fastest low-risk improvement is not a major
algorithm rewrite. It is:

1. mark or shrink redundant slow tests
2. vectorize scalar GIG PDF loops
3. use CPU backend for tiny GH/GIG smoke tests
4. reserve JAX/JAX GH incremental EM for an explicit slow/GPU test

The deeper engineering improvement is a stable, jitted GIG solver and a
scan-based JAX incremental EM path.
