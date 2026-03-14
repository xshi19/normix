# Investigating why GH EM can be slower on JAX GPU than CPU

## Scope

This note investigates the current `normix` implementation of the
Generalized Hyperbolic (GH) EM algorithm, with a focus on whether the M-step
and the GIG `eta -> theta` L-BFGS solve are the main bottleneck when a caller
uses JAX/GPU arrays.

No library behavior was changed. The investigation used:

- code inspection of the GH EM path
- a standalone timing script: `scripts/profile_gh_em_breakdown.py`
- local CPU benchmarks on this cloud machine
- public JAX issue reports and discussions about GPU slowdowns

## Important limitation

This machine did **not** have JAX or a visible GPU installed initially, so I
could not run a true local CPU-vs-GPU benchmark for the exact same script.
The conclusions below combine:

1. direct timing of the current checked-in code
2. code-path inspection showing where the implementation leaves the JAX world
3. public reports about when JAX GPU is slower than CPU

## Main conclusion

The current GH EM path in this repository is **not JAX-native**.

The hottest parts of the implementation use **NumPy + SciPy on the host**:

- `normix/distributions/mixtures/generalized_hyperbolic.py`
  - `fit(...)` starts with `X = np.asarray(X)`
  - `_conditional_expectation_y_given_x(...)` starts with `X = np.asarray(X)`
- `normix/base/exponential_family.py`
  - `_expectation_to_natural(...)` calls `scipy.optimize.minimize(..., method="L-BFGS-B")`
- `normix/utils/bessel.py`
  - `log_kv(...)` uses `scipy.special.kve`

That means:

1. passing a JAX array into GH `fit(...)` does **not** keep the workload on GPU
2. the per-iteration tail solve is done by SciPy L-BFGS-B on the host
3. the E-step Bessel work is also host-side SciPy/NumPy work

So if a caller starts from JAX GPU arrays, the current code is very likely to:

- coerce data back to host NumPy
- run the heavy work on CPU anyway
- potentially pay extra host/device transfer overhead along the way

This already explains a large part of "GPU slower than CPU" for the current
implementation: the code is not structured as a fused JAX workload that can
benefit from GPU execution.

## What one EM iteration actually does

Inside `GeneralizedHyperbolic.fit(...)`, each iteration is:

1. regularize parameters
2. E-step via `_conditional_expectation_y_given_x(X)`
3. M-step via `_m_step(X, cond_exp, ...)`
4. regularize again
5. convergence check

The M-step is split into:

- a **closed-form** update for `(mu, gamma, Sigma)`
- a **numerical** update for the GIG tail parameters `(p, a, b)`

The GIG update is:

- `sub.set_expectation_params(sub_eta, theta0=sub.natural_params)`
- which reaches `ExponentialFamily._expectation_to_natural(...)`
- which uses **SciPy L-BFGS-B**

So the L-BFGS hypothesis is directionally correct, but only for the tail part
of the M-step, not the whole GH iteration.

## Local benchmark script

I added:

- `scripts/profile_gh_em_breakdown.py`

It times, per EM iteration:

- E-step
- M-step total
- time spent specifically in the GIG `set_expectation_params(...)` call
- M-step minus GIG time
- extra `logpdf` cost that appears when `verbose >= 1`

It also prints `cProfile` summaries for:

- one E-step
- one isolated GIG `eta -> theta` solve

## Benchmark results

### Case 1: moderate problem, `n=3000`, `d=6`

Average per iteration:

- total: `0.0091 s`
- E-step: `0.0057 s` (`62%`)
- M-step total: `0.0023 s` (`25%`)
- GIG optimizer only: `0.0019 s` (`21%`)
- M-step excluding GIG: `0.0004 s` (`5%`)
- extra `logpdf` pass: `0.0010 s` (`11%`)

Interpretation:

- the GIG L-BFGS solve is **most of the M-step**
- but the **E-step is still the largest part** of the whole iteration

### Case 2: larger mixed regime, `n=20000`, `d=12`

Average per iteration:

- total: `0.1266 s`
- E-step: `0.0653 s` (`52%`)
- M-step total: `0.0401 s` (`32%`)
- GIG optimizer only: `0.0351 s` (`28%`)
- M-step excluding GIG: `0.0050 s` (`4%`)
- extra `logpdf` pass: `0.0210 s` (`17%`)

Interpretation:

- here the GIG solve is a **large** part of the iteration
- but the E-step is still slightly larger overall

### Case 3: high-sample regime, `n=100000`, `d=6`

Average per iteration:

- total: `0.2529 s`
- E-step: `0.1972 s` (`78%`)
- M-step total: `0.0132 s` (`5%`)
- GIG optimizer only: `0.0025 s` (`1%`)
- M-step excluding GIG: `0.0107 s` (`4%`)
- extra `logpdf` pass: `0.0424 s` (`17%`)

Interpretation:

- once `n` gets large, the **samplewise E-step dominates**
- the GIG solve becomes almost a constant-cost side term

## What `cProfile` showed

### E-step

The E-step profile is dominated by Bessel-related functions:

- `normix/utils/bessel.py:log_kv`
- `normix/utils/bessel.py:kv_ratio`
- `normix/utils/bessel.py:log_kv_derivative_v`

These rely on SciPy special functions and, for `log_kv_derivative_v`, can use
finite differences that trigger multiple Bessel evaluations.

### GIG solve

The GIG profile is dominated by:

- `scipy.optimize._minimize.minimize`
- `scipy.optimize._lbfgsb_py._minimize_lbfgsb`
- repeated calls into GIG expectation calculations
- which again call the Bessel utilities

So the GIG solve is a real hotspot, but it is a **host-side SciPy hotspot**.

## Extra cost that is easy to miss

### `verbose >= 1` adds another full GH logpdf pass

The convergence logic computes the marginal log-likelihood for reporting when
verbosity is enabled. In the local runs, that extra pass added about:

- `11%` per iteration in the small case
- `17%` per iteration in the larger cases

That is not the main bottleneck, but it is large enough to matter.

### Initialization is not free

`GeneralizedHyperbolic._initialize_params(...)` runs short fits for three
special cases (NIG, VG, NInvG) and picks the best warm start.

That is a one-time cost, not a per-iteration EM cost, but in large problems it
can still be noticeable. On the `n=100000, d=6` run it took about `1.95 s`.

## Answer to the original question

### "Is the bottleneck the M-step and the L-BFGS optimization?"

**Short answer:** partly, but not always.

- The GIG L-BFGS solve is definitely the dominant part of the **M-step**
- However, it is **not consistently the dominant part of the whole EM iteration**
- For moderate and especially large `n`, the **E-step Bessel work is larger**

So the more precise statement is:

- **M-step bottleneck:** yes, the GIG `eta -> theta` solve
- **whole-iteration bottleneck:** usually the E-step for large sample sizes,
  and a mix of E-step plus GIG solve for moderate sizes

## Why JAX GPU can still be slower

Based on the code path, if you call this GH code from a JAX workflow:

1. JAX arrays are coerced to NumPy arrays early
2. SciPy special functions and SciPy L-BFGS-B run on the host
3. the workload is made of many relatively small host-side operations, not one
   large JAX/XLA program

That is exactly the kind of setup where GPU often loses:

- host/device transfer overhead can dominate
- small iterative optimization loops do not saturate the GPU well
- small matrix operations and repeated synchronizations can cost more than the
  arithmetic itself

## Similar issues reported in JAX

The public reports I found are consistent with what the code suggests:

- Optax issue discussing LBFGS being much slower than SciPy on CPU
- JAX issues discussing GPU execution being slower than CPU for small iterative
  workloads
- JAX issues about slow host-to-device and device-to-host transfers
- JAX issues about batched/small matrix operations underperforming on GPU

These reports do not prove behavior for `normix` specifically, but they are
very consistent with the current GH implementation, which is:

- not JAX-native in the hot path
- heavily iterative
- dependent on SciPy optimization and special functions

## Practical takeaway

For the current checked-in code:

1. **Do not expect JAX GPU to accelerate GH EM automatically**
2. If GH is slow, first suspect:
   - E-step Bessel evaluations
   - GIG `eta -> theta` SciPy L-BFGS-B solve
   - extra `logpdf` work from verbose convergence reporting
3. Whether the GIG solve is the biggest cost depends on problem size:
   - moderate `n`: often important
   - large `n`: usually E-step dominated

## Recommended next experiment on a real GPU machine

Run the new script twice on the same dataset:

1. with plain NumPy input
2. with `--input-backend jax` so the synthetic dataset is first placed on a JAX
   device before entering the current GH path

Things to compare:

- wall time for the initial `fit(...)` call
- per-iteration E-step and M-step breakdown
- whether the total time difference is much larger than the measured GIG solve
- whether `verbose=1` materially changes the observed slowdown

Suggested commands:

```bash
python scripts/profile_gh_em_breakdown.py --n-samples 20000 --dim 12 --iters 5
python scripts/profile_gh_em_breakdown.py --n-samples 20000 --dim 12 --iters 5 --input-backend jax
python scripts/profile_gh_em_breakdown.py --n-samples 100000 --dim 6 --iters 3
```

If you test on your own GPU machine from a JAX workflow, the key question is
not "is L-BFGS slow on GPU?" alone. It is:

> how much of the GH path is still host-side SciPy/NumPy work, and how much
> transfer/synchronization is happening because the code is entered from JAX?
