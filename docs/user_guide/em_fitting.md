# Fitting with EM

The subordinator $Y$ in a normal variance-mean mixture is latent, so fitting is
a missing-data problem solved by the **EM algorithm**
({ref}`Dempster1977 <dempster1977>`; for GH,
{ref}`Hu2005 <hu2005>`, {ref}`Shi2016 <shi2016>`). normix exposes it through a
one-line convenience method and two configurable fitters.

## The algorithm

EM alternates two steps until the likelihood stops improving:

- **E-step** — given the current model, compute the conditional expectations of
  the sufficient statistics, $\mathbb{E}[t(Y) \mid X]$. For these mixtures the
  posterior $Y \mid X$ is again GIG-like, so the moments are available through
  `joint.conditional_expectations`.
- **M-step** — set the new expectation parameters $\eta$ to those conditional
  means and convert $\eta \mapsto \theta$ with `from_expectation`.

Because the M-step is just `from_expectation`, EM reuses the exact same
machinery as ordinary maximum likelihood (see {doc}`exponential_family`).

## The quick path

```python
from normix import NormalInverseGaussian

model = NormalInverseGaussian.default_init(X)
result = model.fit(X, max_iter=200, tol=1e-3)
fitted = result.model
```

`fit` returns an `EMResult` with fields `model`, `converged`, `n_iter`,
`log_likelihoods`, `param_changes`, and `elapsed_time`. The likelihood is
guaranteed non-decreasing across iterations — a handy invariant when debugging.

`solver_failed` and `diverged` are separate flags on that result.
A non-finite mixing distribution sets `solver_failed`.
On the Python loop, `SolverDidNotConverge` or `ValueError` sets that flag too.
A CPU GH warm start does not: a failed GIG solve keeps the previous GIG, and EM continues.

A non-finite normal-block update sets `diverged`.
Each flag keeps the last finite model.
`fit` raises `ValueError` when the sample is empty or non-finite.
It also raises `ValueError` when a backend is not `'jax'` or `'cpu'`.

## Batch EM

For full control, build a `BatchEMFitter` directly:

```python
from normix.fitting.em import BatchEMFitter

fitter = BatchEMFitter(
    max_iter=200, tol=1e-3, verbose=1,
    regularization="det_sigma_one",
    e_step_backend="cpu", m_step_backend="cpu", m_step_method="newton")
result = fitter.fit(model, X)
```

Key options:

- **`regularization`** fixes the scale gauge (a mixture is only identified up to
  a split between $\Sigma$ and $Y$): `'none'`, `'det_sigma_one'`
  ($\lvert\Sigma\rvert = 1$), `'det_sigma_x'` ($\lvert\Sigma\rvert$ held at its
  initial value), or `'a_eq_b'` (GIG with $a = b$).
- **`e_step_backend` / `m_step_backend`** select `'jax'` or `'cpu'`. The CPU
  path is the same `log_kv` kernel in NumPy (no JAX dispatch on the 3-D GIG
  problem).
- **`m_step_method`** is `'newton'`, `'lbfgs'`, or `'bfgs'`.

See {doc}`../tutorials/em/01_batch_em` for diagnostics and worked examples.

### Sample weights

Pass `sample_weight` to both initialization and fitting. For example,
exponentially decaying weights give recent observations more influence:

```python
import jax.numpy as jnp

weights = 0.98 ** jnp.arange(X.shape[0] - 1, -1, -1)
model = NormalInverseGaussian.default_init(X, sample_weight=weights)
result = model.fit(X, sample_weight=weights, track_ll=True)
# The direct fitter accepts the same argument:
result = fitter.fit(model, X, sample_weight=weights)
```

Weights must have shape `(n,)`, be finite and nonnegative, and contain
at least one positive entry. Invalid weights raise `ValueError` in eager
calls; traced weights have a runtime domain check.
normix normalizes them to sum to one. They weight sufficient statistics,
centering, initial moments, and mean log-likelihood. GH initialization
also uses them when fitting and ranking its candidate models.
The same API works for factor models, MCECM, and both E-step backends.
`sample_weight=None` preserves the unweighted path.

`result.log_likelihoods` records the weighted objective before each
M-step. Use `result.model.marginal_log_likelihood(X, sample_weight=weights)`
to evaluate the returned model. Scalar MAP shrinkage adds its penalty
to the trace, as in unweighted fitting.

Eta update rules receive $n_{\mathrm{eff}}=(\sum_iw_i)^2/\sum_iw_i^2$
as `batch_size`. `Shrinkage.tau` remains a relative penalty strength.
Integer weights match replicated data for likelihood EM; count-dependent
eta rules instead use this effective sample size.
See {doc}`../theory/em_algorithm` for the weighted formulas.
Exponential families also accept `sample_weight` in `fit`, `fit_mle`, and
`default_init`. Incremental EM keeps its existing step-weight API.

## Incremental (mini-batch) EM

For streaming or very large data, `IncrementalEMFitter` updates from
mini-batches ({ref}`Cappe2009 <cappe2009>`), blending each batch estimate
$\hat\eta$ into a running $\eta_t$ through an **$\eta$-update rule**:

```python
from normix.fitting.em import IncrementalEMFitter
from normix import RobbinsMonroUpdate
import jax

fitter = IncrementalEMFitter(
    batch_size=256, max_steps=200,
    eta_update=RobbinsMonroUpdate(tau0=10.0))
result = fitter.fit(model, X, key=jax.random.PRNGKey(0))
```

Six rules are available — `IdentityUpdate`, `RobbinsMonroUpdate`,
`SampleWeightedUpdate`, `EWMAUpdate`, `AffineUpdate`, and `Shrinkage` — trading
off responsiveness against variance. `Shrinkage` combined with the `eta0_*`
targets regularizes the online estimate toward a chosen structure. See
{doc}`../tutorials/em/02_incremental_em`.

## Initialization and robustness

- `default_init(X)` builds a moment-matched starting model — use it rather than
  guessing parameters.
- EM finds a *local* optimum; for rugged likelihoods, run several starts and
  keep the best, or warm-start the solver with `theta0`.
- The pure-JAX `from_expectation` solve `vmap`s, so many starts or bootstrap
  resamples can run in parallel.

These are covered in {doc}`../tutorials/em/03_initialization_and_multistart`,
and the EM vs MCECM comparison in {doc}`../tutorials/em/04_em_vs_mcecm`
reproduces a benchmark from the literature.

## Choosing an algorithm

`fit` and the fitters accept `algorithm='em'` (default) or `algorithm='mcecm'`
(Multi-Cycle ECM; {ref}`MengRubin1993 <mengrubbin1993>`,
{ref}`McNeil2010 <mcneil2010>`). EM is the right choice for almost all cases;
the two agree at the optimum.

Adam or L-BFGS on the observed-data likelihood is not a supported fitter.
The comparison, and a bring-your-own-optimiser snippet for surrounding
models, is in {doc}`../design/why_not_gradient_descent`.
