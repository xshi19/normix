# Fitting with EM

The subordinator $Y$ in a normal variance-mean mixture is latent, so fitting is
a missing-data problem solved by the **EM algorithm**. normix exposes it through
a one-line convenience method and two configurable fitters.

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
model = NormalInverseGaussian.default_init(X)
result = model.fit(X, max_iter=200, tol=1e-3)
fitted = result.model
```

`fit` returns an `EMResult` with fields `model`, `converged`, `n_iter`,
`log_likelihoods`, `param_changes`, and `elapsed_time`. The likelihood is
guaranteed non-decreasing across iterations — a handy invariant when debugging.

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
  Bessel path is markedly faster for the GIG/NIG E-step on CPU.
- **`m_step_method`** is `'newton'`, `'lbfgs'`, or `'bfgs'`.

See {doc}`../tutorials/em/01_batch_em` for diagnostics and worked examples.

## Incremental (mini-batch) EM

For streaming or very large data, `IncrementalEMFitter` updates from
mini-batches, blending each batch estimate $\hat\eta$ into a running $\eta_t$
through an **$\eta$-update rule**:

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
(a Monte Carlo conditional variant). EM is the right choice for almost all
cases; the two agree at the optimum.
