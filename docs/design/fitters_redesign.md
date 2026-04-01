# Fitters Redesign

> **Status:** Design (Phase 0 of [package improvement roadmap](../plans/package_improvement_roadmap.md), item D1).
>
> **Problem:** `OnlineEMFitter` and `MiniBatchEMFitter` are algorithmically
> mis-specified (see [package review §P1](../reviews/package_review_2026-03-30.md)).
> This document designs their replacement.

## Motivation

All EM variants for normal variance-mean mixtures share the same pipeline:

1. **E-step** — given batch $X$ and current $\theta$, compute per-observation
   $E[t(X,Y) \mid X=x_i, \theta]$.
2. **Aggregate** — average over the batch to get $\hat\eta_{\text{batch}}$.
3. **Combine** — apply an update rule
   $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta_{\text{batch}}$.
4. **M-step** — convert $\eta_t \to \theta_t$ (closed-form for normal params,
   Bregman solver for subordinator).

Every fitter variant is a choice of $(a, b, c)$ and iteration strategy:

| Variant | Data per step | Inner iters | $a$ | $b$ | $c$ |
|---------|--------------|-------------|-----|-----|-----|
| Batch EM | Full dataset | 1 | 0 | 0 | 1 |
| Batch EM + shrinkage | Full dataset | 1 | $\frac{\tau}{1+\tau}\eta_0$ | 0 | $\frac{1}{1+\tau}$ |
| Fine-tuning | Mini-batch | $k \geq 1$ | 0 | 0 | 1 |
| Online (Robbins-Monro) | Mini-batch | 1 | 0 | $1 - \tau_t^{-1}$ | $\tau_t^{-1}$ |
| Online (sample-weighted) | Mini-batch | 1 | 0 | $\frac{n}{m+n}$ | $\frac{m}{m+n}$ |
| Online (EWMA) | Mini-batch | 1 | 0 | $1-w$ | $w$ |
| Online + shrinkage | Mini-batch | 1 | nonzero | nonzero | nonzero |

The current codebase has three fitter classes that do not share this structure.
The online and mini-batch fitters compute `step` but never apply it, run
ordinary batch E/M on each slice, and return `converged=True` unconditionally.

---

## Design

### `EtaSummary` — first-class expectation parameters

The six averaged sufficient statistics become an explicit pytree:

```python
class EtaSummary(eqx.Module):
    E_log_Y: jax.Array      # scalar
    E_inv_Y: jax.Array      # scalar
    E_Y: jax.Array          # scalar
    E_X: jax.Array          # (d,)
    E_X_inv_Y: jax.Array    # (d,)
    E_XXT_inv_Y: jax.Array  # (d, d)
```

**Why a pytree, not a flat vector?**  The six components have heterogeneous
shapes (scalars, vectors, matrices).  A flat vector requires packing/unpacking
and knowing $d$.  A pytree is more readable and `jax.tree.map` applies the
affine weights naturally.

Two new methods on `NormalMixture`:

- `compute_eta(X, expectations) -> EtaSummary` — factored out of the current
  `_normal_params_from_expectations`.
- `m_step_from_eta(eta: EtaSummary, **kwargs) -> NormalMixture` — applies the
  closed-form normal M-step + subordinator solver.

The existing `m_step(X, expectations)` becomes sugar for
`m_step_from_eta(compute_eta(X, expectations))`.

A new method `compute_eta_from_model() -> EtaSummary` reconstructs $\eta$
from the model's own parameters (needed to initialise the running average):

$$\eta_1..3 \text{ from subordinator.expectation\_params()}, \quad
\eta_4 = \mu + \gamma E[Y], \quad
\eta_5 = \mu E[1/Y] + \gamma, \quad
\eta_6 = \Sigma + \mu\mu^\top E[1/Y] + \gamma\gamma^\top E[Y]
         + \mu\gamma^\top + \gamma\mu^\top$$

### Affine combination

Since every update rule is $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta$,
a single function suffices:

```python
def affine_combine(
    eta_prev: EtaSummary, eta_new: EtaSummary,
    b: float, c: float, a: EtaSummary | None = None,
) -> EtaSummary:
```

This is JIT-able (pure function on pytrees, scalar weights).

### `EtaUpdateRule` — strategy objects

Each rule computes $(a, b, c)$ and manages any mutable state (cumulative
sample count, step counter).

```python
class EtaUpdateRule(ABC):
    @abstractmethod
    def weights(self, step: int, batch_size: int, state: dict
                ) -> tuple[EtaSummary | None, float, float, dict]:
        """Returns (a, b, c, updated_state)."""

    def initial_state(self) -> dict:
        return {}
```

Concrete rules:

| Rule | $a$ | $b$ | $c$ | State |
|------|-----|-----|-----|-------|
| `IdentityUpdate` | `None` | 0 | 1 | — |
| `RobbinsMonroUpdate(tau0)` | `None` | $1 - 1/\tau_t$ | $1/\tau_t$ | step counter (implicit) |
| `SampleWeightedUpdate` | `None` | $n/(m{+}n)$ | $m/(m{+}n)$ | `cumulative_n` |
| `EWMAUpdate(w)` | `None` | $1{-}w$ | $w$ | — |
| `ShrinkageUpdate(eta0, tau)` | $\frac{\tau}{1+\tau}\eta_0$ | 0 | $\frac{1}{1+\tau}$ | — |
| `AffineUpdate(a, b, c)` | user | user | user | user-defined |

For `AffineUpdate`, $a$, $b$, $c$ can be callables
`(step, batch_size, state) -> value` for time-varying schedules.

### Fitter classes

Two fitters with distinct convergence semantics.

**`BatchEMFitter`** (existing, minimally changed):

- Processes full dataset per iteration, monitors convergence via $\Delta\theta$.
- New optional `eta_update` parameter (default `IdentityUpdate`).
- Enables penalized batch EM (shrinkage) without a separate fitter class.

**`IncrementalEMFitter`** (replaces `OnlineEMFitter` + `MiniBatchEMFitter`):

- Processes data in batches, fixed iteration budget.
- The `eta_update` rule is mandatory (default `RobbinsMonroUpdate(tau0=10)`).
- `inner_iter` parameter: 1 for online EM, $>1$ for fine-tuning.

```python
class IncrementalEMFitter:
    eta_update: EtaUpdateRule    # how to combine η
    batch_size: int              # observations per batch
    max_steps: int               # number of batches to process
    inner_iter: int              # 1 = online, >1 = fine-tuning

    def fit(self, model, X, *, key) -> EMResult: ...
```

**Why keep `BatchEMFitter` separate?**  Batch EM has fundamentally different
convergence guarantees (monotone log-likelihood increase, well-defined
stopping criterion).  Merging it with the incremental fitter would muddy the
API: `tol` is meaningful for batch EM but not for online EM.

---

## `IncrementalEMFitter.fit` pseudocode

```
eta_prev = model.compute_eta_from_model()
state = eta_update.initial_state()

for step in range(max_steps):
    X_batch = sample_batch(X, batch_size, key)

    if inner_iter > 1:
        # Fine-tuning: run multiple EM iterations on this batch
        for _ in range(inner_iter):
            expectations = model.e_step(X_batch)
            model = model.m_step(X_batch, expectations)
        eta_new = model.compute_eta_from_model()
    else:
        # Online: one E-step, aggregate
        expectations = model.e_step(X_batch)
        eta_new = compute_eta(X_batch, expectations)

    a, b, c, state = eta_update.weights(step, len(X_batch), state)
    eta_prev = affine_combine(eta_prev, eta_new, b, c, a)
    model = model.m_step_from_eta(eta_prev)

return EMResult(model=model, ...)
```

---

## Use cases

**Batch EM** (unchanged):

```python
BatchEMFitter(max_iter=200, tol=1e-3).fit(model, X)
```

**Batch EM with shrinkage** ([theory](../theory/shrinkage.rst)):

```python
eta_prior = model_prior.compute_eta_from_model()
BatchEMFitter(
    eta_update=ShrinkageUpdate(eta_prior, tau=0.5),
    max_iter=200, tol=1e-3,
).fit(model, X)
```

**Online EM — Robbins-Monro** ([theory](../theory/online_em.rst)):

```python
IncrementalEMFitter(
    eta_update=RobbinsMonroUpdate(tau0=10),
    batch_size=1, max_steps=n,
).fit(model, X, key=key)
```

**Online EM — sample-weighted batches:**

```python
IncrementalEMFitter(
    eta_update=SampleWeightedUpdate(),
    batch_size=256, max_steps=100,
).fit(model, X, key=key)
```

**Fine-tuning (mini-batch EM, few inner steps):**

```python
IncrementalEMFitter(
    eta_update=IdentityUpdate(),
    batch_size=256, max_steps=50, inner_iter=5,
).fit(model, X, key=key)
```

**Online EM with shrinkage:**

```python
IncrementalEMFitter(
    eta_update=AffineUpdate(
        a=lambda s, m, st: jax.tree.map(lambda x: 0.1 * x, eta_prior),
        b=0.85, c=0.05,
    ),
    batch_size=64, max_steps=500,
).fit(model, X, key=key)
```

---

## Implementation plan

### Changes to existing code

| File | Change |
|------|--------|
| `normix/fitting/eta.py` (new) | `EtaSummary`, `affine_combine` |
| `normix/fitting/eta_rules.py` (new) | `EtaUpdateRule` hierarchy |
| `normix/mixtures/marginal.py` | Add `compute_eta`, `m_step_from_eta`, `compute_eta_from_model` |
| `normix/fitting/em.py` | Add `eta_update` param to `BatchEMFitter`; new `IncrementalEMFitter` |
| `normix/fitting/em.py` | Delete `OnlineEMFitter`, `MiniBatchEMFitter` |
| `normix/fitting/__init__.py` | Update exports |

### What stays unchanged

- `BatchEMFitter._fit_scan` and `_fit_loop` paths
- `NormalMixture.e_step`, `m_step`, `fit`, `default_init`
- All distribution and joint-mixture code
- Solver infrastructure (`solvers.py`)

---

## Mathematical connection to theory docs

The affine combination $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta$ unifies:

- **Online EM** ([online_em.rst](../theory/online_em.rst)): the update rule
  $\eta_t = \eta_{t-1} + \tau_t^{-1}(\bar t(x_t|\theta_{t-1}) - \eta_{t-1})$
  is $b = 1 - \tau_t^{-1}$, $c = \tau_t^{-1}$, $a = 0$.

- **Shrinkage** ([shrinkage.rst](../theory/shrinkage.rst)): the penalised
  M-step computes shrunk sufficient statistics
  $\hat\eta = \frac{1}{1+\tau}\hat\eta_{\text{batch}} + \frac{\tau}{1+\tau}\eta_0$,
  which is $a = \frac{\tau}{1+\tau}\eta_0$, $b = 0$, $c = \frac{1}{1+\tau}$.

- **Sample-weighted**: when previous state represents $n$ observations and
  the new batch has $m$, the weight $c = m/(m+n)$ yields the standard
  incremental mean update. The fitter tracks cumulative $n$ in its state.

- **EWMA**: constant $c = w$ gives exponentially weighted moving average
  semantics. Older batches decay geometrically.
