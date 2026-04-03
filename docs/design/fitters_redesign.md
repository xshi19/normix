# Fitters Redesign

> **Status:** Design (Phase 0 of [package improvement roadmap](../plans/package_improvement_roadmap.md), item D1).
>
> **Problem:** `OnlineEMFitter` and `MiniBatchEMFitter` are algorithmically
> mis-specified (see [package review §P1](../reviews/package_review_2026-03-30.md)).
> This document designs their replacement.

## Motivation

All EM variants for normal variance-mean mixtures share the same pipeline:

1. **E-step** — given batch $X$ and current $\theta$, compute per-observation
   conditional expectations and aggregate into
   $\hat\eta_{\text{batch}}$ (`NormalMixtureEta`).
2. **Combine** — apply an update rule
   $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta_{\text{batch}}$.
3. **M-step** — convert $\eta_t \to \theta_t$ (closed-form for normal params,
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

### `NormalMixtureEta` — first-class aggregated expectation parameters

The six batch-averaged sufficient statistics become an explicit pytree.
The name `NormalMixtureEta` distinguishes these from the generic exponential
family $\eta = \nabla\psi(\theta)$ (which remains a flat `jax.Array`
in `ExponentialFamily`).

```python
class NormalMixtureEta(eqx.Module):
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

**Why keep all three scalar fields?**  Only GIG (GeneralizedHyperbolic)
needs all three for its subordinator M-step.  Special cases drop one:
VG ignores `E_inv_Y`, NInvG ignores `E_Y`, NIG ignores `E_log_Y`.
However, the normal M-step always needs `E_inv_Y` and `E_Y`, and the
E-step always computes all three (posterior is GIG-like).  The savings
of omitting one scalar is negligible; a uniform struct keeps
`affine_combine` and `jax.tree.map` simple.

**Connection to efax.**  `NormalMixtureEta` is the *expectation
parametrization* of `JointNormalMixture`.  The six fields are
$\eta = E[t(X,Y)]$ where $t$ is the joint sufficient statistics.
The EM E-step estimates $\hat\eta$ from incomplete data; the M-step
is the $\eta \to \theta$ inverse map — exactly
[efax](https://github.com/NeilGirdhar/efax)'s
`ExpectationParametrization.to_nat()`.  We stop short of the full
dual-class pattern (where EP would also provide `pdf`, `rvs`, etc.)
because the M-step is not a stateless conversion: it requires solver
warm-start from the current model, backend/method kwargs, and the
MCECM two-cycle split.  These concerns belong on the model, not on
the parametrization.  If the M-step becomes simpler in the future
(e.g. closed-form $\eta \to \theta$ for all subordinators), adopting
efax's pattern would be a natural next step.

### E-step and M-step — clean decomposition

**Problem with current code.**  The responsibilities are tangled:

- `e_step(X)` returns only per-observation subordinator expectations
  `{E_log_Y: (n,), E_inv_Y: (n,), E_Y: (n,)}` — it is really
  `_e_step_subordinator`.
- `m_step(X, expectations)` takes raw data `X` plus that dict, then calls
  `_normal_params_from_expectations` which *both* aggregates batch
  statistics (E-step work: computing $E[X]$, $E[X/Y]$, $E[XX^\top/Y]$)
  *and* solves for $\mu, \gamma, \Sigma$ (M-step work).
- `_mcecm_step` in `BatchEMFitter` manually aggregates `gig_eta`
  from the per-observation dict, duplicating the aggregation logic.
- `m_step_subordinator(gig_eta)` takes a positional 3-vector; each
  concrete subclass picks out indices (`gig_eta[0]`, `gig_eta[2]`, etc.)
  with no readable names.

The true EM step is: E-step produces `NormalMixtureEta`, M-step consumes it.

#### New method signatures on `NormalMixture`

**`e_step`** — full E-step: subordinator conditionals + batch aggregation:

```python
def e_step(self, X, backend='jax') -> NormalMixtureEta:
```

**`m_step`** — full M-step: update normal params + subordinator from eta:

```python
def m_step(self, eta: NormalMixtureEta, **kwargs) -> NormalMixture:
```

`e_step` no longer returns a dict.  `m_step` no longer takes data `X`.

**MCECM helpers** (public, used by `BatchEMFitter` for split updates):

```python
def m_step_normal(self, eta: NormalMixtureEta) -> NormalMixture:
    """CM cycle 1: update (mu, gamma, Sigma), subordinator unchanged."""

def m_step_subordinator(self, eta: NormalMixtureEta, **kwargs) -> NormalMixture:
    """CM cycle 2: update subordinator, normal params unchanged."""
```

**`compute_eta_from_model() -> NormalMixtureEta`** reconstructs $\eta$
from the model's own parameters (needed to initialise the running average
in `IncrementalEMFitter`):

$$\eta_1..3 \text{ from subordinator.expectation\_params()}, \quad
\eta_4 = \mu + \gamma E[Y], \quad
\eta_5 = \mu E[1/Y] + \gamma, \quad
\eta_6 = \Sigma + \mu\mu^\top E[1/Y] + \gamma\gamma^\top E[Y]
         + \mu\gamma^\top + \gamma\mu^\top$$

#### E-step internals

`e_step` is a thin composition of two internal helpers:

```python
def e_step(self, X, backend='jax') -> NormalMixtureEta:
    sub_exp = self._e_step_subordinator(X, backend=backend)
    return self._aggregate_eta(X, sub_exp)
```

`_e_step_subordinator` is the current `e_step` — per-observation
subordinator conditional expectations via `jax.vmap` (or the CPU path).
Returns `{E_log_Y: (n,), E_inv_Y: (n,), E_Y: (n,)}`.  Bessel functions
for GIG are the expensive part.

`_aggregate_eta` averages the per-observation quantities and forms the
data-weighted statistics.  This is the logic currently buried inside
`_normal_params_from_expectations`:

```python
@staticmethod
def _aggregate_eta(X, sub_exp) -> NormalMixtureEta:
    E_inv_Y = sub_exp['E_inv_Y']   # (n,)
    return NormalMixtureEta(
        E_log_Y     = jnp.mean(sub_exp['E_log_Y']),
        E_inv_Y     = jnp.mean(E_inv_Y),
        E_Y         = jnp.mean(sub_exp['E_Y']),
        E_X         = jnp.mean(X, axis=0),                          # (d,)
        E_X_inv_Y   = jnp.mean(X * E_inv_Y[:, None], axis=0),      # (d,)
        E_XXT_inv_Y = jnp.mean(
            jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y), axis=0),    # (d,d)
    )
```

`_normal_params_from_expectations` is deleted.

#### M-step internals

`m_step` calls `m_step_normal` then `m_step_subordinator`:

```python
def m_step(self, eta: NormalMixtureEta, **kwargs) -> NormalMixture:
    model = self.m_step_normal(eta)
    return model.m_step_subordinator(eta, **kwargs)
```

`m_step_normal` calls `JointNormalMixture._mstep_normal_params(eta)`,
which reads `eta.E_X`, `eta.E_X_inv_Y`, `eta.E_XXT_inv_Y`,
`eta.E_inv_Y`, `eta.E_Y` by name (replacing the current five positional
arguments).

`m_step_subordinator` is abstract; each concrete subclass reads the
fields it needs from `eta`:

| Distribution | Current | New |
|---|---|---|
| VG | `Gamma.from_expectation([gig_eta[0], gig_eta[2]])` | `Gamma.from_expectation([eta.E_log_Y, eta.E_Y])` |
| NInvG | `InverseGamma.from_expectation([-gig_eta[1], gig_eta[0]])` | `InverseGamma.from_expectation([-eta.E_inv_Y, eta.E_log_Y])` |
| NIG | `InverseGaussian.from_expectation([gig_eta[2], gig_eta[1]])` | `InverseGaussian.from_expectation([eta.E_Y, eta.E_inv_Y])` |
| GH | `GIG.from_expectation(gig_eta)` | `GIG.from_expectation(jnp.array([eta.E_log_Y, eta.E_inv_Y, eta.E_Y]))` |

The named fields replace cryptic index access.

#### How the fitter calls them

EM (`_em_step`):

```python
eta = model.e_step(X, backend=self.e_step_backend)
model = model.m_step(eta, backend=self.m_step_backend, method=self.m_step_method)
model = self._regularize(model)
```

MCECM (`_mcecm_step`):

```python
eta = model.e_step(X, backend=self.e_step_backend)
model = model.m_step_normal(eta)
model = self._regularize(model)
eta = model.e_step(X, backend=self.e_step_backend)
model = model.m_step_subordinator(eta, backend=self.m_step_backend,
                                  method=self.m_step_method)
```

The fitter's `_m_step` helper is deleted — `model.m_step(eta, **kwargs)`
handles backend/method directly.

### Affine combination

Since every update rule is $\eta_t = a + b\,\eta_{t-1} + c\,\hat\eta$,
a single function suffices:

```python
def affine_combine(
    eta_prev: NormalMixtureEta, eta_new: NormalMixtureEta,
    b: float, c: float, a: NormalMixtureEta | None = None,
) -> NormalMixtureEta:
```

This is JIT-able (pure function on pytrees, scalar weights).

### `EtaUpdateRule` — strategy objects

Each rule computes $(a, b, c)$ and manages any mutable state (cumulative
sample count, step counter).

```python
class EtaUpdateRule(ABC):
    @abstractmethod
    def weights(self, step: int, batch_size: int, state: dict
                ) -> tuple[NormalMixtureEta | None, float, float, dict]:
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
            eta = model.e_step(X_batch)
            model = model.m_step(eta)
        eta_new = model.compute_eta_from_model()
    else:
        # Online: one E-step gives aggregated eta directly
        eta_new = model.e_step(X_batch)

    a, b, c, state = eta_update.weights(step, len(X_batch), state)
    eta_prev = affine_combine(eta_prev, eta_new, b, c, a)
    model = model.m_step(eta_prev)

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

Three phases.  Each phase ends with all tests passing.

### Phase 1 — `NormalMixtureEta` + E-step / M-step refactor

Introduce the pytree, refactor `e_step` and `m_step` on `NormalMixture`,
and update all concrete distributions and the fitter to match.

**Source code:**

| File | Change |
|------|--------|
| `normix/fitting/eta.py` (new) | `NormalMixtureEta` class |
| `normix/mixtures/marginal.py` | Rename current `e_step` → `_e_step_subordinator`; `_e_step_cpu` stays as internal helper of `_e_step_subordinator`; add `_aggregate_eta`; new `e_step` composing both → returns `NormalMixtureEta`; `m_step(eta, **kwargs)` replaces `m_step(X, expectations, **kwargs)`; `m_step_normal(eta)` replaces `m_step_normal(X, expectations)`; abstract `m_step_subordinator` signature changes from `(gig_eta, **kwargs)` to `(eta: NormalMixtureEta, **kwargs)`; delete `_normal_params_from_expectations` |
| `normix/mixtures/joint.py` | `_mstep_normal_params(eta: NormalMixtureEta)` — reads named fields instead of 5 positional args |
| `normix/distributions/variance_gamma.py` | `m_step_subordinator(eta, **kwargs)` — read `eta.E_log_Y`, `eta.E_Y` instead of `gig_eta[0]`, `gig_eta[2]` |
| `normix/distributions/normal_inverse_gamma.py` | `m_step_subordinator(eta, **kwargs)` — read `-eta.E_inv_Y`, `eta.E_log_Y` instead of `-gig_eta[1]`, `gig_eta[0]` |
| `normix/distributions/normal_inverse_gaussian.py` | `m_step_subordinator(eta, **kwargs)` — read `eta.E_Y`, `eta.E_inv_Y` instead of `gig_eta[2]`, `gig_eta[1]` |
| `normix/distributions/generalized_hyperbolic.py` | `m_step_subordinator(eta, **kwargs)` — assemble `jnp.array([eta.E_log_Y, eta.E_inv_Y, eta.E_Y])` for `GIG.from_expectation` |
| `normix/fitting/em.py` | `_em_step`: `eta = model.e_step(X, ...)` + `model.m_step(eta, ...)`; `_mcecm_step`: `m_step_normal(eta)` / `m_step_subordinator(eta, ...)`; delete `_m_step` helper and manual `gig_eta` aggregation |

**Tests updated:**

| Test file | Change |
|-----------|--------|
| `tests/test_jax_distributions.py` | `e_step` returns `NormalMixtureEta`; `m_step(X, exp)` → `m_step(eta)` |
| `tests/test_em_cpu_vs_jax.py` | Same signature migration; compare `NormalMixtureEta` fields instead of dict keys |
| `tests/test_mcecm.py` | `m_step_normal(X, exp)` → `m_step_normal(eta)`; `m_step_subordinator(gig_eta)` → `m_step_subordinator(eta)` |
| `tests/test_cpu_bessel_backend.py` | `e_step` returns `NormalMixtureEta`; compare `.E_log_Y` etc. instead of dict keys |

### Phase 2 — `IncrementalEMFitter` + eta update rules

Build the new fitter infrastructure on top of the Phase 1 API.

| File | Change |
|------|--------|
| `normix/fitting/eta.py` | Add `affine_combine` function |
| `normix/fitting/eta_rules.py` (new) | `EtaUpdateRule` ABC; `IdentityUpdate`, `RobbinsMonroUpdate`, `SampleWeightedUpdate`, `EWMAUpdate`, `ShrinkageUpdate`, `AffineUpdate` |
| `normix/mixtures/marginal.py` | Add `compute_eta_from_model() -> NormalMixtureEta` |
| `normix/fitting/em.py` | Add optional `eta_update` param to `BatchEMFitter`; new `IncrementalEMFitter`; delete `OnlineEMFitter`, `MiniBatchEMFitter` |
| `normix/fitting/__init__.py` | Export new public names |

**New tests:** `tests/test_incremental_em.py` — online, sample-weighted,
EWMA, fine-tuning, and shrinkage variants.

### Phase 3 — docs, notebooks, scripts

No source logic changes.  Update all references to old signatures.

| File | Change |
|------|--------|
| `docs/ARCHITECTURE.md` | Update E-step / M-step descriptions (lines ~149, 156, 159) |
| `docs/design/design.md` | Update `_mstep_normal_params` description and `m_step` example |
| `docs/design.rst` | Update `m_step` call example |
| `docs/architecture.rst` | Update `_mstep_normal_params` description |
| `docs/tech_notes/em_gpu_profiling.md` | Update `e_step` / `m_step` usage |
| `docs/plans/migration_plan.md` | Update `e_step` reference |
| `docs/references/distribution_packages.md` | Update `m_step` example |
| `notebooks/hellinger_distance.ipynb` | Update `_mstep_normal_params` call |
| `scripts/benchmark_comprehensive.py` | Update all `e_step` / `m_step` call sites |

### What stays unchanged

- `BatchEMFitter._fit_scan` and `_fit_loop` iteration machinery
- `NormalMixture.fit`, `default_init`, `log_prob`, `pdf`, `rvs`
- `JointNormalMixture`: `conditional_expectations`,
  `_compute_posterior_expectations`, `_posterior_gig_params`, `_quad_forms`,
  `_precision_quantities`, `_assemble_natural_params`, `_parse_joint_theta`,
  `sufficient_statistics`, `log_base_measure`, `natural_params`,
  `_log_partition_from_theta`
- Solver infrastructure (`solvers.py`)
- `ExponentialFamily` base class — `theta` and `eta` remain flat `jax.Array`
- Natural/classical parameter representations (no structured theta pytrees)

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
