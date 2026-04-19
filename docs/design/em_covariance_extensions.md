# EM Extensions: Shrinkage and Factor Analysis

**Date:** 2026-04-17
**Status:** Proposed — design finalised, implementation pending
**Scope:** `normix/fitting/eta.py`, `normix/fitting/eta_rules.py`,
`normix/fitting/em.py`, `normix/mixtures/marginal.py`,
new `normix/mixtures/factor.py`
**Theory:** `docs/theory/shrinkage.rst`, `docs/theory/factor_analysis.rst`
**Companion docs (planned):** `docs/design/penalised_em.md`

---

## 1. Scope

This note finalises the design for two related extensions of the current EM
machinery:

1. **Penalised EM with flexible shrinkage** — give the user the ability to
  shrink any subset of the six expectation parameters (notably the `Σ`
   block alone), and to compose shrinkage with any incremental update rule.
2. **Factor-analysis mixture family** — implement
  `Σ = F Fᵀ + D` as a sibling of the existing full-covariance family,
   without forcing it into `JointNormalMixture`'s six-statistic exponential
   structure.

It also makes a small set of EM-API changes that both extensions need:

- generalise `NormalMixtureEta` field naming and ordering;
- generalise the convergence hook in `BatchEMFitter` and
`IncrementalEMFitter`;
- introduce a thin `MarginalMixture` ABC so the fitter does not need to know
whether the model is full-covariance or factor-structured.

Out of scope:

- A `DispersionModel` / `FullDispersion` / `FactorDispersion` abstraction.
Deferred until at least three storage variants are needed (see §8).
- SVD-based full covariance. Same reason.
- A new finance layer. See `docs/design/finance_architecture.md`.

---

## 2. Current Design Recap

The fitter contract today is intentionally narrow:

- `model.e_step(X) -> NormalMixtureEta` — six aggregated expectation
parameters as an `eqx.Module` pytree.
- `model.m_step(eta) -> NormalMixture` — closed-form normal-parameter
update plus a numerical subordinator update.
- `model.compute_eta_from_model() -> NormalMixtureEta` — used by the
incremental fitter to initialise its running η.
- `EtaUpdateRule.weights(step, batch_size, state) -> (a, b, c, state)`
drives the affine combination
`η_t = a + b · η_{t-1} + c · η̂` via `affine_combine`.
- Convergence is measured by `_param_change(mu, gamma, L_Sigma, …)` —
hard-coded to the three full-covariance fields.

`ShrinkageUpdate(eta0, tau)` already implements the penalised-EM theory in
`docs/theory/shrinkage.rst` for the special case of batch EM with uniform
`τ` over all six statistics.

This design is good enough for the standard full-covariance models. The
gaps are:

- shrinkage cannot target `Σ` alone (or any subset of the six statistics);
- shrinkage cannot be combined with `RobbinsMonro` / `EWMA` /
`SampleWeighted` running updates (composing them today would discard the
running state, see §4.3);
- the convergence check refers to `mu, gamma, L_Sigma` directly, so a
factor-analysis model with `(mu, gamma, F, D)` cannot reuse the fitter
unchanged;
- no marginal interface exists for fitters and downstream code to depend
on, only the concrete `NormalMixture` class.

---

## 3. Stats Pytrees (`NormalMixtureEta`, `FactorMixtureStats`)

### 3.1 Naming and ordering

Both stats classes use **descriptive field names** in **theory order**
(`s_1, s_2, …` from `docs/theory/shrinkage.rst` and
`docs/theory/factor_analysis.rst`). The first six fields are identical
between the two so that shrinkage targets, weights, and tests written for
the standard family port directly to the factor family.

```python
class NormalMixtureEta(eqx.Module):
    E_inv_Y:     jax.Array  # s_1 = E[Y^{-1}],   scalar
    E_Y:         jax.Array  # s_2 = E[Y],        scalar
    E_log_Y:     jax.Array  # s_3 = E[log Y],    scalar
    E_X:         jax.Array  # s_4 = E[X],        (d,)
    E_X_inv_Y:   jax.Array  # s_5 = E[X / Y],    (d,)
    E_XXT_inv_Y: jax.Array  # s_6 = E[X X^T/Y],  (d, d)


class FactorMixtureStats(eqx.Module):
    E_inv_Y:           jax.Array   # s_1
    E_Y:               jax.Array   # s_2
    E_log_Y:           jax.Array   # s_3
    E_X:               jax.Array   # s_4
    E_X_inv_Y:         jax.Array   # s_5
    E_XXT_inv_Y:       jax.Array   # s_6
    E_XZT_inv_sqrtY:   jax.Array   # s_7  = E[X Z^T Y^{-1/2}],  (d, r)
    E_Z_inv_sqrtY:     jax.Array   # s_8  = E[Z Y^{-1/2}],      (r,)
    E_Z_sqrtY:         jax.Array   # s_9  = E[Z Y^{1/2}],       (r,)
    E_ZZT:             jax.Array   # s_10 = E[Z Z^T],           (r, r)
```

Field order in `NormalMixtureEta` changes from
`(E_log_Y, E_inv_Y, E_Y, E_X, E_X_inv_Y, E_XXT_inv_Y)` to the theory
order above. This is a one-time, pre-1.0 break; all existing references
get updated in the same commit.

### 3.2 Why a separate `FactorMixtureStats`

The factor-analysis complete-data log-likelihood (`factor_analysis.rst`,
§Log-Likelihood Function) depends on four additional statistics involving
the latent factor `Z`. Reusing `NormalMixtureEta` would either pad it
with unused fields or add a discriminated union — both worse than a
sibling pytree. The fitter only needs to know the model's stats type
through `model.e_step` / `model.m_step` / `model.compute_eta_from_model`.

---

## 4. Affine Combination and Shrinkage Combinator

### 4.1 Generalised `affine_combine`

`a, b, c` may be either:

- a scalar (Python `float` or 0-d array) — broadcast to all leaves,
matching today's behaviour, **or**
- a stats-shaped pytree (`NormalMixtureEta` for the standard family,
`FactorMixtureStats` for the factor family) — applied per leaf.

Each leaf of `a, b, c` may itself be a scalar (broadcast over the leaf's
shape) or an array matching that leaf's shape (per-element control). This
gives three useful regimes through one API:


| Regime          | `b, c` form                                         |
| --------------- | --------------------------------------------------- |
| Uniform (today) | scalars                                             |
| Per-field       | stats pytree with scalar leaves                     |
| Per-element     | stats pytree with leaves matching `eta` leaf shapes |


Implementation sketch:

```python
def affine_combine(eta_prev, eta_new, b, c, a=None):
    is_pytree = isinstance(b, type(eta_prev))
    if is_pytree:
        result = jax.tree.map(
            lambda p, n, bi, ci: bi * p + ci * n,
            eta_prev, eta_new, b, c,
        )
    else:
        result = jax.tree.map(
            lambda p, n: b * p + c * n, eta_prev, eta_new,
        )
    if a is not None:
        result = jax.tree.map(lambda r, ai: r + ai, result, a)
    return result
```

JIT-ability is preserved: scalar leaves are JAX scalars, pytree weights
are leaves of the same kind as `eta`.

### 4.2 Shrinkage as a combinator

`ShrinkageUpdate(eta0, tau)` is replaced by a combinator that wraps **any**
base rule:

```python
class Shrinkage(EtaUpdateRule):
    """Pull the running η toward a prior, on top of any base update rule.

    Composition (single equation, per-field):

        η_t = (τ / (1+τ)) ⊙ η_0
            + (1 / (1+τ)) ⊙ (a_b + b_b ⊙ η_{t-1} + c_b ⊙ η̂)

    where (a_b, b_b, c_b) come from `base.weights(...)`, ⊙ is per-field
    multiplication (broadcast for scalar τ), and τ may be either:

    - a Python scalar / 0-d array — uniform shrinkage on all six
      sufficient statistics (matches the penalised-MLE in
      `docs/theory/shrinkage.rst`);
    - a NormalMixtureEta with scalar leaves — per-field shrinkage, e.g.
      τ = NormalMixtureEta(0, 0, 0, 0, 0, τ_Σ) shrinks only η_6 = Σ.
    """
    base: EtaUpdateRule
    eta0: NormalMixtureEta
    tau: Union[jax.Array, NormalMixtureEta]

    def initial_state(self):
        return self.base.initial_state()

    def weights(self, step, batch_size, state):
        a_b, b_b, c_b, state = self.base.weights(step, batch_size, state)
        # τ → factor_a = τ/(1+τ), factor_rest = 1/(1+τ); supports
        # scalar or pytree τ via jax.tree.map.
        ...
        return a_new, b_new, c_new, state
```

Resulting concrete patterns:


| Use case                              | Construction                                    |
| ------------------------------------- | ----------------------------------------------- |
| Batch EM, uniform shrinkage           | `Shrinkage(IdentityUpdate(), eta0, tau)`        |
| Batch EM, Σ-only shrinkage            | `Shrinkage(IdentityUpdate(), eta0, tau_pytree)` |
| Robbins–Monro online + shrinkage      | `Shrinkage(RobbinsMonroUpdate(τ0), eta0, tau)`  |
| EWMA online + shrinkage               | `Shrinkage(EWMAUpdate(w), eta0, tau)`           |
| Sample-weighted incremental + shrink. | `Shrinkage(SampleWeightedUpdate(), eta0, tau)`  |


The current `ShrinkageUpdate` class is removed (pre-1.0 rename). The base
rules (`IdentityUpdate`, `RobbinsMonroUpdate`, `SampleWeightedUpdate`,
`EWMAUpdate`, `AffineUpdate`) are unchanged.

### 4.3 Why a combinator (not `ShrunkX` copies)

Composing shrinkage with a running rule by hand requires getting the
algebra right twice (once for `b_b, c_b`, once for the prior). A
combinator does it in one place. The four hypothetical
`ShrunkRobbinsMonroUpdate` / `ShrunkEWMAUpdate` / `ShrunkSampleWeighted` /
`ShrunkIdentityUpdate` classes collapse into one `Shrinkage(base, eta0, tau)`, with no behaviour lost.

### 4.4 Shrinkage targets

A short helper module `normix/fitting/shrinkage_targets.py` will provide
constructors for common priors:

```python
def eta0_from_model(joint: JointNormalMixture) -> NormalMixtureEta: ...
def eta0_isotropic(joint: JointNormalMixture, sigma2: float
                   ) -> NormalMixtureEta: ...
def eta0_diagonal(joint: JointNormalMixture, diag: jax.Array
                  ) -> NormalMixtureEta: ...
def eta0_with_sigma(joint: JointNormalMixture, Sigma0: jax.Array
                    ) -> NormalMixtureEta: ...
```

All four return a complete six-field `NormalMixtureEta`. The `Σ`-only
variants reuse the current model's `(μ, γ, p, a, b)` to fill the other
five fields, so the result is still a coherent prior expectation
parameter — the user's per-field `tau` then decides which fields are
actually shrunk. This keeps the public contract simple ("η_0 is always a
full prior") while supporting "shrink Σ only" through `tau`, not through
half-built `eta0`.

---

## 5. Convergence Hook Generalisation

### 5.1 `em_convergence_params` on the marginal

Each marginal mixture exposes a pytree of parameters whose change measures
convergence:

```python
class NormalMixture(MarginalMixture):
    def em_convergence_params(self) -> "ConvergencePytree":
        j = self._joint
        return (j.mu, j.gamma, j.L_Sigma)


class FactorNormalMixture(MarginalMixture):
    def em_convergence_params(self) -> "ConvergencePytree":
        Sigma = self.F @ self.F.T + jnp.diag(self.D)
        return (self.mu, self.gamma, Sigma)
```

Subordinator parameters (`p, a, b`) are still excluded — that decision
stays as in `design.md` (their solver has its own tolerance, and including
them inflates iteration counts).

### 5.2 `_param_change` on pytrees

The fitter computes the max relative L2 change across leaves:

```python
def _param_change(new_params, old_params) -> jax.Array:
    leaves_new = jax.tree.leaves(new_params)
    leaves_old = jax.tree.leaves(old_params)
    rels = [
        jnp.linalg.norm(n - o) /
            jnp.maximum(jnp.linalg.norm(o), _PARAM_EPS)
        for n, o in zip(leaves_new, leaves_old)
    ]
    return jnp.max(jnp.stack(rels))
```

`BatchEMFitter._em_step` and `_mcecm_step` call `model.em_convergence_params()`
once before and once after each iteration; `IncrementalEMFitter` does the
same per batch. No model-specific code in the fitter.

### 5.3 `F` identifiability

`F` is identifiable only up to a right `r × r` orthogonal rotation, so
`(μ, γ, F, D)` would never converge in norm. Returning
`Σ = F Fᵀ + D` from `em_convergence_params` sidesteps the issue with no
extra orthogonalisation step.

---

## 6. Marginal Mixture API

### 6.1 ABC

A new `normix/mixtures/marginal.py::MarginalMixture` ABC formalises the
interface that fitters and the divergences module already rely on:

```python
class MarginalMixture(eqx.Module):
    # --- distribution surface ---
    def log_prob(self, x: jax.Array) -> jax.Array: ...
    def pdf(self, x: jax.Array) -> jax.Array: ...
    def mean(self) -> jax.Array: ...
    def cov(self) -> jax.Array: ...
    def rvs(self, n: int, seed: int = 42) -> jax.Array: ...
    def marginal_log_likelihood(self, X: jax.Array) -> jax.Array: ...

    # --- EM hooks (stats type chosen by subclass) ---
    def e_step(self, X: jax.Array, *, backend: str = 'jax'): ...
    def m_step(self, eta, **kwargs) -> "MarginalMixture": ...
    def m_step_normal(self, eta) -> "MarginalMixture": ...
    def m_step_subordinator(self, eta, **kwargs) -> "MarginalMixture": ...
    def compute_eta_from_model(self): ...
    def em_convergence_params(self): ...

    # --- convenience ---
    def fit(self, X, **kwargs) -> "EMResult": ...
```

Existing `NormalMixture` becomes a subclass of `MarginalMixture`; its
implementation does not change. `FactorNormalMixture` is the second
subclass. The fitter type-hints against `MarginalMixture`.

### 6.2 `compute_eta_from_model` for the factor family

For `FactorNormalMixture`, this returns a `FactorMixtureStats`. The first
six fields are the same closed-form expressions as in `NormalMixture`
(using `Σ = F Fᵀ + D`); the four `Z`-related fields are computed from the
deterministic relations in `factor_analysis.rst` §E-Step using the
current model's `β`. This is well-defined because the latent `Z` has
known posterior conditional on `(X, Y, parameters)`.

---

## 7. Factor Analysis Model Family

### 7.1 Class hierarchy

```
MarginalMixture(eqx.Module)
├── NormalMixture                       # owns JointNormalMixture (full Σ)
│   ├── VarianceGamma
│   ├── NormalInverseGamma
│   ├── NormalInverseGaussian
│   └── GeneralizedHyperbolic
└── FactorNormalMixture                 # stores (μ, γ, F, D, subordinator)
    ├── FactorVarianceGamma
    ├── FactorNormalInverseGamma
    ├── FactorNormalInverseGaussian
    └── FactorGeneralizedHyperbolic
```

`FactorNormalMixture` does **not** own a `JointNormalMixture` — the
complete-data structure is over `(X, Y, Z)`, not `(X, Y)`, and its
sufficient statistics are different. Trying to share storage would force
the joint hierarchy to lie about its exponential-family signature.

### 7.2 Parameter storage

```python
class FactorNormalMixture(MarginalMixture):
    mu:    jax.Array          # (d,)
    gamma: jax.Array          # (d,)
    F:     jax.Array          # (d, r)
    D:     jax.Array          # (d,) diagonal entries of diag(D)
    subordinator: ExponentialFamily
```

`subordinator` is stored as a full `ExponentialFamily` field (per
`design.md` D2 and the user-confirmed convention), not as raw `(p, a, b)`
on the mixture. This matches the user-visible API of the existing joint
classes that also expose `subordinator()` as a method.

`D` is stored as a `(d,)` vector of diagonal entries; it is constrained to
be positive by floor `D = jnp.maximum(D, D_FLOOR)` after each M-step
(value defined in `utils/constants.py`).

### 7.3 Closed-form Woodbury helpers

Private methods on `FactorNormalMixture` provide the linear algebra
without a public dispersion class. All operate in `O(d r²)` or `O(r³)`,
never forming a dense `d × d` solve:

```python
def _M(self) -> jax.Array:
    # M = I_r + F^T D^{-1} F,  shape (r, r)
    return jnp.eye(self.F.shape[1]) + (self.F.T / self.D) @ self.F

def _solve(self, x: jax.Array) -> jax.Array:
    # Σ^{-1} x via Woodbury
    Dinv_x = x / self.D
    inner = jax.scipy.linalg.solve(self._M(), self.F.T @ Dinv_x,
                                   assume_a='pos')
    return Dinv_x - (self.F @ inner) / self.D

def _quad_form(self, x: jax.Array) -> jax.Array:
    return jnp.dot(x, self._solve(x))

def _log_det_sigma(self) -> jax.Array:
    # log|Σ| = log|D| + log|I_r + F^T D^{-1} F|
    sign, logdet_M = jnp.linalg.slogdet(self._M())
    return jnp.sum(jnp.log(self.D)) + logdet_M

def _beta(self) -> jax.Array:
    # β = F^T D^{-1} - F^T D^{-1} F · M^{-1} · F^T D^{-1}
    FtDinv = self.F.T / self.D                       # (r, d)
    return FtDinv - jax.scipy.linalg.solve(
        self._M(), FtDinv @ self.F, assume_a='pos'
    ) @ FtDinv                                       # (r, d)
```

`_beta` is computed once per E-step pass (not per observation) and never
materialises `(F Fᵀ + D)^{-1}` densely.

### 7.4 E-step

For each observation `x_j`, using `_quad_form` and posterior GIG params
from `factor_analysis.rst` §Conditional Expectations:

1. `a_post = a + γᵀ Σ^{-1} γ`,
  `b_post = b + (x_j-μ)ᵀ Σ^{-1} (x_j-μ)`,
   `p_post = p - d/2`.
2. Compute `E[Y^{-1}|x_j]`, `E[Y|x_j]`, `E[log Y|x_j]` from the GIG
  posterior — same code as the standard family.
3. Aggregate `s_1, …, s_6` over the batch.
4. Compute `β` once via `_beta()`.
5. Compute `s_7, s_8, s_9, s_10` from `s_1..s_6`, current `(μ, γ, F)`,
  and `β` via the deterministic formulas in `factor_analysis.rst`
   §E-Step. These are O(d·r) or O(d²·r) reductions over the batch already
   reduced.
6. Return a `FactorMixtureStats`.

### 7.5 M-step

Closed-form updates from `factor_analysis.rst` §M-Step:

1. Auxiliaries `q_1..q_5` from `s_5, s_7, s_8, s_9, s_10`.
2. `μ, γ` from the `q_i` (one `2 × 2` linear solve per output coordinate).
3. `F = (s_7 - μ s_8ᵀ - γ s_9ᵀ) s_10^{-1}`.
4. `D = diag(...)` from the long expression in `factor_analysis.rst`,
  then floored for positivity.
5. Subordinator update: same `m_step_subordinator(eta_first_three)` path
  as the standard family.

`F`'s rotational gauge is left unfixed; convergence is measured on
`Σ = F Fᵀ + D` (§5).

### 7.6 Marginal density

`log_prob(x)` reuses each subordinator's existing closed-form GH-family
density formula, substituting `Σ = F Fᵀ + D`. Solves and log-determinant
go through `_solve` and `_log_det_sigma`, so the density evaluation never
forms a dense inverse.

---

## 8. Why No `DispersionModel` Yet

The previous draft proposed introducing a `DispersionModel` ABC with
`FullDispersion(L_Sigma)` and `FactorDispersion(F, D)`. We are deferring
this for now:

- only two storage variants exist today (Cholesky for full, `(F, D)` for
factor);
- pulling out an interface for two implementations adds indirection
without observable benefit;
- the abstraction would have to be designed without knowing the third
variant's needs (SVD? banded? low-rank-plus-diagonal?), which is the
worst time to lock down an interface.

When at least three storage variants are needed (e.g. a research branch
adds SVD-based full covariance for ill-conditioned data), introduce
`DispersionModel` then with the actual API surface those variants
require. A reasonable starting sketch:

```python
class DispersionModel(eqx.Module):
    def solve(self, x: jax.Array) -> jax.Array: ...
    def solve_matrix(self, X: jax.Array) -> jax.Array: ...
    def quad_form(self, x: jax.Array) -> jax.Array: ...
    def log_det(self) -> jax.Array: ...
    def sigma(self) -> jax.Array: ...           # dense, slow path
    def sample_noise(self, key, shape=()) -> jax.Array: ...
```

`CholeskyDispersion(L_Sigma)`, `SVDDispersion(U, S)`, and
`FactorDispersion(F, D)` would all satisfy this contract. Until that
work is actually scheduled, `JointNormalMixture._quad_forms` and
`FactorNormalMixture._solve` (etc.) carry the linear algebra inline.

The interface lives only in this section so that future work has a
starting point, not as committed code.

---

## 9. Implementation Plan

### Phase 1 — EM API generalisation (no new model)

1. Reorder and rename `NormalMixtureEta` fields to `(E_inv_Y, E_Y,
  E_log_Y, E_X, E_X_inv_Y, E_XXT_inv_Y)`.
2. Generalise `affine_combine` to accept scalar or pytree weights.
3. Add `MarginalMixture` ABC; make `NormalMixture` inherit from it.
4. Add `em_convergence_params()` to `NormalMixture` returning
  `(mu, gamma, L_Sigma)`.
5. Replace `_param_change` with the pytree-based version; have the fitter
  call `model.em_convergence_params()` for old/new state.
6. All existing tests (full-covariance EM, MCECM, batch + incremental,
  shrinkage) must pass unchanged after this phase.

### Phase 2 — Shrinkage combinator

1. Implement `Shrinkage(base, eta0, tau)` in `eta_rules.py`.
2. Remove `ShrinkageUpdate`; users construct
  `Shrinkage(IdentityUpdate(), eta0, tau)` instead.
3. Add `normix/fitting/shrinkage_targets.py` with the four target builders
  from §4.4.
4. Tests:
  - `Shrinkage(base, eta0, tau=0)` ≡ `base` for any base rule;
  - `Shrinkage(IdentityUpdate(), eta0, scalar_tau)` matches the previous
  `ShrinkageUpdate` numerically;
  - per-field `tau` with only `E_XXT_inv_Y` non-zero leaves the
  subordinator updates untouched;
  - `Shrinkage(RobbinsMonroUpdate(...), …)` retains the running mean
  (regression test for the bug noted in §2).

### Phase 3 — Penalised EM design doc

1. Create `docs/design/penalised_em.md` covering:
  - the η-affine derivation from `docs/theory/shrinkage.rst`;
  - the per-field `tau` convention;
  - the combinator API and target builders;
  - guidance on choosing `tau` and `Σ_0`.
2. Cross-link from `docs/ARCHITECTURE.md` and from this doc.

### Phase 4 — Factor analysis model family

1. New file `normix/mixtures/factor.py`:
  - `FactorMixtureStats` pytree;
  - `FactorNormalMixture(MarginalMixture)` abstract base with `_M`,
  `_solve`, `_quad_form`, `_log_det_sigma`, `_beta`,
  `e_step`, `m_step`, `m_step_normal`, `m_step_subordinator`,
  `compute_eta_from_model`, `em_convergence_params`;
  - four concrete subclasses (`FactorVarianceGamma`,
  `FactorNormalInverseGamma`, `FactorNormalInverseGaussian`,
  `FactorGeneralizedHyperbolic`) sharing posterior-GIG code with the
  existing joints.
2. Reuse `BatchEMFitter` and `IncrementalEMFitter` unchanged.
3. Tests:
  - synthetic data with known `(F, D)` recovers them up to the
   orthogonal rotation gauge;
  - with `r = d - 1` and large `n`, `Σ = F Fᵀ + D` matches the
  full-covariance fit within tolerance;
  - convergence based on `Σ` succeeds even when `F` rotates between
  iterations;
  - shrinkage combinator works on the factor stats type as well
  (per-field `tau` over the first six fields).

### Phase 5 — Documentation

1. Update `docs/ARCHITECTURE.md`:
  - add `mixtures/factor.py` and `fitting/shrinkage_targets.py` to the
   module tree;
  - update the sufficient-statistics description to reference theory
  order;
  - add `MarginalMixture` to the mixture hierarchy diagram.
2. Update `docs/design/design.md` decision table:
  - shrinkage combinator (§4.2);
  - per-field affine combination (§4.1);
  - convergence hook generalisation (§5);
  - factor analysis as a sibling family (§7);
  - deferral of `DispersionModel` (§8).
3. Add a row pointing to `docs/design/penalised_em.md`.

### Out of scope / explicitly deferred

- `DispersionModel` abstraction — see §8.
- SVD-based full covariance — same.
- Tortora-style alternating CM steps for FA M-step — closed-form is
enough for the paper-baseline; ECM can be added later if needed.
- Identifiability gauge fixing for `F` — convergence on `Σ` removes the
practical need.

