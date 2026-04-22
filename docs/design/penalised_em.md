## Penalised EM with Flexible Shrinkage

**Date:** 2026-04-19
**Status:** Implemented (Phase 1 + Phase 2 of the EM-extension plan)
**Scope:** `normix/fitting/eta.py`, `normix/fitting/eta_rules.py`,
`normix/fitting/shrinkage_targets.py`, `normix/fitting/em.py`
**Theory:** `docs/theory/shrinkage.rst`
**Parent design:** `docs/design/em_covariance_extensions.md` §4

---

## 1. Why Penalised EM

### 1.1 Numerical motivation

Setting $|\Sigma|=1$ (the regularisation already shipped by
`BatchEMFitter`) keeps $\Sigma$ invertible at every iteration but does not
control its *condition number*. When the sample size $n$ is comparable to
the dimension $d$, the M-step formula

$$
\Sigma_{k+1} = E[XX^\top/Y] - E[X/Y]\mu^\top - \mu E[X/Y]^\top
             + E[1/Y]\mu\mu^\top - E[Y]\gamma\gamma^\top
$$

inherits the same ill-conditioning as the sample covariance. Ill-conditioned
$\Sigma$ ruins downstream Cholesky solves, log-density evaluation, and the
GIG posterior in the next E-step (`docs/theory/shrinkage.rst` §1).

### 1.2 Statistical motivation

Equivalently, penalised EM can be viewed as Bayesian regularisation: each
sufficient statistic is pulled toward a prior expectation $\eta_0$, which
on the natural-parameter side maps to maximising

$$
\frac{1}{n}\sum_{j=1}^n \log f(x_j\mid\theta) - \tau\,D_{KL}(\theta_0\,\|\,\theta).
$$

Both motivations point to the *same* algorithmic recipe — convex
combinations of empirical and prior expectation parameters — so we
implement them together in a single combinator.

---

## 2. Theory Recap (η-Affine Derivation)

For an exponential family with log-partition $\psi$, the penalised
M-step solves

$$
\theta_{k+1} = \arg\max_\theta\,\theta^\top\!
\Bigl(\tfrac{1}{n}\sum_j E[t(X,Y)\mid x_j,\theta_k] + \tau\,\eta_0\Bigr)
- (1+\tau)\psi(\theta).
$$

Dividing the linear term by $1+\tau$ shows that the M-step coincides with
ordinary maximum likelihood on a **shrunk expectation**

$$
\hat\eta_k^{\,\text{shrunk}} =
\frac{1}{1+\tau}\,\hat\eta_k + \frac{\tau}{1+\tau}\,\eta_0,
\qquad
\hat\eta_k = \tfrac{1}{n}\sum_j E[t(X,Y)\mid x_j,\theta_k].
$$

This is an η-affine update. It generalises in two directions used by
Phase 2:

1. **Per-field $\tau$.** Replacing the scalar $\tau$ by a pytree
   $\tau\in\eta$-shape applies a different shrinkage weight to each
   sufficient statistic. With $\tau$ non-zero only on the
   $E[XX^\top/Y]$ block, only $\Sigma_{k+1}$ is shrunk — the
   subordinator updates and the $\mu,\gamma$-coupling updates are
   untouched.
2. **Composition with running rules.** Replacing $\hat\eta_k$ by
   $\mathrm{base}(\eta_{k-1},\hat\eta_k)$ — for any incremental rule
   such as Robbins–Monro, EWMA, or sample-weighted averaging — keeps the
   shrinkage step intact while still benefiting from the running
   estimator. The combinator does this in one place; users no longer
   have to derive a `ShrunkRobbinsMonro`-style rule by hand.

The justification for both extensions is mechanical: each leaf of $\eta$
is updated by an independent affine map of the corresponding leaf of
$\hat\eta$ and $\eta_0$, and `affine_combine` already handles
`tree.map`-broadcast weights (see
`docs/design/em_covariance_extensions.md` §4.2).

---

## 3. API

### 3.1 The `Shrinkage` combinator

```python
from normix.fitting.eta_rules import Shrinkage, IdentityUpdate

rule = Shrinkage(IdentityUpdate(), eta0, tau=0.5)
```

`Shrinkage(base, eta0, tau)` is an `EtaUpdateRule`. It computes

$$
\eta_t \;=\; \frac{\tau}{1+\tau}\odot\eta_0
        \;+\; \frac{1}{1+\tau}\odot\mathrm{base}(\eta_{t-1},\hat\eta),
$$

where $\odot$ is per-field multiplication (broadcast for scalar $\tau$).
The combinator inherits `base.initial_state()` and threads the base
rule's state through every call, so composing shrinkage with a stateful
rule (`SampleWeightedUpdate`'s `cumulative_n`, a future
`AdamPredictor`'s moment buffers, …) does not lose memory.

`Shrinkage` is **not** an `AffineRule`: even though its action on $\eta$
is affine, it can wrap an arbitrary `EtaUpdateRule` (including
non-affine predictors), so it lives one level above `AffineRule` in the
class hierarchy.

### 3.2 Per-field $\tau$

`tau` accepts two forms:

| Form                       | Effect                                       |
|----------------------------|----------------------------------------------|
| scalar / 0-d `jax.Array`   | uniform shrinkage on every sufficient stat   |
| `NormalMixtureEta` pytree  | per-field shrinkage; each leaf weights one   |
|                            | sufficient statistic independently           |

Two recurring per-field patterns:

- **Shrink $\Sigma$ alone.** Set the `E_XXT_inv_Y` leaf of `tau` to a
  positive scalar (broadcasts to a $(d,d)$ matrix), all other leaves to
  $0$. The subordinator $(p,a,b)$ and the $\mu,\gamma$ coupling are
  estimated by ordinary EM; only the dispersion is regularised.
- **Anisotropic shrinkage.** Use a pytree-shape `tau` whose leaves are
  themselves arrays — e.g. weighting some directions of `E_XXT_inv_Y`
  more strongly than others. This is the building block for
  Ledoit–Wolf-style targets and for "shrink only the off-diagonal of
  $\Sigma$".

The pytree contract is checked structurally (`isinstance(tau,
NormalMixtureEta)`), so missing leaves are not silently zero-filled.

### 3.3 Target builders

`normix/fitting/shrinkage_targets.py` provides four constructors. Each
returns a complete six-field `NormalMixtureEta`, even when the user
only intends to shrink one statistic — keeping the public contract
"$\eta_0$ is always a full prior" simple.

```python
from normix.fitting.shrinkage_targets import (
    eta0_from_model,    # prior = current model's η
    eta0_isotropic,     # Σ_0 = σ² I_d
    eta0_diagonal,      # Σ_0 = diag(diag)
    eta0_with_sigma,    # Σ_0 = arbitrary user-supplied PSD matrix
)
```

All four read $(\mu,\gamma,p,a,b)$ from a `NormalMixture` and rebuild
the six expectation fields via the closed-form expressions from
`docs/theory/shrinkage.rst` (Shrunk Sufficient Statistics, last
display). The dispersion-substitution variants (`isotropic`,
`diagonal`, `with_sigma`) override only the $E[XX^\top/Y]$ block.

### 3.4 Wire-up with the fitters

`Shrinkage` slots into both fitters via the existing `eta_update`
hook — no extra plumbing:

```python
from normix.fitting import (
    BatchEMFitter, IncrementalEMFitter,
    Shrinkage, IdentityUpdate, RobbinsMonroUpdate,
    eta0_isotropic,
)

eta0 = eta0_isotropic(model, sigma2=jnp.var(X))

# Batch EM with uniform shrinkage on all six statistics:
BatchEMFitter(
    eta_update=Shrinkage(IdentityUpdate(), eta0, tau=0.5),
).fit(model, X)

# Incremental EM, Robbins–Monro running mean composed with shrinkage:
IncrementalEMFitter(
    eta_update=Shrinkage(RobbinsMonroUpdate(tau0=10.0), eta0, tau=0.1),
    batch_size=200, max_steps=100,
).fit(model, X, key=key)
```

When `eta_update` is set, `BatchEMFitter` switches off its `lax.scan`
fast path and uses the Python-loop path. This is intentional: the
combinator wraps an arbitrary base rule, so we cannot make blanket
JIT-friendliness assumptions at the fitter level. (A future
optimisation could detect the all-affine case and re-enable `scan`; not
in scope for Phase 2.)

### 3.3 Inspecting a prior

A shrinkage target $\eta_0$ can be turned back into a concrete model via
the η→model classmethod
:meth:`~normix.mixtures.marginal.NormalMixture.from_expectation`. This
is the cleanest way to verify what an `eta0_*` builder actually produces
(see also `docs/design/em_covariance_extensions.md` §6.3):

```python
prior_model = type(model).from_expectation(eta0_isotropic(model, sigma2))

prior_model.sigma()           # → sigma2 * I_d   (when γ = 0)
prior_model.mu                # → model.mu       (forwarded from the joint)
prior_model.alpha             # → fit Gamma shape (subordinator forwarder)
```

Any subordinator parameters returned by the inversion come from each
subordinator's own `from_expectation` (closed-form for VG, NInvG, NIG;
numerical for GH).

---

## 4. Choosing $\tau$ and $\Sigma_0$

These guidelines summarise practical experience and the relevant
sections of `docs/theory/shrinkage.rst`. Concrete numerical demos live
in `notebooks/em_shrinkage_demo.py` (marimo).

### 4.1 Choosing $\tau$

The scalar $\tau$ has the interpretation "the prior contributes the
same weight as $n_\text{prior}=\tau\,n$ pseudo-observations". Practical
ranges:

| Regime              | Suggested $\tau$           | Notes                                    |
|---------------------|----------------------------|------------------------------------------|
| $n\gg d$            | $0$ — $0.05$               | Shrinkage rarely helps; data dominates   |
| $n\approx d$        | $0.1$ — $1$                | Sample $\Sigma$ ill-conditioned          |
| $n<d$ (e.g. high-d) | $\geq 1$                   | Required for invertibility of $\Sigma$   |
| Online streaming    | match base rule's horizon  | E.g. $\tau\sim 1/T$ for warm-up windows  |

Cross-validation on a held-out $\log f(x_\text{test})$ is the
non-parametric default. The combinator makes this cheap because $\tau$
is a single `jax.Array` leaf — running fits across a $\tau$-grid maps
to `jax.vmap` over the rule's leaves.

For **per-field** $\tau$ (e.g. shrink $\Sigma$ only), the same scalar
guidance applies to the non-zero leaf; the zero leaves leave the
corresponding sufficient statistic untouched.

### 4.2 Choosing $\Sigma_0$

In rough order of regularisation strength:

1. **Isotropic** $\Sigma_0=\sigma^2 I$ — strongest shrinkage; collapses
   all cross-correlation. Useful for high-dimensional warm-starts and
   when the user has no view on covariance structure. `eta0_isotropic`.
2. **Diagonal** $\Sigma_0=\mathrm{diag}(s^2)$ — preserves marginal
   variances, kills cross-asset correlation. Default choice for
   financial returns where univariate variance estimates are reliable
   but pairwise correlations are noisy. `eta0_diagonal`.
3. **Block / factor structure** — provided as raw $\Sigma_0$ via
   `eta0_with_sigma`. Suitable when sector or factor exposures give a
   structured prior; effectively a Ledoit–Wolf-style target.
4. **Self-prior** — `eta0_from_model(model)` shrinks toward the current
   model's own covariance. This is what an L2 trust-region looks like
   in η-space; useful for stabilising incremental EM when batches are
   small.

A useful sanity check: if `eta0` and the data give nearly identical
$\hat\eta$, then shrinkage degenerates to a no-op for any $\tau$ —
consistent with the formula $\eta_t=\eta_0=\hat\eta$.

### 4.3 What about the subordinator?

By default, scalar $\tau$ shrinks the GIG sufficient statistics
$E[\log Y], E[1/Y], E[Y]$ as well. For most subordinators this has the
same direction as ridge regularisation on $(p,a,b)$ but the mapping is
non-linear, so a moderate scalar $\tau$ is benign and large
$\tau\gg 1$ can slow GIG convergence. If GIG instability is a concern,
use a per-field $\tau$ that zeroes out the first three leaves.

---

## 5. Worked Example: Σ-Only Shrinkage

```python
import jax.numpy as jnp
from normix import VarianceGamma
from normix.fitting import (
    BatchEMFitter, Shrinkage, IdentityUpdate, eta0_diagonal,
)
from normix.fitting.eta import NormalMixtureEta

X = ...                           # (n, d), e.g. n=200, d=50
model = VarianceGamma.default_init(X)

# Diagonal prior built from per-asset sample variance.
diag_var = jnp.var(X, axis=0)
eta0 = eta0_diagonal(model, diag=diag_var)

# Per-field τ: shrink only Σ.
d = X.shape[1]
tau = NormalMixtureEta(
    E_inv_Y=jnp.float64(0.0),
    E_Y=jnp.float64(0.0),
    E_log_Y=jnp.float64(0.0),
    E_X=jnp.zeros(d),
    E_X_inv_Y=jnp.zeros(d),
    E_XXT_inv_Y=jnp.full((d, d), 0.5),
)

result = BatchEMFitter(
    eta_update=Shrinkage(IdentityUpdate(), eta0, tau=tau),
    max_iter=200, tol=1e-3,
).fit(model, X)
```

Expected effects vs unshrunk EM at $n\sim d$:

- $\Sigma$ has lower condition number (`jnp.linalg.cond(Σ)`).
- Test log-likelihood improves.
- Per-iteration $\Delta\Sigma$ is smaller, so the fitter converges in
  fewer iterations.
- $\mu,\gamma$ and the GIG parameters are unchanged compared to a fit
  at the same data with $\tau=0$ (modulo coupling through the
  posterior).

---

## 6. Cross-References

- Theory: `docs/theory/shrinkage.rst`
- Phase plan / decision log: `docs/design/em_covariance_extensions.md`
  §4 (rule abstraction, generalised `affine_combine`, combinator) and
  §6 (`MarginalMixture` ABC).
- API surface: `docs/design/design.md` § "EM Framework: Model/Fitter
  Separation".
- Companion demo: `notebooks/em_shrinkage_demo.py` (marimo).
- Tests: `tests/test_incremental_em.py` covers the four guarantees
  required by the design plan
  (`Shrinkage(base, eta0, tau=0)≡base`, scalar-$\tau$ closed form,
  per-field $\tau$ leaving non-target fields untouched, state
  threading through running base rules).

---

## 7. Out of Scope

- A `ShrinkageBatchEMFitter` convenience subclass — `eta_update`
  already exposes this.
- `lax.scan` JIT path for `Shrinkage`-wrapped rules (see §3.4).
- Adaptive / data-driven $\tau$ schedules. These are easy to write as
  `EtaUpdateRule` subclasses (e.g. a wrapper that re-evaluates $\tau$
  per iteration from a held-out LL); deferred until a use case appears.
- Penalisation in **natural-parameter** space (KL with $\theta$ as the
  argument rather than $\eta$). Mathematically equivalent for
  exponential families but algorithmically different; a separate design
  if a user requests it.
