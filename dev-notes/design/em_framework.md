# EM Framework

> **Scope.** Why model and fitter are separate, what the η-update rule
> abstraction buys, how the `Shrinkage` combinator generalises penalised
> EM, and what the four covariance-regularisation modes do.
>
> **Where things live.** Public API is in `../ARCHITECTURE.md` § *EM
> Algorithm*. Parameter facade and `from_expectation` dispatch are in
> `mixtures.md`. Solver internals are in
> `solvers_and_bessel.md`.

---

## 1. Model Knows Math, Fitter Knows Iteration

Following the GMMX style: distributions implement E/M-step math and
return immutable models; fitters implement iteration, convergence, and
diagnostics. Fitters are **plain Python classes**, not `eqx.Module`
(they carry no JAX state — moving them onto pytrees would buy nothing
and make their attributes traceable).

```python
class BatchEMFitter:           # full-dataset EM with convergence monitoring
class IncrementalEMFitter:     # mini-batch / online with fixed budget
```

Both return `EMResult`:

| Field | Always set | Notes |
|---|---|---|
| `model` | yes | the fitted pytree |
| `param_changes` | yes | hybrid-RMS diagnostic on `em_convergence_params()` (not the stop) |
| `n_iter` | yes | iterations actually run |
| `converged` | yes (Batch) / `None` (Incremental) | Aitken remaining gap $<$ `tol` for batch; incremental is fixed-budget |
| `log_likelihoods` | optional (`track_ll` / verbose ≥ 1) | conjugacy mean $\ell_n$ per iteration (the Aitken sequence) |
| `elapsed_time` | yes | wall clock |

### 1.1 One `fit()` signature: `_fit_defaults()` (DEC-3)

> Decision row: `design.md` § *2026-07 review Phase 0*, DEC-3. Decided
> 2026-07-20; implementation lands with roadmap item D3 (Phase 6).

The VG/NInvG/GH `fit()` overrides existed only to flip default backends
(and, for GH, the regularization), re-listed every parameter, and
*narrowed* the API — `track_ll`, `eta_update`, and `m_step_kwargs` were
unreachable without dropping to `BatchEMFitter`. DEC-3 replaces them
with a per-family defaults classmethod consumed by one base signature:

```python
class MarginalMixture(eqx.Module):
    @classmethod
    def _fit_defaults(cls) -> dict:
        """Family-specific BatchEMFitter defaults, merged UNDER user kwargs."""
        return {}

    def fit(self, X, *, alpha_min=None, **fitter_kwargs) -> "EMResult":
        # kwargs = {**cls._fit_defaults(), **fitter_kwargs}
        # alpha_min resolved via self.d into m_step_kwargs (VG semantics),
        # then BatchEMFitter(**kwargs).fit(self, X)
        ...
```

- `VarianceGamma`/`NormalInverseGamma`: `{'e_step_backend': 'cpu'}`
  (CPU E-step is faster for the degenerate-GIG posterior).
- `GeneralizedHyperbolic`:
  `{'e_step_backend': 'cpu', 'regularization': 'det_sigma_one'}`
  (scale non-identifiability).
- Everything not in a family's `_fit_defaults` falls through to
  `BatchEMFitter.__init__` defaults — the **single source of default
  values**; the base `fit` stops re-listing them.
- `alpha_min` stays an explicit keyword: it needs `self.d` to resolve
  the `'density'`/`'inverse_moment'` sentinels, and it merges into any
  user-supplied `m_step_kwargs` (the explicit argument wins).

Alternative rejected: keeping per-family `fit` overrides but widening
each signature — three copies of a growing parameter list is exactly
the drift that produced the narrowing bug.

---

## 2. EM Steps on Marginal Mixtures

```python
e_step(X, *, backend='jax'|'cpu')        -> EStepResult  # eta + conjugacy ℓ_n
m_step(eta, **kw)                        -> MarginalMixture       # full update
m_step_normal(eta)                       -> MarginalMixture       # MCECM cycle 1
m_step_subordinator(eta, **kw)           -> MarginalMixture       # MCECM cycle 2
compute_eta_from_model()                 -> stats pytree          # incremental warm-start
em_convergence_params()                  -> pytree                # hybrid-RMS diagnostic
```

`e_step` returns `EStepResult`: aggregated expectation parameters
(`eta`) plus the conjugacy mean log-likelihood $\ell_n(\theta)$ at the
E-step parameters. $\ell_n$ is *not* a field of `NormalMixtureEta` —
`affine_combine` / shrinkage would mix it — and is not a cache on the
model. The first six fields of `FactorMixtureStats` are identical to
`NormalMixtureEta`, so shrinkage targets and rule weights port across
the two families (see `mixtures.md` §7).
`marginal_log_likelihood` stays the public standalone method.

### 2.1 Stopping: Aitken remaining gap of $\ell_n$

E3. `converged` is the Aitken remaining gap of the mean log-likelihood
$\ell_n$ in nats per observation (McLachlan & Krishnan 2008).
With three consecutive values $\ell_{t-1},\ell_t,\ell_{t+1}$,

$$
\Delta_1 = \ell_t - \ell_{t-1},\qquad
\Delta_2 = \ell_{t+1} - \ell_t,\qquad
a = \Delta_2/\Delta_1,
$$

stop when $\ell_\infty - \ell_{t+1} = a\Delta_2/(1-a) < \mathtt{tol}$
for $\Delta_1 > 0$ and $0 \le a < 1$. Earliest stop is `n_iter=3`.
Default $\mathtt{tol}=10^{-3}$ is a remaining gap of $10^{-3}$ nats/obs.
A negative $\Delta\ell$ under plain EM is a bug detector (Aitken is
not a stop); under scalar-$\tau$ MAP shrinkage the tracked sequence is
the penalised objective $\ell_n + \tau(\theta\cdot\eta_0 - \psi)$.

$\ell_n(\theta_t)$ comes from the E-step at $\theta_t$. Posterior GIG
conjugacy already evaluates $\psi_{\mathrm{post}}$ from
`BesselMoments.log_k`:

$$
\log f(x) = \log h(x) + \psi_{\mathrm{post}} - \psi,
\qquad
\log h(x) = -\tfrac{d}{2}\log(2\pi) - \tfrac12\log|\Sigma|
+ \gamma^\top\Sigma^{-1}(x-\mu).
$$

$\psi$ is the subordinator log-partition in the GIG convention
(`subordinator().to_gig().log_partition()`). InverseGaussian stores
$\tfrac12\log(2\pi)$ in $h$, so its own $\psi$ is the wrong gauge for
this identity. Gamma / InverseGamma / GIG already match.

Zero extra *per-observation* Bessel calls relative to the E-step (the
prior $\psi$ is a scalar). Same $\ell_n$ for every
regularisation (`none` / `det_sigma_one` / `det_sigma_x` / `a_eq_b`).
MCECM uses $\ell_n$ from the first E-step of the cycle.
`EMResult.log_likelihoods` is that E-step sequence of length `n_iter`
(the returned model is one M-step ahead).

The previous stop — hybrid-RMS on `em_convergence_params()` =
$(\mu,\gamma,L_\Sigma)$, subordinator excluded — reported
`converged=True` while the mixer was still travelling. The 2026-09-05
review VG experiment (2000 draws, $Y_i\sim\mathrm{Gamma}(0.7,0.7)$,
init $\alpha=\beta=2$) stopped at $\alpha\approx 1.59$; continuation
recovered the generating shape $0.7$. The additive $1$ in the RMS
denominator also stopped $X\mapsto 0.001\,X$ after one iteration.

Parameter change became the stop in `e516372`.
[PR #75](https://github.com/xshi19/normix/pull/75) made hybrid-RMS the stop and left the log-likelihood as a diagnostic.
[PR #117](https://github.com/xshi19/normix/pull/117) superseded both with the Aitken gap above.

Public twin: `../../docs/design/em_framework.md` § 2.1.

### 2.2 Hybrid-RMS diagnostic

`em_convergence_params()` still returns a pytree whose leaf-wise
hybrid-RMS change is stored in `EMResult.param_changes`:

| Marginal | Returns |
|---|---|
| `NormalMixture` | $(\mu, \gamma, L_\Sigma)$ |
| `FactorNormalMixture` | $(\mu, \gamma, \Sigma = F F^\top + \mathrm{diag}(D))$ |

Returning $\Sigma$ from `FactorNormalMixture` (rather than $(F, D)$)
sidesteps the $r \times r$ orthogonal gauge of $F$. `_param_change`
is

$$
\max_{\text{leaves}}\;
\frac{\mathrm{rms}(\Delta)}{1 + \mathrm{rms}(\theta)},
\qquad
\mathrm{rms}(v) = \|v\|_2 / \sqrt{m},\quad m = |v|.
$$

The additive $1$ avoids near-zero-$\mu$ inflation from a pure relative
$\|\Delta\|/\|\theta\|$. RMS keeps a fixed diagnostic roughly
dimension-free. It is not `converged`.

### GIG statistic conversion

Option C (accepted, 2026-10-29; session authorization) preserves the GIG array
order `(E_log_Y, E_inv_Y, E_Y)` and mixture field order
`(E_inv_Y, E_Y, E_log_Y)`. Private `_pack_gig_eta` and `_unpack_gig_eta` in
`fitting/eta.py` own the conversion. They stack or select the last axis and
support scalar and batched moments.
Reordering either public layout was rejected because it changes positional
consumers. Reopen only for an approved coordinate migration.
Independent mapping tests and the named MAP-penalty regression pin both orders.

---

## 3. η-Update Rules: Two Layers

Online and shrinkage updates compose. The current abstraction has two
layers so future ML-style predictors can plug in without an API revision.

```python
class EtaUpdateRule(eqx.Module):
    """Most general: η_t = rule(η_{t-1}, η̂)."""
    def initial_state(self) -> dict: ...
    def __call__(self, eta_prev, eta_new, step, batch_size, state)
        -> tuple[EtaLike, dict]: ...

class AffineRule(EtaUpdateRule):
    """Specialisation: η_t = a + b·η_{t-1} + c·η̂."""
    def weights(self, step, batch_size, state)
        -> tuple[Optional[EtaLike], Weight, Weight, dict]: ...
```

`__call__` (rather than `predict` / `apply`) is the Equinox idiom:
a module **is** its forward pass, like `eqx.nn.Linear(x)`.

| Rule (concrete) | Layer | Stored | $(a, b, c)$ |
|---|---|---|---|
| `IdentityUpdate` | affine | — | $(0, 0, 1)$ |
| `EWMAUpdate` | affine | `w` | $(0, 1-w, w)$ |
| `RobbinsMonroUpdate` | affine | `tau0` | $c = 1/(\tau_0 + t)$ |
| `SampleWeightedUpdate` | affine | (state-only `cumulative_n`) | $n/(n+m), m/(n+m)$ |
| `AffineUpdate` | affine | `a, b, c` directly | as-is |
| `Shrinkage` | non-affine combinator | `base, eta0, tau` | derived (see §4) |

### 3.1 Generalised `affine_combine`

`affine_combine(eta_prev, eta_new, b, c, a=None)` accepts three weight
forms via `jax.tree.map`:

| Form | Math | Use case |
|---|---|---|
| Scalar (or 0-d `jax.Array`) | $b\cdot I_n$ broadcast to every leaf | EWMA, Robbins–Monro, uniform shrinkage |
| Stats-shape pytree | block-diagonal: leaf-wise multiply | per-field / per-element shrinkage |
| Callable `η → η` | arbitrary linear operator | user-supplied; e.g. wrap `eqx.nn.Linear` |

`eta_prev` is rebased to `eta_new.origin` first, and the result keeps that
origin. Weight pytrees and the shift `a` are not coordinates: their `origin`
leaves are ignored, so a six-field weight such as a Σ-only `tau` works
unchanged. `a` is added in `eta_new`'s coordinates.

We do **not** ship a flat $n \times n$ matrix form. That would force
the public API to commit to a flatten order across $(s_1,\dots,s_6)$,
leaking the contract into every user script. Users who want a true
$n \times n$ linear operator wrap an `eqx.nn.Linear` inside a custom
rule and use `jax.flatten_util.ravel_pytree` locally.

### 3.2 State and trainability

`eqx.Module` is immutable. Anything that genuinely changes per iteration
(cumulative sample count, RNN hidden state, momentum buffers) lives in
`state`, threaded by the fitter:

```python
state = rule.initial_state()
for step in range(...):
    eta_t, state = rule(eta_prev, eta_new, step, batch_size, state)
```

Pure rules round-trip an empty `state = {}` at zero cost. Every JAX-array
leaf of an `eqx.Module` is automatically a parameter visible to
`jax.grad` / `optax`. The architecture leaves the door open for
meta-learning step-size schedules end-to-end (the `lax.scan` fast path
satisfies the no-Python-branches requirement); meta-learning is **not**
a goal of any current implementation phase.

---

## 4. Penalised EM via the `Shrinkage` Combinator

### 4.1 Theory recap (η-affine derivation)

For an exponential family with log-partition $\psi$, the penalised
M-step solves

$$
\theta_{k+1} = \arg\max_\theta\,\theta^\top\Bigl(\hat\eta_k + \tau\,\eta_0\Bigr) - (1+\tau)\psi(\theta)
$$

which is identical to ordinary MLE on the **shrunk expectation**

$$
\hat\eta_k^{\,\text{shrunk}} = \tfrac{1}{1+\tau}\hat\eta_k + \tfrac{\tau}{1+\tau}\eta_0.
$$

This generalises in two directions:

1. **Per-field $\tau$.** Replace the scalar by a pytree
   $\tau \in \eta$-shape. Setting $\tau$ non-zero only on the
   $E[X X^\top/Y]$ leaf shrinks **only** $\Sigma_{k+1}$.
2. **Composition with running rules.** Replace $\hat\eta_k$ by
   $\mathrm{base}(\eta_{k-1},\hat\eta_k)$ — Robbins–Monro, EWMA,
   sample-weighted, etc. Each leaf of η is updated by an independent
   affine map; `affine_combine` already handles `tree.map`-broadcast
   weights.

### 4.2 The combinator

```python
class Shrinkage(EtaUpdateRule):
    """η_t = (τ/(1+τ)) ⊙ η_0 + (1/(1+τ)) ⊙ base(η_{t-1}, η̂).

    base : EtaUpdateRule    (e.g. IdentityUpdate, RobbinsMonroUpdate)
    eta0 : NormalMixtureEta or FactorMixtureStats   (the prior)
    tau  : scalar | stats-shape pytree
    """
    base: EtaUpdateRule
    eta0: EtaLike
    tau:  Union[jax.Array, EtaLike]

    def initial_state(self):
        return self.base.initial_state()
```

`Shrinkage` lives one level above `AffineRule`: even though its action
on η is affine, its `base` may be an arbitrary `EtaUpdateRule` (including
non-affine predictors).

The combinator detects per-field τ via `type(tau) is type(eta0)`, so the
same code works for both `NormalMixtureEta` and `FactorMixtureStats`.

### 4.3 Why a combinator (not `ShrunkX` copies)

Composing shrinkage with a running rule by hand requires getting the
algebra right twice. The combinator does it in one place. The four
hypothetical `ShrunkRobbinsMonro` / `ShrunkEWMA` / `ShrunkSampleWeighted`
/ `ShrunkIdentity` classes collapse into one `Shrinkage(base, eta0, tau)`
with no behaviour lost.

### 4.4 Usage patterns

| Use case | Construction |
|---|---|
| Batch EM, uniform shrinkage | `Shrinkage(IdentityUpdate(), eta0, tau)` |
| Batch EM, Σ-only shrinkage | `Shrinkage(IdentityUpdate(), eta0, tau_pytree)` |
| Robbins–Monro online + shrinkage | `Shrinkage(RobbinsMonroUpdate(τ0), eta0, tau)` |
| EWMA + shrinkage | `Shrinkage(EWMAUpdate(w), eta0, tau)` |
| Sample-weighted + shrinkage | `Shrinkage(SampleWeightedUpdate(), eta0, tau)` |

The current `ShrinkageUpdate` class was removed (pre-1.0 rename); users
now construct `Shrinkage(IdentityUpdate(), eta0, tau)` explicitly.

When `eta_update` is set, `BatchEMFitter` switches off its `lax.scan`
fast path and uses the Python-loop path. The combinator wraps an
arbitrary base rule so we cannot make blanket JIT-friendliness
assumptions at the fitter level.

### 4.5 Shrinkage targets

`normix/fitting/shrinkage_targets.py` provides four constructors. Each
returns a complete six-field `NormalMixtureEta`, even when the user
intends to shrink only one statistic — keeping the public contract
"η₀ is always a full prior" simple.

| Builder | Effect |
|---|---|
| `eta0_from_model(model)` | prior = current model's η (L2 trust-region in η-space) |
| `eta0_isotropic(model, σ²)` | $\Sigma_0 = \sigma^2 I_d$ |
| `eta0_diagonal(model, diag)` | $\Sigma_0 = \mathrm{diag}(s^2)$ |
| `eta0_with_sigma(model, Σ₀)` | arbitrary user-supplied PSD matrix |

The dispersion-substitution variants reuse the model's
$(\mu, \gamma, p, a, b)$ to fill the other five fields, so the result
is still a coherent prior expectation parameter — the user's per-field
`tau` then decides which fields are actually shrunk. Inverting a target
is `type(model).from_expectation(eta0_isotropic(model, σ²))`.

### 4.6 Choosing $\tau$ and $\Sigma_0$

Practical guidance (the scalar τ has the interpretation "the prior
contributes the same weight as $n_{\text{prior}} = \tau\,n$
pseudo-observations"):

| Regime | Suggested $\tau$ | Notes |
|---|---|---|
| $n \gg d$ | $0$ – $0.05$ | Shrinkage rarely helps |
| $n \approx d$ | $0.1$ – $1$ | Sample $\Sigma$ ill-conditioned |
| $n < d$ | $\geq 1$ | Required for invertibility |
| Online streaming | match base rule's horizon | E.g. $\tau \sim 1/T$ |

Cross-validation on held-out $\log f(x_{\text{test}})$ is the default;
running fits across a τ-grid maps to `jax.vmap` over the rule's leaves.

---

## 5. Covariance Regularisations

After each M-step the fitter optionally rescales the model. The
regularisation family is enumerated by
`BatchEMFitter._REGULARIZATIONS`:

| Mode | What it enforces | Implementation |
|---|---|---|
| `'none'` (default) | identity | — |
| `'det_sigma_one'` | $\|\Sigma\| = 1$ (the original GH convention) | `model.regularize_det_sigma(0.0)` |
| `'det_sigma_x'` | $\log\|\Sigma\| = \log\|\Sigma_0\|$, the **initial** model's log-determinant | `model.regularize_det_sigma(target)`; target captured once in `fitter._target_log_det = model.log_det_sigma()` at the start of `fit` |
| `'a_eq_b'` | $a = b = \sqrt{ab}$ on the GIG subordinator (orbit invariant) | `model.regularize_a_eq_b()`; no-op for VG / NInvG / MVN |

All four are reparametrisations: the joint density is unchanged. They
move the model along the orbit $Y \to s\,Y$, $\Sigma \to \Sigma/s$,
$\gamma \to \gamma/s$, with the subordinator absorbing the scale.

### 5.1 Why three Σ-targeting modes

| Mode | Use when |
|---|---|
| `'none'` | EM should leave the scale alone (e.g. when downstream code reads $\Sigma$ directly) |
| `'det_sigma_one'` | Comparing across distributions where only the orbit matters; classical GH convention |
| `'det_sigma_x'` | Running multiple distributions on the same data and you want their displayed $(a, b, \gamma)$ on a comparable scale (e.g. the SP500 study compares VG / NIG / NInvG / GH side by side; only `det_sigma_x` keeps GH's `(a, b)` aligned with the others) |

### 5.2 Why `'a_eq_b'` matters separately

GH's $(p, a, b, \gamma)$ has a one-parameter orbit $s$ that rescales
$a, b, \gamma$. The most useful **canonical representative** sets
$a = b = \sqrt{ab}$ — the orbit-invariant pair $(a\cdot b)$ becomes
visible directly. NIG, where $a = \lambda/\mu_{IG}^2$ and $b = \lambda$,
has the same orbit; the canonical representative there is $\mu_{IG} = 1$.
VG (Gamma subordinator, $a = 0$) and NInvG (InverseGamma, $b = 0$) are
already on a degenerate orbit — `'a_eq_b'` is a no-op for those.

### 5.3 The `_rescale` / `_build_rescaled` pattern

Each marginal owns the linear-algebra side of the rescale; the
subordinator-side is delegated to a per-subclass `_build_rescaled`:

```python
class NormalMixture(MarginalMixture):
    def _rescale(self, scale):
        # Σ → Σ/s, γ → γ/s, then defer subordinator to subclass
        L_new = self._joint.L_Sigma / jnp.sqrt(scale)
        gamma_new = self._joint.gamma / scale
        return self._build_rescaled(self._joint.mu, gamma_new, L_new, scale)

    def regularize_det_sigma(self, target_log_det=0.0):
        s = jnp.exp((self._joint.log_det_sigma() - target_log_det) / d)
        return self._rescale(s)

    def regularize_det_sigma_one(self):
        return self.regularize_det_sigma(0.0)

    def regularize_a_eq_b(self):
        # default no-op; overridden by GH and NIG
        return self
```

Per-subclass `_build_rescaled` knows how the subordinator absorbs
$Y \to s\,Y$:

| Subordinator | Stored | $Y \to s\,Y$ rule |
|---|---|---|
| Gamma($\alpha,\beta$) (VG) | `alpha, beta` | $\beta \to \beta/s$, $\alpha$ unchanged |
| InverseGamma($\alpha,\beta$) (NInvG) | `alpha, beta` | $\beta \to \beta\cdot s$, $\alpha$ unchanged |
| InverseGaussian($\mu_{IG},\lambda$) (NIG) | `mu_ig, lam` | $\mu_{IG} \to s\mu_{IG}$, $\lambda \to s\lambda$ |
| GIG($p, a, b$) (GH) | `p, a, b` | $a \to a/s$, $b \to b\cdot s$, $p$ unchanged |

`FactorNormalMixture` follows the same pattern: $F \to F/\sqrt{s}$,
$D \to D/s$, plus the same subordinator rules.

The lesson learned during Phase 4: NIG's `_build_rescaled` was originally
`mu_ig/scale, lam/scale`, which is **wrong** — the correct rule is
`mu_ig·scale, lam·scale`. The orbit invariance test in
`tests/test_regularizations.py` would have caught it (and now does).

---

## 6. Loop Dispatch (Batch)

```python
use_scan = (
    algorithm == 'em'
    and e_step_backend == 'jax'
    and m_step_backend == 'jax'
    and verbose <= 1
    and eta_update is None
)
```

Otherwise → Python `for` loop (CPU backends, verbose tables, or any
`eta_update`).

`IncrementalEMFitter` runs `lax.scan` over minibatch steps when both
backends are `'jax'` and `verbose == 0`; `inner_iter > 1` nests
`lax.fori_loop`. RNG keys are pre-stacked (`_materialize_incremental_subkeys`).

---

## 7. Cross-References

- Architecture surface: `../ARCHITECTURE.md` § *EM Algorithm*.
- Solvers (η→θ): `solvers_and_bessel.md`.
- Theory: `../../docs/theory/em_algorithm.md`, `../../docs/theory/shrinkage.md`,
  `../../docs/theory/factor_analysis.md`.
- Companion notebook: `notebooks/em_shrinkage_demo.py`.
- Historical / archived: `../archive/design/em_covariance_extensions.md`,
  `../archive/design/penalised_em.md`.

## Weighted batch EM (E18)

Status: chosen for batch EM/MCECM and the data-fitting convenience APIs.
Decision and recording date: 2026-11-15. Authority: the owner's weighted
batch EM task in this session, session-reported, no durable source.
E3, E9, E10, E16, E17, and S9 remain governing constraints.

Normalize finite nonnegative observation weights to sum to one. Divide by
the maximum before summation so finite large weights cannot overflow the
normalizer. Reject wrong shapes, negative/nonfinite weights, and zero
total weight. Eager calls raise `ValueError`; traced weights retain an
Equinox runtime check. Keep the original reductions when weights are absent.

Use the same weights for every E-step statistic, conjugacy likelihood,
sample center, and initial moment. Factor Z statistics follow from the
weighted first six statistics by the existing affine identities.
GH's candidate warm starts and likelihood ranking must also be weighted.
E16's shrinkage-target translation uses that weighted center. The
2026-11-20 E16 amendment records the statistic origin explicitly.

Pass $n_{\mathrm{eff}}=(\sum_iw_i)^2/\sum_iw_i^2$ as the eta rule's
`batch_size`. The owner explicitly requested this choice. It preserves
weight-scale invariance and measures the concentration of observation
mass. `Shrinkage.tau` is already a relative penalty strength; neither it
nor the scale-gauge regularizations use sample count.

Raw weight sums were not selected as sample counts because they depend on
weight units. Frequency-count corrections to the initial covariance would
break weight-scale invariance. A new weighted-fitter abstraction is unnecessary:
the existing aggregation and fit arguments suffice. Integer replication
is the oracle for likelihood EM. It is not the oracle for count-dependent
eta updates, whose effective sample size differs from the replicated count.

Evidence added 2026-11-15: `tests/test_weighted_em.py` checks replication,
uniform/absent weights, all factor fields, both backends and loops,
MCECM, and EWMA convergence. It also covers initialization, centering,
effective sample size, MLE, invalid inputs, and transformed E-steps.
Zero-weight observations are masked before posterior calculations, so an
ignored extreme observation cannot contaminate a matrix statistic through
zero times infinity. Exact commands and measured
results belong to the task report. External review remains pending.

Reopen if frequency-weight uncertainty estimates need replicated counts,
or if weighted online observations become a separate requested feature.


## Explicit statistic origins (E16)

Status: chosen; supersedes E16's implicit-origin and equality-inference clauses.
Decision, evidence, and recording date: 2026-11-20. Authority: owner's four-fix
audit task, session-reported, no durable source. The owner explicitly reopened
E16 and authorized completion without waiting for external review.

The previous decision protected covariance at a large location but lost the
location itself. With Gamma(3,2) mixing, $\mu=10^8$, $\gamma=0.3$, and
$\Sigma=1$, `compute_eta_from_model` stored $s_4=0.45$ and the M-step returned
$\mu=0$. Commit `be6ef67` had introduced the implicit centered store to prevent
covariance cancellation. That reason still holds; its origin must survive.

`NormalMixtureEta.origin` is a dynamic array, default zero. Its three location
fields are moments of $X-\mathrm{origin}$. `FactorMixtureStats` inherits these
fields and also translates the X–Z cross moment. Producers select the existing
`MOMENT_CANCEL_ATOL` threshold before accumulation; ordinary inputs retain the
original reductions. M-steps add the origin after covariance reconstruction.
`rebase` changes coordinates, and `affine_combine` aligns inputs to the new
estimate's origin. The origin is coordinate metadata, never a weighted moment.
Per-field shrinkage uses the same combination path. A callable weight operates
on statistics in those aligned coordinates.

Fit preprocessing still uses the weighted mean under E18. Translating a prior
first shifts its recorded origin and then rebases to zero. The former model
comparison is removed. Non-self priors now retain their location too. MAP
objectives express the model in the prior's coordinates. Custom-covariance
shrinkage targets use the same origin selection. E13's prior moment policy,
E16's support/MLE rules, and S9's backend placement remain unchanged.

Not selected: raw float64 second moments (lose covariance); an implicit origin
(lose location); equality-based inference (fails for other-model priors).
Reopen if unnormalized statistic measures need their own mass coordinate or if
origins separated by a large distance must be combined without loss of covariance.
Already-rounded raw moments cannot recover their lost digits through `rebase`.

Evidence: `tests/test_robustness.py` covers the complete normal roundtrip,
weighted/unweighted E-steps, both mixture hierarchies, distinct shrinkage priors,
custom covariance, per-field shrinkage, JIT, and vmap. Self-prior conversion now
uses explicit moment translation, so floating-point rounding can differ from the
old recomputation of model moments; measured deltas belong to the task report.

## GH exception scope (E17)

Status: existing contract enforced on 2026-11-20; no new fallback policy.
Authority: owner's audit task, session-reported, no durable source.
Both GH CPU instance updates catch only `SolverDidNotConverge` and `ValueError`.
The constructor path still propagates failures. A broad `Exception` catch had
hidden `TypeError` and `RuntimeError` as successful reuse of the old GIG.
`test_gh_warm_failure_exception_contract` injects all four classes into full and
factor GH and checks either exact retention or propagation of the same exception.
Reopen only if a further numerical exception gets an explicit recovery contract.

## Posterior floor scope (E12)

Status: corrected on 2026-11-20. Authority: owner's audit task,
session-reported, no durable source. The prior-to-posterior conjugacy map is
unchanged. Apply `B_POST_FLOOR` only when the prior's GIG $b=0$ (VG, including
an exact Gamma boundary represented as GIG). Preserve every positive prior $b$.
The old reasoning that positive means above $10^{-6}$ was false: at $x=\mu$,
GH $(p,a,b)=(1,1,10^{-12})$ had its posterior $b$ raised a millionfold.

The CPU batch path also used $\sqrt{\max(ab,\mathrm{tiny})}$. Removing the
posterior floor exposed that alteration at the inverse-gamma boundary:
$a=0$, $b=10^{-12}$, and posterior $p=-2.5$ gave $E[Y]=4.9722\times10^{-11}$
instead of $10^{-12}/3$. For subnormal products, computing
$\sqrt{a}\sqrt{b}$ after the existing input floors preserves the scale without
underflow. Normal products retain their original arithmetic. The Bessel kernel
and its unused derivative outputs are unchanged.

A general floor was not selected: it changes valid GH/NIG/NInvG laws without a
requested approximation contract. Removing the VG floor was not selected: it
would restore the known mode singularity. Tests cover both hierarchies and both
backends against half-order GH and inverse-gamma formulas and SciPy NIG ratios.
Reopen if a positive-scale posterior cannot be evaluated within its stated
accuracy domain; any approximation would need a separate error contract.
