# EM Framework

> **Scope.** Why model and fitter are separate, what the η-update rule
> abstraction buys, how the `Shrinkage` combinator generalises penalised
> EM, and what the four covariance-regularisation modes do.
>
> **Where things live.** Public API is in `docs/ARCHITECTURE.md` § *EM
> Algorithm*. Parameter facade and `from_expectation` dispatch are in
> `docs/design/mixtures.md`. Solver internals are in
> `docs/design/solvers_and_bessel.md`.

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
| `param_changes` | yes | max relative L2 change in `em_convergence_params()` per iteration |
| `n_iter` | yes | iterations actually run |
| `converged` | yes (Batch) / `None` (Incremental) | `bool` for batch; incremental is fixed-budget |
| `stop_reason` | optional | `'tol'`, `'max_iter'`, or `'budget'` |
| `log_likelihoods` | optional (verbose ≥ 1) | per-iteration LL trace |
| `elapsed_time` | yes | wall clock |

---

## 2. EM Steps on Marginal Mixtures

```python
e_step(X, *, backend='jax'|'cpu')        -> NormalMixtureEta | FactorMixtureStats
m_step(eta, **kw)                        -> MarginalMixture       # full update
m_step_normal(eta)                       -> MarginalMixture       # MCECM cycle 1
m_step_subordinator(eta, **kw)           -> MarginalMixture       # MCECM cycle 2
compute_eta_from_model()                 -> stats pytree          # incremental warm-start
em_convergence_params()                  -> pytree                # convergence hook
```

`e_step` returns aggregated expectation parameters as an `eqx.Module`
pytree, not raw per-observation dicts. The first six fields of
`FactorMixtureStats` are identical to `NormalMixtureEta`, so shrinkage
targets and rule weights port across the two families (see
`docs/design/mixtures.md` §7).

### 2.1 Convergence on a pytree

`em_convergence_params()` returns a pytree whose leaf-wise change
defines convergence:

| Marginal | Returns |
|---|---|
| `NormalMixture` | `(μ, γ, L_Σ)` |
| `FactorNormalMixture` | `(μ, γ, Σ = F F^\top + \mathrm{diag}(D))` |

Subordinator parameters $(p, a, b)$ are excluded — their solver has
its own tolerance and including them inflates iteration counts.
Returning $\Sigma$ from `FactorNormalMixture` (rather than $(F, D)$)
sidesteps the $r \times r$ orthogonal gauge of $F$ — the Σ-recovery
test in `tests/test_factor_mixture.py` passes without an
orthogonalisation step.

`_param_change(new, old)` takes max relative L2 change across leaves,
clamping the denominator at `_PARAM_EPS`.

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

- Architecture surface: `docs/ARCHITECTURE.md` § *EM Algorithm*.
- Solvers (η→θ): `docs/design/solvers_and_bessel.md`.
- Theory: `docs/theory/em_algorithm.rst`, `docs/theory/shrinkage.rst`,
  `docs/theory/factor_analysis.rst`.
- Companion notebook: `notebooks/em_shrinkage_demo.py`.
- Historical / archived: `docs/archive/design/em_covariance_extensions.md`,
  `docs/archive/design/penalised_em.md`.
