# normix Design

> **What this file is.** Design philosophy + the canonical
> decision table. Topical rationale lives in the sibling files listed
> in `index.md`.
>
> **What this file is not.** A duplicate of `../ARCHITECTURE.md`.
> The module hierarchy, distribution storage table, and constants
> table are in ARCHITECTURE; design decisions and their *why* are
> here.

---

## Philosophy

Priority: **Elegance > Numerical efficiency & robustness > Mathematical
clarity > Simplicity.**

- **Elegance.** Reading and using normix should be enjoyable. Code,
  mathematics, documentation, and agent instructions are all part of
  this. Think in high-level abstractions — modules, base classes,
  object hierarchies — and optimise for long-term maintainability.
  When a new feature feels like it needs a quick-fix function, ask
  whether the underlying design should be refactored instead.
- **Numerical efficiency & robustness.** normix targets the same
  standard as professional scientific-computing libraries: exploit
  Cholesky structure, `assume_a` flags for triangular / PSD solves,
  log-space arithmetic where magnitudes vary widely, large-$z$
  asymptotics for `log_kv`, etc.
- **Mathematical clarity.** Maintain a clean correspondence between
  symbols and code variables ($\theta \leftrightarrow$ `theta`,
  $\eta \leftrightarrow$ `eta`, $\psi \leftrightarrow$
  `log_partition`). Avoid ad-hoc notation in docstrings or theory
  docs.
- **Simplicity.** All else equal, simpler is better. Removing
  something for equal-or-better results is a simplification win.
  But simplicity must not sacrifice the higher-priority concerns.
  Routing a closed-form special case (NIG, VG, NInvG) through a
  general-case implementation (GH) is **not** simpler — it is
  mathematically more complex and numerically wasteful.

`eqx.Module` is a frozen dataclass that is automatically a JAX pytree.
Distributions are mathematical objects → immutable matches semantics;
no batch norm, no running averages, no `_fitted` flags. Parameter
updates go through `eqx.tree_at` or the `replace(**)` facade.

---

## Core Dependencies

| Package | Role |
|---|---|
| `jax` | Array computation, autodiff, JIT, vmap |
| `equinox` | Pytree-based modules (immutable, filterable) |
| `jaxopt` | L-BFGS-B for GIG η→θ constrained optimization |
| `tensorflow_probability.substrates.jax` | `log_bessel_kve` only |
| `scipy` | CPU Bessel evaluation via `kve` (EM hot path) |
| `optax` | optional, for meta-learning experiments |

Module-level functions are forbidden. Distribution behaviour lives on
the class as `@classmethod` or `@staticmethod`.

---

## Decision Table

> One row per design decision. Subsystem detail and rationale are
> linked from the right column.

### Foundational

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| F1 | Base module | `eqx.Module` (not Flax NNX) | Immutable matches math; no mutable state |
| F2 | Parametrizations | One class, three constructors | `from_classical`, `from_natural`, `from_expectation` — see `exponential_family.md` |
| F3 | Autodiff | `jax.grad` on `_log_partition_from_theta` | Single source of truth |
| F4 | Triad classmethods | `_grad_log_partition`, `_hessian_log_partition`, plus CPU triad | `exponential_family.md` § 2 |
| F5 | Unbatched core | `log_prob(x)` for single obs | Clean; batch via `jax.vmap` |
| F6 | Constraints | `jnp.maximum(x, LOG_EPS)` (clamp) | `exponential_family.md` § 4.3 |
| F7 | `jnp.where` over `lax.cond` in log-partition | `jnp.where` | `vmap`-compatible; clamping prevents NaN gradients |
| F8 | Precision | float64 throughout | Bessel + EM convergence require it |
| F9 | Numerical constants | Centralised in `utils/constants.py` | No scattered magic numbers |
| F10 | Module-level functions | **forbidden**; classmethods or staticmethods | Keeps the interface on the class |

### Mixtures

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| M1 | Mixture design | Joint + Marginal classes | Joint **is** an EF, marginal is not — `mixtures.md` § 1 |
| M2 (D2) | Joint public API | Public `ExponentialFamily` exports; `log_prob` on flat `concat(x,[y])` | `mixtures.md` § 2 |
| M3 (D3) | `MultivariateNormal` | Full `ExponentialFamily` with analytical η ↔ θ | `exponential_family.md` § 4.1 |
| M4 | Factor analysis | **Sibling** of `NormalMixture`, not subclass | 10-stat complete data; Woodbury — `mixtures.md` § 6 |
| M5 | `from_expectation` η→model | Canonical map on both layers; closed-form pytree path + Bregman fallback | `mixtures.md` § 5 |
| M6 | Parameter facade | `replace(**)` on `NormalMixture`; subordinator forwarders per subclass | `mixtures.md` § 4 |
| M7 | `DispersionModel` ABC | **deferred** | Only two storage variants today — `mixtures.md` § 6.5 |
| M8 | Convergence on FA $F$ | Compare $\Sigma = F F^\top + \mathrm{diag}(D)$, not $(F, D)$ | $F$ is gauge-only — `em_framework.md` § 2.1 |

### EM and η-update rules

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| E1 | EM separation | Model + Fitter (GMMX-style) | `em_framework.md` § 1 |
| E2 | EM return value | `EMResult` (not bare model) | Diagnostics, timing, optional LL trace |
| E3 | Convergence criterion | Max hybrid-scale L2 change `‖Δ‖/(1+‖θ‖)` in `em_convergence_params()`; subordinator excluded | Avoids near-zero `μ` inflation; `em_framework.md` § 2.1 |
| E4 | Fitter classes | `BatchEMFitter` + `IncrementalEMFitter` (D1) | Replaces obsolete `OnlineEMFitter` / `MiniBatchEMFitter` |
| E5 | η-update rule abstraction | Two layers: `EtaUpdateRule.__call__` + `AffineRule.weights` | Future ML-style predictors plug in; `em_framework.md` § 3 |
| E6 | `EtaUpdateRule` | `eqx.Module` (not plain ABC) | Hyperparams are JAX leaves — JIT-compatible, differentiable |
| E7 | `affine_combine` weight forms | scalar / stats-shape pytree / callable | Per-field shrinkage, custom linear ops; `em_framework.md` § 3.1 |
| E8 | `Shrinkage` | Combinator wrapping any base rule (replaces `ShrinkageUpdate`) | `em_framework.md` § 4 |
| E9 | Sufficient statistics | Pytree (`NormalMixtureEta`, `FactorMixtureStats`) in **theory order** | Readable; first 6 fields shared across families |
| E10 | `lax.scan` EM | Both backends JAX, low verbosity, no `eta_update` | JIT-friendly; otherwise Python loop |
| E11 | Cold vs warm η→θ | `from_expectation(theta0=None)` defaults to `jnp.zeros_like(eta)`; instance `fit` uses `natural_params()` | GIG overrides cold start with multistart |

### EM numerical robustness (VG inverse-moment / unbounded-likelihood)

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| E12 | Posterior-GIG map | One `_posterior_gig_params` / `_floored_posterior_gig_params` pair per hierarchy via `subordinator().to_gig()`; 8 subclass overrides deleted (R2) | Same conjugacy in GIG coordinates for all four families; single `B_POST_FLOOR` chokepoint. `tech_notes/vg_em_inverse_moment_singularity.md` § 7 |
| E13 | Prior moment at `α ≤ 1` | Distribution-specific floor on the `(α−1)` denominator of `β/(α−1)` (`ALPHA_MOMENT_MARGIN`); keep the two finite moments exact (B2) | **Not** the lifted-GIG `expectation_params()` — that injects ~1.0 abs error in `E[log Y]` at `α=0.2` and routes a closed-form special case through general Bessel machinery |
| E14 | VG unbounded likelihood | Opt-in `fit(alpha_min=…)` clamps the Gamma shape (`ghyp` fix-λ analogue); VG-only, default `None` = no change (F4) | The b_post floor keeps EM *finite* but not *bounded*: for `α ≤ d/2` the VG density diverges at `x=μ`. `alpha_min` restricts the estimand. Threaded as a static `m_step_kwargs` entry; resolves `'density'`→`d/2+ε`, `'inverse_moment'`→`d/2+1+ε`. VG alone needs it (unique `b=0` family); NInvG/NIG/GH have prior `b>0`. `tech_notes/vg_em_inverse_moment_singularity.md` § 5 |

### Covariance regularisations (after each M-step)

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| R1 | `'none'` | Identity (default) | Don't move the model |
| R2 | `'det_sigma_one'` | $\|\Sigma\| = 1$ — classical GH convention | `em_framework.md` § 5 |
| R3 | `'det_sigma_x'` | $\log\|\Sigma\| = \log\|\Sigma_0\|$, captured at the start of `fit` | Aligns GH / FactorGH display with VG / NInvG / NIG when comparing fits |
| R4 | `'a_eq_b'` | Rescale GIG so $a = b = \sqrt{ab}$ (orbit invariant); NIG: $\mu_{IG} = 1$; no-op for VG / NInvG / MVN | `em_framework.md` § 5.2 |
| R5 | `_rescale` / `_build_rescaled` pattern | Marginal handles Σ, γ; subclass `_build_rescaled` handles subordinator | `em_framework.md` § 5.3 |

### Solvers and Bessel

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| S1 | Bregman divergence | Generic `f` (not `log_partition_fn`) | Decouple optimization from EFs — `solvers_and_bessel.md` § 1 |
| S2 | Solver interface | `grad_fn` + `hess_fn` (both θ-space) — solver applies chain rule | Distributions never see φ-space |
| S3 | Newton implementation | Hand-rolled `lax.while_loop`; module-level JIT cache | No JAX library has Newton minimizer with custom Hessian |
| S4 | Multi-start | `vmap` for JAX, Python `for` for CPU | Orthogonal to solver name |
| S5 | GIG η→θ | η-rescaled + CPU L-BFGS-B; warm-start defaults to `cpu/lbfgs` | Ill-conditioning + GPU dispatch overhead |
| S6 | `BregmanResult` typing | Loose `Any` for scalars | Survives `lax.scan` carries |
| S7 (D4) | `jaxopt` migration | Keep for now; `DeprecationWarning` suppressed at import | LBFGSB is uniquely useful — `exponential_family.md` § 4.2 |
| S8 | Bessel | Pure-JAX (4 regimes) + CPU `scipy.kve` backend | JAX for JIT/autodiff; CPU for EM throughput |
| S9 | Hybrid backend | Quad forms in JAX, Bessel + GIG solve on CPU | 15× E-step, ~500× M-step on SP500 GH benchmark |

### Conventions

| # | Decision | Choice | Why / Detail |
|---|---|---|---|
| C1 | Class naming | `GeneralizedInverseGaussian` primary, `GIG` alias | Full name canonical; short alias for legacy code |
| C2 | Cholesky factors | always named `L_Sigma` | Single convention |
| C3 | Sufficient statistics field naming | descriptive (`E_inv_Y`, `E_X_inv_Y`, …) in theory order | Heterogeneous shapes, `tree.map` works naturally |

---

## Cross-References

- Topical detail: `index.md` (TOC).
- Architecture: `../ARCHITECTURE.md`.
- Implementation surface: `AGENTS.md` and `.cursor/rules/coding-conventions.mdc`.
- Theory: `../../docs/theory/`.
- Tech notes: `../tech_notes/`.
- Historical / archived design proposals: `../archive/design/`.
- Unimplemented proposals: `../plans/finance_architecture.md`.
