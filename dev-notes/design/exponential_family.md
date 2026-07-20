# Exponential Family Core

> **Scope.** Why every distribution lives behind one log-partition function,
> how the triad pattern keeps gradient/Hessian and JAX/CPU evaluation
> consistent, and how these primitives plug into the η→θ Bregman solver.
>
> **Where things live.** Module hierarchy and the full triad table are in
> `../ARCHITECTURE.md` § *Exponential Family Core*. This file records
> the rationale.

---

## 1. One Function, Three Parametrizations

Each distribution is described by a single convex log-partition
$\psi(\theta)$. Everything else — densities, expectation parameters,
Fisher information, Bregman inversion — is derived from it. Three
parametrizations form a triangle:

```
classical (μ, σ², α, β, …)   ←→   natural θ   ←→   expectation η = ∇ψ(θ)
                  via from_classical / natural_params                ↑
                                                                from_expectation
```

`from_natural`, `natural_params`, and `from_expectation` are the only three
constructors a user ever needs. `from_expectation(η)` runs the universal
Bregman solver `min_θ [ψ(θ) − θ·η]` from `fitting/solvers.py`.

Why it matters: distributions never need to know about parameter
conversions other than their own classical mapping. The conversion graph
collapses into derivatives and Bregman minimisation of a single function.

---

## 2. The Log-Partition Triad

Three functions × two backends:

```
                     JAX (JIT-able)                  CPU (numpy/scipy)
log-partition        _log_partition_from_theta        _log_partition_cpu
gradient             _grad_log_partition              _grad_log_partition_cpu
Hessian              _hessian_log_partition           _hessian_log_partition_cpu
```

| Tier | Default | Override when |
|------|---------|---------------|
| 1: `_log_partition_from_theta` | abstract | always (it *is* the distribution) |
| 2: `_grad_*`, `_hessian_*` (JAX) | `jax.grad` / `jax.hessian` | analytical formula avoids recompilation or saves Bessel calls |
| 3: `_log_partition_cpu` etc.    | numpy wrappers around the JAX version | distribution calls `log_kv` (Bessel) and EM hot path needs CPU evaluation |

### 2.1 Why `@classmethod` for the triad

Tier 2 needs to differentiate the **subclass**'s `_log_partition_from_theta`,
not the parent's. `@classmethod` gives `cls`, which routes to the right
implementation. `@staticmethod` cannot — it would always call the parent.
`cls` is not traced by JAX, so this is JIT-safe.

### 2.2 Why the CPU tier exists

`scipy.special.kve` is **fast** for vectorised Bessel evaluation in the
EM hot path; a `vmap` over JAX's `lax.cond`-dispatched `log_kv` triggers
separate kernel launches per regime check and is much slower. Distributions
that don't call `log_kv` (Gamma, InverseGamma, InverseGaussian) inherit
the default `np.asarray(jax_version(...))` wrappers — they pay nothing for
the CPU tier.

### 2.3 Why we don't share Bessel calls between gradient and Hessian

A combined `_grad_hess` could share Bessel evaluations (≈ 7 calls vs
12 for separate). We keep them separate anyway: the saving is small,
and combining them re-introduces a phi-space chain rule the *distribution*
must understand. Keeping `_grad_log_partition` and `_hessian_log_partition`
as θ-space-only lets the solver (§3) own the chain rule generically.

---

## 3. Bregman Solver Interface

`fitting/solvers.py` exposes `solve_bregman(f, η, θ₀, ...)` for any convex
$f$ — not just log-partitions. The interface uses **θ-space only** for
gradient/Hessian; bounds are handled by the solver via
`_setup_reparam` (φ ↔ θ). Distributions never see the reparametrization.

```python
solve_bregman(f, eta, theta0, *, backend, method, bounds,
              grad_fn, hess_fn, max_steps, tol, verbose)
```

| Decision | Choice | Why |
|---|---|---|
| `grad_fn` + `hess_fn` (vs combined `grad_hess_fn`) | separate | Distributions don't know about φ-space; solver applies chain rule via `jax.jacobian(to_theta)` |
| Newton implementation | hand-rolled `lax.while_loop` | No JAX library provides a Newton **minimizer** that accepts a user-supplied Hessian (Optimistix Newton is root-finding only, JAXopt has no Newton) |
| Multi-start | `jax.vmap` for JAX, Python `for` for CPU | Orthogonal wrapper, not baked into solver name |
| Result type | `BregmanResult` with `Any`-typed scalars | Survives `lax.scan` without `ConcretizationTypeError` |
| Cached JIT | module-level `_gig_jax_newton_jit` via `make_jit_newton_solver` | The GIG warm-start hot path otherwise retraces per call (see `../tech_notes/jax_overhead_diagnosis.md`) |

**`backend × method` matrix:**

| backend | method | bounds | What runs |
|---|---|---|---|
| `jax` | `newton` | reparam | hand-rolled `while_loop`, autodiff or supplied `grad_fn`/`hess_fn` |
| `jax` | `lbfgs` | native (LBFGSB) | `jaxopt.LBFGSB` |
| `jax` | `bfgs`  | reparam | `jaxopt.BFGS` |
| `cpu` | `newton`| reparam | `scipy.optimize.minimize(method='trust-exact')` |
| `cpu` | `lbfgs` | native | `scipy.optimize.minimize(method='L-BFGS-B')` |
| `cpu` | `bfgs`  | none   | `scipy.optimize.minimize(method='BFGS')` |

GIG warm-start hot path: `backend='cpu', method='lbfgs'` (scipy's L-BFGS-B
+ `scipy.kve`) avoids GPU dispatch on this 3-D scalar problem. See
`../tech_notes/gig_eta_to_theta.md` for the η-rescaling derivation and
benchmarks.

---

## 4. Pre-1.0 Decisions Recorded Here

### 4.1 D3 — `MultivariateNormal` as `ExponentialFamily`

Promoted from a plain `eqx.Module` to a full `ExponentialFamily`. EF
structure:

| Component | Expression |
|---|---|
| $t(x)$ | $[x,\;\operatorname{vec}(xx^\top)]$, shape $(d + d^2,)$ |
| $\theta$ | $[\Sigma^{-1}\mu,\;-\tfrac12\operatorname{vec}(\Sigma^{-1})]$ |
| $\log h(x)$ | $0$ |
| $\psi(\theta)$ | $\tfrac12\theta_1^\top\Lambda^{-1}\theta_1 - \tfrac12\log\|\Lambda\| + \tfrac d2\log(2\pi)$, $\Lambda = -2\,\mathrm{reshape}(\theta_2)$ |

`vec` uses row-major (`ravel()`) throughout. All conversions are
analytical (Tier 2 override) — no Bregman solver is ever invoked for
MVN. `_log_partition_from_theta` uses Cholesky of $\Lambda$ for numerical
stability; `log_prob` overrides the inherited EF formula with a direct
Cholesky path (more efficient).

### 4.2 D4 — Keep `jaxopt` for now

JAXopt is unmaintained (last release 0.8.3) and emits a `DeprecationWarning`
on import. We keep it: it is the only pure-JAX library with a native box-
constrained quasi-Newton (`LBFGSB`). Migrating to `optax.scale_by_lbfgs`
loses the convergence loop; `optimistix.LBFGS` lacks box constraints.

The deprecation warning is suppressed in `normix/__init__.py`. Migration
recipe (when JAXopt breaks): wrap `optax.scale_by_lbfgs` in
`jax.lax.while_loop` (~100 lines), then drop `jaxopt`.

### 4.3 Constraints handling

`jnp.maximum(x, LOG_EPS)` (clamp), not `paramax`. Reasons:

- The reparametrization we need is 8 lines, fully understandable.
- EM does not need gradients through the constraints (it parameterises θ
  in the constrained space, not log-space).
- No extra dependency.

`jnp.where` is preferred over `lax.cond` whenever possible: it is
`vmap`-compatible without changing the trace, and the clamping prevents
NaN gradients without branch divergence.

### 4.4 Module-level functions are forbidden

Distribution behaviour lives on the class as `@classmethod` or
`@staticmethod`. No `_helper(self.alpha, ...)` module-level functions
that close over attributes — they leak the class API into module globals
and are hard to override.

---

## 5. Divergences in Gauge Coordinates (DEC-1)

> Decision row: `design.md` § *2026-07 review Phase 0*, DEC-1. Status:
> decided 2026-07-20, implementation lands with roadmap item B1.

### 5.1 The failure and its geometry

Tier-2 divergences evaluated `type(p)`'s ψ at both operands' θ. That is
correct only when ψ is faithful on the whole θ-segment the formulas
touch: the Hellinger affinity needs ψ at the **midpoint**
$(\theta_p + \theta_q)/2$; the KL Bregman form needs ψ at both ends and
$\nabla\psi$ at $\theta_p$. Each special-case ψ is the restriction of
GH's (or GIG's) ψ to the family's embedded sub-manifold and ignores the
constrained coordinates — valid *on* the manifold, undefined off it.

Whether the divergence formulas stay on the manifold is a question of
embedding geometry:

**A sub-family ψ restriction is divergence-faithful iff its embedding
is θ-affine.**

| Family | Constraint in θ | Affine? | Same-family divergences today |
|---|---|---|---|
| Gamma ⊂ GIG | θ₂ ≡ 0 | yes | correct |
| InverseGamma ⊂ GIG | θ₃ ≡ 0 | yes | correct |
| InverseGaussian ⊂ GIG | θ₁ ≡ −3/2 | yes | correct |
| NIG ⊂ GH | θ₁ ≡ −3/2 − d/2 | yes | correct |
| VG ⊂ GH | θ₂ = −½μᵀΛμ (b = 0) | **no** (quadratic in θ₄…θ₆) | **wrong** |
| NInvG ⊂ GH | θ₃ = −½γᵀΛγ (a = 0) | **no** | **wrong** |

For curved embeddings the midpoint leaves the manifold, and the
restricted ψ silently drops the lost direction. Measured (d = 2,
μ_p = [0.1, −0.2], μ_q = [0.9, 0.5], γ = [0.3, 0.1],
Σ = [[1.0, 0.3], [0.3, 0.8]]):

| Pair | Today | Quadrature truth |
|---|---|---|
| H²(VG(μ_p), VG(μ_q)), α=3, β=1.5 | **0.0** | 0.0813770940 |
| KL(VG(μ_p) ‖ VG(μ_q)) | **0.0** | 0.3517605634 |
| H²(NInvG(γ₁), NInvG(γ₂)), α=3, β=2 | **0.0** | 0.1528534803 |
| H²(Gamma(2, 1.5), InvGamma(3, 2)) | 0.100563 | 0.0592459797 |
| H²(NIG, GH_p=0.7) vs H²(GH, NIG) | 0.064060 / 0.026949 | 0.026949 (symmetric) |

The cross-family case (review §1.1) is the same disease: Gamma's ψ
reading an InverseGamma θ, or NIG's ψ reading a GH θ, evaluates the
wrong restriction. Same-family VG/NInvG breakage means a
**type-mismatch trigger is insufficient** — the roadmap's original "when
`type(p) is not type(q)`, lift" sketch is superseded.

### 5.2 The decision: unconditional gauge canonicalization

Every operand is canonicalized into its *divergence gauge* — the
ambient family whose ψ is faithful on the convex hull of all lifted
members: GIG for the univariate subordinator tree, `JointGH` for the
joint-mixture tree, `self` for everything else (MVN, GIG, JointGH).
One hook, no faithfulness flags, no dual code path:

```python
class ExponentialFamily(eqx.Module):
    def _divergence_lift(self) -> "ExponentialFamily":
        """Return self in its divergence gauge. Default: identity."""
        return self
```

`Gamma/InverseGamma/InverseGaussian._divergence_lift → self.to_gig()`;
`JointNormalMixture._divergence_lift` assembles the GH joint from
`subordinator().to_gig()` (one base implementation — the same assembly
every `to_joint_generalized_hyperbolic` performs); `JointGH` returns
`self`. Tier-2 `squared_hellinger`/`kl_divergence` lift both operands,
require `type(lp) is type(lq)` (else `TypeError`; equal gauge with
different `d` → `ValueError`), then call Tier 1 with the gauge ψ.
`NormalMixture`'s delegation to joints and the Tier-3 module functions
are unchanged and inherit the fix. The public Tier-2 methods remain
overridable; the hooks are the default extension points.

Boundary semantics: **`boundary_eps = 0` always.** The lifts are
θ-preserving, and GIG's degenerate ψ branch (`GIG_DEGEN_THRESHOLD`)
evaluates boundary θ exactly — measured
ψ_GH(θ_VG) = ψ_VG(θ_VG) = −0.955414807309 to machine precision, and
every lifted H² above matches quadrature to ≤ 1e-10, including
both-operands-on-different-boundaries (Gamma vs InverseGamma: midpoint
is interior GIG). An ε > 0 would *introduce* error.

### 5.3 KL: exact η with a tail split

$\mathrm{KL}(p\|q) = \psi(\theta_q) - \psi(\theta_p) -
\Delta\theta^\top \eta_p$ needs $\eta_p = E_p[t]$ in gauge coordinates.
Two sourcing options fail:

- `jax.grad` of the gauge ψ at boundary θ differentiates through the
  degenerate-branch `jnp.where` and clamps: the E[1/Y] slot returns
  12896.0 where the truth is 1.0 (VG α = 2.5 lift).
- GIG's analytical Bessel-ratio gradient is exact in the *used* slots
  but the boundary-unused moment degrades near its pole (E[1/Y]
  relative error 4e-16 at α = 2.5, 1.9e-7 at 1.5, 1e-3 at 1.1, garbage
  for α ≤ 1 where the truth is +∞) — the same reason row E13 rejected
  lifted-Bessel moments for `compute_eta_from_model`.

So moments come from the source family's closed forms, through a
**tail-split** hook that represents possibly-infinite moments without
materializing `0·inf`:

```python
def _divergence_eta(self) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact (eta_fin, m, v) with η = eta_fin + m·v in gauge coordinates.

    m ≥ 0 is the one possibly-divergent scalar moment (may be jnp.inf);
    v is its structural direction in the gauge t-layout. At most one
    moment can diverge (a source sits on at most one GIG boundary).
    """
```

Tier 1 gains one pure sibling (the existing `kl_divergence_from_psi`
is untouched):

$$
\mathrm{KL} = \psi(\theta_q) - \psi(\theta_p)
- \Delta\theta^\top \eta_{\mathrm{fin}} + m \cdot A,
\qquad A = \max(-\Delta\theta^\top v,\, 0),
$$

with a double-`jnp.where` masking `m` when A = 0 (F7). The coefficient
A is provably nonnegative and has closed form — for a Gamma-boundary
source (VG):

$$
A = \tfrac{b_q}{2}
  + \tfrac{1}{2}(\mu_p - \mu_q)^\top \Lambda_q (\mu_p - \mu_q),
$$

and symmetrically $A = a_q/2 + \tfrac12 (\gamma_p - \gamma_q)^\top
\Lambda_q (\gamma_p - \gamma_q)$ for the InverseGamma boundary. So
KL = +∞ **exactly when the infinite moment couples to a nonzero chord
coefficient**: VG(α ≤ 1) vs any b_q > 0 target or shifted μ is honestly
infinite, while two VGs with identical (μ, Σ) differing only in (α, β)
have A = 0 and stay finite — the mathematically correct extended-real
semantics. The E13 `ALPHA_MOMENT_MARGIN` floor stays EM-scoped and
never enters divergences.

Per-family η overrides: Gamma/InverseGamma closed forms with the split
(unfloored, honest ±∞ pole); InverseGaussian inherits the interior GIG
Bessel path; GIG gets a degenerate-aware split mirroring its ψ branches
(so hand-built boundary GIGs work); `JointNormalMixture` gets one base
assembly from `subordinator()._divergence_eta()` plus the normal block —
the same algebra as `compute_eta_from_model`, factored into a shared
helper so the moment→η map has one source of truth.

Verified end-to-end against quadrature: KL(VG ‖ GH) = 0.1047209744,
KL(GH ‖ VG) = 0.0878955379, KL(Gamma(2.5, 1.5) ‖ InvGamma(3, 2)) =
0.4799889676 — all exact through the gauge formulas.

### 5.4 Reconciliation with the Simplicity row

The design.md Simplicity paragraph ("special cases must use their own
analytical formulas rather than routing through GH") binds
per-observation hot paths — densities and estimators — where the
special-case forms are strictly better. It does not bind divergences:
off the sub-manifold the special-case ψ is not "less optimal", it is
**undefined** (§ 5.1), so the gauge ψ is the only correct coordinate
system. At the boundary the GIG degenerate branch *is* the special-case
closed form (Gamma–Gamma through the gauge evaluates the identical
`gammaln(α) − α log β` expression), so nothing closed-form is routed
through approximating machinery. The row's spirit survives in the η
sourcing, where lifted-Bessel moments are explicitly rejected.

### 5.5 Arena synthesis note

Three candidates (Fable, Sol xhigh, Grok families) converged on
unconditional lift + Tier-2 ownership + exact η + `boundary_eps = 0` —
strong consensus on the shape. Base: the Fable candidate (gauge
concept, tail-split η, Tier-1 KL sibling with the A ≥ 0 coefficient —
the only candidate whose infinity semantics survive the
zero-chord-coefficient edge). Grafts: friendly `ValueError` on
dimension mismatch within a gauge, and preserving the public Tier-2
override contract (both from the Sol candidate); "ambient family"
prose terminology (Grok candidate). Rejected: a per-class
`_native_divergence_is_faithful` boolean with a native fast path for
affine same-family pairs (second source of truth re-encoding the
affineness criterion; measured accuracy gain nil — the degenerate
branch is formula-identical, and divergences are diagnostics, not hot
paths); blanket `any(~isfinite(η)) → +inf` (wrong for the
zero-coefficient edge); raising on all cross-family pairs (turns
measured-exact comparisons into errors); quadrature/MC fallbacks
(slow, non-JIT, unnecessary).

Out of scope, deliberately: marginal-of-X divergences (intractable;
the joint upper-bound semantics of `NormalMixture` delegation are
retained), factor-mixture divergences (no joint EF layer),
gradient semantics of an infinite KL (documented as undefined),
`boundary_eps` plumbing.

---

## 6. Cross-References

- Architecture surface: `../ARCHITECTURE.md` § *Exponential Family Core*.
- η→θ optimization for GIG: `solvers_and_bessel.md`.
- Tech notes: `../tech_notes/gig_eta_to_theta.md`,
  `../tech_notes/bessel_implementations_survey.md`,
  `../tech_notes/jax_overhead_diagnosis.md`,
  `../tech_notes/distribution_conversions.md` (lift/projection duality,
  boundary_eps rationale).
- Historical / archived rationale: `../archive/design/log_partition_triad.md`,
  `../archive/design/solver_redesign.md`.
