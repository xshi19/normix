# Why normix does not fit by gradient descent — record audit

**Date:** 2026-08-23
**Trigger:** the GPJax review (`../references/gpjax_review.md` § 2.2, § 5) —
GPJax, FlowJAX, and efax all constrain parameters model-side (paramax /
softplus) and optimise in unconstrained space with gradient-based methods,
while normix fits by EM with constrained Bregman η→θ solves. The question:
did we record *why* we rejected the gradient-descent approach — in
particular the memory that it "does not work well in the GIG case with the
log Bessel function" — and where?

**Verdict in one line:** the decision is well recorded across git history,
dev-notes, and one chat transcript, but the pieces were scattered and
nothing user-facing explained the choice. The missing head-to-head is now
[`../tech_notes/gradient_fitting_comparison.md`](../tech_notes/gradient_fitting_comparison.md):
L-BFGS+softplus matches `fit_mle` on interior GIG; Adam lags; degenerate
$(a,b)$ still needs η-rescaling; `log_kv` $\partial_\nu$ is *not* the
blocker. Remaining: user-facing `docs/design/` page + `design.md` row.

---

## 1. The decision in question

Fit GH-family models by EM: E-step computes $E[t(Y) \mid X]$, M-step solves
$\min_\theta [\psi(\theta) - \theta \cdot \eta]$ (constrained, θ-space
bounds, warm-started quasi-Newton/Newton) — instead of running Adam/SGD on
the marginal negative log-likelihood over softplus-reparametrised classical
parameters, as GPJax (`fit` + Optax), FlowJAX (`fit_to_data`), and efax
(softplus search space) do.

## 2. What the record says

### 2.1 The founding decision — git history only

Two design docs from the JAX-migration week exist only in git history
(written 2026-03-07, superseded the same month, never migrated to
`dev-notes/`):

- **`docs/jax_normix_fit_design.md`** (commit `5c679e8`) — the decision
  document. Direct quotes:
  - "This means **fitting an exponential family is the same operation as
    converting expectation parameters to natural parameters**. There is no
    gradient descent involved — it's either a closed-form formula or a
    convex optimization problem."
  - "For exponential families, this [FlowJAX's SGD loop] is wasteful. You
    already know $\hat\eta = \overline{t(x)}$ — there's nothing to optimize
    over the data. … Gradient descent would be like using SGD to compute a
    sample mean."
  - A three-row cost table: closed-form MLE (1 step) vs convex η→θ
    (~10–50 Newton steps) vs gradient descent (~1000 epochs × batches).
  - § 5.3 item 6 reserved a FlowJAX-style `GradientFitter` "for edge
    cases"; it was never implemented.
- **`docs/jax_fitting_design_analysis.md`** (commit `485282e`, replaced by
  the above the same day) — source-level analysis of FlowJAX
  `fit_to_data` and efax `ExpToNat`. Conclusion: "Don't reuse FlowJAX's
  loop directly."

Recover both with `git show 5c679e8:docs/jax_normix_fit_design.md` and
`git show 485282e:docs/jax_fitting_design_analysis.md`.

### 2.2 The paramax rejections — living docs, but a different argument

- `../design/exponential_family.md` § 4.3 (design.md row F6): clamp
  (`jnp.maximum(x, LOG_EPS)`), not paramax — because the needed
  reparametrisation is 8 lines, EM does not need gradients through the
  constraints, and it avoids a dependency.
- `../archive/design/solver_redesign.md` § 2.3 (2026-03-18): "keep
  hand-rolled … Adding `paramax` as a dependency for this alone is not
  justified."
- `../references/gpjax_review.md` § 5: the sharpest surviving statement of
  the trade-off — "GPJax stores the natural/expectation parameters as raw
  `Real` fields and lets Adam walk them (no PSD projection), whereas normix
  solves η→θ as a Bregman problem with domain guarantees."

Note these reject paramax **as a mechanism**, on simplicity and dependency
grounds. None of them says "gradient descent was tried and failed."

### 2.3 The Bessel-gradient evidence

The strongest technical objection on record concerns $\partial_\nu \log
K_\nu(z)$, which any gradient-based fit of GIG/GH needs (the order $\nu$
is the shape parameter $p$):

- The Bessel comparison notebook (commit `b391543`; quoted in the
  2026-03-08 transcript): "**Neither TFP nor logbesselk computes d/dv
  correctly (both return 0). We need custom_jvp with finite differences
  for d/dv.**" Autodiff through off-the-shelf Bessel implementations
  produced silently wrong (zero) ν-gradients — any gradient-based
  optimiser over $p$ would have been broken without noticing.
- The deleted research doc `docs/jax_bessel_research.md` (recover:
  `git show 0681fa2^:docs/jax_bessel_research.md`): TFP documents
  $\nabla_\nu$ as undefined; "This means TFP is insufficient for GIG/GH
  distributions where $p$ … must be optimized."
- The fix that lives today: `normix/utils/bessel.py` — `@jax.custom_jvp`
  with exact recurrence for $\partial_z$ and **central finite differences
  (ε = `BESSEL_EPS_V`) for $\partial_\nu$**. FD accuracy was measured at
  ~1e-7–1e-8 relative across $v \le 500$, $z \in [10^{-6}, 10^3]$
  (2026-03-08 transcript) — fine for a quasi-Newton solve to `gtol=1e-10`
  on a 3-D problem, but a hard ceiling on gradient quality, and each
  ν-tangent costs two extra Bessel evaluations.

### 2.4 The empirical GIG failure — real, but for L-BFGS-B, not Adam

A test notebook (`notebooks/gig_optimization_test.ipynb`, built and
executed in the 2026-03-08 chat "Paramax package and Parameterize
functionality", id `ea8048c3`) swept (p, a, b) including edge cases and
found the failures the user remembers:

| Regime | η err | param err | Root cause |
|---|---|---|---|
| Normal range, Gamma/IG limits | < 1e-7 | < 1e-3 | clean |
| Large $a = 10^5$, $b = 1$ | 2e-9 | 1.50 | non-identifiability at huge $\sqrt{ab}$ |
| Large $\sqrt{ab} = 1000$ | 1.3e-5 | 2.50 | Fisher ill-conditioning |
| $a = 10^4$, $b = 10^{-3}$ | 4.9e-3 | 1.59 | optimization failure |

These are failures of **unrescaled L-BFGS-B on the Bregman η→θ
objective**, not of Adam on the likelihood. They produced the η-rescaling
now recorded in `../design/solvers_and_bessel.md` § 2 ("Fisher information
can be ill-conditioned (condition number up to $10^{30}$) … Vanilla
L-BFGS-B fails without rescaling") and `../tech_notes/gig_eta_to_theta.md`.
**The notebook itself was never committed** — it is absent from git history
and from `notebooks/` today; the results survive only in the transcript.

### 2.5 The pre-JAX era

The 2025-11 refactor chat ("Refactor generalized hyperbolic distribution
repo", id `622b86d9`) sketched a `fit` via natural gradient descent
($\eta^{t+1} = \eta^t + \alpha G^{-1}\nabla L$), and the old ROADMAP
(commit `4096d77`) lists GIG "Fitting methods: MLE, moments, natural
gradient" as planned. What actually landed in the numpy-era
`pygh/base/exponential_family.py` was scipy L-BFGS-B with bounds for
`expectation_to_natural` — the natural-gradient fit was never implemented.
(The archived `solver_redesign.md` § 6.2 later noted the reason it would
have been redundant: Newton on the Bregman objective *is* natural gradient
descent for exponential families.)

## 3. What is inferred (not directly recorded)

Assembling the recorded facts, the full case against likelihood gradient
descent for this family is:

1. **Structural waste** (recorded, § 2.1): the M-step/MLE is moment
   matching plus a 3-D convex solve; SGD re-derives a sample mean.
2. **Monotonicity** (recorded in the `ea8048c3` transcript): EM guarantees
   monotone likelihood; gradient descent on the Bessel-heavy marginal does
   not.
3. **Gradient quality through `log_kv`** (recorded, § 2.3): $\partial_\nu$
   exists only as an ε=1e-5 finite difference; autodiff without the custom
   JVP silently returns 0.
4. **Curvature** (recorded for the Bregman problem, § 2.4; *inferred* for
   the likelihood): Fisher condition numbers up to $10^{30}$ at asymmetric
   (a, b). Given that curvature, a first-order method on the likelihood
   would plausibly stall where even quasi-Newton needed η-rescaling —
   softplus reparametrisation changes the domain, not the conditioning.
5. **Boundary families** (*inferred* from the transform tables): softplus/
   exp parametrisations cannot represent $a = 0$ or $b = 0$ exactly
   (φ → −∞), yet normix's special cases sit exactly on those boundaries
   (VG has $b = 0$; NInvG has $a = 0$). Solver-side bounds plus exact
   degenerate branches keep them reachable.

Point 4 and 5 are inference chains, not measurements. Nobody ever ran
Adam + softplus on the GIG or GH likelihood and wrote down what happened —
the user's memory of "gradient descent not working with log Bessel"
compresses § 2.3 (zero ν-gradients under plain autodiff) and § 2.4
(unrescaled quasi-Newton failures) into one recollection.

## 4. Gaps

1. **No committed experiment.** `gig_optimization_test.ipynb` exists only
   in a chat transcript; no benchmark compares gradient-based likelihood
   fitting against EM (`benchmarks/bench_gig_solvers.py` covers only
   Bregman backends/methods).
2. **No user-facing rationale.** `docs/design/` explains what the Bregman
   solver does, never why gradient descent was not adopted. `rg "gradient
   descent" docs/` matches only the CVaR tutorial (unrelated weight-space
   optimisation).
3. **No design.md row.** F6 covers clamp-vs-paramax; no row states
   "EM + constrained η→θ over likelihood gradient descent."
4. **Founding docs unarchived.** The two 2026-03-07 fit-design docs are
   git-only; `dev-notes/archive/design/` does not contain them.

## 5. Recommendations

In order — the study first, so the docs page cites measurements rather
than folklore (the user proposed exactly this sequencing):

1. **Run the clean empirical study.** **Done (2026-08-23):**
   [`../tech_notes/gradient_fitting_comparison.md`](../tech_notes/gradient_fitting_comparison.md)
   + `benchmarks/bench_gradient_fitting.py`. Headline: L-BFGS+softplus
   *matches* `fit_mle` on interior GIG (so the ML recipe is a valid MLE);
   Adam lags; degenerate $(a,b)$ still needs η-rescaling; `log_kv`
   $\partial_\nu$ is *not* the blocker (rel. err. $\sim 10^{-9}$).
   VG $\alpha\le d/2$ unbounded-likelihood was left out (separate note).
2. **Publish the rationale.** A section in `docs/design/em_framework.md`
   (or a sibling page) titled "Why not gradient descent", written from the
   study's numbers: exponential-family identity + timing tables. Do **not**
   lead with a stale Bessel-$\partial_\nu$ story (H1 was rejected). Natural
   home for GPJax-review adoption item #1 (sharp-bits / method-selection).
   Follow `.cursor/rules/docs-cross-links.mdc`.
3. **Add the design.md row** (EM section): decision "EM + constrained
   Bregman η→θ, not likelihood gradient descent", linking the tech note
   and the docs page.
4. **Archive the founding docs**: copy the two git-only fit-design docs
   into `dev-notes/archive/design/` with an archival header, so the
   rationale is browsable without git archaeology.

## Sources consulted

- `../design/design.md` — rows F6, S1–S9; no gradient-descent row (gap).
- `../design/exponential_family.md` § 4.2–4.3 — clamp-vs-paramax rationale.
- `../design/solvers_and_bessel.md` § 1–2 — Bregman solver, η-rescaling,
  ill-conditioning record.
- `../design/em_framework.md` — no gradient-descent discussion (only the
  note that eqx leaves are optax-visible, § 3).
- `../tech_notes/gig_eta_to_theta.md` — η-rescaling + multistart; brief.
- `../archive/design/solver_redesign.md` § 2.3, § 6.2 — paramax rejection;
  natural gradient = Newton on Bregman.
- `../archive/plans/migration_plan.md` — no fitting-method discussion.
- `../references/distribution_packages.md`, `../references/gpjax_review.md`
  — ecosystem context (FlowJAX/efax/GPJax fitting styles).
- `docs/design/`, `docs/theory/` — no rationale published (gap).
- Git history: commits `5c679e8`, `485282e`, `0681fa2` (+parent),
  `b391543`, `4096d77`, `906a4d6`; pickaxe for "natural gradient",
  "gradient descent", "optax", "fit_mle".
- Chat transcripts: `ea8048c3` (2026-03-08, paramax + GIG optimization
  tests — primary empirical source), `622b86d9` (2025-11-18,
  natural-gradient sketch; transcript not in this project's folder,
  located via conversation search).
- `normix/` source + `benchmarks/` — no gradient-descent code or baseline.
