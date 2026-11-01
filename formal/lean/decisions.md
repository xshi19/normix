# W2 M0 decisions: build baseline, dependency audit, axiom policy

Roadmap reference: [W2 M0 and remaining M1](../../dev-notes/research/formal_verification/04_integrated_roadmap.md#w2-hand-written-lean-mathematics-under-formallean). W2 M0 implementation merged in PR #161 (`80073f3`): 31 exported theorems, three modules, and 31 audit entries. Statement review remains pending for all six cards; **merged** is separate from **statement-reviewed**.

Historical planning baseline: `51e053c57237a29c71b56d3a2a86988277477bd9`. Current source correspondence: master `d557399905c533b3f15825941be6f04a6e897584`. Build results below are the merged record, not a new execution.

Assurance labels for this baseline: **math theorem** — checked, statement review pending; **program-translation certificate**, **approximation-error bound**, **floating-point guarantee** — not covered. Acceptance records must state all four separately. AI edits proofs only under human-reviewed statements; specification changes require visible revisions and renewed review.

## Pinned environment

| Item | Value |
|---|---|
| Lean toolchain | `leanprover/lean4:v4.35.0` (`lean-toolchain`) |
| Mathlib | tag `v4.35.0`, commit `56a005855ea17e0b0958f73ec801ae140b0a1679` (`lakefile.toml`, `lake-manifest.json`) |
| Transitive packages | plausible, LeanSearchClient, importGraph, proofwidgets, aesop, Qq, batteries, Cli (versions in `lake-manifest.json`) |
| Platform tested | Linux x86_64, `elan` 4.2.4 installed without root |
| Cache | `lake exe cache get` retrieved 9031 Mathlib artifacts for this tag; no Mathlib compilation was needed |

Reproduce:

```bash
cd formal/lean
lake exe cache get
lake build
lake env lean AxiomAudit.lean
```

`lake build` builds the `Normix` library, whose root module `Normix.lean` imports every production module. `AxiomAudit.lean` is not part of the library and must be run explicitly. No Lean CI workflow is installed by this update. The [Lean CI plan](../../dev-notes/research/formal_verification/04_integrated_roadmap.md#lean-ci-plan) contains the proposed workflow and later bridge regeneration check.

## Mathlib dependency audit

Mathlib at this tag uses the module system. Library files start with `module`, use `public import` for re-exported dependencies, and declare `@[expose] public section`. The legacy path `Mathlib.Data.Real.Basic` is deprecated; use `Mathlib.Basic.Real.Basic`.

Library facts the proofs rely on (all resolved by the pinned build):

| Need | Mathlib source |
|---|---|
| Real arithmetic, `max`, order | `Mathlib.Basic.Real.Basic`, `Mathlib.Tactic.{Ring,Linarith,FieldSimp}` |
| Finite sums and sum exchange | `Mathlib.Algebra.BigOperators.Field` (`Finset.sum_comm`, `Finset.sum_mul_sum`, `Finset.sum_pos`) |
| Exponential and logarithm | `Real.exp_pos`, `Real.exp_sub`, `Real.exp_log` via `Mathlib.Analysis.SpecialFunctions.Log.Deriv` |
| Derivative of a finite sum and of `log` | `HasDerivAt.fun_sum`, `HasDerivAt.log`, `hasDerivAt_mul_const` |
| Positive semidefinite matrices | `Matrix.PosSemidef.of_dotProduct_mulVec_nonneg`, `Matrix.isHermitian_iff_isSymm` (`Mathlib.LinearAlgebra.Matrix.PosDef`) |

Not used in this milestone and still to verify before M2-M4: Gamma integral and `Real.Gamma` shifted-shape lemmas, integrability and differentiation under the integral, measure-theoretic Gaussian integrals, and any Bessel $K_\nu$ definition. No Mathlib definition of $K_\nu$ was searched for here.

## Axiom policy

- Formal exported modules contain no `sorry`, no new `axiom`, no `native_decide`, no `unsafe` or `partial` definitions.
- Accepted foundational axioms: `propext`, `Classical.choice`, `Quot.sound`. Any additional axiom, including `sorryAx`, must fail CI. The current `#print axioms` command only reports dependencies; the planned CI wrapper enforces the allowlist.
- `AxiomAudit.lean` carries one `#print axioms` line per exported theorem. A new theorem must be added there in the same change.
- `Scratch/` is reserved for unfinished work, may hold at most five open proof holes during the pilot, and is never imported by `Normix`. It does not exist yet.
- An axiom list cannot detect a wrong definition or an overly strong premise; statement cards are reviewed separately (`StatementCards.md`).

## Scope of this milestone

Included: parameter identities for Gamma and GIG, normalization of finite positive log weights, the centered Gram covariance of a finite weighted family, and one fixed-geometry first derivative. All results are exact real arithmetic. They are statements about mathematical models of the Python formulas, not about floating-point execution, JAX tracing, or the generated bridge (W3).

Excluded: continuous integrals, Gamma/GIG normalization, Bessel analysis, Hessian of the finite log partition, matrix-valued statistic coordinates, floors beyond the historical GIG `max(., 0)` clamp, runtime validation, and any link to a generated jaxpr certificate.

Layout notes: Lean names follow the roadmap rules (UpperCamelCase modules and namespaces, lowerCamelCase definitions, snake_case theorems). The GIG module is `GigParameters` so that the file name equals the module path component.

## EF-06 Python correspondence update (2026-10-31)

PR #162 replaces the GIG `from_natural` clamp with `eqx.error_if` for positive
rate coordinates. Accepted inputs call `_from_natural_unchecked`; zero rates
remain accepted without proving normalizability for every boundary tuple.
No tolerance is applied. Historical clamp claims do not cover current rejection;
EF-06's strict-interior theorem excludes zero rates. Coverage points to current
master and the unclamped validity theorem. Any future certificate limited to
unchecked algebra must name that method and retain the public error-path gap.
Python rejection tests provide separate evidence.

## Duplication scanning (2026-10-31)

Accepted documentation policy, authorized by the current task; scanner execution
remains future work. Dryer-style tools may scan and report similarities. They
must never force deduplication. Explicit distribution formulas and independent
specifications can justify repetition.

[`duplication_allowlist.toml`](duplication_allowlist.toml) records each exception's
`id`, repository-relative `paths`, qualified `symbols`, `reason`, and in-repo
`source`. Later scans skip the listed fragments. A listed path does not exempt
the whole file. Review an entry when its symbols, scope, or rationale change.
Automatic extraction was rejected because structural similarity does not establish
equivalent mathematical domains or justify coupling a specification to its code.

## Many-to-many correspondence and partial coverage (2026-10-31)

Accepted documentation policy, authorized by the current task. The schema is in
[`coverage.toml`](coverage.toml); the checker and bridge remain proposed.
Each record names one claim/theorem, its fully qualified Lean name, all qualified
Python function/method `targets`, a `target_hashes` entry per target, and a
`fallback_hypothesis_test`. Targets can occur in several records. `checklist_ids`
link implementation claims such as `EF-04-var` to the existing EF-04 card.
This map is partial; it does not replace the complete theorem axiom audit.

Hashes initially cover the bytes of each target's defining Python module.
This conservative choice includes local helpers and avoids importing normix.
`PLACEHOLDER:UNREVIEWED` is not a digest or accepted baseline. `source_commit`
retains the recorded correspondence reference; advance it and the hashes only
after review. Changes flag **every** claim referencing an affected target,
including Python-only edits. Cross-module dependencies, constants, and tool pins
also require review; module hashes alone do not discover that dependency graph.

Coverage `status` is `proved`, `tested`, or `blocked`. `proved` requires a reviewed
statement, checked proof, and axiom audit for the declared scope. `tested` records
executed Hypothesis evidence only; it never promotes to `proved`. An unresolved
proof claim stays `blocked`, with any passing fallback recorded separately.
The migrated entries remain blocked pending statement review and hash baselining;
the earlier checked Lean build evidence is retained. All named fallback tests
are marked `PLANNED`, not executed. Checklist/card states remain separate.

Cover what is feasible now. Partial coverage is acceptable, with no full-coverage
quota. Missing Mathlib lemmas or jax-lean translation support leave claims blocked.
Use a fallback on the real Python entry point when an independent oracle or useful
invariant exists; otherwise record the gap. Do not weaken the statement to pass.

Two synchronization paths apply:

- `jax-lean`: auto-regenerate traceable pure-JAX functions, then rebuild translation certificates and application proofs against the fixed specification.
- `manual`: check hashes, then have a human or agent update the hand-written correspondence and proofs. Specification changes still require statement review.

`EF-04-var` maps the existing variance/mean identity to `Gamma.var` and
`Gamma.mean`. `EF-04-var-bridge` reserves the planned W3 M1 theorem
`Normix.Bridge.GammaVar.gamma_var_eq_spec`, linking actual `Gamma.var` to EF-04's
fixed $\alpha/\beta^2$ formula for $\alpha,\beta>0$. Its module does not yet exist.
A wrong variance edit flags both claims now; after bridge regeneration it must
break the application theorem. A hand-written proof does not automatically see
Python changes. Revisit the hash granularity when conservative review noise or
new dependency structure warrants a better checker, without dropping these limits.

## Synchronization and Lean failure triage (2026-10-31)

Accepted documentation policy, authorized by the current task. A Lean failure
after updating the Lean side is not automatically a code bug. Possible causes
include a real code bug that falsifies the statement, a broken or stale proof,
and a toolchain or library issue. A changed hash establishes only review work.

1. Save the previous check result, inspect changes since the correspondence baseline, and review the current statement and mapping.
2. Update the Lean side through the recorded synchronization path; retain the independent specification and rerun the relevant checks.
3. If Lean fails, investigate the statement, proof, and toolchain separately. Declare `bug` only with a concrete Hypothesis counterexample or hand computation violating the applicable statement.
4. Check the counterexample's domain, oracle, tolerance, and source correspondence. Replay generated inputs on the real entry point; record dtype/backend/device for numerical failures. Without that evidence, report `blocked`.

| Sync state | Meaning |
|---|---|
| `in sync` | Existing correspondence reviewed; relevant checks pass without verification changes. |
| `updated` | Verification material changed, correspondence reviewed, and relevant checks rerun; report the results separately. |
| `bug` | A concrete counterexample demonstrates that the current code violates the applicable statement. |
| `blocked` | Proof, toolchain, missing support, or unresolved correspondence prevents a conclusion; no established code counterexample. |

These maintenance outcomes are separate from coverage status and assurance.
`updated + bug` is possible when synchronization exposes a counterexample;
`updated` alone is not a proof claim. Do not advance the baseline while blocked
or refresh hashes merely to clear an alert. Passing fallback tests never establish
a universal theorem. Reopen a blocked result when proof/toolchain repair or new
counterexample evidence resolves its cause.
