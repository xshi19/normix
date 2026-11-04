---
name: normix-formal-verify
description: >-
  Prove one normix claim in formal/lean, translate a function with the
  xshi19/jax-lean fork, or update formal/lean/coverage.toml. Use when
  choosing the next formal target, marking a claim blocked, checking a
  Lean proof, or recording partial coverage after the EF-04-var pilot.
---

# Formal verification

Prove one feasible slice of one claim.
Record every excluded layer as `blocked` or `not covered`.

**Why:** The EF-04-var pilot proved one exact-real identity and left translation, rounding, and the bridge blocked. A green Python test is not a universal theorem. A green `lake build` does not reject `sorry`. The fork and this repository pin different JAX and Lean versions. Forcing a missing translation or a missing Mathlib lemma hides the gap.

Policy, the pilot record, and triage live in `formal/lean/decisions.md`.
The inventory lives in `dev-notes/research/formal_verification/05_math_checklist.md`.
Do not treat the roadmap sample workflow as installed CI.

## Pick the next target

**Pattern:**

- Read `formal/lean/coverage.toml` and `dev-notes/research/formal_verification/05_math_checklist.md`.
- Pick one checklist row that you can state on its own.
- Read its card in `formal/lean/StatementCards.md` when that card exists.
- Prefer a row that already has a Lean theorem.
- Write the statement card before any new proof.
- Set the claim to `manual` for a hand-written formula.
- Set the claim to `jax-lean` for a traced-function certificate.
- Accept a small covered part.
- Do not wait for a full-family proof.
- Keep `EF-04-var-bridge` blocked until the fork certifies actual `Gamma.var`.
- Vary both parameters in that future certificate.
- Leave distributional moments, rounding, and approximation bounds for later claims.
- Skip Bessel, EM, and `custom_jvp` until a smaller slice is recorded.
- Use checklist statuses only in the checklist.
- Do not copy those statuses into `coverage.toml`.
- Leave a compiling theorem `blocked` until correspondence review and real hashes exist.
- The step is done when one claim id and its excluded layers are named.

## Translate with jax-lean

Study translation stays in the fork [xshi19/jax-lean](https://github.com/xshi19/jax-lean) of [srush/jax-lean](https://github.com/srush/jax-lean).
Do not vendor fork sources into normix.
Reuse permission is still open.

Read the fork revision you trace.
The audited `main` tip `570aed6` pins JAX `0.8.0` and Lean `v4.33.0`.
Read `formal/lean/lean-toolchain` and `formal/lean/lake-manifest.json` for the normix pins.
Do not downgrade normix to match the fork.

**Pattern:**

- Trace with `jax.make_jaxpr` on the real function.
- Call `transpile` for a readable Lean function.
- Call `certify` on that `ClosedJaxpr` for one program.
- Call `certify_module` for a pure `jit` graph with one result per call.
- Import them from `jaxlean`.
- Treat a translation and a certificate as different results.
- Check the emitted file with Lean before you trust it.
- Put a new example under `examples/<name>/` with `code.py`, `generate.py`, `generated/`, and `proofs/`.
- Run `python -m examples.generate` in the fork.
- Run `python -m examples.generate --check` before you trust those files.
- Build application proofs with `lake build JaxLeanExamples`.
- `lake build` checks only the `JaxLean` library.
- Run `make check` for freshness, both Lake targets, and the fast Python tests.
- Read `SUPPORTED_JAXPR.md` in the fork before you add a primitive.
- Stop when `translate` raises `TranslationError`.
- A JAX version other than `0.8.0` is one such stop.
- The certificate path rejects `scan`, `custom_jvp_call`, captured float arrays, and `reduce_prod`.
- It also rejects multiple outputs, `while`, and `cond`.
- Stop on `no certificate rule` or `captured floating arrays are not yet certified`.
- Do not add a certificate rule inside normix.
- Ordinary translation may unroll `scan` to `max_scan_length` (default 32).
- That unroll is not a loop certificate.
- A trace that needs the normix JAX graph is blocked when the fork cannot import it.
- Keep generated fork text out of `formal/lean/` until a bridge theorem checks it.
- Do not add `Normix/Bridge/GammaVar.lean` until that certificate exists.
- The step is done when Lean accepts the certificate, or the claim records the fork error.

```python
from jaxlean import certify, certify_module, transpile

text = transpile(fn, *example_args, name="gammaVar", namespace="JaxLean.Generated")
closed = jax.make_jaxpr(fn)(*example_args)
obligation = certify(closed, name="gammaVar", namespace="JaxLean.Generated")
```

## Write and check the proof

Hand-written proofs live under `formal/lean/` and follow the EF-04-var shape in `Normix/Distributions/GammaParameters.lean`.
`gammaVar` is an independent real formula.
It is not "whatever `Gamma.var` returns."

**Pattern:**

- Model the formula with an independent real definition.
- Put the domain in the theorem statement.
- Use UpperCamelCase modules under `Normix/`.
- Use lowerCamelCase definitions and snake_case theorems.
- Keep the package directory `formal/lean/` in lowercase.
- Start a library file with `module` and one namespace.
- Use `public import` and `@[expose] public section`.
- Add every production module to `Normix.lean` with `public import`.
- Import reals from `Mathlib.Basic.Real.Basic`.
- Keep `sorry`, `axiom`, `native_decide`, `unsafe`, and `partial` out of exported modules.
- Add one `#print axioms` line in `AxiomAudit.lean` for each new theorem.
- Run `lake exe cache get` in `formal/lean`.
- Run `lake build` in `formal/lean`.
- Run `lake env lean AxiomAudit.lean` in `formal/lean`.
- Accept only `propext`, `Classical.choice`, and `Quot.sound`.
- Read the printed axioms.
- A successful build does not apply this allowlist.
- Change `StatementCards.md` when the statement changes.
- Leave human review pending until a person finishes that card.
- Add a Hypothesis check in `tests/test_formal_properties.py` when an independent oracle exists.
- Call the public Python API in that test.
- Record dtype, domain, and tolerance beside the claim.
- Do not copy the EF-04-var numbers onto another claim.
- Follow `.agents/skills/running-tests/SKILL.md` for pytest.
- Follow `.cursor/rules/testing-guidelines.mdc` for tolerances.
- Run `git checkout uv.lock` after every `uv run`.
- The step is done when `lake build` passes and the audit lists only the three allowed axioms.

## Mark a claim blocked

**Pattern:**

- Set `status = "blocked"` when the fork rejects the program.
- Do the same when Mathlib has no lemma you can cite.
- Write the missing fact in `blocked_reason`.
- Name the primitive, the certificate exclusion, or the Mathlib gap.
- Keep `PLACEHOLDER:UNREVIEWED` hashes on that claim.
- Do not run `--update` to clear the alert.
- Keep a hand-written partial theorem on `sync_path = "manual"`.
- Do not label that file a translation certificate.
- Do not add `sorry` to force a green build.
- Do not weaken the statement so the proof goes through.
- Report sync state `blocked` when no counterexample shows a code bug.
- Report `bug` only with a replayed counterexample on the real entry point.
- The step is done when `blocked_reason` names the gap and the hashes stay placeholders.

Leave `program_translation_certificate`, `approximation_error_bound`, and `floating_point_guarantee` as `not covered` until checked.

| Field | Layer |
|---|---|
| `math_theorem` | Checked proof for the written scope |
| `program_translation_certificate` | Fork certificate that Lean accepted |
| `approximation_error_bound` | Explicit error bound |
| `floating_point_guarantee` | Machine rounding |

## Record coverage

`formal/lean/check_coverage.py` is the checker.
Keep `schema_version = 2` and `hash_scope = "function-source-sha256"`.
The hash covers one normalized function: decorators, signature, docstring, comments, and body.
Edits elsewhere in the module do not change it.
Helpers, constructors, constants, and tool pins stay outside the hash.
Review those by hand.

**Pattern:**

- Give each theorem its own `[[claim]]`.
- Repeat a Python target on every claim that uses it.
- Point `targets` at dotted functions, as in `normix.distributions.gamma.Gamma.var`.
- Set `sync_path` to `manual` or `jax-lean` for the path you used.
- Set `status` to `proved` only for the scope you checked.
- Require a checked proof and a clean axiom audit for that scope.
- Write the review you did in `spec_review` and `assurance.math_theorem`.
- Keep human statement-card review pending in `StatementCards.md`.
- Use `tested` for an executed Hypothesis check with no Lean proof.
- Never change `tested` to `proved` for that reason.
- Name an existing test in `fallback_hypothesis_test`.
- Keep the `PLANNED:` prefix until that test exists.
- Run `uv run python formal/lean/check_coverage.py --update --claim ID` only after you review that claim.
- Repeat `--claim` for several claims.
- Without `--claim`, `--update` rewrites every non-blocked claim.
- Pass `--claim` only together with `--update`.
- Run `uv run python formal/lean/check_coverage.py` after the update.
- The checker always scans every claim.
- Let the script replace quoted hashes.
- Do not edit `status`, `spec_review`, or `source_commit` with it.
- Advance `source_commit` yourself after correspondence review.
- Run `git checkout uv.lock` after the checker.
- The step is done when `check_coverage.py` exits 0.

## CI

**Pattern:**

- Expect `.github/workflows/ci.yml` to run `uv run pytest tests/` on Python 3.12.
- That job includes `tests/test_formal_properties.py` and `tests/test_coverage_check.py`.
- Do not treat a green pytest job as a Lean proof.
- Do not treat it as a hash check.
- No workflow runs `lake build`, `AxiomAudit.lean`, or `check_coverage.py`.
- Ignore the sample Lean workflow in the roadmap.
- That sample still requires `defining-module-sha256`, and it is not installed.

## Gotchas

- A hash mismatch asks for review.
- It does not by itself prove a code bug.
- The checker skips blocked placeholders.
- The checker checks a real hash on a blocked claim.
- `EF-04-var` is `proved` for the exact-real identity only.
- Its bridge stays `blocked`.
- `proved` on that claim leaves human card review pending.
- Dryer scans may report duplication.
- Do not deduplicate a listed formula.
- See `formal/lean/duplication_allowlist.toml`.
- `Scratch/` may hold at most five proof holes.
- Leave `Scratch/` unimported.
- That directory does not exist yet.
- Write cards, reasons, and status text in English.
- Omit personal names and email addresses from those records.
