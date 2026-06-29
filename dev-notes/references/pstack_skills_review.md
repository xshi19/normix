# pstack Skills: Survey and Fit Assessment for normix

**Date:** 2026-05-25 · **Revised:** 2026-06-28 (Part II added)
**Plugin source:** [`cursor/plugins/pstack`](https://github.com/cursor/plugins/tree/main/pstack)
**Author of pstack:** poteto (lauren / @poteto — React Compiler core team, ex-Meta/Netflix/Cursor)

> **Part I (§1–8)** is the original skill-by-skill fit survey.
> **Part II (§9–15)** adds learnings from poteto's *Loops You Can Trust*
> article, the new pstack skills / principles / playbooks / `benny` automation
> pack, and **how Cursor implements the Slack "control room" and the
> screenshot/video verification** poteto relies on — with concrete
> recommendations for loops, multi-agent, and human-reviewer efficiency on
> normix. If you only read one part, read Part II.

---

## 1. Scope

`pstack` is a Cursor plugin shipping one subagent (`poteto-agent`),
twelve workflow skills, and eighteen single-principle skills, plus the
`poteto-mode` router. The plugin's stated goal is "write less, but
higher quality code" via rigorous, parallelizable engineering
playbooks.

This document answers three questions per skill:

1. **What it does.**
2. **Does it fit normix?** (against the design philosophy in
   `../design/design.md`: Elegance > Numerical efficiency &
   robustness > Mathematical clarity > Simplicity. Cross-checked
   against the four existing normix skills:
   `agent-maintenance`, `git-conventions`, `running-tests`,
   `docs-publish`.)
3. **A concrete normix-flavoured example** and the with/without
   delta.

**Caveat.** pstack was authored for a multi-developer product
codebase with rich MCP integration (Linear, Slack, Datadog, Sentry,
Snowflake). normix is a single-author scientific-computing library
with no such MCPs. That mismatch drives several of the "skip" /
"trim" verdicts below.

---

## 2. Verdict Summary

| Verdict | Meaning | Skills |
|---|---|---|
| **Adopt** | Use as shipped | `principle-laziness-protocol`, `principle-foundational-thinking`, `principle-redesign-from-first-principles`, `principle-subtract-before-you-add`, `principle-minimize-reader-load`, `principle-type-system-discipline`, `principle-prove-it-works`, `principle-fix-root-causes`, `principle-guard-the-context-window`, `principle-never-block-on-the-human`, `principle-encode-lessons-in-structure`, `principle-exhaust-the-design-space`, `principle-boundary-discipline`, `unslop` (with two carve-outs), `principle-build-the-lever` (new — §10), `principle-sequence-verifiable-units` (new — §10) |
| **Adopt with adaptation** | Useful but needs a normix-specific wrapper or trimmed scope | `architect`, `arena`, `how`, `tdd`, `interrogate`, `reflect`, `principle-outcome-oriented-execution`, `principle-migrate-callers-then-delete-legacy-apis`, `blast-radius` (new — §10) |
| **Use rarely / on demand** | Right tool for a narrow class of problem | `figure-it-out`, `show-me-your-work` (now the backbone of any loop — §12–13), `recall` (new — §10), `principle-make-operations-idempotent`, `principle-separate-before-serializing-shared-state` |
| **Skip / config-only** | Wrong target audience, relies on missing infrastructure, or one-time setup | `poteto-mode` as default router, `automate-me`, `why`, `typescript-best-practices`, `principle-experience-first`, `poteto-agent` subagent; `setup-pstack` (one-time model config — §10) |

---

## 3. Workflow Skills

### 3.1 `poteto-mode` — the umbrella router

- **What it does.** Master entry point. Reads request, picks one of
  twelve playbooks (Investigation, Bug fix, Perf, Runtime forensics,
  Feature, Prototype, Visual parity, Authoring a skill, Eval,
  Autonomous run, Multi-phase, Opening a PR), copies its steps
  verbatim into a todolist, then calls the leaf skills (`how`,
  `architect`, `arena`, `unslop`, …) at the prescribed steps. Also
  imposes prose style: short declarative sentences, em-dash banned,
  mid-sentence colon banned.
- **Fit.** Mixed. The playbook *idea* fits normix well — distribution
  additions, EM refactors, and Bessel work all have repeatable
  shapes. The *prose rules* clash with normix's design-doc voice
  (longer mathematical prose, em-dashes are normal in derivations).
  The *infra assumptions* (PR-per-task, `cursor-team-kit`'s
  `/deslop`, `control-ui`/`control-cli`, `babysit`) only partially
  apply: normix already has a PR-and-release-please workflow but
  doesn't ship `cursor-team-kit`.
- **Recommendation.** Don't make `/poteto-mode` the default entry
  point. Cherry-pick its playbook structure for the next time we
  write a normix-specific orchestrator skill (e.g.
  `add-distribution-playbook` that wires together
  `agent-maintenance` + `running-tests` + `git-conventions`).
- **Example.** Adding `MeixnerDistribution`:
  - *With pstack `/poteto-mode` Feature playbook:* opens 15+
    todos, runs `architect` (which fans out to arena across four
    models), runs `unslop` on docstrings — heavy.
  - *Without:* the existing `agent-maintenance` skill already lists
    the six update points for "New Distribution Added". Faster,
    normix-native.

### 3.2 `architect` — sketch types before implementing

- **What it does.** Five-phase flow: Ground (run `how`), Sketch (run
  `arena` to fan out 4 model candidates), Agree (default proceed),
  Implement, Scrap if friction recurs.
- **Fit.** Strong fit for **structural decisions only**: a new ABC
  layer (e.g. `DispersionModel` per design row M7), the η-update
  rule abstraction (E5–E8), Joint vs Marginal mixture shape (M1).
  Overkill for routine work (new analytical Hessian, bug fix in a
  pdf, doc edit). Aligns with normix's philosophy ("ask whether the
  underlying design should be refactored instead of patching
  around it").
- **Recommendation.** Adopt for one-way-door design changes;
  document the trigger explicitly so it isn't invoked on every
  task. Pair with normix's existing `../design/design.md`
  decision table (the rationale rows are the natural sketch
  artifact).
- **Example.** Designing M5's "canonical map on both layers;
  closed-form pytree path + Bregman fallback":
  - *With:* arena returns four shape proposals; synthesized design
    becomes a row in `design.md` and a sketch in
    `../plans/`.
  - *Without:* one model produces one shape; nuance from the
    rejected alternatives is lost.

### 3.3 `arena` — N parallel candidates, graft the winners

- **What it does.** Six-phase fan-out: Frame, Fan out N (default 4
  models), Cross-judge, Pick base, Graft from losers, Verify.
- **Fit.** Same trigger profile as `architect`: structural
  decisions where the design space is genuinely open. Costly —
  spawns five subagents per invocation. Not appropriate for
  numerical-method choices (those are decided by benchmarks, not by
  model judgment).
- **Recommendation.** Adopt as a callee of `architect`. Do not
  invoke directly for routine work. Cap usage to maybe 1–2 times
  per quarter on normix.
- **Example.** Choosing the GIG η→θ solver (decision S5):
  - *With:* four candidates (pure JAX Newton, scipy.optimize wrap,
    custom L-BFGS-B port, jaxopt LBFGSB). Cross-judge spots the
    ill-conditioning hazard. The benchmark still decides, but the
    arena widens the candidate set.
  - *Without:* we pick `jaxopt.LBFGSB` because it's familiar; the
    eta-rescaling hack is found later under benchmark pressure.

### 3.4 `interrogate` — four-model adversarial review of a diff

- **What it does.** Same prompt to four different models, each
  reviews the diff, parent synthesizes into Act / Consider /
  Noted / Dismissed.
- **Fit.** Good fit for risky numerical PRs: Bessel regime
  boundaries, EM convergence-criterion changes, new
  log-partition implementations. Lower value for doc PRs or
  obvious refactors.
- **Recommendation.** Adopt for PRs touching
  `normix/_bessel.py`, GIG / GH solvers, EM fitter, or any
  function with `@jax.custom_jvp`. Skip for docs / typing-only
  PRs.
- **Example.** Reviewing the large-z Hankel-regime patch in
  `log_kv`:
  - *With:* one model flags a sign error in the asymptotic
    series; another spots a missing `jnp.where` guard that
    would NaN at z=0; cross-validation gives high confidence.
  - *Without:* single review misses the corner case; bug
    surfaces in an EM fit weeks later.

### 3.5 `how` — codebase walkthrough / placement question

- **What it does.** Two modes: Explain (default) and Critique.
  Explain spawns 2–4 parallel explorer subagents for complex
  questions, then a synthesizer.
- **Fit.** Useful when onboarding a new agent to a subsystem
  (Joint/Marginal mixture machinery, the η-update rule layer, the
  solver chain rule). Lower marginal value than in a 1M-LoC
  codebase: normix is small enough that `Grep` + `Read` usually
  beats spawning explorer subagents.
- **Recommendation.** Adopt the Critique mode for periodic
  architecture audits. Default mode is overkill — prefer direct
  exploration for normix-sized questions.
- **Example.** "How does the φ↔θ chain rule actually flow through
  the solver?":
  - *With `how`:* three explorers (solver interface, EF base
    class, log-partition triad) feed a synthesizer producing an
    onboarding-grade explanation.
  - *Without:* agent reads `../design/exponential_family.md`
    §2 and `solvers_and_bessel.md` §1 directly. Comparable
    quality at lower token cost.

### 3.6 `why` — historical rationale across seven MCP categories

- **What it does.** Enumerates available MCPs (source control,
  issue tracker, long-form docs, real-time chat, infra
  observability, error tracking, analytics warehouse), spawns one
  investigator per category in parallel, synthesizes a
  confidence-weighted answer.
- **Fit.** **Six of the seven categories are not configured for
  normix.** Only source control (`git`, `gh`) is available. The
  remaining six would all be "skipped — no MCP" rows.
- **Recommendation.** Skip. normix already encodes design
  rationale in three first-class locations: `../design/` (why),
  `../archive/design/` (superseded why), and PR descriptions.
  The lightweight question "why did we pick X" is answered by
  the decision table in `../design/design.md` plus
  `git log -p`.
- **Example.** "Why does GIG η→θ default to CPU L-BFGS-B?":
  - *With `why`:* six "no MCP" gaps, one source-control investigator,
    a synthesizer that mostly cites `../design/design.md` rows
    S2 and S5 anyway.
  - *Without:* same answer, found in 30s by reading row S5.

### 3.7 `figure-it-out` — design a custom playbook

- **What it does.** When no bundled playbook fits, designs one:
  Frame (define done as a falsifiable predicate), Design the
  workflow, Run the loop, Audit trail, Verify and hand back.
- **Fit.** Good fit for genuinely novel work where the shape isn't
  predetermined: a new family of distributions, a fundamentally
  different solver, a multi-week refactor. Don't reach for it
  reflexively.
- **Recommendation.** Use on demand for cross-cutting work.
  Routine distribution additions are well-served by
  `agent-maintenance`.
- **Example.** Adding incremental EM for streaming data:
  - *With:* explicit predicate (likelihood-per-batch
    monotonicity within tolerance), units sequenced
    riskiest-first (sufficient-stats accumulation before
    mini-batch sequencing).
  - *Without:* agent starts coding, discovers the η→θ warm-start
    coupling halfway through, rewrites.

### 3.8 `tdd` — failing test before fix, with carve-outs

- **What it does.** Standard TDD loop but with an explicit
  "skip if the test path is unclear/expensive/integration-heavy"
  off-ramp.
- **Fit.** Strong fit. normix already has clean
  property-based and contract markers. The skill's "prefer no
  new test over a bad test" stance matches normix conventions.
- **Recommendation.** Adopt as-is for bug fixes with a clean
  test path. Reference the existing `running-tests` skill for
  marker selection.
- **Example.** Bug: `pdf(x)` returns NaN at `x=mu` for VG with
  small `alpha`:
  - *With `tdd`:* failing `pytest -k vg_pdf_at_mu`, fix the
    `log_kv` clamp, regression test stays.
  - *Without:* eyeball-test the fix, miss the property-based
    suite, regression rides on luck.

### 3.9 `unslop` — strip AI patterns from prose

- **What it does.** Thirty-one detection patterns covering
  significance inflation, AI vocabulary, em-dash overuse,
  inline-header lists, etc.
- **Fit.** Good fit for `README.md`, public docs, and PR
  descriptions. **Two carve-outs:**
  - Rule 13 (em-dash banned). normix uses em-dashes in
    mathematical asides and in the decision-table rationale
    column. Treat as guidance, not absolute.
  - Rule 14 (mid-sentence colon banned). Math docs sometimes
    use colons to introduce equation labels. Treat as guidance.
  Rules 1–12, 15–31 apply cleanly.
- **Recommendation.** Adopt with the two carve-outs documented.
  Add a note to `.cursor/skills/` referencing it.
- **Example.** Rewriting a docstring that says "leverages the
  intricate interplay of Bessel asymptotics":
  - *With:* "uses the Hankel-regime asymptotic for `log_kv`."
  - *Without:* the slop ships to PyPI and Sphinx.

### 3.10 `automate-me` — capture personal working style

- **What it does.** Mines transcripts and asks structured
  questions to draft a personal `<name>-mode` skill.
- **Fit.** This is a personal-mode capture tool. Not
  appropriate as a normix project skill — the project should
  not depend on one person's `-mode` skill.
- **Recommendation.** Skip at the project level. The user can
  invoke `/automate-me` once at the user level
  (`~/.cursor/skills/`) if they want a personal style skill.
- **Example.** N/A for normix.

### 3.11 `reflect` — mine transcript, route lessons to skill edits

- **What it does.** Three reviewer subagents (judgment / tooling /
  divergent) → Opus synthesizer → routed skill edits.
- **Fit.** Good fit, but normix already has the
  `agent-maintenance` skill covering the same outcome (capture a
  recurring failure mode → update the right rule/skill). pstack's
  `reflect` adds the multi-reviewer fan-out, which is novel.
- **Recommendation.** Adopt as the front-end to
  `agent-maintenance`. After a long task, run `reflect` first to
  surface candidates, then funnel approved items into
  `agent-maintenance`'s update points.
- **Example.** After a multi-day GH refactor:
  - *With:* three reviewers spot that "we kept forgetting to
    update `project-overview.mdc` after adding a distribution",
    which becomes a row in the agent-maintenance triggers.
  - *Without:* lesson lives only in the user's head until the
    next time they remember to update it.

### 3.12 `show-me-your-work` — TSV decision log for autonomous runs

- **What it does.** Append-only TSV (`ts | phase | decision | why |
  evidence | result`) for runs the human reviews after the fact.
- **Fit.** Limited fit. normix's typical session is short and
  reviewed live. Useful for the rare overnight benchmark sweep
  or multi-day refactor.
- **Recommendation.** Use on demand for long autonomous runs only.
  Most normix work doesn't need it.
- **Example.** Benchmarking GH-path implementations across 50
  parameter combinations:
  - *With:* TSV row per parameter set, committed alongside
    benchmark output for reproducibility.
  - *Without:* benchmark log is markdown free-text; harder for
    a reviewer to grep.

### 3.13 `typescript-best-practices`

- **What it does.** TypeScript-specific grounding for
  `type-system-discipline`.
- **Fit.** normix has zero TypeScript. Skip.
- **Recommendation.** Skip.

---

## 4. The `poteto-agent` Subagent

- **What it is.** A `subagent_type: poteto-agent` value. When
  spawned, the subagent reads `poteto-mode/SKILL.md` in full
  before doing any work.
- **Fit.** Inherits the same `poteto-mode` mismatch (prose rules,
  missing MCPs, product-oriented playbooks). For normix, a
  `generalPurpose` subagent that reads
  `.cursor/skills/agent-maintenance/SKILL.md` and the relevant
  rule file is sufficient.
- **Recommendation.** Skip. If we want a normix-flavoured
  subagent later, author one that points at our existing skill
  set.

---

## 5. Principle Skills (the eighteen leaves)

These are short, one-page rules. The `poteto-mode` index
references each. Even without adopting the router, the leaves
read as a high-quality principles library that can be cited from
normix code review.

| Principle | Verdict | Why for normix |
|---|---|---|
| `laziness-protocol` | Adopt | Mirrors design row F10 ("module-level functions forbidden; keep the interface on the class") and the Simplicity priority. |
| `foundational-thinking` | Adopt | Mirrors the Elegance priority ("Think in high-level abstractions — modules, base classes, object hierarchies"). |
| `redesign-from-first-principles` | Adopt | Mirrors normix's "refactor instead of patch" stance from `../design/design.md`. |
| `subtract-before-you-add` | Adopt | Mirrors Simplicity ("Removing something for equal-or-better results is a simplification win"). |
| `minimize-reader-load` | Adopt | Mirrors Elegance ("Reading and using normix should be enjoyable"). |
| `outcome-oriented-execution` | Adopt with adaptation | Useful for `../plans/` migration plans. The "intermediate breakage is acceptable when planned" rule needs the carve-out that **EM convergence tests must stay green** between phases. |
| `experience-first` | Skip | Product/UX-oriented. normix is a library; user delight comes via mathematical clarity and numerical correctness, not transitions and spacing. |
| `exhaust-the-design-space` | Adopt | Pairs with `architect` and `arena`. Aligns with normix's preference for "ask whether the underlying design should be refactored". |
| `boundary-discipline` | Adopt | Maps directly onto normix's "constructors clamp inputs; internal code trusts them" pattern (decision row F6). |
| `type-system-discipline` | Adopt | normix uses `eqx.Module` (frozen dataclass) and JAX pytrees; this principle is the philosophical companion. Applies even though Python's type checker is weaker. |
| `make-operations-idempotent` | Use rarely | Only relevant for `benchmarks/run_all.py` resumption and the docs publish script. Most normix code is pure-functional and naturally idempotent. |
| `migrate-callers-then-delete-legacy-apis` | Adopt with adaptation | Mirrors decision row E4 ("BatchEMFitter + IncrementalEMFitter replaces obsolete OnlineEMFitter / MiniBatchEMFitter"). But normix is pre-1.0 and on PyPI; "delete in the same wave" still needs a `DeprecationWarning` cycle for any public API. |
| `separate-before-serializing-shared-state` | Use rarely | Only relevant for benchmark worktrees or multi-agent runs. Most normix code has no shared mutable state. |
| `prove-it-works` | Adopt | Mirrors normix's contract-test culture and the "EM convergence numerical predicate" mindset. |
| `fix-root-causes` | Adopt | Mirrors normix's debugging culture (Bessel regime boundary bugs, η→θ ill-conditioning are root-cause work). |
| `guard-the-context-window` | Adopt | Already implicit in `agent-maintenance` ("`project-overview.mdc` is always loaded. Every line costs context"). The pstack version states it cleanly. |
| `never-block-on-the-human` | Adopt | normix changes are reversible (PR + tests + release-please). Reflexive permission-asking is wasteful. |
| `encode-lessons-in-structure` | Adopt | Mirrors normix's preference for constants-in-`utils/constants.py`, ruff config, pre-commit hooks over textual reminders. |

---

## 6. Overlap and Duplication Map

| Existing normix skill / rule | pstack overlap | Recommendation |
|---|---|---|
| `agent-maintenance` (triggers for updating docs/rules) | `reflect` (mines lessons), `figure-it-out` (designs custom playbook) | Keep `agent-maintenance` as the canonical update list. Front it with `reflect` after long tasks. |
| `git-conventions` (conventional commits, pre-commit checklist) | `poteto-mode` "Opening a PR" playbook | Keep `git-conventions`. It already encodes the release-please workflow which pstack doesn't know about. |
| `running-tests` (markers, profiling recipes) | `tdd` (test-first loop) | Both compose cleanly. `tdd` cites `running-tests` for marker selection. |
| `docs-publish` (Sphinx build + gh-pages) | None | No overlap. Keep. |
| `.cursor/rules/maintain-skills.mdc` (skill structure) | `poteto-mode` "authoring-a-skill" playbook + Cursor's `create-skill` | The normix rule is the canonical structure. `create-skill` is fine to use; the pstack playbook adds an evaluation harness step that's overkill for normix. |
| `.cursor/rules/coding-conventions.mdc` | `principle-*` (especially `boundary-discipline`, `type-system-discipline`) | Principles complement coding-conventions; they don't duplicate. |

**No existing normix skill duplicates `architect`, `arena`,
`interrogate`, or the principle library.** These are the
highest-value additions.

---

## 7. Concrete Recommendations

### 7.1 Adopt without ceremony

The principle leaves (minus `experience-first`,
`make-operations-idempotent`, `separate-before-serializing-shared-state`)
and `unslop` (with the two carve-outs) read as high-quality,
project-agnostic rules. Cite them from normix code review and PR
descriptions. No wrapper needed.

### 7.2 Adopt for narrow triggers

- **`architect`** — when a design row in `../design/design.md`
  is about to be added (one-way-door).
- **`arena`** — only as a callee of `architect`, capped at 1–2
  invocations per quarter.
- **`interrogate`** — for PRs touching `_bessel.py`, GIG/GH
  solvers, EM fitter, or `@jax.custom_jvp` functions.
- **`tdd`** — for bug fixes with a clean test path.
- **`reflect`** — after long autonomous runs, as a front-end to
  `agent-maintenance`.
- **`figure-it-out`** — for genuinely novel work (new family of
  distributions, fundamental solver redesign).

### 7.3 Skip

- **`poteto-mode`** as default router. (The prose style and
  product-codebase assumptions don't match normix.)
- **`automate-me`** at the project level.
- **`why`** — six of seven evidence categories have no MCP
  configured.
- **`typescript-best-practices`** — no TypeScript in normix.
- **`poteto-agent`** subagent — `generalPurpose` with a pointer
  to normix's own skills is sufficient.
- **`principle-experience-first`** — product-UX focus.
- **`show-me-your-work`** — only for long autonomous runs.

### 7.4 Open question for the maintainer

Should we author a thin normix-specific umbrella skill that plays
the same role as `poteto-mode` (route a request to one of a small
set of playbooks) but with normix's actual recurring shapes?
Candidate playbooks:

- **New distribution** → routes through `agent-maintenance`'s
  "New Distribution Added" trigger list.
- **New numerical method** → benchmark-first; cite `prove-it-works`,
  call out CPU/JAX hybrid trade-off per decision rows S8/S9.
- **Doc-only change** → run `unslop` (with carve-outs), then
  `docs-publish`.
- **EM/solver refactor** → run `architect` for structural
  changes, gate on `running-tests -m "slow or stress"`.
- **Bessel touch** → mandatory `interrogate`.

If yes, the new skill belongs in `.cursor/skills/` and follows
`.cursor/rules/maintain-skills.mdc` for structure.

---

## Part II — "Loops You Can Trust": new learnings (revised 2026-06-28)

Added after poteto published [*Loops You Can Trust*](https://x.com/poteto/article/2069824386283319343)
(local copy: `../../docs/pdfs/lauren on X_ _Loops You Can Trust_ _ X.pdf`) and
pstack grew new skills, principles, playbooks, and the `benny` automation pack.
Part I surveyed *which skills fit*. Part II answers *how to run them as loops*,
*whether multi-agent helps normix*, *how the human reviewer stays efficient*,
and the two mechanisms you asked about — the **Slack control room** and
**screenshot/video verification** — in concrete Cursor terms.

---

## 9. The thesis: verification is the limiting step

poteto frames agent management with Andy Grove's *High Output Management*
"breakfast factory" (the egg is the slow step that paces the whole plate):

- **Verification is the long pole.** Reproduction and checking take the most
  time and usually need a human. Until an agent can *verify its own work*,
  running loops "just creates compounding slop and more work for humans." Build
  verification **before** you build the loop.
- **Managerial leverage / build the lever.** A human authors a reusable
  tool or skill once; every future run inherits it. "Give me a lever long
  enough …" Now a first-class pstack principle (§10).
- **Trust, but verify → "show me your work."** "I fixed it" is not enough.
  An *artifact* you can inspect (failing + passing test, before/after plot,
  trace, screenshot) "beats a plausible sounding explanation which may be
  wrong," and means you don't replay the whole run to trust the result.
- **Catch defects at the lowest-value stage.** A failed check on one small unit
  is cheap; a regression found after a giant change is expensive. (His StyleX
  migration went from a 400K-line PR to *one leaf component per day*, each PR
  small with before/after evidence.)
- **Every stage can stop the line.** Triage may say "not a bug," repro may say
  "can't reproduce," the fixer may say "too risky." Those are *useful* outcomes:
  they keep bad work from flowing downstream where it is costlier to fix.
- **Decide whether a step needs an agent at all.** "If a script can do it
  deterministically, use the script. Agents are for the fuzzy parts: choosing
  hypotheses, interpreting evidence, deciding when something needs judgment."

His five closing rules:

1. Do the work by hand first, so you know what good looks like.
2. Give agents the same tools and signals you use.
3. Make every stage prove its work and stop when it doesn't meet the bar.
4. Read the transcripts; turn recurring failures into tools, skills, or evals.
5. Make loops autonomous only after they earn your trust.

This maps cleanly onto normix's priority order (Elegance > Numerical efficiency
& robustness > Mathematical clarity > Simplicity): "prove it works" *is* the
numerical-robustness discipline, pointed at agents instead of code.

## 10. What's new in pstack since this survey

Five skills/principles and five playbooks were added, plus the automations pack.

| New item | Kind | normix verdict |
|---|---|---|
| `principle-build-the-lever` | principle | **Adopt.** normix already half-does this: `benchmarks/utils.py::save_result` (JSON + git hash) and `benchmarks/compare.py` (before/after delta table) *are* the lever. Make "build the rerunnable tool, don't hand-do it" the explicit default for any non-trivial sweep. |
| `principle-sequence-verifiable-units` | principle | **Adopt.** "Red→green per unit, never batch the check to the end." Maps to: keep EM/solver tests green between refactor phases; stack PRs failing-test-first. |
| `blast-radius` | skill | **Adopt for narrow triggers.** Companion to `interrogate`. Its rule — *prove the one safety fact by running code, not by a convincing writeup* — matches normix's "benchmarks decide" culture. Use when a change touches the `ExponentialFamily` base, `log_kv`, or a shared solver. |
| `recall` | skill | **Use rarely / adapt.** Its shared-record sweep needs MCPs normix lacks (same gap as `why`). The chat-history mining over `agent-transcripts/` + `git`/`gh` verification is still useful for a returning solo maintainer. |
| `setup-pstack` | skill | **One-time config.** Writes `~/.cursor/rules/pstack-models.mdc` mapping each role (code, judgment, the arena/architect/interrogate panels) to a model. Run once if you adopt pstack, otherwise every panel collapses to one model and the cross-model value is lost. |
| Playbooks `hillclimb`, `trace-forensics`, `refactoring`, `session-pickup`, `pause-safely` | poteto-mode | `hillclimb` is the high-value one for normix (§12). `session-pickup` / `pause-safely` pair with `agent-transcripts/` for resuming work. |
| `benny` automation pack | automations | A reference Slack triage→repro→fix loop. Not directly usable (normix has no Slack issue stream), but its *contract* — control-adapter, stop-the-line gates, fail-closed, draft-PR-only — is the template for any normix automation (§11, §14). |

`/poteto-mode` now also advertises that it "works extremely well with Cursor's
`/loop`" for multi-hour autonomous runs.

## 11. The two mechanisms you asked about, in Cursor terms

### 11.1 Slack as the "control room" → Cursor Automations

poteto's self-running loops are **Cursor Automations**: cloud agents that fire
on a trigger, do work, and report. Slack is simply where he *watches the line
and jumps in when something looks wrong*. You configure these at
[cursor.com/automations](https://cursor.com/automations) or with the
`/automate` skill. Key facts ([docs](https://cursor.com/docs/cloud-agent/automations)):

- **Triggers:** schedule/cron, GitHub/GitLab PR events (opened, pushed, merged,
  commented, CI completed), **Slack** (new message, emoji reaction), webhook,
  Linear, Sentry, PagerDuty. The run fires when *any* trigger matches.
- **Tools:** open / comment-on PR, request reviewers, **Send to / Read Slack**,
  MCP servers, **Memories** (notes persisted across runs), and **Computer use**
  (browser + screenshots/recordings — see 11.2).
- **Execution:** always cloud agents, always Max Mode, billed as cloud-agent
  usage. Identity: Slack posts as the Cursor bot; a *private* automation opens
  PRs as *your* GitHub account.
- **Repository scope:** none / single / multi-repo.

The `benny` pack shows the discipline that makes a Slack loop *trustworthy*:
freeze the source thread coordinates; the coordinator is the **only** Slack
poster; subagents are read-only and never receive Slack credentials; every
stage can **fail closed**; **draft PRs only**. (See
`automations/benny/skills/*/SKILL.md` and `references/control-adapter.md`.)

**normix fit (honest).** normix is a solo library with **no inbound Slack issue
stream**, so the *Slack-triaged* benny shape does not apply. Your real "control
room" is the **Cursor Agents Window + chat** (optionally GitHub PR comments).
The automations that pay off for normix are **schedule- or GitHub-triggered,
not Slack** — see §14.

### 11.2 Screenshots / videos to verify → Computer use + the browser MCP

poteto's `/control-glass` launched a dev build with the **Chrome DevTools
Protocol (CDP)** enabled, so the agent could click, screenshot, record video,
throttle CPU/network, capture CPU profiles and heap snapshots — "agents need to
see what I see, and use the tools I use." Cursor exposes the same capability two
ways:

- **In cloud Automations:** the **Computer use** tool (on by default) lets the
  agent "operate a browser, produce screenshots or recordings." `benny`
  abstracts this behind a **control-adapter contract**
  (`bring up → drive real UI → inspect state read-only → screenshot → record →
  cleanup`), requires the discriminating broken state to appear **twice**, then
  has *another* agent confirm the media actually shows it.
- **Locally, in the IDE:** the **`cursor-ide-browser` MCP** (available in this
  workspace) is normix's accessible `/control-glass`: `browser_navigate`,
  `browser_take_screenshot`, `browser_click/type`, and `browser_cdp` for
  `Profiler.*`, `Performance.getMetrics`, and DOM/CSS inspection. The
  **`browser-use` subagent** wraps it for longer interaction sequences.

**normix analog — what "show me the artifact" means for a numerical library.**
normix has no UI, so the verifiable artifact is a **plot or a numeric table**,
not a web-page screenshot:

- **matplotlib PNGs** via the existing `normix.utils.plotting` harness
  (`plot_pdf_cdf_comparison`, `plot_mle_fit`, `plot_em_convergence`, `savefig`
  at dpi 200). The agent saves the figure and **embeds it inline in chat** with
  `![](path)` — you see the convergence curve / PDF overlay directly, exactly
  like a before/after screenshot.
- **Benchmark JSON + `compare.py`** before/after delta tables — already a
  deterministic, rerunnable lever.
- Reach for the browser MCP only to screenshot the **built Sphinx site** or a
  rendered notebook, i.e. when the artifact genuinely is a web page.

So poteto's rule — *hand back an inspectable artifact, not a claim* —
translates for normix to: **an EM-fit reply should carry the convergence plot
and the `compare.py` delta, not the sentence "it converged."**

## 12. Do loops and multi-agent help normix?

Yes, for a specific shape of work — and the project is already half-built for it.

**Loops (`/loop` + the `autonomous-run` / `hillclimb` playbooks).**

- **High value:** numeric *hillclimbing* — drive a benchmark metric toward a
  target (GIG/GH solver iterations or wall-time, Bessel accuracy vs speed, EM
  iterations-to-converge). The discipline is exactly normix's own: *one change,
  one measurement against the frozen `compare.py` harness, keep or revert; the
  data decides, never code inspection.* Also good for overnight benchmark sweeps
  and doc link/build-fixing loops.
- **Requirement:** a **checkable predicate** ("≥ X% faster *and* ≥ N iterations,
  with `slow`/`stress` tests still green"). A vague goal makes a loop spin.
- Pair every loop with `show-me-your-work`: an append-only decision log plus an
  end-of-run **cross-model audit** (a subagent from a *different* model family
  reviews the trail and flags weak evidence / skipped checks).

**Multi-agent.**

- **`best-of-n-runner` (git worktrees)** is the normix-native version of
  poteto's worktree parallelism. Use it for solver/algorithm bake-offs (e.g. the
  GIG η→θ candidates in §3.3) where each attempt needs an isolated checkout.
- **`arena` / `architect`** — one-way-door *design* only (Part I verdicts hold).
- **`interrogate` / `blast-radius`** — adversarial review of Bessel/solver/EM
  diffs.
- **Not** for routine work: normix is small enough that `Grep`+`Read` beats
  spawning explorers, and numerical choices are settled by benchmarks, not by
  model vote. Every fan-out spawns N subagents (real token cost) — gate by an
  explicit trigger, don't reach for it reflexively.

## 13. Making *you* (the human reviewer) more efficient

The highest-leverage part of the article for a solo maintainer. Concrete habits:

1. **Demand artifacts, not prose.** Make your standard ask: *"show me the
   failing test and the passing test; the EM convergence plot; the `compare.py`
   delta vs the last result."* You inspect the artifact in seconds instead of
   re-deriving the claim. Bake the recurring asks into a skill so agents produce
   them unprompted.
2. **Front-load a predicate, then review the predicate — not the transcript.**
   State "done = …" as a number or a test before the agent starts. A
   self-terminating loop hands you pass/fail, not a wall of text.
3. **Let reversible work run; reserve attention for one-way doors.** normix
   changes are reversible (PR + tests + release-please), so
   `never-block-on-the-human` applies — stop interrupting for permission. Spend
   the saved attention on design forks (use **Plan mode** to discuss, or
   `architect`/`arena` for genuine one-way doors).
4. **Read thinking blocks to find failure modes, then codify them.** poteto:
   that is how he decides *whether a step needs an agent at all*. When you see a
   mistake twice, route it through `reflect` → `agent-maintenance` into a rule, a
   constant, or a lint — not a one-off correction.
5. **"Do it by hand once, then build the lever."** For repetitive work (adding a
   distribution, a parameter sweep) do the first unit yourself to learn the
   recipe, then have the agent write the script/checker and review *that*. The
   lever reruns for free; a hand-done change can only be re-checked by redoing it.
6. **Stop-the-line, honestly.** Wire normix loops so a stage *refuses* rather
   than fudges: a benchmark regression beyond tolerance **stops and reports** —
   it must never "fix" the metric by relaxing the predicate or weakening a test.
7. **For overnight runs, commit the decision log.** `show-me-your-work` gives a
   one-row-per-decision TSV you skim instead of replaying the session, plus a
   cross-model "Attention" section.

Cursor skills already on this machine that support the above: **`/automate`**
(create an Automation), **`/loop`** (run a prompt on an interval or until a
condition), **`babysit`** (keep a PR merge-ready).

## 14. Concrete next steps for normix

Prioritized, lowest-effort first:

1. **Adopt three principles by citation** (no new infra): `build-the-lever`,
   `sequence-verifiable-units`, `prove-it-works`. Cite them in review/PRs;
   normix already embodies the first via `compare.py`.
2. **Add a "verification artifact" convention** to a rule or a small
   `verify-results` skill: *any EM/solver/benchmark change reply embeds the
   relevant `normix.utils.plotting` figure and/or the `compare.py` delta.* This
   is the single highest-leverage change for *your* review time.
3. **One scheduled Automation — the normix-shaped `benny`:** nightly cron runs
   `benchmarks/run_all.py`, then `compare.py` against the last committed result;
   if any metric regresses beyond a threshold, it **opens a GitHub issue / PR
   comment** with the delta table (and a plot). Schedule + GitHub triggered,
   reported to GitHub, no Slack needed — fail-closed and draft-only, per benny.
4. **One GitHub-triggered Automation (optional):** on a PR touching
   `normix/utils/bessel.py`, the GIG/GH solvers, or the EM fitter, run
   `interrogate` and post the synthesis as a PR comment (the §3.4 trigger).
5. **`/loop` a hillclimb** next time a solver/Bessel path is too slow or
   imprecise: frozen `compare.py` harness, a numeric predicate, a decision log.
6. **Run `setup-pstack` once** if you adopt pstack, so the review panels use
   distinct model families (the cross-model value depends on it).
7. **Revisit the §7.4 umbrella-skill question with a loop lens:** the candidate
   "New numerical method → benchmark-first" playbook is now concretely
   `hillclimb` + `build-the-lever` + `show-me-your-work`.

---

## 15. References

- pstack source: [`cursor/plugins/pstack`](https://github.com/cursor/plugins/tree/main/pstack)
- pstack marketplace page: [cursor.com/marketplace/cursor/pstack](https://cursor.com/marketplace/cursor/pstack)
- poteto, *Loops You Can Trust* (2026-06-24): [x.com/poteto/article/2069824386283319343](https://x.com/poteto/article/2069824386283319343); local copy `../../docs/pdfs/lauren on X_ _Loops You Can Trust_ _ X.pdf`.
- Cursor Automations docs: [cursor.com/docs/cloud-agent/automations](https://cursor.com/docs/cloud-agent/automations).
- `benny` automation pack: [`cursor/plugins/pstack/automations/benny`](https://github.com/cursor/plugins/tree/main/pstack/automations/benny) and its [`FOR_AGENTS.md`](https://github.com/cursor/plugins/blob/main/pstack/automations/benny/FOR_AGENTS.md).
- New pstack skills referenced in Part II: `blast-radius`, `recall`, `setup-pstack`, `principle-build-the-lever`, `principle-sequence-verifiable-units`, playbook `hillclimb`.
- normix verification surfaces: `benchmarks/utils.py`, `benchmarks/compare.py`, `normix/utils/plotting.py`.
- normix philosophy: `../design/design.md` §Philosophy.
- normix agent-instruction design: `../design/agent_instructions_design.md`.
- Existing normix skills: `.cursor/skills/`.
- Existing normix rules: `.cursor/rules/`.
