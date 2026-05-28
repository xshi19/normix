# pstack Skills: Survey and Fit Assessment for normix

**Date:** 2026-05-25
**Plugin source:** [`cursor/plugins/pstack`](https://github.com/cursor/plugins/tree/main/pstack)
**Author of pstack:** poteto (React Compiler core team, ex-Meta/Netflix/Cursor)

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
| **Adopt** | Use as shipped | `principle-laziness-protocol`, `principle-foundational-thinking`, `principle-redesign-from-first-principles`, `principle-subtract-before-you-add`, `principle-minimize-reader-load`, `principle-type-system-discipline`, `principle-prove-it-works`, `principle-fix-root-causes`, `principle-guard-the-context-window`, `principle-never-block-on-the-human`, `principle-encode-lessons-in-structure`, `principle-exhaust-the-design-space`, `principle-boundary-discipline`, `unslop` (with two carve-outs) |
| **Adopt with adaptation** | Useful but needs a normix-specific wrapper or trimmed scope | `architect`, `arena`, `how`, `tdd`, `interrogate`, `reflect`, `principle-outcome-oriented-execution`, `principle-migrate-callers-then-delete-legacy-apis` |
| **Use rarely / on demand** | Right tool for a narrow class of problem | `figure-it-out`, `show-me-your-work`, `principle-make-operations-idempotent`, `principle-separate-before-serializing-shared-state` |
| **Skip** | Wrong target audience or relies on missing infrastructure | `poteto-mode` as default router, `automate-me`, `why`, `typescript-best-practices`, `principle-experience-first`, `poteto-agent` subagent |

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

## 8. References

- pstack source: [`cursor/plugins/pstack`](https://github.com/cursor/plugins/tree/main/pstack)
- pstack marketplace page: [cursor.com/marketplace/cursor/pstack](https://cursor.com/marketplace/cursor/pstack)
- normix philosophy: `../design/design.md` §Philosophy.
- normix agent-instruction design: `../design/agent_instructions_design.md`.
- Existing normix skills: `.cursor/skills/`.
- Existing normix rules: `.cursor/rules/`.
