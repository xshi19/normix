# Agent Instructions Design

> How we structure AGENTS.md, docs/, .cursor/rules/, and skills to maximize
> agent effectiveness in the normix codebase.

## Motivation

As AI coding agents (Cursor, Claude Code, Codex) become primary contributors
to normix, the quality of their output depends less on the model and more on
the **environment we give them**: what context they see, when they see it, and
how they know when they're done.

This document codifies the design principles, structure, and maintenance
discipline for our agent-facing knowledge system.

### References

| Source | Key Insight |
|---|---|
| [Harness Engineering (OpenAI, Feb 2026)](https://openai.com/index/harness-engineering/) | AGENTS.md = table of contents, not encyclopedia; progressive disclosure; enforce architecture mechanically; repository knowledge is the system of record |
| [How To Be A World-Class Agentic Engineer (nonsensee, Mar 2026)](https://nonsensee.medium.com/how-to-be-a-world-class-agentic-engineer-413783d24388) | Context is everything — give only what's needed; CLAUDE.md is a nested directory of where to find context; rules = preferences, skills = recipes; start barebones, iterate, consolidate |
| [Lessons from Building Claude Code: How We Use Skills (Anthropic, Mar 2026)](https://x.com/trq212/status/2033949937936085378) | Skills are folders, not just files; progressive disclosure via file system; gotchas sections are highest signal; don't state the obvious; don't railroad the agent |
| [pstack plugin (poteto, 2026)](https://github.com/cursor/plugins/tree/main/pstack) | Workflow skills with narrow triggers; a principles library cited by name from workflows. Adapted into normix skills 2026-07 — survey archived at `../archive/references/pstack_skills_review.md` (decision row C4). In-repo model panels from that adaptation are retired. |
| [poteto, *Loops You Can Trust* (Jun 2026)](https://x.com/poteto/article/2069824386283319343) | Verification is the limiting step — build it before the loop; demand artifacts, not claims; build the lever; loops earn autonomy only after they earn trust. Drives `../plans/loops_and_orchestration.md` |
| [humanlayer/skills (2026)](https://github.com/humanlayer/skills) | Smallest-view visual explanations (`show-me`); control-loop vocabulary (sensor / controller / actuator, memory file, PR bounding). Adapted 2026-08 — survey at `../references/humanlayer_skills_review.md` (decision row C5 in `design.md`) |

---

## Principles

### 1. Map, Not Manual

> "Give the agent a map, not a 1,000-page instruction manual."
> — OpenAI Harness Engineering

`AGENTS.md` is a **table of contents**. The length cap is in
`.cursor/rules/rule-authoring.mdc`. It tells the agent *where to look*, not
*everything it needs to know*. Deep knowledge lives
in `docs/`, `.cursor/rules/`, and skills. The agent reads deeper context only
when it needs it.

### 2. Progressive Disclosure

Context arrives in layers. Which layer loads by itself depends on the host.
See § Host support.

```
Layer 0   AGENTS.md                  Startup map on Cursor and on Codex.
Layer 1   .cursor/rules/*.mdc        Cursor: glob match, or alwaysApply.
                                     Codex: read from the AGENTS.md route.
Layer 2   dev-notes/design/*.md      Read on demand for a design decision.
Layer 3   dev-notes/tech_notes/*.md  Read on demand for a numerical problem.
Layer 4   .cursor/skills/            Cursor: description in the skill list; body on selection.
                                     Codex: description only if .agents/skills/ links the skill.
```

Each layer adds context **only when relevant**. An agent fixing a test never
loads the Bessel feasibility study. An agent writing a new distribution
never loads notebook guidelines.

### 3. Context Is Everything

> "You want to give your agents only the exact amount of information they need
> to do their tasks and nothing more."
> — nonsensee

Every rule, doc, and skill must earn its place. Redundant, contradictory, or
stale content actively degrades agent performance. When adding new content,
ask: *does this push the agent out of its default behavior in a useful way?*
If the agent would already do the right thing, don't add a rule for it.

### 4. Rules Encode Preferences, Skills Encode Recipes

**Rules** (`.cursor/rules/*.mdc`) tell the agent *what* to do and *what not*
to do — coding style, naming, numerical patterns, test structure.

**Skills** tell the agent *how* to accomplish a multi-step workflow — running
benchmarks, adding a new distribution, debugging Bessel convergence.

Keep these concerns separated. A skill should not re-state coding conventions;
it should reference the rule.

### 5. Enforce Mechanically, Not Verbally

Prefer tests and linters over prose. When a constraint matters:

1. First choice: encode it in a test (e.g. "all distributions must round-trip through natural params")
2. Second choice: encode it in a linter or CI check
3. Last resort: write it as a rule in prose

Prose rules are fragile. Tests are self-enforcing.

### 6. Single Source of Truth

Every fact appears in exactly one place. Other documents **point** to it.

| Fact | Canonical Location | Others Point To It |
|---|---|---|
| Module hierarchy | `../ARCHITECTURE.md` | `AGENTS.md`, `design.md` |
| Coding conventions | `.cursor/rules/coding-conventions.mdc` | `AGENTS.md` context map |
| Distribution catalog | `.cursor/rules/project-overview.mdc` | `AGENTS.md` |
| Numerical constants | `normix/utils/constants.py` | `.cursor/rules/coding-conventions.mdc` |
| Design rationale | `design.md` | `AGENTS.md` |
| Mathematical derivations | `../../docs/theory/*.md` | `AGENTS.md`, `../ARCHITECTURE.md` |
| Bessel implementation | `../tech_notes/bessel_*.md` | `../ARCHITECTURE.md` |
| Instruction length caps | `.cursor/rules/rule-authoring.mdc` | `maintain-agents-md.mdc`, `maintain-cursor-rules.mdc`, this file |

Duplicating facts across files leads to contradictions when one is updated
and the other isn't.

---

## Skills Architecture (2026-07, post-pstack)

Adopting the pstack skill library (survey and verdicts:
`../archive/references/pstack_skills_review.md`; decision row C4 in
`design.md`) settled three structural questions worth recording.

### Where do principles live?

The four priorities (Elegance > Numerical efficiency & robustness >
Mathematical clarity > Simplicity) stay in `AGENTS.md`; operational
principles live in **one** consolidated skill
(`.cursor/skills/principles/`).

- `AGENTS.md` is in context at startup on both hosts. In Cursor,
  `alwaysApply` rules are also in context. Moving the value hierarchy
  into a skill would make it conditional. AGENTS.md carries *what to
  optimize*; the skill carries *how to execute under it*.
- One skill, not eighteen leaves. In Cursor, every skill description
  is listed in the system prompt. Eighteen principle skills would cost
  eighteen description lines in every Cursor session.
  One description line, named sections inside (§ Prove it works,
  § Subtract before you add, …).
- No duplication: a pstack principle that restated an AGENTS.md priority
  contributed only its operational test (">3 files to trace → flatten",
  the 30-second reader test) plus normix grounding (design rows, the
  `compare.py` lever).

### How principles stay visible

1. `AGENTS.md` holds the four priorities. Both hosts load that file at startup.
2. In Cursor, the `principles` description is in the skill list.
   In Codex, that description is visible only when `.agents/skills/` links the skill.
   This repo does not link `principles` yet. Read the body from the AGENTS.md route.
3. Workflow skills (`architect`, `arena`, `interrogate`, `figure-it-out`,
   `tdd`) cite principle sections by name. That citation is the read trigger.

Rule of thumb: a principle that matters *everywhere* goes to AGENTS.md;
one that matters *at specific moments* goes to the skill and gets cited
by the workflows that own those moments.

### Workflow skills: narrow triggers, recorded outputs

- Each workflow skill states its trigger in the description and its
  non-trigger in the body (architect: "would this add a `design.md`
  row? if not, skip"). Skills produce records; docs remain the record —
  architect *lands* a design.md row rather than replacing the table.
- Cross-family bake-offs run outside this repo. Cursor sessions here use
  Grok only (`grok-4.7-xhigh`). GPT runs only in Codex. Normix Engineer
  runs one Codex GPT candidate, one Cursor Grok candidate, and its own
  candidate. It compares the three and grafts a synthesis.
- Architect Phase B prepares that hand-off and stops. On Cursor, the
  default interrogate tier is Grok 4.7 reviewers with distinct lenses.
  On Codex, a sequential lens pass is not that review.
  Escalation uses § Host support.
  Use it for Bessel, the GIG/GH solvers, the EM fitter, or `@jax.custom_jvp`.
  Use it when the user asks for a cross-family review.
  Do not copy a model list into another skill.
- Checkpoint policy: routine reversible work proceeds within the established contract.
  Structural or execution-contract changes follow architect Phase C.
  Existing user authorization applies; a new contract change needs agreement on that change.

### Visual artifacts (show-me, 2026-08)

Conversation-scoped explanations that chat mermaid / LaTeX can't carry
— a side-by-side parametrization, a layout, a research-note page —
go through the `show-me` skill (row C5). Smallest view first; HTML
is the last resort, matching `docs/_static/normix.css` tokens.

HTML is not a third documentation system. Durable notes still land in
`docs/` or `dev-notes/`. The desktop menu and the cloud screenshot path
are Cursor-only. Delivery steps are in `.cursor/skills/show-me/SKILL.md`.

On Codex, write HTML outside the git tree.
Give an ordinary link to that file.
Do not assume Mermaid or `/opt/cursor/artifacts/`.
Survey: `../references/humanlayer_skills_review.md`.

### Completion criteria in skill bodies

Each numbered step or phase of a workflow skill states what "done"
looks like in one line. Borrowed from humanlayer's loop skills;
encoded in `.cursor/rules/maintain-skills.mdc`. The criterion is a
checkable predicate, not a restatement of the step.

---

## Structure

Detailed rules for each document type live in their own `.cursor/rules/` files.
In Cursor, a rule loads when its glob matches, or when `alwaysApply` is true.
In Codex, the agent reads the one rule the route names.
This avoids loading every maintenance rule for one edit.

| Document | Maintenance Rule |
|---|---|
| `AGENTS.md` | `.cursor/rules/maintain-agents-md.mdc` |
| `../ARCHITECTURE.md` | `.cursor/rules/maintain-architecture-md.mdc` |
| Design docs (this dir), `../tech_notes/`, `../references/` | `.cursor/rules/maintain-design-docs.mdc` |
| `../../docs/theory/` | `.cursor/rules/maintain-theory-docs.mdc` |
| `.cursor/rules/*.mdc` | `.cursor/rules/maintain-cursor-rules.mdc` |
| `.cursor/skills/` | `.cursor/rules/maintain-skills.mdc` |

### Maintenance Workflows

Maintenance triggers and periodic reviews are codified in the
`agent-maintenance` skill (`.cursor/skills/agent-maintenance/`).

### Commit Conventions

Git commit conventions for docs and all other changes are codified in the
`git-conventions` skill (`.cursor/skills/git-conventions/`).

---

## Information Flow

```
                         AGENTS.md
                    startup map + pointers
                               |
         +---------------------+---------------------+
         |                     |                     |
         v                     v                     v
  .cursor/rules/        dev-notes/            .cursor/skills/
  Cursor: glob or       design/               canonical bodies
  alwaysApply.          tech_notes/
  Codex: read from      references/           .agents/skills/
  the AGENTS.md route.  ARCHITECTURE.md       directory symlinks
                        read on demand        for Codex discovery
         |
         v
       docs/
       design/ theory/ research/ tutorials/
       published site
```

**The agent's journey for a typical task:**

1. Cursor starts with `AGENTS.md`, `project-overview.mdc`, and `rule-authoring.mdc`.
   Codex starts with `AGENTS.md`. Codex then reads other files from that route.
2. In Cursor, opening a `.py` file injects `coding-conventions.mdc`.
   In Codex, read that rule from the route before editing Python.
3. Needs to understand GIG optimization → reads `../tech_notes/gig_eta_to_theta.md`
4. Needs to add a new distribution → follows the `agent-maintenance`
   skill's "New Distribution Added" trigger list
5. Proposes a new ABC layer → `architect` skill (checkpoint, then a
   `design.md` row)
6. Needs a side-by-side math comparison → `show-me` skill (HTML last
   resort). Cursor may screenshot. Codex uses a writable path and an ordinary link.
7. Commits → uses `git-conventions` skill

At no point does the agent load everything. Context grows only as needed.

---

## Host support

Cursor and Codex read the same files. They do not load those files the same way.

### Cursor

- A rule with `alwaysApply: true` loads in every session.
  Today those rules are `project-overview.mdc` and `rule-authoring.mdc`.
- Any other `.mdc` rule loads when its `globs` match an open file.
- A skill `description` is in the skill list. The body loads when the skill is selected.
- `disable-model-invocation: true` blocks implicit selection. The file stays readable.

### Codex

- In this checkout, project startup text is the root `AGENTS.md`.
  Codex can also load a global file and a nested `AGENTS.md`.
  This checkout has no nested `AGENTS.md`.
  Codex does not evaluate `.mdc` frontmatter.
- Read each selected rule or skill body from the route in `AGENTS.md`.
- `.agents/skills/` holds directory symlinks into `.cursor/skills/`.
  Codex can match those skills by description.
- The canonical body stays under `.cursor/skills/`. Do not keep a second copy.
- Do not link `figure-it-out` or `arena` for catalog coverage.
  `figure-it-out` stays explicit-only.
- `.codex/agents/` holds project custom agents.
  `comment-sicko.toml` names `comment_sicko` and sets no `model`.
  The agent inherits the parent model.
  Docs: https://learn.chatgpt.com/docs/agent-configuration/subagents
- A Codex subagent inherits the parent sandbox.
  Read-only behavior needs a read-only parent sandbox.
  A "do not edit" line in the toml is not that sandbox.
- Set `allow_implicit_invocation: false` in `agents/openai.yaml` to block implicit selection.
  It does not hide the file.
- Cursor `Task` arguments are not Codex settings.
  `subagent_type`, `readonly`, and model slugs are not Codex settings.
  Use the tools this host has.
- A sequential lens pass in one session is not an independent cross-family review.

### Discovery smoke

Read-only. Do not add skill links to grow the catalog.

1. Check that each `.agents/skills/*` symlink resolves.
2. Check that `.agents/skills/figure-it-out` is absent.
3. Check that `.agents/skills/arena` is absent.
4. Check that `.codex/agents/comment-sicko.toml` sets `name = "comment_sicko"`.
5. If `codex` is on `PATH`, run the read-only command below.

   ```bash
   codex exec --sandbox read-only --ask-for-approval never \
     "List the skill names in your startup catalog. Do not edit files."
   ```

   Expect the linked skill names.
   Do not expect `figure-it-out`.

### Normix Engineer hand-off

Normix Engineer is an external assistant. This repo has no call, endpoint, or import for it.

Build the package the calling skill names. Then stop.

**Architect Phase B.** Include the design task and the Phase A grounding.
Include `.cursor/skills/architect/references/runner-prompt.md` and `.cursor/skills/architect/references/rationale-template.md`.
Include a rubric of 3–6 gradeable criteria.

Candidates do not see the rubric.
Name a concrete row, such as S2.
Write "the sketch keeps `grad_fn` in θ-space per row S2".
Do not write "the code is correct".

**Interrogate escalation.** Include the diff, the touched class, and its base.
Include the theory or tech-note page.
Include the goal, constraints, proposed mechanism, and authority packet from interrogate step 2.
Identify the base and head revisions.
Include pre-change decision rows, related unchanged rows, and relevant rejected alternatives.

The user relays the package to Normix Engineer. Do not invent a transport.

Normix Engineer returns one written synthesis.
For a design bake-off, that note names the base, the grafts, and the rejections.
For a review, it names the reviewers, reviewed revision, constraints checked, and findings.
A draft PR, CI success, or a prepared package does not supply that return.
If the diff changes afterward, identify which findings still apply and what needs renewed review.

Paste that return into the session before you call the work reviewed.
With no return, the state is "ready for external review".
Never write "reviewed" unless that synthesis is back in the session.

`architect`, `interrogate`, and `arena` point here. Do not copy this contract into those skills.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Do This Instead |
|---|---|---|
| Monolithic AGENTS.md (>300 lines) | Context bloat; agent can't find what matters | Keep it as a map; move detail to rules/docs |
| Duplicating facts across files | Contradictions after partial updates | Single source of truth + pointers |
| Rules that state the obvious | Wastes context; agent already knows this | Focus on project-specific, non-obvious conventions |
| Over-specified skills ("run this exact command") | Brittle; breaks when commands change | State goals and constraints, not exact steps |
| Never updating rules | Stale rules cause worse behavior than no rules | Treat rules as living code; maintain them |
| Too many rules loaded at once | Context pollution → confused agent | In Cursor, use globs. In Codex, read only the routed rule. |
| Rules contradicting each other | Agent picks one arbitrarily, often wrong | Periodic consolidation; single source of truth |

---

## Evolution

This design is intentionally minimal. As the project grows:

- **Add rules only when an agent makes a recurring mistake.** Don't
  preemptively write rules for problems that haven't occurred.
- **Add skills only when a workflow is repeated >3 times.** Don't create
  skills for one-off tasks.
- **Consolidate periodically.** When rules exceed ~10 files or skills exceed
  ~15, review for redundancy and contradictions.
- **Measure.** Track which rules and skills are actually invoked. Remove
  those that aren't used.
