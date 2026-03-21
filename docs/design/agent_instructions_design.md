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

---

## Principles

### 1. Map, Not Manual

> "Give the agent a map, not a 1,000-page instruction manual."
> — OpenAI Harness Engineering

`AGENTS.md` is a **table of contents** — roughly 100 lines — that tells the
agent *where to look*, not *everything it needs to know*. Deep knowledge lives
in `docs/`, `.cursor/rules/`, and skills. The agent reads deeper context only
when it needs it.

### 2. Progressive Disclosure

Context is injected in layers:

```
Layer 0   AGENTS.md               Always in context. Map + pointers.
Layer 1   .cursor/rules/*.mdc     Auto-injected by glob match (e.g. *.py triggers coding-conventions).
Layer 2   docs/design/*.md        Read on demand when the agent is making design decisions.
Layer 3   docs/tech_notes/*.md    Read on demand when the agent hits a specific numerical/algorithmic problem.
Layer 4   skills/                 Invoked explicitly for specific workflows.
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
| Module hierarchy | `docs/ARCHITECTURE.md` | `AGENTS.md`, `docs/design/design.md` |
| Coding conventions | `.cursor/rules/coding-conventions.mdc` | `AGENTS.md` context map |
| Distribution catalog | `.cursor/rules/project-overview.mdc` | `AGENTS.md` |
| Numerical constants | `normix/utils/constants.py` | `.cursor/rules/coding-conventions.mdc` |
| Design rationale | `docs/design/design.md` | `AGENTS.md` |
| Mathematical derivations | `docs/theory/*.rst` | `AGENTS.md`, `docs/ARCHITECTURE.md` |
| Bessel implementation | `docs/tech_notes/bessel_*.md` | `docs/ARCHITECTURE.md` |

Duplicating facts across files leads to contradictions when one is updated
and the other isn't.

---

## Structure

Detailed rules for each document type live in their own `.cursor/rules/` files,
auto-injected when editing that type. This avoids loading all maintenance rules
when only one is relevant.

| Document | Maintenance Rule |
|---|---|
| `AGENTS.md` | `.cursor/rules/maintain-agents-md.mdc` |
| `docs/ARCHITECTURE.md` | `.cursor/rules/maintain-architecture-md.mdc` |
| `docs/design/`, `docs/tech_notes/`, `docs/references/` | `.cursor/rules/maintain-design-docs.mdc` |
| `docs/theory/` | `.cursor/rules/maintain-theory-docs.mdc` |
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
                    ┌─────────────────────────┐
                    │       AGENTS.md          │  ← Always in context
                    │   (map + pointers)       │
                    └──────────┬──────────────┘
                               │ points to
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                      ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ .cursor/rules/  │  │ docs/            │  │ .cursor/skills/  │
│ (auto-injected  │  │  ├── design/     │  │ (on-demand       │
│  by glob match) │  │  ├── tech_notes/ │  │  workflows)      │
└─────────────────┘  │  ├── theory/     │  └──────────────────┘
                     │  ├── references/ │
                     │  └── ARCH..      │
                     │ (read on demand) │
                     └──────────────────┘
```

**The agent's journey for a typical task:**

1. Agent starts → sees `AGENTS.md` (map) + `project-overview.mdc` (always applied)
2. Opens a `.py` file → `coding-conventions.mdc` auto-injected
3. Needs to understand GIG optimization → reads `docs/tech_notes/gig_eta_to_theta.md`
4. Needs to add a new distribution → invokes `add-distribution` skill
5. Commits → uses `git-conventions` skill

At no point does the agent load everything. Context grows only as needed.

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails | Do This Instead |
|---|---|---|
| Monolithic AGENTS.md (>300 lines) | Context bloat; agent can't find what matters | Keep it as a map; move detail to rules/docs |
| Duplicating facts across files | Contradictions after partial updates | Single source of truth + pointers |
| Rules that state the obvious | Wastes context; agent already knows this | Focus on project-specific, non-obvious conventions |
| Over-specified skills ("run this exact command") | Brittle; breaks when commands change | State goals and constraints, not exact steps |
| Never updating rules | Stale rules cause worse behavior than no rules | Treat rules as living code; maintain them |
| Too many rules loaded at once | Context pollution → confused agent | Use glob-based conditional injection |
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
