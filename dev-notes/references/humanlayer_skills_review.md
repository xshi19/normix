# humanlayer/skills: Survey and Fit for normix

**Date:** 2026-08-12
**Source:** [`humanlayer/skills`](https://github.com/humanlayer/skills) (Claude Code plugin marketplace; SKILL.md format, same spec as `.cursor/skills/`)
**Trigger:** review whether any of the five skills improve the agent framework in `../design/agent_instructions_design.md`, with particular interest in `show-me` HTML review artifacts on Cursor desktop and iOS.

Unrelated to pstack's on-hold `show-me-your-work` (decision-log TSVs). Durable learnings promoted to `../design/agent_instructions_design.md` (row C5 in `design.md`); remaining loop mechanics live in `../plans/loops_and_orchestration.md`.

## Verdicts

| Skill | Verdict | Why |
|---|---|---|
| `show-me` | **Adapt** | Smallest-view hierarchy + last-resort HTML. Needs a Cursor-aware delivery section (desktop Open in Browser, cloud screenshot bridge, no iOS HTML). Implemented as `.cursor/skills/show-me/`. |
| `design-control-loop` | **Harvest** | Sensor / controller / actuator / set-point vocabulary, per-loop memory file, PR bounding, optional dampener. Concepts only — Cursor automations stay the runner. Folded into `../plans/loops_and_orchestration.md`. |
| `build-iterated-agentic-loop` | **Reference** | GH Actions + headless-CLI scaffold. Checklist ideas (one-PR bound, memory file, response template as PR body) overlap the harvest above. Don't install the workflow. |
| `improve-claude-md` | **Skip** | Compensates for Claude Code's "this context may or may not be relevant" disclaimer via `<important if>` blocks. Cursor glob-scoped `.mdc` rules are the mechanical version of the same idea; we already use them. Don't wrap `AGENTS.md` in XML. |
| `narrow-react-prop-types` | **Skip** | React-specific. Keep only as a specimen of a well-shaped loop-task skill (live call sites as source of truth; stories/tests adapt). |

## What we took from `show-me`

Not "generate HTML" — a taste hierarchy: pick the smallest view that makes the key point clear. Pseudocode → call tree → file tree → Mermaid → diff-shaped sketches (diffing a call tree or file tree, not just code) → HTML only when the concept is too dense for Mermaid. Match the product's design tokens (`docs/_static/normix.css`).

normix additions: LaTeX in chat for formulas; matplotlib PNG for distribution plots; KaTeX-in-HTML for side-by-side math; delivery that matches the surface the user is on.

## Cursor viewing (verified 2026-08-12)

| Surface | Rendered HTML? | Path |
|---|---|---|
| Cursor desktop, file in repo | Yes | Right-click → **Open in Browser** (built-in Chromium; Design Mode `Cmd+Shift+D` annotates) |
| Cursor desktop, omnibar `file://` | Buggy | Omnibar rewrites `file://` → `https://`. Don't paste those URLs. |
| Cloud agent VM → user | Bridge | Headless Chrome → PNG/video artifact. iPhone-width screenshot confirmed readable on iOS (2026-08-12). |
| Cursor iOS app | No | No file browser. Diffs, logs, screenshots, videos only. |
| Any phone browser | Yes | Publish to GitHub Pages (`docs-publish` skill). |

Headless-Chrome gotchas recorded in the skill: `--virtual-time-budget` hung; `--timeout=20000` worked. Mermaid ESM modules fail under `file://`; use the UMD build.

Related rendering gaps (not HTML, but they push work toward HTML): classic markdown preview (`Cmd+Shift+V`) renders KaTeX, the inline Preview tab does not; chat mermaid supports flowchart / sequence / state / class / ER / xychart only.

## What we took from `design-control-loop`

The control-theory framing, not the GH-Actions plumbing:

- **Set point / sensor / controller / actuator** under disturbances — names the parts of items 1–3 in the loops plan.
- **Flow control** — scheduled runs no-op while an open PR/issue from this loop already exists.
- **Memory file** — standing reviewer feedback loaded every run. Litmus: deleting it would lose future-run context, not just history.
- **Dampener** — advisory regression gate on PRs so the measured problem can't get worse while the scheduled loop chips away. Item 2 (PR-triggered interrogate) is a dampener, not a loop of its own.
- **Completion criterion** per step — adopted as a skill-authoring convention in `.cursor/rules/maintain-skills.mdc` and applied to `architect` / `interrogate`.

## What we did not take

- `<important if>` blocks in `AGENTS.md`. Progressive disclosure here is glob-scoped rules + on-demand skills, not inline XML weighting.
- A GitHub Actions coding-agent runner. The 2026-07-11 decision (Cursor automations, no Slack) stands.
- A third documentation system of committed HTML reports. Conversation-scoped; promote to `docs/` or `dev-notes/` if it should last.
