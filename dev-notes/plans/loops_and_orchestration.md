# Loops, Automations, and the Umbrella Orchestrator

> **Status:** not started (2026-07-11). Remaining work from the pstack
> adoption; full analysis in the archived survey
> `../archive/references/pstack_skills_review.md` (§7.4, §9–14).
> The skill layer (architect/arena, interrogate, how/why, tdd, unslop,
> figure-it-out, principles) is done — see `../design/design.md` row C4.
> Loop mechanics (sensor/controller/actuator, flow control, memory file)
> harvested 2026-08 from humanlayer; see `../references/humanlayer_skills_review.md`
> and row C5. Items 1–3 below are still unbuilt.

## Candidate work items (lowest effort first)

1. **Nightly benchmark automation** — the normix-shaped `benny`. Cron
   trigger → `benchmarks/run_all.py` → `compare.py` against the last
   committed result → if any metric regresses beyond tolerance, open a
   GitHub issue (or PR comment) with the delta table and a plot. Fail
   closed; report-only, never auto-"fix" a metric. Configure at
   cursor.com/automations (schedule + GitHub triggers; no chat surface
   needed).
2. **GitHub-triggered interrogate** — on a PR touching
   `normix/utils/bessel.py`, the GIG/GH solvers, or the EM fitter, run
   the interrogate skill and post the synthesized verdict as a PR
   comment (its mandatory-trigger list, automated).
3. **`/loop` hillclimb** — next time a solver/Bessel path is too slow or
   imprecise: frozen `compare.py` harness, a numeric predicate
   ("≥ X% faster and `slow or stress` green"), decision log per the
   figure-it-out skill Phases C–D. One change, one measurement, keep or
   revert; the data decides.
4. **Umbrella orchestrator skill** (archived survey §7.4) — route a
   request to a playbook: new distribution → agent-maintenance trigger
   list; new numerical method → hillclimb + build-the-lever
   (benchmark-first); doc-only → unslop + docs-publish; EM/solver
   refactor → architect, gated on `-m "slow or stress"`; Bessel touch →
   interrogate. Decide after items 1–3 exist, so the playbooks route to
   real loops rather than aspirations.

## Prerequisites

- Every loop needs a checkable predicate before it starts (figure-it-out
  Phase A). A vague goal makes a loop spin.
- Decision trail: a log table in the plan/investigation file (the
  figure-it-out Phase D convention). Revisit a committed TSV
  (`show-me-your-work`, on hold) only for overnight runs.
- Automations run as cloud agents (Max Mode, usage-billed): keep
  schedules coarse (nightly, not hourly) until the benchmark suite's
  runtime and cost are measured once.

## Loop mechanics (humanlayer, 2026-08)

Harvested from [`humanlayer/skills`](https://github.com/humanlayer/skills)
`design-control-loop` (survey:
`../references/humanlayer_skills_review.md`). The GH-Actions / headless-CLI
plumbing is not adopted — Cursor automations stay the runner. The
vocabulary and three bounding mechanisms are.

Name the parts before building a loop:

| Part | Meaning | Item 1 (nightly bench) | Item 2 (PR interrogate) | Item 3 (`/loop` hillclimb) |
|---|---|---|---|---|
| **Set point** | Desired property | no metric regresses beyond tolerance vs last committed result | deep-math diffs get an adversarial verdict before merge | predicate from figure-it-out Phase A (`≥ X%` faster and `slow or stress` green) |
| **Sensor** | Repeatable measurement | `benchmarks/run_all.py` + `compare.py` | path filter on `bessel.py` / solvers / EM fitter | the same frozen `compare.py` harness |
| **Controller** | Next reviewable increment | fail closed, report-only — never auto-"fix" a metric | run interrogate, post the synthesized verdict | one change, one measurement, keep or revert |
| **Actuator** | What applies the change | cloud agent opens a GitHub issue (or PR comment) with the delta table | cloud agent posts the interrogate comment | the coding agent in the hillclimb session |
| **Disturbance** | Outside the loop | commits landing on master, dependency bumps | the PR author's concurrent pushes | the rest of the working tree |

Item 2 is a **dampener** (advisory regression gate on PRs) more than a
loop of its own: it keeps the measured problem from getting worse while
a scheduled loop chips away. Offer a dampener when building item 1;
don't invent a second scheduled agent for it.

When items 1–3 are built, each loop also gets:

1. **Flow control.** Scheduled runs no-op while an open issue/PR from
   *this* loop already exists (label-scoped). Manual dispatch bypasses
   the bound. Default: one open artifact per loop. Without this, a
   nightly run piles up faster than it can be reviewed.
2. **Memory file.** A short markdown file loaded every run, carrying
   standing reviewer feedback — permanent scope exclusions, known
   false-positive areas, selection preferences. Litmus: deleting it
   would lose future-run context, not just history. Not a run log.
   Suggested location: `dev-notes/agent-memory/<loop-slug>.md` (or
   `.github/agent-memory/` if the automation wants it next to the
   workflow). One source of truth: don't repeat the skill's rules here.

Run each component locally (sensor, then controller on that output, then
actuator on a selected target) before wiring the schedule. The
automation should only orchestrate pieces already runnable by hand.

## Control-room decision: no Slack (2026-07-11)

In poteto's loops, Slack is the **control room**, not the engine: the
place to watch runs, get notified, and steer a running automation by
replying in-thread. The automations themselves are Cursor cloud agents;
Slack adds (a) an inbound issue stream to triage, (b) team-wide
visibility, (c) chat-native trigger/steering.

For this repo none of the three applies: solo maintainer, no inbound
issue stream, and the trigger/steer/monitor role is already covered by
the existing setup — Cursor IDE on two desktops, Cursor iOS (push
notifications, follow-ups to cloud agents), cursor.com/agents and
cursor.com/automations from any browser, plus GitHub notifications for
automation-opened issues/PRs (items 1–2 report to GitHub by design).
Slack would also require running a personal workspace just to host the
bot. **Verdict: unnecessary.** Revisit only if normix gains
collaborators or a user-facing support channel.
