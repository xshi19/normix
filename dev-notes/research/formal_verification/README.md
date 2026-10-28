# Formal verification research (2026-10-24)

Offline research reports (Codex GPT 6 Astra, xhigh). They are proposals, not accepted design decisions.

| File | Contents |
|---|---|
| [`01_lean_math_feasibility.md`](01_lean_math_feasibility.md) | Lean 4 + Mathlib for normix theory: feasibility, design, project architecture |
| [`02_jaxpr_to_lean_feasibility.md`](02_jaxpr_to_lean_feasibility.md) | jaxpr to Lean (the jax-lean idea): feasibility, pilots, roadmap |
| [`03_tla_vs_hypothesis.md`](03_tla_vs_hypothesis.md) | TLA+ vs Hypothesis for application code, with a beginner primer |
| [`04_integrated_roadmap.md`](04_integrated_roadmap.md) | Prioritized roadmap combining the three reports |
| [`05_math_checklist.md`](05_math_checklist.md) | Item-by-item math inventory with status tracking |

The hand-written Lean project (W2) lives in `formal/lean/`; see `formal/lean/decisions.md` for the pinned versions, dependency audit, and axiom policy.

Mentions of `.research-inputs/jax-lean/` refer to a local copy of https://github.com/srush/jax-lean used while writing; it is not part of this repo. That project has no LICENSE file.
