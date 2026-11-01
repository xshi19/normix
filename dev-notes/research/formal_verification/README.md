# Formal verification research (2026-10-24)

Offline research reports (Codex GPT 6 Astra, xhigh). They are proposals, not accepted design decisions.

| File | Contents |
|---|---|
| [`01_lean_math_feasibility.md`](01_lean_math_feasibility.md) | Lean 4 + Mathlib for normix theory: feasibility, design, project architecture |
| [`02_jaxpr_to_lean_feasibility.md`](02_jaxpr_to_lean_feasibility.md) | jaxpr to Lean (the jax-lean idea): feasibility, pilots, roadmap |
| [`03_tla_vs_hypothesis.md`](03_tla_vs_hypothesis.md) | TLA+ vs Hypothesis for application code, with a beginner primer |
| [`04_integrated_roadmap.md`](04_integrated_roadmap.md) | Prioritized roadmap combining the three reports |
| [`05_math_checklist.md`](05_math_checklist.md) | Item-by-item math inventory with status tracking |

The hand-written Lean project (W2) lives in `formal/lean/`. W2 M0 merged in PR #161 (`80073f3`): 31 theorems; statement review remains pending. See [decisions.md](../../../formal/lean/decisions.md) for pins and audit policy, and the [updated roadmap](04_integrated_roadmap.md) for Lean CI, the narrow Gamma pilot, and remaining finite calculus.

Mentions of `.research-inputs/jax-lean/` refer to the historical local copy of https://github.com/srush/jax-lean used while writing; it is not part of this repo. That copy had no root LICENSE file. Audit the existing `xshi19/jax-lean` fork's current revision, capabilities, and permission record before integration.
