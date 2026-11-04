---
name: benchmarking
description: "Use when measuring numerical speedups, regressions, or implementation tradeoffs."
disable-model-invocation: true
---

# Benchmarking

- Fix the accuracy, failure, transformation, and placement contracts before comparing speed; a faster candidate that changes them answers a different question.
- Inspect the harness for setup, transfers, execution, and validation boundaries. Include each cost according to the claimed workload.
- Tune both sides for that workload; preserve versions, dtype, inputs, thread counts, and cache state. An unfairly configured side leaves adoption unresolved.
- Validate outputs outside timing unless validation is part of the workload. Profile separately; explain the limiting resource and end-to-end impact.
- Alternate repeated runs; retain raw samples, units, count, and spread. A gap within ordinary variation is unresolved; label requested single-run estimates as ballparks.
- Preserve commands, revision, environment, and correctness results with the measurements. Timeouts, failed outputs, unexplained timings, or missing hardware evidence limit the claim.
- If the code runs on an async or JIT backend (JAX, PyTorch CUDA), read [references/async-jit-backends.md](references/async-jit-backends.md); otherwise skip it.
- If pstack is installed, use `benchmark-checklist` before claiming a speedup to review completed work, fairness, and measurement evidence.
