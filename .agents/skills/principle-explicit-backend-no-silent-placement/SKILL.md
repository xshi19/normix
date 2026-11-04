---
name: principle-explicit-backend-no-silent-placement
description: "Use when changing numerical backends, execution placement, fallbacks, or transformation support."
disable-model-invocation: true
---

# Explicit Backend, No Silent Placement

Treat promised execution placement and transformation support as part of the numerical API. Make fallback behavior an explicit policy the caller can select and inspect.

**Why:** A host detour can return the right values on the expected device while adding transfers, serializing accelerator work, or breaking differentiation. Value-only tests can therefore pass while a training loop loses gradients or a supposedly native kernel spends its time moving data.

**Pattern:**

- Preserve caller-selected execution policies, including explicitly selected automatic placement.
- Disclose transfer costs, precision changes, and derivative limits before enabling a fallback that weakens the requested contract.
- Validate value agreement, requested transformations, and placement as separate promises.
- Report unsupported paths explicitly; an unavailable backend must not silently select a different execution policy.
- Review array conversions and scalar extraction for hidden host work; the output device does not prove where computation occurred.
- If the code uses GPU execution or a traced backend (JAX, PyTorch CUDA), read [references/accelerator-placement.md](references/accelerator-placement.md); otherwise skip it.
