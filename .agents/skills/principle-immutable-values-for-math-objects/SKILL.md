---
name: principle-immutable-values-for-math-objects
description: "Use when designing mathematical values, fitting results, or reusable numerical artifacts."
disable-model-invocation: true
---

# Immutable Values for Math Objects

Represent a mathematical value with fixed parameters. Keep fitting progress, random state, and reusable numerical artifacts separate from that value.

**Why:** Mutating parameters behind a cached factorization or table can combine quantities from different models without any shape error. Shared mutable state also makes results depend on earlier calls, so parallel fits and repeated experiments can disagree despite receiving the same nominal inputs.

**Pattern:**

- Keep fixed parameters separate from fitting state; return the value with convergence status and diagnostics. No fitter-class hierarchy is required.
- Produce a new value for a parameter update so previously returned results retain their meaning.
- Make reusable tables caller-held artifacts tied to the parameters and inputs that produced them; avoid hidden invalidation inside the value.
- Check buffer ownership and writable aliases; a frozen wrapper around mutable arrays does not preserve value semantics.
- Pass random state and mutable workspaces explicitly so sampling or fitting cannot silently alter later evaluations of the value.
- Immutability alone does not establish transformation support; verify the requested operations separately.
