# Accelerator placement

- A GPU result can conceal CPU computation followed by a transfer. Inspect traces or profiles for host callbacks and transfers when native execution is promised.
- For traced programs, inspect nested bodies as well as the top-level trace. Validate a structural scanner with a known callback as a positive control.
- Exercise promised compilation, batching, and differentiation through public callers; value agreement alone covers none of these.
- An explicit CPU fallback must disclose transfers and derivative limits. Missing accelerator access remains a coverage gap even if CPU tests pass.
