# Transformed execution review

- **JAX:** `jnp.where` can select a finite value yet produce a NaN gradient from an inactive expression. Inspect both branches and test gradients under batching. Make inactive inputs safe without altering selected values; changing control-flow primitives alone is not evidence of a fix.
- Static metadata, array leaves, unsupported primitives, and Python control flow can break transformations despite immutable objects. Exercise promised compilation, batching, and autodiff independently.
- Host fallbacks can preserve values while violating placement. Inspect traces or profiles when native execution is promised; a returned GPU array does not exclude host work. Missing device access stays an evidence gap.
