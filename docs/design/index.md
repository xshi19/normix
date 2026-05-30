# Design Rationale

User-facing design documentation for normix — the **why** behind the API and
algorithms. For class and method reference see the [API Reference](../api/index).

```{toctree}
:maxdepth: 1
:caption: Topics

exponential_family
mixtures
em_framework
solvers_and_bessel
```

## Quick lookups

| Question | Where to read |
|---|---|
| Why two classes for each mixture distribution? | {doc}`mixtures` § 1 |
| Why three classmethod tiers for the log-partition? | {doc}`exponential_family` § 2 |
| What does `'det_sigma_x'` regularisation do? | {doc}`em_framework` § 5 |
| Why CPU backend for Bessel and GIG solve? | {doc}`solvers_and_bessel` § 4 |

## See also

- [Mathematical Background](../theory/index) — formal derivations
- [API Reference](../api/index) — module and class reference
- [Quickstart](../getting_started/quickstart) — install and first fit
