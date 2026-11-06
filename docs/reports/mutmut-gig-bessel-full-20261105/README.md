# Verified full mutmut re-run (2026-11-05)

Artifacts from one full mutmut 3.8.1 run on git SHA `c79dda8`
(`cursor/mutmut-gig-bessel-6335`), modules
`normix/distributions/generalized_inverse_gaussian.py` and
`normix/utils/bessel.py`.

| File | Role |
| --- | --- |
| `run-info.txt` | SHA, tool versions, wall clock, exit code |
| `mutmut-cicd-stats.json` | Aggregate counts (killed, survived, timeout, …) |
| `summary.txt` | Count of each status from `status.tsv` |
| `status.tsv` | One line per mutant: `name<TAB>status` (3251 lines) |

Score: `(killed + timeout) / total` = `(2856 + 45) / 3251` = 2901/3251 ≈ 89.23%.

Reproduce with `scripts/run_mutmut.sh` and `[tool.mutmut]` in `pyproject.toml`.
The large `mutmut-run.log` and `mutmut-results.txt` are not stored here.
