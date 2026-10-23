# Quadrature scenario implementation

> **ARCHIVED — 2026-10-23.** Stages 1–3 shipped in [#131](https://github.com/xshi19/normix/pull/131) (`06abc51`), released in v0.4.0. Adaptive step refinement stays deferred.

Owner-approved fixed-measure stages 1–3; decision FIN-5 in
[design.md](../../design/design.md). Public math:
[quadrature scenarios](../../../docs/theory/quadrature_scenarios.md).

| Stage | State | Evidence / remaining gate |
|---|---|---|
| Ground / sketch / agree | Complete | Owner supplied and approved the design and derivation report |
| Probability measure and weighted conditional kernels | Complete | `DiscreteMeasure`, RU value, transformed quantile, legacy `_mc` imports; existing tests untouched |
| Distribution quadrature and diagnostics | Complete | 512 nodes, eight sinh panels, S10 helpers reused unchanged; moment masks and step resolution |
| Default finance integration | Complete | CVaR, functional, optimization, transaction costs, ENB |
| Theory, design, tutorial migration | Complete | New theory/design/tutorial pages; tutorials 04–07 and user guide use quadrature |
| Accuracy reference | Complete | 18 NumPy/SciPy adaptive-integral cases, independent of normix; source `benchmarks/quadrature_reference.py` |
| Tests, benchmark, demo comparison, commits | Complete | [#131](https://github.com/xshi19/normix/pull/131) merged as `06abc51`; in v0.4.0 and later |
| Adaptive step refinement | Deferred | Requires accuracy through local step ratio 3000 and cost at most three times the fixed rule |

The notes below record the pre-merge session.

Implementation adjustments: the edge helper already exists, so no Bessel
refactor is needed. The source-law mean may not exist while positive skew
keeps lower-tail CVaR finite; use a geometric center in that case. Retain
`_mc` imports to leave existing regression tests unchanged.

Reproduction after dependencies are available:

```bash
uv run pytest tests/
uv run pytest tests/finance/ -m slow
uv run python benchmarks/bench_quadrature_cvar.py --save --output /tmp/quadrature-cvar.json
uv run python benchmarks/compare_finance_demos.py --output /tmp/finance-after.json
uv run make -C docs clean-cache
uv run make -C docs html-strict
bash scripts/check_doc_links.sh
```

Baseline package and tutorial copies for this session are under
`/tmp/normix-quad-baseline/`. They are local validation artifacts, not tracked
sources. Run `compare_finance_demos.py --module-root /tmp/normix-quad-baseline
--tutorial-dir /tmp/normix-quad-baseline/finance --output /tmp/finance-before.json`
for the unchanged baseline. Then run the saved old tutorial sources against
the current package to isolate MC compatibility before comparing new defaults.

Session checks: Python sources and all 40 finance tutorial code cells parse;
`check_doc_links.sh` and `git diff --check` pass. A separate NumPy/SciPy
audit supports the smooth-regime accuracy; it does not execute normix/JAX.
The required test, benchmark, demo, and strict-build commands all stopped
during dependency downloads (DNS unavailable). Git staging also failed:
`.git/index.lock` is on a read-only filesystem. Full measurements and logs
are recorded in `/tmp/codex-quad-report.md`; no commits were created.
