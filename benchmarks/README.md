# Deep-dive benchmarks

Investigation scripts: SP500, MCECM, solver tables, first-call vs cached
JIT. JSON via `--save` into `results/`; diff two files with `compare.py`.

Trend tracking is the ASV suite in [`../asv_bench/`](../asv_bench/)
(dashboard `/benchmarks/` on the docs site). Do not add SP500 or long
sweeps there.

```bash
uv run python benchmarks/run_all.py
uv run python benchmarks/run_all.py --only bessel,gig
uv run python benchmarks/compare.py benchmarks/results/OLD.json benchmarks/results/NEW.json
```

| Script | What it times |
|---|---|
| `bench_bessel.py` | `log_kv` JAX vs CPU, scalar vs batch |
| `bench_gig_solvers.py` | GIG η→θ warm/cold start, batched `expectation_params` |
| `bench_jit_solvers.py` | First-call compile vs cached Newton (GIG, Gamma) |
| `bench_em_mixture.py` | Mixture EM to convergence × backend; `--large` / `--mcecm` |
| `bench_incremental_em.py` | Incremental EM Python loop vs `lax.scan` |
| `bench_gh_paths.py` | GH batch vs incremental, CPU vs JAX, by $(n, d)$ |
| `bench_gradient_fitting.py` | EF MLE / EM vs likelihood gradient descent |

| `bench_quadrature_cvar.py` | Independent-reference VaR/CVaR/mean/derivative errors, MC seed RMSE, warm CPU costs, equal-accuracy counts, optimized portfolios and near-step diagnostics |
| `quadrature_reference.py` | Independent NumPy/SciPy adaptive integrals; regenerates `tests/data/quadrature_cvar_reference.json` |
| `compare_finance_demos.py` | Executes finance MyST pages 04–07 and saves numeric before/after snapshots |

```bash
uv run python benchmarks/bench_quadrature_cvar.py --save --output /tmp/quadrature-cvar.json
uv run python benchmarks/compare_finance_demos.py --output /tmp/finance-demos.json
```

The CVaR benchmark records generation and evaluation separately. Compilation
is excluded; each timed result is synchronized. A missing equal-accuracy
count means the target was not reached in the measured sweep. MC RMSE uses
eight seeds by default; increase `--seeds` for a sampling-error study.
