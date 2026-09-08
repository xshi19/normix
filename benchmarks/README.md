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
