# ASV micro-suite

Curated, seconds-scale trend tracking. Investigation scripts stay in
`../benchmarks/` (`run_all.py`, `compare.py`, SP500 / MCECM sweeps). Plan:
[`../dev-notes/plans/asv_benchmarking.md`](../dev-notes/plans/asv_benchmarking.md).

ASV paths are relative to **cwd**. Run every command from this directory.

```bash
cd asv_bench
uv run asv machine --yes                 # once per machine
uv run asv check --python=same
uv run asv run --python=same --quick     # working tree; results not saved
uv run asv run HEAD^!                    # isolated uv env, this commit only
uv run asv publish
uv run asv preview
```

`--python=same --quick` is the authoring loop. Historical / release runs
(`asv run TAGS`, `asv run <prev>..<new>`) always go through isolated envs.
Canonical numbers come from the maintainer's Linux desktop at each GitHub
release — not from CI, not from this smoke path.

Phase 1 is CPU only (`JAX_PLATFORMS=cpu` in `asv.conf.json`). CUDA is Phase 2.

## JAX timing rules

1. Every `time_*` body ends in `block_until_ready` — otherwise dispatch is timed, not work.
2. Steady-state benchmarks: jit warm-up call in `setup()`.
3. Compile-time benchmarks: `timeraw_*` methods (fresh subprocess per measurement). Phase 3.
4. Synthetic data only, fixed seeds — no SP500 CSV, so any machine can run the suite.
5. Machine identity pinned via `asv machine`; never benchmark on battery or thermally throttled hardware.

## Layout

| Path | Role |
|---|---|
| `asv.conf.json` | project, uv envs, jax==0.9.1 pin, CPU `env_nobuild` |
| `benchmarks/bessel.py` | `Bessel`: `log_kv` scalar / batch / grad × regime |
| `benchmarks/gig.py` | `GIGFromExpectation`: backend jax/cpu × easy/hard η |
| `results/` | JSON keyed by (machine, env, commit). Committed from Phase 4 |
| `.asv/` | isolated envs + published HTML. gitignored |

`environment_type` is `uv` (asv ≥ 0.6.6). Fall back to `virtualenv` only if
the installed asv is older. No conda.
