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
uv run asv run 'HEAD^!'                  # isolated uv env, this commit only (quote so bash does not eat `!`)
# Tag backfill: asv 0.6.6 `TAGS` records tag names, not SHAs. Resolve first:
git tag | while read t; do git rev-list -n 1 "$t"; done > /tmp/asv-tags
uv run asv run HASHFILE:/tmp/asv-tags
uv run asv publish
uv run asv preview
```

`--python=same --quick` is the authoring loop. Historical / release runs
(`asv run HASHFILE:…` of tag SHAs, `asv run <prev>..<new>`) always go through isolated envs.
Canonical numbers come from the maintainer's Linux desktop at each GitHub
release — not from CI, not from this smoke path.

Two device series share one install (`install_command` + `_asv_install.py`
adds `jax[cuda12]` on Linux). `JAX_PLATFORMS` is `env_nobuild`: cpu and cuda
are separate trend lines, no rebuild. Non-linux has no `jax[cuda12]` wheels;
`exclude` drops the cuda series there. `--python=same` on a CPU-only
interpreter skips the cuda series (`NotImplementedError` in setup).

To time the cuda series from the working tree:

```bash
uv sync --extra cuda12
cd asv_bench && uv run asv run --python=same --quick
```

## JAX timing rules

1. Every `time_*` body ends in `block_until_ready` — otherwise dispatch is timed, not work.
2. Steady-state benchmarks: jit warm-up call in `setup()`.
3. Compile-time benchmarks: `timeraw_*` methods (fresh subprocess per measurement).
4. Synthetic data only, fixed seeds — no SP500 CSV, so any machine can run the suite.
5. Machine identity pinned via `asv machine`; never benchmark on battery or thermally throttled hardware.

## Layout

| Path | Role |
|---|---|
| `asv.conf.json` | project, uv envs, jax==0.9.1 pin, cpu/cuda `env_nobuild` |
| `_asv_install.py` | `install_command`: wheel `[cuda12]` + Linux `jax[cuda12]==0.9.1` |
| `benchmarks/__init__.py` | skip cuda series when `jax_cuda12_plugin` is missing |
| `benchmarks/bessel.py` | `Bessel`: `log_kv` scalar / batch / grad × regime |
| `benchmarks/gig.py` | `GIGFromExpectation` (backend × easy/hard η); `Sampling` (`GIG.rvs` Devroye) |
| `benchmarks/em.py` | `EStep` (dist × backend × N); `EMIteration` (dist, 5 steps) |
| `benchmarks/compile.py` | `Compile`: `timeraw_from_expectation` GIG Newton / Gamma digamma |
| `results/` | JSON keyed by (machine, env, commit). `wukong/` + `benchmarks.json` committed |
| `.asv/` | isolated envs + published HTML. gitignored |

`environment_type` is `uv` (asv ≥ 0.6.6). Fall back to `virtualenv` only if
the installed asv is older. No conda.

Docs CI / `make html` copies `.asv/html` into `_build/html/benchmarks/`
(`scripts/publish_asv_html.sh`). Never `asv gh-pages`. Do not add
`docs/benchmarks.md` (GitHub Pages `/benchmarks` vs `/benchmarks/` clash).
