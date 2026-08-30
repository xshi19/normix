# ASV Benchmarking: trend tracking for normix

> **IN PROGRESS — Phase 1 done 2026-08-28.** Drafted 2026-08-23.
> **Supersedes:** the "skip ASV" verdict in
> [`../references/gpjax_review.md`](../references/gpjax_review.md) §6.2/§8.
> That verdict weighed ASV as a replacement for the ad-hoc scripts; this plan
> adopts it as a *separate layer* (curated micro-suite for trends) while the
> existing deep-dive scripts stay.
> **Scope:** new `asv_bench/` tree, `pyproject.toml` dev deps,
> `.github/workflows/` (Phase 5 only), AGENTS.md context map,
> eventual pruning of `scripts/benchmark_*.py`.
> **Does not touch:** `normix/` source, `tests/`, the deep-dive scripts in
> `benchmarks/` (they remain the investigation layer).

---

## Motivation

Current benchmarking is ad-hoc: `benchmarks/bench_*.py` scripts written for
specific investigations, JSON snapshots in `benchmarks/results/`, manual
two-file diffs via `compare.py`. Nothing tracks trends across commits, so a
performance regression is only noticed if someone happens to re-run the right
script and eyeball the right table. GPJax runs
[ASV](https://asv.readthedocs.io/) continuously and catches regressions from
per-commit trend series (gpjax_review §6.2).

Runs happen on the maintainer's Linux desktop (GPU). GitHub-hosted runners
have no GPU and noisy timing, so CI is at most a coarse tripwire, never the
canonical results source.

## ASV 101

[ASV](https://asv.readthedocs.io/) (airspeed velocity, `pip install asv`,
0.6.x; used by NumPy, SciPy, pandas, astropy, GPJax) benchmarks a package
*over its git history*. Benchmarks are written once; ASV checks out commits,
builds each one, runs the suite, stores results as JSON keyed by
(machine, environment, commit), and renders a static HTML dashboard with
per-benchmark trend lines and automatic step-detection that flags
regressions. The dashboard is static files — publishable straight into the
existing gh-pages site.

### Benchmark conventions

Benchmarks are convention-named methods in the configured `benchmark_dir`:

```python
class GIGSolver:
    params = [["jax", "cpu"]]        # cartesian product of parameter axes
    param_names = ["backend"]

    def setup(self, backend):
        self.eta = ...               # build inputs; jit warm-up goes here

    def time_from_expectation(self, backend):
        GIG.from_expectation(self.eta, backend=backend)  # block_until_ready
```

The prefix selects the metric: `time_*` (wall time), `timeraw_*` (wall time,
fresh subprocess per measurement), `peakmem_*` (process peak RSS), `mem_*`
(object size), `track_*` (any number the method returns). Each value
combination in `params` becomes its own trend line, plotted together.
Attributes like `repeat`, `number`, `warmup_time`, `timeout` tune the timer.

### The environment model

ASV never benchmarks the dev environment. Per run it creates isolated
environments in `.asv/env/`, one per combination of:

- **Python version** — `"pythons": ["3.12"]`;
- **dependency versions** — `"matrix" → "req"`, e.g. benchmark against two
  JAX versions;
- **environment variables** — `"matrix" → "env"` (part of the build key) and
  `"env_nobuild"` (set at benchmark time only; no rebuild).

For each commit it checks out the source into a hidden clone, builds a
wheel, installs it into every environment, and runs the suite there.
Environments are cached, so only the first run pays setup cost.

**`environment_type` is which tool ASV uses to create those isolated
envs**, not which Python you currently have activated. The options:

| Type | What it is | When to use |
|---|---|---|
| `virtualenv` | Classic [virtualenv](https://virtualenv.pypa.io/) — a copy of a Python interpreter plus a `site-packages`. ASV finds `python3.12` on `PATH` and clones it. | Always available; ASV's default if nothing else is on PATH. |
| `uv` | Same idea, but ASV calls [`uv venv`](https://docs.astral.sh/uv/) instead of virtualenv. Faster env create/install; can fetch Python versions if needed. | Prefer for normix — the project already uses uv, and ASV ≥ 0.6.6 ships this plugin (`pip install asv[envs]` or just have `uv` on PATH). |
| `conda` / `rattler` | Conda-forge environments. Needed when deps are not pip-installable. | Not needed here (jax/equinox/scipy are all on PyPI). |

The earlier open question "is the uv backend released yet at install
time?" meant: when the plan was drafted, `uv` was documented on ASV's
`main` branch and it was unclear whether a *PyPI release* already
contained the plugin. It does — **asv 0.6.6** (current PyPI) added it
(`asv/plugins/uv.py`; optional extra `envs`). Use
`"environment_type": "uv"`. If `asv` is older than 0.6.6, fall back to
`virtualenv` — same isolation model, slower setup.

Two escape hatches:

- `asv run --python=same --quick` bypasses isolation and uses the currently
  active interpreter — the smoke-test loop while authoring benchmarks
  (current checkout only; no historical runs in this mode).
- `asv machine` records hardware identity in `~/.asv-machine.json`; results
  from different machines are stored and plotted as separate series, never
  mixed. This is what keeps desktop-GPU and CI numbers apart.

Historical runs (`asv run NEW`, `asv run TAGS`) always go through isolated
envs.

### Day-to-day commands

| Command | Purpose |
|---|---|
| `asv run NEW` | benchmark commits since the last run on this machine |
| `asv run v0.1.0..master` / `asv run TAGS` | backfill history |
| `asv continuous master mybranch --factor 1.1` | PR regression check (fails on >10% slowdown) |
| `asv compare A B` | table diff of two commits |
| `asv publish` + `asv preview` | build and serve the dashboard |
| `asv find` | bisect a regression to the offending commit |
| `asv profile` | profile a single benchmark |

How normix's cpu/jax split maps onto `params` vs environments vs machines is
the subject of the next section.

### What `matrix` is

`matrix` in `asv.conf.json` is a cartesian product of *install-time and
runtime knobs* that ASV turns into separate environments (and therefore
separate trend series on the dashboard). It is **not** the same as
pytest-style parametrization (`params` on a benchmark class):

- **`params`**: same install, same env; the benchmark method is called
  several times with different arguments. Cheap. Use for `backend='jax'|'cpu'`.
- **`matrix`**: ASV builds *N* environments. Expensive (N× install time).
  Use only when the knob is actually an install or a process-level env var.

The three sub-keys:

```json
"matrix": {
    "req": {
        "jax": ["0.5.0", "0.6.0"]
    },
    "env": {
        "SOME_BUILD_FLAG": ["0", "1"]
    },
    "env_nobuild": {
        "JAX_PLATFORMS": ["cpu", "cuda"]
    }
}
```

| Key | Meaning | Rebuilds the env? |
|---|---|---|
| `req` | Third-party **pip/conda packages and versions** installed *into* the env, before the project itself. Empty string `""` = latest; `null` = don't install. Lists cartesian-product. | Yes — each version combo is a distinct env. |
| `env` | Environment variables that are part of the **build key** (e.g. compile flags). | Yes. |
| `env_nobuild` | Environment variables set only when **running** the benchmarks. Same installed env, different runtime. | No. |

**`matrix.req` example.** `"req": {"jax": ["0.5.0", "0.6.0"]}` makes ASV
install two copies of the world, one pinned to each JAX, and plot two
series. That is how you would catch "we got slower because JAX 0.7
changed XLA", independent of any normix commit. Cost: every historical
commit is timed twice. **Skip this until a JAX upgrade is imminent**;
one env (latest jax from the project's own deps) keeps the suite fast.

For the CPU-vs-CUDA device split we *do* want a matrix entry, but it
belongs in `env_nobuild` (`JAX_PLATFORMS`), not `req` — same jax
install, different device at run time.

## Design: two layers, three axes

### Layer 1 — ASV curated suite (`asv_bench/`, new)

Small, fast (seconds per benchmark), synthetic data only, fixed seeds.
Purpose: per-commit trend lines and regression detection. This is the only
layer ASV owns.

### Layer 2 — deep-dive scripts (`benchmarks/`, existing, unchanged)

SP500 data, MCECM variants, large sweeps, rich diagnostic tables. Purpose:
one-off investigations feeding tech notes. Too slow and too tabular for ASV.
`run_all.py` + `compare.py` stay for this layer.

### The three axes and their ASV mechanisms

The "cpu"/"jax" question decomposes into *different* axes with different
ASV mechanisms — conflating them is the main design trap:

| Axis | What it is | ASV mechanism |
|---|---|---|
| `backend='jax'\|'cpu'` (triad, `e_step_backend`) | code path, same install | **benchmark `params`** — one env, two trend lines per plot |
| JAX device (CPU vs CUDA) | runtime device, same install | **`matrix.env_nobuild: {"JAX_PLATFORMS": ["cpu", "cuda"]}`** — two environment series, no rebuild |
| Hardware (desktop vs CI runner) | physical machine | **`asv machine`** — results keyed per machine, never mixed |

Benchmark parametrization example:

```python
class GIGFromExpectation:
    params = [["jax", "cpu"]]
    param_names = ["backend"]

    def setup(self, backend):
        self.eta = ...          # fixed-seed synthetic η
        GIG.from_expectation(self.eta, backend=backend)  # jit warm-up

    def time_from_expectation(self, backend):
        gig = GIG.from_expectation(self.eta, backend=backend)
        jax.block_until_ready(gig.p)
```

### Config sketch (`asv_bench/asv.conf.json`)

```json
{
    "version": 1,
    "project": "normix",
    "project_url": "https://xshi19.github.io/normix/",
    "repo": "..",
    "branches": ["master"],
    "environment_type": "uv",
    "pythons": ["3.12"],
    "benchmark_dir": "benchmarks",
    "env_dir": ".asv/env",
    "results_dir": "results",
    "html_dir": ".asv/html",
    "build_command": [
        "python -m pip install build",
        "python -m build --wheel -o {build_cache_dir} {build_dir}"
    ],
    "install_command": [
        "in-dir={env_dir} python {conf_dir}/_asv_install.py {wheel_file}"
    ],
    "matrix": {
        "req": {
            "jax": ["0.9.1"],
            "jaxlib": ["0.9.1"]
        },
        "env_nobuild": {
            "JAX_PLATFORMS": ["cpu", "cuda"]
        }
    }
}
```

Notes:
- `environment_type`: `uv` (asv ≥ 0.6.6). Fall back to `virtualenv` only
  if the installed asv is older. No conda.
- ASV's uv plugin does `uv venv` then **pip install** — it does **not**
  read `uv.lock`. The single-valued `matrix.req` (`jax==0.9.1`) is how
  ASV gets the same pin as `[tool.uv] constraint-dependencies`. One
  version, not a version-comparison matrix; bump it in lockstep with
  the uv constraints.
- GPU series need CUDA-enabled JAX inside the ASV-managed env.
  `install_command` runs `_asv_install.py`: `{wheel_file}[cuda12]` plus,
  on Linux, `jax[cuda12]==0.9.1` (lockstep with `matrix.req`). The extra
  in `pyproject.toml` is Linux-only; non-linux `exclude` drops the cuda
  `env_nobuild` series. `--python=same` on a CPU-only interpreter skips
  cuda via `NotImplementedError` in setup.
- x64 is enabled by normix at import; no env var needed.
- Smoke-test loop while authoring: `asv run --python=same --quick`.
- `build_command` must leave **one** wheel in `{build_cache_dir}`.
  `pip wheel` without `--no-deps` dumps every dependency and `{wheel_file}`
  is ambiguous.

### JAX timing rules (adapted from GPJax's five rules)

Recorded in `asv_bench/README.md`:

1. Every `time_*` body ends in `block_until_ready` — otherwise dispatch is
   timed, not work.
2. Steady-state benchmarks: jit warm-up call in `setup()`.
3. Compile-time benchmarks: `timeraw_*` methods (fresh subprocess per
   measurement — no cache contamination, `number=1` enforced). This replaces
   the first-call/cached-call split in `bench_jit_solvers.py`.
4. Synthetic data only, fixed `jax.random` keys — no SP500 CSV dependency,
   so any machine (or CI) can run the suite.
5. Machine identity pinned via `asv machine`; never benchmark on battery /
   thermally-throttled hardware.

### Initial suite (port from existing scripts, shrunk to seconds)

| ASV class | Source | Params |
|---|---|---|
| `Bessel` (`time_log_kv_*`, `time_grad_log_kv`) | `bench_bessel.py` | regime (small-z / mid / Hankel), scalar vs batch |
| `GIGFromExpectation` | `bench_gig_solvers.py` | backend jax/cpu × easy/hard η |
| `EStep` (`time_conditional_expectations`) | `bench_em_mixture.py` | dist VG/NIG/GH × e_backend jax/cpu × N ∈ {1e3, 1e4} |
| `EMIteration` (small synthetic fit, fixed iters) | `bench_em_mixture.py` | dist |
| `Compile` (`timeraw_*` jit solver first call) | `bench_jit_solvers.py` | dist |
| `Sampling` (`time_gig_rvs`) | `bench_gig_solvers.py` / `scripts/benchmark_gig_rvs.py` | — |

## Recommendations (summary)

1. **Adopt ASV, scoped**: trends only; the deep-dive scripts remain the
   investigation tool. Do not port the SP500/MCECM benchmarks.
2. **Desktop is the canonical machine.** Run `asv run` locally **at each
   GitHub release** (plus an optional `asv run NEW` before a suspected
   performance-sensitive merge). Commit results with the release. CI is
   optional and never authoritative.
3. **Backend via `params`, device via `env_nobuild`, machine via
   `asv machine`** — per the axis table above.
4. **Results committed to master** under `asv_bench/results/` (small JSON).
   Revisit an orphan `asv-results` branch (GPJax pattern) only if the commit
   noise annoys.
5. **Publish the dashboard** into the existing docs site at `/benchmarks/`
   (static HTML from `asv publish`; the docs-publish skill owns gh-pages).

## Phases

### Phase 1 — Scaffold (local, CPU only) ✅

- [x] `uv add --dev asv`
- [x] Create `asv_bench/{asv.conf.json, benchmarks/__init__.py, README.md}`
      (timing rules in the README)
- [x] `asv machine` profile on the desktop
- [x] Port `Bessel` + `GIGFromExpectation` (backend params)
- [x] Verify with `asv run --python=same --quick`, then a real
      `asv run 'HEAD^!'` (PR branch; `master^!` after merge); `asv preview`
      the dashboard

Gotchas from the first isolated run:

- Quote `'HEAD^!'` / `'master^!'` so bash history expansion does not
  swallow `!`.
- `asv publish` on a non-master commit warns `Couldn't find HASH in
  branches (master)`. Expected until the suite is on master.

### Phase 2 — Device matrix (desktop) ✅

- [x] `install_command` with the `cuda12` extra; confirm CUDA jax inside the
      ASV env
- [x] `env_nobuild: JAX_PLATFORMS = ["cpu", "cuda"]`; confirm two environment
      series appear in the dashboard
- [x] Sanity-check GPU numbers against `benchmarks/results/` history

Gotchas:

- `matrix.req` installs `jax==0.9.1` *without* extras. `{wheel_file}[cuda12]`
  alone does not grow extras on an already-installed jax. `_asv_install.py`
  then does `jax[cuda12]==0.9.1` on Linux. Keep `--force-reinstall` on the
  wheel (ASV default; versions may not change between commits).
- `JAX_PLATFORMS=cuda` with CPU-only jax fails at first device placement
  (`normix.utils.bessel` module-level `jnp.asarray`). `benchmarks/__init__.py`
  falls back to `JAX_PLATFORMS=cpu` *before* importing jax when
  `jax_cuda12_plugin` is missing; `setup` then skips. Isolated Linux GPU
  runs do not take that path.
- Non-linux: `exclude` on `sys_platform` drops the cuda series (no
  `jax[cuda12]` wheels).
- cpu and cuda share one `.asv/env/<hash>/` (`env_nobuild` is not in the
  build key). Result filenames and the dashboard still split on
  `env-JAX_PLATFORMS`.
- Micro-suite GPU is slower than CPU (kernel launch vs a 90 μs CPU
  `log_kv`). Historical `bench_bessel.py` / `bench_gig_solvers.py` GPU
  scalars (~345 ms / ~2.4 s) were unjitted; ASV's jitted cuda scalar
  `log_kv` is ~0.3 ms and GIG jax-Newton ~260–710 ms. Direction matches
  (jax ≫ cpu on this problem size); do not compare the old milliseconds
  to ASV μs.

Verified on wukong (RTX 4090, WSL2): `jax-cuda12-plugin==0.9.1` in the
ASV env; `JAX_PLATFORMS=cuda` → `CudaDevice(id=0)`; `asv publish`
`params.env-JAX_PLATFORMS = ["cpu", "cuda"]`.

### Phase 3 — Suite build-out + first history

- [ ] Port `EStep`, `EMIteration`, `Compile` (`timeraw_*`), `Sampling`
- [ ] Backfill **tags only** (matches the per-release cadence):
      `asv run TAGS`. Do not time every commit on master.
- [ ] Wall-clock budget check: full suite ≤ ~10 min/commit on the desktop;
      shrink Ns if over

### Phase 4 — Publishing

- [ ] Commit `asv_bench/results/`; `asv publish` output wired into the docs
      site under `/benchmarks/` (update the docs-publish skill)
- [ ] AGENTS.md context map: benchmarks row → mention both layers

### Phase 5 — CI tripwire (optional; decide after Phase 3)

- [ ] PR workflow: `asv continuous master HEAD --factor 1.5` on a CPU-only
      subset (`-b` regex selecting Bessel/GIG micro-benchmarks)
- [ ] Loose threshold on purpose — shared runners jitter ±20%; this is a
      smoke check, not the trend source

### Phase 6 — Consolidation

- [ ] Retire `scripts/benchmark_*.py` where the ASV suite or
      `benchmarks/bench_*.py` covers them (`benchmark_comprehensive.py`,
      `benchmark_gig_rvs.py`, `benchmark_mixture_em.py`)
- [ ] State the two-layer split in `benchmarks/` (docstring or short README)

## Decisions (was: open questions)

- **uv env backend — use it.** asv 0.6.6 on PyPI ships `"environment_type":
  "uv"`. Prefer that; `virtualenv` is the fallback if asv < 0.6.6.
  See the environment-model subsection above.
- **JAX version matrix — skip (no old-vs-new JAX).** Pin one version
  in two places that must stay in lockstep: (1) `[tool.uv]
  constraint-dependencies` (`jax==0.9.1`, `jaxlib==0.9.1`) so `uv sync`
  / `uv lock --upgrade` cannot silently move JAX; (2) a *single-valued*
  `matrix.req` in ASV, because ASV's uv plugin does `uv venv` + pip and
  **does not read `uv.lock`**. That is not a version-comparison matrix.
  Published metadata stays `jax>=0.4.38`. Bump both pins on a
  deliberate upgrade.
- **Result cadence — per GitHub release**, not per merge. A full
  desktop run (CPU + CUDA series × the curated suite) is too expensive
  to attach to every merge. At release: `asv run <prev-tag>..<new-tag>`
  (or `asv run NEW` if that range is empty) on the desktop, commit
  `asv_bench/results/`, publish the dashboard. For a suspected
  performance-sensitive PR, optionally run `asv continuous master HEAD`
  locally before merging — that is a two-commit compare, not a history
  backfill. Phase 3 still backfills tags once so the dashboard has a
  trend line on day one.
