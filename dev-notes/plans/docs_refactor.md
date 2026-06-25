# Documentation Refactor: MyST + `myst-nb`, internal/external split

> **IN PROGRESS — Phases 1–3 done and live; Phases 4–5 pending; Phase 6 optional.**
> **Date:** 2026-05-25 (Phase 3 authored 2026-05-28; status refreshed 2026-06-25)
> **Status:** The MyST + `myst-nb` site is **built and published** to
> https://xshi19.github.io/normix/ via `.github/workflows/docs.yml` (gh-pages
> deploy on push to `master`). All 18 tutorials + getting_started + user_guide +
> design pages render with executed outputs. **What remains:**
> - **Phase 4** — retire `nbsphinx` and delete `notebooks/` (both still present:
>   `nbsphinx` is in `docs/conf.py`, `pyproject.toml`, and the docs-publish skill;
>   `notebooks/` still holds 16 `.ipynb` + 1 `.py`).
> - **Phase 5** — add the release-tier `docs-full.yml` (force full re-execution on
>   `v*` tags / `workflow_dispatch`); not yet created.
> - **Phase 6** — optional polish (theory `.rst` → MyST, theme swap, PR previews).
> **Scope:** `docs/`, `notebooks/`, `.github/workflows/docs.yml`, the docs-publish skill.
> **Does not touch:** `normix/` source, `tests/`, `benchmarks/`, the EM / finance roadmaps.

---

## Motivation

The current docs surface mixes three different concerns and two different
toolchains:

- **Sphinx + autodoc** for the API reference (`docs/api/`).
- **Sphinx + nbsphinx** for the (un-executed) Jupyter notebooks in
  `notebooks/`, currently with `nbsphinx_execute = 'never'`.
- **`.rst` + `.md` mixed** for prose: `docs/index.rst`, `docs/design.rst`,
  `docs/architecture.rst`, `../../docs/theory/*.rst` on the published side; and
  `../design/*.md`, `../plans/*.md`, `../tech_notes/*.md`,
  `../ARCHITECTURE.md`, `../investigations/*`, `../reviews/*`,
  `../archive/*` on the internal side.

Two problems follow:

1. **Repo bloat from notebook outputs.** `notebooks/` is ~9 MB, dominated
   by base64-encoded PNG figures embedded in `.ipynb` JSON. Diffs are
   unreadable, code review on tutorial changes is essentially blind, and
   the published site shows only the last-committed run — not a live
   execution against the current source.
2. **Internal / external boundary is policy, not structure.** `docs/conf.py`
   excludes internal folders via `exclude_patterns`; one missed entry leaks
   internal material to the website. Duplicate-feeling pairs
   (`design.rst` vs `design/`, `architecture.rst` vs `ARCHITECTURE.md`) add
   navigation cost.

## Goals

- Keep **Sphinx as the build engine** — same toolchain the JAX/Equinox/JAXopt
  ecosystem uses, full `autodoc` support for our docstring conventions.
- Replace `nbsphinx` with **`myst-nb`** so the same MyST `.md` parser handles
  prose, tutorials, and notebook cells.
- **Executable tutorials** authored as MyST `.md` with code cells; cell
  outputs are produced at docs-build time, not stored in source.
- **Tiered execution** — fast for PRs, full re-execution for releases.
- **Comprehensive demo** that walks through every public-facing feature, plus
  real-data experiments. Not a 1-to-1 port of the current 16 notebooks.
- **Structural split** between the published website and the agent/dev-internal
  knowledge base.

## Non-goals

- A migration to Jupyter Book v1 (in maintenance) or to `mystmd` /
  Jupyter Book v2 (no mature Sphinx-autodoc parity yet). See `agent-transcripts`
  discussion 2026-05-25 for rationale.
- A bulk rewrite of `../../docs/theory/*.rst` — they coexist with MyST cleanly.
  File-by-file migration when touched is fine; no big-bang.
- Changes to `normix/` source or the EM / finance roadmaps.

---

## Target ARCHITECTURE

### File layout after the refactor

```
normix/                              # unchanged
tests/                               # unchanged
benchmarks/                          # unchanged
scripts/                             # unchanged (data downloaders used by tutorials)
data/                                # unchanged (small CSVs checked in)

docs/                                # PUBLISHED website source only
├── conf.py                          # Sphinx + myst-parser + myst-nb
├── index.md                         # landing page (MyST)
├── getting_started/
│   ├── install.md
│   ├── quickstart.md                # 30-line copy-paste example
│   └── first_model.md               # narrative walkthrough
├── user_guide/                      # prose explanations of concepts
│   ├── distributions.md
│   ├── exponential_family.md
│   ├── em_fitting.md
│   ├── divergences.md
│   └── finance.md
├── tutorials/                       # executable MyST .md (code cells)
│   ├── core/                        # framework concepts
│   ├── distributions/               # distribution tour
│   ├── em/                          # EM variants in practice
│   ├── stats/                       # divergences, goodness of fit
│   └── finance/                     # real-data experiments
├── theory/                          # math derivations (.rst, gradual .md migration)
│   └── *.rst
├── reference/                       # autodoc API (was docs/api/)
│   └── index.md
├── design/                          # PUBLIC-FACING design rationale subset
│   ├── index.md
│   ├── exponential_family.md
│   ├── mixtures.md
│   ├── em_framework.md
│   └── solvers_and_bessel.md
├── changelog.md                     # link / include of root CHANGELOG.md
├── _static/                         # CSS, logos
├── _templates/                      # if needed
└── Makefile

dev-notes/                           # NOT published; agent + dev-internal
├── README.md                        # index for agents
├── ARCHITECTURE.md                  # was ../ARCHITECTURE.md
├── design/
│   ├── agent_instructions_design.md
│   └── design.md                    # philosophy + canonical decision table
├── plans/                           # was ../plans/
├── tech_notes/                      # was ../tech_notes/
├── investigations/                  # was ../investigations/
├── reviews/                         # was ../reviews/
├── references/                      # was ../references/
└── archive/                         # was ../archive/

.github/workflows/
├── ci.yml                           # unchanged
├── docs.yml                         # PR + master: cached execution
└── docs-full.yml                    # release tags + workflow_dispatch: force re-exec
```

Key structural decisions:

- **`docs/` contains nothing that is not meant to be published.** The
  `exclude_patterns` list in `conf.py` shrinks to just `_build`, `Thumbs.db`,
  `.DS_Store`. No more "internal stuff hiding in `docs/`".
- **`dev-notes/`** is the new home for plans, tech notes, archive, investigations,
  reviews, references, and the internal design material. The AGENTS.md context
  map points there. Nothing in this tree is rendered.
- **`notebooks/` disappears.** Source-of-truth tutorial files live at
  `docs/tutorials/**/*.md`. No `.ipynb` is checked in.
- **`../ARCHITECTURE.md` moves to `../ARCHITECTURE.md`.** It is an
  agent-facing blueprint, not a user-facing document. The user-facing
  architecture overview lives in `docs/user_guide/exponential_family.md` and
  `../design/`.
- **The `design.rst` / `design/` and `architecture.rst` / `ARCHITECTURE.md`
  duals are resolved:**
  - The external face is `../design/` (MyST `.md`).
  - The internal face is `dev-notes/design/design.md` (philosophy + decision table).
  - `docs/design.rst` and `docs/architecture.rst` are deleted.

### Build pipeline

```
   .md / .rst sources             docstrings (normix/*.py)
       │                                 │
       ▼                                 ▼
  myst-parser / docutils            sphinx.ext.autodoc + napoleon
       │                                 │
       └────────────┬────────────────────┘
                    ▼
                  Sphinx
                    │
                    ▼
        myst-nb (executes code cells in
        tutorials/**.md, caches outputs in
        docs/_build/.jupyter_cache)
                    │
                    ▼
           docs/_build/html
                    │
                    ▼
       gh-pages branch via peaceiris/actions-gh-pages
```

`conf.py` extension list after the refactor:

```python
extensions = [
    "myst_parser",                  # MyST .md for prose
    "myst_nb",                      # MyST .md with executable code cells
    "sphinx.ext.napoleon",          # numpy-style docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # math: provided by myst_nb / myst_parser via dollarmath + amsmath
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]

nb_execution_mode = "cache"         # PR + master default
nb_execution_timeout = 600          # 10 min per cell (raise per-file if needed)
nb_execution_raise_on_error = True  # fail the build on a broken tutorial
nb_execution_cache_path = "_build/.jupyter_cache"
```

**Theme** — `sphinx-book-theme` (Phase 1) plus a custom CSS overlay derived
from the Kami-style visual language already used by `incerto-wiki`. See
**§ Visual style** below.

### Visual style

normix shares a maintainer and an aesthetic with `incerto-wiki`. Adopting the
same "clean mathematical notes on good paper" language gives a unified house
style across both projects without locking the two repositories together
technically.

The visual contract is the one from `incerto-wiki/../design/VISUAL_STYLE.md`,
adapted for an API-heavy library:

| Layer | Source | normix file |
|---|---|---|
| Base Sphinx theme | `sphinx-book-theme` | `docs/conf.py` (`html_theme`) |
| CSS overrides | port of `assets/css/incerto.css` tokens | `docs/_static/normix.css` |
| Matplotlib `rcParams` | port of `incerto.figures.set_theme()` | `normix/utils/plotting.py` (extend `set_theme()`) |
| Code-block / API-signature styling | Sphinx-book defaults + token-aware overrides | `docs/_static/normix.css` |

Design tokens (mirroring `VISUAL_STYLE.md` § 2 verbatim so the two sites match):

| Token | Value | Use |
|---|---|---|
| `paper` | `#F5F4ED` | Page background |
| `surface` | `#FAF9F5` | Code blocks, figure panels, API signature blocks |
| `sand` | `#E8E6DC` | Subtle fills, plot grid lines |
| `rule` | `#D8D4C8` | Borders, dividers, table rules |
| `ink` | `#141413` | Primary text |
| `muted` | `#6B6A64` | Secondary text, ticks, inactive navigation |
| `accent` | `#1B365D` | Links, active nav, primary plot series |
| `accent_light` | `#2D5A8A` | Hover, secondary link emphasis |

Typography mirrors `incerto-wiki`: Charter serif body, JetBrains Mono code,
17–18px body size, ~1.58 line height, ~68–78 character measure.

normix-specific adaptations (where the API-heavy nature differs from a wiki):

- **API signature blocks** get a quiet `surface` background, `rule` left border,
  monospace at body line-height. No bright colored language tag chips.
- **Autodoc parameter tables** use the `incerto-wiki` table conventions
  (transparent / `surface` header, horizontal rules only, no zebra by default).
- **Math** continues to render via Sphinx mathjax / KaTeX — same as before.
  `dollarmath` MyST extension means we can write `$\psi(\theta)$` inline in
  prose without `:math:` directives.
- **Tutorial figures** call `from normix.utils.plotting import set_theme;
  set_theme()` at the top of every notebook. The theme function is the
  port of `incerto.figures.set_theme()` and uses the same tokens, so a plot
  from a normix tutorial and a plot from `incerto-wiki` are visually
  interchangeable.

The detailed CSS port (`incerto.css` → `normix.css`) is a Phase 1 deliverable,
sized at roughly 300–400 lines. Matching matplotlib helper port to
`normix/utils/plotting.py` is roughly 40–80 lines.

### Internal / external boundary

A single check at PR time verifies the structural split:

```bash
# in CI: docs.yml
test -z "$(find docs -name 'plans' -o -name 'investigations' -o -name 'reviews' \
                     -o -name 'tech_notes' -o -name 'archive' -o -name 'references')"
```

If any of those names appear inside `docs/`, the build fails. This replaces
the `exclude_patterns` policy with a structural invariant.

---

## Workflow

### Authoring a tutorial (MyST `.md` with code cells)

A tutorial file at `docs/tutorials/em/batch_em.md` looks like:

````markdown
---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 180
---

# Batch EM in practice

Brief motivation in prose.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from normix import GeneralizedHyperbolic
```

Inline math like $\psi(\theta)$ works through `dollarmath`.

```{code-cell} python
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (1000, 3))
model = GeneralizedHyperbolic.default_init(X)
result = model.fit(X, max_iter=100)
result.n_iter, result.converged
```
````

Authoring choices for contributors:

- **Pure `.md`** — edit in any editor / Cursor. No `.ipynb` is checked in.
- **`jupytext`-paired** — for contributors who prefer the JupyterLab GUI,
  `jupytext --set-formats md:myst,ipynb docs/tutorials/em/batch_em.md` creates
  a local `.ipynb` pair. The `.ipynb` is `.gitignore`d.

### CI / release tiers

| Trigger | Workflow | `nb_execution_mode` | Cache | Typical runtime |
|---|---|---|---|---|
| PR (any branch) | `docs.yml` | `cache` | restored from prior master build | 5–15 min |
| Push to `master` | `docs.yml` | `cache` | shared key per `notebooks/**` + `uv.lock` hash | 5–20 min |
| Release tag `v*` | `docs-full.yml` | `force` | ignored | 30 min – 3 h |
| `workflow_dispatch` | `docs-full.yml` | `force` | optional | same |
| Local | `make html` | `cache` | local `.jupyter_cache` | 1–10 min |

Public repo → **GitHub Actions minutes are unlimited**, so the cost is wall-clock
time, not money. The 6-hour per-job timeout is the only hard ceiling; we
budget the worst tutorial to stay under 30 min and rely on parallelism for the
rest.

Heavy tutorials (`tutorials/finance/sp500_multivariate.md`,
`tutorials/finance/cvar_optimization.md`) may carry per-file front-matter:

```yaml
mystnb:
  execution_mode: off       # ship pre-executed cached outputs
  # or:
  execution_mode: force
  execution_timeout: 1800
```

### Cross-link discipline

The internal/external split is structurally enforced (folder layout + CI
invariant check), but **link discipline** between the two trees needs a
separate layer. Three enforcement levels, each catching different failure
modes:

#### 1. Preventive — `.cursor/rules/docs-cross-links.mdc`

A new agent rule, scoped to docs and docstrings, that codifies the link
contract before bad links are written. Sketch:

```
---
description: Cross-link conventions between docs/, dev-notes/, normix/ docstrings, README
globs:
  - docs/**/*.{md,rst}
  - dev-notes/**/*.md
  - normix/**/*.py
  - README.md
alwaysApply: false
---

# Cross-link conventions

normix has two doc trees:
- `docs/` — published Sphinx website (public-facing)
- `dev-notes/` — agent/dev-internal, NOT built or published

## In `docs/**` (published)
- NEVER link to `dev-notes/`. The website cannot resolve those paths.
- Cross-reference other `docs/` pages via MyST `{doc}` or `{ref}` roles, not
  raw relative paths (so Sphinx can validate the link).
- Cross-reference `normix.*` symbols via `{py:class}`, `{py:func}`, `{py:meth}`
  (intersphinx-style).
- External URLs are fine; the linkcheck builder validates them in CI.

## In `dev-notes/**` (internal)
- May link freely to `docs/`, other `dev-notes/`, and source code paths.
- Use relative paths (e.g. `../design/em_framework.md`) so links work in IDE
  preview and on GitHub.
- Do not direct end-users into `dev-notes/` from any public surface
  (README, error messages, docstrings, `docs/`).

## In `normix/**/*.py` (docstrings → autodoc → public website)
- Reference public docs via Sphinx `:doc:` or `:ref:` roles, not relative paths.
- NEVER reference `dev-notes/` paths — they are not in the autodoc output.
- Do not link to `../plans/`, `../tech_notes/`, etc. (those moved to
  `dev-notes/` anyway).

## In `README.md` and `AGENTS.md`
- `README.md` is for users: link to the published site or `docs/`, not
  `dev-notes/`.
- `AGENTS.md` is for agents: may link to `dev-notes/` freely (it is the
  context map for internal material).
```

This is the only place link policy is written in human language; everything
below mechanically enforces it.

#### 2. Structural — CI grep invariants

Lightweight regex checks in `docs.yml`. Cheap to run on every push, hard to
silently bypass:

```bash
# forbidden: docs/ references dev-notes/
! rg -n 'dev-notes/' docs/ || (echo "docs/ must not reference dev-notes/"; exit 1)

# forbidden: source docstrings reference dev-notes/
! rg -n 'dev-notes/' normix/ || (echo "docstrings must not reference dev-notes/"; exit 1)

# forbidden: README sends users to dev-notes/
! rg -n 'dev-notes/' README.md || (echo "README must not reference dev-notes/"; exit 1)
```

If any of these triggers, the docs build job fails. This is the primary
gatekeeper.

#### 3. Semantic — Sphinx itself

Sphinx already knows how to check links; we just turn the strictness up.

- **`nitpicky = True`** in `conf.py` — any unresolved cross-reference
  (`{ref}`, `{doc}`, `{py:class}`, etc.) becomes a warning. Combined with
  `sphinx-build -W` this becomes a hard error in CI.
- **`sphinx-build -b linkcheck`** — runs as a separate CI step (allowed to fail
  on master, hard-fail on release tag). Validates every external URL and
  every internal cross-reference.
- **`myst_heading_anchors = 3`** — generates stable heading anchors so MyST
  files can cross-link by header without breaking on rename.

Suggested CI step ordering in `docs.yml`:

```yaml
- name: Cross-link invariants
  run: scripts/check_doc_links.sh           # the grep block above

- name: Build (nitpicky, warnings as errors)
  run: uv run sphinx-build -W -b html docs docs/_build/html

- name: External link check
  run: uv run sphinx-build -b linkcheck docs docs/_build/linkcheck
  continue-on-error: true                   # report-only on PR
```

#### 4. Editorial — `dev-notes/README.md`

Top of `dev-notes/README.md` explicitly states:

> Material in `dev-notes/` is **not published** to the docs website. Do not
> link to it from `docs/`, from `normix/` docstrings, or from `README.md`.
> If a doc here should be public, promote it to `docs/` (typically
> `../design/`) first, then update incoming links.

This keeps the policy discoverable to any human reading the dev-notes tree
for the first time.

### Release path

Adding a docs-execution step to the release flow:

1. Maintainer triggers `docs-full.yml` (or pushes a `v*` tag).
2. All tutorials execute fresh against `master`. A failing cell blocks the
   release-please PR from merging.
3. After tag merges and the wheel is built, `docs.yml` runs once more to
   publish the executed site to `gh-pages`.

This gives a real "integration test layer before release" — exactly the
property the user asked for — without changing how the wheel itself is built.

### Local developer commands (added to `AGENTS.md`)

| Command | Purpose |
|---|---|
| `uv run make -C docs html` | Cached build, fast iteration |
| `uv run make -C docs html-strict` | `nb_execution_mode=force`, full re-execute |
| `uv run make -C docs clean` | Drop `_build/` (cache survives) |
| `uv run make -C docs clean-cache` | Drop `_build/.jupyter_cache` too |
| `uv run jupytext --to ipynb docs/tutorials/path.md` | Local interactive editing |

---

## Tutorial content plan

We are **building a comprehensive demo**, not porting `notebooks/` 1-to-1.
The current 16 notebooks are heavily redundant (one per distribution, each
re-exercising the same EF triad / EM / sampling). The new tutorial tree
follows feature coverage, with one tutorial per concept and shorter,
focused real-data experiments.

Coverage targets every public feature exported from `normix/__init__.py`.

### `docs/tutorials/core/` — framework concepts (4 tutorials)

| File | Demonstrates | Replaces / consolidates |
|---|---|---|
| `01_exponential_family.md` | `ExponentialFamily` ABC, three parametrizations (classical ↔ θ ↔ η), `_log_partition_from_theta`, the log-partition triad, `expectation_params(backend=)`, `fisher_information(backend=)` | parts of `exponential_distribution.ipynb` |
| `02_gh_family_tour.md` | The GH hierarchy: GIG → GH; degenerate limits to Gamma, IG, IG; how VG, NIG, NInvG arise as subordinator special cases | `distribution_conversions.ipynb` |
| `03_bessel_and_log_kv.md` | `log_kv(v, z, backend='jax'\|'cpu')`, four-regime dispatch, derivative recurrences, when to pick which backend | new |
| `04_random_sampling.md` | `rvs` on every distribution, Devroye TDR for GIG, PINV via `utils.rvs`, comparison vs scipy where available | scattered across current notebooks |

### `docs/tutorials/distributions/` — distribution tour (5 tutorials)

Consolidates the per-distribution notebooks into thematic groups. Each
tutorial demonstrates: `from_classical`, `from_natural`, `from_expectation`,
`pdf`, `cdf` (where available), `mean` / `var` / `std`, `rvs`, and either
moment-matching MLE or EM where applicable.

| File | Covers |
|---|---|
| `01_univariate_positive.md` | `Gamma`, `InverseGamma`, `InverseGaussian` — closed-form M-step, analytical triad overrides |
| `02_gig.md` | `GeneralizedInverseGaussian` — Bessel-heavy log-partition, η-rescaled multi-start solver, two sampling methods |
| `03_multivariate_normal.md` | `MultivariateNormal` — `L_Sigma` parametrization, EF round-trip, log-prob via `solve_triangular` |
| `04_normal_mixtures.md` | All four marginals (`VarianceGamma`, `NormalInverseGamma`, `NormalInverseGaussian`, `GeneralizedHyperbolic`); joint vs marginal layers; `Univariate*` for `d=1` |
| `05_factor_mixtures.md` | `FactorVarianceGamma`, `FactorNormalInverseGamma`, `FactorNormalInverseGaussian`, `FactorGeneralizedHyperbolic` — Woodbury, when `F F^T + diag(D)` wins over full `Σ` |

### `docs/tutorials/em/` — EM in practice (3 tutorials)

| File | Demonstrates |
|---|---|
| `01_batch_em.md` | `BatchEMFitter`, `EMResult`, convergence diagnostics, regularizations (`det_sigma_one`, `det_sigma_x`, `a_eq_b`), CPU vs JAX backends |
| `02_incremental_em.md` | `IncrementalEMFitter` with all six `EtaUpdateRule`s (Identity, RobbinsMonro, SampleWeighted, EWMA, Shrinkage, Affine); shrinkage targets; mini-batch curves |
| `03_initialization_and_multistart.md` | `default_init`, `from_expectation`, `theta0` warm-starts, multi-start via `jax.vmap`, comparison with single-start |

### `docs/tutorials/stats/` — statistical analysis (2 tutorials)

| File | Demonstrates |
|---|---|
| `01_divergences.md` | `squared_hellinger`, `kl_divergence`, the three-tier `*_from_psi` functional API, model comparison via Hellinger |
| `02_goodness_of_fit.md` | QQ plots, empirical vs fitted CDF, Kolmogorov-style diagnostics on synthetic + real data |

### `docs/tutorials/finance/` — real-data experiments (4 tutorials)

| File | Data | Demonstrates |
|---|---|---|
| `01_univariate_index.md` | S&P 500 daily returns | `Univariate*` fits to an index series, log-return tail behaviour, model comparison via Hellinger |
| `02_multivariate_stocks.md` | small basket from S&P 500 sample | Full-Σ `NormalMixture` fit, conditional expectations, EM convergence on real data |
| `03_factor_mixture_portfolios.md` | Dow Jones 30 daily returns | `FactorGeneralizedHyperbolic` for `d=30`, Woodbury solves, latent factor inspection |
| `04_cvar_optimization.md` | DJ30 returns | `normix.finance.projection.project_portfolio`, `CVaR(alpha)`, gradient / Hessian in `(μ̃, γ̃, σ̃)` and in `w`, Monte Carlo verification |

### Tutorial count summary

| Section | Tutorials | Approximate execution budget |
|---|---|---|
| core | 4 | < 5 min each |
| distributions | 5 | < 5 min each |
| em | 3 | < 10 min each |
| stats | 2 | < 5 min each |
| finance | 4 | 5–30 min each (heaviest) |
| **Total** | **18** | **< 2.5 h on a fresh full build** |

### What is NOT a tutorial

Anything that should not run in CI lives in `dev-notes/`:

- Performance benchmarks → `benchmarks/`
- Profiling notes → `dev-notes/tech_notes/`
- Investigations / dead ends → `dev-notes/investigations/`
- Numerical experiments comparing implementations → `dev-notes/tech_notes/`

---

## Migration phases

Each phase produces a green build, a working `gh-pages` site, and is mergeable
on its own.

### Phase 1 — Bootstrap MyST infrastructure + theme (1 PR, ~1 day) ✅

**Goal:** Sphinx still builds; `myst-nb` works alongside `nbsphinx` on one
prototype tutorial; the Kami-derived visual style is live on the prototype.

- [x] Add `myst-parser`, `myst-nb`, `sphinx-book-theme` to
  `[project.optional-dependencies].docs`.
- [x] Update `docs/conf.py`:
  - Add extensions and `nb_execution_*` settings (see § Build pipeline).
  - Switch `html_theme = "sphinx_book_theme"`.
  - Enable `myst_enable_extensions = ["dollarmath", "amsmath",
    "colon_fence", "deflist"]`.
  - `nitpicky = False` for now (docstring xref cleanup deferred; enable with
    `-W` in CI once warnings are cleared).
- [x] Port Kami tokens → `docs/_static/normix.css` (~350 lines).
- [x] Port `incerto.figures.set_theme()` → `normix/utils/plotting.py`
  (`set_theme()`, `COLORS`, `style_axes`, `savefig`).
- [x] Add `.cursor/rules/docs-cross-links.mdc` per § Cross-link discipline.
- [x] Add `scripts/check_doc_links.sh` and wire into `docs.yml` + `docs/Makefile`.
- [x] Convert `notebooks/em_vs_mcecm.ipynb` →
  `docs/tutorials/em/01_em_vs_mcecm.md` via jupytext; wired in `index.rst`.
- [x] Wire myst-nb cache into `docs.yml` (`actions/cache@v4`).
- [x] Keep `nbsphinx` enabled so existing `.ipynb` references still render.
- [x] Update `docs-publish` skill and `AGENTS.md` docs commands.
- [x] Verify gh-pages publishes after merge (CI deploy step) — `docs.yml`
  deploys `docs/_build/html` to `gh-pages` on push to `master`; site is live.

**Exit:**
- [x] Prototype tutorial renders with executed outputs locally (~132 s first run).
- [x] Site uses sphinx-book-theme + normix.css (warm paper, Charter serif, ink-blue accent).
- [x] `scripts/check_doc_links.sh` passes locally and in CI.

### Phase 2 — Structural split (1 PR, ~half day, source-only) ✅

**Goal:** Move internal material out of `docs/`; resolve naming duals.

- [x] Move plans, tech_notes, investigations, reviews, references, archive,
  ARCHITECTURE.md, and internal design/ → `dev-notes/`
- [x] Create public `docs/design/` (four topical docs + index.md)
- [x] Delete `docs/design.rst`, `docs/architecture.rst`
- [x] Minimal `exclude_patterns` in `docs/conf.py`
- [x] Update `AGENTS.md`, rules, skills, and cross-references
- [x] Structural CI invariant + updated `scripts/check_doc_links.sh`
- [x] `dev-notes/README.md` editorial header

**Exit:**
- [x] No legacy `docs/(plans|…|references)/` paths in agent-facing files
- [x] Live site verified after merge (design pages via MyST; architecture.rst removed) — published

### Phase 3 — Author the new tutorial tree (3–4 PRs) ✅ (authored and live)

**Goal:** Build the 18-tutorial demo from scratch, drawing material from
current notebooks but not constrained by their boundaries.

Cluster the work by section to keep PRs reviewable:

- [x] PR 3a: `tutorials/core/` (4 files) + `tutorials/distributions/` (5 files)
- [x] PR 3b: `tutorials/em/` (3 new files; existing prototype renumbered to
  `em/04_em_vs_mcecm.md`) + `tutorials/stats/` (2 files)
- [x] PR 3c: `tutorials/finance/` (4 files) — uses checked-in
  `data/sp500_returns.csv`
- [x] PR 3d: `getting_started/` (3 files) + `user_guide/` (5 files) +
  `index.md` toctree (+ `tutorials/index.md` for grouped nav)

Each PR runs the new section under `nb_execution_mode=cache` in CI; failures
block merge.

**Exit:**
- [x] All 18 tutorials author-complete and render with executed outputs in a
  local cached `make -C docs html`.
- [x] Toctree is complete (`index.md` → getting_started / user_guide /
  tutorials / theory / design / api).
- [x] Live site verified after merge — published to https://xshi19.github.io/normix/.

**Implementation notes (2026-05-28):**
- Source-of-truth landing page moved from `index.rst` to `index.md`;
  `quickstart.rst` content folded into `getting_started/` + `user_guide/`.
- **Finance data:** the standalone DJ30 CSV is *not* checked in
  (`scripts/download_dj30.py` needs `yfinance` + network, unavailable in CI), but
  the DJ30 constituents live inside the committed `data/sp500_returns.csv`.
  Tutorials read from that panel: an equal-weight index proxy for `01`, a
  5-stock basket for `02`/`04`, and the **DJ30 constituents** for `03` (29 of 30
  present — WBA, removed from the index in 2024, is absent from the panel).
- `VarianceGamma` is dropped from the `finance/01` index comparison (Normal vs
  NIG vs GH), but **not** because of "light tails" — that earlier explanation was
  wrong. The VG EM fits competitively (mll ≈ 3.21, improving monotonically) until
  it drives the Gamma subordinator shape α below 1, where the inverse moment
  E[1/Y] = β/(α−1) diverges; the covariance M-step's posterior 1/Y weights then
  blow up to `nan` (~iteration 19). Independent of regularization / M-step
  backend. Full repro + root cause:
  `dev-notes/investigations/variance_gamma_em_nan.{py,md}` (marimo). Suggested
  library fix (floor α > 1 or clamp the posterior 1/Y weights in the VG Σ update)
  is noted there, not yet implemented.
- **Cross-family divergences.** NIG, VG, NInvG *are* GH special cases, and
  `to_generalized_hyperbolic()` re-expresses them in the GH parametrization
  (shared `psi`), so a closed-form divergence between them is well-defined —
  `stats/01` now demonstrates this. What is **not** implemented: a closed-form
  *marginal* (over X) divergence. `squared_hellinger`/`kl_divergence` on mixtures
  return the **joint** (X, Y) divergence — the joint is exponential-family, the
  marginal is not — which is an upper bound on the marginal one. The bound is
  tight when models nearly agree (estimate vs target) but loose when latent
  subordinator parametrizations differ. For ranking *different* families as fits
  to data, use out-of-sample log-likelihood (`finance/01`) or a Monte-Carlo
  marginal Hellinger (`stats/01`).
- `api/index.rst` kept in place (not renamed to `reference/`); that rename is
  optional polish, out of Phase 3 scope.
- **myst-nb execution cache (not an error).** With `nb_execution_mode="cache"`,
  myst-nb hashes each tutorial's code + kernel and stores executed outputs under
  `docs/_build/.jupyter_cache`. On rebuild, a page whose hash is unchanged is
  served from cache instead of being re-executed (`Using cached notebook: ID=…`).
  The plan log noting `em/04_em_vs_mcecm.md` "hits the cache after rename" just
  means the rename did not change cell contents, so its prior cached run was
  reused — a feature (fast incremental builds), not a problem. Every code cell
  was also independently verified to execute against the live source. A forced
  full re-execution is `make -C docs html-strict` (`NB_EXECUTION_MODE=force`).

### Phase 4 — Retire `nbsphinx` and `notebooks/` (1 PR) ⬜ PENDING

**Goal:** Drop the old toolchain; the new tutorials are the only ones.

> **Not started.** `nbsphinx` is still listed in `docs/conf.py` (extensions +
> `nbsphinx_*` settings), in `pyproject.toml`'s docs extra, and in the
> docs-publish skill. `notebooks/` still holds the 16 legacy `.ipynb` (plus
> `em_shrinkage_demo.py`). `finance_phase_d_cvar_demo.ipynb` is superseded by
> `docs/tutorials/finance/04_cvar_optimization.md` and can be dropped here.

- Remove `nbsphinx` from extensions and from `[project.optional-dependencies].docs`.
- Delete `docs/conf.py` `nbsphinx_*` settings.
- `git rm -r notebooks/` (history preserves the old files).
- Add `notebooks/` and `*.ipynb` to `.gitignore`.
- Update `AGENTS.md` and any rule files that mention `notebooks/`.
- Update the README and `docs/getting_started/quickstart.md` if they link to
  `notebooks/*.ipynb`.

**Exit:** `rg -l '\.ipynb' docs/ AGENTS.md README.md .cursor/` returns no hits;
`docs.yml` builds without `nbsphinx`.

### Phase 5 — Add the release tier (1 PR) ⬜ PENDING

**Goal:** A second workflow that forces full re-execution on release tags
and `workflow_dispatch`.

> **Not started.** `.github/workflows/docs-full.yml` does not exist yet. The
> existing `docs.yml` already supports `workflow_dispatch` and respects the
> `NB_EXECUTION_MODE` env var (`docs/conf.py` reads it), so the release tier is
> mostly a copy with `NB_EXECUTION_MODE=force` and a longer `timeout-minutes`.

- Add `.github/workflows/docs-full.yml` (copy `docs.yml`, set
  `NB_EXECUTION_MODE=force`, longer `timeout-minutes`).
- Wire it to `on: push: tags: ['v*']` and `workflow_dispatch:`.
- Document the trigger flow in `dev-notes/plans/docs_refactor.md` (status
  update) and the `docs-publish` skill.

**Exit:** Dispatching the new workflow runs every tutorial fresh and
publishes to `gh-pages`.

### Phase 6 — Polish (open-ended, optional) ⬜ OPTIONAL

- Migrate `../../docs/theory/*.rst` to MyST `.md` opportunistically when touched.
- Swap theme (`furo` or `pydata-sphinx-theme`) for better navigation.
- Add PR-preview deploys (Netlify-style) if maintainer load justifies it.
- Add a `docs/changelog.md` that includes `CHANGELOG.md` directly.

No exit criterion; tackled as time allows.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `myst-nb` cache misses on every PR if cache key is wrong | Key on `hashFiles('docs/tutorials/**', 'uv.lock', 'pyproject.toml')`. Manual cache bust via re-running `docs-full.yml`. |
| Heavy tutorial silently takes >6 h and burns CI | Per-file `execution_timeout`; tutorials over 30 min carry `execution_mode: off` and ship cached outputs. |
| Non-deterministic outputs (random seeds, timing logs) thrash the cache | Each tutorial cell-1 sets seeds explicitly; timing prints suppressed via cell tags. Enforce in PR review. |
| Live site goes dark during phases | Each phase is independently mergeable. Phase 1 keeps `nbsphinx` for backward compat; phases 2–3 don't touch the published toctree until ready. |
| Internal docs accidentally leak to website | Structural invariant check in `docs.yml` (Phase 2 exit criterion). |
| Cross-links break (e.g. `docs/design.md → ../plans/...`) | One pass with `rg -l 'docs/(plans|tech_notes|investigations|reviews|archive|references)/'` at Phase 2 end. CI link-check via `sphinx-build -b linkcheck` added in Phase 5. |
| Tutorials drift out of sync with API | Tutorials run in CI on every push. A renamed symbol breaks the build before the PR merges. |

## Resolved decisions (2026-05-25)

- **Internal-tree name.** `dev-notes/`. Visible in `ls`, semantically clear.
- **Base theme.** `sphinx-book-theme` + `docs/_static/normix.css` port of the
  Kami-derived tokens from `incerto-wiki/../design/VISUAL_STYLE.md`.
- **Matplotlib helper.** Extend `normix/utils/plotting.set_theme()` to install
  the Incerto-token `rcParams` so tutorial plots match the site visually.
- **Cross-link enforcement.** Four layers: agent rule, CI grep invariants,
  Sphinx `nitpicky` + `-W` + `linkcheck`, and editorial header in
  `dev-notes/README.md`. See § Cross-link discipline.

## Open questions

- **Versioned docs.** Should `gh-pages` carry `latest/` + tagged versions
  (`v0.2.x/`, `v0.3.x/`), or only `latest/`? Recommendation: defer; current
  pace doesn't justify it yet.
  Answer: defer, only latest for now
- **ReadTheDocs fallback.** Worth setting up as a parallel target so we can
  cut over if `docs.yml` ever becomes a bottleneck? Recommendation: keep as
  a documented escape hatch, do not configure yet.
  Answer: agree
- **Matplotlib theme rollout.** Apply `set_theme()` to every tutorial in
  Phase 3, or only to new ones? Recommendation: apply universally so the
  finished tutorial tree is visually homogeneous from day one.
  Answer: agrer, apply to all

## Related plans / docs (current state)

- `../archive/plans/migration_plan.md` — JAX migration, complete (archived).
- `../archive/plans/package_improvement_roadmap.md` — review-driven cleanup,
  complete (archived).
- `finance_architecture.md` — Phase D done; E, F still proposed.
- `.cursor/skills/docs-publish/SKILL.md` — current build/publish recipe; will
  be updated in Phase 1 and Phase 5.
- `AGENTS.md` § Context Map — internal-doc index, updated in Phase 2.
