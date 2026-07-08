---
name: docs-publish
description: >-
  Build and publish the normix documentation website. Use when asked to compile
  Sphinx docs, refresh https://xshi19.github.io/normix/, or debug why published
  docs are stale. Use when changing docs/ and needing the website updated. This
  skill encodes the correct deploy target: the public normix docs site is served
  from the normix repo's gh-pages branch, not from the xshi19.github.io repo.
---

# Docs Publish

Use this skill when the task is to build or publish the `normix` docs website.

## Canonical facts

- Docs source lives under `docs/`
- Build toolchain: Sphinx + `myst-parser` + `myst-nb` + `sphinx-book-theme`
- Executable tutorials live at `docs/tutorials/**/*.md` (MyST notebooks)
- Build command is from `docs/Makefile` (runs `scripts/check_doc_links.sh` first)
- Build output is `docs/_build/html`
- myst-nb execution cache: `docs/_build/.jupyter_cache`
- Public URL `https://xshi19.github.io/normix/` is served from the **`normix` repo `gh-pages` branch**
- It is **not** served from the `xshi19.github.io` repo

## Local build

```bash
uv sync --extra docs --extra plotting
uv run make -C docs html          # cached tutorial execution (fast)
uv run make -C docs html-strict   # NB_EXECUTION_MODE=force, -W (full re-execute)
uv run make -C docs clean         # drop html/doctrees
uv run make -C docs clean-cache   # also drop .jupyter_cache
```

For interactive editing of a MyST tutorial:

```bash
uv run jupytext --to ipynb docs/tutorials/path/to/tutorial.md
# local .ipynb is not checked in
```

## Normal publish path (CI)

Docs are deployed automatically by `.github/workflows/docs.yml` on every push
to `master` / `main`. The workflow:

1. Runs `scripts/check_doc_links.sh` (cross-link invariants)
2. Restores myst-nb cache keyed on `normix/**`, `docs/tutorials/**`, `uv.lock`,
   `pyproject.toml` (the `normix/**` component forces re-execution when library
   source changes)
3. Builds HTML via `uv run sphinx-build -b html docs docs/_build/html`
4. Runs linkcheck (report-only, `continue-on-error`)
5. Deploys to `gh-pages` on push to default branch

**Do not run the local publish script unless CI is broken or you need to
publish from a non-default branch.**

To trigger a docs update: merge or push to `master` / `main` and let the workflow run.

## Release publish path (full re-execution)

`.github/workflows/docs-full.yml` is a second workflow for release tags. Unlike
`docs.yml`, it forces every tutorial to re-execute from scratch (no myst-nb
cache) and turns linkcheck into a hard gate instead of a report:

1. Triggers: `push: tags: ['v*']` (fires automatically alongside `publish.yml`
   on a release-please tag) or `workflow_dispatch` (manual run from the
   Actions tab, any ref).
2. Builds with `NB_EXECUTION_MODE=force` — no cache restore/save step, so a
   stale cached figure can never slip into a release build.
3. Runs `sphinx-build -b linkcheck` **without** `continue-on-error`: a broken
   external link fails the workflow.
4. Deploys to `gh-pages` unconditionally (tag push or manual dispatch both
   deploy).
5. `timeout-minutes: 360` — full re-execution of the tutorial tree is much
   slower than the cached `docs.yml` build; budget for it.
6. Separate concurrency group (`pages-release`, `cancel-in-progress: false`)
   so a routine push to `master` running `docs.yml` cannot cancel an
   in-progress release build (they'd otherwise share `docs.yml`'s `pages`
   group and `cancel-in-progress: true` would kill the release run).

Use `workflow_dispatch` to manually force a full re-execution + republish
without cutting a release (e.g. after suspecting a stale cached figure, or
before a release to catch linkcheck failures early).

## Fallback: local build + publish

Use only when the CI workflow is unavailable or broken.

1. If the user asked for latest changes, pull the desired branch first.
2. If the repo has local notebook/worktree changes, preserve them before pulling.
   - Stash if safe and clearly restore afterward.
   - Avoid clobbering notebook work.
3. Build docs locally:

```bash
uv sync --extra docs --extra plotting
uv run make -C docs clean
uv run make -C docs html
```

Build warnings do not necessarily block publishing. If HTML is produced under
`docs/_build/html`, the site can still be deployed unless the user asked to fix
warnings first.

4. Publish to the `gh-pages` branch of **this repo**:

```bash
.cursor/skills/docs-publish/scripts/publish_gh_pages.sh
```

Run it from the repo root after a successful build.

## Verification

After push, check the GitHub Pages run for `gh-pages` and verify the live URL:

- `https://xshi19.github.io/normix/`
- `https://xshi19.github.io/normix/index.html`
- Tutorials landing: `https://xshi19.github.io/normix/tutorials/index.html`
- EM vs MCECM tutorial: `https://xshi19.github.io/normix/tutorials/em/04_em_vs_mcecm.html`

Do not assume the site is updated until the `pages build and deployment` run on
`gh-pages` is complete.

## Gotchas

- **Wrong repo trap**: Updating `xshi19.github.io/normix/` does not update the public normix docs URL.
- **Pages source**: The public docs site comes from `normix:gh-pages`.
- **Jekyll trap**: Keep `.nojekyll` on `gh-pages` so Sphinx `_static/`, `_sources/`, `_modules/` are served correctly.
- **Worktree trap**: If using a temporary worktree for `gh-pages`, do not delete the worktree's `.git` pointer file while clearing content.
- **Notebook trap**: Pull/build requests may happen while local notebooks are dirty. Preserve and restore them.
- **Library-fix / stale-figure trap**: myst-nb's `execution_mode: cache` keys the
  execution cache on notebook *cell source* only — not on the `normix` library
  that produces the outputs. A fix to `normix/**` therefore does NOT invalidate
  cached figures, so a normal cached build republishes the old (buggy) figure
  even though the workflow "succeeded". The CI cache key includes `hashFiles('normix/**')`
  to force re-execution on library changes. For a one-off local republish after a
  library fix, use `uv run make -C docs html-strict` (force) or `clean-cache` first.
- **Tutorial runtime**: the full tutorial tree (`docs/tutorials/**`, 18 executable pages) runs in a few minutes on a fresh build; `em/04_em_vs_mcecm` is the long pole (~2 min, sweeps 21 values of $p$ with EM + MCECM). CI restores the myst-nb cache from prior master builds.
- **`notebooks/` is not published**: it's a personal research workspace (two-tier `.ipynb`-scratch / jupytext-`.py`-preserved policy, see `notebooks/README.md`), not built by Sphinx.

## Related files

- `docs/Makefile`
- `docs/conf.py`
- `docs/_static/normix.css`
- `scripts/check_doc_links.sh`
- `.github/workflows/docs.yml`
- `.github/workflows/docs-full.yml`
- `.cursor/skills/docs-publish/scripts/publish_gh_pages.sh`
- `dev-notes/plans/docs_refactor.md` — migration status
- `AGENTS.md`
