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
2. Restores myst-nb cache keyed on `docs/tutorials/**`, `uv.lock`, `pyproject.toml`
3. Builds HTML via `uv run sphinx-build -b html docs docs/_build/html`
4. Runs linkcheck (report-only, `continue-on-error`)
5. Deploys to `gh-pages` on push to default branch

**Do not run the local publish script unless CI is broken or you need to
publish from a non-default branch.**

To trigger a docs update: merge or push to `master` / `main` and let the workflow run.

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
- Prototype tutorial: `https://xshi19.github.io/normix/tutorials/em/01_em_vs_mcecm.html`

Do not assume the site is updated until the `pages build and deployment` run on
`gh-pages` is complete.

## Gotchas

- **Wrong repo trap**: Updating `xshi19.github.io/normix/` does not update the public normix docs URL.
- **Pages source**: The public docs site comes from `normix:gh-pages`.
- **Jekyll trap**: Keep `.nojekyll` on `gh-pages` so Sphinx `_static/`, `_sources/`, `_modules/` are served correctly.
- **Worktree trap**: If using a temporary worktree for `gh-pages`, do not delete the worktree's `.git` pointer file while clearing content.
- **Notebook trap**: Pull/build requests may happen while local notebooks are dirty. Preserve and restore them.
- **Tutorial runtime**: `01_em_vs_mcecm` sweeps 21 values of $p$ with EM + MCECM; first uncached build can take ~2 min. CI restores cache from prior master builds.
- **Legacy notebooks**: `notebooks/*.ipynb` still render via `nbsphinx` until Phase 4 retires them.

## Related files

- `docs/Makefile`
- `docs/conf.py`
- `docs/_static/normix.css`
- `scripts/check_doc_links.sh`
- `.github/workflows/docs.yml`
- `.cursor/skills/docs-publish/scripts/publish_gh_pages.sh`
- `dev-notes/plans/docs_refactor.md` — migration status
- `AGENTS.md`
