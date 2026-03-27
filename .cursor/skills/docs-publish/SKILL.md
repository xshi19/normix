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
- Build command is from `docs/Makefile`
- Successful local build pattern:
  - activate `.venv`
  - `cd docs`
  - `make clean`
  - `make html`
- Build output is `docs/_build/html`
- Public URL `https://xshi19.github.io/normix/` is served from the **`normix` repo `gh-pages` branch**
- It is **not** served from the `xshi19.github.io` repo

## Recommended workflow

1. If the user asked for latest changes, pull the desired branch first.
2. If the repo has local notebook/worktree changes, preserve them before pulling.
   - Stash if safe and clearly restore afterward.
   - Avoid clobbering notebook work.
3. Build docs locally.
4. Publish `docs/_build/html` to `gh-pages`.
5. Ensure `.nojekyll` exists on `gh-pages`.
6. Verify the live site after the Pages deploy completes.

## Build

From the repo root:

```bash
. .venv/bin/activate
cd docs
make clean
make html
```

Build warnings do not necessarily block publishing. If HTML is produced under
`docs/_build/html`, the site can still be deployed unless the user asked to fix
warnings first.

## Publish target

Publish to the `gh-pages` branch of **this repo**.

Use the helper script:

```bash
.cursor/skills/docs-publish/scripts/publish_gh_pages.sh
```

Run it from the repo root after a successful build.

## Verification

After push, check the GitHub Pages run for `gh-pages` and verify the live URL:

- `https://xshi19.github.io/normix/`
- `https://xshi19.github.io/normix/index.html`

Do not assume the site is updated until the `pages build and deployment` run on
`gh-pages` is complete.

## Gotchas

- **Wrong repo trap**: Updating `xshi19.github.io/normix/` does not update the public normix docs URL.
- **Pages source**: The public docs site comes from `normix:gh-pages`.
- **Jekyll trap**: Keep `.nojekyll` on `gh-pages` so Sphinx `_static/`, `_sources/`, `_modules/` are served correctly.
- **Worktree trap**: If using a temporary worktree for `gh-pages`, do not delete the worktree's `.git` pointer file while clearing content.
- **Notebook trap**: Pull/build requests may happen while local notebooks are dirty. Preserve and restore them.

## Related files

- `docs/Makefile`
- `docs/conf.py`
- `.cursor/skills/docs-publish/scripts/publish_gh_pages.sh`
- `AGENTS.md`
