# notebooks/

This directory is a **personal research workspace**, not published content.
The published, executable demo surface is `docs/tutorials/` (built to
https://xshi19.github.io/normix/). Nothing here is built or deployed.

## Two-tier policy

**Tier 1 — scratch (default).** `*.ipynb` is `.gitignore`d; nothing is
committed. Work in Jupyter/Cursor as usual with zero friction — create,
edit, and delete notebooks freely.

**Tier 2 — preserved research.** A notebook worth keeping across machines
is committed as a **jupytext percent-format `.py`** pair instead of the
`.ipynb` itself:

```bash
uv run jupytext --set-formats ipynb,py:percent notebooks/my_study.ipynb
# thereafter: edit either file, then sync with
uv run jupytext --sync notebooks/my_study.ipynb
```

- The `.py` percent file is committed; the paired `.ipynb` stays
  `.gitignore`d.
- Linguist counts it as **Python** (readable diffs, no output/image bloat).
- Jupyter Lab opens percent files as notebooks natively (double-click →
  *Open With* → *Notebook*), so the GUI workflow is unchanged.
- Outputs/figures are not stored in git — acceptable for research.
  Anything demo-worthy graduates to `docs/tutorials/`, where outputs are
  produced fresh at build time.

`varentropy_validation.py` (jupytext percent) is the current Tier 2
notebook. `em_shrinkage_demo.py` is a marimo notebook — marimo's native
format is already plain Python, so it needs no jupytext pairing; run it
with `uv run marimo edit notebooks/em_shrinkage_demo.py`.

## Why not delete this directory

Jupyter `.ipynb` remains the most ergonomic scratchpad for exploratory work
(first-class Cursor autocomplete; marimo was evaluated and rejected for lack
of it). The problems this policy solves are narrower than "no notebooks":
keeping GitHub's language stats Python-dominant, avoiding repo bloat from
committed outputs, and not duplicating the published tutorial tree. See
`dev-notes/plans/docs_refactor.md` (Phase 4) for the full rationale.
