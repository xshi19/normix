# Docs & Notebooks: remaining work

> **ACTIVE — refreshed 2026-07-04.**
> **Done (archived):** Phases 1–3 — MyST + `myst-nb` infrastructure, Kami visual
> theme, internal/external split (`docs/` vs `dev-notes/`), and the full tutorial
> tree (21 tutorials + getting_started + user_guide + design pages) are **built
> and published** to https://xshi19.github.io/normix/. Full historical plan:
> [`../archive/plans/docs_refactor_phases_1_3.md`](../archive/plans/docs_refactor_phases_1_3.md).
> **Done:** Phase 4 — notebook strategy (revised two-tier policy; `nbsphinx`
> retired). Phase 5 — website correctness (version string, API reference
> completeness + restructure, changelog page).
> **Remaining:** distribution gallery (Phase 6), release execution tier
> (Phase 7), open-ended polish (Phase 8).
> **Scope:** `docs/`, `notebooks/`, `.gitattributes`, `.github/workflows/`,
> `docs/conf.py`, the docs-publish skill, notebook-guidelines rule.
> **Does not touch:** `normix/` source (except `conf.py`-adjacent metadata and
> surgical docstring fixes — Phase 5 fixed broken `:doc:` cross-links and
> Sphinx-xref-ambiguous docstring formatting that Phase 5b's completeness work
> newly exposed), `tests/`, `benchmarks/`.

---

## Where we are

- The MyST website is the primary public demo surface. All tutorial content is
  authored as executable MyST `.md` under `docs/tutorials/` and runs in CI on
  every push.
- `notebooks/` now follows the two-tier policy (Phase 4, done): `*.ipynb` is
  `.gitignore`d scratch; `em_shrinkage_demo.py` (marimo) and
  `varentropy_validation.py` (jupytext percent) are the committed Tier 2
  notebooks. The 16 legacy tutorial notebooks (superseded by the tutorial
  tree) were removed with `git rm`; history preserves them.
- `nbsphinx` has been removed from `docs/conf.py`, `pyproject.toml`, and
  `uv.lock`.
- `*.ipynb linguist-vendored` is in `.gitattributes`, so any accidentally
  committed `.ipynb` no longer skews GitHub's language stats.
- `docs/conf.py` derives `release`/`version` from installed package metadata
  (Phase 5a, done) — the site header can no longer drift from `pyproject.toml`.
  `github_version` fixed to `master`.
- The API reference is restructured into `docs/api/{distributions,mixtures,
  fitting,finance,utils}.rst` plus a slim `index.rst` landing page (Phase 5b,
  done); every public module in `normix/__init__.py` is now reachable.
- `docs/changelog.md` `{include}`s the root `CHANGELOG.md`, wired into the
  Reference toctree (Phase 5c, done).

---

## Phase 4 — Notebook strategy (revised) ✅ DONE

The original Phase 4 said "delete `notebooks/`". **Revised decision:** the MyST
site remains the only *published* demo, but Jupyter notebooks stay as the
*personal research* workspace — they are the most ergonomic scratchpad
(marimo was evaluated and rejected: no Cursor autocomplete; Jupyter `.ipynb`
has first-class Cursor support). The problems to solve are narrower:

1. GitHub language stats must not be dominated by Jupyter.
2. Repo bloat / unreadable diffs from committed outputs.
3. Redundancy: 16 legacy tutorial notebooks duplicate the published tutorials.

### Decision: two-tier notebook policy

**Tier 1 — scratch (default).** `notebooks/` is a local research workspace.
`*.ipynb` is `.gitignore`d; nothing is committed. Work in Jupyter/Cursor as
usual with zero friction.

**Tier 2 — preserved research.** A notebook worth keeping across machines is
committed as a **jupytext percent-format `.py`** pair:

```bash
uv run jupytext --set-formats ipynb,py:percent notebooks/my_study.ipynb
# thereafter: edit either file; sync with
uv run jupytext --sync notebooks/my_study.ipynb
```

- The `.py` percent file is committed; the paired `.ipynb` stays gitignored.
- Linguist counts it as **Python**; diffs are readable; no output bloat.
- Cursor autocomplete on a percent `.py` is plain-Python-grade (better than
  `.ipynb`); Jupyter Lab opens percent files as notebooks natively (double-click
  → Open With → Notebook), so the GUI workflow is preserved.
- Outputs/figures are not stored in git — acceptable for research; anything
  demo-worthy graduates to `docs/tutorials/` where outputs are produced at
  build time. `notebooks/em_shrinkage_demo.py` already follows this pattern.

**Belt-and-braces:** add linguist overrides to `.gitattributes` so that any
`.ipynb` that *does* get committed (accidentally or deliberately) never skews
language stats:

```gitattributes
*.ipynb linguist-vendored
```

This one line fixes the ">90% Jupyter" display immediately, independent of the
cleanup below (stats refresh on the next push to the default branch).

### Checklist (1 PR, ~half day)

- [x] Add `*.ipynb linguist-vendored` to `.gitattributes` (quick win — can ship
  first, alone).
- [x] `git rm` the 16 legacy tutorial notebooks (superseded by
  `docs/tutorials/`; history preserves them). Includes
  `finance_phase_d_cvar_demo.ipynb` (superseded by
  `tutorials/finance/04_cvar_optimization.md`).
- [x] Convert `varentropy_validation.ipynb` → jupytext percent `.py` (Tier 2);
  remove the `.ipynb`.
- [x] Add `notebooks/*.ipynb` to `.gitignore`; keep `notebooks/` with a short
  `README.md` documenting the two-tier policy and the jupytext commands.
- [x] Remove `nbsphinx` from `docs/conf.py` (extension + `nbsphinx_*` settings)
  and from `[project.optional-dependencies].docs` in `pyproject.toml` (also
  dropped the stray `nbsphinx` pin in `[dependency-groups].dev`; `uv lock`
  refreshed).
- [x] Update `.cursor/rules/notebook-guidelines.mdc` to describe the two-tier
  policy (scratch vs jupytext-paired) instead of `.ipynb` conventions.
- [x] Update `AGENTS.md` (notebooks row in context map), the docs-publish
  skill, and README if they reference `.ipynb` paths.

**Exit (verified):** `git ls-files '*.ipynb'` is empty; `rg -l nbsphinx docs/
pyproject.toml` is empty; local `uv run make -C docs html` builds green
without `nbsphinx`. GitHub language bar Python-dominance confirms on the
next push to the default branch.

---

## Phase 5 — Website correctness fixes ✅ DONE

Bugs and drift on the live site, cheapest-first:

### 5a. Version string (trivial) ✅

`docs/conf.py` now derives `release`/`version` from installed package
metadata via `importlib.metadata.version("normix")`, so the header can never
drift from `pyproject.toml` again. `html_context['github_version']` fixed
`'main'` → `'master'` (the actual default branch; "edit on GitHub" links were
404ing).

### 5b. API reference completeness ✅

`docs/api/index.rst` was missing entire public subpackages. All now
documented — `normix.divergences`, `normix.mixtures.factor`,
`normix.fitting.{eta,eta_rules,shrinkage_targets}`,
`normix.finance.{projection,risk,optimization,functional}`,
`normix.utils.{plotting,rvs,gammainc,validation}`. (`Factor*`/`Univariate*`
distribution variants were already covered automatically — they live in the
same module as their parent joint/marginal class, so the existing
`automodule :members:` picked them up without any new directive.)

Restructured into one page per subpackage: `api/distributions.rst`,
`api/mixtures.rst`, `api/fitting.rst`, `api/finance.rst`, `api/utils.rst`,
with `api/index.rst` as a slim landing page (Base Classes + Divergences +
toctree). The single page was already ~160 lines and would only have grown.

Exposing these previously-unrendered modules surfaced pre-existing docstring
bugs that only manifest through Sphinx (invisible in code review): broken
`:doc:` cross-links using a doc-relative path convention that doesn't hold
once a docstring is included from a different page
(`normix/fitting/eta.py`, `eta_rules.py`, `shrinkage_targets.py`,
`normix/mixtures/joint.py` — fixed by switching to source-root-absolute
`:doc:` paths, e.g. `` :doc:`/theory/shrinkage` ``), a numpydoc `Attributes`
section on `NormalMixtureEta`/`FactorMixtureStats` that duplicated autodoc's
automatic `eqx.Module` field listing (fixed by moving the per-field math into
`#:` attribute comments, matching the convention used by every other
distribution in the codebase — no other class uses an `Attributes` section),
and several numpydoc `Parameters` entries with a bare dimension letter `d` in
the type position (`normix/fitting/{em,solvers,shrinkage_targets}.py`) that
Napoleon tried to cross-reference, ambiguously, against every distribution's
`.d` property (fixed by moving shape info into the description text). These
were docstring-only, surgical fixes — no logic changes.

### 5c. Changelog page ✅

Added `docs/changelog.md` that `{include}`s the root `CHANGELOG.md`
(`:start-line: 1` skips the duplicate `# Changelog` H1; release-please keeps
the source current). Wired into the `Reference` toctree in `docs/index.md`.

**Exit (verified):** local build shows `normix 0.2.7` in the header;
`git ls-files` of every module referenced by `normix/__init__.py`'s `__all__`
has a matching `automodule`/`autoclass` directive in `docs/api/*.rst`
(spot-checked via `rg -o` diff); `docs/changelog.html` renders the version
history; `uv run make -C docs clean && uv run sphinx-build -b html docs
docs/_build/html` builds green — the only remaining warnings (9) are
pre-existing docutils/RST formatting nits unrelated to this phase (verified
present before Phase 5 by re-running on `git stash`).

---

## Phase 6 — Distribution gallery ⬜ PROPOSED

The old repo had one demo notebook per distribution; the tutorial tree
deliberately consolidated them into 5 thematic tours, so there is no longer a
per-distribution landing page. That is a real gap: "what does normix's NIG look
like and how do I use it" deserves a direct, linkable answer (the way
`scipy.stats` has one page per distribution) — without resurrecting the 16
redundant notebooks.

**Proposal:** `docs/distributions/` — one compact executable MyST page per
distribution (9 core + factor variants grouped on one page ≈ 10 pages), each
following a fixed template:

1. **Density gallery** — pdf across 3–4 parameter settings (one small figure,
   `set_theme()` styled).
2. **Parametrization table** — classical ↔ natural $\theta$ ↔ expectation
   $\eta$, with the stored-attribute names.
3. **Quick usage** — `from_classical` → `pdf` / `rvs` / `fit` in ~10 lines.
4. **Cross-links** — API class, theory page, and the thematic tutorial that
   exercises it in depth.

Each page is small (< 1 min execution) so CI cost is negligible. The landing
page gets a visual index (grid of density thumbnails) — doubles as the "hero"
demo of the family tree.

Order pages by hierarchy: GIG → (Gamma, InverseGamma, InverseGaussian limits) →
MultivariateNormal → mixtures (VG, NInvG, NIG, GH) → factor variants — the
gallery then *narrates* the GH family structure, which is normix's core pitch.

**Exit:** every distribution in the README table has a linkable page with a
density plot; gallery index wired into the main toctree between User guide and
Tutorials.

---

## Phase 7 — Release execution tier ⬜ PENDING (unchanged from original plan)

A second workflow forcing full tutorial re-execution on release tags.

- [ ] Add `.github/workflows/docs-full.yml` (copy `docs.yml`, set
  `NB_EXECUTION_MODE=force`, `timeout-minutes: 360`, ignore the myst-nb cache).
- [ ] Wire to `on: push: tags: ['v*']` and `workflow_dispatch:`.
- [ ] Promote linkcheck from report-only to hard-fail in this workflow only.
- [ ] Document the trigger flow here and in the docs-publish skill.

`docs/conf.py` already reads `NB_EXECUTION_MODE` from the env, so this is
mostly workflow YAML.

**Exit:** dispatching the workflow re-executes every tutorial fresh and
publishes to `gh-pages`.

---

## Phase 8 — Polish (open-ended, optional)

Website structure ideas beyond correctness, roughly in value order:

- **Landing page cards.** Use `sphinx-design` grid cards for the main sections
  (Getting started / User guide / Tutorials / Theory / API) instead of the
  current prose-only "Where to start". Add a hero figure — the distribution
  gallery thumbnail grid from Phase 6 is the natural candidate.
- **Intersphinx** to `jax`, `equinox`, `scipy` so type references in autodoc
  link out.
- **`api/` → `reference/` rename** (deferred from Phase 3; the 5b restructure
  happened without it — still optional, now a standalone rename).
- **Migrate `docs/theory/*.rst` → MyST `.md`** opportunistically when touched.
- **PR-preview deploys** if maintainer load ever justifies it.
- **PyPI badge / install matrix** on the landing page once the package is
  published to PyPI.

No exit criterion; tackled as time allows.

---

## Suggested ordering

1. ~~**Phase 4**~~ — done (notebook cleanup + nbsphinx retirement).
2. ~~**Phase 5**~~ — done (version string, API reference restructure +
   completeness, changelog).
3. **Phase 6** — distribution gallery (the largest new-content item).
4. **Phase 7** — release tier.
5. **Phase 8** — as time allows.

## Risks

| Risk | Mitigation |
|---|---|
| jupytext pairing drifts (edits `.py` without syncing `.ipynb` or vice versa) | pairing is per-file metadata; `jupytext --sync` is idempotent. Scratch tier has no pairing at all. |
| Deleting legacy notebooks loses figures not yet ported | they are superseded page-for-page by `docs/tutorials/`; git history preserves the `.ipynb` blobs regardless. |
| Gallery pages duplicate tutorial content | fixed 4-block template keeps them reference-style, not narrative; deep dives stay in tutorials. |
| API restructure breaks inbound `{py:class}` links | Sphinx resolves by object name, not page; `linkcheck` in Phase 7 catches stragglers. |

## Open questions

- **Repo size.** History already carries ~90 MB of packfiles (old notebook
  outputs). Rewriting history (`git filter-repo`) would shrink clones but breaks
  forks/clones; **recommendation: don't** — stop adding blobs (Phase 4) and let
  it be.
- **Gallery scope.** Include the `Univariate*` marginal wrappers as separate
  pages or as sections of their multivariate parents? Recommendation: sections —
  they share parameters.

## Related plans / docs

- [`../archive/plans/docs_refactor_phases_1_3.md`](../archive/plans/docs_refactor_phases_1_3.md)
  — archived Phases 1–3 (target architecture, visual style, cross-link
  discipline, tutorial content plan).
- `finance_architecture.md` — Phase E merged (mean-risk optimization); F proposed.
- `.cursor/skills/docs-publish/SKILL.md` — build/publish recipe; updated in
  Phase 4 (`notebooks/` gotcha), update again in Phase 7.
- `.cursor/rules/notebook-guidelines.mdc` — rewritten in Phase 4.
- `notebooks/README.md` — two-tier policy, added in Phase 4.
- `AGENTS.md` § Context Map — notebooks row checked in Phase 4 (no change needed).
