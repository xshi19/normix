---
name: show-me
description: >-
  Explain the current topic visually: the smallest diagram, sketch, or
  HTML artifact that makes the point. Use when the user asks to show,
  visualize, sketch, or diagram a design, algorithm, derivation, or
  code shape; when a research note or comparison is easier to review
  as a page than as chat; or when they say show-me. Not for durable
  docs (those go to docs/ or dev-notes/) and not for plots matplotlib
  already covers.
---

# Show me

Help the user see the current topic. Skip the preamble; keep prose
brief. Pick the **smallest view** that makes the key point clear —
you will rarely use every form below.

This skill produces **conversation-scoped** artifacts. Durable notes
belong in `docs/` or `dev-notes/` (promote, don't accumulate a third
doc tree). Unrelated to pstack's on-hold `show-me-your-work`
(decision-log TSVs in `dev-notes/plans/loops_and_orchestration.md`).

Copyable HTML skeleton: `references/html-template.html`. Worked
example: `references/specimen.html`.

## View hierarchy (smallest first)

- **Logic / algorithm** → short pseudocode:

```text
on(save)
  if content is unchanged
    return cached result
  write new content
  return fresh result
```

- **Runtime control flow** → call tree:

```text
fit
  e_step
    posterior_gig
  m_step
    from_expectation
```

- **Module responsibility / refactor shape** → shallow file tree,
  one-line comments on the directories that matter.
- **Component interaction, control flow, data flow** → Mermaid in
  chat. Stick to types Cursor chat actually renders: `flowchart`,
  `sequenceDiagram`, `stateDiagram`, `classDiagram`, `erDiagram`,
  `xychart`. `timeline`, `gantt`, `mindmap`, `pie`, `gitGraph` fall
  back to a syntax error — use a flowchart or an HTML artifact.
  The Mermaid type list is Cursor-only.
  On Codex, use pseudocode, a call tree, a file tree, a diff, or HTML.
- **What changes**, when the surrounding shape already exists → a
  `diff` of that shape (call tree, file tree, pseudocode, or code),
  not a wall of new prose.
- **Formulas** → LaTeX in chat (`$...$` / `$$...$$`). Cursor chat
  renders it. On Codex, leave the LaTeX in the reply.
  Do not assume a Cursor math preview.
  Reach for KaTeX-in-HTML only when several formulas
  must sit next to each other or next to a diagram (e.g. the three
  parametrizations across GIG / NIG / VG).
- **Distribution plots, traces, benchmark curves** → matplotlib PNG
  via `normix.utils.plotting` (golden-ratio figsize, `dpi=110`). Don't
  rebuild those in HTML canvas.
- **A layout, state comparison, or concept too dense for Mermaid** →
  one focused HTML file. Match the tokens in
  `references/html-template.html` (paper `#f5f4ed`, ink `#141413`,
  accent `#1b365d` — same as `docs/_static/normix.css`). Real labels
  and data; desktop and mobile. Then deliver it (next section).

Place each visual next to the short text it supports. Keep only the
calls, files, states, and boundaries needed for the current question.
Write that text and the labels STE-style (unslop `references/ste.md`).
The visual carries the structure; each sentence carries one fact.

## Delivering an HTML artifact

Write the file outside the git tree.
Never commit a conversation report.
If the page should last, promote it to `docs/` or `dev-notes/`.

### Codex

Do not use the Cursor branch below.
Do not assume Cursor menus, Mermaid, or `/opt/cursor/artifacts/`.
Write the HTML on a path this session can write.
Prefer `/tmp/show-me/<name>.html` when `/tmp` is writable.
If the sandbox blocks `/tmp`, pick another writable path outside the git tree.
Give that absolute path as an ordinary markdown link.

Text views stay in the reply.

Codex completion criterion: the reply links to HTML outside the git tree.
Or the reply contains a text view.

### Cursor

The table below is Cursor-only.
The menu steps below are Cursor-only.
The screenshot command below is Cursor-only.
Local HTML goes in `/tmp/show-me/`.
Cloud HTML goes in `/opt/cursor/artifacts/`.
Cloud PNGs go in `/opt/cursor/artifacts/screenshots/`.

Then pick the path that matches where the user will look:

| Where they are | What to do |
|---|---|
| Cursor desktop (local agent) | Write the HTML. Tell them to right-click the file → **Open in Browser** (built-in Chromium; Design Mode `Cmd+Shift+D` annotates). Do **not** paste a `file://` URL into the omnibar — it rewrites to `https://`. |
| Cloud agent → desktop or iOS | Render with headless Chrome and embed the PNGs in the reply (and in a PR body if one exists). Ship **two** widths: desktop `1280` and mobile `390` (`--force-device-scale-factor=2` on mobile). The iOS app has no file browser — screenshots and videos are the review surface. |
| Durable / any phone browser | Don't leave it as a one-off HTML file. Promote to `docs/` and publish (the `docs-publish` skill); GitHub Pages is the phone-browser channel. |

Headless Chrome on the cloud VM (gotchas from a real hang):

```bash
timeout 60 google-chrome --headless=new --disable-gpu --no-sandbox \
  --hide-scrollbars --timeout=20000 --window-size=1280,1800 \
  --screenshot=/opt/cursor/artifacts/screenshots/<name>-desktop.png \
  file://<absolute-path-to-html>
```

Use `--timeout=20000`, not `--virtual-time-budget` (the latter hung
the process). Mermaid must be the UMD build (`mermaid.min.js`), not
the ESM module — `file://` pages can't import ESM from a CDN.

Cursor completion criterion: the user can open the visual without a `file://` omnibar URL.
No HTML file sits untracked in the repo.

## Gotchas

- Cursor's **classic** markdown preview (`Cmd+Shift+V`) renders
  KaTeX; the newer inline **Preview \| Markdown** tab does not.
  Don't tell the user to "just preview the `.md`" for math.
- Cursor chat Mermaid can vanish when you switch threads.
  An HTML file, or its screenshot, is the stable copy.
  On Codex, do not rely on Mermaid.
- One file, one point. A slide deck of everything you know is the
  failure mode.
- CDN assets need network. That's fine on cloud VMs and desktops;
  don't inline KaTeX/Mermaid unless the user asks for an offline file.
