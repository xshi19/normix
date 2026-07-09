#!/usr/bin/env python3
"""Convert docs/theory/*.rst (except index) to MyST Markdown.

Preserves Sphinx ``:eq:`` / ``:math:`` / ``:meth:`` roles and citation
targets. Rewrites titles, ``.. math::`` (with labels), and admonitions to
MyST syntax. Citation definitions are wrapped in ``{eval-rst}`` so Sphinx
keeps resolving ``[Key]_`` references.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1] / "docs" / "theory"


def _is_underline(line: str) -> bool:
    s = line.rstrip()
    return bool(s) and len(set(s)) == 1 and s[0] in "=-~^\"'"


def _heading_level(ch: str) -> int:
    return {"=": 1, "-": 2, "~": 3, "^": 4, '"': 5, "'": 6}.get(ch, 2)


def _collect_indented(lines: list[str], start: int) -> tuple[list[str], int]:
    """Collect a blank-tolerant indented block starting at ``start``."""
    body: list[str] = []
    i = start
    n = len(lines)
    while i < n:
        L = lines[i]
        if not L.strip():
            if i + 1 < n and (lines[i + 1].startswith("   ") or lines[i + 1].startswith("\t")):
                body.append("")
                i += 1
                continue
            break
        if L.startswith("   "):
            body.append(L[3:])
            i += 1
            continue
        if L.startswith("\t"):
            body.append(L.lstrip("\t"))
            i += 1
            continue
        break
    return body, i


def convert(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Section title + underline
        if (
            i + 1 < n
            and line.strip()
            and not line.startswith(" ")
            and not line.startswith("..")
            and _is_underline(lines[i + 1])
        ):
            level = _heading_level(lines[i + 1].lstrip()[0])
            out.append("#" * level + " " + line.strip())
            i += 2
            continue

        # .. math::
        if re.match(r"^\.\. math::\s*$", line):
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            label = None
            if i < n:
                m = re.match(r"^\s+:label:\s+(\S+)\s*$", lines[i])
                if m:
                    label = m.group(1)
                    i += 1
                    while i < n and not lines[i].strip():
                        i += 1
            body, i = _collect_indented(lines, i)
            out.append("```{math}" + (f"\n:label: {label}\n" if label else ""))
            out.extend(body)
            out.append("```")
            continue

        # Admonitions
        m = re.match(r"^\.\. (note|warning|tip|important|caution)::\s*(.*)$", line)
        if m:
            kind, rest = m.group(1), m.group(2).strip()
            i += 1
            body, i = _collect_indented(lines, i)
            if rest:
                body = [rest, *body]
            out.append(f"```{{{kind}}}")
            out.extend(body)
            out.append("```")
            continue

        # Citation definitions .. [Key] ...
        if re.match(r"^\.\. \[", line):
            block = [line]
            i += 1
            while i < n and (
                lines[i].startswith("   ")
                or (
                    not lines[i].strip()
                    and i + 1 < n
                    and lines[i + 1].startswith("   ")
                )
            ):
                block.append(lines[i])
                i += 1
            out.append("```{eval-rst}")
            out.extend(block)
            out.append("```")
            continue

        out.append(line)
        i += 1

    text_out = "\n".join(out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip() + "\n"
    # RST roles -> MyST roles
    text_out = re.sub(r":eq:`([^`]+)`", r"{eq}`\1`", text_out)
    text_out = re.sub(r":math:`([^`]+)`", r"$\1$", text_out)
    text_out = re.sub(
        r":(meth|class|func|mod|obj|data|attr|exc|ref|doc):`([^`]+)`",
        r"{\1}`\2`",
        text_out,
    )
    # Inline [Key]_ -> Markdown link to Sphinx citation anchor
    text_out = re.sub(
        r"\[([A-Za-z][A-Za-z0-9]+)\]_",
        lambda m: f"[{m.group(1)}](#{m.group(1).lower()})",
        text_out,
    )
    return text_out


def main() -> int:
    dry = "--dry-run" in sys.argv
    files = sorted(p for p in THEORY.glob("*.rst") if p.name != "index.rst")
    if not files:
        print("No theory .rst content files found", file=sys.stderr)
        return 1
    for src in files:
        md = convert(src.read_text(encoding="utf-8"))
        dst = src.with_suffix(".md")
        if dry:
            print(f"would write {dst} ({len(md)} bytes)")
            continue
        dst.write_text(md, encoding="utf-8")
        src.unlink()
        print(f"{src.name} -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
