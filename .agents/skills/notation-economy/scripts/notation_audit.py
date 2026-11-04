#!/usr/bin/env python3
"""Report candidate notation introductions and later math-token occurrences.

Usage: python3 notation_audit.py path/to/page.md
Accepts files or directories (recursively scans *.md). Findings always exit 0;
invalid inputs exit nonzero. No writes, dependencies, or build integration.

Heuristic only: recognizes simple assignments/names following Let/Write/Set/
Define/denote (also Put/Defining), and := or \\coloneqq in math. Counts after
the defining math span, excluding its RHS. Functions count across arguments;
subscripts and simple accents stay distinct. Counts include later restatements
and redefinitions, without understanding scope or meaning. Misses prose-only
uses, declarations by inequalities, macros, and complex/implicit definitions.
Braced accents may be split into spurious tokens.
Skips frontmatter, inline code and ordinary fenced code; supports dollar math
and fenced MyST math, and retains math in figure captions. Not a TeX parser.
"""

import argparse
import re
from pathlib import Path


SYMBOL = r"(?:\\(?:widehat|hat|bar)\s*)?(?:\\[A-Za-z]+|[A-Za-z])(?:_(?:\{[^{}]+\}|\\[A-Za-z]+|[A-Za-z0-9]))?(?:\^\*|\^\{\*\})?"
TOKEN = re.compile(SYMBOL)
HEAD = re.compile(rf"\s*(?P<symbol>{SYMBOL})(?:\([^()\n]*\))?\s*(?P<op>:=|\\coloneqq|=|$)")
EXPLICIT = re.compile(rf"(?P<symbol>{SYMBOL})(?:\([^()\n]*\))?\s*(?::=|\\coloneqq)")
CUE = re.compile(r"\b(?:let|write|set|define|defining|denote|put)\b", re.I)
MATH = re.compile(r"\$\$(.*?)\$\$|(?<![\\$])\$(?!\$)(.*?)(?<!\\)\$", re.S)


def blank(text):
    return re.sub(r"[^\n]", " ", text)


def mask_nonmath(text):
    """Preserve offsets and line numbers while hiding common non-prose blocks."""
    text = re.sub(r"\A---\n.*?\n---(?:\n|$)", lambda m: blank(m[0]), text, flags=re.S)
    lines = []
    fence = None
    mode = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"\s*(`{3,}|~{3,})(.*)", line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                lines.append("$$" + blank(line)[2:] if mode == "math" else blank(line))
                fence = mode = None
            elif mode == "skip" or (mode == "math" and line.lstrip().startswith(":")):
                lines.append(blank(line))
            else:
                lines.append(line)
        elif marker:
            fence = marker[1]
            info = marker[2].strip()
            mode = "math" if info.startswith("{math}") else "keep" if info.startswith("{") else "skip"
            lines.append("$$" + blank(line)[2:] if mode == "math" else blank(line))
        else:
            lines.append(line)
    return re.sub(r"(`+)[^\n]*?\1", lambda m: blank(m[0]), "".join(lines))


def normalize(symbol):
    symbol = re.sub(r"\s+", "", symbol)
    symbol = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", symbol).replace("^{*}", "^*")
    return re.sub(r"\\(widehat|hat|bar)(?=[A-Za-z])", r"\\\1 ", symbol)


def audit(text):
    text = mask_nonmath(text)
    spans = []
    tokens = []
    for match in MATH.finditer(text):
        group = 1 if match[1] is not None else 2
        start = match.start(group)
        spans.append((match, match[group], start))
        tokens.extend((normalize(t[0]), start + t.start()) for t in TOKEN.finditer(match[group]))
    seen = set()
    rows = []
    for match, formula, start in spans:
        candidates = list(EXPLICIT.finditer(formula))
        head = HEAD.match(formula)
        # Include the preceding paragraph for a standalone display.
        prefix = text[:match.start()].rstrip()
        paragraph = prefix[prefix.rfind("\n\n") + 2:] if "\n\n" in prefix else prefix
        if head and CUE.search(paragraph):
            if head["op"] or re.match(r"\s+(?:for|denote)\b", text[match.end():], re.I):
                candidates.append(head)
        for candidate in sorted(candidates, key=lambda c: c.start("symbol")):
            symbol = normalize(candidate["symbol"])
            if symbol in seen:
                continue
            seen.add(symbol)
            line = text.count("\n", 0, start + candidate.start("symbol")) + 1
            uses = sum(name == symbol and pos >= match.end() for name, pos in tokens)
            rows.append((line, symbol, uses))
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    files = set()
    for path in args.paths:
        if not path.exists():
            parser.error(f"path does not exist: {path}")
        files.update(path.rglob("*.md") if path.is_dir() else [path])
    print("Notation audit (heuristic; REVIEW means <=2 later math occurrences).")
    for path in sorted(files):
        try:
            rows = audit(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as error:
            parser.error(str(error))
        print(f"\n{path}")
        print("  line  symbol          later  review")
        for line, symbol, uses in rows:
            print(f"  {line:4}  {symbol:14}  {uses:5}  {'REVIEW' if uses <= 2 else '-'}")
        if not rows:
            print("  (no recognized introductions)")


if __name__ == "__main__":
    main()
