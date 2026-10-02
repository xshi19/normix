#!/usr/bin/env python3
"""Advisory STE-style checker for Markdown prose (stdlib only).

Flags sentence length by type, paragraph length, common wordy words,
progressive/perfect verb forms, and normix term drift. Code, math,
tables, headings, and front matter are skipped. This is not an
ASD-STE100 compliance check: there is no STE dictionary here.

Usage:
    python .cursor/skills/unslop/scripts/ste_check.py PATH... [--summary] [--strict]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_PROCEDURE, MAX_DESCRIPTION, MAX_PARAGRAPH = 20, 25, 6

WORDS = {
    r"utili[sz]e[sd]?|utili[sz]ing": "use",
    r"in order to": "to",
    r"prior to": "before",
    r"commenc\w*": "start",
    r"approximately": "about",
    r"ensur(?:e|es|ed|ing)": "make sure",
    r"facilitat\w*": "help",
    r"numerous": "many",
    r"subsequently": "then / after",
    r"in the event that": "if",
    r"due to the fact that": "because",
    r"perform(?:s|ed|ing)?": "do / run",
}
TERMS = {
    r"canonical param\w*": "natural parameters",
    r"mean param\w*": "expectation parameters",
    r"log partition": "log-partition",
}
IMPERATIVE = set(
    "add apply ask avoid call check choose cite commit copy create cut define "
    "delete do don't draft edit enable fix follow give import install keep "
    "label link list make mark match move name never open pass pick place "
    "prefer push put quote reach read record remove render replace report "
    "review rewrite run say see set ship show skip split start state stop "
    "tell update use verify wait write".split()
)
NOT_PROGRESSIVE = re.compile(
    r"\w*thing|during|missing|existing|following|remaining|underlying|"
    r"corresponding|interesting|confusing|appealing|misleading|string|ring|"
    r"king|bring|willing|evolving"
)
PROGRESSIVE = re.compile(r"\b(?:am|is|are|was|were|be|been)\s+(\w+ing)\b", re.I)
PERFECT = re.compile(r"\b(?:has|have|had)\s+(?:been|\w+ed|\w+en)\b", re.I)
FENCE = re.compile(r"^\s*(```|~~~)")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
LEAD_IN = re.compile(r"^\*\*[^*]+\*\*(?:[.:]|\s+[—-])\s*")
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])")
QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”")
ABBREV = re.compile(r"\b(e\.g|i\.e|vs|etc|cf|Eq|Fig|Sec)\.", re.I)


def clean(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$|\$[^$]+\$", "MATH", text)
    text = re.sub(r"\{\w+\}`[^`]*`", "REF", text)
    text = re.sub(r"`[^`]+`", "CODE", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return ABBREV.sub(r"\1", text)


def blocks(lines: list[str]):
    """Yield (line_no, is_list_item, text) prose blocks."""
    in_fence = in_math = False
    start, buf, is_item = 0, [], False
    if lines and lines[0].strip() == "---":
        end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "---"), 0)
        lines = [""] * (end + 1) + lines[end + 1 :]
    for no, raw in enumerate(lines, 1):
        line = raw.rstrip()
        if FENCE.match(line):
            in_fence = not in_fence
        elif line.strip() == "$$":
            in_math = not in_math
        skip = (in_fence or in_math or FENCE.match(line) or line.strip() == "$$"
                or re.match(r"^\s*(#|\||<|:::|\{|\.\. )", line))
        stripped = re.sub(r"^\s*>\s?", "", line).strip()
        new_item = bool(LIST_ITEM.match(line))
        if skip or not stripped or new_item:
            if buf:
                yield start, is_item, " ".join(buf)
            buf = []
            if skip or not stripped:
                continue
        if not buf:
            start, is_item = no, new_item
        buf.append(LIST_ITEM.sub("", stripped))


def check(path: Path):
    findings, n_sent = [], 0
    for line_no, is_item, text in blocks(path.read_text(encoding="utf-8").splitlines()):
        sentences = [s for s in SPLIT.split(clean(text)) if len(s.split()) > 2]
        n_sent += len(sentences)
        if not is_item and len(sentences) > MAX_PARAGRAPH:
            findings.append((line_no, "STE-PARA", f"{len(sentences)} sentences"))
        for s in sentences:
            body = LEAD_IN.sub("", s.strip())
            words = body.split()
            first = words[0].lower().strip("*_\"'") if words else ""
            procedure = first in IMPERATIVE
            limit = MAX_PROCEDURE if procedure else MAX_DESCRIPTION
            if len(words) > limit:
                kind = "STE-LEN-P" if procedure else "STE-LEN-D"
                findings.append((line_no, kind, f"{len(words)} words: {body[:60]}..."))
            unquoted = QUOTED.sub("QUOTE", body)
            for table, rule in ((WORDS, "STE-WORD"), (TERMS, "STE-TERM")):
                for pat, alt in table.items():
                    for m in re.finditer(rf"\b(?:{pat})\b", unquoted, re.I):
                        findings.append((line_no, rule, f'"{m.group(0)}" -> {alt}'))
            for m in PROGRESSIVE.finditer(unquoted):
                if not NOT_PROGRESSIVE.fullmatch(m.group(1).lower()):
                    findings.append((line_no, "STE-VERB", f'progressive "{m.group(0)}"'))
            for m in PERFECT.finditer(unquoted):
                findings.append((line_no, "STE-VERB", f'perfect "{m.group(0)}"'))
    return n_sent, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--summary", action="store_true", help="one line per file")
    parser.add_argument("--strict", action="store_true", help="exit 1 on findings")
    args = parser.parse_args()
    files = sorted(
        f for p in args.paths
        for f in ([p] if p.is_file() else [*p.rglob("*.md"), *p.rglob("*.mdc")])
        if "_build" not in f.parts
    )
    total = 0
    for f in files:
        n_sent, findings = check(f)
        total += len(findings)
        if args.summary:
            counts = {r: sum(1 for _, k, _ in findings if k == r) for r in
                      ("STE-LEN-P", "STE-LEN-D", "STE-PARA", "STE-WORD", "STE-VERB", "STE-TERM")}
            print(f"{f}\t{n_sent}\t" + "\t".join(f"{k}={v}" for k, v in counts.items()))
        else:
            for line_no, rule, msg in findings:
                print(f"{f}:{line_no}: {rule} {msg}")
    return 1 if args.strict and total else 0


if __name__ == "__main__":
    sys.exit(main())
