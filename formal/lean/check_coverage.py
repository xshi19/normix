"""Check reviewed function-source hashes in the Lean correspondence map."""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HASH_SCOPE = "function-source-sha256"


def target_hash(target: str, root: Path) -> str:
    """Hash a function's dedented source, including decorators and docstrings."""
    parts = target.split(".")
    definitions = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for size in range(len(parts) - 1, 0, -1):
        module = root.joinpath(*parts[:size]).with_suffix(".py")
        if not module.is_file():
            continue
        source = module.read_text(encoding="utf-8")
        node = ast.parse(source)
        for name in parts[size:]:
            node = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, definitions) and child.name == name
                ),
                None,
            )
            if node is None:
                raise ValueError(f"missing symbol: {target}")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError(f"expected function or method: {target}")
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        lines = source.splitlines(keepends=True)[start - 1 : node.end_lineno]
        normalized = textwrap.dedent("".join(lines)).rstrip() + "\n"
        return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    raise ValueError(f"missing module: {target}")


def _update_hashes(source: str, updates: dict[str, dict[str, str]]) -> str:
    blocks = re.split(r"(?m)(?=^\[\[claim\]\])", source)
    for index, block in enumerate(blocks):
        if not block.startswith("[[claim]]"):
            continue
        claim_id = tomllib.loads(block)["claim"][0]["id"]
        for target, digest in updates.get(claim_id, {}).items():
            pattern = r'("' + re.escape(target) + r'"\s*=\s*")[^"]*(")'
            block, count = re.subn(pattern, r"\g<1>" + digest + r"\g<2>", block)
            if count != 1:
                raise ValueError(
                    f"expected one quoted hash entry: {claim_id} -> {target}"
                )
        blocks[index] = block
    return "".join(blocks)


def main(argv: list[str] | None = None) -> int:
    """Check all claims, or update selected baselines and then check all claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage", type=Path, default=ROOT / "formal/lean/coverage.toml"
    )
    parser.add_argument(
        "--update", action="store_true", help="record reviewed source hashes"
    )
    parser.add_argument(
        "--claim", action="append", help="claim to update; repeat for several"
    )
    args = parser.parse_args(argv)
    if args.claim and not args.update:
        parser.error("--claim requires --update; checks always cover every claim")

    try:
        source = args.coverage.read_text(encoding="utf-8")
        coverage = tomllib.loads(source)
        if coverage["schema_version"] != 2 or coverage["hash_scope"] != HASH_SCOPE:
            raise ValueError(f"expected schema_version 2 and hash_scope {HASH_SCOPE}")
        claims = coverage["claim"]
        ids = [claim["id"] for claim in claims]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate claim IDs")
        for claim in claims:
            if claim["status"] not in {"proved", "tested", "blocked"}:
                raise ValueError(f"unknown status: {claim['id']}")
            targets = claim["targets"]
            if not targets or len(set(targets)) != len(targets):
                raise ValueError(f"empty or duplicate targets: {claim['id']}")
            if set(targets) != set(claim["target_hashes"]):
                raise ValueError(f"targets and target_hashes differ: {claim['id']}")

        if args.update:
            selected = (
                set(args.claim)
                if args.claim
                else {claim["id"] for claim in claims if claim["status"] != "blocked"}
            )
            if selected - set(ids):
                raise ValueError(f"unknown claims: {sorted(selected - set(ids))}")
            updates = {
                claim["id"]: {
                    target: target_hash(target, ROOT) for target in claim["targets"]
                }
                for claim in claims
                if claim["id"] in selected
            }
            updated = _update_hashes(source, updates)
            if updated != source:
                args.coverage.write_text(updated, encoding="utf-8")
            for claim in claims:
                if claim["id"] in updates:
                    claim["target_hashes"] = updates[claim["id"]]
                    print(f"updated: {claim['id']}")

        failed = False
        checked = skipped = 0
        for claim in claims:
            claim_id = claim["id"]
            fallback = claim.get("fallback_hypothesis_test", "")
            if claim["status"] != "blocked" and (
                not fallback or fallback.startswith(("PLANNED", "PLACEHOLDER"))
            ):
                print(f"needs-review: {claim_id}: placeholder fallback_hypothesis_test")
                failed = True
            for target, recorded in claim["target_hashes"].items():
                label = f"{claim_id} -> {target}"
                if claim["status"] == "blocked" and recorded.startswith("PLACEHOLDER"):
                    print(f"skipped: {label}: blocked placeholder")
                    skipped += 1
                    continue
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", recorded):
                    print(f"needs-review: {label}: placeholder or invalid hash")
                    failed = True
                    continue
                try:
                    current = target_hash(target, ROOT)
                except (OSError, TypeError, ValueError, SyntaxError) as error:
                    print(f"needs-review: {label}: {error}")
                    failed = True
                    continue
                if current != recorded:
                    print(
                        f"needs-review: {label}: hash mismatch ({recorded} != {current})"
                    )
                    failed = True
                else:
                    print(f"ok: {label}")
                    checked += 1
        print(f"{checked} targets matched; {skipped} blocked placeholders skipped")
        return int(failed)
    except (OSError, KeyError, TypeError, ValueError, SyntaxError) as error:
        print(f"coverage error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
