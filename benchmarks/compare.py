"""
Compare two benchmark result files side by side.

Usage:
    uv run python benchmarks/compare.py benchmarks/results/OLD.json benchmarks/results/NEW.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.utils import fmt_time, hdr, sep


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def delta_str(old: float, new: float) -> str:
    """Format a percentage change with sign."""
    if old == 0:
        return "N/A"
    pct = 100 * (new - old) / old
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def compare_em(old: dict, new: dict):
    """Compare EM benchmark results."""
    W = 130
    hdr("EM Benchmark Comparison", W)

    old_rows = {
        (r["dist_name"], r["algorithm"], r["e_backend"], r["m_backend"], r["m_method"]): r
        for r in old.get("results", [])
    }
    new_rows = {
        (r["dist_name"], r["algorithm"], r["e_backend"], r["m_backend"], r["m_method"]): r
        for r in new.get("results", [])
    }

    all_keys = sorted(set(old_rows) | set(new_rows))
    if not all_keys:
        print("  No EM results to compare.")
        return

    print(
        f"  {'Dist':<6} {'Algo':<6} {'E':<5} {'M':<5} {'Meth':<7} "
        f"{'Old/iter':>10} {'New/iter':>10} {'Δ':>8}  "
        f"{'Old iters':>9} {'New iters':>9}  "
        f"{'Old LL':>12} {'New LL':>12}")
    sep(W)

    prev_dist = None
    for key in all_keys:
        dist = key[0]
        if prev_dist is not None and dist != prev_dist:
            sep(W)
        prev_dist = dist

        o = old_rows.get(key)
        n = new_rows.get(key)

        if o is None or n is None or o.get("error") or n.get("error"):
            status = "MISSING" if (o is None or n is None) else "ERROR"
            print(f"  {key[0]:<6} {key[1]:<6} {key[2]:<5} {key[3]:<5} {key[4]:<7} "
                  f"  {status}")
            continue

        o_per = o["total_time"] / max(o["n_iter"], 1)
        n_per = n["total_time"] / max(n["n_iter"], 1)

        print(
            f"  {key[0]:<6} {key[1]:<6} {key[2]:<5} {key[3]:<5} {key[4]:<7} "
            f"{fmt_time(o_per):>10} {fmt_time(n_per):>10} "
            f"{delta_str(o_per, n_per):>8}  "
            f"{o['n_iter']:>9} {n['n_iter']:>9}  "
            f"{o['final_ll']:>12.4f} {n['final_ll']:>12.4f}")

    print(f"{'=' * W}")


def compare_bessel(old: dict, new: dict):
    """Compare Bessel benchmark results."""
    W = 100
    hdr("Bessel Comparison", W)

    for label in ("scalar", "batch"):
        o_data = old.get(label)
        n_data = new.get(label)
        if o_data is None or n_data is None:
            continue

        if label == "scalar":
            print(f"  {'Case':<40} {'Old JAX μs':>12} {'New JAX μs':>12} {'Δ':>8}")
            sep(W)
            for o_row, n_row in zip(o_data, n_data):
                print(
                    f"  {o_row['label']:<40} "
                    f"{o_row['jax_us']:>12.1f} {n_row['jax_us']:>12.1f} "
                    f"{delta_str(o_row['jax_us'], n_row['jax_us']):>8}")
        else:
            sep(W)
            print(
                f"  batch N={o_data['n']}: "
                f"JAX {o_data['jax_ms']:.1f}→{n_data['jax_ms']:.1f} ms "
                f"({delta_str(o_data['jax_ms'], n_data['jax_ms'])})")

    print(f"{'=' * W}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result files")
    parser.add_argument("old", help="Path to older result JSON")
    parser.add_argument("new", help="Path to newer result JSON")
    args = parser.parse_args()

    old = load(args.old)
    new = load(args.new)

    old_sys = old.get("system", {})
    new_sys = new.get("system", {})
    print(f"\nComparing benchmarks:")
    print(f"  OLD: {os.path.basename(args.old)}  "
          f"(git={old_sys.get('git_hash', '?')}, {old_sys.get('timestamp', '?')})")
    print(f"  NEW: {os.path.basename(args.new)}  "
          f"(git={new_sys.get('git_hash', '?')}, {new_sys.get('timestamp', '?')})")

    bench_type = old.get("benchmark", new.get("benchmark"))
    if bench_type == "em_mixture":
        compare_em(old, new)
    elif bench_type == "bessel":
        compare_bessel(old, new)
    elif bench_type == "gig_solvers":
        print("\n  GIG solver comparison not yet implemented "
              "(inspect JSON files directly).")
    else:
        print(f"\n  Unknown benchmark type: {bench_type}")
        print("  Dumping system info diff only.")


if __name__ == "__main__":
    main()
