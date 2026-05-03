"""
Run all normix benchmarks and save results.

Usage:
    uv run python benchmarks/run_all.py
    uv run python benchmarks/run_all.py --large --mcecm
    uv run python benchmarks/run_all.py --only em
    uv run python benchmarks/run_all.py --only bessel,gig
"""

import argparse
import subprocess
import sys
import os

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARKS = {
    "bessel": "benchmarks/bench_bessel.py",
    "gig": "benchmarks/bench_gig_solvers.py",
    "jit": "benchmarks/bench_jit_solvers.py",
    "em": "benchmarks/bench_em_mixture.py",
}


def run_benchmark(name: str, script: str, extra_args: list[str]) -> int:
    """Run a single benchmark script as a subprocess."""
    cmd = [sys.executable, script, "--save"] + extra_args
    print(f"\n{'#' * 80}")
    print(f"# Running: {name}")
    print(f"# Command: {' '.join(cmd)}")
    print(f"{'#' * 80}\n", flush=True)
    result = subprocess.run(cmd, cwd=os.path.dirname(_BENCH_DIR))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all normix benchmarks")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated benchmark names: bessel,gig,em")
    parser.add_argument("--large", action="store_true",
                        help="Pass --large to EM benchmark")
    parser.add_argument("--mcecm", action="store_true",
                        help="Pass --mcecm to EM benchmark")
    parser.add_argument("--n-stocks", type=int, default=None,
                        help="Pass --n-stocks to EM benchmark")
    args = parser.parse_args()

    names = list(BENCHMARKS.keys())
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        for n in names:
            if n not in BENCHMARKS:
                print(f"Unknown benchmark: {n!r}. "
                      f"Available: {', '.join(BENCHMARKS)}")
                sys.exit(1)

    results = {}
    for name in names:
        extra = []
        if name == "em":
            if args.large:
                extra.append("--large")
            if args.mcecm:
                extra.append("--mcecm")
            if args.n_stocks is not None:
                extra.extend(["--n-stocks", str(args.n_stocks)])
        rc = run_benchmark(name, BENCHMARKS[name], extra)
        results[name] = rc
        if rc != 0:
            print(f"\n  [WARN] {name} exited with code {rc}", flush=True)

    print(f"\n{'=' * 80}")
    print("Benchmark Summary")
    print(f"{'=' * 80}")
    for name, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  {name:<12} {status}")
    print(f"{'=' * 80}")

    failed = [n for n, rc in results.items() if rc != 0]
    if failed:
        print(f"\n  {len(failed)} benchmark(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll benchmarks passed. Results saved to benchmarks/results/")


if __name__ == "__main__":
    main()
