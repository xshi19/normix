"""
Bessel function benchmark: log_kv — JAX vs CPU, scalar vs batch.

Usage:
    uv run python benchmarks/bench_bessel.py
    uv run python benchmarks/bench_bessel.py --save
"""

import os
import sys
import argparse

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.utils import timeit, jax_timeit, save_result, hdr, sep

from normix.utils.bessel import log_kv


SCALAR_CASES = [
    ("small-z  (v=0.5, z=0.001)",   0.5,  0.001),
    ("quadrature (v=1.0, z=2.0)",   1.0,  2.0),
    ("large-z  (v=1.0, z=30.0)",    1.0,  30.0),
    ("large-v  (v=50.0, z=5.0)",    50.0, 5.0),
    ("extreme  (v=100.0, z=200.0)", 100.0, 200.0),
]


def bench_scalar() -> list[dict]:
    """Benchmark scalar log_kv calls."""
    results = []
    for label, v, z in SCALAR_CASES:
        v_j, z_j = jnp.array(v), jnp.array(z)
        t_jax = jax_timeit(lambda: log_kv(v_j, z_j), n_runs=50, warmup=5)
        t_cpu = timeit(
            lambda: log_kv(float(v), float(z), backend='cpu'),
            n_runs=50, warmup=5)
        ratio = t_jax / t_cpu if t_cpu > 0 else float('inf')
        results.append({
            "label": label, "v": v, "z": z,
            "jax_us": t_jax * 1e6, "cpu_us": t_cpu * 1e6, "ratio": ratio,
        })
    return results


def bench_batch(n_batch: int = 2552) -> dict:
    """Benchmark batch log_kv (vmap JAX vs vectorized CPU)."""
    v_batch = jnp.full(n_batch, 0.5, dtype=jnp.float64)
    z_batch = jnp.linspace(0.5, 5.0, n_batch, dtype=jnp.float64)
    v_np = np.full(n_batch, 0.5)
    z_np = np.linspace(0.5, 5.0, n_batch)

    log_kv_jit = jax.jit(lambda v, z: jax.vmap(log_kv)(v, z))
    _ = log_kv_jit(v_batch, z_batch).block_until_ready()

    t_jax = jax_timeit(lambda: log_kv_jit(v_batch, z_batch), n_runs=100, warmup=10)
    t_cpu = timeit(lambda: log_kv(v_np, z_np, backend='cpu'), n_runs=100, warmup=10)
    ratio = t_jax / t_cpu if t_cpu > 0 else float('inf')
    return {
        "n": n_batch,
        "jax_ms": t_jax * 1e3, "cpu_ms": t_cpu * 1e3, "ratio": ratio,
    }


def print_results(scalar_results, batch_result):
    W = 90
    hdr("Bessel: log_kv(v, z)", W)

    def row(label, *vals):
        print(f"  {label:<40}" + "".join(f"{v:>12}" for v in vals), flush=True)

    row("", "JAX μs", "CPU μs", "ratio")
    sep(W)
    for r in scalar_results:
        row(r["label"],
            f"{r['jax_us']:.1f}", f"{r['cpu_us']:.1f}", f"{r['ratio']:.1f}×")

    sep(W)
    b = batch_result
    row(f"batch N={b['n']} (ms)",
        f"{b['jax_ms']:.1f}", f"{b['cpu_ms']:.1f}", f"{b['ratio']:.1f}×")
    print(f"{'=' * W}")


def bench_accuracy() -> list[dict]:
    """Sweep (n, D, T) against tests/data/bessel_reference.json (CPU kernel)."""
    import json
    from pathlib import Path
    from normix.utils.bessel import _moments_xp

    ref_path = Path(__file__).resolve().parent.parent / "tests" / "data" / "bessel_reference.json"
    table = json.loads(ref_path.read_text())
    rows = [r for r in table["points"] if r["z"] >= 1e-10]
    keys = [
        ("log_k", "log_k"), ("d_order", "d_v"), ("d_arg", "d_z"),
        ("d2_order", "d_vv"), ("d2_order_arg", "d_vz"), ("d2_arg", "d_zz"),
    ]

    def err(got, ref, v, key):
        if abs(v) == 0.0 and key in ("d_v", "d_vz"):
            return abs(got - ref)
        return abs(got - ref) / max(1.0, abs(ref))

    results = []
    print("\nAccuracy sweep vs mpmath table (z ≥ 1e-10)", flush=True)
    print(f"  {'n':>4} {'D':>4} {'T':>3} {'worst':>10}  at", flush=True)
    for n in (64, 96, 128, 192, 256):
        for D in (40, 50, 60):
            for T in (1, 2, 3):
                worst = 0.0
                where = None
                for row in rows:
                    m = _moments_xp(
                        row["v"], row["z"], np,
                        n_nodes=n, log_drop=float(D), tilt=T,
                    )
                    for attr, key in keys:
                        got = float(np.asarray(
                            m.log_k if attr == "log_k" else getattr(m, attr)
                        ))
                        e = err(got, row[key], row["v"], key)
                        if e > worst:
                            worst = e
                            where = f"{key}({row['v']:g},{row['z']:g})"
                rec = {"n": n, "D": D, "T": T, "worst": worst, "where": where}
                results.append(rec)
                print(f"  {n:4d} {D:4d} {T:3d} {worst:10.3e}  {where}", flush=True)
    best = min(results, key=lambda r: r["worst"])
    print(f"best: n={best['n']} D={best['D']} T={best['T']}  {best['worst']:.3e}",
          flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Bessel log_kv benchmark")
    parser.add_argument("--save", action="store_true",
                        help="Save results to benchmarks/results/")
    parser.add_argument("--accuracy", action="store_true",
                        help="Sweep quadrature constants against the mpmath table")
    args = parser.parse_args()

    print(f"\nnormix Bessel Benchmark", flush=True)
    print(f"Python {sys.version.split()[0]}, JAX {jax.__version__}", flush=True)
    print(f"Devices: {jax.devices()}", flush=True)

    if args.accuracy:
        acc = bench_accuracy()
        if args.save:
            path = save_result("bessel_accuracy", {"benchmark": "bessel_accuracy",
                                                  "rows": acc})
            print(f"\nResults saved to {path}")
        return

    scalar_results = bench_scalar()
    batch_result = bench_batch()
    print_results(scalar_results, batch_result)

    if args.save:
        data = {
            "benchmark": "bessel",
            "scalar": scalar_results,
            "batch": batch_result,
        }
        path = save_result("bessel", data)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
