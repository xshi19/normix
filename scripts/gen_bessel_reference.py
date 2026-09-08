"""Generate the committed mpmath reference table for log K_ν derivatives.

Writes ``tests/data/bessel_reference.json``. mpmath is an optional generator
dependency, not a runtime or pytest requirement:

    uv run --with mpmath python scripts/gen_bessel_reference.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "bessel_reference.json"

NU_GRID = [0.0, 0.3, -0.3, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5,
           10.0, -10.0, 25.0, -25.0, 100.0, -100.0, 300.0, -300.0]
Z_GRID = [1e-10, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1e4, 1e6, 1e8]
EXTRA = [
    (0.0, 1.0), (0.5, 1.0), (25.0, 0.1), (25.0, 1.0), (25.0, 1e-6),
    (1.0, 1e4), (5.0, 1e6), (0.0, 1e-4), (1.0, 1e-30), (1.0, 1e-50),
]


def _logk(nu, z):
    return mp.log(mp.besselk(nu, z))


def _spot(v: float, z: float) -> dict:
    nu, zz = mp.mpf(v), mp.mpf(z)
    L = _logk(nu, zz)
    dv = mp.diff(lambda t: _logk(t, zz), nu)
    dz = mp.diff(lambda t: _logk(nu, t), zz)
    dvv = mp.diff(lambda t: _logk(t, zz), nu, 2)
    dzz = mp.diff(lambda t: _logk(nu, t), zz, 2)
    dvz = mp.diff(lambda t: mp.diff(lambda s: _logk(t, s), zz), nu)
    return {
        "v": float(v),
        "z": float(z),
        "log_k": float(L),
        "d_v": float(dv),
        "d_z": float(dz),
        "d_vv": float(dvv),
        "d_vz": float(dvz),
        "d_zz": float(dzz),
    }


def _points() -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for v in NU_GRID:
        for z in Z_GRID:
            key = (float(v), float(z))
            if key not in seen:
                seen.add(key)
                out.append(key)
    for v, z in EXTRA:
        key = (float(v), float(z))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    mp.mp.dps = args.dps
    pts = _points()
    rows = []
    t0 = time.perf_counter()
    for i, (v, z) in enumerate(pts, 1):
        t1 = time.perf_counter()
        row = _spot(v, z)
        dt = time.perf_counter() - t1
        rows.append(row)
        print(f"[{i:3d}/{len(pts)}] v={v:g} z={z:g}  {dt:.2f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dps": args.dps,
        "generated_by": "scripts/gen_bessel_reference.py",
        "n_points": len(rows),
        "elapsed_s": time.perf_counter() - t0,
        "points": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} ({len(rows)} points)")


if __name__ == "__main__":
    main()
