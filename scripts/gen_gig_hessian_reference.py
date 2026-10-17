r"""Generate GIG log-partition Hessians with 40-digit mpmath arithmetic.

Writes ``tests/data/gig_hessian_reference.json``. Run with mpmath available:

    uv run --with mpmath python scripts/gen_gig_hessian_reference.py

mpmath is needed only for generation. Hessians use the natural coordinates
:math:`\theta = (p - 1, -b/2, -a/2)` for :math:`a, b > 0` and are stored
as float64 values, matching the Bessel reference table convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "gig_hessian_reference.json"
POINTS = [
    (0.5, 1.0, 1.0),
    (-1.5, 2.0, 0.7),
    (2.5, 0.5, 3.0),
    (0.5, 1e-4, 1e4),
]


def _log_partition(t0: mp.mpf, t1: mp.mpf, t2: mp.mpf) -> mp.mpf:
    p, b, a = t0 + 1, -2 * t1, -2 * t2
    return mp.log(2 * mp.besselk(p, mp.sqrt(a * b))) + p / 2 * mp.log(b / a)


def _reference(p: float, a: float, b: float) -> dict:
    theta = tuple(mp.mpf(t) for t in (p - 1, -b / 2, -a / 2))
    hessian = [
        [
            float(mp.diff(_log_partition, theta, tuple(int(k == i) + int(k == j) for k in range(3))))
            for j in range(3)
        ]
        for i in range(3)
    ]
    return {"p": p, "a": a, "b": b, "hessian": hessian}


def main() -> None:
    with mp.workdps(40):
        points = [_reference(p, a, b) for p, a, b in POINTS]
    payload = {
        "dps": 40,
        "generated_by": "scripts/gen_gig_hessian_reference.py",
        "n_points": len(points),
        "points": points,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
