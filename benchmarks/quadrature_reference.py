"""Independent adaptive-integral references for conditional mixture CVaR.

This module uses only NumPy and SciPy. Run it to regenerate the frozen
finance table: ``python benchmarks/quadrature_reference.py --output PATH``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr


CASES = {
    "VG": (2.0, 2.0, 0.0),
    "NIG": (-0.5, 0.5, 0.5),
    "GH": (1.0, 1.0, 0.5),
    "NInvG3": (-3.0, 0.0, 2.0),
    "NInvG2": (-2.0, 0.0, 1.0),
    "NInvG1.5": (-1.5, 0.0, 0.5),
    "VG0.1": (0.1, 1.0, 0.0),
    "GH_boundary": (-1.5, 5e-9, 0.5),
    "GH_concentrated": (1.0, 5000.0, 5000.0),
}


class Reference:
    """Adaptive integration of a normalized log-coordinate mixing density."""

    def __init__(self, nu: float, A: float, B: float) -> None:
        self.nu, self.A, self.B = nu, A, B
        kappa = math.hypot(nu, 2 * math.sqrt(A * B))
        ac = (kappa + nu) / 2 if nu >= 0 else 2 * A * B / (kappa - nu)
        bc = (kappa - nu) / 2 if nu <= 0 else 2 * A * B / (kappa + nu)
        self.ac, self.bc = ac, bc
        self.s0 = math.log(ac / A) if nu >= 0 else math.log(B / bc)
        self.tail_index = math.inf if A > 0 else -nu
        tilt = 1.5 if self.tail_index > 1.5 else (1.0 if self.tail_index > 1 else 0.0)
        self.lo = max(self._edge(-1, 0.0), -700.0 - self.s0)
        self.hi = self._edge(1, tilt)
        self.mass = self._integrate(lambda y: 1.0, [0.0])

    def _log_density(self, x: float) -> float:
        return (self.nu * x - (self.ac * math.expm1(x) if self.ac else 0.0)
                - (self.bc * math.expm1(-x) if self.bc else 0.0))

    def _edge(self, side: int, tilt: float) -> float:
        f = lambda r: self._log_density(side * r) + tilt * side * r + 80.0
        hi = 1.0
        while f(hi) > 0:
            hi *= 2
        return side * brentq(f, 0.0, hi)

    def _integrate(self, fn, points: list[float]) -> float:
        cuts = sorted({self.lo, self.hi, *[p for p in points if self.lo < p < self.hi]})
        return sum(quad(
            lambda x: fn(math.exp(self.s0 + x)) * math.exp(self._log_density(x)),
            lo, hi, epsabs=1e-14, epsrel=3e-13, limit=500,
        )[0] for lo, hi in zip(cuts[:-1], cuts[1:]))

    def mean(self) -> float:
        """Return the first mixing moment."""
        return self._integrate(lambda y: y, [0.0]) / self.mass

    def evaluate(self, mu: float, gamma: float, sigma: float, alpha: float) -> dict:
        """Return VaR, CVaR, gradient, and Hessian from adaptive integrals."""
        def points(x):
            yc = (x - mu) / gamma if gamma else -1
            if yc <= 0:
                return [0.0]
            sc = math.log(yc) - self.s0
            width = sigma / (abs(gamma) * math.sqrt(yc))
            return [0.0, *[sc + k * width for k in (-10, -1, 0, 1, 10)]]

        def E(fn, x):
            return self._integrate(fn, points(x)) / self.mass

        def z(y, x):
            return (x - mu - gamma * y) / (sigma * math.sqrt(y))

        phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        F = lambda x: E(lambda y: ndtr(z(y, x)), x) - alpha
        width = sigma + abs(gamma) + 1.0
        lo, hi = mu - width, mu + width
        while F(lo) > 0:
            width *= 2
            lo = mu - width
        while F(hi) < 0:
            width *= 2
            hi = mu + width
        x = brentq(F, lo, hi, xtol=5e-15, rtol=1e-15)
        value = -x + E(lambda y: (x - mu - gamma * y) * ndtr(z(y, x))
                        + sigma * math.sqrt(y) * phi(z(y, x)), x) / alpha
        dg = -E(lambda y: y * ndtr(z(y, x)), x) / alpha
        ds = E(lambda y: math.sqrt(y) * phi(z(y, x)), x) / alpha
        num = ds * alpha
        den = E(lambda y: phi(z(y, x)) / math.sqrt(y), x)
        hgg = E(lambda y: math.sqrt(y) * phi(z(y, x)) * (y - num / den), x) / (alpha * sigma)
        hgs = -gamma / sigma * hgg
        hss = -gamma / sigma * hgs
        return {"var": -x, "cvar": value, "mean_y": self.mean(),
                "gradient": [-1.0, dg, ds],
                "hessian": [[0.0, 0.0, 0.0], [0.0, hgg, hgs], [0.0, hgs, hss]]}


def reference_table() -> list[dict]:
    """Generate the smooth-regime regression table, including boundary laws."""
    rows = []
    for name, params in CASES.items():
        for gamma, sigma, alpha in [(-0.5, 1.0, 0.05), (0.5, 1.0, 0.01)]:
            result = Reference(*params).evaluate(0.0, gamma, sigma, alpha)
            rows.append({"case": name, "params": params, "mu": 0.0,
                         "gamma": gamma, "sigma": sigma, "alpha": alpha, **result})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference_table(), indent=2) + "\n")
