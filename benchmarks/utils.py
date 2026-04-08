"""Shared utilities for normix benchmarks."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timeit(fn, n_runs: int = 100, warmup: int = 5) -> float:
    """Return median elapsed time (seconds) over *n_runs* calls."""
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def jax_timeit(fn, n_runs: int = 100, warmup: int = 5) -> float:
    """Timeit for JAX functions — blocks until device transfer complete."""
    def blocked():
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for v in result:
                if hasattr(v, "block_until_ready"):
                    v.block_until_ready()
        elif isinstance(result, dict):
            for v in result.values():
                if hasattr(v, "block_until_ready"):
                    v.block_until_ready()
    return timeit(blocked, n_runs=n_runs, warmup=warmup)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sp500_returns.csv")


def load_sp500_data(n_stocks: int | None = None) -> np.ndarray:
    """Load SP500 returns.  *n_stocks=None* loads all columns."""
    import pandas as pd
    if not os.path.exists(_DATA_PATH):
        raise FileNotFoundError(
            f"SP500 data not found at {_DATA_PATH}\n"
            "  Run: uv run python scripts/download_sp500_data.py")
    df = pd.read_csv(_DATA_PATH, index_col=0, parse_dates=True).dropna(axis=1)
    X = df.values.astype(np.float64)
    if n_stocks is not None:
        X = X[:, :n_stocks]
    return X


# ---------------------------------------------------------------------------
# System / git info
# ---------------------------------------------------------------------------

def system_info() -> dict:
    """Collect runtime metadata for result persistence."""
    import jax
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
    }
    try:
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            text=True,
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            text=True,
        ).strip()
    except Exception:
        info["git_hash"] = "unknown"
        info["git_branch"] = "unknown"
    return info


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def save_result(bench_name: str, data: dict) -> str:
    """Save benchmark result as JSON.  Returns the written path."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    info = system_info()
    date = datetime.now().strftime("%Y-%m-%d")
    git_hash = info.get("git_hash", "unknown")
    filename = f"{date}_{git_hash}_{bench_name}.json"
    path = os.path.join(_RESULTS_DIR, filename)
    payload = {"system": info, **data}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_result(path: str) -> dict:
    """Load a benchmark result JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_time(t: float) -> str:
    """Human-readable time string."""
    if t >= 1.0:
        return f"{t:.2f}s"
    if t >= 0.01:
        return f"{t * 1e3:.0f}ms"
    if t >= 0.001:
        return f"{t * 1e3:.1f}ms"
    return f"{t * 1e6:.0f}μs"


def hdr(title: str, width: int = 110) -> None:
    print(f"\n{'=' * width}\n  {title}\n{'=' * width}", flush=True)


def sep(width: int = 110) -> None:
    print(f"  {'-' * (width - 2)}", flush=True)
