"""Session-wide pytest fixtures for the normix test suite."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

# Float64 everywhere — must run before any JAX computation in collected tests.
jax.config.update("jax_enable_x64", True)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_csv(path: Path, n_stocks: int):
    if not path.exists():
        pytest.skip(f"SP500 data not found at {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna(axis=1)
    return jnp.asarray(df.values[:, :n_stocks], dtype=jnp.float64)


@pytest.fixture(scope="session")
def sp500_returns():
    """SP500 log-returns matrix from ``data/sp500_returns.csv`` (5 stocks)."""
    return _load_csv(_DATA_DIR / "sp500_returns.csv", n_stocks=5)


@pytest.fixture(scope="session")
def sp500_sample():
    """SP500 log-returns matrix from ``data/sp500_sample.csv`` (10 stocks)."""
    return _load_csv(_DATA_DIR / "sp500_sample.csv", n_stocks=10)
