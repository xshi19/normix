#!/usr/bin/env python3
"""
Download Dow Jones 30 daily log returns and save to CSV.

Mirrors the data used in [Shi2016] for Figures 10, 11, and 14 of
``docs/pdfs/generalized_hyperbolic_finance.pdf``: Dow Jones Industrial
Average constituents, daily prices 2005-01-01 through 2015-12-31.

Usage::

    python scripts/download_dj30.py [--output data/dj30_returns.csv]
                                    [--start 2005-01-01] [--end 2015-12-31]

The current DJIA membership is hard-coded; survivorship bias is a known
limitation but matches the convention in the paper (it backtests the
ex-post membership). For an exact reproduction one would substitute the
historical-membership list per sub-period.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yfinance as yf


DJIA_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    "XOM",
]


def download_dj30(
    output_path: str = "data/dj30_returns.csv",
    start: str = "2005-01-01",
    end: str = "2015-12-31",
    verbose: bool = True,
) -> pd.DataFrame:
    """Download DJIA log returns and write them to ``output_path``."""
    if verbose:
        print(f"Downloading {len(DJIA_TICKERS)} DJIA tickers "
              f"from {start} to {end}...")

    raw = yf.download(
        DJIA_TICKERS, start=start, end=end,
        progress=verbose, threads=True, auto_adjust=False,
    )

    if "Adj Close" in raw.columns.get_level_values(0):
        prices = raw["Adj Close"]
    else:
        prices = raw["Close"]

    log_returns = np.log(prices / prices.shift(1)).iloc[1:]

    nan_frac = log_returns.isna().mean()
    keep = nan_frac[nan_frac < 0.05].index.tolist()
    log_returns = log_returns[keep].ffill().dropna(axis=1, how="any")

    if verbose:
        print(f"Final shape: {log_returns.shape} "
              f"({log_returns.index[0].date()} to {log_returns.index[-1].date()})")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    log_returns.to_csv(output_path)
    if verbose:
        print(f"Wrote {output_path} "
              f"({os.path.getsize(output_path) / 1024:.1f} KB)")
    return log_returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="data/dj30_returns.csv")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2015-12-31")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()
    download_dj30(
        output_path=args.output, start=args.start, end=args.end,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
