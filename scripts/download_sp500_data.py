#!/usr/bin/env python3
"""
Download S&P 500 stock data and save to CSV.

This script fetches S&P 500 constituent tickers from Wikipedia and downloads
historical price data from Yahoo Finance. The resulting log returns are saved
to a CSV file for use in analysis notebooks.

Usage:
    python scripts/download_sp500_data.py [--output OUTPUT] [--years YEARS] [--min-days MIN_DAYS]
    
Example:
    python scripts/download_sp500_data.py --output data/sp500_returns.csv --years 10
"""

import argparse
import os
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


def get_sp500_tickers_from_wikipedia() -> list[str]:
    """
    Fetch S&P 500 tickers from Wikipedia using requests with proper headers.
    
    Returns
    -------
    list[str]
        List of S&P 500 ticker symbols, cleaned for Yahoo Finance compatibility.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    # Parse the HTML tables (try lxml first, fall back to html.parser)
    try:
        tables = pd.read_html(StringIO(response.text), flavor='lxml')
    except ImportError:
        tables = pd.read_html(StringIO(response.text), flavor='html.parser')
    sp500_table = tables[0]  # First table contains current constituents
    
    # Extract ticker symbols
    tickers = sp500_table['Symbol'].tolist()
    # Clean up ticker symbols (replace . with - for Yahoo Finance compatibility)
    tickers = [str(t).replace('.', '-') for t in tickers]
    
    return tickers


def download_sp500_data(
    output_path: str = 'data/sp500_returns.csv',
    years: int = 10,
    min_days: int = 2400,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Download S&P 500 stock data and save log returns to CSV.
    
    Parameters
    ----------
    output_path : str
        Path to save the CSV file.
    years : int
        Number of years of historical data to download.
    min_days : int
        Minimum number of trading days required for a stock to be included.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of log returns with dates as index and tickers as columns.
    """
    # Get S&P 500 tickers
    if verbose:
        print("Fetching S&P 500 constituent list from Wikipedia...")
    
    try:
        sp500_tickers = get_sp500_tickers_from_wikipedia()
        if verbose:
            print(f"Successfully fetched {len(sp500_tickers)} tickers from Wikipedia")
    except Exception as e:
        if verbose:
            print(f"Could not fetch from Wikipedia: {e}")
            print("Using curated list of S&P 500 stocks...")
        sp500_tickers = get_curated_sp500_tickers()
        if verbose:
            print(f"Using {len(sp500_tickers)} curated tickers")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years + 60)  # Extra buffer
    
    if verbose:
        print(f"\nDownloading data from {start_date.date()} to {end_date.date()}...")
        print("This may take a few minutes...")
    
    # Download all at once for efficiency
    all_data = yf.download(
        sp500_tickers, 
        start=start_date, 
        end=end_date, 
        progress=verbose,
        threads=True
    )
    
    # Extract close prices
    if 'Close' in all_data.columns.get_level_values(0):
        prices = all_data['Close']
    else:
        prices = all_data['Adj Close']
    
    if verbose:
        print(f"\nDownloaded {len(prices.columns)} stocks with {len(prices)} trading days")
    
    # Compute log returns
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.iloc[1:]  # Remove first NaN row
    
    if verbose:
        print(f"Log returns shape: {log_returns.shape}")
    
    # Filter stocks with sufficient history
    if verbose:
        print(f"\nFiltering stocks with at least {min_days} trading days...")
    
    valid_counts = log_returns.notna().sum()
    valid_tickers = valid_counts[valid_counts >= min_days].index.tolist()
    
    if verbose:
        print(f"Stocks with at least {min_days} trading days: {len(valid_tickers)}")
    
    # Keep stocks with at least 99% non-NaN data, then forward-fill small gaps
    log_returns_subset = log_returns[valid_tickers]
    nan_frac = log_returns_subset.isna().mean()
    nearly_complete = nan_frac[nan_frac < 0.01].index.tolist()
    
    if verbose:
        print(f"Stocks with <1% missing data: {len(nearly_complete)}")
    
    # Forward-fill remaining small gaps (e.g., trading halts), then drop any
    # stocks that still have NaN at the very start (no prior value to fill from)
    log_returns_filtered = (
        log_returns[nearly_complete]
        .ffill()
        .dropna(axis=1, how='any')
    )
    
    if verbose:
        print(f"Stocks after forward-fill cleanup: {len(log_returns_filtered.columns)}")
        print(f"Final shape: {log_returns_filtered.shape}")
        print(f"Date range: {log_returns_filtered.index[0].date()} to {log_returns_filtered.index[-1].date()}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    log_returns_filtered.to_csv(output_path)
    
    if verbose:
        print(f"\nSaved log returns to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return log_returns_filtered


def main():
    parser = argparse.ArgumentParser(
        description='Download S&P 500 stock data and save to CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/sp500_returns.csv',
        help='Output CSV file path (default: data/sp500_returns.csv)'
    )
    parser.add_argument(
        '--years', '-y',
        type=int,
        default=10,
        help='Number of years of historical data (default: 10)'
    )
    parser.add_argument(
        '--min-days', '-m',
        type=int,
        default=2400,
        help='Minimum trading days required per stock (default: 2400)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    download_sp500_data(
        output_path=args.output,
        years=args.years,
        min_days=args.min_days,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
