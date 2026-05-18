"""
data_loader.py
--------------
Downloads historical price data for a list of tickers using yfinance.
Falls back to synthetic data if yfinance is unavailable (e.g. offline).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Try to import yfinance; fall back gracefully ──────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def download_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str   e.g. ["AAPL", "MSFT", "GOOGL"]
    start   : str           e.g. "2020-01-01"
    end     : str           e.g. "2024-01-01"

    Returns
    -------
    pd.DataFrame  — columns = tickers, index = date
    """
    if YFINANCE_AVAILABLE:
        print(f"Downloading data for: {tickers}")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        # yfinance returns MultiIndex when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices.dropna(how="all", inplace=True)
        print(f"  Downloaded {len(prices)} rows, {prices.shape[1]} tickers.")
        return prices
    else:
        print("yfinance not available — generating synthetic data for demonstration.")
        return _generate_synthetic_data(tickers, start, end)


def _generate_synthetic_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Generate realistic-looking synthetic price data using Geometric Brownian Motion.
    Used as a fallback when yfinance is not installed.
    """
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq="B")  # business days only
    n = len(dates)

    # Realistic starting prices and drift/volatility per ticker
    params = {
        "AAPL":  {"S0": 130.0, "mu": 0.0003, "sigma": 0.018},
        "MSFT":  {"S0": 240.0, "mu": 0.0004, "sigma": 0.016},
        "GOOGL": {"S0": 140.0, "mu": 0.0003, "sigma": 0.019},
        "AMZN":  {"S0": 170.0, "mu": 0.0002, "sigma": 0.021},
        "JPM":   {"S0":  140.0, "mu": 0.0002, "sigma": 0.015},
    }

    data = {}
    for ticker in tickers:
        p = params.get(ticker, {"S0": 100.0, "mu": 0.0003, "sigma": 0.018})
        shocks = np.random.normal(p["mu"], p["sigma"], n)
        prices = p["S0"] * np.exp(np.cumsum(shocks))
        data[ticker] = np.round(prices, 2)

    df = pd.DataFrame(data, index=dates)
    print(f"  Generated {len(df)} rows of synthetic data for {tickers}.")
    return df


def load_or_cache(tickers: list, start: str, end: str,
                  cache_path: str = "data/prices.csv") -> pd.DataFrame:
    """
    Load from CSV cache if available, otherwise download and save.
    Avoids repeated API calls during development.
    """
    try:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded cached data from '{cache_path}' ({len(df)} rows).")
        return df
    except FileNotFoundError:
        df = download_data(tickers, start, end)
        df.to_csv(cache_path)
        print(f"Saved data to '{cache_path}'.")
        return df
