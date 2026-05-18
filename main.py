"""
main.py
-------
Entry point for the Market Data Analyzer.
Run this file to execute the full pipeline:
  1. Load data
  2. Quality check
  3. Analysis
  4. Visualisation
  5. Save summary report
"""

import pandas as pd
import os
from data_loader    import load_or_cache
from quality_checker import run_quality_report
from analysis       import (compute_returns, compute_log_returns,
                             correlation_matrix, drawdown_analysis,
                             performance_summary)
from visualiser     import (plot_normalised_prices, plot_daily_returns,
                             plot_rolling_volatility, plot_correlation_heatmap,
                             plot_drawdowns, plot_moving_averages,
                             plot_performance_summary)

# ── Configuration ─────────────────────────────────────────────────────
TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
START_DATE = "2020-01-01"
END_DATE   = "2024-01-01"
CACHE_PATH = "data/prices.csv"


def main():
    print("\n" + "="*60)
    print("  MARKET DATA ANALYZER")
    print("  Zunaira Haider — github.com/ZunairaHaider")
    print("="*60)

    # ── Step 1: Load Data ─────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    prices = load_or_cache(TICKERS, START_DATE, END_DATE, CACHE_PATH)

    # ── Step 2: Quality Check ─────────────────────────────────────────
    print("\n[2/5] Running quality checks...")
    quality_report = run_quality_report(prices)

    # ── Step 3: Analysis ──────────────────────────────────────────────
    print("\n[3/5] Running analysis...")
    returns  = compute_returns(prices)
    log_ret  = compute_log_returns(prices)
    corr     = correlation_matrix(returns)
    dd       = drawdown_analysis(prices)
    summary  = performance_summary(prices)

    print("\n  Performance Summary:")
    print(summary.to_string())

    # ── Step 4: Visualisations ────────────────────────────────────────
    print("\n[4/5] Generating charts...")
    plot_normalised_prices(prices)
    plot_daily_returns(returns)
    plot_rolling_volatility(prices)
    plot_correlation_heatmap(corr)
    plot_drawdowns(dd)
    plot_moving_averages(prices, TICKERS[0])   # detailed chart for first ticker
    plot_performance_summary(summary)

    # ── Step 5: Save Report ───────────────────────────────────────────
    print("\n[5/5] Saving summary report...")
    os.makedirs("outputs", exist_ok=True)
    summary.to_csv("outputs/performance_summary.csv")
    corr.to_csv("outputs/correlation_matrix.csv")
    print("  Saved: outputs/performance_summary.csv")
    print("  Saved: outputs/correlation_matrix.csv")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Charts saved to: outputs/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
