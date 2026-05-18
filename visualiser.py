"""
visualiser.py
-------------
All charts for the project. Saves PNG files to the outputs/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# ── Global style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f8f9fa",
    "axes.grid"        : True,
    "grid.color"       : "#e0e0e0",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.6,
    "font.family"      : "sans-serif",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

COLORS = ["#003399", "#e63946", "#2a9d8f", "#e9c46a", "#f4a261"]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_normalised_prices(prices: pd.DataFrame):
    """
    Plot all tickers normalised to 100 at start date.
    Makes it easy to compare performance regardless of price level.
    """
    normed = prices / prices.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ticker in enumerate(normed.columns):
        ax.plot(normed.index, normed[ticker], label=ticker,
                color=COLORS[i % len(COLORS)], linewidth=1.8)

    ax.set_title("Normalised Price Performance (Base = 100)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Indexed Price")
    ax.set_xlabel("")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_normalised_prices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_daily_returns(returns: pd.DataFrame):
    """
    Plot daily return distributions as histograms with KDE overlay.
    Shows the return profile and fat tails of each asset.
    """
    n = len(returns.columns)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for i, ticker in enumerate(returns.columns):
        ax = axes[i]
        data = returns[ticker].dropna() * 100
        ax.hist(data, bins=60, color=COLORS[i % len(COLORS)],
                alpha=0.7, edgecolor="white", density=True)

        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x   = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, kde(x), color="black", linewidth=1.5)

        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(ticker, fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        if i == 0:
            ax.set_ylabel("Density")

    fig.suptitle("Daily Return Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_return_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rolling_volatility(prices: pd.DataFrame, window: int = 30):
    """
    Plot 30-day rolling annualised volatility for all tickers.
    Volatility clustering is a well-known financial phenomenon.
    """
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, ticker in enumerate(rolling_vol.columns):
        ax.plot(rolling_vol.index, rolling_vol[ticker], label=ticker,
                color=COLORS[i % len(COLORS)], linewidth=1.5, alpha=0.85)

    ax.set_title(f"{window}-Day Rolling Annualised Volatility (%)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_rolling_volatility.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """
    Plot correlation heatmap with annotations.
    Useful for understanding diversification between assets.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        annot_kws={"size": 11}, ax=ax, square=True
    )
    ax.set_title("Return Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_drawdowns(drawdowns: pd.DataFrame):
    """
    Plot drawdown series for each ticker.
    Max drawdown is one of the most important risk metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ticker in enumerate(drawdowns.columns):
        ax.fill_between(drawdowns.index, drawdowns[ticker], 0,
                        alpha=0.4, color=COLORS[i % len(COLORS)], label=ticker)
        ax.plot(drawdowns.index, drawdowns[ticker],
                color=COLORS[i % len(COLORS)], linewidth=1.2)

    ax.set_title("Drawdown Analysis (%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_drawdowns.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_moving_averages(prices: pd.DataFrame, ticker: str):
    """
    Plot price with 20, 50, 200-day moving averages for one ticker.
    Classic technical analysis chart used in every finance context.
    """
    s   = prices[ticker]
    ma20  = s.rolling(20).mean()
    ma50  = s.rolling(50).mean()
    ma200 = s.rolling(200).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.index,    s,     label="Price",   color="#003399", linewidth=1.5, alpha=0.9)
    ax.plot(ma20.index, ma20,  label="MA 20",   color="#e63946", linewidth=1.2, linestyle="--")
    ax.plot(ma50.index, ma50,  label="MA 50",   color="#2a9d8f", linewidth=1.2, linestyle="--")
    ax.plot(ma200.index,ma200, label="MA 200",  color="#f4a261", linewidth=1.5, linestyle="-.")

    ax.set_title(f"{ticker} — Price & Moving Averages", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_moving_averages_{ticker}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_performance_summary(summary: pd.DataFrame):
    """
    Bar chart comparing annualised return and volatility side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("Ann. Return (%)",    "#003399", "Annualised Return (%)"),
        ("Ann. Volatility (%)", "#e63946", "Annualised Volatility (%)"),
        ("Sharpe Ratio",       "#2a9d8f", "Sharpe Ratio"),
    ]

    for ax, (col, color, title) in zip(axes, metrics):
        vals = summary[col]
        bars = ax.bar(vals.index, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticklabels(vals.index, rotation=30)
        ax.axhline(0, color="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.01 * abs(vals.max())),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Performance Summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/07_performance_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
