"""
analysis.py
-----------
Core time-series analysis functions:
  - Daily returns
  - Rolling statistics (moving average, volatility)
  - Correlation matrix
  - Drawdown analysis
  - Annualised performance metrics
"""

import pandas as pd
import numpy as np


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns."""
    returns = prices.pct_change().dropna()
    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns — more statistically robust for modelling."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret


def rolling_statistics(prices: pd.DataFrame,
                        windows: list = [20, 50, 200]) -> dict:
    """
    Compute rolling mean (moving average) and rolling volatility
    for each ticker across multiple window sizes.

    Returns dict: {ticker: {"ma": DataFrame, "vol": DataFrame}}
    """
    result = {}
    for ticker in prices.columns:
        s = prices[ticker]
        ma  = pd.DataFrame({f"MA_{w}": s.rolling(w).mean() for w in windows})
        vol = pd.DataFrame({f"Vol_{w}d": s.pct_change().rolling(w).std() * np.sqrt(252)
                            for w in windows})
        result[ticker] = {"price": s, "moving_averages": ma, "volatility": vol}
    return result


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix of daily returns.
    Useful for understanding co-movement between assets.
    """
    return returns.corr().round(4)


def drawdown_analysis(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute drawdown series for each ticker.
    Drawdown = how far price has fallen from its running peak.
    Max drawdown is a key risk metric in asset management.
    """
    result = {}
    for ticker in prices.columns:
        s          = prices[ticker]
        peak       = s.cummax()
        drawdown   = (s - peak) / peak * 100   # expressed as %
        result[ticker] = drawdown
    return pd.DataFrame(result)


def performance_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualised performance metrics for each ticker:
      - Total return
      - Annualised return
      - Annualised volatility
      - Sharpe ratio (risk-free rate = 2%)
      - Max drawdown
    """
    returns   = compute_returns(prices)
    trading_days = 252
    rf        = 0.02   # risk-free rate assumption

    records = []
    for ticker in prices.columns:
        r      = returns[ticker].dropna()
        total  = (prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1) * 100
        ann_r  = ((1 + r.mean()) ** trading_days - 1) * 100
        ann_v  = r.std() * np.sqrt(trading_days) * 100
        sharpe = (ann_r / 100 - rf) / (ann_v / 100) if ann_v != 0 else np.nan

        dd     = drawdown_analysis(prices[[ticker]])
        max_dd = dd[ticker].min()

        records.append({
            "Ticker"           : ticker,
            "Total Return (%)" : round(total, 2),
            "Ann. Return (%)"  : round(ann_r, 2),
            "Ann. Volatility (%)": round(ann_v, 2),
            "Sharpe Ratio"     : round(sharpe, 3),
            "Max Drawdown (%)" : round(max_dd, 2),
        })

    summary = pd.DataFrame(records).set_index("Ticker")
    return summary
