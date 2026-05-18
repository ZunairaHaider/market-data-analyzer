"""
quality_checker.py
------------------
Detects and reports data quality issues in a price DataFrame.
This is the core of the project — directly mirrors what market data
teams do in asset management every day.
"""

import pandas as pd
import numpy as np


def run_quality_report(df: pd.DataFrame) -> dict:
    """
    Run all quality checks on a price DataFrame.

    Returns a dictionary with all findings — can be printed or saved.
    """
    print("\n" + "="*60)
    print("  MARKET DATA QUALITY REPORT")
    print("="*60)

    report = {}
    report["shape"]            = _check_shape(df)
    report["missing_values"]   = _check_missing(df)
    report["outliers"]         = _check_outliers(df)
    report["zero_prices"]      = _check_zeros(df)
    report["duplicate_dates"]  = _check_duplicates(df)
    report["gaps"]             = _check_date_gaps(df)
    report["negative_prices"]  = _check_negative(df)

    _print_summary(report)
    return report


# ── Individual checks ────────────────────────────────────────────────

def _check_shape(df: pd.DataFrame) -> dict:
    result = {"rows": len(df), "columns": len(df.columns), "tickers": list(df.columns)}
    print(f"\n[INFO] Dataset: {result['rows']} trading days × {result['columns']} tickers")
    print(f"       Period : {df.index[0].date()} → {df.index[-1].date()}")
    return result


def _check_missing(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    result  = {}
    for ticker in df.columns:
        result[ticker] = {"count": int(missing[ticker]), "pct": float(pct[ticker])}

    total = missing.sum()
    if total == 0:
        print("\n[✓] Missing values   : None detected")
    else:
        print(f"\n[!] Missing values   : {total} total")
        for t, v in result.items():
            if v["count"] > 0:
                print(f"      {t}: {v['count']} missing ({v['pct']}%)")
    return result


def _check_outliers(df: pd.DataFrame, threshold: float = 3.0) -> dict:
    """
    Flag daily returns beyond `threshold` standard deviations from the mean.
    Uses Z-score method — standard in financial data quality pipelines.
    """
    returns = df.pct_change().dropna()
    result  = {}
    total   = 0

    for ticker in returns.columns:
        col    = returns[ticker].dropna()
        mean   = col.mean()
        std    = col.std()
        z      = (col - mean) / std
        flags  = z[np.abs(z) > threshold]
        result[ticker] = {
            "count": len(flags),
            "dates": [str(d.date()) for d in flags.index],
            "values": [round(v * 100, 2) for v in col[flags.index].values]
        }
        total += len(flags)

    if total == 0:
        print(f"[✓] Outliers (|z|>{threshold}): None detected")
    else:
        print(f"[!] Outliers (|z|>{threshold}): {total} flagged across all tickers")
        for t, v in result.items():
            if v["count"] > 0:
                print(f"      {t}: {v['count']} outlier(s)")
                for d, val in zip(v["dates"], v["values"]):
                    print(f"        → {d}  daily return: {val:+.2f}%")
    return result


def _check_zeros(df: pd.DataFrame) -> dict:
    zeros  = (df == 0).sum()
    result = {t: int(zeros[t]) for t in df.columns}
    total  = zeros.sum()
    if total == 0:
        print("[✓] Zero prices      : None detected")
    else:
        print(f"[!] Zero prices      : {total} detected — likely data feed errors")
    return result


def _check_duplicates(df: pd.DataFrame) -> dict:
    dupes  = df.index.duplicated().sum()
    result = {"count": int(dupes)}
    if dupes == 0:
        print("[✓] Duplicate dates  : None detected")
    else:
        print(f"[!] Duplicate dates  : {dupes} found — index integrity issue")
    return result


def _check_date_gaps(df: pd.DataFrame) -> dict:
    """
    Detect unexpectedly large gaps between consecutive trading days.
    Gaps > 5 calendar days (excluding normal weekends/holidays) are flagged.
    """
    diffs = pd.Series(df.index).diff().dropna()
    large = diffs[diffs > pd.Timedelta(days=5)]
    result = {"count": len(large), "gaps": []}

    for idx in large.index:
        result["gaps"].append({
            "from": str(df.index[idx - 1].date()),
            "to"  : str(df.index[idx].date()),
            "days": int(diffs.iloc[idx - 1].days)
        })

    if len(large) == 0:
        print("[✓] Date gaps        : No unusual gaps detected")
    else:
        print(f"[!] Date gaps        : {len(large)} gap(s) > 5 days")
        for g in result["gaps"]:
            print(f"      {g['from']} → {g['to']}  ({g['days']} days)")
    return result


def _check_negative(df: pd.DataFrame) -> dict:
    negs   = (df < 0).sum()
    result = {t: int(negs[t]) for t in df.columns}
    total  = negs.sum()
    if total == 0:
        print("[✓] Negative prices  : None detected")
    else:
        print(f"[!] Negative prices  : {total} — critical data error")
    return result


def _print_summary(report: dict):
    issues = (
        sum(v["count"] for v in report["missing_values"].values()) +
        sum(v["count"] for v in report["outliers"].values()) +
        sum(v for v in report["zero_prices"].values()) +
        report["duplicate_dates"]["count"] +
        report["gaps"]["count"] +
        sum(v for v in report["negative_prices"].values())
    )
    print("\n" + "-"*60)
    if issues == 0:
        print("  RESULT: Data passed all quality checks ✓")
    else:
        print(f"  RESULT: {issues} issue(s) found — review flagged items above")
    print("="*60 + "\n")
