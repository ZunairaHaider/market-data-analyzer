# Financial Time Series Analysis

A personal Python project I built to apply what I learned in **Foundations of Data Analysis** and **Causal Inference in Time Series** at TU Munich to something more concrete than problem sets.

The idea was to take real stock price data, check whether it's actually clean and reliable, and then do proper time series analysis on it. Turned out there's a lot more to "clean data" than I expected.

---

## What it does

**Five steps, all run automatically:**

1. **Data ingestion** — downloads 4 years of daily prices for AAPL, MSFT, GOOGL, AMZN, JPM via `yfinance`, with CSV caching so you don't re-download every run
2. **Quality checks** — automatically flags missing values, outliers (Z-score > 3σ), zero/negative prices, duplicate dates, and unexpected date gaps
3. **Time series analysis** — daily returns, rolling volatility, correlation matrix, drawdown analysis, Sharpe ratio
4. **Visualisations** — 7 charts saved as PNG
5. **Summary report** — performance metrics exported to CSV

The quality check pipeline is the part I'm most happy with — it's structured the way a real market data team would run it each morning before trading starts.

---

## Charts produced

| # | Chart |
|---|-------|
| 01 | Normalised price performance (base = 100) |
| 02 | Daily return distributions with normal fit overlay |
| 03 | 30-day rolling annualised volatility |
| 04 | Pearson correlation heatmap |
| 05 | Drawdown from peak (%) |
| 06 | Moving averages — MA20, MA50, MA200 |
| 07 | Performance comparison — return, volatility, Sharpe |

---

## How to run

```bash
git clone https://github.com/ZunairaHaider/market-data-analyzer.git
cd market-data-analyzer
pip install -r requirements.txt

# Option 1: script
python main.py

# Option 2: notebook (more visual)
jupyter notebook Financial_Time_Series_Analysis.ipynb
```

---

## Project structure

```
market-data-analyzer/
│
├── Financial_Time_Series_Analysis.ipynb   ← main notebook (start here)
├── main.py                                ← script version
│
├── data_loader.py       # data download + caching
├── quality_checker.py   # quality validation
├── analysis.py          # time series analysis
├── visualiser.py        # charts
│
├── requirements.txt
└── README.md
```

---

## Key things I learned building this

- Even data from Yahoo Finance has outliers that need investigating — never assume it's clean
- Financial returns have fat tails (kurtosis > 3) — normality assumptions underestimate extreme losses
- Volatility clustering is very visible — the basis for GARCH models I want to try next
- High correlations between large US stocks (~0.7) give less diversification than it looks like on paper

---

## Background

Built alongside my MSc coursework at TU Munich.  
Relevant courses: MA4800 Foundations of Data Analysis · CIT4230006 Causal Inference in Time Series

**Author:** Zunaira Haider — [github.com/ZunairaHaider](https://github.com/ZunairaHaider)
