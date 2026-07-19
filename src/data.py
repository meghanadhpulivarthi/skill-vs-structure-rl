"""ETF price/return panel via yfinance, cached to a git-ignored data/ dir.

Universe (spec §8): liquid multi-asset ETFs spanning equity sectors, duration,
gold, and international equity — chosen for a long survivorship-clean daily
history through 2008 / 2020 / 2022 stress. Adjusted close only (splits/dividends
handled by the vendor). The cache makes runs restartable and keeps the large raw
data out of the repo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Config — edit these directly
TICKERS = [
    "SPY",                                  # US large-cap equity
    "XLK", "XLF", "XLE", "XLV", "XLU",      # sector SPDRs (tech/fin/energy/health/utilities)
    "IEF", "TLT",                           # 7-10y and 20y+ US Treasuries (the safe-haven sleeve)
    "GLD",                                  # gold
    "EFA", "EEM",                           # developed-ex-US and emerging equity
]
START = "2005-01-01"
END = "2025-01-01"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "etf_panel.parquet"


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().iloc[1:]
    # A NaN after the first row means a ticker lacks full history over the window.
    # Drop it loudly rather than forward-filling (which would fabricate returns).
    bad = returns.columns[returns.isna().any()].tolist()
    if bad:
        print(f"compute_returns: dropping {len(bad)} tickers with gaps: {bad}")
        returns = returns.drop(columns=bad)
    return returns


def download_prices(tickers: list, start: str, end: str, cache_path: Path) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"download_prices: loading cache {cache_path}")
        return pd.read_parquet(cache_path)
    import yfinance as yf
    print(f"download_prices: downloading {len(tickers)} tickers {start}..{end} from yfinance")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    print(f"download_prices: cached {prices.shape[0]} rows x {prices.shape[1]} tickers to {cache_path}")
    return prices


def load_etf_panel(cache_path: Path = DEFAULT_CACHE, start: str = START, end: str = END) -> dict:
    prices = download_prices(TICKERS, start, end, cache_path)
    returns = compute_returns(prices)
    print(f"load_etf_panel: {returns.shape[0]} return rows x {returns.shape[1]} tickers "
          f"({returns.index[0].date()}..{returns.index[-1].date()})")
    return {
        "returns": returns.to_numpy(dtype=float),
        "dates": returns.index.to_numpy(),
        "tickers": list(returns.columns),
    }
