"""Data acquisition and cleaning (Sect. 4 of the paper)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Fallback mapping from Yahoo Finance suffix to listing currency.
# Used only when yfinance metadata is unavailable.
SUFFIX_CURRENCY = {
    ".SA": "BRL", ".SZ": "CNY", ".SS": "CNY", ".T": "JPY", ".SG": "EUR",
    ".DU": "EUR", ".F": "EUR", ".BE": "EUR", ".VI": "EUR", ".HK": "HKD",
    ".KS": "KRW", ".BK": "THB", ".TWO": "TWD", ".SR": "SAR", ".NS": "INR",
    ".BO": "INR", ".IL": None,  # LSE IOB lines: currency varies -> must be checked
}


def load_universe(path: Path, only_included: bool = True) -> pd.DataFrame:
    uni = pd.read_csv(path, dtype={"ticker": str})
    if only_included:
        uni = uni[uni["included"] == 1]
    return uni.reset_index(drop=True)


def guess_currency(ticker: str) -> str | None:
    for suffix, cur in SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return cur
    return "USD" if "." not in ticker else None


def download_prices(tickers: list[str], start: str, end: str, interval: str = "1wk",
                    field: str = "Close") -> pd.DataFrame:
    """Download weekly adjusted prices from Yahoo Finance with explicit dates.

    ``end`` is inclusive. Using explicit dates (instead of ``period="6y"``)
    makes the download reproducible.
    """
    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    raw = yf.download(tickers=tickers, start=start, end=end_excl, interval=interval,
                      auto_adjust=True, progress=False, group_by="column")
    prices = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.loc[start:end]
    return prices[sorted(prices.columns)]


def download_currencies(tickers: list[str]) -> pd.Series:
    """Listing currency of every ticker (yfinance metadata, suffix fallback)."""
    import yfinance as yf

    out = {}
    for t in tickers:
        cur = None
        try:
            cur = yf.Ticker(t).fast_info.get("currency")
        except Exception:  # network / delisted
            pass
        out[t] = (cur or guess_currency(t) or "UNKNOWN").upper()
    return pd.Series(out, name="currency")


def download_fx_to_usd(currencies: list[str], start: str, end: str,
                       interval: str = "1wk") -> pd.DataFrame:
    """Weekly FX rates expressed as USD per 1 unit of local currency."""
    import yfinance as yf

    fx = {}
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    for cur in sorted(set(currencies) - {"USD", "UNKNOWN"}):
        s = yf.download(f"{cur}USD=X", start=start, end=end_excl, interval=interval,
                        auto_adjust=True, progress=False)["Close"].squeeze()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        fx[cur] = s
    fx = pd.DataFrame(fx)
    fx["USD"] = 1.0
    return fx


def clean_prices(prices: pd.DataFrame, max_ffill: int = 4) -> pd.DataFrame:
    """Align to a Monday weekly grid and forward-fill short gaps.

    No back-filling: back-filling copies future prices into the past and
    leaks information into the forecasting and optimisation stages.
    Leading NaNs (before a stock's first quote) are kept as NaN.
    """
    prices = prices.sort_index()
    grid = pd.date_range(prices.index.min(), prices.index.max(), freq="W-MON")
    prices = prices.reindex(prices.index.union(grid)).ffill(limit=max_ffill).reindex(grid)
    prices.index.name = "Date"
    return prices


def to_usd(prices: pd.DataFrame, currency: pd.Series, fx: pd.DataFrame) -> pd.DataFrame:
    fx = fx.reindex(prices.index).ffill().bfill()
    out = prices.copy()
    for t in prices.columns:
        cur = currency.get(t, "USD")
        if cur not in fx.columns:
            raise KeyError(f"No FX series for {t} ({cur}).")
        out[t] = prices[t] * fx[cur]
    return out


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def load_risk_free(path: Path, dates: pd.DatetimeIndex, periods_per_year: int = 52) -> pd.Series:
    """ECB deposit facility rate (annual %, step function) -> weekly rate.

    Expected CSV: ``date;deposit`` with dates dd/mm/YYYY and decimal comma,
    as published by the ECB Statistical Data Warehouse.
    """
    ti = pd.read_csv(path, sep=";", decimal=",")
    ti["date"] = pd.to_datetime(ti["date"], dayfirst=True)
    ti = ti.set_index("date")["deposit"].astype(float).sort_index()
    daily = ti.reindex(pd.date_range(ti.index.min(), max(ti.index.max(), dates.max()), freq="D")).ffill()
    annual = daily.reindex(dates).ffill().fillna(0.0) / 100.0
    return (1.0 + annual) ** (1.0 / periods_per_year) - 1.0


def market_caps(prices_usd: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """Weekly market capitalisation in USD (prices x shares outstanding)."""
    shares = shares.reindex(prices_usd.columns)
    if shares.isna().any():
        missing = shares[shares.isna()].index.tolist()
        raise ValueError(f"Shares outstanding missing for: {missing}")
    return prices_usd.mul(shares, axis=1)


def split_dates(index: pd.DatetimeIndex, train_end: str, val_end: str):
    """Boolean masks for the 70 / 10 / 20 chronological split (Sect. 4.1)."""
    train = index <= pd.Timestamp(train_end)
    val = (index > pd.Timestamp(train_end)) & (index <= pd.Timestamp(val_end))
    test = index > pd.Timestamp(val_end)
    return np.asarray(train), np.asarray(val), np.asarray(test)
