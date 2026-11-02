"""Shared utilities for the one-week-ahead price forecasts (Sect. 3.1, 4.1).

Conventions used by every model
-------------------------------
* Input: the previous ``window`` (=30) weekly prices; output: next week's price.
* Chronological split by *target date*: train <= train_end < val <= val_end < test.
* Any scaler is fitted on training data only.
* Each model returns a DataFrame of one-step-ahead predictions for every
  validation and test date (rows) and asset (columns). Each prediction uses
  only prices observed up to the previous week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(series: pd.Series, window: int):
    """Return (X, y, target_dates) for one-step-ahead forecasting."""
    s = series.dropna()
    vals = s.to_numpy(dtype=float)
    X = np.lib.stride_tricks.sliding_window_view(vals[:-1], window)
    y = vals[window:]
    dates = s.index[window:]
    return X.copy(), y.copy(), dates


def split_masks(dates: pd.DatetimeIndex, train_end: str, val_end: str):
    d = pd.DatetimeIndex(dates)
    train = d <= pd.Timestamp(train_end)
    val = (d > pd.Timestamp(train_end)) & (d <= pd.Timestamp(val_end))
    test = d > pd.Timestamp(val_end)
    return train, val, test


class TrainMinMax:
    """Min-max scaler fitted on the training targets/inputs only."""

    def fit(self, values: np.ndarray) -> "TrainMinMax":
        self.lo = float(np.nanmin(values))
        self.hi = float(np.nanmax(values))
        self.span = self.hi - self.lo if self.hi > self.lo else 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.lo) / self.span

    def inverse(self, x):
        return np.asarray(x, dtype=float) * self.span + self.lo


def forecast_metrics(actual: pd.DataFrame, pred: pd.DataFrame) -> pd.Series:
    """MAE, RMSE and MAPE (%) pooled over all assets and dates (Table 2)."""
    a, p = actual.align(pred, join="inner")
    err = (p - a).to_numpy().ravel()
    act = a.to_numpy().ravel()
    ok = ~np.isnan(err) & ~np.isnan(act) & (act != 0)
    err, act = err[ok], act[ok]
    return pd.Series({
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE": 100 * np.mean(np.abs(err / act)),
    })
