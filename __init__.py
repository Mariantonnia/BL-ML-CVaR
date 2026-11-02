"""One-week-ahead price forecasters: LSTM, GRU, RNN, Random Forest, ARIMA."""
from __future__ import annotations

import pandas as pd


def run_model(name: str, prices: pd.DataFrame, cfg: dict):
    """Dispatch to the requested model. Returns (predictions, best_params)."""
    fc, seed = cfg["forecast"], cfg["seed"]
    if name in ("lstm", "gru", "rnn"):
        from . import deep

        return deep.forecast(name, prices, fc, seed)
    if name == "rf":
        from . import random_forest

        return random_forest.forecast(prices, fc, seed)
    if name == "arima":
        from . import arima

        return arima.forecast(prices, fc)
    raise ValueError(f"Unknown model {name!r}")
