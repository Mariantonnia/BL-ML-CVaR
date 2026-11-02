"""ARIMA forecaster re-estimated every week on a rolling 30-week window.

For each target week the order (p, d, q) is selected by AIC with
d in {0, .., max_d} chosen by an ADF test and p, q in {0, .., max_p/q}.
Fits whose forecast is non-finite or outside [P_t/3, 3 P_t] are discarded;
if none is left the random-walk forecast P_t is used.
"""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd

from .common import split_masks

log = logging.getLogger(__name__)


def _choose_d(x: np.ndarray, max_d: int) -> int:
    from statsmodels.tsa.stattools import adfuller

    for d in range(max_d + 1):
        z = np.diff(x, n=d) if d else x
        try:
            if np.ptp(z) == 0 or adfuller(z, autolag="AIC")[1] < 0.05:
                return d
        except Exception:
            return d
    return max_d


def one_step(x: np.ndarray, max_p: int, max_d: int, max_q: int) -> float:
    from statsmodels.tsa.arima.model import ARIMA

    d = _choose_d(x, max_d)
    best_aic, best_fc = np.inf, x[-1]  # random-walk fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
            try:
                res = ARIMA(x, order=(p, d, q)).fit()
            except Exception:
                continue
            fc = float(res.forecast(1)[0])
            # discard explosive fits (short 30-week windows can produce them)
            if np.isfinite(res.aic) and res.aic < best_aic and np.isfinite(fc) \
                    and x[-1] / 3 < fc < 3 * x[-1]:
                best_aic, best_fc = res.aic, fc
    return best_fc


def forecast_asset(series: pd.Series, cfg_fc: dict):
    a = cfg_fc["arima"]
    w = a["rolling_window"]
    s = series.dropna()
    dates = s.index[w:]
    _, va, te = split_masks(dates, cfg_fc["train_end"], cfg_fc["val_end"])
    out_dates = dates[va | te]
    vals = s.to_numpy(dtype=float)
    pos = {d: i for i, d in enumerate(s.index)}
    preds = [one_step(vals[pos[d] - w:pos[d]], a["max_p"], a["max_d"], a["max_q"]) for d in out_dates]
    return pd.Series(preds, index=out_dates, name=series.name)


def forecast(prices: pd.DataFrame, cfg_fc: dict, n_jobs: int = -1):
    from joblib import Parallel, delayed

    res = Parallel(n_jobs=n_jobs)(delayed(forecast_asset)(prices[t], cfg_fc) for t in prices.columns)
    return pd.DataFrame({s.name: s for s in res}), {}
