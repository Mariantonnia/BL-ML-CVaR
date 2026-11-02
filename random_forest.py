"""Random-forest forecaster with 30 lagged prices as features (Table 1)."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .common import make_windows, split_masks

log = logging.getLogger(__name__)


def forecast_asset(series: pd.Series, cfg_fc: dict, seed: int):
    rf_cfg = cfg_fc["rf"]
    X, y, dates = make_windows(series, rf_cfg["n_lags"])
    tr, va, te = split_masks(dates, cfg_fc["train_end"], cfg_fc["val_end"])
    grid = GridSearchCV(
        RandomForestRegressor(random_state=seed),
        rf_cfg["param_grid"],
        cv=TimeSeriesSplit(n_splits=rf_cfg["cv_splits"]),  # chronological folds
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X[tr], y[tr])
    out = va | te
    pred = grid.best_estimator_.predict(X[out])
    return pd.Series(pred, index=dates[out], name=series.name), grid.best_params_


def forecast(prices: pd.DataFrame, cfg_fc: dict, seed: int):
    preds, params = {}, {}
    for t in prices.columns:
        log.info("RF  %s", t)
        preds[t], params[t] = forecast_asset(prices[t], cfg_fc, seed)
    return pd.DataFrame(preds), params
