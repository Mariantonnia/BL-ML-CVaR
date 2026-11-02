"""Accuracy-weighted aggregation of the five forecasts (Sect. 3.1, Eqs. 1-3).

For asset j and week t the weight of technique i is computed from the
absolute errors of the *previous* week (t-1), which are known when the
forecast for week t is issued:

    Re_i   = E_i / sum_k E_k                      (Eq. 1)
    w_i    = (1 - Re_i) / sum_k (1 - Re_k)        (Eq. 2)
    NewPred = sum_i w_i * Pred_i                  (Eq. 3)

With 5 techniques sum_k (1 - Re_k) = 4, so the weights always sum to 1.
For the first week (no previous error) equal weights are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align(preds: dict[str, pd.DataFrame], actual: pd.DataFrame):
    idx = actual.index
    cols = actual.columns
    for p in preds.values():
        idx = idx.intersection(p.index)
        cols = cols.intersection(p.columns)
    return {k: v.loc[idx, cols] for k, v in preds.items()}, actual.loc[idx, cols]


def accuracy_weights(preds: dict[str, pd.DataFrame], actual: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Weights w_i (Eq. 2) to be applied at week t, using errors of week t-1."""
    preds, actual = _align(preds, actual)
    errors = {k: (p - actual).abs() for k, p in preds.items()}
    total = sum(errors.values())
    rel = {k: (e / total) for k, e in errors.items()}                    # Eq. 1
    n = len(preds)
    # If all errors are zero for an asset/week, fall back to equal weights.
    rel = {k: r.where(total > 0, 1.0 / n) for k, r in rel.items()}
    denom = sum((1 - r) for r in rel.values())
    weights = {k: ((1 - r) / denom).shift(1) for k, r in rel.items()}    # Eq. 2, lagged
    return {k: w.fillna(1.0 / n) for k, w in weights.items()}


def aggregate(preds: dict[str, pd.DataFrame], actual: pd.DataFrame):
    """Return the aggregated forecast (Eq. 3) and the weights used."""
    w = accuracy_weights(preds, actual)
    preds, _ = _align(preds, actual)
    agg = sum(w[k] * preds[k] for k in preds)
    return agg, w


def winners_per_week(preds: dict[str, pd.DataFrame], actual: pd.DataFrame) -> pd.DataFrame:
    """Number of assets for which each technique has the lowest error (Table A2)."""
    preds, actual = _align(preds, actual)
    names = list(preds)
    err = np.stack([(preds[k] - actual).abs().to_numpy() for k in names])  # (model, week, asset)
    best = np.nanargmin(np.where(np.isnan(err), np.inf, err), axis=0)
    counts = {k: (best == i).sum(axis=1) for i, k in enumerate(names)}
    return pd.DataFrame(counts, index=actual.index)
