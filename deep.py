"""LSTM, GRU and simple RNN forecasters (Table 1, Sect. 4.1).

Architecture: three stacked recurrent layers (the first two return sequences)
each followed by Dropout, and a Dense(1) output. Hyper-parameters (units per
layer, dropout per layer, batch size) are tuned per asset on the validation
split, either with Optuna/TPE or with an exhaustive grid.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from .common import TrainMinMax, make_windows, split_masks

log = logging.getLogger(__name__)


def _layer(kind: str):
    from tensorflow.keras import layers

    return {"lstm": layers.LSTM, "gru": layers.GRU, "rnn": layers.SimpleRNN}[kind]


def build_model(kind: str, window: int, units, dropouts):
    from tensorflow import keras
    from tensorflow.keras import layers

    Rec = _layer(kind)
    model = keras.Sequential([keras.Input(shape=(window, 1))])
    for k, (u, d) in enumerate(zip(units, dropouts)):
        model.add(Rec(int(u), return_sequences=k < len(units) - 1))
        model.add(layers.Dropout(float(d)))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _fit(kind, params, window, Xtr, ytr, Xva, yva, epochs):
    from tensorflow import keras

    keras.backend.clear_session()
    units = (params["units_1"], params["units_2"], params["units_3"])
    drops = (params["dropout_1"], params["dropout_2"], params["dropout_3"])
    model = build_model(kind, window, units, drops)
    hist = model.fit(Xtr[..., None], ytr, validation_data=(Xva[..., None], yva),
                     epochs=epochs, batch_size=int(params["batch_size"]), verbose=0, shuffle=False)
    return model, float(hist.history["val_loss"][-1])


def _grid(space: dict):
    def pts(rng):  # continuous dropout range -> 3 grid points
        lo, hi = rng
        return sorted({lo, (lo + hi) / 2, hi})

    keys = ["units_1", "units_2", "units_3", "dropout_1", "dropout_2", "dropout_3", "batch_size"]
    vals = [space["units_1"], space["units_2"], space["units_3"],
            pts(space["dropout_1"]), pts(space["dropout_2"]), pts(space["dropout_3"]),
            space["batch_size"]]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def tune(kind, window, Xtr, ytr, Xva, yva, cfg_deep, seed):
    space = cfg_deep["tuning"]
    epochs = cfg_deep["epochs"]
    if space["method"] == "grid":
        best, best_loss = None, np.inf
        for params in _grid(space):
            _, loss = _fit(kind, params, window, Xtr, ytr, Xva, yva, epochs)
            if loss < best_loss:
                best, best_loss = params, loss
        return best

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "units_1": trial.suggest_categorical("units_1", space["units_1"]),
            "units_2": trial.suggest_categorical("units_2", space["units_2"]),
            "units_3": trial.suggest_categorical("units_3", space["units_3"]),
            "dropout_1": trial.suggest_float("dropout_1", *space["dropout_1"]),
            "dropout_2": trial.suggest_float("dropout_2", *space["dropout_2"]),
            "dropout_3": trial.suggest_float("dropout_3", *space["dropout_3"]),
            "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        }
        return _fit(kind, params, window, Xtr, ytr, Xva, yva, epochs)[1]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=space["n_trials"])
    return study.best_params


def forecast_asset(kind: str, series: pd.Series, cfg_fc: dict, seed: int):
    """Tune, train and produce val+test one-step predictions for one asset."""
    w = cfg_fc["window"]
    X, y, dates = make_windows(series, w)
    tr, va, te = split_masks(dates, cfg_fc["train_end"], cfg_fc["val_end"])
    if tr.sum() < 10 or va.sum() == 0:
        raise ValueError(f"{series.name}: not enough history")
    scaler = TrainMinMax().fit(np.concatenate([X[tr].ravel(), y[tr]]))
    Xs, ys = scaler.transform(X), scaler.transform(y)

    best = tune(kind, w, Xs[tr], ys[tr], Xs[va], ys[va], cfg_fc["deep"], seed)
    model, _ = _fit(kind, best, w, Xs[tr], ys[tr], Xs[va], ys[va], cfg_fc["deep"]["epochs"])
    out = va | te
    pred = scaler.inverse(model.predict(Xs[out][..., None], verbose=0).ravel())
    return pd.Series(pred, index=dates[out], name=series.name), best


def forecast(kind: str, prices: pd.DataFrame, cfg_fc: dict, seed: int):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    preds, params = {}, {}
    for t in prices.columns:
        log.info("%s  %s", kind.upper(), t)
        preds[t], params[t] = forecast_asset(kind, prices[t], cfg_fc, seed)
    return pd.DataFrame(preds), params
