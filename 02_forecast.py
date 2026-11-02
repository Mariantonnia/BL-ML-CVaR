"""Step 2 - One-week-ahead price forecasts with LSTM, GRU, RNN, RF and ARIMA.

Usage:
    python scripts/02_forecast.py --model lstm
    python scripts/02_forecast.py --model all

Outputs (results/predictions/): pred_<model>.csv (rows = target dates of the
validation + test periods) and params_<model>.json (tuned hyper-parameters).
"""
import json

from _common import read_prices, setup

from blmlcvar.forecasting import run_model


def args(ap):
    ap.add_argument("--model", default="all",
                    help="lstm | gru | rnn | rf | arima | all")
    ap.add_argument("--tickers", nargs="*", help="optional subset (for quick tests)")


def main():
    cfg, a = setup(__doc__, args)
    prices = read_prices(cfg, "local")
    if a.tickers:
        prices = prices[a.tickers]
    models = cfg["forecast"]["models"] if a.model == "all" else [a.model]
    out = cfg["paths"]["predictions"]
    for m in models:
        pred, params = run_model(m, prices, cfg)
        pred.index.name = "Date"
        pred.to_csv(out / f"pred_{m}.csv")
        with open(out / f"params_{m}.json", "w") as fh:
            json.dump(params, fh, indent=1, default=float)
        print(f"{m}: {pred.shape} -> {out / f'pred_{m}.csv'}")


if __name__ == "__main__":
    main()
