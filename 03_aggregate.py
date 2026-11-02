"""Step 3 - Accuracy-weighted aggregation of the forecasts (Eqs. 1-3).

Outputs:
    results/predictions/pred_aggregated.csv
    results/predictions/weights_<model>.csv
    results/tables/table2_forecast_metrics.csv   (test period)
    results/tables/tableA2_best_technique_per_week.csv
"""
import pandas as pd
from _common import read_prices, setup

from blmlcvar.aggregation import aggregate, winners_per_week
from blmlcvar.data import split_dates
from blmlcvar.forecasting.common import forecast_metrics

LABELS = {"lstm": "LSTM", "rf": "RF", "rnn": "RNN", "gru": "GRU", "arima": "ARIMA"}


def main():
    cfg, _ = setup(__doc__)
    paths = cfg["paths"]
    actual = read_prices(cfg, "local")
    preds = {m: pd.read_csv(paths["predictions"] / f"pred_{m}.csv", index_col="Date", parse_dates=True)
             for m in cfg["forecast"]["models"]}

    agg, weights = aggregate(preds, actual)
    agg.index.name = "Date"
    agg.to_csv(paths["predictions"] / "pred_aggregated.csv")
    for m, w in weights.items():
        w.to_csv(paths["predictions"] / f"weights_{m}.csv")

    _, _, test = split_dates(agg.index, cfg["forecast"]["train_end"], cfg["forecast"]["val_end"])
    tdates = agg.index[test]
    rows = {LABELS.get(m, m): forecast_metrics(actual.loc[tdates], p.loc[p.index.intersection(tdates)])
            for m, p in preds.items()}
    rows["Aggregated"] = forecast_metrics(actual.loc[tdates], agg.loc[tdates])
    table2 = pd.DataFrame(rows).T.round(2)
    table2.to_csv(paths["tables"] / "table2_forecast_metrics.csv")
    print(table2)

    win = winners_per_week({LABELS.get(m, m): p for m, p in preds.items()}, actual.loc[tdates])
    win.to_csv(paths["tables"] / "tableA2_best_technique_per_week.csv")
    print("\nMinimum number of assets won per week:\n", win.min())


if __name__ == "__main__":
    main()
