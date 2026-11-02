"""Step 5 - Weekly BL-ML-CVaR / BL-ML-CVaR-ESG portfolios and benchmarks.

Runs every CVaR threshold of the sensitivity analysis (Table 4) in the same
loop, so the copula scenarios are simulated once per week.

Outputs (results/portfolios/): weights_<strategy>.csv, returns.csv,
asset_returns.csv, diagnostics.csv
"""
import pandas as pd
from _common import read_prices, setup

from blmlcvar import backtest
from blmlcvar.data import load_risk_free, market_caps, split_dates
from blmlcvar.prospect import pt_value


def args(ap):
    ap.add_argument("--only-target", action="store_true",
                    help="run only th_CVaR = portfolio.cvar_target (skip Table 4 sweep)")
    ap.add_argument("--weeks", type=int, help="limit number of test weeks (quick test)")


def main():
    cfg, a = setup(__doc__, args)
    paths, pc, dc = cfg["paths"], cfg["portfolio"], cfg["data"]

    local = read_prices(cfg, "local")
    usd = read_prices(cfg, "usd")
    prices = usd if dc["return_currency"].upper() == "USD" else local
    shares = pd.read_csv(dc["shares_file"], sep=None, engine="python").set_index("ticker")["shares"]
    caps = market_caps(usd, shares.astype(float))
    rf = load_risk_free(dc["risk_free_file"], prices.index)

    agg = pd.read_csv(paths["predictions"] / "pred_aggregated.csv", index_col="Date", parse_dates=True)
    esg_pt = pd.read_csv(paths["processed"] / "esg_pt.csv", index_col="asset")
    esg_pt = esg_pt[list(cfg["esg"]["pillars"].values())]
    pt = cfg["esg"]["prospect_theory"]
    esg_pt.attrs["threshold_pt"] = float(pt_value(pc["esg_score_threshold"], pt["reference_point"],
                                                  pt["kappa"], pt["theta"]))

    _, _, test = split_dates(agg.index, cfg["forecast"]["train_end"], cfg["forecast"]["val_end"])
    test_dates = agg.index[test]
    if a.weeks:
        test_dates = test_dates[: a.weeks]
    ths = [pc["cvar_target"]] if a.only_target else sorted(set(pc["cvar_sensitivity"]) | {pc["cvar_target"]})

    res = backtest.run(prices, local, caps, agg, esg_pt, rf, cfg, test_dates, ths)

    out = paths["portfolios"]
    for name, w in res.weights.items():
        w.index.name = "Date"
        w.to_csv(out / f"weights_{name.replace('@', '_th')}.csv")
    res.returns.to_csv(out / "returns.csv", index_label="Date")
    res.asset_returns.to_csv(out / "asset_returns.csv", index_label="Date")
    res.diagnostics.to_csv(out / "diagnostics.csv")
    fb = res.diagnostics.filter(like="|fallback").sum()
    if fb.any():
        print("Weeks solved with min-CVaR fallback (target infeasible):\n", fb[fb > 0])
    print(f"nu (t-copula d.o.f.) range: {res.diagnostics['nu'].min():.2f} - {res.diagnostics['nu'].max():.2f}")


if __name__ == "__main__":
    main()
