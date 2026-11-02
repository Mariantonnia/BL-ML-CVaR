"""Step 6 - Tables and figures of Sect. 4.3 and 5.

Outputs (results/):
    tables/table4_cvar_sensitivity.csv
    tables/table5_portfolio_metrics.csv
    tables/table6_composition_esg.csv, table7_composition_noesg.csv
    tables/tableA1_descriptive_returns.csv
    figures/fig5_cumulative_returns.png, fig6_buy_and_hold.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from _common import read_prices, setup

from blmlcvar.backtest import buy_and_hold, performance


def main():
    cfg, _ = setup(__doc__)
    paths, pc = cfg["paths"], cfg["portfolio"]
    port = paths["portfolios"]
    R = pd.read_csv(port / "returns.csv", index_col="Date", parse_dates=True)
    AR = pd.read_csv(port / "asset_returns.csv", index_col="Date", parse_dates=True)
    th = pc["cvar_target"]
    main_cols = {f"BL-ML-CVaR@{th:.2f}": "BL-ML-CVaR", f"BL-ML-CVaR-ESG@{th:.2f}": "BL-ML-CVaR-ESG",
                 "MV": "MV", "MV-ESG": "MV-ESG", "Naive": "Naive"}

    # Table 4 -- sensitivity to th_CVaR (cumulative realised return)
    rows = {}
    for c in R.columns:
        if "@" in c:
            kind, t = c.split("@")
            rows.setdefault(float(t), {})[f"{kind} return"] = (1 + R[c]).prod() - 1
    pd.DataFrame(rows).T.sort_index().rename_axis("th_CVaR").to_csv(paths["tables"] / "table4_cvar_sensitivity.csv")

    # Table 5 -- performance
    t5 = pd.DataFrame({v: performance(R[k], pc["cvar_alpha"]) for k, v in main_cols.items() if k in R}).T
    t5.to_csv(paths["tables"] / "table5_portfolio_metrics.csv")
    print((t5 * 100).round(2))

    # Fig. 5 -- cumulative returns
    styles = ["-o", "-s", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(10, 6))
    for (k, v), st in zip(main_cols.items(), styles):
        if k in R:
            ax.plot((1 + R[k]).cumprod() - 1, st, label=v, lw=1.8, ms=3)
    ax.axhline(0, color="grey", ls="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(color="lightgrey", ls="--")
    ax.legend(loc="upper left", fontsize="small")
    ax.set_ylabel("Cumulative return")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig5_cumulative_returns.png", dpi=300)
    plt.close(fig)

    # Fig. 6 -- first portfolios without rebalancing
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in list(main_cols)[:2]:
        w = pd.read_csv(port / f"weights_{k.replace('@', '_th')}.csv", index_col="Date").iloc[0]
        bh = buy_and_hold(w, AR)
        ax.plot((1 + bh).cumprod() - 1, label=main_cols[k], lw=2)
        print(f"Buy-and-hold {main_cols[k]}: {(1 + bh).prod() - 1:.2%}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(color="lightgrey", ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig6_buy_and_hold.png", dpi=300)
    plt.close(fig)

    # Tables 6-7 -- composition in the last week
    raw = pd.read_csv(cfg["data"]["esg_file"], sep=None, engine="python").set_index("asset")
    cols = [cfg["esg"]["composite"], *cfg["esg"]["pillars"].values()]
    for k, name in zip(list(main_cols)[:2], ["table7_composition_noesg", "table6_composition_esg"]):
        w = pd.read_csv(port / f"weights_{k.replace('@', '_th')}.csv", index_col="Date", parse_dates=True)
        last = w.iloc[-1]
        comp = raw.reindex(last[last > 1e-3].index)[cols].assign(Weight=last[last > 1e-3].round(2))
        comp.to_csv(paths["tables"] / f"{name}.csv", index_label=f"Asset ({w.index[-1].date()})")

    # Table A1 -- descriptive statistics of weekly returns (full sample)
    rets = read_prices(cfg, cfg["data"]["return_currency"].lower()).pct_change(fill_method=None)
    rets.describe().T.to_csv(paths["tables"] / "tableA1_descriptive_returns.csv")


if __name__ == "__main__":
    main()
