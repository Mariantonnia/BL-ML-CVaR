"""Create a synthetic dataset with the same layout as the real one.

Lets anyone run the full pipeline without Yahoo Finance access or a LSEG
licence. Results obtained with this data are meaningless.

    python scripts/make_synthetic_data.py --n-assets 8
"""
import numpy as np
import pandas as pd
from _common import setup

from blmlcvar.data import load_universe


def args(ap):
    ap.add_argument("--n-assets", type=int, default=None, help="use only the first N tickers")


def main():
    cfg, a = setup(__doc__, args)
    rng = np.random.default_rng(cfg["seed"])
    tick = load_universe(cfg["data"]["universe_file"])["ticker"].tolist()
    if a.n_assets:
        tick = tick[: a.n_assets]
        uni = pd.read_csv(cfg["data"]["universe_file"])
        uni.loc[~uni["ticker"].isin(tick), "included"] = 0
        uni.to_csv(cfg["data"]["universe_file"], index=False)
    dates = pd.date_range(cfg["data"]["start"], cfg["data"]["end"], freq="W-MON", name="Date")
    n, T = len(tick), len(dates)

    # factor model with fat tails
    beta = rng.uniform(0.5, 1.5, n)
    f = 0.002 + 0.025 * rng.standard_t(5, T)
    e = 0.03 * rng.standard_t(4, (T, n))
    r = f[:, None] * beta + e
    prices = pd.DataFrame(50 * np.exp(np.cumsum(np.log1p(r), 0)), index=dates, columns=tick)
    out = cfg["paths"]["processed"]
    prices.to_csv(out / "prices_local.csv")
    prices.to_csv(out / "prices_usd.csv")
    pd.Series("USD", index=tick, name="currency").to_csv(out / "currencies.csv", index_label="ticker")

    raw = cfg["paths"]["raw"]
    pd.DataFrame({"ticker": tick, "shares": rng.uniform(1e7, 1e9, n).round()}).to_csv(
        raw / "shares_outstanding.csv", index=False)
    esg = pd.DataFrame({"asset": tick})
    for c in [cfg["esg"]["composite"], *cfg["esg"]["pillars"].values()]:
        esg[c] = rng.uniform(25, 95, n).round(2)
    esg.to_csv(raw / "esg_scores.csv", index=False)
    pd.DataFrame({"date": ["18/09/2019", "27/07/2022", "14/09/2022", "02/11/2022",
                           "21/12/2022", "08/02/2023", "22/03/2023", "10/05/2023",
                           "21/06/2023", "20/09/2023", "12/06/2024"],
                  "deposit": ["-0,50", "0,00", "0,75", "1,50", "2,00", "2,50", "3,00",
                              "3,25", "3,50", "4,00", "3,75"]}).to_csv(
        raw / "ecb_deposit_facility.csv", sep=";", index=False)
    print(f"Synthetic data: {n} assets x {T} weeks")


if __name__ == "__main__":
    main()
