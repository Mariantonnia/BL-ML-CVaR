"""Step 1 - Download weekly prices (Yahoo Finance) and FX rates.

Outputs (data/processed/):
    prices_local.csv   adjusted close in listing currency, Monday grid, ffill only
    prices_usd.csv     same prices converted to USD
    currencies.csv     listing currency per ticker  (CHECK the .IL tickers by hand)

Shares outstanding (data/raw/shares_outstanding.csv) and the ECB deposit rate
(data/raw/ecb_deposit_facility.csv) are not downloaded here; see data/README.md.
"""
from _common import setup

import pandas as pd

from blmlcvar import data as D


def main():
    cfg, _ = setup(__doc__)
    dc = cfg["data"]
    uni = D.load_universe(dc["universe_file"], only_included=False)
    tickers = uni["ticker"].tolist()

    raw = D.download_prices(tickers, dc["start"], dc["end"], dc["interval"], dc["price_field"])
    prices = D.clean_prices(raw, dc["max_ffill_weeks"])
    out = cfg["paths"]["processed"]
    prices.to_csv(out / "prices_local.csv")

    cur = D.download_currencies(tickers)
    cur.to_csv(out / "currencies.csv", header=True, index_label="ticker")
    unknown = cur[cur == "UNKNOWN"].index.tolist()
    if unknown:
        print(f"WARNING: currency unknown for {unknown}; edit currencies.csv before step 5.")
        return
    fx = D.download_fx_to_usd(cur.tolist(), dc["start"], dc["end"], dc["interval"])
    D.to_usd(prices, cur, fx).to_csv(out / "prices_usd.csv")
    print(f"Saved {prices.shape[1]} series x {prices.shape[0]} weeks to {out}")


if __name__ == "__main__":
    main()
