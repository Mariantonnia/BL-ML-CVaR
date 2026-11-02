"""Step 4 - Prospect-Theory transformation of LSEG ESG scores (Eq. 4).

Input : data/raw/esg_scores.csv  (columns: asset, ESG Score, Environmental
        Pillar Score, Social Pillar Score, Governance Pillar Score; 0-100)
Output: data/processed/esg_pt.csv, results/tables/table3_esg_descriptives.csv,
        results/figures/fig2_value_function.png, fig3_esg_raw.png, fig4_esg_pt.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import setup

from blmlcvar.data import load_universe
from blmlcvar.prospect import pt_value, transform_esg


def hist4(df, cols, titles, path, xlabel):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, c, t in zip(axes.ravel(), cols, titles):
        ax.hist(df[c].dropna(), bins=15, color="0.55", edgecolor="black")
        ax.set_title(t)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    cfg, _ = setup(__doc__)
    ec, pt = cfg["esg"], cfg["esg"]["prospect_theory"]
    paths = cfg["paths"]
    tickers = load_universe(cfg["data"]["universe_file"])["ticker"]

    raw = pd.read_csv(cfg["data"]["esg_file"], sep=None, engine="python")
    raw = raw.set_index("asset").reindex(tickers)
    cols = [ec["composite"], *ec["pillars"].values()]
    for c in cols:
        raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", "."), errors="coerce")
    if raw[cols].isna().any().any():
        raise ValueError(f"Missing ESG scores for: {raw[raw[cols].isna().any(axis=1)].index.tolist()}")

    kw = dict(reference_point=pt["reference_point"], kappa=pt["kappa"], theta=pt["theta"])
    esg_pt = transform_esg(raw[cols], cols, **kw)
    esg_pt.index.name = "asset"
    esg_pt.to_csv(paths["processed"] / "esg_pt.csv")

    thr = pt_value(cfg["portfolio"]["esg_score_threshold"], **kw)
    print(f"ESG threshold {cfg['portfolio']['esg_score_threshold']} -> v(sc) = {thr:.4f}")

    desc = pd.DataFrame({c: {"Q1": raw[c].quantile(.25), "Median": raw[c].median(),
                             "Q3": raw[c].quantile(.75), "Mean": raw[c].mean()} for c in cols}).T.round(2)
    desc.to_csv(paths["tables"] / "table3_esg_descriptives.csv")
    print(desc)
    print("Mean PT-transformed scores:\n", esg_pt.mean().round(2))

    titles = ["(a) ESG Score", "(b) Environmental Pillar", "(c) Social Pillar", "(d) Governance Pillar"]
    hist4(raw, cols, titles, paths["figures"] / "fig3_esg_raw.png", "Score")
    hist4(esg_pt, cols, titles, paths["figures"] / "fig4_esg_pt.png", "v(sc)")

    x = np.linspace(0, 100, 501)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, pt_value(x, **kw), color="black", lw=2)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(pt["reference_point"], color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("ESG score (sc)")
    ax.set_ylabel("v(sc)")
    ax.set_title(f"rp={pt['reference_point']}, κ={pt['kappa']}, θ={pt['theta']}")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig2_value_function.png", dpi=300)


if __name__ == "__main__":
    main()
