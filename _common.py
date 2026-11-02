"""Helpers shared by the pipeline scripts."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blmlcvar.config import load_config, set_seed  # noqa: E402


def setup(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    if extra:
        extra(ap)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    return cfg, args


def read_prices(cfg, kind: str = "local"):
    import pandas as pd

    from blmlcvar.data import load_universe

    p = pd.read_csv(cfg["paths"]["processed"] / f"prices_{kind}.csv", index_col="Date", parse_dates=True)
    keep = load_universe(cfg["data"]["universe_file"])["ticker"].tolist()
    missing = sorted(set(keep) - set(p.columns))
    if missing:
        raise KeyError(f"Tickers missing from prices_{kind}.csv: {missing}")
    return p[sorted(keep)]
