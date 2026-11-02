"""Configuration loading and global seeding."""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load ``config.yaml`` and turn every entry of ``paths`` into an absolute Path."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (ROOT / v) for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    for key in ("universe_file", "esg_file", "shares_file", "risk_free_file"):
        cfg["data"][key] = ROOT / cfg["data"][key]
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:  # only seed TF if it is in use (LSTM/GRU/RNN)
        sys.modules["tensorflow"].keras.utils.set_random_seed(seed)
