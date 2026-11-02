"""Prospect-Theory transformation of ESG scores (Sect. 3.2, Eq. 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pt_value(score, reference_point: float = 50.0, kappa: float = 0.8, theta: float = 2.0):
    """Value function of Eq. (4).

    v(sc) =  ((sc - rp) / rp) ** kappa            if sc >= rp
    v(sc) = -theta * ((rp - sc) / rp) ** kappa    if sc <  rp
    """
    sc = np.asarray(score, dtype=float)
    gain = np.clip(sc - reference_point, 0, None) / reference_point
    loss = np.clip(reference_point - sc, 0, None) / reference_point
    out = np.where(sc >= reference_point, gain ** kappa, -theta * loss ** kappa)
    out = np.where(np.isnan(sc), np.nan, out)
    return out if out.ndim else float(out)


def transform_esg(esg: pd.DataFrame, columns: list[str], **pt_kwargs) -> pd.DataFrame:
    """Apply :func:`pt_value` column-wise and return a new DataFrame."""
    out = esg.copy()
    for c in columns:
        out[c] = pt_value(pd.to_numeric(esg[c], errors="coerce"), **pt_kwargs)
    return out
