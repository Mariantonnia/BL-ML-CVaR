"""Black-Litterman with machine-learning views (Sect. 3.3.1-3.3.3, Eqs. 5-10).

All quantities are in WEEKLY units (returns, covariances, risk-free rate).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def market_risk_aversion(market_returns: pd.Series, rf: float | pd.Series = 0.0) -> float:
    """lambda_m = (E[r_m] - r_f) / sigma_m^2  (Eq. 6), from the cap-weighted market."""
    excess = market_returns - (rf if np.isscalar(rf) else rf.reindex(market_returns.index).fillna(0))
    excess = excess.dropna()
    lam = excess.mean() / excess.var()
    if not np.isfinite(lam) or lam <= 0:
        log.warning("Non-positive market risk aversion (%.4f); prior returns set to 0.", lam)
        lam = 0.0
    return float(lam)


def implied_returns(cov: np.ndarray, w_mkt: np.ndarray, lam: float) -> np.ndarray:
    """Pi = lambda_m * Sigma * w_m  (Eq. 5)."""
    return lam * cov @ w_mkt


def views_from_forecast(pred_price: pd.Series, last_price: pd.Series) -> pd.Series:
    """Q: expected next-week return implied by the aggregated price forecast."""
    return (pred_price - last_price) / last_price


def omega_from_rmae(agg_pred_hist: pd.DataFrame, actual_hist: pd.DataFrame,
                    squared: bool = False, floor: float = 1e-6) -> np.ndarray:
    """Diagonal Omega from the relative MAE of past forecasts (Eq. 7).

    RMAE_i = (1/n) sum_t |Y_it - Yhat_it| / mean_t(Y_it)

    Only forecast errors observed up to the decision date must be passed in.
    """
    a, p = actual_hist.align(agg_pred_hist, join="inner")
    rmae = (p - a).abs().mean() / a.mean()
    rmae = rmae.fillna(rmae.median()).clip(lower=floor)
    diag = rmae.to_numpy() ** (2 if squared else 1)
    return np.diag(diag)


def posterior(cov: np.ndarray, pi: np.ndarray, Q: np.ndarray, omega: np.ndarray,
              tau: float, P: np.ndarray | None = None):
    """Black-Litterman posterior mean and covariance (Eqs. 8-10).

    M        = [(tau Sigma)^-1 + P' Omega^-1 P]^-1                 (Eq. 9)
    mu_BL    = M [(tau Sigma)^-1 Pi + P' Omega^-1 Q]               (Eq. 8)
    Sigma_BL = Sigma + M                                           (Eq. 10)
    """
    n = cov.shape[0]
    P = np.eye(n) if P is None else P
    ts_inv = np.linalg.inv(tau * cov)
    om_inv = np.linalg.inv(omega)
    M = np.linalg.inv(ts_inv + P.T @ om_inv @ P)
    mu = M @ (ts_inv @ pi + P.T @ om_inv @ Q)
    sigma_bl = cov + M
    return mu, 0.5 * (sigma_bl + sigma_bl.T)
