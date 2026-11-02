"""Benchmark portfolios of Sect. 5: MV, MV-ESG and naive (1/N)."""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def _solver(name: str):
    return name if name in cp.installed_solvers() else "CLARABEL"


def naive(assets) -> pd.Series:
    return pd.Series(1.0 / len(assets), index=assets)


def mean_variance(returns: pd.DataFrame, rf: float, max_weight: float,
                  objective: str = "max_sharpe", esg: pd.DataFrame | None = None,
                  esg_threshold: float | None = None, solver: str = "ECOS") -> pd.Series:
    """Markowitz portfolio with historical mean and sample covariance.

    max_sharpe uses the standard homogenisation y = kappa * w so that the
    problem stays convex with bounds and ESG constraints.
    """
    r = returns.dropna(how="any")
    mu = r.mean().to_numpy()
    cov = r.cov().to_numpy()
    cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(len(mu))
    n = len(mu)
    E = None if esg is None else esg.loc[r.columns].to_numpy()

    if objective == "max_sharpe" and np.any(mu - rf > 0):
        y = cp.Variable(n, nonneg=True)
        kappa = cp.Variable(nonneg=True)
        cons = [(mu - rf) @ y == 1, y <= max_weight * kappa, cp.sum(y) == kappa]
        if E is not None:
            cons += [E[:, k] @ y >= esg_threshold * kappa for k in range(E.shape[1])]
        prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))), cons)
        prob.solve(solver=_solver("CLARABEL"))
        if prob.status in ("optimal", "optimal_inaccurate") and kappa.value > 1e-12:
            w = np.clip(y.value / kappa.value, 0, None)
            return pd.Series(w / w.sum(), index=r.columns)
    # min-variance (also fallback when no asset beats the risk-free rate)
    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1, w <= max_weight]
    if E is not None:
        cons += [E[:, k] @ w >= esg_threshold for k in range(E.shape[1])]
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), cons).solve(solver=_solver("CLARABEL"))
    wv = np.clip(w.value, 0, None)
    return pd.Series(wv / wv.sum(), index=r.columns)
