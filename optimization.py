"""Multi-objective BL-ML-CVaR(-ESG) model solved by epsilon-constraints
(Sect. 3.3.5, Eqs. 17-20).

    max  mu' w
    s.t. xi + 1/(S(1-alpha)) sum_s v_s <= th_CVaR        (CVaR, Rockafellar-Uryasev)
         v_s >= -r_s' w - xi,  v_s >= 0
         pts_E' w >= th_E,  pts_S' w >= th_S,  pts_G' w >= th_G
         sum w = 1,  0 <= w_i <= M_i
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class Solution:
    weights: pd.Series
    status: str
    expected_return: float
    cvar: float
    fallback: bool = False


def _solver(name: str):
    return name if name in cp.installed_solvers() else "CLARABEL"


def scenario_cvar(w: np.ndarray, scen: np.ndarray, alpha: float) -> float:
    losses = -scen @ w
    var = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())


def solve_bl_cvar(mu: pd.Series, scenarios: np.ndarray, alpha: float, cvar_target: float,
                  max_weight: float, esg: pd.DataFrame | None = None,
                  esg_threshold: float | None = None, solver: str = "ECOS") -> Solution:
    """Solve Eq. (19). ``esg`` holds PT-transformed pillar scores (rows = assets).

    If the problem is infeasible (CVaR target too tight given the ESG
    constraints) the minimum-CVaR portfolio satisfying the remaining
    constraints is returned and flagged with ``fallback=True``.
    """
    n = len(mu)
    if n * max_weight < 1 - 1e-12:
        raise ValueError(f"{n} assets with max weight {max_weight} cannot sum to 1.")
    S = scenarios.shape[0]
    w = cp.Variable(n, nonneg=True)
    xi = cp.Variable()
    v = cp.Variable(S, nonneg=True)
    cvar = xi + cp.sum(v) / (S * (1 - alpha))
    base = [v >= -scenarios @ w - xi, cp.sum(w) == 1, w <= max_weight]
    if esg is not None:
        E = esg.loc[mu.index].to_numpy()
        base += [E[:, k] @ w >= esg_threshold for k in range(E.shape[1])]

    prob = cp.Problem(cp.Maximize(mu.to_numpy() @ w), base + [cvar <= cvar_target])
    prob.solve(solver=_solver(solver))
    fallback = False
    if prob.status not in ("optimal", "optimal_inaccurate"):
        fallback = True
        prob = cp.Problem(cp.Minimize(cvar), base)
        prob.solve(solver=_solver(solver))
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Optimisation failed ({prob.status}): the ESG thresholds and weight bounds cannot be met simultaneously.")
    wv = np.clip(w.value, 0, None)
    wv = wv / wv.sum()
    return Solution(pd.Series(wv, index=mu.index), prob.status, float(mu.to_numpy() @ wv),
                    scenario_cvar(wv, scenarios, alpha), fallback)
