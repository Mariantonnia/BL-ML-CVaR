"""Weekly out-of-sample back-test of all strategies (Sect. 4.3 and 5).

Timing convention
-----------------
Decision date ``t`` (Monday close) -> forecasts for ``t+1`` are already
available (they only use prices up to ``t``) -> weights chosen at ``t`` ->
realised return from ``t`` to ``t+1``. Nothing dated after ``t`` is used to
build the portfolio held during (t, t+1].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import benchmarks
from . import black_litterman as bl
from .copula import cbl_scenarios, copula_scenarios
from .optimization import solve_bl_cvar

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    returns: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None


def strategy_name(kind: str, th: float | None = None) -> str:
    return kind if th is None else f"{kind}@{th:.2f}"


def run(prices: pd.DataFrame, local_prices: pd.DataFrame, caps_usd: pd.DataFrame,
        agg_pred: pd.DataFrame, esg_pt: pd.DataFrame, rf_weekly: pd.Series,
        cfg: dict, test_dates: pd.DatetimeIndex, thresholds: list[float] | None = None,
        benchmarks_on: bool = True) -> BacktestResult:
    """Run the back-test.

    prices       prices in the return currency (local or USD)
    local_prices prices in listing currency (the space in which forecasts live)
    caps_usd     market capitalisation in USD
    agg_pred     aggregated forecasts indexed by TARGET date (val + test)
    esg_pt       PT-transformed E, S, G pillar scores (rows = assets)
    test_dates   target dates of the out-of-sample period
    """
    pc = cfg["portfolio"]
    thresholds = thresholds or [pc["cvar_target"]]
    th_esg = float(esg_pt.attrs.get("threshold_pt", np.nan))
    rng = np.random.default_rng(cfg["seed"])
    rets = prices.pct_change(fill_method=None)
    assets = list(prices.columns)

    W: dict[str, list] = {}
    realised, diag = [], []
    for target in test_dates:
        t = prices.index[prices.index.get_loc(target) - 1]          # decision date
        hist = rets.loc[:t].iloc[1:]
        hist = hist.dropna(axis=0, how="any")

        # --- Step 1: market equilibrium (Eqs. 5-6)
        cap_t = caps_usd.loc[t, assets]
        w_mkt = (cap_t / cap_t.sum()).to_numpy()
        caps_lag = caps_usd[assets].shift(1).loc[hist.index]
        mkt = (hist * caps_lag.div(caps_lag.sum(1), axis=0)).sum(1)
        lam = bl.market_risk_aversion(mkt, rf_weekly)
        cov_hist = hist.cov().to_numpy()
        pi = bl.implied_returns(cov_hist, w_mkt, lam)

        # --- Step 2: ML views and confidence (Eq. 7)
        Q = bl.views_from_forecast(agg_pred.loc[target, assets], local_prices.loc[t, assets])
        known = agg_pred.index[agg_pred.index <= t]
        omega = bl.omega_from_rmae(agg_pred.loc[known, assets], local_prices.loc[known, assets],
                                   squared=pc["omega"] == "rmae_squared", floor=pc["omega_floor"])

        # --- Step 4 (prior covariance from the copula) and Step 3 (posterior)
        scen, nu = copula_scenarios(hist, pc["copula"]["n_simulations"], rng, pc["copula"]["garch"])
        cov_c = np.cov(scen.to_numpy(), rowvar=False)
        mu_bl, sigma_bl = bl.posterior(cov_c, pi, Q.to_numpy(), omega, pc["tau"])
        r_cbl = cbl_scenarios(scen.to_numpy(), mu_bl, sigma_bl)
        mu_s = pd.Series(mu_bl, index=assets)

        row = {"date": t, "target": target, "lambda_m": lam, "nu": nu}
        # --- Step 5: epsilon-constraint model (Eqs. 17-20)
        for th in thresholds:
            for kind, esg in (("BL-ML-CVaR", None), ("BL-ML-CVaR-ESG", esg_pt)):
                sol = solve_bl_cvar(mu_s, r_cbl, pc["cvar_alpha"], th, pc["max_weight"],
                                    esg, th_esg, pc["solver"])
                name = strategy_name(kind, th)
                W.setdefault(name, []).append(sol.weights.rename(t))
                row[f"{name}|fallback"] = sol.fallback
                row[f"{name}|exp_ret"] = sol.expected_return
                row[f"{name}|cvar_in_sample"] = sol.cvar

        if benchmarks_on:
            rf_t = float(rf_weekly.get(t, 0.0))
            W.setdefault("MV", []).append(benchmarks.mean_variance(
                hist, rf_t, pc["max_weight"], cfg["benchmarks"]["mv_objective"]).rename(t))
            W.setdefault("MV-ESG", []).append(benchmarks.mean_variance(
                hist, rf_t, pc["max_weight"], cfg["benchmarks"]["mv_objective"],
                esg_pt, th_esg).rename(t))
            W.setdefault("Naive", []).append(benchmarks.naive(assets).rename(t))

        realised.append(rets.loc[target, assets].fillna(0.0).rename(target))
        diag.append(row)
        log.info("%s  nu=%.2f  lambda=%.3f", t.date(), nu, lam)

    res = BacktestResult()
    res.weights = {k: pd.DataFrame(v) for k, v in W.items()}
    R = pd.DataFrame(realised)
    res.returns = pd.DataFrame({k: (w.to_numpy() * R.to_numpy()).sum(1) for k, w in res.weights.items()},
                               index=R.index)
    res.diagnostics = pd.DataFrame(diag).set_index("date")
    res.asset_returns = R
    return res


def buy_and_hold(first_weights: pd.Series, asset_returns: pd.DataFrame) -> pd.Series:
    """Portfolio return path when the first-week weights are never rebalanced (Fig. 6)."""
    growth = (1 + asset_returns[first_weights.index]).cumprod()
    value = (growth * first_weights).sum(1)
    return value.pct_change().fillna(value.iloc[0] - 1)


def performance(r: pd.Series, alpha: float = 0.95, periods: int = 52) -> pd.Series:
    """Performance metrics used in Table 5 (all explicitly defined)."""
    r = r.dropna()
    losses = -r
    var = losses.quantile(alpha)
    return pd.Series({
        "Cumulative return": (1 + r).prod() - 1,
        "Annualised mean return": r.mean() * periods,
        "Annualised volatility": r.std() * np.sqrt(periods),
        "Annualised variance": r.var() * periods,
        f"Weekly hist. CVaR{int(alpha * 100)}": losses[losses >= var].mean(),
    })
