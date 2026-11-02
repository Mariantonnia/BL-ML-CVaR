"""GARCH(1,1) margins + multivariate Student-t copula (Sect. 3.3.4, Eqs. 11-16).

Steps for every rebalancing date
1. Fit a GARCH(1,1) with Student-t innovations to each asset's weekly returns
   (Eq. 14) and keep the standardised residuals z_jt and the one-step-ahead
   conditional volatility.
2. Transform residuals to pseudo-observations and fit a t-copula (Eq. 15):
   the shape matrix R from Kendall's tau (R = sin(pi*tau/2)) and the degrees of
   freedom nu by maximum likelihood.
3. Simulate S scenarios of residuals from the copula, map them back with the
   empirical marginal quantiles, rescale with the GARCH forecast volatility.
4. CBL step (Sahamkhadam et al., 2022; Eq. 16): whiten the copula scenarios with
   their own covariance and re-colour them with the BL posterior,
       r_CBL = mu_CBL + Sigma_CBL^{1/2} eta.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- margins
def fit_garch(returns: pd.DataFrame, p: int = 1, q: int = 1, dist: str = "t"):
    """Return standardised residuals (T x n), one-step vol (n,) and mean (n,)."""
    from arch import arch_model

    z, vol, mu = {}, {}, {}
    for c in returns.columns:
        y = 100 * returns[c].dropna()  # rescale: arch recommends values in [1, 1000]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y, mean="Constant", vol="GARCH", p=p, q=q, dist=dist).fit(disp="off")
        z[c] = res.std_resid
        vol[c] = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])) / 100
        mu[c] = float(res.params["mu"]) / 100
    z = pd.DataFrame(z).dropna()
    return z, pd.Series(vol), pd.Series(mu)


# ----------------------------------------------------------------------------- copula
def _nearest_corr(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (R + R.T))
    R = vecs @ np.diag(np.clip(vals, eps, None)) @ vecs.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def pseudo_obs(z: pd.DataFrame) -> np.ndarray:
    return z.rank().to_numpy() / (len(z) + 1)


def t_copula_loglik(u: np.ndarray, R: np.ndarray, nu: float) -> float:
    x = stats.t.ppf(u, df=nu)
    n = R.shape[0]
    mvt = stats.multivariate_t(loc=np.zeros(n), shape=R, df=nu)
    return float(np.sum(mvt.logpdf(x)) - np.sum(stats.t.logpdf(x, df=nu)))


def fit_t_copula(z: pd.DataFrame, nu_bounds=(2.05, 50.0)):
    """Fit (R, nu): R via Kendall's tau inversion, nu via profile ML."""
    u = pseudo_obs(z)
    tau = z.corr(method="kendall").to_numpy()
    R = _nearest_corr(np.sin(np.pi * tau / 2))
    res = optimize.minimize_scalar(lambda nu: -t_copula_loglik(u, R, nu),
                                   bounds=nu_bounds, method="bounded",
                                   options={"xatol": 1e-2})
    return R, float(res.x)


def simulate_t_copula(R: np.ndarray, nu: float, n_sim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform draws (n_sim x n) from a t-copula."""
    n = R.shape[0]
    L = np.linalg.cholesky(R)
    g = rng.standard_normal((n_sim, n)) @ L.T
    w = rng.chisquare(nu, size=(n_sim, 1)) / nu
    x = g / np.sqrt(w)
    return stats.t.cdf(x, df=nu)


def empirical_quantile(z: pd.DataFrame, u: np.ndarray) -> np.ndarray:
    zs = np.sort(z.to_numpy(), axis=0)
    out = np.empty_like(u)
    grid = (np.arange(1, len(zs) + 1)) / (len(zs) + 1)
    for j in range(u.shape[1]):
        out[:, j] = np.interp(u[:, j], grid, zs[:, j])
    return out


def copula_scenarios(returns: pd.DataFrame, n_sim: int, rng: np.random.Generator,
                     garch_cfg: dict | None = None):
    """Simulated next-week return scenarios (n_sim x n) and fitted nu."""
    g = garch_cfg or {}
    z, vol, mu = fit_garch(returns, g.get("p", 1), g.get("q", 1), g.get("dist", "t"))
    R, nu = fit_t_copula(z)
    u = simulate_t_copula(R, nu, n_sim, rng)
    eta = empirical_quantile(z, u)
    eta = (eta - eta.mean(0)) / eta.std(0)
    scen = mu.to_numpy() + eta * vol.to_numpy()
    return pd.DataFrame(scen, columns=returns.columns), nu


# ----------------------------------------------------------------------------- CBL
def _sqrtm_psd(S: np.ndarray, inverse: bool = False) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
    vals = np.clip(vals, 1e-12, None)
    p = -0.5 if inverse else 0.5
    return vecs @ np.diag(vals ** p) @ vecs.T


def cbl_scenarios(scen: np.ndarray, mu_bl: np.ndarray, sigma_bl: np.ndarray) -> np.ndarray:
    """r_CBL = mu_BL + Sigma_BL^{1/2} eta, eta = whitened copula scenarios (Eq. 16)."""
    mu_c = scen.mean(0)
    eta = (scen - mu_c) @ _sqrtm_psd(np.cov(scen, rowvar=False), inverse=True)
    return mu_bl + eta @ _sqrtm_psd(sigma_bl)
