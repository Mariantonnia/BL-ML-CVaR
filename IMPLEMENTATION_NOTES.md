# Implementation notes

This file records the implementation choices made where the article leaves details open, and the points where this code differs from the preliminary notebooks used during the research. Authors: review it before making the repository public.

## 1. Data
* Prices are downloaded with explicit `start`/`end` dates and `auto_adjust=True`, not `period="6y"`, so the download can be reproduced.
* Missing prices are forward-filled for at most 4 weeks. The notebooks also back-filled them, which copies future prices into the past.
* Market capitalisations are always converted to USD. Without that conversion, caps quoted in KRW, INR, JPY, etc. cannot be compared. Returns stay in each stock's local currency by default, as in the paper (`data.return_currency`).

## 2. Forecasting (Sect. 4.1)
* All models use the same input window of 30 weeks. The preliminary LSTM and RNN notebooks used 12.
* The split is chronological by target date: train ≤ 2022-12-26, validation ≤ 2023-06-26, test 2023-07-03 → 2024-07-01. Hyper-parameters are tuned on the validation split only. The test weeks are never used for tuning.
* The min–max scaler is fitted on the training split only.
* LSTM, GRU and RNN share one architecture (3 recurrent layers + Dense). Each layer has its own dropout parameter, and the model that is tuned is exactly the model that is trained.
* Hyper-parameter search uses Optuna (TPE, 25 trials, seeded) over the search space of Sect. 4.1, which is what the authors ran. An exhaustive grid is available with `tuning.method: grid`.
* Random Forest tuning uses `TimeSeriesSplit`, so no fold trains on the future.
* ARIMA is re-estimated every week on the last 30 prices, with the order selected by AIC.
* Models are trained once and then produce one-step-ahead forecasts. Each forecast only uses prices up to the previous week.

## 3. Aggregation (Eqs. 1–3)
* Each week, all five techniques get the weights computed from the previous week's absolute errors. The weights sum to 1. In the first week the weights are equal.

## 4. Black–Litterman (Eqs. 5–10)
* All quantities are weekly.
* λ_m comes from the cap-weighted market portfolio of the 55 assets (Eq. 6).
* P = I. Q is the expected weekly return implied by the aggregated price forecast.
* Ω = diag(RMAE_i) (Eq. 7). RMAE_i only uses forecast errors known at the decision date. `portfolio.omega: rmae_squared` gives Ω in variance units instead.
* The posterior follows Eqs. 8–10 with the identity pick matrix.
* τ = 0.25. The paper does not report τ; this is the value used in the notebooks.

## 5. Copula and scenarios (Eqs. 11–16)
* GARCH(1,1) margins with Student-t innovations are fitted on returns × 100.
* A multivariate t-copula is fitted to the pseudo-observations. R comes from Kendall's τ, and ν is estimated by maximum likelihood.
* 10 000 scenarios are simulated. They are whitened with their own covariance and re-coloured with the BL posterior (Eq. 16).
* The prior covariance used in the posterior (Eqs. 8–10) is the covariance of the copula scenarios. The equilibrium returns of Eq. 5 use the historical covariance.

## 6. Optimisation (Eqs. 17–20)
* The model is a Rockafellar–Uryasev linear program solved in cvxpy (ECOS). The constraints are Σw = 1, 0 ≤ w ≤ 0.10, and the three PT-transformed pillar constraints ≥ v(65) = 0.38.
* If the CVaR target cannot be met in a given week, the minimum-CVaR portfolio is used for that week and recorded in `diagnostics.csv`.

## 7. Benchmarks and metrics (Sect. 5)
* MV is the maximum-Sharpe portfolio with historical mean and sample covariance (`benchmarks.mv_objective`). It uses the same bounds as the BL portfolios. MV-ESG adds the ESG constraints. Naive is 1/N.
* Table 5 reports cumulative return, annualised mean, annualised volatility, annualised variance, and weekly historical CVaR95, each under an explicit name.

## 8. Open points for the authors
- [ ] Give the reason for excluding 6254.T.
- [ ] Record the source and date of the shares-outstanding data.
- [ ] Confirm the MV benchmark objective.
- [ ] Re-run the pipeline and compare with Tables 2, 4, 5, 6 and 7.
