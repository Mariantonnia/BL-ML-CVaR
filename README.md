# BL-ML-CVaR-ESG

Code for

> Truyols-Pont, M. A., Arenas-Parra, M., & Bilbao-Terol, A. (2026).
> **Artificial intelligence for socially responsible investment: a multi-objective decision-making framework.**
> *Annals of Operations Research*. https://doi.org/10.1007/s10479-026-07270-2

The framework combines five price forecasters (LSTM, GRU, RNN, Random Forest and ARIMA), aggregates them by past accuracy, and feeds the result as views into a Black–Litterman model. Scenarios come from a GARCH–Student-t copula, and the final optimisation is a CVaR model with Prospect-Theory-transformed ESG constraints, solved by ε-constraints. The empirical case is the 55 constituents of the iShares Global Water UCITS ETF (S&P Global Water index), with weekly data from 2019-07-01 to 2024-07-01.

## Pipeline

| Step | Script | Paper | Main output |
|---|---|---|---|
| 1 | `scripts/01_download_data.py` | Sect. 4 | `data/processed/prices_{local,usd}.csv` |
| 2 | `scripts/02_forecast.py --model {lstm,gru,rnn,rf,arima,all}` | Sect. 3.1, 4.1, Table 1 | `results/predictions/pred_<model>.csv` |
| 3 | `scripts/03_aggregate.py` | Eqs. 1–3, Table 2, Table A2 | `pred_aggregated.csv` |
| 4 | `scripts/04_esg.py` | Sect. 3.2, Eq. 4, Table 3, Figs. 2–4 | `data/processed/esg_pt.csv` |
| 5 | `scripts/05_portfolios.py` | Sect. 3.3, Eqs. 5–20 | `results/portfolios/*` |
| 6 | `scripts/06_results.py` | Tables 4–7, A1, Figs. 5–6 | `results/tables`, `results/figures` |

All parameters (dates, splits, hyper-parameter search spaces, PT parameters `rp=50, κ=0.8, θ=2`, `τ`, CVaR level, ESG threshold 65 → v=0.38, `M_i=0.10`, 10 000 copula scenarios, seed) are set in [`config.yaml`](config.yaml).

## Installation

```bash
git clone https://github.com/<user>/bl-ml-cvar-esg.git
cd bl-ml-cvar-esg
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
```

## Data

| Data | Source | Included? |
|---|---|---|
| Weekly prices | Yahoo Finance (downloaded by step 1) | No (re-downloadable) |
| ESG scores, July 2024 | LSEG Data & Analytics | **No (proprietary licence)** |
| Shares outstanding | see `data/README.md` | No |
| ECB deposit facility rate | ECB Data Portal (public) | Add `data/raw/ecb_deposit_facility.csv` |

See [`data/README.md`](data/README.md) for the expected file formats.

## Reproducing the paper

```bash
python scripts/01_download_data.py
python scripts/02_forecast.py --model all     # long: per-asset tuning of 3 deep models
python scripts/03_aggregate.py
python scripts/04_esg.py
python scripts/05_portfolios.py
python scripts/06_results.py
```

or simply `make all`.

**Quick test without licensed data:**

```bash
python scripts/make_synthetic_data.py --n-assets 12   # writes fake data with the real layout
make all
```

The results of this test have no meaning. It only checks that the code runs.

## Tests

```bash
pytest -q
```

The unit tests check each component against its equation in the paper, for example `v(65)=0.38`, aggregation weights that sum to 1 and use only the previous week's errors, BL posterior properties, recovery of the copula degrees of freedom, and the CVaR LP constraints.

## Repository layout

```
config.yaml            all experiment parameters
data/                  tickers.csv (universe and exclusions), raw/, processed/
src/blmlcvar/          library: data, prospect, forecasting/, aggregation,
                       black_litterman, copula, optimization, benchmarks, backtest
scripts/               pipeline steps 01-06 and synthetic data generator
tests/                 unit tests
docs/IMPLEMENTATION_NOTES.md   implementation choices and differences w.r.t. the
                               preliminary notebooks
results/               generated tables, figures, predictions, portfolios
```

## Citation

If you use this code, please cite the article (see [`CITATION.cff`](CITATION.cff)).

```bibtex
@article{TruyolsPont2026,
  author  = {Truyols-Pont, M. Antonia and Arenas-Parra, Mar and Bilbao-Terol, Amelia},
  title   = {Artificial intelligence for socially responsible investment: a multi-objective decision-making framework},
  journal = {Annals of Operations Research},
  year    = {2026},
  doi     = {10.1007/s10479-026-07270-2}
}
```

## Licence

Code released under the MIT licence (see `LICENSE`). Third-party data keep their own terms.
