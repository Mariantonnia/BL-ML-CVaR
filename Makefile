PY ?= python
all: data forecast aggregate esg portfolios results
data:       ; $(PY) scripts/01_download_data.py
forecast:   ; $(PY) scripts/02_forecast.py --model all
aggregate:  ; $(PY) scripts/03_aggregate.py
esg:        ; $(PY) scripts/04_esg.py
portfolios: ; $(PY) scripts/05_portfolios.py
results:    ; $(PY) scripts/06_results.py
test:       ; $(PY) -m pytest -q
.PHONY: all data forecast aggregate esg portfolios results test
