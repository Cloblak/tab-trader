# tab-trader — everything runs on CPU. TabPFN needs TABPFN_TOKEN (free: https://ux.priorlabs.ai).
PY ?= .venv/bin/python

.PHONY: install quick benchmark thinking controls analyze report notebooks verify all

install:
	uv sync

quick:            ## one test week per market, all models (~10 min on 8 cores)
	$(PY) -m tabtrader.run --sub quick --market 15m --n 5000 --models market,classic_default,tabpfn --blocks 8
	$(PY) -m tabtrader.run --sub quick --market 1h  --n 5000 --models market,classic_default,tabpfn --blocks 5
	$(PY) -m tabtrader.quick

benchmark:        ## full walk-forward benchmark + learning curves (hours; resumable)
	PY=$(PY) bash scripts/run_all.sh

thinking:         ## TabPFN-3.5 Thinking via the Prior Labs API (uses API credits)
	$(PY) -m tabtrader.run --market 15m --n 5000 --models thinking
	$(PY) -m tabtrader.run --market 1h  --n 5000 --models thinking

controls:         ## planted-signal and calibrated-null harness checks
	$(PY) -m tabtrader.controls

analyze:          ## cached predictions -> results/summary.json
	$(PY) -m tabtrader.analyze

report:           ## results -> docs/index.html (GitHub Pages)
	$(PY) -m tabtrader.report
	$(PY) -m tabtrader.readme

notebooks:        ## regenerate and execute notebooks/
	$(PY) scripts/make_notebooks.py
	cd notebooks && for n in *.ipynb; do ../$(PY) -m jupyter nbconvert --to notebook --execute --inplace $$n --ExecutePreprocessor.timeout=900; done

verify:           ## leakage, reproducibility, claim and secret checks
	$(PY) scripts/verify.py

all: benchmark thinking controls analyze report notebooks verify
