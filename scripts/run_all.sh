#!/usr/bin/env bash
# Full benchmark (CPU). Resumable: finished folds are cached in results/preds/.
# Requires TABPFN_TOKEN in the environment (https://ux.priorlabs.ai).
set -euo pipefail
PY=${PY:-.venv/bin/python}
for m in 15m 1h; do $PY -m tabtrader.run --market $m --n 5000 --models market,classic_tuned; done
for m in 15m 1h; do $PY -m tabtrader.run --market $m --n 5000 --models tabpfn; done
for m in 15m 1h; do $PY -m tabtrader.run --market $m --n all  --models classic_tuned; done
for n in 250 500 1000 2000 5000; do
  for m in 15m 1h; do $PY -m tabtrader.run --market $m --n $n --models market,classic_default,tabpfn; done
done
echo ALL_DONE
