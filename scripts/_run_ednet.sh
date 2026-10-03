#!/usr/bin/env bash
# ednet, 22 models x 5 folds, then rebuild the combined table across all five
# datasets.
#
# Two steps rather than one because `summarize()` only emits the datasets passed
# in --datasets: writing ednet straight to experiment/baseline_table.md would
# replace the four finished datasets with an ednet-only table.  So ednet goes to
# its own file and a second --summarize-only pass rebuilds the combined one.
#
# Resumable: a cell whose best_metrics.json exists is skipped, so Ctrl-C is safe
# and re-running this script continues where it stopped.

set -eu
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MODELS="dkt dkt+ dkvmn deep_irt sakt saint saint_plus akt atkt kqn atdkt qikt keenkt ukt simplekt stablekt sparsekt robustkt extrakt folibikt cskt fluckt"

echo "[ednet] start $(date +%H:%M:%S)"
"$PY" scripts/run_baseline_table.py \
    --datasets ednet --folds 0 1 2 3 4 --seed 3407 \
    --models $MODELS \
    --save-root saved_model/baseline_table \
    --log-dir logs/baseline_table \
    --out experiment/_ednet.md

echo "[ednet] rebuilding combined table $(date +%H:%M:%S)"
"$PY" scripts/run_baseline_table.py --summarize-only \
    --datasets assist2009 algebra2005 assist2017 statics2011 ednet \
    --folds 0 1 2 3 4 --seed 3407 \
    --models $MODELS \
    --save-root saved_model/baseline_table \
    --out experiment/baseline_table.md

echo "[ednet] done $(date +%H:%M:%S)"
