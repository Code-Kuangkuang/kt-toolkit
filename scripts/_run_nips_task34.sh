#!/usr/bin/env bash
# nips_task34 的 21 模型基线扫（与 bridge2algebra2006 同一组模型，不含 keenkt）。
# run_baseline_table.py 本身可断点续跑：已有 best_metrics.json 的格子会跳过，
# 所以这个脚本重复执行是安全的。
set -eu

PY="${PYTHON:-python}"
cd "$(dirname "$0")/.." || exit 1

MODELS="dkt dkt+ dkvmn deep_irt sakt saint saint_plus akt atkt kqn qikt ukt \
simplekt stablekt sparsekt robustkt atdkt cskt extrakt fluckt folibikt"

"$PY" scripts/run_baseline_table.py \
  --datasets nips_task34 \
  --models $MODELS \
  --folds 0 1 2 3 4
