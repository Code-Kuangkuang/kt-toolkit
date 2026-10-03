#!/usr/bin/env bash
# Wait for the assist2017 sweep to finish, then run statics2011 and rebuild the
# combined table.  Deliberately not a chained `&&`: the running sweep was started
# in another shell, so the only thing to synchronise on is its manifest.
#
# Done = the last cell of the running command (assist2017 / fluckt / fold 4) has
# a manifest row, OR nothing new has landed for 35 minutes (the longest single
# run on assist2017 is keenkt at ~12 min, so 35 means it stopped, one way or
# another).  Max wait 9 h.

set -eu
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
MANIFEST=saved_model/baseline_table/manifest.jsonl
MODELS="dkt dkt+ dkvmn deep_irt sakt saint saint_plus akt atkt kqn atdkt qikt keenkt ukt simplekt stablekt sparsekt robustkt extrakt folibikt cskt fluckt"

deadline=$(( $(date +%s) + 9*3600 ))
last_size=$(wc -c < "$MANIFEST")
last_change=$(date +%s)

while :; do
    if "$PY" -c "
import json,sys
rows=[json.loads(l) for l in open('$MANIFEST')]
sys.exit(0 if any(r['dataset']=='assist2017' and r['model']=='fluckt' and r['fold']==4
                  for r in rows) else 1)
"; then
        echo "[queue] assist2017 sweep finished at $(date +%H:%M:%S)"
        break
    fi

    now=$(date +%s)
    size=$(wc -c < "$MANIFEST")
    if [ "$size" != "$last_size" ]; then
        last_size=$size
        last_change=$now
    elif [ $(( now - last_change )) -gt 2100 ]; then
        echo "[queue] no new manifest row for 35 min - treating the sweep as stopped"
        break
    fi

    if [ "$now" -gt "$deadline" ]; then
        echo "[queue] 9 h deadline hit, giving up without starting statics2011"
        exit 1
    fi
    sleep 60
done

sleep 30   # let the sweep write its own summary before touching anything

echo "[queue] starting statics2011 at $(date +%H:%M:%S)"
"$PY" scripts/run_baseline_table.py \
    --datasets statics2011 --folds 0 1 2 3 4 --seed 3407 \
    --models $MODELS \
    --save-root saved_model/baseline_table \
    --log-dir logs/baseline_table \
    --out experiment/_statics2011.md

echo "[queue] rebuilding the combined table at $(date +%H:%M:%S)"
"$PY" scripts/run_baseline_table.py --summarize-only \
    --datasets assist2009 algebra2005 assist2017 statics2011 \
    --folds 0 1 2 3 4 --seed 3407 \
    --models $MODELS \
    --save-root saved_model/baseline_table \
    --out experiment/baseline_table.md

echo "[queue] done at $(date +%H:%M:%S)"
