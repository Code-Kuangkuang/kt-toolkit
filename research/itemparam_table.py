"""Paired five-fold comparison of SimpleKT's three item-parameter widths.

The three runs differ in one config key, `emb_type`, which selects how many free
parameters each question gets in `models/simplekt.py`:

    qid_norasch  ->  0    the Rasch block at L188 is skipped entirely
    qid_scalar   ->  1    difficult_param is Embedding(num_pid+1, 1)
    qid          ->  d    difficult_param is Embedding(num_pid+1, embed_l)

Everything else -- backbone, trainer, seed, folds, protocol stamp -- is held
fixed, so the difference between rows is the parameterisation and nothing else.
This is the single-variable version of the cross-model comparison in
`experiment/baseline_table.md`, where `akt` (1 per item) and `simplekt` (d per
item) also differ in attention mechanism, learning rate and block count.

Refuses to compare runs whose protocol stamps differ, for the reason
docs/evaluation_protocol.md §1.5 gives.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = [
    ("qid_norasch", "itemparam_norasch", 0),
    ("qid_scalar", "itemparam_scalar", 1),
    ("qid", "itemparam_full", None),  # None -> read embed_l from the config
]

METRIC = "best_window_test_auc"


def read_fold(save_root: str, fold: int, seed: int):
    base = os.path.join(save_root, f"assist2009__simplekt__fold{fold}__seed{seed}")
    hits = sorted(glob.glob(os.path.join(base, "*", "best_metrics.json")))
    if not hits:
        return None
    run_dir = os.path.dirname(hits[-1])
    metrics = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    config = json.loads(
        Path(os.path.join(run_dir, "run_config.json")).read_text(encoding="utf-8")
    )
    return metrics, config


def paired_t(delta: np.ndarray) -> tuple[float, float]:
    """Paired t over folds, and the sd of the per-fold differences."""
    sd = float(delta.std(ddof=1))
    if sd == 0:
        return float("nan"), sd
    return float(delta.mean() / (sd / np.sqrt(len(delta)))), sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="saved_model")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    protocols: dict[str, str] = {}
    results: dict[str, list[float]] = {}
    accs: dict[str, list[float]] = {}
    epochs: dict[str, list[int]] = {}
    widths: dict[str, int] = {}

    for emb_type, subdir, k in VARIANTS:
        save_root = os.path.join(args.root, subdir)
        aucs, ac, ep = [], [], []
        for fold in args.folds:
            got = read_fold(save_root, fold, args.seed)
            if got is None:
                print(f"missing: {emb_type} fold {fold}")
                return 1
            metrics, config = got
            if config.get("emb_type") != emb_type:
                print(f"{save_root} fold {fold} stamped emb_type="
                      f"{config.get('emb_type')!r}, expected {emb_type!r}")
                return 1
            protocols.setdefault(
                emb_type, json.dumps(config.get("protocol"), sort_keys=True)
            )
            aucs.append(metrics[METRIC])
            ac.append(metrics.get("best_window_test_acc"))
            ep.append(metrics.get("best_epoch"))
            if k is None:
                widths[emb_type] = int(config.get("emb_size") or config.get("d_model") or 256)
            else:
                widths[emb_type] = k
        results[emb_type] = aucs
        accs[emb_type] = ac
        epochs[emb_type] = ep

    stamps = set(protocols.values())
    if len(stamps) != 1:
        print("protocol stamps differ across variants; refusing to compare:")
        for k_, v in protocols.items():
            print(f"  {k_}: {v}")
        return 1
    print(f"protocol (identical across all three): {stamps.pop()}")
    print(f"seed {args.seed}, folds {args.folds}, metric {METRIC}\n")

    print(f"| emb_type | free params per item | WIN AUC (mean) | sd over folds |")
    print(f"|---|---|---|---|")
    for emb_type, _, _ in VARIANTS:
        a = np.array(results[emb_type])
        print(f"| `{emb_type}` | {widths[emb_type]} | {a.mean():.5f} | {a.std(ddof=1):.5f} |")

    print("\npaired over the five folds:\n")
    print("| contrast | delta | sd(delta) | t (df=4) | folds won |")
    print("|---|---|---|---|---|")
    pairs = [("qid_norasch", "qid_scalar"), ("qid_scalar", "qid"), ("qid_norasch", "qid")]
    per_fold = {}
    for a_name, b_name in pairs:
        a = np.array(results[a_name])
        b = np.array(results[b_name])
        d = b - a
        t, sd = paired_t(d)
        per_fold[f"{a_name} -> {b_name}"] = d
        print(f"| `{a_name}` → `{b_name}` | {d.mean():+.5f} | {sd:.5f} | "
              f"{t:+.2f} | {int((d > 0).sum())}/{len(d)} |")

    print("\nper-fold deltas:\n```")
    for name, d in per_fold.items():
        print(f"{name:32s} " + "  ".join(f"{x:+.5f}" for x in d))
    print("```")
    print("\n|t| would need ~2.78 for p < 0.05 at df = 4.")

    print("\nper-fold WIN AUC:\n```")
    for emb_type, _, _ in VARIANTS:
        print(f"{emb_type:12s} " + "  ".join(f"{x:.5f}" for x in results[emb_type])
              + f"   epochs {epochs[emb_type]}")
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
