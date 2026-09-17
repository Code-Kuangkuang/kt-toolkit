"""Run a baseline comparison sweep and emit a protocol-checked markdown table.

Two things this does that a shell loop does not:

  * It is resumable.  A run whose `best_metrics.json` already exists is skipped,
    so an interrupted sweep continues where it stopped instead of starting over.
  * It refuses to hand you a table whose rows were produced under different
    protocols.  Every run stamps `run_config.json` with a `protocol` block
    (dataset_mode / concept_mode / concepts_visible / score_repeated_kc /
    eval_window); rows are grouped by that stamp and any mismatch is reported
    loudly.  Mixing `concept_mode: multi` with `first`, or `eval_window: true`
    with `false`, is exactly how a comparison table ends up measuring the
    harness instead of the models -- see docs/evaluation_protocol.md.

Run the sweep (hours; resumable, so Ctrl-C is safe):

    python scripts/run_baseline_table.py --datasets assist2009 --folds 0

Then emit the table without re-running anything:

    python scripts/run_baseline_table.py --summarize-only --out experiment/baseline_table.md
"""

import argparse
import datetime
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Standard baselines that appear in KT comparison tables.  `gkt` and `rekt` are
# included but will always stamp `concepts_visible: first_of_K` -- they keep
# per-KC state arrays and cannot take pooled concepts (docs §3.4).
DEFAULT_MODELS = [
    "dkt", "dkt+", "dkvmn", "deep_irt", "sakt", "saint", "saint_plus",
    "akt", "atkt", "gkt", "kqn", "skvmn", "dimkt", "atdkt",
    "simplekt", "stablekt", "sparsekt", "robustkt", "dtransformer",
    "iekt", "qikt", "ukt", "rekt", "lpkt",
]

METRIC_COLUMNS = [
    ("best_window_test_auc", "WIN AUC"),
    ("best_window_test_acc", "WIN ACC"),
    ("best_test_auc", "AUC"),
    ("best_test_acc", "ACC"),
]


def run_dir_name(dataset, model, fold, seed):
    return f"{dataset}__{model}__fold{fold}__seed{seed}"


def find_result(save_root, dataset, model, fold, seed):
    """The finished run for this cell, or None.  Presence of best_metrics.json
    is the completion marker -- it is written only after test evaluation."""
    base = Path(save_root) / run_dir_name(dataset, model, fold, seed)
    if not base.exists():
        return None
    for metrics_path in sorted(base.glob("*/best_metrics.json")):
        config_path = metrics_path.parent / "run_config.json"
        if not config_path.exists():
            continue
        try:
            return {
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "config": json.loads(config_path.read_text(encoding="utf-8")),
                "dir": str(metrics_path.parent),
            }
        except json.JSONDecodeError:
            continue
    return None


def train_one(dataset, model, fold, seed, save_root, extra_args, log_dir):
    save_dir = Path(save_root) / run_dir_name(dataset, model, fold, seed)
    log_path = Path(log_dir) / f"{run_dir_name(dataset, model, fold, seed)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--dataset-name", dataset,
        "--model-name", model,
        "--fold", str(fold),
        "--seed", str(seed),
        "--use-wandb", "0",
        "--save-dir", str(save_dir),
        *extra_args,
    ]
    started = datetime.datetime.now()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    elapsed = (datetime.datetime.now() - started).total_seconds() / 60
    return completed.returncode, elapsed, log_path


def protocol_key(config):
    """Group runs by every field of the protocol block, not a subset.

    A field recorded but not grouped on is decorative: `feature_fit_scope` would
    sit in the artifact while a `train_folds` run and a `train_valid_test` run
    landed in the same table anyway. The key therefore covers all eight fields.

    Missing fields become "unknown" rather than a default. A run produced before
    a field existed did not record it, and assuming it matches a current run is
    the assumption this whole mechanism exists to stop making -- the same
    reasoning that already refuses a run with no protocol block at all.
    """
    p = config.get("protocol")
    if not p:
        # Runs from before the protocol stamp existed cannot be placed in a
        # table, because there is no way to tell what protocol produced them.
        return None

    def field(name, default="unknown"):
        value = p.get(name)
        return default if value is None else value

    return (
        field("dataset_mode"),
        field("concept_mode"),
        field("concepts_visible"),
        field("max_concepts"),
        bool(p.get("score_repeated_kc")),
        bool(p.get("eval_window", True)),
        field("feature_fit_scope"),
        field("graph_scope"),
    )


def describe_protocol(key):
    mode, concept, visible, width, repeated, window, feature_scope, graph_scope = key
    parts = [f"{mode}/{concept}", f"concepts={visible}", f"max_concepts={width}"]
    if repeated:
        parts.append("**score_repeated_kc=TRUE (leaky)**")
    if not window:
        parts.append("**no windowed metric**")
    for label, scope in (("features", feature_scope), ("graph", graph_scope)):
        if scope == "train_valid_test":
            parts.append(f"**{label} fitted on test too (transductive)**")
        elif scope == "unknown":
            parts.append(f"**{label} scope unrecorded**")
    return ", ".join(parts)


def summarize(save_root, datasets, models, folds, seed, out_path):
    rows, missing, unstamped = [], [], []
    for dataset in datasets:
        for model in models:
            for fold in folds:
                found = find_result(save_root, dataset, model, fold, seed)
                if found is None:
                    missing.append((dataset, model, fold))
                    continue
                key = protocol_key(found["config"])
                if key is None:
                    unstamped.append((dataset, model, fold))
                    continue
                rows.append({
                    "dataset": dataset, "model": model, "fold": fold,
                    "protocol": key, "metrics": found["metrics"],
                })

    lines = ["# 基线对比表", "",
             f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M}",
             f"结果目录：`{save_root}`　seed：{seed}", ""]

    groups = {}
    for row in rows:
        groups.setdefault(row["protocol"], []).append(row)

    if len(groups) > 1:
        lines += [
            "> **警告：这些结果产于多种不同协议，不能放进同一张表。**",
            "> 下面按协议分组列出。只有同一组内的行才可以互相比较。",
            "> 协议含义见 `docs/evaluation_protocol.md` §1.5。", "",
        ]

    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        group = groups[key]
        if len(groups) > 1:
            lines += [f"## 协议：{describe_protocol(key)}", ""]
        elif key[3] or not key[4]:
            lines += [f"协议：{describe_protocol(key)}", ""]

        for dataset in datasets:
            subset = [r for r in group if r["dataset"] == dataset]
            if not subset:
                continue
            lines += [f"### {dataset}", "",
                      "| model | " + " | ".join(c[1] for c in METRIC_COLUMNS) + " | epoch |",
                      "|---" * (len(METRIC_COLUMNS) + 2) + "|"]
            for row in sorted(subset, key=lambda r: models.index(r["model"])):
                m = row["metrics"]
                cells = []
                for field, _ in METRIC_COLUMNS:
                    value = m.get(field)
                    cells.append(f"{value:.5f}" if isinstance(value, (int, float)) and value >= 0 else "—")
                lines.append(f"| `{row['model']}` | " + " | ".join(cells)
                             + f" | {m.get('epoch', '—')} |")
            lines.append("")

    if unstamped:
        lines += ["## 无协议指纹（产于 2026-09-16 修复之前，不可用）", ""]
        lines += [f"- {d} / `{m}` / fold {f}" for d, m, f in unstamped] + [""]
    if missing:
        lines += ["## 缺失（未跑或跑失败）", ""]
        lines += [f"- {d} / `{m}` / fold {f}" for d, m, f in missing] + [""]

    text = "\n".join(lines)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"表已写入 {out_path}")
    else:
        print(text)
    print(f"\n完成 {len(rows)} 格，缺失 {len(missing)} 格，无指纹 {len(unstamped)} 格，"
          f"协议分组 {len(groups)} 个")
    return 1 if (len(groups) > 1 or unstamped) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["assist2009"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-root", default="saved_model/baseline_table")
    parser.add_argument("--log-dir", default="logs/baseline_table")
    parser.add_argument("--out", default="experiment/baseline_table.md")
    parser.add_argument("--summarize-only", action="store_true",
                        help="只汇总已有结果，不跑任何训练")
    parser.add_argument("--rerun", action="store_true",
                        help="忽略已完成的结果，全部重跑")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要跑哪些格子")
    parser.add_argument("--train-args", default="",
                        help='原样传给 scripts/train.py 的参数，整体加引号，'
                             '例如 --train-args "--num-epochs 1 --patience 5"')
    args = parser.parse_args()

    extra = shlex.split(args.train_args)

    if not args.summarize_only:
        todo = [(d, m, f) for d in args.datasets for m in args.models for f in args.folds
                if args.rerun or find_result(args.save_root, d, m, f, args.seed) is None]
        done = len(args.datasets) * len(args.models) * len(args.folds) - len(todo)
        print(f"待跑 {len(todo)} 格，已完成 {done} 格（已完成的会跳过，用 --rerun 强制重跑）\n")
        if args.dry_run:
            for d, m, f in todo:
                print(f"  {d} / {m} / fold {f}")
            return 0

        failures = []
        for i, (dataset, model, fold) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {dataset} / {model} / fold {fold} ...", flush=True)
            code, minutes, log_path = train_one(
                dataset, model, fold, args.seed, args.save_root, extra, args.log_dir)
            if code == 0:
                print(f"      完成，用时 {minutes:.1f} 分钟")
            else:
                failures.append((dataset, model, fold, log_path))
                print(f"      失败（退出码 {code}），日志：{log_path}")
        if failures:
            print(f"\n{len(failures)} 格失败：")
            for dataset, model, fold, log_path in failures:
                print(f"  {dataset} / {model} / fold {fold}  →  {log_path}")
        print()

    return summarize(args.save_root, args.datasets, args.models,
                     args.folds, args.seed, args.out)


if __name__ == "__main__":
    sys.exit(main())
