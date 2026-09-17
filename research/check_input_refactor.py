"""Prove that moving a model's extra inputs into an `Inputs` spec changed nothing.

The migration described in docs/architecture.md under "Known Structural Debt"
moves per-model feature preparation out of `core/train_runner.py` and into the
model. It is pure code motion, so a migrated model must produce *bit-identical*
metrics -- same seed, same fold, same numbers to full precision. "Close enough"
is not a pass: a small drift means the RNG stream moved, which means a hook ran
after `set_seed`, which means every later result is off by an unknown amount.

Usage -- before touching the model:

    python research/check_input_refactor.py record --models gkt --fold 0 --epochs 1

then after moving it:

    python research/check_input_refactor.py verify --models gkt --fold 0 --epochs 1

`record` writes a baseline under .cache/input_refactor/; `verify` re-runs and
diffs. Run it per model rather than batching the whole migration -- if five
models move and the check fails, it does not say which one broke.

`run_config.json` is diffed too, but as a warning: a spec reporting its config
changes through `ModelInputs` instead of mutating dictionaries can legitimately
reorder keys. Every difference should still be explainable; an unexplained one
usually means a `model_cfg` key was left behind.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / ".cache" / "input_refactor"

# Compared exactly. These are the numbers a paper would quote.
METRIC_KEYS = (
    "valid_auc",
    "valid_acc",
    "best_test_auc",
    "best_test_acc",
    "best_window_test_auc",
    "best_window_test_acc",
    "last_test_auc",
    "last_test_acc",
    "epoch",
)

# Differ per run by construction, so they are excluded from the config diff.
VOLATILE_CONFIG_KEYS = {"run_name", "timestamp", "ckpt_dir", "save_dir", "device"}


def run_one(model, dataset, fold, epochs, out_root):
    """Train one fold and return (best_metrics, run_config)."""
    save_dir = out_root / f"{dataset}_{model}_fold{fold}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--dataset_name", dataset,
        "--model_name", model,
        "--fold", str(fold),
        "--num_epochs", str(epochs),
        "--save_dir", str(save_dir),
        "--use_wandb", "0",
    ]
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    # Streamed rather than captured: GKT's windowed test pass alone runs for
    # many minutes, and a silent subprocess is indistinguishable from a hung one.
    # The tail is kept only so a failure can be reported without scrollback.
    tail = []
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
    if proc.wait() != 0:
        raise SystemExit(f"训练失败 ({model}, fold {fold}):\n{''.join(tail)}")

    runs = sorted(save_dir.glob("*/best_metrics.json"), key=os.path.getmtime)
    if not runs:
        raise SystemExit(f"没找到 best_metrics.json，训练可能没写出产物: {save_dir}")
    metrics = json.loads(runs[-1].read_text(encoding="utf-8"))
    config_path = runs[-1].parent / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return metrics, config


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in VOLATILE_CONFIG_KEYS:
                continue
            yield from flatten(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from flatten(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare(model, before, after):
    """Return True when the run is bit-identical on the metrics that matter."""
    b_metrics, b_config = before
    a_metrics, a_config = after

    print(f"\n{'=' * 60}\n{model}\n{'=' * 60}")
    ok = True

    print("指标（必须完全相同）:")
    for key in METRIC_KEYS:
        old, new = b_metrics.get(key), a_metrics.get(key)
        if old is None and new is None:
            continue
        if old == new:
            print(f"  OK   {key:<24} {old}")
        else:
            ok = False
            delta = ""
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                delta = f"   (diff {new - old:+.10f})"
            print(f"  FAIL {key:<24} {old}  ->  {new}{delta}")

    extra = (set(b_metrics) | set(a_metrics)) - set(METRIC_KEYS)
    for key in sorted(extra):
        old, new = b_metrics.get(key), a_metrics.get(key)
        if old != new:
            ok = False
            print(f"  FAIL {key:<24} {old}  ->  {new}   (metric not in the known list)")

    b_flat, a_flat = dict(flatten(b_config)), dict(flatten(a_config))
    diffs = [
        (k, b_flat.get(k), a_flat.get(k))
        for k in sorted(set(b_flat) | set(a_flat))
        if b_flat.get(k) != a_flat.get(k)
    ]
    if diffs:
        print(f"\nrun_config 差异 {len(diffs)} 处（警告，不判定失败——但每一处都要能解释）:")
        for key, old, new in diffs[:25]:
            print(f"  ~ {key}: {old!r}  ->  {new!r}")
        if len(diffs) > 25:
            print(f"  ... 还有 {len(diffs) - 25} 处")
    else:
        print("\nrun_config: 无差异")

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "verify"])
    parser.add_argument("--models", required=True, help="逗号分隔，一次只查一个最好")
    parser.add_argument("--dataset", default="assist2009")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_fold{args.fold}_ep{args.epochs}"

    failures = []
    for model in models:
        baseline_path = BASELINE_DIR / f"{model}_{tag}.json"
        print(f"\n[{args.mode}] {model} on {args.dataset} fold {args.fold}, {args.epochs} epoch(s)")
        metrics, config = run_one(
            model, args.dataset, args.fold, args.epochs, BASELINE_DIR / args.mode
        )

        if args.mode == "record":
            baseline_path.write_text(
                json.dumps({"metrics": metrics, "config": config}, indent=2),
                encoding="utf-8",
            )
            print(f"  基线已写入 {baseline_path.relative_to(ROOT)}")
            for key in METRIC_KEYS:
                if metrics.get(key) is not None:
                    print(f"    {key:<24} {metrics[key]}")
            continue

        if not baseline_path.exists():
            raise SystemExit(
                f"没有 {model} 的基线。先在改动前跑：\n"
                f"  python research/check_input_refactor.py record "
                f"--models {model} --dataset {args.dataset} "
                f"--fold {args.fold} --epochs {args.epochs}"
            )
        saved = json.loads(baseline_path.read_text(encoding="utf-8"))
        if not compare(model, (saved["metrics"], saved["config"]), (metrics, config)):
            failures.append(model)

    if args.mode == "verify":
        print(f"\n{'=' * 60}")
        if failures:
            print(f"不等价: {', '.join(failures)}")
            print("指标变了就说明不是纯搬家。最常见的原因是某个 prepare 跑到了 set_seed 之后。")
            raise SystemExit(1)
        print(f"全部等价（{len(models)} 个模型，指标逐位相同）")


if __name__ == "__main__":
    main()
