"""Rebuild CV summaries from completed fold directories.

Usage:
    python scripts/aggregate_cv_results.py saved_model/cv-assist2009-dkt-YYYYMMDD-HHMMSS
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cv_results import aggregate_fold_metrics, print_cv_summary, save_cv_summary


def find_fold_dirs(cv_dir: str) -> List[tuple]:
    cv_path = Path(cv_dir)
    if not cv_path.exists():
        raise ValueError(f"CV directory not found: {cv_dir}")

    fold_dirs = []
    for item in cv_path.iterdir():
        if not item.is_dir():
            continue
        for part in item.name.split("-"):
            if not part.startswith("fold"):
                continue
            try:
                fold_dirs.append((int(part[4:]), item))
            except ValueError:
                pass
            break
    return sorted(fold_dirs, key=lambda item: item[0])


def load_fold_result(fold_dir: Path) -> Dict[str, Any]:
    result = {
        "fold": None,
        "run_name": fold_dir.name,
        "ckpt_dir": str(fold_dir),
        "emb_type": None,
        "best_metrics": {},
        "best_path": None,
    }

    best_metrics_path = fold_dir / "best_metrics.json"
    if best_metrics_path.exists():
        result["best_metrics"] = json.loads(best_metrics_path.read_text(encoding="utf-8"))

    run_config_path = fold_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        result["fold"] = run_config.get("fold")
        result["emb_type"] = run_config.get("emb_type")
        result["run_name"] = run_config.get("run_name", result["run_name"])
    return result


def main(cv_dir: str) -> None:
    cv_dir = os.path.abspath(cv_dir)
    print(f"Scanning CV directory: {cv_dir}")
    fold_dirs = find_fold_dirs(cv_dir)
    if not fold_dirs:
        raise ValueError("No fold directories found.")

    fold_results = [load_fold_result(path) for _, path in fold_dirs]
    aggregate = aggregate_fold_metrics(fold_results)
    payload = {
        "cv_run_name": Path(cv_dir).name,
        "timestamp": None,
        "dataset_name": None,
        "model_name": None,
        "emb_type": fold_results[0].get("emb_type"),
        "folds": [result.get("fold") for result in fold_results],
        "seed": None,
        "save_dir": str(Path(cv_dir).parent),
        "cv_dir": cv_dir,
        "per_fold": fold_results,
        "aggregate": aggregate,
    }

    for result in fold_results:
        run_config_path = Path(result["ckpt_dir"]) / "run_config.json"
        if not run_config_path.exists():
            continue
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key in ("timestamp", "dataset_name", "model_name", "seed"):
            if payload[key] is None:
                payload[key] = run_config.get(key)

    save_cv_summary(cv_dir, payload, fold_results)
    print_cv_summary(aggregate, cv_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit("CV directory path is required.")
    main(sys.argv[1])
