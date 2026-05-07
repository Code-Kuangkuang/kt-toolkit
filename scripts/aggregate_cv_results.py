"""
Tool to aggregate CV (cross-validation) results from fold directories.
Useful when some folds were rerun manually and the final summary was not executed.

Usage:
    python aggregate_cv_results.py <cv_dir>

Example:
    python aggregate_cv_results.py saved_model/cv-assist2009-dkt-20260331-170826
"""

import json
import os
import sys
import csv
import statistics
from pathlib import Path
from typing import List, Dict, Any


def find_fold_dirs(cv_dir: str) -> List[tuple]:
    """
    Find all fold directories in the CV directory.
    Returns list of (fold_id, fold_path) sorted by fold_id.
    
    Fold directories are expected to match pattern like:
    - assist2009-dkt-fold0-20260331-170827
    - assist2009-dkt-fold1-20260331-170858
    etc.
    """
    cv_path = Path(cv_dir)
    if not cv_path.exists():
        raise ValueError(f"CV directory not found: {cv_dir}")
    
    fold_dirs = []
    for item in cv_path.iterdir():
        if item.is_dir():
            # Try to extract fold_id from directory name (e.g., "fold0" from name)
            parts = item.name.split("-")
            for part in parts:
                if part.startswith("fold"):
                    try:
                        fold_id = int(part[4:])  # Extract number after "fold"
                        fold_dirs.append((fold_id, item))
                        break
                    except (ValueError, IndexError):
                        pass
    
    # Sort by fold_id
    fold_dirs.sort(key=lambda x: x[0])
    return fold_dirs


def load_fold_result(fold_dir: Path) -> Dict[str, Any]:
    """Load fold result from best_metrics.json and run_config.json"""
    best_metrics_path = fold_dir / "best_metrics.json"
    run_config_path = fold_dir / "run_config.json"
    
    result = {
        "fold": None,
        "run_name": fold_dir.name,
        "ckpt_dir": str(fold_dir),
        "emb_type": None,
        "best_metrics": {},
        "best_path": None,
    }
    
    # Load best_metrics
    if best_metrics_path.exists():
        with open(best_metrics_path, "r", encoding="utf-8") as f:
            result["best_metrics"] = json.load(f)
    
    # Load run_config for additional info
    if run_config_path.exists():
        with open(run_config_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)
            result["fold"] = run_config.get("fold")
            result["emb_type"] = run_config.get("emb_type")
            result["run_name"] = run_config.get("run_name", result["run_name"])
    
    return result


def aggregate_fold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all folds (mean and std)"""
    numeric_keys = set()
    for r in fold_results:
        bm = r.get("best_metrics") or {}
        for k, v in bm.items():
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    
    # Also include last-epoch test metrics
    for r in fold_results:
        for k in ("last_test_auc", "last_test_acc"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    
    summary = {}
    for k in sorted(numeric_keys):
        values = []
        for r in fold_results:
            bm = r.get("best_metrics") or {}
            v = bm.get(k)
            if isinstance(v, (int, float)):
                values.append(float(v))
        if not values:
            continue
        mean_v = statistics.mean(values)
        std_v = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[k] = {
            "mean": mean_v,
            "std": std_v,
            "values": values,
        }
    return summary


def save_cv_summary(cv_dir: str, cv_payload: Dict[str, Any], fold_results: List[Dict[str, Any]]) -> None:
    """Save cv_summary.json and cv_summary.csv"""
    # Save JSON
    json_path = os.path.join(cv_dir, "cv_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cv_payload, f, indent=2, ensure_ascii=True)
    print(f"✓ Saved: {json_path}")
    
    # Save CSV
    csv_path = os.path.join(cv_dir, "cv_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold", "valid_auc", "valid_acc",
                "best_test_auc", "best_test_acc",
                "last_test_auc", "last_test_acc",
                "best_epoch", "ckpt_dir", "best_path", "run_name"
            ],
        )
        writer.writeheader()
        for r in fold_results:
            bm = r.get("best_metrics") or {}
            writer.writerow({
                "fold": r.get("fold"),
                "valid_auc": bm.get("valid_auc"),
                "valid_acc": bm.get("valid_acc"),
                "best_test_auc": bm.get("best_test_auc"),
                "best_test_acc": bm.get("best_test_acc"),
                "last_test_auc": bm.get("last_test_auc"),
                "last_test_acc": bm.get("last_test_acc"),
                "best_epoch": bm.get("epoch"),
                "ckpt_dir": r.get("ckpt_dir"),
                "best_path": r.get("best_path"),
                "run_name": r.get("run_name"),
            })
    print(f"✓ Saved: {csv_path}")


def print_cv_summary(agg: Dict[str, Any], cv_dir: str) -> None:
    """Print CV summary statistics"""
    print("\n" + "=" * 60)
    print("CV SUMMARY")
    print("=" * 60)
    
    if "valid_auc" in agg:
        m = agg["valid_auc"]["mean"]
        s = agg["valid_auc"]["std"]
        print(f"✓ valid_auc:     mean={m:.6f}  std={s:.6f}")
    if "valid_acc" in agg:
        m = agg["valid_acc"]["mean"]
        s = agg["valid_acc"]["std"]
        print(f"✓ valid_acc:     mean={m:.6f}  std={s:.6f}")
    if "best_test_auc" in agg:
        m = agg["best_test_auc"]["mean"]
        s = agg["best_test_auc"]["std"]
        print(f"✓ best_test_auc: mean={m:.6f}  std={s:.6f}")
    if "best_test_acc" in agg:
        m = agg["best_test_acc"]["mean"]
        s = agg["best_test_acc"]["std"]
        print(f"✓ best_test_acc: mean={m:.6f}  std={s:.6f}")
    if "last_test_auc" in agg:
        m = agg["last_test_auc"]["mean"]
        s = agg["last_test_auc"]["std"]
        print(f"  last_test_auc:  mean={m:.6f}  std={s:.6f}")
    if "last_test_acc" in agg:
        m = agg["last_test_acc"]["mean"]
        s = agg["last_test_acc"]["std"]
        print(f"  last_test_acc:  mean={m:.6f}  std={s:.6f}")
    
    print("=" * 60)
    print(f"CV results saved to: {cv_dir}")
    print("=" * 60 + "\n")


def main(cv_dir: str) -> None:
    """Main function"""
    cv_dir = os.path.abspath(cv_dir)
    
    print(f"\n[*] Scanning CV directory: {cv_dir}")
    
    # Find all fold directories
    fold_dirs = find_fold_dirs(cv_dir)
    if not fold_dirs:
        print("✗ No fold directories found!")
        sys.exit(1)
    
    print(f"✓ Found {len(fold_dirs)} fold(s): {[f[0] for f in fold_dirs]}")
    
    # Load all fold results
    print("\n[*] Loading fold results...")
    fold_results = []
    for fold_id, fold_path in fold_dirs:
        result = load_fold_result(fold_path)
        fold_results.append(result)
        bm = result.get("best_metrics", {})
        valid_auc = bm.get("valid_auc", -1)
        print(f"  fold{fold_id}: valid_auc={valid_auc:.4f}" if valid_auc >= 0 else f"  fold{fold_id}: valid_auc=N/A")
    
    # Aggregate metrics
    print("\n[*] Aggregating fold metrics...")
    agg = aggregate_fold_metrics(fold_results)
    
    # Extract CV info from first fold's run_config if available
    cv_payload = {
        "cv_run_name": Path(cv_dir).name,
        "timestamp": None,
        "dataset_name": None,
        "model_name": None,
        "emb_type": fold_results[0].get("emb_type") if fold_results else None,
        "folds": [r.get("fold") for r in fold_results],
        "seed": None,
        "save_dir": str(Path(cv_dir).parent),
        "cv_dir": cv_dir,
        "per_fold": fold_results,
        "aggregate": agg,
    }
    
    # Try to extract more info from run_config files
    for result in fold_results:
        ckpt_dir = result.get("ckpt_dir")
        if ckpt_dir:
            run_config_path = Path(ckpt_dir) / "run_config.json"
            if run_config_path.exists():
                with open(run_config_path, "r", encoding="utf-8") as f:
                    run_config = json.load(f)
                    if not cv_payload["timestamp"]:
                        cv_payload["timestamp"] = run_config.get("timestamp")
                    if not cv_payload["dataset_name"]:
                        cv_payload["dataset_name"] = run_config.get("dataset_name")
                    if not cv_payload["model_name"]:
                        cv_payload["model_name"] = run_config.get("model_name")
                    if not cv_payload["seed"]:
                        cv_payload["seed"] = run_config.get("seed")
                    if not cv_payload["timestamp"]:
                        break
    
    # Save summary files
    print("\n[*] Saving summary files...")
    save_cv_summary(cv_dir, cv_payload, fold_results)
    
    # Print summary
    print_cv_summary(agg, cv_dir)
    print("✓ Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: CV directory path is required")
        sys.exit(1)
    
    cv_dir = sys.argv[1]
    try:
        main(cv_dir)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
