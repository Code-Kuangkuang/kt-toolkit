import csv
import json
import os
import statistics

from rich import print


def aggregate_fold_metrics(fold_results):
    numeric_keys = set()
    for result in fold_results:
        best_metrics = result.get("best_metrics") or {}
        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    summary = {}
    for key in sorted(numeric_keys):
        values = []
        for result in fold_results:
            best_metrics = result.get("best_metrics") or {}
            value = best_metrics.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def save_cv_summary(cv_dir, cv_payload, fold_results):
    os.makedirs(cv_dir, exist_ok=True)
    json_path = os.path.join(cv_dir, "cv_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cv_payload, f, indent=2, ensure_ascii=True)

    csv_path = os.path.join(cv_dir, "cv_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "valid_auc",
                "valid_acc",
                "best_test_auc",
                "best_test_acc",
                "last_test_auc",
                "last_test_acc",
                "best_epoch",
                "ckpt_dir",
                "best_path",
                "run_name",
            ],
        )
        writer.writeheader()
        for result in fold_results:
            best_metrics = result.get("best_metrics") or {}
            writer.writerow(
                {
                    "fold": result.get("fold"),
                    "valid_auc": best_metrics.get("valid_auc"),
                    "valid_acc": best_metrics.get("valid_acc"),
                    "best_test_auc": best_metrics.get("best_test_auc"),
                    "best_test_acc": best_metrics.get("best_test_acc"),
                    "last_test_auc": best_metrics.get("last_test_auc"),
                    "last_test_acc": best_metrics.get("last_test_acc"),
                    "best_epoch": best_metrics.get("epoch"),
                    "ckpt_dir": result.get("ckpt_dir"),
                    "best_path": result.get("best_path"),
                    "run_name": result.get("run_name"),
                }
            )


def print_cv_summary(aggregate, cv_dir):
    _print_metric(aggregate, "valid_auc", "CV valid_auc", style="green")
    _print_metric(aggregate, "valid_acc", "CV valid_acc", style="green")
    _print_metric(aggregate, "best_test_auc", "CV best_test_auc", style="green")
    _print_metric(aggregate, "best_test_acc", "CV best_test_acc", style="green")
    _print_metric(aggregate, "last_test_auc", "CV last_test_auc ", style="cyan")
    _print_metric(aggregate, "last_test_acc", "CV last_test_acc ", style="cyan")
    print("")
    print(f"CV summary saved to: [bold]{cv_dir}[/bold]")


def _print_metric(aggregate, key, label, style):
    if key not in aggregate:
        return
    mean_value = aggregate[key]["mean"]
    std_value = aggregate[key]["std"]
    if style == "green":
        print(f"[green][bold]{label} mean={mean_value:.6f} std={std_value:.6f}[/bold][/green]")
    else:
        print(f"[{style}]{label} mean={mean_value:.6f} std={std_value:.6f}[/{style}]")
