import csv
import json
import os
import statistics

from rich import print


def aggregate_fold_metrics(fold_results):
    """Average each metric over the folds that reported it.

    Keys are the union across folds, so a metric can be missing from some of
    them -- most often because building the test loader failed for one fold and
    `train_one_fold` swallowed the error to keep the sweep alive. Each entry
    therefore carries `n` against `n_folds` and the ids it is missing, so a
    four-fold mean cannot be read as a five-fold one.
    """
    numeric_keys = set()
    for result in fold_results:
        best_metrics = result.get("best_metrics") or {}
        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    n_folds = len(fold_results)
    summary = {}
    for key in sorted(numeric_keys):
        values = []
        missing = []
        for result in fold_results:
            best_metrics = result.get("best_metrics") or {}
            value = best_metrics.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                missing.append(result.get("fold"))
        if not values:
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
            "n": len(values),
            "n_folds": n_folds,
            "missing_folds": missing,
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
                # The pyKT-comparable pair. Absent from this file until now,
                # which meant the one column you can line up against published
                # tables was missing from the artifact people actually open.
                "best_window_test_auc",
                "best_window_test_acc",
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
                    "best_window_test_auc": best_metrics.get("best_window_test_auc"),
                    "best_window_test_acc": best_metrics.get("best_window_test_acc"),
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
    # The pyKT-comparable pair: one row per position with a full history, rather
    # than non-overlapping chunks that leave boundary positions with almost none.
    _print_metric(aggregate, "best_window_test_auc", "CV window_test_auc", style="green")
    _print_metric(aggregate, "best_window_test_acc", "CV window_test_acc", style="green")
    _print_metric(aggregate, "last_test_auc", "CV last_test_auc ", style="cyan")
    _print_metric(aggregate, "last_test_acc", "CV last_test_acc ", style="cyan")
    _print_incomplete_warning(aggregate)
    print("")
    print(f"CV summary saved to: [bold]{cv_dir}[/bold]")


def _print_metric(aggregate, key, label, style):
    if key not in aggregate:
        return
    entry = aggregate[key]
    mean_value = entry["mean"]
    std_value = entry["std"]
    n = entry.get("n", len(entry.get("values", [])))
    n_folds = entry.get("n_folds", n)
    body = f"{label} mean={mean_value:.6f} std={std_value:.6f} n={n}/{n_folds}"
    if n < n_folds:
        # Averaging fewer folds than were run also narrows std, so the number
        # looks tighter at the same time as it becomes incomparable.
        print(f"[red][bold]{body}  <-- INCOMPLETE[/bold][/red]")
    elif style == "green":
        print(f"[green][bold]{body}[/bold][/green]")
    else:
        print(f"[{style}]{body}[/{style}]")


def _print_incomplete_warning(aggregate):
    incomplete = {
        key: entry
        for key, entry in aggregate.items()
        if entry.get("n", 0) < entry.get("n_folds", 0)
    }
    if not incomplete:
        return
    print("")
    print("[red][bold]WARNING: some metrics are averaged over fewer folds than were run.[/bold][/red]")
    for key in sorted(incomplete):
        entry = incomplete[key]
        missing = ", ".join(str(f) for f in entry.get("missing_folds", []))
        print(
            f"[red]  {key}: {entry['n']}/{entry['n_folds']} folds"
            f" (missing fold {missing})[/red]"
        )
    print("[red]  Most likely a test loader failed to build; check the fold logs for"
          " 'Could not build test loader'. Do not put these means in a table.[/red]")
