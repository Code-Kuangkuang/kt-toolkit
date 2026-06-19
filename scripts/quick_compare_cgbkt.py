import copy
import csv
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Optional

import typer
from rich import print


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.trainers  # noqa: F401
import datasets.init_dataset  # noqa: F401
import models  # noqa: F401
from core.config import load_cfg
from core.train_runner import train_one_fold


app = typer.Typer(add_completion=False)


GBKT_BASE_MODEL_UPDATES = {
    "use_concept_readout": False,
    "history_mode": "none",
    "use_sequence_distance_attention": False,
    "use_time_distance_attention": False,
    "use_time_aware_kab": False,
    "use_time_forgetting": False,
    "use_timestamps": False,
    "use_dynamic_fusion": False,
    "max_concept_fusion_weight": 0.0,
    "use_scalar_item_difficulty": False,
    "lambda_item_difficulty": 0.0,
    "lambda_ball": 0.0,
    "lambda_concept_next": 0.0,
    "use_bbp_radius_normalization": True,
    "use_radius_discrimination": True,
    "use_kab_radius_features": True,
    "use_radius_state_update": True,
    "use_point_space": False,
    "response_function": "irt",
}


CANDIDATES = {
    "cgbkt": {
        "model_name": "cgbkt",
        "title": "CGBKT",
        "model_updates": {},
        "train_updates": {"use_timestamps": False},
    },
    "gbktv5": {
        "model_name": "cgbkt",
        "title": "CGBKT",
        "model_updates": {},
        "train_updates": {"use_timestamps": False},
    },
    "gbktv4": {
        "model_name": "gbktv4",
        "title": "GBKTV4 Full",
        "model_updates": {},
        "train_updates": {},
    },
    "gbkt_final": {
        "model_name": "gbkt_final",
        "title": "GBKTFinal Slim",
        "model_updates": {},
        "train_updates": {},
    },
    "gbkt-base": {
        "model_name": "gbktv4",
        "title": "GBKT-Base",
        "model_updates": GBKT_BASE_MODEL_UPDATES,
        "train_updates": {"use_timestamps": False},
    },
    "base-point": {
        "model_name": "gbktv4",
        "title": "GBKT-Base Point",
        "model_updates": {**GBKT_BASE_MODEL_UPDATES, "use_point_space": True},
        "train_updates": {"use_timestamps": False},
    },
}


DATASET_REFERENCE_DIR = {
    "assist2017": ("assist2017", "cv-assist2017-gbktv4-*"),
    "nips_task34": ("nips34", "cv-nips_task34-gbktv4-*"),
    "nips34": ("nips34", "cv-nips_task34-gbktv4-*"),
    "peiyou": ("peiyou", "cv-peiyou-gbktv4-*"),
}


def parse_folds(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return [0]
    if "-" in spec and "," not in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def normalize_candidate(name: str):
    key = name.strip().lower()
    aliases = {
        "base": "gbkt-base",
        "ball-base": "gbkt-base",
        "point": "base-point",
        "coverage": "cgbkt",
        "gbkt-coverage": "cgbkt",
        "gbkt-final": "gbkt_final",
        "final": "gbkt_final",
        "slim": "gbkt_final",
        "gbkt-slim": "gbkt_final",
    }
    key = aliases.get(key, key)
    if key not in CANDIDATES:
        raise typer.BadParameter(f"Unknown candidate '{name}'. Known: {', '.join(CANDIDATES)}")
    return key


def apply_candidate_config(kt_cfg_raw, candidate_key):
    spec = CANDIDATES[candidate_key]
    kt_cfg = copy.deepcopy(kt_cfg_raw)
    model_name = spec["model_name"]
    if model_name not in kt_cfg:
        raise typer.BadParameter(f"Model '{model_name}' not found in kt_config.")
    kt_cfg[model_name].update(spec.get("model_updates", {}))
    kt_cfg["train_config"].update(spec.get("train_updates", {}))
    return kt_cfg


def metric_value(result, key):
    metrics = result.get("best_metrics") or {}
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def summarize(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def fmt_metric(summary):
    if summary["mean"] is None:
        return ""
    return f"{summary['mean']:.6f}±{summary['std']:.6f}"


def latest_reference_summary(dataset_name: str, train_model_root: Path):
    key = dataset_name.lower()
    if key not in DATASET_REFERENCE_DIR:
        return None
    folder, pattern = DATASET_REFERENCE_DIR[key]
    root = train_model_root / folder
    matches = [p for p in root.glob(pattern) if p.is_dir()]
    if not matches:
        return None
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    csv_path = latest / "cv_summary.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    valid_auc = [float(r["valid_auc"]) for r in rows if r.get("valid_auc")]
    test_auc = [float(r["test_auc"]) for r in rows if r.get("test_auc") and float(r["test_auc"]) >= 0]
    return {
        "candidate": "gbktv4_existing",
        "title": "Existing GBKTV4 Full",
        "model_name": "gbktv4",
        "folds": ",".join(r.get("fold", "") for r in rows),
        "num_epochs": "",
        "elapsed_seconds": "",
        "valid_auc": fmt_metric(summarize(valid_auc)),
        "test_auc": fmt_metric(summarize(test_auc)),
        "valid_auc_mean": summarize(valid_auc)["mean"],
        "test_auc_mean": summarize(test_auc)["mean"],
        "output_dir": str(latest),
    }


@app.command()
def main(
    dataset_name: str = typer.Option("assist2017", "--dataset_name", "--dataset-name"),
    candidates: str = typer.Option(
        "cgbkt,gbktv4",
        "--candidates",
        help="Comma-separated candidates: cgbkt, gbkt_final, gbktv4, gbkt-base, base-point.",
    ),
    folds: str = typer.Option("0", "--folds", help="Examples: 0, 0-2, 0,1,3."),
    num_epochs: int = typer.Option(20, "--num_epochs", "--num-epochs"),
    patience: Optional[int] = typer.Option(5, "--patience"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size", "--batch-size"),
    learning_rate: Optional[float] = typer.Option(None, "--learning_rate", "--learning-rate"),
    gpu: int = typer.Option(0, "--gpu"),
    seed: int = typer.Option(3407, "--seed"),
    save_dir: str = typer.Option("saved_model", "--save_dir", "--save-dir"),
    use_wandb: int = typer.Option(0, "--use_wandb", "--use-wandb"),
    kt_config: str = typer.Option("configs/kt_config.json", "--kt_config", "--kt-config"),
    data_config_path: str = typer.Option("configs/data_config.json", "--data_config", "--data-config"),
    wandb_config: str = typer.Option("configs/wandb.json", "--wandb_config", "--wandb-config"),
    include_existing_reference: int = typer.Option(
        1,
        "--include-existing-reference",
        help="Add existing train_model GBKTV4 summary when available.",
    ),
    train_model_root: str = typer.Option(
        r"E:\project\knowledgeTracing\train_model",
        "--train-model-root",
    ),
    dry_run: int = typer.Option(0, "--dry-run"),
):
    kt_cfg_raw = load_cfg(kt_config)
    data_config_raw = load_cfg(data_config_path)
    candidate_keys = [normalize_candidate(item) for item in candidates.split(",") if item.strip()]
    fold_ids = parse_folds(folds)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(save_dir) / f"quick-cgbkt-compare-{dataset_name}-{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    wandb_cfg = None
    if use_wandb == 1 and os.path.exists(wandb_config):
        wandb_cfg = load_cfg(wandb_config)

    overrides = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "patience": patience,
    }

    print(f"[bold]Quick CGBKT comparison[/bold]")
    print(f"Dataset: {dataset_name}")
    print(f"Candidates: {', '.join(candidate_keys)}")
    print(f"Folds: {fold_ids}")
    print(f"Epochs: {num_epochs}, patience: {patience}")
    print(f"Output: {out_root}")

    rows = []
    payload = {
        "dataset_name": dataset_name,
        "timestamp": ts,
        "folds": fold_ids,
        "num_epochs": num_epochs,
        "patience": patience,
        "candidates": [],
    }

    for candidate_key in candidate_keys:
        spec = CANDIDATES[candidate_key]
        model_name = spec["model_name"]
        candidate_dir = out_root / candidate_key
        candidate_dir.mkdir(parents=True, exist_ok=True)

        if dry_run == 1:
            print(f"[yellow]Dry run:[/yellow] {candidate_key} -> model_name={model_name}, save={candidate_dir}")
            continue

        fold_results = []
        start = time.perf_counter()
        kt_cfg_candidate = apply_candidate_config(kt_cfg_raw, candidate_key)
        cv_run_name = f"{out_root.name}-{candidate_key}"

        for fid in fold_ids:
            print(f"\n[bold magenta]===== {candidate_key} fold {fid} =====[/bold magenta]")
            result = train_one_fold(
                dataset_name=dataset_name,
                model_name=model_name,
                emb_type=None,
                fold_id=fid,
                root_dir=str(ROOT),
                kt_cfg_raw=kt_cfg_candidate,
                data_config_raw=data_config_raw,
                seed=seed,
                save_root=str(candidate_dir),
                add_uuid=0,
                wandb_cfg=wandb_cfg,
                kt_config_path=kt_config,
                data_config_path=data_config_path,
                wandb_config_path=wandb_config,
                cv_run_name=cv_run_name,
                overrides=overrides,
                gpu_id=gpu,
            )
            fold_results.append(result)

        elapsed = time.perf_counter() - start
        valid_auc = summarize([metric_value(r, "valid_auc") for r in fold_results])
        test_auc = summarize([metric_value(r, "test_auc") for r in fold_results])
        valid_acc = summarize([metric_value(r, "valid_acc") for r in fold_results])
        test_acc = summarize([metric_value(r, "test_acc") for r in fold_results])

        row = {
            "candidate": candidate_key,
            "title": spec["title"],
            "model_name": model_name,
            "folds": ",".join(str(f) for f in fold_ids),
            "num_epochs": num_epochs,
            "elapsed_seconds": f"{elapsed:.3f}",
            "valid_auc": fmt_metric(valid_auc),
            "test_auc": fmt_metric(test_auc),
            "valid_acc": fmt_metric(valid_acc),
            "test_acc": fmt_metric(test_acc),
            "valid_auc_mean": valid_auc["mean"],
            "test_auc_mean": test_auc["mean"],
            "output_dir": str(candidate_dir),
        }
        rows.append(row)
        payload["candidates"].append(
            {
                **row,
                "model_updates": spec.get("model_updates", {}),
                "train_updates": spec.get("train_updates", {}),
                "per_fold": fold_results,
            }
        )

    if include_existing_reference == 1:
        ref = latest_reference_summary(dataset_name, Path(train_model_root))
        if ref is not None:
            rows.append(ref)
            payload["existing_reference"] = ref

    if not rows:
        return

    csv_path = out_root / "quick_compare_summary.csv"
    fieldnames = [
        "candidate",
        "title",
        "model_name",
        "folds",
        "num_epochs",
        "elapsed_seconds",
        "valid_auc",
        "test_auc",
        "valid_acc",
        "test_acc",
        "valid_auc_mean",
        "test_auc_mean",
        "output_dir",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    json_path = out_root / "quick_compare_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print("\n[bold]Summary[/bold]")
    for row in rows:
        print(
            f"{row['candidate']:>16s}  valid_auc={row.get('valid_auc', '')}  "
            f"test_auc={row.get('test_auc', '')}  seconds={row.get('elapsed_seconds', '')}"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    app()
