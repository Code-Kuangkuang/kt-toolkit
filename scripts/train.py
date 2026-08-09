import datetime
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Optional

import typer
from rich import print


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.trainers  # register trainers
import datasets.init_dataset  # register dataset builders
import models  # register models
from core.config import load_cfg
from core.dataset_names import normalize_dataset_name
from core.train_runner import aggregate_fold_metrics, print_cv_summary, save_cv_summary, train_one_fold

app = typer.Typer(add_completion=False)

def _parse_folds_spec(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return [0, 1, 2, 3, 4]
    if "-" in spec and "," not in spec:
        left, right = spec.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))
    folds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    return folds


def _resolve_existing_cv_dir(cv_run_dir: str, save_dir: str):
    raw = Path(cv_run_dir)
    if raw.is_absolute():
        candidates = [raw]
    else:
        save_root = Path(save_dir)
        if not save_root.is_absolute():
            save_root = ROOT / save_root
        candidates = [ROOT / raw, save_root / raw]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise typer.BadParameter(f"--cv-run-dir not found. Tried: {tried}")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _find_best_model_path(run_dir: Path):
    candidates = [p for p in run_dir.glob("*.pt") if p.name != "last_epoch_model.pt"]
    if not candidates:
        return None
    return str(sorted(candidates)[0])


def _load_completed_fold(cv_dir: Path, fold_id: int, dataset_name: str, model_name: str):
    for run_config_path in sorted(cv_dir.glob("*/run_config.json")):
        run_dir = run_config_path.parent
        best_metrics_path = run_dir / "best_metrics.json"
        if not best_metrics_path.exists():
            continue
        try:
            run_config = _load_json(run_config_path)
            best_metrics = _load_json(best_metrics_path)
        except Exception:
            continue
        if int(run_config.get("fold", -999)) != int(fold_id):
            continue
        if run_config.get("dataset_name") != dataset_name:
            continue
        if run_config.get("model_name") != model_name:
            continue
        return {
            "fold": fold_id,
            "run_name": run_config.get("run_name", run_dir.name),
            "ckpt_dir": str(run_dir),
            "emb_type": run_config.get("emb_type"),
            "best_metrics": best_metrics,
            "best_path": _find_best_model_path(run_dir),
            "skipped": True,
        }
    return None


@app.command()
def main(
    # Dataset options
    dataset_name: str = typer.Option(
        ...,
        "--dataset_name", "--dataset-name",
        help="Name of the dataset to use. E.g., assist2009, assist2015"
        ),

    # Model options
    model_name: str = typer.Option(
        "dkt",
        "--model_name", "--model-name",
        help="Name of the model to use. E.g., dkt, dkvmn"
        ),
    emb_type: Optional[str] = typer.Option(
        None,
        "--emb_type", "--emb-type",
        help="Embedding type (e.g., qid, iekt). If omitted, uses the model's default from kt_config."
        ),
    emb_size: Optional[int] = typer.Option(
        None,
        "--emb_size", "--emb-size",
        help="Embedding size for training"
        ),

    # Attention options
    d_model: Optional[int] = typer.Option(
        256,
        "--d_model", "--d-model",
        help="Dimension of the model"
        ),
    d_ff: Optional[int] = typer.Option(
        512,
        "--d_ff", "--d-ff",
        help="Dimension of the feed forward network"
        ),
    num_attn_heads: Optional[int] = typer.Option(
        8,
        "--num_attn_heads", "--num-attn-heads",
        help="Number of attention heads"
        ),
    n_blocks: Optional[int] = typer.Option(
        4,
        "--n_blocks", "--n-blocks",
        help="Number of transformer blocks"
        ),

    # Training options
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch_size", "--batch-size",
        help="Batch size for training"
        ),
    num_epochs: Optional[int] = typer.Option(
        None,
        "--num_epochs", "--num-epochs",
        help="Number of epochs for training"
        ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning_rate", "--learning-rate",
        help="Learning rate for training"
        ),
    dropout: Optional[float] = typer.Option(
        None,
        "--dropout",
        help="Dropout rate for training"
        ),
    patience: Optional[int] = typer.Option(
        None,
        "--patience",
        help="Early stopping patience. Use -1 to disable early stopping.",
        ),

    # Experiment options
    fold: int = typer.Option(
        0,
        "--fold",
        help="Fold number for cross-validation. [0-4]"
        ),
    cv: int = typer.Option(
        0,
        "--cv",
        help="Run cross-validation over multiple folds in one command (sequential).",
    ),
    folds: str = typer.Option(
        "0-4",
        "--folds",
        help="Folds to run when --cv=1. Examples: 0-4 or 0,1,3",
    ),
    cv_run_dir: Optional[str] = typer.Option(
        None,
        "--cv-run-dir",
        help="Existing CV directory to continue, e.g. saved_model/cv-assist2012-iekt-20260511-120000.",
    ),
    skip_completed: int = typer.Option(
        0,
        "--skip-completed",
        help="When --cv=1, skip folds that already have best_metrics.json in --cv-run-dir.",
    ),
    seed: int = typer.Option(
        3407,
        "--seed",
        help="Random seed for reproducibility"
        ),

    # GPU options
    gpu: int = typer.Option(
        0,
        "--gpu",
        help="GPU device number to use for training"
        ),

    # Save and logging options
    save_dir: str = typer.Option(
        "saved_model",
        "--save_dir", "--save-dir",
        help="Directory to save the model"
        ),
    use_wandb: int = typer.Option(
        1,
        "--use_wandb", "--use-wandb",
        help="Whether to use Weights and Biases for logging"
        ),
    add_uuid: int = typer.Option(
        0,
        "--add_uuid", "--add-uuid",
        help="Whether to add a unique identifier to the run name"
        ),

    # Config paths
    kt_config: str = typer.Option(
        "configs/kt_config.json",
        "--kt_config", "--kt-config",
        help="Path to the kt_config.json file"
        ),
    data_config_path: str = typer.Option(
        "configs/data_config.json",
        "--data_config", "--data-config",
        help="Path to the data_config.json file"
        ),
    wandb_config: str = typer.Option(
        "configs/wandb.json",
        "--wandb_config", "--wandb-config",
        help="Path to the wandb.json file"
        ),
):
    dataset_name = normalize_dataset_name(dataset_name)
    kt_cfg_raw = load_cfg(kt_config)
    data_config_raw = load_cfg(data_config_path)

    wandb_cfg = None
    if use_wandb == 1 and os.path.exists(wandb_config):
        wandb_cfg = load_cfg(wandb_config)

    def _train_one_fold(fold_id: int, save_root: str, cv_run_name: Optional[str] = None):
        overrides = {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "emb_size": emb_size,
            "dropout": dropout,
            "patience": patience,
        }
        return train_one_fold(
            dataset_name=dataset_name,
            model_name=model_name,
            emb_type=emb_type,
            fold_id=fold_id,
            root_dir=str(ROOT),
            kt_cfg_raw=kt_cfg_raw,
            data_config_raw=data_config_raw,
            seed=seed,
            save_root=save_root,
            add_uuid=add_uuid,
            wandb_cfg=wandb_cfg,
            kt_config_path=kt_config,
            data_config_path=data_config_path,
            wandb_config_path=wandb_config,
            cv_run_name=cv_run_name,
            overrides=overrides,
            gpu_id=gpu,
        )

    if cv == 1:
        fold_ids = _parse_folds_spec(folds)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if cv_run_dir:
            cv_dir_path = _resolve_existing_cv_dir(cv_run_dir, save_dir)
            cv_run_name = cv_dir_path.name
            cv_dir = str(cv_dir_path)
            print(f"[bold]Continuing CV run directory:[/bold] {cv_dir}")
        else:
            cv_run_name = f"cv-{dataset_name}-{model_name}-{ts}"
            if add_uuid == 1:
                cv_run_name = f"{cv_run_name}-{uuid.uuid4()}"
            cv_dir = os.path.join(save_dir, cv_run_name)
            cv_dir_path = Path(cv_dir)
            os.makedirs(cv_dir, exist_ok=True)

        fold_results = []
        for fid in fold_ids:
            if skip_completed == 1:
                completed = _load_completed_fold(cv_dir_path, fid, dataset_name, model_name)
                if completed is not None:
                    print(f"\n[yellow]===== CV Fold {fid} skipped: completed run found =====[/yellow]\n")
                    fold_results.append(completed)
                    continue
            print(f"\n[bold]===== CV Fold {fid} / {fold_ids} =====[/bold]\n")
            fold_results.append(_train_one_fold(fid, save_root=cv_dir, cv_run_name=cv_run_name))

        agg = aggregate_fold_metrics(fold_results)
        cv_payload = {
            "cv_run_name": cv_run_name,
            "timestamp": ts,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "emb_type": emb_type,
            "folds": fold_ids,
            "seed": seed,
            "save_dir": save_dir,
            "cv_dir": cv_dir,
            "per_fold": fold_results,
            "aggregate": agg,
        }
        save_cv_summary(cv_dir, cv_payload, fold_results)

        print_cv_summary(agg, cv_dir)
    else:
        _train_one_fold(fold, save_root=save_dir, cv_run_name=None)


if __name__ == "__main__":
    app()
