import datetime
import copy
import csv
import json
import os
import uuid
from pathlib import Path
import sys
import statistics
from typing import Optional

import typer
from rich import print

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.trainers  # register trainers
import datasets.init_dataset  # register dataset builders
import models  # register models
from core.config import load_cfg
from core.factory import build_dataset, build_model, build_trainer
from core.hooks import BestMetricsHook, SaveBestHook, WandbHook

app = typer.Typer(add_completion=False)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_run_config(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


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


def _aggregate_fold_metrics(fold_results):
    numeric_keys = set()
    for r in fold_results:
        bm = r.get("best_metrics") or {}
        for k, v in bm.items():
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


@app.command()
def main(
    # 数据集相关配置
    dataset_name: str = typer.Option(
        ..., 
        "--dataset_name", "--dataset-name",
        help="Name of the dataset to use. E.g., assist2009, assist2015"
        ),
    
    # 模型相关配置
    model_name: str = typer.Option(
        "dkt", 
        "--model_name", "--model-name",
        help="Name of the model to use. E.g., dkt, dkvmn"
        ),
    emb_type: str = typer.Option(
        "qid", 
        "--emb_type", "--emb-type",
        help="Type of embedding to use. E.g., qid"
        ),
    emb_size: Optional[int] = typer.Option(
        None, 
        "--emb_size", "--emb-size",
        help="Embedding size for training"
        ),
    
    # 注意力相关
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

    # 训练相关配置
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

    # 实验设置
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
    seed: int = typer.Option(
        3407, 
        "--seed",
        help="Random seed for reproducibility"
        ),
    
    # 保存和日志相关配置
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
        1, 
        "--add_uuid", "--add-uuid",
        help="Whether to add a unique identifier to the run name"
        ),

    # 配置文件路径
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
    kt_cfg_raw = load_cfg(kt_config)
    data_config_raw = load_cfg(data_config_path)

    wandb_cfg = None
    if use_wandb == 1 and os.path.exists(wandb_config):
        wandb_cfg = load_cfg(wandb_config)

    def _train_one_fold(fold_id: int, save_root: str, cv_run_name: Optional[str] = None):
        kt_cfg = copy.deepcopy(kt_cfg_raw)
        train_cfg_local = kt_cfg["train_config"]
        model_cfg_local = kt_cfg[model_name]

        if batch_size is not None:
            train_cfg_local["batch_size"] = batch_size
        if num_epochs is not None:
            train_cfg_local["num_epochs"] = num_epochs
        if learning_rate is not None:
            model_cfg_local["learning_rate"] = learning_rate
        if emb_size is not None:
            model_cfg_local["emb_size"] = emb_size
        if dropout is not None:
            model_cfg_local["dropout"] = dropout

        model_kwargs = {k: v for k, v in model_cfg_local.items() if k != "learning_rate"}

        data_config = copy.deepcopy(data_config_raw)
        dataset_cfg_local = data_config[dataset_name]
        dataset_cfg_local["dpath"] = os.path.normpath(os.path.join(ROOT, dataset_cfg_local["dpath"]))

        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_model(
            model_name,
            num_c=dataset_cfg_local["num_c"],
            num_q=dataset_cfg_local["num_q"],
            emb_type=emb_type,
            **model_kwargs,
        ).to(device)

        print(f"Training on device: [bold]{device}[/bold]")
        print(f"Model_Info: [green][bold]{model}[/bold][/green]")

        train_loader, valid_loader = build_dataset(
            "kt_default",
            dataset_name=dataset_name,
            data_config=data_config,
            fold=fold_id,
            batch_size=train_cfg_local["batch_size"],
        )

        optimizer_name = train_cfg_local.get("optimizer", "adam").lower()
        if optimizer_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=model_cfg_local["learning_rate"], momentum=0.9)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=model_cfg_local["learning_rate"])

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{dataset_name}-{model_name}-fold{fold_id}-{ts}"
        if add_uuid == 1:
            run_name = f"{run_name}-{uuid.uuid4()}"
        ckpt_dir = os.path.join(save_root, run_name)

        wandb_entity = None
        if wandb_cfg:
            wandb_entity = wandb_cfg.get("entity") or wandb_cfg.get("uid")

        run_config = {
            "run_name": run_name,
            "timestamp": ts,
            "save_dir": save_root,
            "ckpt_dir": ckpt_dir,
            "device": device,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "emb_type": emb_type,
            "fold": fold_id,
            "seed": seed,
            "use_wandb": bool(use_wandb),
            "add_uuid": bool(add_uuid),
            "cv": {
                "enabled": bool(cv),
                "cv_run_name": cv_run_name,
                "folds_spec": folds if cv else None,
            },
            "config_paths": {
                "kt_config": kt_config,
                "data_config": data_config_path,
                "wandb_config": wandb_config if wandb_cfg else None,
            },
            "train_config": train_cfg_local,
            "model_config": model_cfg_local,
            "dataset_config": dataset_cfg_local,
            "wandb": {
                "enabled": bool(wandb_cfg),
                "project": wandb_cfg.get("project") if wandb_cfg else None,
                "entity": wandb_entity,
                "mode": wandb_cfg.get("mode") if wandb_cfg else None,
            },
            "argv": sys.argv,
        }
        save_run_config(os.path.join(ckpt_dir, "run_config.json"), run_config)

        hooks = [
            BestMetricsHook(metric_key="valid_auc", mode="max"),
            SaveBestHook(save_dir=ckpt_dir, filename=f"{emb_type}_model.pt"),
        ]

        if wandb_cfg:
            wandb_kwargs = {
                "enabled": True,
                "run_name": run_name,
                "config": {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "fold": fold_id,
                    "train_config": train_cfg_local,
                    "model_config": model_cfg_local,
                    "cv_run_name": cv_run_name,
                },
                "project": wandb_cfg.get("project"),
                "entity": wandb_entity,
                "api_key": wandb_cfg.get("api_key"),
            }
            if cv_run_name is not None:
                wandb_kwargs["group"] = cv_run_name
            mode = wandb_cfg.get("mode")
            if mode is not None:
                wandb_kwargs["mode"] = mode
            hooks.append(WandbHook(**wandb_kwargs))

        trainer = build_trainer(
            model_name,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=opt,
            num_epochs=train_cfg_local["num_epochs"],
            device=device,
            hooks=hooks,
        )
        trainer.run()

        best_metrics = getattr(trainer, "best_metrics", None)
        if best_metrics is not None:
            save_run_config(os.path.join(ckpt_dir, "best_metrics.json"), best_metrics)

        return {
            "fold": fold_id,
            "run_name": run_name,
            "ckpt_dir": ckpt_dir,
            "best_metrics": best_metrics,
            "best_path": getattr(trainer, "best_path", None),
        }

    if cv == 1:
        fold_ids = _parse_folds_spec(folds)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        cv_run_name = f"cv-{dataset_name}-{model_name}-{ts}"
        if add_uuid == 1:
            cv_run_name = f"{cv_run_name}-{uuid.uuid4()}"
        cv_dir = os.path.join(save_dir, cv_run_name)
        os.makedirs(cv_dir, exist_ok=True)

        fold_results = []
        for fid in fold_ids:
            print(f"\n[bold]===== CV Fold {fid} / {fold_ids} =====[/bold]\n")
            fold_results.append(_train_one_fold(fid, save_root=cv_dir, cv_run_name=cv_run_name))

        agg = _aggregate_fold_metrics(fold_results)
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
        save_run_config(os.path.join(cv_dir, "cv_summary.json"), cv_payload)

        csv_path = os.path.join(cv_dir, "cv_summary.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["fold", "valid_auc", "valid_acc", "best_epoch", "ckpt_dir", "best_path", "run_name"],
            )
            writer.writeheader()
            for r in fold_results:
                bm = r.get("best_metrics") or {}
                writer.writerow(
                    {
                        "fold": r.get("fold"),
                        "valid_auc": bm.get("valid_auc"),
                        "valid_acc": bm.get("valid_acc"),
                        "best_epoch": bm.get("epoch"),
                        "ckpt_dir": r.get("ckpt_dir"),
                        "best_path": r.get("best_path"),
                        "run_name": r.get("run_name"),
                    }
                )

        if "valid_auc" in agg:
            m = agg["valid_auc"]["mean"]
            s = agg["valid_auc"]["std"]
            print(f"\n[green][bold]CV valid_auc mean={m:.6f} std={s:.6f}[/bold][/green]")
        if "valid_acc" in agg:
            m = agg["valid_acc"]["mean"]
            s = agg["valid_acc"]["std"]
            print(f"[green][bold]CV valid_acc mean={m:.6f} std={s:.6f}[/bold][/green]\n")
        print(f"CV summary saved to: [bold]{cv_dir}[/bold]")
    else:
        _train_one_fold(fold, save_root=save_dir, cv_run_name=None)


if __name__ == "__main__":
    app()
