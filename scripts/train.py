import datetime
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
        0, 
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
        overrides = {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "emb_size": emb_size,
            "dropout": dropout,
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
        )

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
