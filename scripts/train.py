import datetime
import os
import uuid
from pathlib import Path
import sys
from typing import Optional

import typer
from rich import print

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.trainers  # register trainers
import datasets.init_dataset  # register dataset builders
import models.dkt  # register models
from core.config import load_cfg
from core.factory import build_dataset, build_model, build_trainer
from core.hooks import SaveBestHook, WandbHook

app = typer.Typer(add_completion=False)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    # 加载配置文件
    kt_cfg = load_cfg(kt_config)
    train_cfg = kt_cfg["train_config"]
    model_cfg = kt_cfg[model_name]

    if batch_size is not None:
        train_cfg["batch_size"] = batch_size
    if num_epochs is not None:
        train_cfg["num_epochs"] = num_epochs
    if learning_rate is not None:
        model_cfg["learning_rate"] = learning_rate
    if emb_size is not None:
        model_cfg["emb_size"] = emb_size
    if dropout is not None:
        model_cfg["dropout"] = dropout

    data_config = load_cfg(data_config_path)
    dataset_cfg = data_config[dataset_name]
    dataset_cfg["dpath"] = os.path.normpath(os.path.join(ROOT, dataset_cfg["dpath"]))

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构建模型
    model = build_model(
        model_name,
        num_c=dataset_cfg["num_c"],
        emb_size=model_cfg["emb_size"],
        dropout=model_cfg.get("dropout", 0.1),
        emb_type=emb_type,
    ).to(device)

    print(f"Training on device: [bold]{device}[/bold]")
    print(f"Model_Info: [green][bold]{model}[/bold][/green]")


    # 构建数据集
    train_loader, valid_loader = build_dataset(
        "kt_default",
        dataset_name=dataset_name,
        data_config=data_config,
        fold=fold,
        batch_size=train_cfg["batch_size"],
    )

    # 构建优化器
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=model_cfg["learning_rate"], momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"])

    # 设置保存目录和日志
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dataset_name}-{model_name}-fold{fold}-{ts}"
    if add_uuid == 1:
        run_name = f"{run_name}-{uuid.uuid4()}"
    ckpt_dir = os.path.join(save_dir, run_name)
    hooks = [SaveBestHook(save_dir=ckpt_dir, filename=f"{emb_type}_model.pt")]
    if use_wandb == 1 and os.path.exists(wandb_config):
        wandb_cfg = load_cfg(wandb_config)
        wandb_kwargs = {
            "enabled": True,
            "run_name": run_name,
            "config": {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "fold": fold,
                "train_config": train_cfg,
                "model_config": model_cfg,
            },
            "project": wandb_cfg.get("project"),
            "entity": wandb_cfg.get("entity") or wandb_cfg.get("uid"),
            "api_key": wandb_cfg.get("api_key"),
        }
        mode = wandb_cfg.get("mode")
        if mode is not None:
            wandb_kwargs["mode"] = mode
        hooks.append(WandbHook(**wandb_kwargs))

    # 构建Trainer并开始训练
    trainer = build_trainer(
        "dkt",
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=opt,
        num_epochs=train_cfg["num_epochs"],
        device=device,
        hooks=hooks,
    )
    trainer.run()


if __name__ == "__main__":
    app()
