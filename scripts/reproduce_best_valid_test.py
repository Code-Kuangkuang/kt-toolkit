import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets  # noqa: F401
import models  # noqa: F401
import core.trainers  # noqa: F401
from core.factory import build_dataset, build_model, build_trainer
from core.train_runner import build_optimizer, set_seed


MODEL_CONFIG_EXCLUDE = {
    "loss_c_all_lambda",
    "loss_q_all_lambda",
    "loss_c_next_lambda",
    "loss_q_next_lambda",
    "output_mode",
    "output_c_all_lambda",
    "output_c_next_lambda",
    "output_q_all_lambda",
    "output_q_next_lambda",
    "emb_type",
    "learning_rate",
    "use_timestamps",
    "dpath",
    "num_at",
    "num_it",
    "booster_strategy",
    "require_fold_embedding",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint(run_dir, model_name, emb_type):
    candidates = [
        run_dir / f"{model_name}_{emb_type}_model.pt",
        run_dir / "qid_model.pt",
        run_dir / "model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    pt_files = sorted(run_dir.glob("*.pt"))
    if len(pt_files) == 1:
        return pt_files[0]
    raise FileNotFoundError(f"No unique checkpoint found in {run_dir}")


def reproduce_run(run_dir, device_arg):
    run_config = load_json(run_dir / "run_config.json")
    dataset_name = run_config["dataset_name"]
    model_name = run_config["model_name"]
    emb_type = run_config.get("emb_type") or run_config["model_config"].get("emb_type", "qid")
    fold = run_config["fold"]
    seed = run_config.get("seed", 3407)
    train_cfg = dict(run_config["train_config"])
    model_cfg = dict(run_config["model_config"])
    dataset_cfg = dict(run_config["dataset_config"])
    dpath = Path(dataset_cfg.get("dpath", ""))
    if not dpath.exists():
        local_dpath = ROOT / "data" / dataset_name
        if local_dpath.exists():
            dataset_cfg["dpath"] = str(local_dpath)
    data_config = {dataset_name: dataset_cfg}

    set_seed(seed)
    if device_arg == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device).index or 0)

    model_kwargs = {k: v for k, v in model_cfg.items() if k not in MODEL_CONFIG_EXCLUDE}
    model = build_model(
        model_name,
        num_c=dataset_cfg["num_c"],
        num_q=dataset_cfg["num_q"],
        emb_type=emb_type,
        seq_len=train_cfg.get("seq_len"),
        device=device,
        dpath=dataset_cfg.get("dpath", ""),
        num_at=model_cfg.get("num_at"),
        num_it=model_cfg.get("num_it"),
        **model_kwargs,
    ).to(device)

    use_timestamps = bool(train_cfg.get("use_timestamps", False) or model_cfg.get("use_timestamps", False))
    dataset_mode = train_cfg.get("dataset_mode")
    train_loader, valid_loader = build_dataset(
        "kt_default",
        dataset_name=dataset_name,
        data_config=data_config,
        fold=fold,
        batch_size=train_cfg["batch_size"],
        model_name=model_name,
        dataset_mode=dataset_mode,
        use_timestamps=use_timestamps,
    )
    test_loader = build_dataset(
        "kt_test",
        dataset_name=dataset_name,
        data_config=data_config,
        batch_size=train_cfg["batch_size"],
        model_name=model_name,
        dataset_mode=dataset_mode,
        use_timestamps=use_timestamps,
    )

    optimizer = build_optimizer(train_cfg, model_cfg, model)
    trainer = build_trainer(
        model_name,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        num_epochs=train_cfg["num_epochs"],
        device=device,
        hooks=[],
        other_config=model_cfg,
        test_loader=test_loader,
    )

    ckpt_path = resolve_checkpoint(run_dir, model_name, emb_type)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    trainer.model.load_state_dict(state)
    metrics = trainer.evaluate_test()

    saved_metrics_path = run_dir / "best_metrics.json"
    saved = load_json(saved_metrics_path) if saved_metrics_path.exists() else {}
    return {
        "fold": fold,
        "run_name": run_config.get("run_name", run_dir.name),
        "best_epoch": saved.get("epoch"),
        "test_auc": metrics.get("test_auc"),
        "test_acc": metrics.get("test_acc"),
        "saved_test_auc": saved.get("best_test_auc", saved.get("test_auc")),
        "saved_test_acc": saved.get("best_test_acc", saved.get("test_acc")),
        "ckpt_path": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce best-valid checkpoint test metrics.")
    parser.add_argument("--cv-dir", type=Path, required=True, help="Directory containing fold run directories.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    run_dirs = sorted([p for p in args.cv_dir.iterdir() if (p / "run_config.json").exists()])
    if not run_dirs:
        raise SystemExit(f"No fold run directories with run_config.json found under {args.cv_dir}")

    rows = [reproduce_run(run_dir, args.device) for run_dir in run_dirs]
    aucs = [float(r["test_auc"]) for r in rows]
    accs = [float(r["test_acc"]) for r in rows]

    for row in rows:
        print(
            f"fold={row['fold']} epoch={row['best_epoch']} "
            f"test_auc={row['test_auc']:.12f} test_acc={row['test_acc']:.12f}"
        )
    print(
        f"mean test_auc={statistics.mean(aucs):.12f} "
        f"std={statistics.pstdev(aucs):.12f}"
    )
    print(
        f"mean test_acc={statistics.mean(accs):.12f} "
        f"std={statistics.pstdev(accs):.12f}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
