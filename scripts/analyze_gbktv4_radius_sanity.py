import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets.init_dataset  # noqa: F401
import models  # noqa: F401
from core.factory import build_dataset, build_model
from core.run_support import set_seed


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
    "lambda_item_difficulty",
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
    pt_files = sorted([p for p in run_dir.glob("*.pt") if p.name != "last_epoch_model.pt"])
    if len(pt_files) == 1:
        return pt_files[0]
    raise FileNotFoundError(f"No unique checkpoint found in {run_dir}")


def concat_full(seqs, shft):
    if seqs is None or seqs.numel() == 0:
        return None
    return torch.cat((seqs[:, :1], shft), dim=1)


def masked_numpy(value, mask):
    return torch.masked_select(value, mask).detach().cpu().numpy()


def safe_auc(target, score):
    if len(target) == 0 or len(np.unique(target)) < 2:
        return math.nan
    return float(metrics.roc_auc_score(target, score))


def corr_pair(x, y):
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return {"pearson": math.nan, "spearman": math.nan}
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def build_loaders(run_config, split, batch_size=None):
    dataset_name = run_config["dataset_name"]
    model_name = run_config["model_name"]
    fold = int(run_config["fold"])
    train_cfg = dict(run_config["train_config"])
    model_cfg = dict(run_config["model_config"])
    dataset_cfg = dict(run_config["dataset_config"])

    dpath = Path(dataset_cfg.get("dpath", ""))
    if not dpath.exists():
        local_dpath = ROOT / "data" / dataset_name
        if local_dpath.exists():
            dataset_cfg["dpath"] = str(local_dpath)

    data_config = {dataset_name: dataset_cfg}
    use_timestamps = bool(train_cfg.get("use_timestamps", False) or model_cfg.get("use_timestamps", False))
    dataset_mode = train_cfg.get("dataset_mode")
    bs = int(batch_size or train_cfg["batch_size"])

    if split in {"train", "valid"}:
        train_loader, valid_loader = build_dataset(
            "kt_default",
            dataset_name=dataset_name,
            data_config=data_config,
            fold=fold,
            batch_size=bs,
            model_name=model_name,
            dataset_mode=dataset_mode,
            use_timestamps=use_timestamps,
        )
        return [train_loader if split == "train" else valid_loader]

    if split == "test":
        return [
            build_dataset(
                "kt_test",
                dataset_name=dataset_name,
                data_config=data_config,
                batch_size=bs,
                model_name=model_name,
                dataset_mode=dataset_mode,
                use_timestamps=use_timestamps,
            )
        ]

    if split == "all":
        train_loader, valid_loader = build_dataset(
            "kt_default",
            dataset_name=dataset_name,
            data_config=data_config,
            fold=fold,
            batch_size=bs,
            model_name=model_name,
            dataset_mode=dataset_mode,
            use_timestamps=use_timestamps,
        )
        test_loader = build_dataset(
            "kt_test",
            dataset_name=dataset_name,
            data_config=data_config,
            batch_size=bs,
            model_name=model_name,
            dataset_mode=dataset_mode,
            use_timestamps=use_timestamps,
        )
        return [train_loader, valid_loader, test_loader]

    raise ValueError(f"Unknown split: {split}")


def build_run_model(run_dir, device):
    run_config = load_json(run_dir / "run_config.json")
    dataset_cfg = dict(run_config["dataset_config"])
    model_cfg = dict(run_config["model_config"])
    train_cfg = dict(run_config["train_config"])
    model_name = run_config["model_name"]
    emb_type = run_config.get("emb_type") or model_cfg.get("emb_type", "qid")

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

    ckpt_path = resolve_checkpoint(run_dir, model_name, emb_type)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return run_config, model


def collect_run(run_dir, split, device, batch_size=None):
    run_config, model = build_run_model(run_dir, device)
    loaders = build_loaders(run_config, split=split, batch_size=batch_size)
    fold = int(run_config["fold"])

    records = {
        "fold": [],
        "target": [],
        "y": [],
        "y_ball": [],
        "fusion_concept_weight": [],
        "r_h_mean": [],
        "r_d_mean": [],
    }

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                qseqs = batch.get("qseqs")
                cseqs = batch.get("cseqs")
                rseqs = batch["rseqs"]
                qshft = batch.get("shft_qseqs")
                cshft = batch.get("shft_cseqs")
                rshft = batch["shft_rseqs"]
                itseqs = batch.get("itseqs")
                itshft = batch.get("shft_itseqs")
                utseqs = batch.get("utseqs")
                utshft = batch.get("shft_utseqs")
                sm = batch["smasks"].to(device)

                qseqs = qseqs.to(device) if qseqs is not None else None
                cseqs = cseqs.to(device) if cseqs is not None else None
                rseqs = rseqs.to(device)
                qshft = qshft.to(device) if qshft is not None else None
                cshft = cshft.to(device) if cshft is not None else None
                rshft = rshft.to(device)
                itseqs = itseqs.to(device) if itseqs is not None else None
                itshft = itshft.to(device) if itshft is not None else None
                utseqs = utseqs.to(device) if utseqs is not None else None
                utshft = utshft.to(device) if utshft is not None else None

                q_full = concat_full(qseqs, qshft)
                c_full = concat_full(cseqs, cshft)
                r_full = concat_full(rseqs, rshft)
                it_full = concat_full(itseqs, itshft)
                ut_full = concat_full(utseqs, utshft)

                outputs = model(
                    q_full.long(),
                    c_full.long(),
                    r_full.float(),
                    it=it_full.long() if it_full is not None else None,
                    ut=ut_full.long() if ut_full is not None else None,
                )

                y = outputs["y"]
                common_len = min(y.size(1), rshft.size(1), sm.size(1))
                if common_len <= 0:
                    continue

                sm = sm[:, :common_len]
                target = rshft[:, :common_len].float()
                aligned = {
                    "target": target,
                    "y": outputs["y"][:, :common_len],
                    "y_ball": outputs["y_ball"][:, :common_len],
                    "fusion_concept_weight": outputs["fusion_concept_weight"][:, :common_len],
                    "r_h_mean": outputs["r_h_mean"][:, :common_len],
                    "r_d_mean": outputs["r_d_mean"][:, :common_len],
                }

                selected_target = masked_numpy(aligned["target"], sm)
                if len(selected_target) == 0:
                    continue
                records["fold"].append(np.full(len(selected_target), fold, dtype=np.int64))
                for key, value in aligned.items():
                    records[key].append(masked_numpy(value, sm))

    return {k: np.concatenate(v) if v else np.array([]) for k, v in records.items()}


def summarize(data, n_bins):
    df = pd.DataFrame(
        {
            "fold": data["fold"],
            "target": data["target"],
            "y": data["y"],
            "y_ball": data["y_ball"],
            "fusion_concept_weight": data["fusion_concept_weight"],
            "r_h_mean": data["r_h_mean"],
            "r_d_mean": data["r_d_mean"],
        }
    )
    df["r_sum"] = df["r_h_mean"] + df["r_d_mean"]

    corr = {
        radius_key: corr_pair(df[radius_key].to_numpy(), df["fusion_concept_weight"].to_numpy())
        for radius_key in ["r_h_mean", "r_d_mean", "r_sum"]
    }

    bins = []
    for radius_key in ["r_h_mean", "r_d_mean", "r_sum"]:
        qbin = pd.qcut(df[radius_key], q=n_bins, labels=False, duplicates="drop")
        tmp = df.assign(radius_bin=qbin)
        for bin_id, group in tmp.groupby("radius_bin", dropna=True):
            bins.append(
                {
                    "radius_key": radius_key,
                    "bin": int(bin_id),
                    "count": int(len(group)),
                    "radius_min": float(group[radius_key].min()),
                    "radius_mean": float(group[radius_key].mean()),
                    "radius_max": float(group[radius_key].max()),
                    "target_rate": float(group["target"].mean()),
                    "y_ball_mean": float(group["y_ball"].mean()),
                    "y_mean": float(group["y"].mean()),
                    "fusion_concept_weight_mean": float(group["fusion_concept_weight"].mean()),
                    "ball_auc": safe_auc(group["target"].to_numpy(), group["y_ball"].to_numpy()),
                    "final_auc": safe_auc(group["target"].to_numpy(), group["y"].to_numpy()),
                    "ball_error": float(np.abs(group["y_ball"] - group["target"]).mean()),
                    "final_error": float(np.abs(group["y"] - group["target"]).mean()),
                }
            )

    return df, corr, bins


def main():
    parser = argparse.ArgumentParser(description="Sanity check GBKTV4 radius/fusion semantics.")
    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=Path("saved_model/cv-assist2009-gbktv4-20260528-000048"),
        help="CV directory containing fold run directories.",
    )
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/radius_sanity"))
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device).index or 0)

    set_seed(3407)
    run_dirs = sorted([p for p in args.cv_dir.iterdir() if (p / "run_config.json").exists()])
    if not run_dirs:
        raise SystemExit(f"No run_config.json found under {args.cv_dir}")

    collected = []
    for run_dir in run_dirs:
        print(f"Collecting {run_dir.name} ({args.split})")
        collected.append(collect_run(run_dir, split=args.split, device=device, batch_size=args.batch_size))

    keys = collected[0].keys()
    data = {key: np.concatenate([item[key] for item in collected]) for key in keys}
    df, corr, bins = summarize(data, n_bins=args.bins)

    per_fold = {}
    for fold_id, group in df.groupby("fold"):
        per_fold[int(fold_id)] = {
            radius_key: corr_pair(group[radius_key].to_numpy(), group["fusion_concept_weight"].to_numpy())
            for radius_key in ["r_h_mean", "r_d_mean", "r_sum"]
        }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.cv_dir.name}_{args.split}"
    rows_path = out_dir / f"{stem}_rows.csv"
    bins_path = out_dir / f"{stem}_bins.csv"
    summary_path = out_dir / f"{stem}_summary.json"

    df.to_csv(rows_path, index=False)
    with open(bins_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bins[0].keys()))
        writer.writeheader()
        writer.writerows(bins)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_dir": str(args.cv_dir),
                "split": args.split,
                "count": int(len(df)),
                "pooled_correlation": corr,
                "per_fold_correlation": per_fold,
                "overall_auc": {
                    "ball_auc": safe_auc(df["target"].to_numpy(), df["y_ball"].to_numpy()),
                    "final_auc": safe_auc(df["target"].to_numpy(), df["y"].to_numpy()),
                },
                "mean_values": {
                    "fusion_concept_weight": float(df["fusion_concept_weight"].mean()),
                    "r_h_mean": float(df["r_h_mean"].mean()),
                    "r_d_mean": float(df["r_d_mean"].mean()),
                    "r_sum": float(df["r_sum"].mean()),
                },
                "outputs": {
                    "rows": str(rows_path),
                    "bins": str(bins_path),
                },
            },
            f,
            indent=2,
        )

    print("\nPooled correlation with fusion_concept_weight:")
    for radius_key, values in corr.items():
        print(
            f"  {radius_key:8s}: pearson={values['pearson']:.6f}, "
            f"spearman={values['spearman']:.6f}"
        )
    print("\nOverall:")
    print(f"  ball_auc={safe_auc(df['target'].to_numpy(), df['y_ball'].to_numpy()):.6f}")
    print(f"  final_auc={safe_auc(df['target'].to_numpy(), df['y'].to_numpy()):.6f}")
    print(f"  fusion_mean={df['fusion_concept_weight'].mean():.6f}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {bins_path}")
    print(f"Wrote {rows_path}")


if __name__ == "__main__":
    main()
