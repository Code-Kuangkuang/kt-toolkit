import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
    "lambda_ball",
    "lambda_concept_next",
    "lambda_theta",
    "lambda_radius",
    "lambda_conf",
    "lambda_item_difficulty",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_run_dirs(run_dir: Path):
    run_dir = run_dir.resolve()
    if (run_dir / "run_config.json").exists():
        return [run_dir]

    direct_children = sorted([p for p in run_dir.iterdir() if (p / "run_config.json").exists()])
    if direct_children:
        return direct_children

    nested = sorted(run_dir.rglob("run_config.json"))
    return [p.parent for p in nested]


def resolve_checkpoint(run_dir: Path, model_name: str, emb_type: str):
    candidates = [
        run_dir / f"{model_name}_{emb_type}_model.pt",
        run_dir / f"{emb_type}_model.pt",
        run_dir / "model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    pt_files = sorted([p for p in run_dir.glob("*.pt") if p.name != "last_epoch_model.pt"])
    if len(pt_files) == 1:
        return pt_files[0]
    if len(pt_files) > 1:
        return pt_files[0]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def torch_load_weights(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def concat_full(seqs, shft):
    if seqs is None or shft is None or seqs.numel() == 0:
        return None
    return torch.cat((seqs[:, :1], shft), dim=1)


def safe_auc(target, score):
    target = np.asarray(target)
    score = np.asarray(score)
    mask = np.isfinite(target) & np.isfinite(score)
    target = target[mask]
    score = score[mask]
    if len(target) == 0 or len(np.unique(target)) < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(target, score))
    except Exception:
        return math.nan


def corr_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return {"count": int(len(x)), "pearson": math.nan, "spearman": math.nan}

    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"count": int(len(x)), "pearson": pearson, "spearman": spearman}


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def build_run_model(run_dir: Path, device):
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
    state = torch_load_weights(ckpt_path, device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        print(f"Strict checkpoint load failed for {run_dir.name}: {exc}")
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        print(
            "Loaded with strict=False. "
            f"missing_keys={missing[:8]}{'...' if len(missing) > 8 else ''}; "
            f"unexpected_keys={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    model.eval()
    return run_config, model, ckpt_path


def build_loaders(run_config, split: str, batch_size=None):
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
    bs = int(batch_size or train_cfg.get("batch_size", 64))

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
        return [(split, train_loader if split == "train" else valid_loader)]

    if split == "test":
        test_loader = build_dataset(
            "kt_test",
            dataset_name=dataset_name,
            data_config=data_config,
            batch_size=bs,
            model_name=model_name,
            dataset_mode=dataset_mode,
            use_timestamps=use_timestamps,
        )
        return [("test", test_loader)]

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
        return [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]

    raise ValueError(f"Unknown split: {split}")


def forward_with_geometry(model, q_full, c_full, r_full, it_full=None, ut_full=None):
    if not hasattr(model, "ball_to_ball_predict"):
        outputs = model(q_full.long(), c_full.long(), r_full.float(), it=it_full, ut=ut_full)
        return outputs, {}

    captured = []
    original = model.ball_to_ball_predict

    def wrapped(*args, **kwargs):
        pred = original(*args, **kwargs)
        needed = {"mu_h_p", "mu_d_p", "radius_sum"}
        if needed.issubset(pred.keys()):
            center_diff = pred["mu_h_p"] - pred["mu_d_p"]
            radius_sum = pred["radius_sum"].clamp_min(1e-6)
            captured.append(
                {
                    "center_dist": torch.linalg.norm(center_diff, dim=-1).detach(),
                    "radius_sum_mean": radius_sum.mean(dim=-1).detach(),
                    "normalized_dist": torch.linalg.norm(center_diff / radius_sum, dim=-1).detach(),
                    "overlap_margin": (radius_sum.mean(dim=-1) - torch.linalg.norm(center_diff, dim=-1)).detach(),
                }
            )
        return pred

    model.ball_to_ball_predict = wrapped
    try:
        outputs = model(
            q_full.long(),
            c_full.long(),
            r_full.float(),
            it=it_full.long() if it_full is not None else None,
            ut=ut_full.long() if ut_full is not None else None,
        )
    finally:
        model.ball_to_ball_predict = original

    if not captured:
        return outputs, {}
    geometry = {key: torch.stack([step[key] for step in captured], dim=1) for key in captured[0]}
    return outputs, geometry


def masked_np(value, mask, fill_value=np.nan):
    n = int(mask.sum().item())
    if value is None:
        return np.full(n, fill_value)
    return torch.masked_select(value, mask).detach().cpu().numpy()


def align_optional(outputs, geometry, key, common_len, like):
    if key in outputs:
        return outputs[key][:, :common_len]
    if key in geometry:
        return geometry[key][:, :common_len]
    return torch.full_like(like, float("nan"))


def concept_features(c_next, common_len, sm):
    if c_next is None:
        like = sm[:, :common_len].long()
        return torch.full_like(like, -1), torch.zeros_like(like)
    c_next = c_next[:, :common_len]
    if c_next.dim() == 3:
        first_concept = c_next[..., 0]
        concept_count = (c_next >= 0).sum(dim=-1)
        return first_concept, concept_count
    concept_count = (c_next >= 0).long()
    return c_next, concept_count


def collect_run(run_dir: Path, split: str, device, batch_size=None, max_batches=None):
    run_config, model, ckpt_path = build_run_model(run_dir, device)
    loaders = build_loaders(run_config, split=split, batch_size=batch_size)
    fold = int(run_config["fold"])

    frames = []
    with torch.no_grad():
        for split_name, loader in loaders:
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

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

                outputs, geometry = forward_with_geometry(model, q_full, c_full, r_full, it_full, ut_full)
                y = outputs["y"]
                common_len = min(y.size(1), rshft.size(1), sm.size(1))
                if common_len <= 0:
                    continue

                sm = sm[:, :common_len].bool()
                target = rshft[:, :common_len].float()
                sm = sm & (target >= 0)
                n = int(sm.sum().item())
                if n == 0:
                    continue

                pred = outputs["y"][:, :common_len]
                pred_ball = outputs.get("y_ball", outputs["y"])[:, :common_len]
                confidence = outputs.get("confidence")
                theta = outputs.get("theta")
                r_h_mean = outputs.get("r_h_mean")
                r_d_mean = outputs.get("r_d_mean")
                fusion_concept_weight = outputs.get("fusion_concept_weight")

                q_next = qshft[:, :common_len] if qshft is not None else None
                first_concept, concept_count = concept_features(cshft, common_len, sm)
                history_len = torch.arange(1, common_len + 1, device=device).unsqueeze(0).expand_as(sm)
                position = history_len - 1

                pred_clamped = pred.clamp(1e-5, 1.0 - 1e-5)
                bce = -(target * torch.log(pred_clamped) + (1.0 - target) * torch.log(1.0 - pred_clamped))
                abs_error = torch.abs(pred - target)
                cls_error = ((pred >= 0.5).float() != target).float()

                like = pred
                frame = pd.DataFrame(
                    {
                        "run_name": np.repeat(run_config["run_name"], n),
                        "checkpoint": np.repeat(str(ckpt_path), n),
                        "split": np.repeat(split_name, n),
                        "fold": np.repeat(fold, n),
                        "batch_idx": np.repeat(batch_idx, n),
                        "position": masked_np(position.float(), sm),
                        "history_len": masked_np(history_len.float(), sm),
                        "q_id": masked_np(q_next.float() if q_next is not None else None, sm, fill_value=-1).astype(np.int64),
                        "first_concept_id": masked_np(first_concept.float(), sm, fill_value=-1).astype(np.int64),
                        "concept_count": masked_np(concept_count.float(), sm),
                        "target": masked_np(target, sm),
                        "pred": masked_np(pred, sm),
                        "pred_ball": masked_np(pred_ball, sm),
                        "theta": masked_np(theta[:, :common_len] if theta is not None else None, sm),
                        "confidence": masked_np(
                            confidence[:, :common_len] if confidence is not None else None,
                            sm,
                        ),
                        "fusion_concept_weight": masked_np(
                            fusion_concept_weight[:, :common_len] if fusion_concept_weight is not None else None,
                            sm,
                        ),
                        "r_h_mean": masked_np(r_h_mean[:, :common_len] if r_h_mean is not None else None, sm),
                        "r_d_mean": masked_np(r_d_mean[:, :common_len] if r_d_mean is not None else None, sm),
                        "center_dist": masked_np(align_optional(outputs, geometry, "center_dist", common_len, like), sm),
                        "radius_sum_mean": masked_np(
                            align_optional(outputs, geometry, "radius_sum_mean", common_len, like),
                            sm,
                        ),
                        "normalized_dist": masked_np(
                            align_optional(outputs, geometry, "normalized_dist", common_len, like),
                            sm,
                        ),
                        "overlap_margin": masked_np(
                            align_optional(outputs, geometry, "overlap_margin", common_len, like),
                            sm,
                        ),
                        "abs_error": masked_np(abs_error, sm),
                        "bce": masked_np(bce, sm),
                        "classification_error": masked_np(cls_error, sm),
                    }
                )
                frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def qbin_summary(df: pd.DataFrame, column: str, n_bins: int):
    valid = df[np.isfinite(df[column])].copy()
    if valid.empty or valid[column].nunique() < 2:
        return pd.DataFrame()
    valid["bin"] = pd.qcut(valid[column], q=n_bins, labels=False, duplicates="drop")
    grouped = valid.groupby("bin", dropna=True)
    rows = []
    for bin_id, group in grouped:
        rows.append(
            {
                "feature": column,
                "bin": int(bin_id),
                "count": int(len(group)),
                "feature_min": float(group[column].min()),
                "feature_mean": float(group[column].mean()),
                "feature_max": float(group[column].max()),
                "correct_rate": float(group["target"].mean()),
                "pred_mean": float(group["pred"].mean()),
                "abs_error": float(group["abs_error"].mean()),
                "bce": float(group["bce"].mean()),
                "classification_error": float(group["classification_error"].mean()),
                "confidence": float(group["confidence"].mean()) if "confidence" in group else math.nan,
                "r_h_mean": float(group["r_h_mean"].mean()) if "r_h_mean" in group else math.nan,
                "r_d_mean": float(group["r_d_mean"].mean()) if "r_d_mean" in group else math.nan,
            }
        )
    return pd.DataFrame(rows)


def history_bucket_summary(df: pd.DataFrame):
    valid = df[np.isfinite(df["history_len"])].copy()
    if valid.empty:
        return pd.DataFrame()
    bins = [0, 5, 10, 20, 50, 100, np.inf]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", ">100"]
    valid["history_bucket"] = pd.cut(valid["history_len"], bins=bins, labels=labels, right=True)
    grouped = valid.groupby("history_bucket", observed=True)
    return grouped.agg(
        count=("target", "size"),
        history_len_mean=("history_len", "mean"),
        r_h_mean=("r_h_mean", "mean"),
        correct_rate=("target", "mean"),
        abs_error=("abs_error", "mean"),
        bce=("bce", "mean"),
        confidence=("confidence", "mean"),
    ).reset_index()


def item_summary(df: pd.DataFrame, min_item_count: int):
    valid = df[df["q_id"] >= 0].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()
    item_df = valid.groupby("q_id").agg(
        count=("target", "size"),
        correct_rate=("target", "mean"),
        pred_mean=("pred", "mean"),
        r_d_mean=("r_d_mean", "mean"),
        concept_count=("concept_count", "mean"),
        abs_error=("abs_error", "mean"),
        bce=("bce", "mean"),
    ).reset_index()
    item_df["incorrect_rate"] = 1.0 - item_df["correct_rate"]
    item_df["response_entropy"] = binary_entropy(item_df["correct_rate"].to_numpy())
    corr_df = []
    frequent = item_df[item_df["count"] >= min_item_count]
    pairs = [
        ("r_d_mean", "response_entropy"),
        ("r_d_mean", "incorrect_rate"),
        ("r_d_mean", "concept_count"),
        ("r_d_mean", "count"),
    ]
    for x, y in pairs:
        values = corr_pair(frequent[x].to_numpy(), frequent[y].to_numpy())
        corr_df.append(
            {
                "level": "item",
                "x": x,
                "y": y,
                "min_item_count": int(min_item_count),
                **values,
            }
        )
    return item_df, pd.DataFrame(corr_df)


def correlation_summary(df: pd.DataFrame, item_corr_df: pd.DataFrame):
    pairs = [
        ("pred_ball", "target"),
        ("pred_ball", "pred"),
        ("pred_ball", "abs_error"),
        ("theta", "target"),
        ("theta", "pred"),
        ("theta", "abs_error"),
        ("r_h_mean", "abs_error"),
        ("r_h_mean", "bce"),
        ("r_h_mean", "classification_error"),
        ("r_h_mean", "confidence"),
        ("history_len", "r_h_mean"),
        ("normalized_dist", "target"),
        ("normalized_dist", "pred"),
        ("normalized_dist", "abs_error"),
        ("center_dist", "target"),
        ("overlap_margin", "target"),
    ]
    rows = []
    for x, y in pairs:
        if x not in df or y not in df:
            continue
        values = corr_pair(df[x].to_numpy(), df[y].to_numpy())
        rows.append({"level": "position", "x": x, "y": y, **values})
    corr_df = pd.DataFrame(rows)
    if item_corr_df is not None and not item_corr_df.empty:
        corr_df = pd.concat([corr_df, item_corr_df], ignore_index=True)
    return corr_df


def try_plot(out_dir: Path, bin_df: pd.DataFrame, history_df: pd.DataFrame, item_df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is unavailable; skip plots: {exc}")
        return []

    paths = []

    def save_current(name):
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()
        paths.append(str(path))

    rh = bin_df[bin_df["feature"] == "r_h_mean"].copy()
    if not rh.empty:
        plt.figure(figsize=(5.2, 3.4))
        plt.plot(rh["feature_mean"], rh["abs_error"], marker="o", label="Abs. error")
        plt.plot(rh["feature_mean"], rh["classification_error"], marker="s", label="Error rate")
        plt.xlabel("Student radius bucket mean")
        plt.ylabel("Error")
        plt.legend()
        save_current("student_radius_vs_error.png")

    nd = bin_df[bin_df["feature"] == "normalized_dist"].copy()
    if not nd.empty:
        plt.figure(figsize=(5.2, 3.4))
        plt.plot(nd["feature_mean"], nd["correct_rate"], marker="o")
        plt.xlabel("Normalized ball distance bucket mean")
        plt.ylabel("Empirical correctness")
        save_current("normalized_distance_vs_correctness.png")

    if not history_df.empty:
        plt.figure(figsize=(5.2, 3.4))
        plt.plot(history_df["history_bucket"].astype(str), history_df["r_h_mean"], marker="o")
        plt.xlabel("Interaction history length")
        plt.ylabel("Mean student radius")
        plt.xticks(rotation=25)
        save_current("history_length_vs_student_radius.png")

    frequent_items = item_df[item_df["count"] >= max(5, int(item_df["count"].quantile(0.25)))] if not item_df.empty else item_df
    if not frequent_items.empty:
        plt.figure(figsize=(5.2, 3.4))
        plt.scatter(frequent_items["r_d_mean"], frequent_items["response_entropy"], s=12, alpha=0.55)
        plt.xlabel("Question radius")
        plt.ylabel("Response entropy")
        save_current("question_radius_vs_entropy.png")

    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate GBKT interpretability tables and plots.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="A single fold run directory or a CV directory containing fold run_config.json files.",
    )
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-item-count", type=int, default=20)
    parser.add_argument("--max-runs", type=int, default=None, help="Debug option: analyze only first N run dirs.")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug option: analyze only first N batches per run.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/gbkt_interpretability"))
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device).index or 0)

    set_seed(3407)
    run_dirs = resolve_run_dirs(args.run_dir)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    if not run_dirs:
        raise SystemExit(f"No run_config.json found under {args.run_dir}")

    print(f"Device: {device}")
    print(f"Found {len(run_dirs)} run dir(s).")

    frames = []
    for idx, run_dir in enumerate(run_dirs, 1):
        print(f"[{idx}/{len(run_dirs)}] Collecting {run_dir}")
        frame = collect_run(
            run_dir,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit("No valid prediction rows collected. Try --split valid for hidden-label test sets.")

    df = pd.concat(frames, ignore_index=True)
    df["r_sum"] = df["r_h_mean"] + df["r_d_mean"]

    out_dir = args.output_dir / args.run_dir.name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "position_rows.csv"
    df.to_csv(rows_path, index=False)

    bin_parts = []
    for feature in [
        "pred_ball",
        "theta",
        "r_h_mean",
        "r_d_mean",
        "r_sum",
        "center_dist",
        "normalized_dist",
        "overlap_margin",
    ]:
        if feature in df:
            part = qbin_summary(df, feature, args.bins)
            if not part.empty:
                bin_parts.append(part)
    bin_df = pd.concat(bin_parts, ignore_index=True) if bin_parts else pd.DataFrame()
    bins_path = out_dir / "feature_bins.csv"
    bin_df.to_csv(bins_path, index=False)

    history_df = history_bucket_summary(df)
    history_path = out_dir / "history_length_bins.csv"
    history_df.to_csv(history_path, index=False)

    item_df, item_corr_df = item_summary(df, min_item_count=args.min_item_count)
    item_path = out_dir / "item_radius_summary.csv"
    item_df.to_csv(item_path, index=False)

    corr_df = correlation_summary(df, item_corr_df)
    corr_path = out_dir / "correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    plot_paths = [] if args.no_plots else try_plot(out_dir, bin_df, history_df, item_df)

    summary = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "num_runs": len(run_dirs),
        "num_rows": int(len(df)),
        "folds": sorted([int(v) for v in df["fold"].dropna().unique().tolist()]),
        "overall": {
            "auc": safe_auc(df["target"].to_numpy(), df["pred"].to_numpy()),
            "ball_auc": safe_auc(df["target"].to_numpy(), df["pred_ball"].to_numpy()),
            "correct_rate": float(df["target"].mean()),
            "pred_mean": float(df["pred"].mean()),
            "abs_error": float(df["abs_error"].mean()),
            "bce": float(df["bce"].mean()),
            "classification_error": float(df["classification_error"].mean()),
            "r_h_mean": float(df["r_h_mean"].mean()),
            "r_d_mean": float(df["r_d_mean"].mean()),
            "normalized_dist": float(df["normalized_dist"].mean()),
        },
        "outputs": {
            "position_rows": str(rows_path),
            "feature_bins": str(bins_path),
            "history_length_bins": str(history_path),
            "item_radius_summary": str(item_path),
            "correlations": str(corr_path),
            "plots": plot_paths,
        },
    }
    summary_path = out_dir / "summary.json"
    dump_json(summary_path, summary)

    print("\nOverall:")
    print(f"  rows={len(df)}")
    print(f"  auc={summary['overall']['auc']:.6f}")
    print(f"  ball_auc={summary['overall']['ball_auc']:.6f}")
    print(f"  abs_error={summary['overall']['abs_error']:.6f}")
    print(f"  r_h_mean={summary['overall']['r_h_mean']:.6f}")
    print(f"  r_d_mean={summary['overall']['r_d_mean']:.6f}")
    print("\nWrote:")
    for path in [summary_path, rows_path, bins_path, history_path, item_path, corr_path]:
        print(f"  {path}")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
