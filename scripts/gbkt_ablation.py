import csv
import datetime
import json
import os
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Optional
import uuid

import typer
from rich import print


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_cfg
from core.run_support import save_run_config


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


def _model_updates(*parts):
    merged = {}
    for part in parts:
        merged.update(part)
    return merged


ABLATIONS = OrderedDict(
    [
        (
            "full",
            {
                "title": "Full GBKT",
                "description": "Full GBKTV4 configuration from kt_config.json.",
                "model": {},
                "train": {},
            },
        ),
        (
            "gbkt-base",
            {
                "title": "GBKT-Base",
                "description": (
                    "Minimal ball-space KT backbone: student/item balls, KAB update, "
                    "and ball-to-ball prediction only."
                ),
                "model": GBKT_BASE_MODEL_UPDATES,
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-point",
            {
                "title": "GBKT-Base Point-Space",
                "description": (
                    "Point-space counterpart of GBKT-Base: remove learnable radii and disable "
                    "all radius use while keeping the same minimal backbone."
                ),
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {"use_point_space": True},
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-plus-history",
            {
                "title": "GBKT-Base + Distance-Aware History",
                "description": "Add historical state attention with sequence-distance bias to GBKT-Base.",
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {
                        "history_mode": "attention",
                        "use_sequence_distance_attention": True,
                    },
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-wo-bbp-radius-norm",
            {
                "title": "GBKT-Base w/o Radius-Normalized BBP",
                "description": (
                    "Remove radius normalization from ball-to-ball prediction: "
                    "theta is estimated from center displacement only."
                ),
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {"use_bbp_radius_normalization": False},
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-wo-radius-discrimination",
            {
                "title": "GBKT-Base w/o Radius-Guided Discrimination",
                "description": (
                    "Estimate discrimination/confidence from center displacement instead of combined radius."
                ),
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {"use_radius_discrimination": False},
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-wo-kab-radius",
            {
                "title": "GBKT-Base w/o Radius-Aware KAB",
                "description": "Remove radius-normalized mismatch and radius_sum from KAB plausibility.",
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {"use_kab_radius_features": False},
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-center-only-geometry",
            {
                "title": "GBKT-Base Center-Only Geometry",
                "description": (
                    "Disable radius use in BBP normalization, discrimination/confidence, and KAB plausibility."
                ),
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {
                        "use_bbp_radius_normalization": False,
                        "use_radius_discrimination": False,
                        "use_kab_radius_features": False,
                    },
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-plus-history-readout",
            {
                "title": "GBKT-Base + History + Concept Readout",
                "description": "Add concept-conditioned readout on top of distance-aware history retrieval.",
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {
                        "history_mode": "attention",
                        "use_sequence_distance_attention": True,
                        "use_concept_readout": True,
                    },
                ),
                "train": {"use_timestamps": False},
            },
        ),
        (
            "base-plus-history-readout-time",
            {
                "title": "GBKT-Base + History + Readout + Time",
                "description": "Further add time-aware KAB and time forgetting.",
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {
                        "history_mode": "attention",
                        "use_sequence_distance_attention": True,
                        "use_concept_readout": True,
                        "use_time_aware_kab": True,
                        "use_time_forgetting": True,
                        "use_timestamps": True,
                    },
                ),
                "train": {"use_timestamps": True},
            },
        ),
        (
            "base-plus-history-readout-time-rasch",
            {
                "title": "GBKT-Base + History + Readout + Time + Rasch",
                "description": "Further add scalar Rasch item-difficulty calibration.",
                "model": _model_updates(
                    GBKT_BASE_MODEL_UPDATES,
                    {
                        "history_mode": "attention",
                        "use_sequence_distance_attention": True,
                        "use_concept_readout": True,
                        "use_time_aware_kab": True,
                        "use_time_forgetting": True,
                        "use_timestamps": True,
                        "use_scalar_item_difficulty": True,
                        "lambda_item_difficulty": 1e-4,
                    },
                ),
                "train": {"use_timestamps": True},
            },
        ),
        (
            "strict-point",
            {
                "title": "GBKT-Point",
                "description": (
                    "Strict point-space ablation: remove learnable radii and disable radius use "
                    "in prediction and state updating while keeping the remaining architecture."
                ),
                "model": {
                    "use_point_space": True,
                },
                "train": {},
            },
        ),
        (
            "wo-prediction-radius",
            {
                "title": "w/o prediction-side radius",
                "description": (
                    "Keep radius-aware state updating, but remove radius normalization and "
                    "radius-guided discrimination/confidence from BBP prediction."
                ),
                "model": {
                    "use_bbp_radius_normalization": False,
                    "use_radius_discrimination": False,
                },
                "train": {},
            },
        ),
        (
            "wo-update-radius",
            {
                "title": "w/o update-side radius",
                "description": (
                    "Keep radius-aware BBP prediction, but remove radius features from KAB "
                    "and stop response-driven student-radius updating."
                ),
                "model": {
                    "use_kab_radius_features": False,
                    "use_radius_state_update": False,
                },
                "train": {},
            },
        ),
        (
            "wo-irt-bbp",
            {
                "title": "w/o IRT-inspired BBP",
                "description": (
                    "Replace the IRT-inspired ball-to-ball response function with an MLP "
                    "over ball geometry features."
                ),
                "model": {
                    "response_function": "mlp",
                    "lambda_theta": 0.0,
                    "lambda_conf": 0.0,
                },
                "train": {},
            },
        ),
        (
            "wo-loss-ball",
            {
                "title": "w/o loss_ball",
                "description": "Disable the ball-branch auxiliary BCE loss while keeping the branch.",
                "model": {"lambda_ball": 0.0},
                "train": {},
            },
        ),
        (
            "wo-loss-concept-next",
            {
                "title": "w/o loss_concept_next",
                "description": "Disable concept-next auxiliary loss but keep concept fusion active.",
                "model": {"lambda_concept_next": 0.0},
                "train": {},
            },
        ),
        (
            "wo-loss-theta",
            {
                "title": "w/o loss_theta",
                "description": "Disable theta auxiliary BCE loss.",
                "model": {"lambda_theta": 0.0},
                "train": {},
            },
        ),
        (
            "wo-loss-item-difficulty",
            {
                "title": "w/o loss_item_difficulty",
                "description": "Disable scalar Rasch L2 regularization but keep the scalar item difficulty term.",
                "model": {"lambda_item_difficulty": 0.0},
                "train": {},
            },
        ),
        (
            "wo-concept-aux",
            {
                "title": "w/o concept auxiliary branch",
                "description": "Disable concept residual fusion and concept-next auxiliary supervision.",
                "model": {"max_concept_fusion_weight": 0.0, "lambda_concept_next": 0.0},
                "train": {},
            },
        ),
        (
            "wo-scalar-rasch",
            {
                "title": "w/o scalar Rasch difficulty",
                "description": "Remove scalar item-difficulty calibration and its regularization.",
                "model": {"use_scalar_item_difficulty": False, "lambda_item_difficulty": 0.0},
                "train": {},
            },
        ),
        (
            "wo-concept-readout",
            {
                "title": "w/o concept-conditioned readout",
                "description": "Predict with the global student ball without target concept/item readout.",
                "model": {"use_concept_readout": False},
                "train": {},
            },
        ),
        (
            "wo-history-attention",
            {
                "title": "w/o history attention",
                "description": "Disable historical state attention entirely.",
                "model": {"history_mode": "none"},
                "train": {},
            },
        ),
        (
            "wo-seq-distance",
            {
                "title": "w/o sequence-distance attention",
                "description": "Keep history attention but remove sequence-distance bias.",
                "model": {"use_sequence_distance_attention": False},
                "train": {},
            },
        ),
        (
            "wo-time-dynamics",
            {
                "title": "w/o temporal uncertainty calibration",
                "description": "Disable time-aware KAB and time forgetting.",
                "model": {
                    "use_time_aware_kab": False,
                    "use_time_forgetting": False,
                    "use_timestamps": False,
                },
                "train": {"use_timestamps": False},
            },
        ),
    ]
)


ABLATION_GROUPS = {
    "geometry": [
        "gbkt-base",
        "base-point",
        "base-wo-bbp-radius-norm",
        "base-wo-radius-discrimination",
        "base-wo-kab-radius",
        "base-center-only-geometry",
    ],
    "incremental": [
        "gbkt-base",
        "base-plus-history",
        "base-plus-history-readout",
        "base-plus-history-readout-time",
        "base-plus-history-readout-time-rasch",
        "full",
    ],
    "losses": [
        "full",
        "wo-loss-ball",
        "wo-loss-concept-next",
        "wo-loss-theta",
        "wo-loss-item-difficulty",
    ],
    "main": [
        "full",
        "strict-point",
        "wo-prediction-radius",
        "wo-update-radius",
        "wo-concept-readout",
        "wo-irt-bbp",
        "wo-history-attention",
        "wo-time-dynamics",
        "wo-concept-aux",
        "wo-scalar-rasch",
    ],
    "strict": [
        "full",
        "strict-point",
        "wo-prediction-radius",
        "wo-update-radius",
        "wo-concept-readout",
        "wo-irt-bbp",
        "wo-time-dynamics",
    ],
    "all": list(ABLATIONS.keys()),
}


ALIASES = {
    "base": "gbkt-base",
    "ball-base": "gbkt-base",
    "gbkt_base": "gbkt-base",
    "gbktbase": "gbkt-base",
    "base_point": "base-point",
    "base-strict-point": "base-point",
    "gbkt-base-point": "base-point",
    "base_history": "base-plus-history",
    "wo_bbp_radius_norm": "base-wo-bbp-radius-norm",
    "wo-bbp-radius-normalization": "base-wo-bbp-radius-norm",
    "wo_radius_discrimination": "base-wo-radius-discrimination",
    "wo-radius-guided-discrimination": "base-wo-radius-discrimination",
    "wo_kab_radius": "base-wo-kab-radius",
    "wo-radius-aware-kab": "base-wo-kab-radius",
    "center-only": "base-center-only-geometry",
    "center_only": "base-center-only-geometry",
    "point-geometry": "base-center-only-geometry",
    "point": "strict-point",
    "strict-point-space": "strict-point",
    "gbkt-point": "strict-point",
    "wo_prediction_radius": "wo-prediction-radius",
    "no-prediction-radius": "wo-prediction-radius",
    "wo_update_radius": "wo-update-radius",
    "no-update-radius": "wo-update-radius",
    "wo_irt_bbp": "wo-irt-bbp",
    "no-irt-bbp": "wo-irt-bbp",
    "mlp-bbp": "wo-irt-bbp",
    "base-readout": "base-plus-history-readout",
    "base_readout": "base-plus-history-readout",
    "base-time": "base-plus-history-readout-time",
    "base_time": "base-plus-history-readout-time",
    "base-rasch": "base-plus-history-readout-time-rasch",
    "base_rasch": "base-plus-history-readout-time-rasch",
    "none": "full",
    "baseline": "full",
    "wo_loss_ball": "wo-loss-ball",
    "no-loss-ball": "wo-loss-ball",
    "wo_loss_concept_next": "wo-loss-concept-next",
    "no-loss-concept-next": "wo-loss-concept-next",
    "wo_loss_theta": "wo-loss-theta",
    "no-loss-theta": "wo-loss-theta",
    "wo_loss_item_difficulty": "wo-loss-item-difficulty",
    "no-loss-item-difficulty": "wo-loss-item-difficulty",
    "wo_concept_aux": "wo-concept-aux",
    "no-concept-aux": "wo-concept-aux",
    "wo_scalar_rasch": "wo-scalar-rasch",
    "no-scalar-rasch": "wo-scalar-rasch",
    "wo_concept_readout": "wo-concept-readout",
    "no-concept-readout": "wo-concept-readout",
    "wo_history_attention": "wo-history-attention",
    "no-history-attention": "wo-history-attention",
    "wo_seq_distance": "wo-seq-distance",
    "no-seq-distance": "wo-seq-distance",
    "wo_time_dynamics": "wo-time-dynamics",
    "no-time-dynamics": "wo-time-dynamics",
    "wo_temporal_calibration": "wo-time-dynamics",
    "no-temporal-calibration": "wo-time-dynamics",
}


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
        if part:
            folds.append(int(part))
    return folds


def _normalize_variant_name(name: str):
    raw = name.strip().lower()
    if not raw:
        return raw
    raw = raw.replace("_", "-")
    return ALIASES.get(raw, raw)


def _parse_variants_spec(spec: str):
    spec = (spec or "losses").strip().lower()
    if spec in ABLATION_GROUPS:
        return list(ABLATION_GROUPS[spec])

    variants = []
    for part in spec.split(","):
        variant = _normalize_variant_name(part)
        if not variant:
            continue
        if variant not in ABLATIONS:
            known = ", ".join(ABLATIONS.keys())
            groups = ", ".join(ABLATION_GROUPS.keys())
            raise typer.BadParameter(
                f"Unknown ablation variant '{part}'. Known variants: {known}. Groups: {groups}."
            )
        if variant not in variants:
            variants.append(variant)
    if not variants:
        raise typer.BadParameter("No ablation variant selected.")
    return variants


def _load_json(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _find_best_model_path(run_dir: Path):
    candidates = [p for p in run_dir.glob("*.pt") if p.name != "last_epoch_model.pt"]
    if not candidates:
        return None
    return str(sorted(candidates)[0])


def _load_completed_fold(cv_dir: Path, fold_id: int, dataset_name: str, model_name: str, variant: str):
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

        ablation = run_config.get("ablation") or {}
        recorded_variant = ablation.get("variant")
        if recorded_variant is not None and recorded_variant != variant:
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


def _apply_variant_config(kt_cfg_raw, model_name: str, variant: str):
    kt_cfg = json.loads(json.dumps(kt_cfg_raw))
    if model_name not in kt_cfg:
        raise typer.BadParameter(f"Model '{model_name}' not found in kt_config.")

    spec = ABLATIONS[variant]
    kt_cfg[model_name].update(spec.get("model", {}))
    kt_cfg["train_config"].update(spec.get("train", {}))
    return kt_cfg


def _build_runtime_overrides(
    *,
    batch_size,
    num_epochs,
    learning_rate,
    emb_size,
    dropout,
    patience,
):
    return {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "emb_size": emb_size,
        "dropout": dropout,
        "patience": patience,
    }


def _annotate_run_config(result, variant: str):
    ckpt_dir = result.get("ckpt_dir")
    if not ckpt_dir:
        return
    run_config_path = Path(ckpt_dir) / "run_config.json"
    if not run_config_path.exists():
        return
    payload = _load_json(run_config_path)
    payload["ablation"] = {
        "variant": variant,
        "title": ABLATIONS[variant]["title"],
        "description": ABLATIONS[variant]["description"],
        "model_updates": ABLATIONS[variant].get("model", {}),
        "train_updates": ABLATIONS[variant].get("train", {}),
    }
    save_run_config(str(run_config_path), payload)


def _save_ablation_summary(ablation_dir: Path, payload: dict):
    save_run_config(str(ablation_dir / "ablation_summary.json"), payload)

    csv_path = ablation_dir / "ablation_summary.csv"
    fieldnames = [
        "variant",
        "title",
        "folds",
        "valid_auc_mean",
        "valid_auc_std",
        "best_test_auc_mean",
        "best_test_auc_std",
        "valid_acc_mean",
        "valid_acc_std",
        "best_test_acc_mean",
        "best_test_acc_std",
        "variant_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload["variants"]:
            agg = item.get("aggregate", {})

            def metric(metric_name, stat):
                data = agg.get(metric_name) or {}
                return data.get(stat)

            writer.writerow(
                {
                    "variant": item["variant"],
                    "title": item["title"],
                    "folds": ",".join(str(x) for x in item["folds"]),
                    "valid_auc_mean": metric("valid_auc", "mean"),
                    "valid_auc_std": metric("valid_auc", "std"),
                    "best_test_auc_mean": metric("best_test_auc", "mean"),
                    "best_test_auc_std": metric("best_test_auc", "std"),
                    "valid_acc_mean": metric("valid_acc", "mean"),
                    "valid_acc_std": metric("valid_acc", "std"),
                    "best_test_acc_mean": metric("best_test_acc", "mean"),
                    "best_test_acc_std": metric("best_test_acc", "std"),
                    "variant_dir": item["variant_dir"],
                }
            )


def _print_variants():
    print("[bold]GBKT ablation variants[/bold]")
    for name, spec in ABLATIONS.items():
        print(f"  [cyan]{name}[/cyan]: {spec['title']} - {spec['description']}")
    print("")
    print("[bold]Groups[/bold]")
    for name, variants in ABLATION_GROUPS.items():
        print(f"  [green]{name}[/green]: {', '.join(variants)}")


@app.command()
def main(
    dataset_name: Optional[str] = typer.Option(None, "--dataset_name", "--dataset-name"),
    model_name: str = typer.Option("gbktv4", "--model_name", "--model-name"),
    variants: str = typer.Option(
        "losses",
        "--variants",
        help="Ablation group or comma-separated variants. Groups: geometry, incremental, losses, main, strict, all.",
    ),
    list_variants: bool = typer.Option(
        False,
        "--list-variants",
        help="Print built-in ablation variants and exit.",
    ),
    cv: int = typer.Option(0, "--cv", help="Run multiple folds sequentially."),
    fold: int = typer.Option(0, "--fold", help="Single fold id when --cv=0."),
    folds: str = typer.Option("0-4", "--folds", help="Fold spec when --cv=1. Examples: 0-4 or 0,1,3."),
    skip_completed: int = typer.Option(
        0,
        "--skip-completed",
        help="Skip completed fold runs inside each variant directory.",
    ),
    seed: int = typer.Option(3407, "--seed"),
    gpu: int = typer.Option(0, "--gpu"),
    batch_size: Optional[int] = typer.Option(None, "--batch_size", "--batch-size"),
    num_epochs: Optional[int] = typer.Option(None, "--num_epochs", "--num-epochs"),
    learning_rate: Optional[float] = typer.Option(None, "--learning_rate", "--learning-rate"),
    emb_size: Optional[int] = typer.Option(None, "--emb_size", "--emb-size"),
    dropout: Optional[float] = typer.Option(None, "--dropout"),
    patience: Optional[int] = typer.Option(None, "--patience"),
    emb_type: Optional[str] = typer.Option(None, "--emb_type", "--emb-type"),
    save_dir: str = typer.Option("saved_model", "--save_dir", "--save-dir"),
    use_wandb: int = typer.Option(0, "--use_wandb", "--use-wandb"),
    add_uuid: int = typer.Option(0, "--add_uuid", "--add-uuid"),
    kt_config: str = typer.Option("configs/kt_config.json", "--kt_config", "--kt-config"),
    data_config_path: str = typer.Option("configs/data_config.json", "--data_config", "--data-config"),
    wandb_config: str = typer.Option("configs/wandb.json", "--wandb_config", "--wandb-config"),
):
    if list_variants:
        _print_variants()
        raise typer.Exit()

    if not dataset_name:
        raise typer.BadParameter("--dataset-name is required unless --list-variants is used.")

    import core.trainers  # register trainers
    import datasets.init_dataset  # register dataset builders
    import models  # register models
    from core.train_runner import (
        aggregate_fold_metrics,
        print_cv_summary,
        save_cv_summary,
        train_one_fold,
    )

    model_name = model_name.lower()
    variant_names = _parse_variants_spec(variants)
    fold_ids = _parse_folds_spec(folds) if cv == 1 else [int(fold)]

    kt_cfg_raw = load_cfg(kt_config)
    data_config_raw = load_cfg(data_config_path)

    wandb_cfg = None
    if use_wandb == 1 and os.path.exists(wandb_config):
        wandb_cfg = load_cfg(wandb_config)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ablation_run_name = f"ablation-{dataset_name}-{model_name}-{ts}"
    if add_uuid == 1:
        ablation_run_name = f"{ablation_run_name}-{uuid.uuid4()}"

    ablation_dir = Path(save_dir) / ablation_run_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    runtime_overrides = _build_runtime_overrides(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        emb_size=emb_size,
        dropout=dropout,
        patience=patience,
    )

    print(f"[bold]GBKT ablation run:[/bold] {ablation_run_name}")
    print(f"[bold]Dataset:[/bold] {dataset_name}  [bold]Model:[/bold] {model_name}")
    print(f"[bold]Variants:[/bold] {', '.join(variant_names)}")
    print(f"[bold]Folds:[/bold] {fold_ids}")
    print(f"[bold]Output:[/bold] {ablation_dir}")

    variant_summaries = []
    for variant in variant_names:
        spec = ABLATIONS[variant]
        variant_dir = ablation_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        cv_run_name = f"{ablation_run_name}-{variant}"

        print(f"\n[bold magenta]===== Ablation: {variant} ({spec['title']}) =====[/bold magenta]")
        print(f"[cyan]{spec['description']}[/cyan]")
        if spec.get("model"):
            print("Model updates: " + json.dumps(spec["model"], ensure_ascii=True))
        if spec.get("train"):
            print("Train updates: " + json.dumps(spec["train"], ensure_ascii=True))

        kt_cfg_variant = _apply_variant_config(kt_cfg_raw, model_name, variant)
        fold_results = []
        for fid in fold_ids:
            if skip_completed == 1:
                completed = _load_completed_fold(variant_dir, fid, dataset_name, model_name, variant)
                if completed is not None:
                    print(f"\n[yellow]===== {variant} fold {fid} skipped: completed run found =====[/yellow]\n")
                    fold_results.append(completed)
                    continue

            print(f"\n[bold]===== {variant} Fold {fid} / {fold_ids} =====[/bold]\n")
            result = train_one_fold(
                dataset_name=dataset_name,
                model_name=model_name,
                emb_type=emb_type,
                fold_id=fid,
                root_dir=str(ROOT),
                kt_cfg_raw=kt_cfg_variant,
                data_config_raw=data_config_raw,
                seed=seed,
                save_root=str(variant_dir),
                add_uuid=0,
                wandb_cfg=wandb_cfg,
                kt_config_path=kt_config,
                data_config_path=data_config_path,
                wandb_config_path=wandb_config,
                cv_run_name=cv_run_name if cv == 1 else None,
                overrides=runtime_overrides,
                gpu_id=gpu,
            )
            _annotate_run_config(result, variant)
            fold_results.append(result)

        agg = aggregate_fold_metrics(fold_results)
        variant_payload = {
            "cv_run_name": cv_run_name,
            "timestamp": ts,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "ablation_variant": variant,
            "ablation_title": spec["title"],
            "ablation_description": spec["description"],
            "ablation_model_updates": spec.get("model", {}),
            "ablation_train_updates": spec.get("train", {}),
            "emb_type": emb_type,
            "folds": fold_ids,
            "seed": seed,
            "save_dir": str(variant_dir),
            "cv_dir": str(variant_dir),
            "per_fold": fold_results,
            "aggregate": agg,
        }
        save_cv_summary(str(variant_dir), variant_payload, fold_results)
        print_cv_summary(agg, str(variant_dir))

        variant_summaries.append(
            {
                "variant": variant,
                "title": spec["title"],
                "description": spec["description"],
                "model_updates": spec.get("model", {}),
                "train_updates": spec.get("train", {}),
                "folds": fold_ids,
                "variant_dir": str(variant_dir),
                "per_fold": fold_results,
                "aggregate": agg,
            }
        )
        _save_ablation_summary(
            ablation_dir,
            {
                "ablation_run_name": ablation_run_name,
                "timestamp": ts,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "seed": seed,
                "folds": fold_ids,
                "variants": variant_summaries,
            },
        )

    print(f"\n[green][bold]Ablation summary saved to: {ablation_dir}[/bold][/green]")
    print(f"  JSON: {ablation_dir / 'ablation_summary.json'}")
    print(f"  CSV : {ablation_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    app()
