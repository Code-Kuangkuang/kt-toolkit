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


def _load_completed_fold(
    cv_dir: Path,
    fold_id: int,
    dataset_name: str,
    model_name: str,
    train_label_flip_ratio: float,
    train_label_flip_seed: int,
    seed: int,
):
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
        flip_config = run_config.get("train_label_flip") or {}
        saved_ratio = float(flip_config.get("requested_ratio", 0.0))
        saved_seed = int(flip_config.get("seed", run_config.get("seed", 3407)))
        if abs(saved_ratio - train_label_flip_ratio) > 1e-12:
            continue
        if train_label_flip_ratio > 0 and saved_seed != train_label_flip_seed:
            continue
        # The seed is part of what produced this number, so a run under a
        # different one is a different experiment, not a completed fold.
        if int(run_config.get("seed", -999)) != int(seed):
            continue
        # A run from before the protocol stamp existed cannot be shown to match
        # the current one, so it is not reusable -- the same rule
        # run_baseline_table.py applies before putting rows in a table.
        if not run_config.get("protocol"):
            continue
        return {
            "fold": fold_id,
            "run_name": run_config.get("run_name", run_dir.name),
            "ckpt_dir": str(run_dir),
            "emb_type": run_config.get("emb_type"),
            "best_metrics": best_metrics,
            "best_path": _find_best_model_path(run_dir),
            "protocol": run_config["protocol"],
            "seed": int(run_config.get("seed", seed)),
            "skipped": True,
        }
    return None


def _assert_one_protocol(fold_results):
    """Refuse to average folds that were not produced the same way.

    `--skip-completed 1` reuses a finished fold from the CV directory, matching
    on dataset, model, fold and the label-flip settings. It used to stop there,
    so changing anything else -- `score_repeated_kc`, `concept_mode`,
    `pykt_transductive`, or regenerating the data -- and resuming would reuse the
    old folds, run the rest under the new settings, and average the two together
    into one reported mean.

    That is precisely the protocol mixing the stamp exists to prevent, happening
    in the one place that combines folds automatically, and announcing itself
    only as "completed run found". The scopes are not knowable until a fold has
    actually run, so the check belongs here, where they all are.
    """
    stamped = [r for r in fold_results if r.get("protocol")]
    if len(stamped) < 2:
        return

    reference = stamped[0]
    mismatches = []
    for result in stamped[1:]:
        diffs = {
            key: (reference["protocol"].get(key), result["protocol"].get(key))
            for key in set(reference["protocol"]) | set(result["protocol"])
            if reference["protocol"].get(key) != result["protocol"].get(key)
        }
        if diffs:
            mismatches.append((reference.get("fold"), result.get("fold"), diffs))

    if not mismatches:
        return

    lines = [
        "Folds in this run were produced under different protocols and cannot "
        "be averaged:",
    ]
    for ref_fold, other_fold, diffs in mismatches:
        for key, (a, b) in sorted(diffs.items()):
            lines.append(f"  fold {ref_fold} {key}={a!r}  vs  fold {other_fold} {key}={b!r}")
    lines.append(
        "This usually means --skip-completed reused folds from before a settings "
        "change. Use a fresh --cv-run-dir, or rerun the stale folds."
    )
    raise SystemExit("\n".join(lines))


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
    frozen_difficulty_alpha: Optional[float] = typer.Option(
        None,
        "--frozen_difficulty_alpha", "--frozen-difficulty-alpha",
        help="Shrinkage strength for SimpleKT's frozen difficulty table "
             "(qid_frozen*). An item is pulled towards its target with weight "
             "alpha/(n_i+alpha), so the same alpha is a different method at "
             "different observations-per-item. Default 10.",
        ),
    frozen_difficulty_group_seed: Optional[int] = typer.Option(
        None,
        "--frozen_difficulty_group_seed", "--frozen-difficulty-group-seed",
        help="Seed for the qid_frozen_grouprand control's group reshuffle.",
        ),
    memory_rule: Optional[str] = typer.Option(
        None, "--memory-rule", help="simplekt_delta: delta, ema, kc_ema, or none.",
        ),
    memory_dim: Optional[int] = typer.Option(
        None, "--memory-dim", help="Dimension of the associative residual memory.",
        ),
    memory_rate: Optional[float] = typer.Option(
        None, "--memory-rate", help="Fixed residual-memory write rate in [0,1].",
        ),

    # CGMKT's three artefact sources. The paper and the released code disagree
    # about what the question graph and the KC graph are (models/cgmkt_graphs.py
    # records the evidence), so which variant a run took has to be stated rather
    # than implied by the model name.
    question_graph_source: Optional[str] = typer.Option(
        None,
        "--question_graph_source", "--question-graph-source",
        help="CGMKT question branch: 'incidence' (the released code's padded "
             "Q-matrix) or 'cooccur' (the paper's A@A.T co-occurrence graph).",
        ),
    kc_graph_source: Optional[str] = typer.Option(
        None,
        "--kc_graph_source", "--kc-graph-source",
        help="CGMKT KC branch: 'sbm_random' (the released pipeline's untrained "
             "forward pass), 'sbm_fit' (a real likelihood fit, train folds "
             "only), 'cooccur', or 'random'.",
        ),
    kc_embedding_source: Optional[str] = typer.Option(
        None,
        "--kc_embedding_source", "--kc-embedding-source",
        help="CGMKT concept vectors: 'bge' (the skill-name text embeddings) or "
             "'random' (same shape and norm, no semantics). Separates sharing "
             "statistical strength from using what the names mean.",
        ),
    use_mastery: Optional[bool] = typer.Option(
        None,
        "--use_mastery/--no-use-mastery", "--use-mastery/--no_use_mastery",
        help="CGMKT's group-level mastery module. --no-use-mastery is the "
             "paper's 'w/o mastery state' arm: gating, propagation, updating "
             "and readout all bypassed, GRU over the fused sequence alone.",
        ),
    group_source: Optional[str] = typer.Option(
        None,
        "--group_source", "--group-source",
        help="CGMKT knowledge groups: 'spectral' (what every released run "
             "silently falls back to), 'sbm_fit', or 'random'.",
        ),
    num_clusters: Optional[int] = typer.Option(
        None,
        "--num_clusters", "--num-clusters",
        help="CGMKT's number of knowledge groups k. The paper sweeps 2-9 and "
             "selects 9, the top of its own grid.",
        ),
    graph_seed: Optional[int] = typer.Option(
        None,
        "--graph_seed", "--graph-seed",
        help="Seed for CGMKT's graph and group construction. Upstream left this "
             "unset, so its graphs were not reproducible across regenerations.",
        ),

    # Attention options. These default to None, like every other hyperparameter
    # option here, because `apply_overrides` writes any non-None value straight
    # into the model config: a concrete default would silently replace the
    # per-model value for every run that did not ask for it. With d_ff=512 it
    # rewrote 20 of the 45 models -- akt from n_blocks 1 to 4, and d_ff 256 to
    # 512 across the whole AKT family -- while the run's own log still printed
    # the number it had overwritten.
    d_model: Optional[int] = typer.Option(
        None,
        "--d_model", "--d-model",
        help="Dimension of the model. Unset: use the model's kt_config value."
        ),
    d_ff: Optional[int] = typer.Option(
        None,
        "--d_ff", "--d-ff",
        help="Dimension of the feed forward network. Unset: use kt_config."
        ),
    num_attn_heads: Optional[int] = typer.Option(
        None,
        "--num_attn_heads", "--num-attn-heads",
        help="Number of attention heads. Unset: use kt_config."
        ),
    n_blocks: Optional[int] = typer.Option(
        None,
        "--n_blocks", "--n-blocks",
        help="Number of transformer blocks. Unset: use kt_config."
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
    train_label_flip_ratio: float = typer.Option(
        0.0,
        "--train_label_flip_ratio", "--train-label-flip-ratio",
        help="Fraction of binary responses to flip in the training split only. Range: [0, 1].",
    ),
    train_label_flip_seed: Optional[int] = typer.Option(
        None,
        "--train_label_flip_seed", "--train-label-flip-seed",
        help="Seed for selecting flipped response positions. Defaults to --seed.",
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
    if not 0.0 <= train_label_flip_ratio <= 1.0:
        raise typer.BadParameter("--train-label-flip-ratio must be in [0, 1].")
    resolved_flip_seed = seed if train_label_flip_seed is None else train_label_flip_seed
    kt_cfg_raw = load_cfg(kt_config)
    data_config_raw = load_cfg(data_config_path)

    wandb_cfg = None
    if use_wandb == 1 and os.path.exists(wandb_config):
        wandb_cfg = load_cfg(wandb_config)

    def _train_one_fold(fold_id: int, save_root: str, cv_run_name: Optional[str] = None):
        # Every CLI hyperparameter option belongs here. --d-model, --d-ff,
        # --num-attn-heads and --n-blocks were accepted and then never forwarded,
        # so a run that passed them trained on the config file's values while its
        # own command line said otherwise.
        overrides = {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "emb_size": emb_size,
            "dropout": dropout,
            "patience": patience,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_attn_heads": num_attn_heads,
            "n_blocks": n_blocks,
            "question_graph_source": question_graph_source,
            "kc_graph_source": kc_graph_source,
            "group_source": group_source,
            "kc_embedding_source": kc_embedding_source,
            "use_mastery": use_mastery,
            "num_clusters": num_clusters,
            "graph_seed": graph_seed,
            "frozen_difficulty_alpha": frozen_difficulty_alpha,
            "frozen_difficulty_group_seed": frozen_difficulty_group_seed,
            "memory_rule": memory_rule,
            "memory_dim": memory_dim,
            "memory_rate": memory_rate,
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
            train_label_flip_ratio=train_label_flip_ratio,
            train_label_flip_seed=resolved_flip_seed,
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
            if train_label_flip_ratio > 0:
                ratio_tag = f"{train_label_flip_ratio:g}".replace(".", "p")
                cv_run_name = f"{cv_run_name}-flip{ratio_tag}"
            if add_uuid == 1:
                cv_run_name = f"{cv_run_name}-{uuid.uuid4()}"
            cv_dir = os.path.join(save_dir, cv_run_name)
            cv_dir_path = Path(cv_dir)
            os.makedirs(cv_dir, exist_ok=True)

        fold_results = []
        for fid in fold_ids:
            if skip_completed == 1:
                completed = _load_completed_fold(
                    cv_dir_path,
                    fid,
                    dataset_name,
                    model_name,
                    train_label_flip_ratio,
                    resolved_flip_seed,
                    seed,
                )
                if completed is not None:
                    print(f"\n[yellow]===== CV Fold {fid} skipped: completed run found =====[/yellow]\n")
                    fold_results.append(completed)
                    continue
            print(f"\n[bold]===== CV Fold {fid} / {fold_ids} =====[/bold]\n")
            fold_results.append(_train_one_fold(fid, save_root=cv_dir, cv_run_name=cv_run_name))

        _assert_one_protocol(fold_results)
        agg = aggregate_fold_metrics(fold_results)
        cv_payload = {
            "cv_run_name": cv_run_name,
            "timestamp": ts,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "emb_type": emb_type,
            "folds": fold_ids,
            "seed": seed,
            "train_label_flip": {
                "requested_ratio": train_label_flip_ratio,
                "seed": resolved_flip_seed,
                "scope": "train_only",
            },
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
