from pathlib import Path
import csv
import json
import sys
from typing import Optional

import pandas as pd
import torch
import typer
from rich import print


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # register models
from core.config import load_cfg
from core.factory import build_model


app = typer.Typer(add_completion=False)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(ckpt_dir: Optional[Path], checkpoint: Optional[Path], model_name: str, emb_type: str) -> Path:
    if checkpoint is not None:
        return checkpoint
    if ckpt_dir is None:
        raise typer.BadParameter("Provide either --ckpt-dir or --checkpoint.")
    candidates = [
        ckpt_dir / f"{model_name}_{emb_type}_model.pt",
        ckpt_dir / f"{model_name}_model.pt",
    ]
    matches = [p for p in candidates if p.exists()]
    if not matches:
        matches = sorted(ckpt_dir.glob("*_model.pt"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}.")
    return matches[0]


def _first_int(value: str) -> int:
    return int(str(value).split("_", 1)[0])


def _build_model_from_config(
    *,
    dataset_name: str,
    model_name: str,
    emb_type: Optional[str],
    run_config_path: Optional[Path],
    kt_config_path: Path,
    data_config_path: Path,
    device: torch.device,
):
    if run_config_path is not None and run_config_path.exists():
        run_config = _load_json(run_config_path)
        model_cfg = dict(run_config["model_config"])
        dataset_cfg = dict(run_config["dataset_config"])
        resolved_emb_type = emb_type or run_config.get("emb_type") or model_cfg.get("emb_type", "qid")
    else:
        kt_cfg = load_cfg(str(kt_config_path))
        data_cfg = load_cfg(str(data_config_path))
        model_cfg = dict(kt_cfg[model_name])
        dataset_cfg = dict(data_cfg[dataset_name])
        resolved_emb_type = emb_type or model_cfg.get("emb_type", "qid")

    model_cfg["emb_type"] = resolved_emb_type
    excluded = {
        "loss_c_all_lambda", "loss_q_all_lambda", "loss_c_next_lambda", "loss_q_next_lambda",
        "output_mode", "output_c_all_lambda", "output_c_next_lambda", "output_q_all_lambda",
        "output_q_next_lambda", "emb_type", "learning_rate", "use_timestamps", "dpath",
        "num_at", "num_it", "booster_strategy", "require_fold_embedding",
    }
    model_kwargs = {k: v for k, v in model_cfg.items() if k not in excluded}
    model = build_model(
        model_name,
        num_c=dataset_cfg["num_c"],
        num_q=dataset_cfg["num_q"],
        emb_type=resolved_emb_type,
        device=str(device),
        dpath=dataset_cfg.get("dpath", ""),
        **model_kwargs,
    ).to(device)
    return model, resolved_emb_type


def _iekt_predict_row(model, questions, concepts, responses, use_pred: bool, device: torch.device):
    net = model.model
    h = torch.zeros(1, net.emb_size, device=device)
    predictions = []

    for q, c, r in zip(questions, concepts, responses):
        if q < 0 or c < 0:
            continue

        q_t = torch.tensor([q], dtype=torch.long, device=device)
        c_t = torch.tensor([c], dtype=torch.long, device=device)

        v = net.get_ques_representation(q=q_t, c=c_t)
        ques_h = torch.cat([v, h], dim=1)
        cog_action = torch.argmax(net.pi_cog_func(ques_h), dim=1)
        cog_emb = net.cog_matrix[cog_action, :]
        h_v, v, logits, _ = net.obtain_v(q=q_t, c=c_t, h=h, x=None, emb=cog_emb)
        prob = torch.sigmoid(logits).squeeze()

        if r == -1:
            predictions.append(float(prob.detach().cpu()))
            if not use_pred:
                continue
            r = 1 if float(prob.detach().cpu()) >= 0.5 else 0

        r_t = torch.tensor([[r]], dtype=torch.float, device=device)
        pred_label = torch.where(
            prob.view(1, 1) > 0.5,
            torch.ones(1, 1, device=device),
            torch.zeros(1, 1, device=device),
        )
        out_x_true = torch.cat(
            [h_v.mul(r_t.repeat(1, h_v.size(-1))), h_v.mul((1 - r_t).repeat(1, h_v.size(-1)))],
            dim=1,
        )
        out_x_pred = torch.cat(
            [h_v.mul(pred_label.repeat(1, h_v.size(-1))), h_v.mul((1 - pred_label).repeat(1, h_v.size(-1)))],
            dim=1,
        )
        sens_action = torch.argmax(net.pi_sens_func(torch.cat([out_x_true, out_x_pred], dim=1)), dim=1)
        sens_emb = net.acq_matrix[sens_action, :]
        h = net.update_state(h, v, sens_emb, r_t)

    return predictions


def _akt_predict_row(model, questions, concepts, responses, use_pred: bool, device: torch.device):
    if any(c < 0 for c in concepts):
        valid_len = next((i for i, c in enumerate(concepts) if c < 0), len(concepts))
        questions = questions[:valid_len]
        concepts = concepts[:valid_len]
        responses = responses[:valid_len]

    input_responses = [0 if r == -1 else r for r in responses]
    q_data = torch.tensor([concepts], dtype=torch.long, device=device)
    pid_data = torch.tensor([questions], dtype=torch.long, device=device) if getattr(model, "n_pid", 0) > 0 else None

    if use_pred:
        probs = []
        running = input_responses[:]
        for pos in [i for i, r in enumerate(responses) if r == -1]:
            target = torch.tensor([running], dtype=torch.long, device=device)
            pred_tensor, _ = model(q_data, target, pid_data) if pid_data is not None else model(q_data, target)
            prob = float(pred_tensor[0, pos].detach().cpu())
            probs.append(prob)
            running[pos] = 1 if prob >= 0.5 else 0
        return probs

    target = torch.tensor([input_responses], dtype=torch.long, device=device)
    pred_tensor, _ = model(q_data, target, pid_data) if pid_data is not None else model(q_data, target)
    return [float(pred_tensor[0, i].detach().cpu()) for i, r in enumerate(responses) if r == -1]


def _predict_all_rows(model, df: pd.DataFrame, model_name: str, use_pred: bool, device: torch.device, label: str):
    rows = []
    total = len(df)
    with torch.no_grad():
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            questions = [_first_int(x) for x in row.questions.split(",")]
            concepts = [_first_int(x) for x in row.concepts.split(",")]
            responses = [int(x) for x in row.responses.split(",")]
            if model_name == "iekt":
                rows.append(_iekt_predict_row(model, questions, concepts, responses, use_pred, device))
            elif model_name == "akt":
                rows.append(_akt_predict_row(model, questions, concepts, responses, use_pred, device))
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")
            if idx == 1 or idx % 100 == 0 or idx == total:
                print(f"{label}: predicted {idx}/{total} rows", flush=True)
    return rows


def _average_prediction_rows(all_model_rows):
    if len(all_model_rows) == 1:
        return all_model_rows[0]
    averaged = []
    for row_group in zip(*all_model_rows):
        lengths = {len(row) for row in row_group}
        if len(lengths) != 1:
            raise ValueError(f"Cannot average predictions with mismatched row lengths: {sorted(lengths)}")
        averaged.append([sum(values) / len(values) for values in zip(*row_group)])
    return averaged


def _cv_run_dirs(cv_dir: Path):
    summary_path = cv_dir / "cv_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        dirs = []
        for item in summary.get("per_fold", []):
            ckpt_dir = Path(item["ckpt_dir"])
            if not ckpt_dir.is_absolute():
                ckpt_dir = ROOT / ckpt_dir
            dirs.append(ckpt_dir)
        if dirs:
            return dirs
    return sorted([p for p in cv_dir.iterdir() if p.is_dir()])


@app.command()
def main(
    cv_dir: Optional[Path] = typer.Option(None, "--cv-dir", help="CV directory containing fold run directories; predictions are averaged."),
    ckpt_dir: Optional[Path] = typer.Option(None, "--ckpt-dir", help="Training run directory containing run_config.json and the best model."),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Direct path to a model checkpoint."),
    test_file: Path = typer.Option(ROOT / "data" / "peiyou" / "pykt_test.csv", "--test-file"),
    output: Path = typer.Option(ROOT / "prediction.csv", "--output"),
    dataset_name: str = typer.Option("peiyou", "--dataset-name"),
    model_name: str = typer.Option("iekt", "--model-name"),
    emb_type: Optional[str] = typer.Option(None, "--emb-type"),
    kt_config: Path = typer.Option(ROOT / "configs" / "kt_config.json", "--kt-config"),
    data_config: Path = typer.Option(ROOT / "configs" / "data_config.json", "--data-config"),
    gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU id. Omit to use CPU."),
    use_pred: bool = typer.Option(False, "--use-pred", help="Accumulative mode: update state with predicted labels for hidden targets."),
):
    if model_name not in {"iekt", "akt"}:
        raise typer.BadParameter("scripts/predict_peiyou.py currently supports IEKT and AKT checkpoints.")
    if cv_dir is not None and (ckpt_dir is not None or checkpoint is not None):
        raise typer.BadParameter("Use --cv-dir by itself, or use --ckpt-dir/--checkpoint for one model.")

    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    if model_name == "akt":
        import models.akt as akt_module

        akt_module.device = device

    df = pd.read_csv(test_file, dtype=str, keep_default_na=False)
    run_dirs = _cv_run_dirs(cv_dir) if cv_dir is not None else [ckpt_dir]
    print(f"Using device: {device}", flush=True)
    print(f"Loaded test file: {test_file} ({len(df)} rows)", flush=True)
    print(f"Models to predict: {len(run_dirs)}", flush=True)

    all_model_rows = []
    loaded = []
    for model_idx, run_dir in enumerate(run_dirs, start=1):
        run_config_path = run_dir / "run_config.json" if run_dir is not None else None
        model, resolved_emb_type = _build_model_from_config(
            dataset_name=dataset_name,
            model_name=model_name,
            emb_type=emb_type,
            run_config_path=run_config_path,
            kt_config_path=kt_config,
            data_config_path=data_config,
            device=device,
        )
        ckpt_path = _resolve_checkpoint(run_dir, checkpoint, model_name, resolved_emb_type)
        print(f"[{model_idx}/{len(run_dirs)}] Loading checkpoint: {ckpt_path}", flush=True)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        all_model_rows.append(
            _predict_all_rows(
                model,
                df,
                model_name,
                use_pred,
                device,
                label=f"[{model_idx}/{len(run_dirs)}]",
            )
        )
        loaded.append(ckpt_path)
        print(f"[{model_idx}/{len(run_dirs)}] Done", flush=True)

    print("Averaging predictions and writing output...", flush=True)
    averaged_rows = _average_prediction_rows(all_model_rows)
    rows = [",".join(f"{p:.10g}" for p in probs) for probs in averaged_rows]
    total_preds = sum(len(row) for row in averaged_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["responses"])
        for value in rows:
            writer.writerow([value])

    for ckpt_path in loaded:
        print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Wrote {len(rows)} rows and {total_preds} probabilities to: {output}")
    print("Zip prediction.csv before submission.")


if __name__ == "__main__":
    app()
