"""Plot a single student's a removed model knowledge-state trajectory with Seaborn.

The heatmaps visualize a coverage-based mastery proxy in [0, 1].  They do not
reinterpret the proxy as a calibrated per-concept response probability.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis_utils import (  # noqa: E402
    build_run_model,
    resolve_device,
    resolve_local_data_path,
)


CONTINUOUS_CMAPS = {"viridis", "mako", "rocket"}
DIVERGING_CMAPS = {"vlag", "coolwarm"}
SUPPORTED_CMAPS = tuple(sorted(CONTINUOUS_CMAPS | DIVERGING_CMAPS))
SUPPORTED_FORMATS = {"png", "pdf", "svg"}
MASTERY_LABEL = "Coverage-based mastery proxy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one student's a removed model knowledge-state trajectory."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--uid", required=True, help="Student uid in the sequence CSV.")
    parser.add_argument(
        "--segment-index",
        type=int,
        default=0,
        help="Zero-based row index when one uid has multiple sequence segments.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Zero-based first interaction to display; earlier history is retained.",
    )
    parser.add_argument("--span", type=int, default=30)
    parser.add_argument(
        "--selected-concepts",
        default="",
        help="Comma-separated concept ids; defaults to the four most frequent.",
    )
    parser.add_argument(
        "--target-concept",
        type=int,
        default=None,
        help="Concept for the line plot; defaults to the most frequent concept.",
    )
    parser.add_argument("--cmap", choices=SUPPORTED_CMAPS, default="viridis")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "student_knowledge_state",
    )
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_integer_sequence(value: Any) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    return [int(float(item)) for item in str(value).split(",") if item != ""]


def parse_concept_sequence(value: Any) -> list[list[int]]:
    if value is None or str(value).strip() == "":
        return []
    result: list[list[int]] = []
    for item in str(value).split(","):
        concepts = [
            int(float(concept))
            for concept in item.split("_")
            if concept != "" and int(float(concept)) >= 0
        ]
        result.append(concepts)
    return result


def parse_selected_concepts(value: str) -> list[int]:
    if not value.strip():
        return []
    concepts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(concepts) != len(set(concepts)):
        raise ValueError(f"Selected concepts contain duplicates: {concepts}")
    return concepts


def resolve_sequence_file(run_config: dict[str, Any], split: str) -> Path:
    dataset_cfg = dict(run_config["dataset_config"])
    data_dir = resolve_local_data_path(run_config)
    dataset_mode = str(run_config["train_config"].get("dataset_mode", ""))
    question_level = dataset_mode == "all_in_one"
    if split == "test":
        keys = (
            ("test_file_quelevel", "test_file")
            if question_level
            else ("test_file", "test_file_quelevel")
        )
    else:
        keys = (
            ("train_valid_file_quelevel", "train_valid_file")
            if question_level
            else ("train_valid_file", "train_valid_file_quelevel")
        )
    for key in keys:
        filename = dataset_cfg.get(key)
        if filename and (data_dir / str(filename)).exists():
            return (data_dir / str(filename)).resolve()
    raise FileNotFoundError(
        f"No sequence file found for split={split}; checked keys={keys} in {data_dir}"
    )


def load_student_sequence(
    run_config: dict[str, Any],
    split: str,
    uid: str,
    segment_index: int,
    start_step: int,
    span: int,
) -> dict[str, Any]:
    if segment_index < 0:
        raise ValueError("segment-index must be non-negative.")
    if span <= 0:
        raise ValueError("span must be positive.")
    if start_step < 0:
        raise ValueError("start-step must be non-negative.")

    sequence_file = resolve_sequence_file(run_config, split)
    frame = pd.read_csv(sequence_file, dtype=str, keep_default_na=False)
    required = {"uid", "questions", "concepts", "responses"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Sequence file is missing columns: {sorted(missing)}")

    fold = int(run_config["fold"])
    if "fold" in frame.columns:
        fold_values = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        if split == "valid":
            frame = frame[fold_values.eq(fold)]
        elif split == "train":
            frame = frame[fold_values.ge(0) & fold_values.ne(fold)]
        else:
            frame = frame[fold_values.eq(-1)]

    matches = frame[frame["uid"].astype(str).eq(str(uid))]
    if matches.empty:
        raise ValueError(
            f"uid={uid!r} was not found in split={split} file {sequence_file}"
        )
    if segment_index >= len(matches):
        raise IndexError(
            f"uid={uid!r} has {len(matches)} segment(s); "
            f"segment-index={segment_index} is out of range."
        )
    row = matches.iloc[segment_index]

    questions = parse_integer_sequence(row["questions"])
    responses = parse_integer_sequence(row["responses"])
    concepts = parse_concept_sequence(row["concepts"])
    if not (len(questions) == len(responses) == len(concepts)):
        raise ValueError(
            "Sequence columns have different lengths: "
            f"questions={len(questions)}, concepts={len(concepts)}, "
            f"responses={len(responses)}"
        )

    valid = [
        question >= 0 and response in (0, 1) and bool(concept_ids)
        for question, response, concept_ids in zip(questions, responses, concepts)
    ]
    valid_length = next((index for index, flag in enumerate(valid) if not flag), len(valid))
    if any(valid[valid_length:]):
        raise ValueError("Found a valid interaction after padding/invalid data.")
    if valid_length == 0:
        raise ValueError(f"uid={uid!r} has no valid interactions to plot.")
    if start_step >= valid_length:
        raise IndexError(
            f"start-step={start_step} is outside the valid sequence length "
            f"{valid_length}."
        )
    window_end = min(valid_length, start_step + span)

    # Preserve all preceding interactions so the state at start_step has its
    # true causal history instead of being reset at the displayed window.
    questions = questions[:window_end]
    responses = responses[:window_end]
    concepts = concepts[:window_end]
    num_c = int(run_config["dataset_config"]["num_c"])
    num_q = int(run_config["dataset_config"]["num_q"])
    for index, question in enumerate(questions):
        if not 0 <= question < num_q:
            raise ValueError(f"Question id out of range at step {index}: {question}")
    for index, concept_ids in enumerate(concepts):
        invalid = [concept for concept in concept_ids if not 0 <= concept < num_c]
        if invalid:
            raise ValueError(f"Concept ids out of range at step {index}: {invalid}")

    width = max(len(concept_ids) for concept_ids in concepts)
    padded_concepts = [
        concept_ids + [-1] * (width - len(concept_ids))
        for concept_ids in concepts
    ]
    return {
        "uid": str(uid),
        "segment_index": int(segment_index),
        "split": split,
        "sequence_file": sequence_file,
        "window_start": int(start_step),
        "window_end": int(window_end),
        "questions": questions,
        "concepts": concepts,
        "padded_concepts": padded_concepts,
        "responses": responses,
    }


def heatmap_kwargs(cmap: str) -> dict[str, Any]:
    if cmap not in CONTINUOUS_CMAPS | DIVERGING_CMAPS:
        raise ValueError(f"Unsupported cmap={cmap!r}; choose from {SUPPORTED_CMAPS}")
    kwargs: dict[str, Any] = {
        "cmap": cmap,
        "vmin": 0.0,
        "vmax": 1.0,
    }
    if cmap in DIVERGING_CMAPS:
        kwargs["center"] = 0.5
    return kwargs


def validate_mastery(mastery: np.ndarray) -> None:
    if mastery.ndim != 2:
        raise ValueError(f"Expected mastery [T, C], got {mastery.shape}")
    if not np.isfinite(mastery).all():
        raise ValueError("Mastery matrix contains NaN or Inf values.")
    if mastery.min() < 0.0 or mastery.max() > 1.0:
        raise ValueError(
            f"Mastery matrix must lie in [0, 1], got [{mastery.min()}, {mastery.max()}]"
        )


def configure_step_ticks(ax, steps: int, step_offset: int = 0) -> None:
    stride = max(1, math.ceil(steps / 30))
    indices = np.arange(0, steps, stride)
    ax.set_xticks(indices + 0.5)
    ax.set_xticklabels(indices + step_offset, rotation=0)
    ax.tick_params(axis="x", length=0)


def save_figure(
    fig,
    output_dir: Path,
    basename: str,
    formats: list[str],
    dpi: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {fmt}")
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def plot_related_concepts(
    mastery: np.ndarray,
    concepts: list[list[int]],
    responses: list[int],
    related: list[int],
    cmap: str,
    step_offset: int = 0,
):
    data = mastery[:, related].T
    width = max(10.0, 0.55 * mastery.shape[0])
    height = max(3.0, 0.48 * len(related) + 1.5)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        data,
        ax=ax,
        **heatmap_kwargs(cmap),
        cbar_kws={"label": MASTERY_LABEL, "shrink": 0.78, "pad": 0.02},
        xticklabels=False,
        yticklabels=[f"c{concept}" for concept in related],
        linewidths=0,
    )
    configure_step_ticks(ax, mastery.shape[0], step_offset=step_offset)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Related concept")
    ax.set_title("Knowledge-state changes on encountered concepts", pad=34)

    identity_colors = sns.color_palette("colorblind", n_colors=len(related))
    color_by_concept = dict(zip(related, identity_colors))
    for label, color in zip(ax.get_yticklabels(), identity_colors):
        label.set_color(color)
    for step, (concept_ids, response) in enumerate(zip(concepts, responses)):
        count = len(concept_ids)
        offsets = np.linspace(-0.22, 0.22, count) if count > 1 else [0.0]
        for offset, concept in zip(offsets, concept_ids):
            color = color_by_concept[concept]
            ax.scatter(
                step + 0.5 + offset,
                -0.52,
                s=55,
                facecolor=color if response == 1 else "white",
                edgecolor=color,
                linewidth=1.8,
                clip_on=False,
                zorder=5,
            )
    ax.set_ylim(len(related), -0.95)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markeredgecolor="black", label="Correct"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", label="Incorrect"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.2),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(pad=1.2)
    return fig


def plot_selected_concepts(
    mastery: np.ndarray,
    questions: list[int],
    responses: list[int],
    selected: list[int],
    cmap: str,
    step_offset: int = 0,
):
    data = mastery[:, selected].T
    width = max(10.0, 0.58 * mastery.shape[0])
    height = max(3.0, 0.52 * len(selected) + 1.7)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        data,
        ax=ax,
        **heatmap_kwargs(cmap),
        cbar_kws={"label": MASTERY_LABEL, "shrink": 0.78, "pad": 0.02},
        xticklabels=False,
        yticklabels=[f"c{concept}" for concept in selected],
        linewidths=0,
    )
    configure_step_ticks(ax, mastery.shape[0], step_offset=step_offset)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Selected concept")
    ax.set_title("Knowledge-state changes on selected concepts", pad=50)
    fontsize = 8 if mastery.shape[0] <= 35 else 6
    for step, (question, response) in enumerate(zip(questions, responses)):
        ax.text(
            step + 0.5,
            -1.02,
            f"q{question}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#343434",
            clip_on=False,
        )
        ax.text(
            step + 0.5,
            -0.38,
            "✓" if response == 1 else "×",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1B7837" if response == 1 else "#B2182B",
            clip_on=False,
        )
    ax.set_ylim(len(selected), -1.35)
    fig.tight_layout(pad=1.2)
    return fig


def plot_single_concept(
    mastery: np.ndarray,
    concepts: list[list[int]],
    responses: list[int],
    target_concept: int,
    step_offset: int = 0,
):
    values = mastery[:, target_concept]
    steps = np.arange(step_offset, step_offset + len(values))
    relation = np.asarray(
        [
            1.0 / len(concept_ids) if target_concept in concept_ids else 0.0
            for concept_ids in concepts
        ],
        dtype=float,
    )
    width = max(10.0, 0.55 * len(values))
    fig, ax = plt.subplots(figsize=(width, 4.1))
    sns.lineplot(
        x=steps,
        y=values,
        ax=ax,
        color="#0F4D92",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    ax.vlines(steps, 0.0, values, colors="#B8B8B8", linestyles="dashed", linewidth=0.8, alpha=0.55)
    ax.set_xlim(step_offset - 0.5, step_offset + len(values) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Interaction step", labelpad=42)
    ax.set_ylabel(MASTERY_LABEL)
    ax.set_title(f"Knowledge tracing over time on c{target_concept}")
    ax.grid(axis="y", alpha=0.18)

    marker_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for local_step, (response, strength) in enumerate(zip(responses, relation)):
        step = step_offset + local_step
        related = strength > 0.0
        ax.scatter(
            step,
            -0.16,
            marker="o" if response == 1 else "X",
            s=42 + 55 * strength,
            facecolor="#1B7837" if response == 1 else "#B2182B",
            edgecolor="black" if related else "#777777",
            linewidth=1.0 if related else 0.5,
            alpha=0.3 + 0.7 * strength if related else 0.28,
            transform=marker_transform,
            clip_on=False,
            zorder=5,
        )
    fig.subplots_adjust(bottom=0.34)
    return fig


def plot_geometry_contribution_case(
    mastery: np.ndarray,
    questions: list[int],
    responses: list[int],
    selected: list[int],
    full_probability: np.ndarray,
    base_probability: np.ndarray,
    cmap: str,
    step_offset: int = 0,
):
    """Combine mastery states with the target-aligned geometry correction."""
    target = np.asarray(responses, dtype=float)
    improvement = (2.0 * target - 1.0) * (full_probability - base_probability)
    if step_offset == 0:
        # The training/evaluation contract starts at the second interaction.
        improvement[0] = np.nan
    finite = improvement[np.isfinite(improvement)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
    contribution_limit = max(0.05, math.ceil(max_abs / 0.05) * 0.05)
    contribution_limit = min(1.0, contribution_limit)

    width = max(10.0, 0.58 * mastery.shape[0])
    height = max(4.4, 0.52 * len(selected) + 2.8)
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(2, 1, height_ratios=[max(len(selected), 2), 1], hspace=0.34)
    state_ax = fig.add_subplot(grid[0])
    contribution_ax = fig.add_subplot(grid[1])

    sns.heatmap(
        mastery[:, selected].T,
        ax=state_ax,
        **heatmap_kwargs(cmap),
        cbar_kws={"label": MASTERY_LABEL, "shrink": 0.82, "pad": 0.02},
        xticklabels=False,
        yticklabels=[f"c{concept}" for concept in selected],
        linewidths=0,
    )
    configure_step_ticks(state_ax, mastery.shape[0], step_offset=step_offset)
    state_ax.set_xlabel("")
    state_ax.set_ylabel("Selected concept")
    fontsize = 8 if mastery.shape[0] <= 35 else 6
    for local_step, (question, response) in enumerate(zip(questions, responses)):
        state_ax.text(
            local_step + 0.5,
            -0.95,
            f"q{question}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#343434",
            clip_on=False,
        )
        state_ax.text(
            local_step + 0.5,
            -0.34,
            "✓" if response == 1 else "×",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1B7837" if response == 1 else "#B2182B",
            clip_on=False,
        )
    state_ax.set_ylim(len(selected), -1.22)

    sns.heatmap(
        improvement.reshape(1, -1),
        ax=contribution_ax,
        cmap="vlag",
        vmin=-contribution_limit,
        vmax=contribution_limit,
        center=0.0,
        cbar_kws={
            "label": "Target-aligned probability shift",
            "shrink": 0.88,
            "pad": 0.02,
        },
        xticklabels=False,
        yticklabels=["Geometry\ncorrection"],
        linewidths=0,
    )
    configure_step_ticks(
        contribution_ax,
        mastery.shape[0],
        step_offset=step_offset,
    )
    contribution_ax.set_xlabel("Interaction step")
    contribution_ax.set_ylabel("")
    contribution_ax.set_title(
        "Positive values move the prediction toward the observed response",
        fontsize=10,
        pad=7,
    )
    fig.suptitle("a removed model geometry-aware student-state case", fontsize=15, y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.08, right=0.93)
    return fig, contribution_limit, improvement


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    if not formats:
        raise ValueError("At least one output format is required.")

    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        font_scale=1.05,
        rc={"svg.fonttype": "none", "axes.linewidth": 1.2},
    )
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    run_config, model, checkpoint = build_run_model(args.run_dir, device)
    if str(run_config["model_name"]).lower() != "removed_model":
        raise ValueError(
            f"This visualization requires a removed model, got {run_config['model_name']!r}."
        )
    if not hasattr(model, "get_state_trajectory") or not hasattr(
        model, "score_concept_mastery"
    ):
        raise TypeError("Loaded a removed model model lacks the state-trajectory API.")

    sequence = load_student_sequence(
        run_config=run_config,
        split=args.split,
        uid=args.uid,
        segment_index=args.segment_index,
        start_step=args.start_step,
        span=args.span,
    )
    questions = torch.tensor(
        [sequence["questions"]], device=device, dtype=torch.long
    )
    concepts = torch.tensor(
        [sequence["padded_concepts"]], device=device, dtype=torch.long
    )
    responses = torch.tensor(
        [sequence["responses"]], device=device, dtype=torch.float32
    )
    with torch.inference_mode():
        trajectory = model.get_state_trajectory(questions, concepts, responses)
        scored = model.score_concept_mastery(
            trajectory["student_point"], trajectory["student_radius"]
        )
    if not bool(trajectory["valid"].all()):
        raise ValueError("The selected student sequence contains invalid/padded steps.")
    mastery_full = scored["mastery"][0].detach().float().cpu().numpy()
    window_start = int(sequence["window_start"])
    window_end = int(sequence["window_end"])
    mastery = mastery_full[window_start:window_end]
    validate_mastery(mastery)

    window_questions = sequence["questions"][window_start:window_end]
    window_concepts = sequence["concepts"][window_start:window_end]
    window_responses = sequence["responses"][window_start:window_end]
    observed = [concept for step in window_concepts for concept in step]
    related = sorted(set(observed))
    frequency_order = [
        concept
        for concept, _ in sorted(
            Counter(observed).items(), key=lambda pair: (-pair[1], pair[0])
        )
    ]
    selected = parse_selected_concepts(args.selected_concepts) or frequency_order[:4]
    target = args.target_concept if args.target_concept is not None else frequency_order[0]
    num_c = mastery.shape[1]
    invalid_selected = [concept for concept in selected if not 0 <= concept < num_c]
    if invalid_selected:
        raise ValueError(f"Selected concepts out of range: {invalid_selected}")
    if not 0 <= target < num_c:
        raise ValueError(f"target-concept must lie in [0, {num_c - 1}], got {target}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, list[str]] = {}
    figures["related_concepts"] = save_figure(
        plot_related_concepts(
            mastery,
            window_concepts,
            window_responses,
            related,
            args.cmap,
            step_offset=window_start,
        ),
        output_dir,
        "related_concepts_heatmap",
        formats,
        args.dpi,
    )
    figures["selected_concepts"] = save_figure(
        plot_selected_concepts(
            mastery,
            window_questions,
            window_responses,
            selected,
            args.cmap,
            step_offset=window_start,
        ),
        output_dir,
        "selected_concepts_heatmap",
        formats,
        args.dpi,
    )
    figures["single_concept"] = save_figure(
        plot_single_concept(
            mastery,
            window_concepts,
            window_responses,
            target,
            step_offset=window_start,
        ),
        output_dir,
        "single_concept_trajectory",
        formats,
        args.dpi,
    )
    full_probability = (
        trajectory["pre_probability"][0, window_start:window_end]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    base_probability = (
        trajectory["pre_base_probability"][0, window_start:window_end]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    evaluation_mask = np.arange(window_start, window_end) > 0
    evaluation_target = np.asarray(window_responses, dtype=float)[evaluation_mask]
    evaluation_full = np.clip(
        full_probability[evaluation_mask], 1e-7, 1.0 - 1e-7
    )
    evaluation_base = np.clip(
        base_probability[evaluation_mask], 1e-7, 1.0 - 1e-7
    )
    full_log_loss = float(
        -np.mean(
            evaluation_target * np.log(evaluation_full)
            + (1.0 - evaluation_target) * np.log(1.0 - evaluation_full)
        )
    )
    base_log_loss = float(
        -np.mean(
            evaluation_target * np.log(evaluation_base)
            + (1.0 - evaluation_target) * np.log(1.0 - evaluation_base)
        )
    )
    geometry_figure, contribution_limit, aligned_improvement = (
        plot_geometry_contribution_case(
            mastery,
            window_questions,
            window_responses,
            selected,
            full_probability,
            base_probability,
            args.cmap,
            step_offset=window_start,
        )
    )
    finite_improvement = aligned_improvement[np.isfinite(aligned_improvement)]
    figures["geometry_contribution"] = save_figure(
        geometry_figure,
        output_dir,
        "geometry_contribution_heatmap",
        formats,
        args.dpi,
    )

    mastery_frame = pd.DataFrame(
        mastery, columns=[f"c{concept}" for concept in range(num_c)]
    )
    mastery_frame.insert(
        0,
        "step",
        np.arange(window_start, window_start + len(mastery_frame)),
    )
    mastery_frame.to_csv(output_dir / "mastery_proxy.csv", index=False)
    sequence_frame = pd.DataFrame(
        {
            "step": np.arange(window_start, window_end),
            "question_id": window_questions,
            "concept_ids": ["_".join(map(str, item)) for item in window_concepts],
            "response": window_responses,
            "student_radius": trajectory["student_radius"][0, window_start:window_end]
            .detach()
            .float()
            .cpu()
            .numpy(),
            "response_update_gate": trajectory["response_update_gate"][
                0, window_start:window_end
            ]
            .detach()
            .float()
            .cpu()
            .numpy(),
        }
    )
    sequence_frame.to_csv(output_dir / "interaction_sequence.csv", index=False)

    manifest = {
        "analysis": "single-student a removed model knowledge-state trajectory",
        "score_name": MASTERY_LABEL,
        "score_definition": "sigmoid(beta * (student_radius - concept_center_distance))",
        "score_warning": (
            "Geometric coverage proxy; not a calibrated per-concept response probability."
        ),
        "causal_alignment": "column t is the post-response state after interaction t",
        "dataset": run_config["dataset_name"],
        "model": run_config["model_name"],
        "fold": int(run_config["fold"]),
        "uid": sequence["uid"],
        "segment_index": sequence["segment_index"],
        "split": sequence["split"],
        "start_step": window_start,
        "end_step_exclusive": window_end,
        "span": len(window_questions),
        "related_concepts": related,
        "selected_concepts": selected,
        "target_concept": target,
        "checkpoint": str(checkpoint.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "sequence_file": str(sequence["sequence_file"]),
        "device": str(device),
        "heatmap": {
            "library": "seaborn",
            "cmap": args.cmap,
            "vmin": 0.0,
            "vmax": 1.0,
            "center": 0.5 if args.cmap in DIVERGING_CMAPS else None,
        },
        "geometry_contribution": {
            "definition": "(2 * response - 1) * (full_probability - base_probability)",
            "interpretation": "positive values move prediction toward the observed response",
            "color_limit": contribution_limit,
            "mean": float(np.mean(finite_improvement)),
            "positive_share": float(np.mean(finite_improvement > 0.0)),
            "full_log_loss": full_log_loss,
            "base_log_loss": base_log_loss,
            "log_loss_gain": base_log_loss - full_log_loss,
        },
        "figures": figures,
        "tables": [
            str((output_dir / "mastery_proxy.csv").resolve()),
            str((output_dir / "interaction_sequence.csv").resolve()),
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"Completed: {output_dir}")


if __name__ == "__main__":
    main()
