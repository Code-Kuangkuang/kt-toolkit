import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DATASET_DEFAULTS = {
    "assist2017": {
        "display": "ASSIST17",
        "folder": "assist2017",
        "pattern": "cv-assist2017-gbktv4-*",
        "split": "test",
    },
    "nips34": {
        "display": "NIPS34",
        "folder": "nips34",
        "pattern": "cv-nips_task34-gbktv4-*",
        "split": "test",
    },
    "peiyou": {
        "display": "AAAI2023/Peiyou",
        "folder": "peiyou",
        "pattern": "cv-peiyou-gbktv4-*",
        "split": "valid",
    },
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=4):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def latest_matching_dir(root: Path, pattern: str):
    matches = [p for p in root.glob(pattern) if p.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def resolve_runs(train_root: Path, datasets):
    runs = []
    for name in datasets:
        if name not in DATASET_DEFAULTS:
            raise SystemExit(f"Unknown dataset alias: {name}. Known: {', '.join(DATASET_DEFAULTS)}")
        spec = DATASET_DEFAULTS[name]
        dataset_root = train_root / spec["folder"]
        run_dir = latest_matching_dir(dataset_root, spec["pattern"])
        if run_dir is None:
            raise SystemExit(f"No GBKTV4 run found: {dataset_root / spec['pattern']}")
        runs.append(
            {
                "name": name,
                "display": spec["display"],
                "run_dir": run_dir,
                "split": spec["split"],
            }
        )
    return runs


def analysis_dir(output_root: Path, run_dir: Path, split: str):
    return output_root / run_dir.name / split


def run_interpretability(run, args):
    out_dir = analysis_dir(args.output_dir, run["run_dir"], run["split"])
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"Skip existing: {summary_path}")
        return

    command = [
        sys.executable,
        str(ROOT / "scripts" / "gbkt_interpretability.py"),
        "--run-dir",
        str(run["run_dir"]),
        "--split",
        run["split"],
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
        "--bins",
        str(args.bins),
        "--min-item-count",
        str(args.min_item_count),
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.no_plots:
        command.append("--no-plots")
    if args.max_runs is not None:
        command.extend(["--max-runs", str(args.max_runs)])
    if args.max_batches is not None:
        command.extend(["--max-batches", str(args.max_batches)])

    print("Running: " + " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def read_corr(corr_df: pd.DataFrame, x: str, y: str):
    if corr_df.empty:
        return math.nan
    rows = corr_df[(corr_df["x"] == x) & (corr_df["y"] == y)]
    if rows.empty:
        return math.nan
    return float(rows.iloc[0].get("spearman", math.nan))


def judge_corr(value, expected: str):
    if value is None or not math.isfinite(value):
        return "insufficient"
    sign_ok = value > 0 if expected == "positive" else value < 0
    mag = abs(value)
    if sign_ok and mag >= 0.10:
        return "support"
    if sign_ok and mag >= 0.03:
        return "weak support"
    if mag < 0.03:
        return "flat"
    if mag >= 0.10:
        return "opposite"
    return "weak opposite"


def overall_judgement(items):
    support = sum(1 for item in items if item in {"support", "weak support"})
    opposite = sum(1 for item in items if item in {"opposite", "weak opposite"})
    if support >= 2 and opposite == 0:
        return "supports"
    if support >= 1 and opposite == 0:
        return "weakly supports"
    if opposite >= 2:
        return "contradicts"
    if support > 0 and opposite > 0:
        return "mixed"
    return "weak/flat"


def top_bin_delta(bin_df: pd.DataFrame, feature: str, target_col: str):
    part = bin_df[bin_df["feature"] == feature].copy()
    if part.empty or target_col not in part:
        return math.nan
    part = part.sort_values("bin")
    return float(part.iloc[-1][target_col] - part.iloc[0][target_col])


def collect_dataset_report(run, output_root: Path):
    out_dir = analysis_dir(output_root, run["run_dir"], run["split"])
    summary_path = out_dir / "summary.json"
    corr_path = out_dir / "correlations.csv"
    bins_path = out_dir / "feature_bins.csv"
    history_path = out_dir / "history_length_bins.csv"
    item_path = out_dir / "item_radius_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    summary = load_json(summary_path)
    corr_df = pd.read_csv(corr_path) if corr_path.exists() else pd.DataFrame()
    bins_df = pd.read_csv(bins_path) if bins_path.exists() else pd.DataFrame()
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    item_df = pd.read_csv(item_path) if item_path.exists() else pd.DataFrame()

    metrics = {
        "pred_ball_target": read_corr(corr_df, "pred_ball", "target"),
        "theta_target": read_corr(corr_df, "theta", "target"),
        "pred_ball_abs_error": read_corr(corr_df, "pred_ball", "abs_error"),
        "theta_abs_error": read_corr(corr_df, "theta", "abs_error"),
        "student_radius_abs_error": read_corr(corr_df, "r_h_mean", "abs_error"),
        "student_radius_bce": read_corr(corr_df, "r_h_mean", "bce"),
        "student_radius_confidence": read_corr(corr_df, "r_h_mean", "confidence"),
        "history_radius": read_corr(corr_df, "history_len", "r_h_mean"),
        "norm_dist_target": read_corr(corr_df, "normalized_dist", "target"),
        "norm_dist_pred": read_corr(corr_df, "normalized_dist", "pred"),
        "overlap_target": read_corr(corr_df, "overlap_margin", "target"),
        "item_radius_entropy": read_corr(corr_df, "r_d_mean", "response_entropy"),
        "item_radius_concept_count": read_corr(corr_df, "r_d_mean", "concept_count"),
    }

    judgements = {
        "student_uncertainty": overall_judgement(
            [
                judge_corr(metrics["student_radius_abs_error"], "positive"),
                judge_corr(metrics["student_radius_bce"], "positive"),
                judge_corr(metrics["student_radius_confidence"], "negative"),
            ]
        ),
        "history_uncertainty": judge_corr(metrics["history_radius"], "negative"),
        "geometric_matching": overall_judgement(
            [
                judge_corr(metrics["norm_dist_target"], "negative"),
                judge_corr(metrics["norm_dist_pred"], "negative"),
                judge_corr(metrics["overlap_target"], "positive"),
            ]
        ),
        "question_coverage": overall_judgement(
            [
                judge_corr(metrics["item_radius_entropy"], "positive"),
                judge_corr(metrics["item_radius_concept_count"], "positive"),
            ]
        ),
        "bbp_matching": overall_judgement(
            [
                judge_corr(metrics["pred_ball_target"], "positive"),
                judge_corr(metrics["theta_target"], "positive"),
            ]
        ),
    }

    deltas = {
        "pred_ball_top_minus_bottom_correct": top_bin_delta(bins_df, "pred_ball", "correct_rate"),
        "theta_top_minus_bottom_correct": top_bin_delta(bins_df, "theta", "correct_rate"),
        "r_h_top_minus_bottom_abs_error": top_bin_delta(bins_df, "r_h_mean", "abs_error"),
        "r_h_top_minus_bottom_confidence": top_bin_delta(bins_df, "r_h_mean", "confidence"),
        "norm_dist_top_minus_bottom_correct": top_bin_delta(bins_df, "normalized_dist", "correct_rate"),
    }

    return {
        "run": run,
        "out_dir": out_dir,
        "summary": summary,
        "metrics": metrics,
        "judgements": judgements,
        "deltas": deltas,
        "history_rows": int(len(history_df)),
        "item_rows": int(len(item_df)),
    }


def table_row(cells):
    return "| " + " | ".join(cells) + " |"


def write_report(reports, output_root: Path):
    lines = []
    lines.append("# GBKT 可解释性自动分析报告")
    lines.append("")
    lines.append("本报告基于训练好的 GBKTV4 checkpoint 做推理分析，不重新训练模型。")
    lines.append("")

    lines.append("## 运行概览")
    lines.append("")
    lines.append(table_row(["Dataset", "Split", "Rows", "AUC", "Ball AUC", "r_h", "r_d", "Output"]))
    lines.append(table_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]))
    for report in reports:
        overall = report["summary"]["overall"]
        lines.append(
            table_row(
                [
                    report["run"]["display"],
                    report["run"]["split"],
                    str(report["summary"]["num_rows"]),
                    fmt(overall.get("auc")),
                    fmt(overall.get("ball_auc")),
                    fmt(overall.get("r_h_mean")),
                    fmt(overall.get("r_d_mean")),
                    str(report["out_dir"]),
                ]
            )
        )
    lines.append("")

    lines.append("## 假设检验")
    lines.append("")
    lines.append(
        table_row(
            [
                "Dataset",
                "Student Radius=Uncertainty",
                "History Len vs Radius",
                "BBP Score Matching",
                "Ball Distance Matching",
                "Question Radius=Coverage",
            ]
        )
    )
    lines.append(table_row(["---", "---", "---", "---", "---", "---"]))
    for report in reports:
        j = report["judgements"]
        lines.append(
            table_row(
                [
                    report["run"]["display"],
                    j["student_uncertainty"],
                    j["history_uncertainty"],
                    j["bbp_matching"],
                    j["geometric_matching"],
                    j["question_coverage"],
                ]
            )
        )
    lines.append("")

    lines.append("## 关键相关性 Spearman")
    lines.append("")
    lines.append(
        table_row(
            [
                "Dataset",
                "BBP-y",
                "theta-y",
                "r_h-error",
                "r_h-BCE",
                "r_h-conf",
                "hist-r_h",
                "normDist-y",
                "overlap-y",
                "r_d-entropy",
                "r_d-concepts",
            ]
        )
    )
    lines.append(table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for report in reports:
        m = report["metrics"]
        lines.append(
            table_row(
                [
                    report["run"]["display"],
                    fmt(m["pred_ball_target"], 3),
                    fmt(m["theta_target"], 3),
                    fmt(m["student_radius_abs_error"], 3),
                    fmt(m["student_radius_bce"], 3),
                    fmt(m["student_radius_confidence"], 3),
                    fmt(m["history_radius"], 3),
                    fmt(m["norm_dist_target"], 3),
                    fmt(m["overlap_target"], 3),
                    fmt(m["item_radius_entropy"], 3),
                    fmt(m["item_radius_concept_count"], 3),
                ]
            )
        )
    lines.append("")

    lines.append("## 分桶差异")
    lines.append("")
    lines.append("这里比较最高分桶和最低分桶，便于直接写图注或正文。")
    lines.append("")
    lines.append(
        table_row(
            [
                "Dataset",
                "High BBP - Low BBP Correct",
                "High theta - Low theta Correct",
                "High r_h - Low r_h AbsErr",
                "High r_h - Low r_h Conf",
                "High NormDist - Low NormDist Correct",
            ]
        )
    )
    lines.append(table_row(["---", "---:", "---:", "---:", "---:", "---:"]))
    for report in reports:
        d = report["deltas"]
        lines.append(
            table_row(
                [
                    report["run"]["display"],
                    fmt(d["pred_ball_top_minus_bottom_correct"], 4),
                    fmt(d["theta_top_minus_bottom_correct"], 4),
                    fmt(d["r_h_top_minus_bottom_abs_error"], 4),
                    fmt(d["r_h_top_minus_bottom_confidence"], 4),
                    fmt(d["norm_dist_top_minus_bottom_correct"], 4),
                ]
            )
        )
    lines.append("")

    lines.append("## 自动结论")
    lines.append("")
    for report in reports:
        j = report["judgements"]
        lines.append(f"### {report['run']['display']}")
        lines.append("")
        if j["student_uncertainty"] in {"supports", "weakly supports"}:
            lines.append("- 学生半径与预测误差/置信度关系支持“学生半径刻画状态不确定性”的解释。")
        elif j["student_uncertainty"] == "mixed":
            lines.append("- 学生半径的不确定性证据是混合的，建议正文谨慎表述，更多强调可分析性而非强因果解释。")
        else:
            lines.append("- 学生半径的不确定性证据较弱，暂时不建议强写“半径越大越不确定”。")

        if j["bbp_matching"] in {"supports", "weakly supports"}:
            lines.append("- BBP 几何匹配分数与真实正确率方向稳定一致，可以作为正文中最主要的可解释性证据。")
        else:
            lines.append("- BBP 几何匹配分数证据不足，需要回看 `pred_ball` 和 `theta` 的分桶曲线。")

        if j["geometric_matching"] in {"supports", "weakly supports"}:
            lines.append("- 球距离/重叠关系与真实正确率方向一致，可以支撑学生-题目几何匹配解释。")
        elif j["geometric_matching"] == "mixed":
            lines.append("- 球距离匹配证据混合，建议先看 `feature_bins.csv` 的分桶曲线是否单调。")
        else:
            lines.append("- 球距离匹配证据较弱，不宜把 AUC 主要归因于半径距离。")

        if j["question_coverage"] in {"supports", "weakly supports"}:
            lines.append("- 题目半径与题目响应熵或概念数相关，能支持“题目覆盖范围/模糊性”解释。")
        else:
            lines.append("- 题目半径与题目复杂度的证据较弱，可作为补充分析而不是主结论。")
        lines.append("")

    lines.append("## 推荐论文表述")
    lines.append("")
    lines.append(
        "更稳的写法是：GBKT 的中心几何匹配提供主要预测能力；球半径提供可解释的不确定性和覆盖范围建模。"
        "如果上面的相关性在多个数据集上方向一致，可以进一步写成：半径与预测不确定性、题目响应模糊性或球距离匹配存在一致关联。"
    )
    lines.append("")

    report_path = output_root / "gbkt_interpretability_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run GBKT interpretability analyses and summarize them.")
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path(r"E:\project\knowledgeTracing\train_model"),
        help="Root directory containing dataset CV folders.",
    )
    parser.add_argument(
        "--datasets",
        default="assist2017,nips34,peiyou",
        help="Comma-separated aliases: assist2017,nips34,peiyou.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/gbkt_interpretability_suite"))
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--min-item-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only summarize existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="Debug option passed to child script.")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug option passed to child script.")
    args = parser.parse_args()

    args.output_dir = (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    runs = resolve_runs(args.train_root, datasets)

    print("Resolved runs:")
    for run in runs:
        print(f"  {run['display']}: {run['run_dir']} ({run['split']})")

    if not args.skip_run:
        for run in runs:
            run_interpretability(run, args)

    if args.dry_run:
        return

    reports = [collect_dataset_report(run, args.output_dir) for run in runs]
    report_path = write_report(reports, args.output_dir)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
