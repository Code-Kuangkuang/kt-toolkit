"""Build navigation for existing experiments without moving or loading checkpoints.

Run from any directory: python scripts/catalog_experiments.py
Only generated README/index files and saved_model/_catalog are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


GROUPS = {
    "baseline": "基线与收敛检查",
    "fokt": "FO-KT 遗忘模型",
    "item": "题目参数、难度与分组",
    "ablation": "模型结构与消融",
    "probe": "诊断与探索分析",
    "smoke": "调试与运行验证",
    "other": "待归类",
}


def category(name):
    name = name.lower().lstrip("_")
    if "smoke" in name or "pytest" in name or name.startswith("review"):
        return "smoke"
    if name.startswith("fokt"):
        return "fokt"
    if name.startswith(("itemparam", "item_parameter", "grp", "alpha", "item_state", "xds_algebra", "xds_assist")):
        return "item"
    if name.startswith(("abl_", "hdkt_ablation", "exp_", "simplekt_delta", "model_lineage")):
        return "ablation"
    if name.startswith(("baseline", "akt_converged", "akt_familycfg", "fixed_", "atdkt_fixed", "atkt_fixed", "ednet", "statics2011")) or name in {"assist2012", "bridge2algebra2006"}:
        return "baseline"
    if name.startswith(("null", "pdiff", "cost", "xds_probe", "evidence", "delta", "prereq", "forgetting")):
        return "probe"
    return "other"


def load_json(path, errors):
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict):
            raise ValueError("expected JSON object")
        return obj
    except (OSError, ValueError) as exc:
        errors.append({"path": str(path), "error": str(exc)})
        return {}


def link(path, base, label=None):
    relative = Path(os.path.relpath(path, base)).as_posix()
    return f"[{label or path.name}]({quote(relative, safe='/._-')})"


def cell(value):
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def revision_note(cfg):
    model, emb = cfg.get("model_name"), cfg.get("emb_type", "") or ""
    if cfg.get("model_correctness_revision"):
        return "已记录修复版本 " + str(cfg["model_correctness_revision"])
    affected = model in {"atdkt", "keenkt"} or (model == "fakt" and "band" in emb) or (model == "stablekt" and ("wha" in emb or "sin" in emb or emb == "qid")) or (model == "dtransformer" and "cl" in emb)
    return "旧结果：受 2026-10-03 修复影响，需复核/重跑" if affected else "未记录修复版本；不据此判断结果无效"


def build(root):
    saved, experiment = root / "saved_model", root / "experiment"
    errors, documents, runs, campaigns = [], [], [], []
    for directory in (experiment, root / "docs", root / "research"):
        for path in sorted(directory.rglob("*.md")):
            if path.name == "README.md" or "index" in path.relative_to(directory).parts:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if directory.name in {"docs", "research"} and not path.name.startswith(("results_", "evidence_", "item_state_", "simplekt_delta", "model_lineage")):
                continue
            heading = next((line.lstrip("# ") for line in text.splitlines() if line.startswith("#")), path.stem)
            mentions = sorted(set(re.findall(r"saved_model[/\\]([\w.-]+)", text)))
            stem = path.stem.removeprefix("results_").removesuffix("_plan")
            documents.append({"path": path.relative_to(root).as_posix(), "title": heading, "category": category(stem), "campaigns": mentions})
    if saved.exists():
        for folder in sorted(saved.iterdir()):
            if not folder.is_dir() or folder.name == "_catalog":
                continue
            paths = [p for p in folder.rglob("*") if p.is_file()]
            configs = [p for p in paths if p.name == "run_config.json"]
            members = []
            for path in sorted(configs):
                cfg = load_json(path, errors)
                parent = path.parent
                standard = bool(cfg.get("model_name") and cfg.get("dataset_name"))
                metrics = parent / "best_metrics.json"
                data = load_json(metrics, errors) if metrics.exists() else {}
                checkpoints = [p for p in parent.glob("*.pt") if p.name != "last_epoch_model.pt"]
                present = metrics.exists() and bool(checkpoints) and (parent / "metrics.jsonl").exists()
                status = "标准训练产物齐全" if standard and present else ("标准训练产物不齐（不代表失败）" if standard else "研究脚本配置（采用独立产物格式）")
                row = {
                    "campaign": folder.name, "category": category(folder.name),
                    "path": parent.relative_to(root).as_posix(),
                    "dataset": cfg.get("dataset_name"), "model": cfg.get("model_name"),
                    "emb_type": cfg.get("emb_type"), "fold": cfg.get("fold", cfg.get("outer_fold")),
                    "seed": cfg.get("seed"), "timestamp": cfg.get("timestamp"),
                    "status": status, "valid_auc": data.get("valid_auc"),
                    "best_epoch": data.get("epoch"), "revision_note": revision_note(cfg),
                    "protocol": cfg.get("protocol", {}),
                }
                runs.append(row)
                members.append(row)
            campaigns.append({
                "name": folder.name, "category": category(folder.name),
                "path": folder.relative_to(root).as_posix(), "run_count": len(members),
                "file_count": len(paths), "bytes": sum(p.stat().st_size for p in paths),
                "datasets": sorted({r["dataset"] for r in members if r["dataset"]}),
                "models": sorted({r["model"] for r in members if r["model"]}),
                "statuses": dict(Counter(r["status"] for r in members)),
                "review_count": sum(r["revision_note"].startswith("旧结果") for r in members),
                "reports": [d["path"] for d in documents if folder.name in d["campaigns"]],
                "cv_count": sum(p.name == "cv_summary.json" for p in paths),
                "other_json": sorted({p.name for p in paths if p.suffix == ".json" and p.name not in {"run_config.json", "best_metrics.json"}})[:12],
            })
    output = saved / "_catalog"
    output.mkdir(parents=True, exist_ok=True)
    index = experiment / "index"
    index.mkdir(parents=True, exist_ok=True)
    catalog = {"generated_at": datetime.now().isoformat(timespec="seconds"), "campaigns": campaigns, "runs": runs, "documents": documents, "errors": errors}
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [k for k in runs[0] if k != "protocol"] if runs else []
    with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    for group, title in GROUPS.items():
        group_campaigns = [c for c in campaigns if c["category"] == group]
        group_docs = [d for d in documents if d["category"] == group]
        lines = [f"# {title}", "", "[返回实验总目录](../README.md)", "", "分类用于导航，依据现有名称和文档中的显式路径；不代表质量排名或最终论文结果。", "", "## 实验产物", ""]
        for campaign in group_campaigns:
            lines += [f"### {campaign['name']}", "", f"目录：{link(root / campaign['path'], index)}；大小 {campaign['bytes'] / 1024**3:.2f} GiB；{campaign['run_count']} 个配置；{campaign['cv_count']} 个 CV 汇总。", ""]
            if campaign["reports"]:
                lines += ["对应报告：" + "、".join(link(root / p, index) for p in campaign["reports"]), ""]
            if campaign["run_count"] == 0:
                lines += ["没有标准 run_config；现有 JSON：" + ("、".join(campaign["other_json"]) or "无") + "。需结合脚本确认用途，不自动判为无用。", ""]
            else:
                lines += ["| 数据集 | 模型 / emb_type | fold / seed | 产物 | valid AUC / epoch | 版本 | 配置 |", "|---|---|---|---|---|---|---|"]
                for row in (r for r in runs if r["campaign"] == campaign["name"]):
                    lines.append("| " + " | ".join(map(cell, [row["dataset"], f"{row['model']} / {row['emb_type']}", f"{row['fold']} / {row['seed']}", row["status"], f"{row['valid_auc']} / {row['best_epoch']}", row["revision_note"], link(root / row["path"] / "run_config.json", index, "配置")])) + " |")
                lines.append("")
        lines += ["## 报告与计划", "", "| 文件 | 原标题 | 显式关联的产物目录 |", "|---|---|---|"]
        for doc in group_docs:
            associations = []
            for name in doc["campaigns"]:
                p = saved / name
                associations.append(link(p, index, name) if p.exists() else name + "（路径缺失）")
            lines.append(f"| {link(root / doc['path'], index)} | {cell(doc['title'])} | {'、'.join(associations) or '未写明路径'} |")
        (index / f"{group}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    overview = ["# 实验总目录", "", "从这里查实验用途和产物，不需要逐个猜目录名。原始报告、配置、checkpoint 保持原路径，现有脚本和续跑入口仍可使用。", "", f"本次盘点：**{len(campaigns)} 组产物、{len(runs)} 个运行配置、{len(documents)} 份报告/计划**；共 {sum(c['bytes'] for c in campaigns)/1024**3:.2f} GiB（文件逻辑大小）。", "", "## 按研究主题查找", "", "| 主题 | 产物组数 | 文档数 |", "|---|---:|---:|"]
    for group, title in GROUPS.items():
        overview.append(f"| [{title}](index/{group}.md) | {sum(c['category']==group for c in campaigns)} | {sum(d['category']==group for d in documents)} |")
    overview += ["", "## 如何判断可以使用的结果", "", "- 表格中的“产物齐全”只表示找到配置、best metrics、metrics JSONL 和 checkpoint，不证明实验正确或已经收敛。", "- 看 run_config 中的 dataset、fold、seed、emb_type 和 protocol；协议不同的结果分开比较。这里不自动平均或选最好的 test 数字。", "- FAKT、ATDKT、KeenKT、StableKT、DTransformer 的部分旧分支受 2026-10-03 修复影响，相关运行已标注复核；缺少版本字段本身不证明其他模型有问题。", "- 调试/单轮验证用于检查程序，不能作为正式模型对比；缺产物也可能是中断、仍在运行或独立研究格式。", "", "## 文件职责", "", "- experiment：逐组结果表和临时诊断报告；标题重复的旧表保留，用本目录辨认。", "- research/results_*：研究结论、解释和汇总；research/*_plan：计划。它们也纳入上面的主题索引。", "- saved_model：真实配置、指标、checkpoint、研究分析输出。", "- research：分析脚本、研究计划与结果文档；docs：框架使用与架构文档；scripts/_run_*：数据集实验启动脚本。", "", "## 完整清单与更新", "", "[全部运行 CSV](../saved_model/_catalog/runs.csv) · [机器可读清单 JSON](../saved_model/_catalog/catalog.json) · [产物总览](../saved_model/README.md)", "", "新增或续跑实验后，在仓库执行：", "", "```bash", "python scripts/catalog_experiments.py", "```", "", "新实验建议用 saved_model/<研究主题>/<批次>/<run_name>，报告写明对应目录、假设、协议和结论；先保留旧路径，避免破坏历史配置与脚本引用。", ""]
    if errors:
        overview += [f"有 {len(errors)} 个 JSON 无法解析，具体错误见 catalog.json 的 errors；这些条目需人工检查。", ""]
    (experiment / "README.md").write_text("\n".join(overview), encoding="utf-8")
    lines = ["# 实验产物目录", "", "[实验总入口](../experiment/README.md) · [_catalog/runs.csv](_catalog/runs.csv)", "", "配置数量与文件大小来自磁盘；不加载 checkpoint，不删除或搬移实验。每个目录的细节、旧版本标记和报告链接见主题页。", "", "| 主题 | 目录 | 数据集 | 模型 | 配置数 | GiB | 需复核旧运行 |", "|---|---|---|---|---:|---:|---:|"]
    for c in sorted(campaigns, key=lambda c: (list(GROUPS).index(c["category"]), c["name"])):
        lines.append("| " + " | ".join(map(cell, [link(index / f"{c['category']}.md", saved, GROUPS[c["category"]]), link(root / c["path"], saved), ", ".join(c["datasets"]) or "独立格式", ", ".join(c["models"]) or "独立格式", c["run_count"], f"{c['bytes']/1024**3:.2f}", c["review_count"]])) + " |")
    root_files = [p.name for p in saved.iterdir() if p.is_file() and p.name != "README.md"]
    lines += ["", "根目录其他文件（历史辅助记录，未自动归类）：" + "、".join(root_files), ""]
    (saved / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"campaigns": len(campaigns), "runs": len(runs), "documents": len(documents), "errors": len(errors), "review_runs": sum(c["review_count"] for c in campaigns)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().root.resolve())
