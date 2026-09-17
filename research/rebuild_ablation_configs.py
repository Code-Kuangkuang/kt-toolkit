"""Rebuild configs/ablation/*.json from the current configs/kt_config.json.

Each ablation config is a full copy of the main config with one setting changed.
Copies drift: the four on disk still carried blocks for models that no longer
exist, plus years of unrelated differences inherited from whatever the main
config looked like when they were made. Regenerating from the current main
config keeps the intended single-variable property true.

Re-run this whenever configs/kt_config.json changes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs" / "kt_config.json"
OUT = ROOT / "configs" / "ablation"

# name -> (purpose, {model_block: {key: value}})
ABLATIONS = {
    "atkt_beta0": (
        "对抗训练的权重置零；与默认 beta=0.2 的差值即对抗训练的贡献",
        {"atkt": {"beta": 0.0}},
    ),
    "dkvmn_first": (
        "多知识点题目只取第一个知识点（旧的截断行为）",
        {"dkvmn": {"concept_mode": "first"}},
    ),
    "dkvmn_multi": (
        "多知识点题目做掩码平均池化；与 first 的差值即截断代价",
        {"dkvmn": {"concept_mode": "multi"}},
    ),
    "akt_long": (
        "AKT 延长训练预算至 600 轮",
        {"akt": {"num_epochs": 600}},
    ),
}


def main() -> None:
    main_config = json.loads(MAIN.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    for name, (purpose, overrides) in ABLATIONS.items():
        config = json.loads(json.dumps(main_config))  # deep copy
        applied = {}
        for block, settings in overrides.items():
            if block not in config:
                raise SystemExit(
                    f"{name}: 主配置里没有 '{block}' 块，无法生成。"
                    "该模型可能已被删除，请更新 ABLATIONS。"
                )
            for key, value in settings.items():
                applied[f"{block}.{key}"] = (config[block].get(key), value)
                config[block][key] = value

        path = OUT / f"{name}.json"
        path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Verify the single-variable property actually holds.
        written = json.loads(path.read_text(encoding="utf-8"))
        drift = []
        for block in set(main_config) | set(written):
            if main_config.get(block) == written.get(block):
                continue
            if not isinstance(main_config.get(block), dict):
                drift.append(block)
                continue
            for key in set(main_config[block]) | set(written[block]):
                if main_config[block].get(key) != written[block].get(key):
                    if f"{block}.{key}" not in applied:
                        drift.append(f"{block}.{key}")
        if drift:
            raise SystemExit(f"{name}: 出现预期之外的差异 {drift}")

        changes = "，".join(
            f"{k}: {old!r} -> {new!r}" for k, (old, new) in applied.items()
        )
        print(f"{name:<16} {changes}")
        print(f"{'':16} {purpose}")

    print(f"\n重建 {len(ABLATIONS)} 个文件，每个与主配置的差异均已逐键校验。")


if __name__ == "__main__":
    main()
