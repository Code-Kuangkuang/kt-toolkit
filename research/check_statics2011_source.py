"""Check a downloaded STATICS2011 file against what the preprocessor expects.

`preprocess/statics2011_preprocess.py` was written against one particular
DataShop export and hardcodes three things that vary between exports:

    pd.read_csv(read_file)                      comma separated
    df["First Attempt"]                         values correct / incorrect / hint
    datetime.strptime(t, "%Y/%m/%d %H:%M")      on First Transaction Time

algebra2005, which this repository already has, is a *different* export: tab
separated, `Correct First Attempt` holding 0/1, and timestamps like
`2005-09-09 12:24:49.0`. So the one cannot be used to predict the other, and
guessing costs a confusing failure part way through preprocessing.

Run this on the extracted file before wiring the dataset up:

    python research/check_statics2011_source.py <path to the extracted file>

It reports what the file actually is and, for each of the three assumptions,
whether it holds. Nothing is modified.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REQUIRED = [
    "Anon Student Id",
    "Problem Name",
    "Step Name",
    "First Transaction Time",
    "First Attempt",
]
TIME_FORMAT = "%Y/%m/%d %H:%M"


def sniff_separator(path):
    """Tab or comma, decided by which yields more columns on the header."""
    head = path.open("r", encoding="utf-8", errors="replace").readline()
    return "\t" if head.count("\t") > head.count(",") else ","


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="解压出来的 STATICS2011 数据文件")
    parser.add_argument("--rows", type=int, default=20000, help="抽查行数")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"文件不存在: {args.path}")

    sep = sniff_separator(args.path)
    print(f"文件: {args.path}")
    print(f"大小: {args.path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"分隔符: {'制表符' if sep == chr(9) else '逗号'}"
          f"{'' if sep == ',' else '   <- 预处理用的是 read_csv（逗号），需要改成 read_table'}")

    frame = pd.read_csv(args.path, sep=sep, nrows=args.rows, dtype=str,
                        keep_default_na=False, low_memory=False)
    print(f"列数: {len(frame.columns)}   抽查行数: {len(frame)}")

    print("\n预处理需要的列:")
    missing = []
    for column in REQUIRED:
        present = column in frame.columns
        print(f"  {column:26} {'有' if present else '缺'}")
        if not present:
            missing.append(column)

    if missing:
        print("\n缺列。DataShop 的另一种导出把答对列叫 `Correct First Attempt`（0/1），"
              "\n而这里期望 `First Attempt`（correct/incorrect/hint）。候选列名:")
        for column in frame.columns:
            low = column.lower()
            if "attempt" in low or "correct" in low or "time" in low:
                print(f"    {column}")

    if "First Attempt" in frame.columns:
        values = frame["First Attempt"].value_counts()
        print("\nFirst Attempt 的取值:")
        for value, count in values.head(8).items():
            print(f"  {value!r:16} {count}")
        unexpected = set(values.index) - {"correct", "incorrect", "hint", ""}
        if unexpected:
            print(f"  预处理只认 correct/incorrect/hint，额外取值: {sorted(unexpected)}")
        # The preprocessor drops `hint` rows outright, which is the CFA
        # convention: needing a hint means the first attempt was not correct.
        hints = int(values.get("hint", 0))
        if hints:
            print(f"  其中 hint {hints} 行会被整行丢弃（占抽查 {100*hints/len(frame):.1f}%）")

    if "First Transaction Time" in frame.columns:
        sample = next((v for v in frame["First Transaction Time"] if v), None)
        print(f"\nFirst Transaction Time 样例: {sample!r}")
        try:
            datetime.strptime(str(sample), TIME_FORMAT)
            print(f"  与预处理的 {TIME_FORMAT!r} 匹配")
        except (ValueError, TypeError):
            print(f"  与预处理的 {TIME_FORMAT!r} 不匹配 -> change2timestamp 会抛异常")
            for candidate in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                              "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S",
                              "%m/%d/%Y %H:%M"):
                try:
                    datetime.strptime(str(sample), candidate)
                    print(f"  实际格式应为 {candidate!r}")
                    break
                except ValueError:
                    continue
            else:
                print("  未能识别，需要手工确定格式")

    # The repository's own expectation, for reference.
    print("\nconfigs/data_config.json 声明 statics2011 为 num_c=1223, num_q=0，"
          "\n即知识点由 `Problem Name----Step Name` 拼成、没有独立题目 id。")
    if all(c in frame.columns for c in ("Problem Name", "Step Name")):
        keys = {f"{p}----{s}" for p, s in
                zip(frame["Problem Name"], frame["Step Name"])}
        print(f"抽查 {len(frame)} 行得到 {len(keys)} 个不同的 KC"
              f"（全量应接近 1223，抽查值偏小是正常的）")


if __name__ == "__main__":
    sys.exit(main())
