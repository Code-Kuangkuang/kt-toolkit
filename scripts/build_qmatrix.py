"""Build `qmatrix.npz` from a dataset's own question-level sequence files.

Several models read `data/<dataset>/qmatrix.npz` -- DenoiseKT's and HCGKT's
question graphs, CGMKT, and the `_cold` difficulty arms -- but nothing in this
repository writes it. The two that exist (assist2009, assist2017) came from a
pipeline that is no longer here, and the error messages that point at
`scripts/run_clean.py` are wrong: that script does not produce it either.

It does not need a pipeline. The file is a float64 `[num_q + 1, num_c + 1]`
matrix, `matrix[q, c] = 1` when question `q` carries concept `c`, with an
all-zero padding row and column (assist2009: 17738 x 124 for num_q=17737,
num_c=123) -- the layout `models/lpkt.py::generate_qmatrix` writes. Every
question-concept pair a model can ever see is already in the `*_quelevel.csv`
sequence files, so the matrix is read off those. Nothing is regenerated, so
results already computed on the dataset stay comparable.

It refuses to overwrite an existing file unless `--force` is given, and
`--check` builds into memory and compares against the file already on disk
instead of writing -- which is how this was validated against assist2009.

Usage:
    python scripts/build_qmatrix.py --dataset-name algebra2005
    python scripts/build_qmatrix.py --dataset-name assist2009 --check
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.feature_utils import parse_concept_lists, parse_int_list  # noqa: E402

# The windowed test file is deliberately absent. It re-slices the same test
# sequences into ~20x as many rows, so it adds no question-concept pair the
# plain test file lacks -- checked on assist2009 and assist2017, where dropping
# it leaves the matrix bit-identical -- and on algebra2005 it was the whole cost
# of a build: several minutes with it, one second without.
SOURCES = (
    "train_valid_sequences_quelevel.csv",
    "test_sequences_quelevel.csv",
)


def build(dpath, num_q, num_c, sources=None):
    # `(num_q + 1, num_c + 1)` float64, zero padding row and column: exactly the
    # layout `models/lpkt.py::generate_qmatrix` writes, and what the existing
    # assist2009 file is. LPKT loads this file instead of building its own when
    # it exists, so a different shape here would change LPKT, not just add a file.
    matrix = np.zeros((num_q + 1, num_c + 1), dtype=np.float64)
    used = []
    for name in dict.fromkeys(SOURCES if sources is None else sources):
        if not name:
            continue
        path = os.path.join(dpath, name)
        if not os.path.exists(path):
            continue
        used.append(name)
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         usecols=["questions", "concepts"])
        for questions, concepts in zip(df["questions"], df["concepts"]):
            for question, cids in zip(parse_int_list(questions),
                                      parse_concept_lists(concepts)):
                if not 0 <= question < num_q:
                    continue
                for concept in cids:
                    if 0 <= concept < num_c:
                        matrix[question, concept] = 1
    if not used:
        raise FileNotFoundError(f"No *_quelevel.csv sequence file in {dpath}.")
    return matrix, used


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--check", action="store_true",
                        help="compare against the existing qmatrix.npz, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing qmatrix.npz")
    args = parser.parse_args()

    cfg = json.load(open(os.path.join(ROOT, "configs", "data_config.json"),
                         encoding="utf-8"))[args.dataset_name]
    dpath = cfg["dpath"] if os.path.isabs(cfg["dpath"]) else os.path.join(ROOT, cfg["dpath"])
    num_q, num_c = int(cfg["num_q"]), int(cfg["num_c"])
    out = os.path.join(dpath, "qmatrix.npz")

    matrix, used = build(dpath, num_q, num_c, sources=[
        cfg.get("train_valid_file_quelevel"), cfg.get("test_file_quelevel"),
    ])
    covered = int(matrix.any(axis=1).sum())
    print(f"{args.dataset_name}: num_q={num_q} num_c={num_c}  from {used}")
    print(f"  questions with >=1 concept: {covered} / {num_q}")
    print(f"  concepts per question (covered): {matrix[matrix.any(1)].sum(1).mean():.2f}")

    if args.check:
        if not os.path.exists(out):
            raise SystemExit(f"--check: nothing to compare against at {out}")
        existing = np.load(out)["matrix"]
        print(f"  existing shape {existing.shape} {existing.dtype}, "
              f"built shape {matrix.shape} {matrix.dtype}")
        if existing.shape == matrix.shape:
            print(f"  byte-for-byte equal: {bool(np.array_equal(existing, matrix))}")
        rows = min(existing.shape[0], matrix.shape[0])
        cols = min(existing.shape[1], matrix.shape[1])
        a = existing[:rows, :cols] > 0
        b = matrix[:rows, :cols] > 0
        mine = b.any(1)
        print(f"  identical on every row this build covers: "
              f"{bool((a[mine] == b[mine]).all())}  ({int(mine.sum())} rows)")
        print(f"  identical overall: {bool((a == b).all())}  "
              f"(rows only the existing file fills: {int((a.any(1) & ~mine).sum())})")
        return

    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite, or --check to compare.")
    np.savez_compressed(out, matrix=matrix)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
