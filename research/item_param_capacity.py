"""How much of SimpleKT's per-item difficulty table is actually used?

SimpleKT's default `emb_type="qid"` gives every question its own `embed_l`-wide
row in `difficult_param` -- 17,738 x 256 = 4.54M parameters on assist2009, against
229,459 training interactions, i.e. ~12.9 observations per item.  Nominally that is
0.05 observations per parameter and should overfit badly.  It does not.

Two candidate reasons, both measurable from a trained checkpoint:

1. `SimpleKT.reset()` zero-initialises every parameter whose first dimension is
   `num_pid + 1`, so `difficult_param` starts at exactly 0.  A question that never
   appears in the training folds receives no gradient and stays at 0, which makes
   its Rasch term vanish and the model fall back to the pure concept embedding.
   So the count of items with a nonzero row, not `num_pid`, is the real width.

2. The gradient w.r.t. one row is `q_embed_diff_data * upstream`, and
   `q_embed_diff` is indexed by concept (124 rows), not by item.  Every item's
   update is therefore steered through a 123-concept subspace, so the rows may
   span far fewer than 256 effective directions.  If the spectrum collapses to
   ~1 direction, the d-dimensional table is behaving like AKT's Rasch scalar and
   the two parameterisations are the same model wearing different parameter counts.

Reports both, plus the norm/frequency relationship that (1) predicts.

Usage:
    python research/item_param_capacity.py --ckpt-root saved_model/itemparam_full
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def train_item_counts(dpath: str, fold: int) -> np.ndarray:
    """Per-question appearance counts in this fold's *training* folds.

    Validation is `fold == fold`; training is every other fold in
    `train_valid_sequences_quelevel.csv`.  Padding is `-1`.  This mirrors
    `datasets/init_dataset.py`'s split, so the counts are the ones the model
    actually saw.
    """
    df = pd.read_csv(os.path.join(dpath, "train_valid_sequences_quelevel.csv"))
    train = df[df["fold"] != fold]
    counts: dict[int, int] = {}
    for qs, sm in zip(train["questions"].astype(str), train["selectmasks"].astype(str)):
        q_ids = qs.split(",")
        masks = sm.split(",")
        for q, m in zip(q_ids, masks):
            q = q.strip()
            if q in ("", "-1"):
                continue
            if m.strip() == "-1":
                continue
            qi = int(q)
            counts[qi] = counts.get(qi, 0) + 1
    size = (max(counts) + 1) if counts else 1
    out = np.zeros(size, dtype=np.int64)
    for qi, n in counts.items():
        out[qi] = n
    return out


def load_difficult_param(ckpt_root: str, fold: int) -> tuple[np.ndarray, str]:
    pattern = os.path.join(ckpt_root, f"*fold{fold}*", "*", "simplekt_*_model.pt")
    paths = [p for p in glob.glob(pattern) if "last_epoch" not in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no checkpoint under {pattern}")
    path = sorted(paths)[-1]
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "difficult_param.weight" not in state:
        raise SystemExit(f"{path} has no difficult_param.weight")
    return state["difficult_param.weight"].float().numpy(), path


def effective_rank(sv: np.ndarray) -> float:
    """exp(entropy of the normalised spectrum) -- the participation ratio.

    1.0 means every row lies on a single direction (a scalar difficulty times a
    fixed vector, i.e. Rasch); `d` means the directions are used evenly.
    """
    p = sv / sv.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-root", default="saved_model/itemparam_full")
    ap.add_argument("--dpath", default="data/assist2009")
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    for fold in args.folds:
        W, path = load_difficult_param(args.ckpt_root, fold)
        counts = train_item_counts(args.dpath, fold)
        n_items, d = W.shape
        if len(counts) < n_items:
            counts = np.pad(counts, (0, n_items - len(counts)))
        counts = counts[:n_items]

        norms = np.linalg.norm(W, axis=1)
        seen = counts > 0
        nonzero = norms > 0

        print(f"\n=== fold {fold} ===")
        print(f"checkpoint: {path}")
        print(f"difficult_param: {n_items} x {d} = {W.size:,} nominal parameters")
        print(f"items seen in training folds : {seen.sum():,} / {n_items:,} "
              f"({100 * seen.mean():.1f}%)")
        print(f"items with a nonzero row     : {nonzero.sum():,} "
              f"({100 * nonzero.mean():.1f}%)")
        print(f"unseen items left at exactly 0: {(~seen & ~nonzero).sum():,} "
              f"of {(~seen).sum():,} unseen")
        print(f"effective nominal width      : {nonzero.sum() * d:,} parameters "
              f"({100 * nonzero.sum() * d / W.size:.1f}% of nominal)")

        # (1) does row magnitude track how often the item was drilled?
        print("\n  row norm by training frequency")
        print(f"  {'freq bucket':>14} | {'items':>7} | {'mean ||w||':>10} | {'median':>8}")
        edges = [0, 1, 2, 4, 8, 16, 32, 64, 10 ** 9]
        labels = ["0 (unseen)", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"]
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            sel = (counts >= lo) & (counts < hi) if lo > 0 else (counts == 0)
            if sel.sum() == 0:
                continue
            print(f"  {lab:>14} | {sel.sum():7,} | {norms[sel].mean():10.4f} "
                  f"| {np.median(norms[sel]):8.4f}")
        if seen.sum() > 2:
            r = np.corrcoef(np.log1p(counts[seen]), norms[seen])[0, 1]
            print(f"  corr(log1p(freq), ||w||) over seen items = {r:+.3f}")

        # (2) how many directions do the rows actually span?
        Wn = W[nonzero]
        if Wn.shape[0] > 1:
            sv = np.linalg.svd(Wn, compute_uv=False)
            energy = sv ** 2
            frac = energy / energy.sum()
            print(f"\n  spectrum of the {Wn.shape[0]:,} nonzero rows "
                  f"(full width {d})")
            print(f"  top-1 direction explains {100 * frac[0]:5.1f}% of the energy")
            for k in (2, 3, 5, 10, 25):
                if k <= len(frac):
                    print(f"  top-{k:<2d}                   {100 * frac[:k].sum():5.1f}%")
            print(f"  effective rank (participation ratio) = {effective_rank(sv):.2f}"
                  f"   [1.00 would be exactly Rasch]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
