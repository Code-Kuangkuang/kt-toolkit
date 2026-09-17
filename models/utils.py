import copy
import ast
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, ReLU, Sequential


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class transformer_FFN(nn.Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len, target_device=None):
    """Upper triangular mask.

    `target_device` follows the convention already used in models/saint.py.
    Falling back to the module-level `device` keeps old callers working, but it
    is the wrong default: that global is fixed at import time from
    `torch.cuda.is_available()`, so on a machine with a GPU the mask lands on
    CUDA even when the model was explicitly built on CPU, and the forward dies
    on a device mismatch. Pass the device of a tensor you already have.
    """
    target_device = target_device or device
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(target_device)


def pos_encode(seq_len, target_device=None):
    """Position encoding indices. See `ut_mask` on `target_device`."""
    target_device = target_device or device
    return torch.arange(seq_len).unsqueeze(0).to(target_device)


def get_clones(module, n):
    """Cloning nn modules."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def resolve_pretrain_path(path, dpath=""):
    """Resolve a possibly relative pretrained file path."""
    if not path:
        return ""
    if os.path.isabs(path):
        return os.path.normpath(path)

    candidates = []
    if dpath:
        candidates.append(os.path.join(dpath, path))
    candidates.append(path)

    for cand in candidates:
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return os.path.normpath(candidates[0])


def _load_pretrained_array(npz_path):
    if not npz_path or not os.path.exists(npz_path):
        return None

    with np.load(npz_path) as data:
        for key in ("pro_final_repre", "pro_repre"):
            if key in data.files:
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Invalid embedding array shape for key '{key}': {arr.shape}")
                return arr

        if not data.files:
            raise ValueError(f"No arrays found in pretrained embedding file: {npz_path}")

        first_key = data.files[0]
        arr = np.asarray(data[first_key], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"No 2D embedding array found in {npz_path}; first key '{first_key}' has shape {arr.shape}"
            )
        return arr


def _fit_num_q(arr, num_q):
    cur_q, dim = arr.shape
    if cur_q == num_q:
        return arr

    if cur_q == num_q + 1 and np.allclose(arr[0], 0.0, atol=1e-8):
        return arr[1:]

    if cur_q > num_q:
        return arr[:num_q]

    pad = np.random.normal(loc=0.0, scale=0.02, size=(num_q - cur_q, dim)).astype(np.float32)
    return np.concatenate([arr, pad], axis=0)


def _load_question_idx_map(path):
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], dict):
        data = data["questions"]

    if not isinstance(data, dict):
        return {}

    out = {}
    for k, v in data.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _load_pretrained_row_map(path):
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return {}

    data = ast.literal_eval(text)
    if not isinstance(data, dict):
        return {}

    out = {}
    for k, v in data.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _find_first_existing(candidates):
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.normpath(cand)
    return ""


def load_pretrained_question_matrix(
    num_q,
    emb_path="",
    dpath="",
    keyid2idx_path="",
    pro_id_dict_path="",
):
    """Load and align question embeddings from a pretrained npz file.

    Returns:
        tuple(np.ndarray | None, dict): (embedding_matrix, meta_info)
    """
    meta = {
        "status": "missing",
        "emb_path": "",
        "keyid2idx_path": "",
        "pro_id_dict_path": "",
        "aligned_count": 0,
    }

    resolved_emb_path = resolve_pretrain_path(emb_path, dpath)
    meta["emb_path"] = resolved_emb_path
    arr = _load_pretrained_array(resolved_emb_path)
    if arr is None:
        return None, meta

    emb_dir = os.path.dirname(resolved_emb_path) if resolved_emb_path else ""
    resolved_keyid2idx = resolve_pretrain_path(keyid2idx_path, dpath) if keyid2idx_path else ""
    resolved_pro_map = resolve_pretrain_path(pro_id_dict_path, dpath) if pro_id_dict_path else ""

    if not resolved_keyid2idx:
        resolved_keyid2idx = _find_first_existing(
            [
                os.path.join(dpath, "keyid2idx.json") if dpath else "",
                os.path.join(emb_dir, "keyid2idx.json") if emb_dir else "",
            ]
        )
    if not resolved_pro_map:
        resolved_pro_map = _find_first_existing(
            [
                os.path.join(emb_dir, "pro_id_dict.txt") if emb_dir else "",
                os.path.join(dpath, "pro_id_dict.txt") if dpath else "",
            ]
        )

    meta["keyid2idx_path"] = resolved_keyid2idx
    meta["pro_id_dict_path"] = resolved_pro_map

    qid2idx = _load_question_idx_map(resolved_keyid2idx)
    qid2row = _load_pretrained_row_map(resolved_pro_map)

    if qid2idx and qid2row:
        aligned = np.random.normal(loc=0.0, scale=0.02, size=(num_q, arr.shape[1])).astype(np.float32)
        matched = 0
        for raw_qid, row_idx in qid2row.items():
            q_idx = qid2idx.get(raw_qid)
            if q_idx is None:
                continue
            if 0 <= q_idx < num_q and 0 <= row_idx < arr.shape[0]:
                aligned[q_idx] = arr[row_idx]
                matched += 1

        meta["status"] = "aligned"
        meta["aligned_count"] = matched
        return aligned, meta

    meta["status"] = "fitted"
    return _fit_num_q(arr, num_q), meta


def build_question_embedding(
    num_q,
    emb_size,
    emb_path="",
    dpath="",
    keyid2idx_path="",
    pro_id_dict_path="",
    freeze_pretrained=False,
):
    """Build question embedding + optional projection from pretrained npz."""
    matrix, meta = load_pretrained_question_matrix(
        num_q=num_q,
        emb_path=emb_path,
        dpath=dpath,
        keyid2idx_path=keyid2idx_path,
        pro_id_dict_path=pro_id_dict_path,
    )

    if matrix is None:
        emb = nn.Embedding(num_q, emb_size)
        proj = nn.Identity()
        return emb, proj, meta

    raw_dim = int(matrix.shape[1])
    emb = nn.Embedding(num_q, raw_dim)
    with torch.no_grad():
        emb.weight.copy_(torch.from_numpy(matrix))
    emb.weight.requires_grad = not bool(freeze_pretrained)

    proj = nn.Linear(raw_dim, emb_size) if raw_dim != emb_size else nn.Identity()
    return emb, proj, meta
