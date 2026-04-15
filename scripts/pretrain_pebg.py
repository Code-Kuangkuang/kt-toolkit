import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_cfg


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def _split_csv_list(text: str):
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",")]


def _parse_int_list(text: str):
    out = []
    for tok in _split_csv_list(text):
        try:
            out.append(int(tok))
        except Exception:
            try:
                out.append(int(float(tok)))
            except Exception:
                out.append(-1)
    return out


def _parse_float_list(text: str):
    out = []
    for tok in _split_csv_list(text):
        try:
            out.append(float(tok))
        except Exception:
            out.append(-1.0)
    return out


def _parse_concept_token(token: str):
    t = str(token).strip()
    if not t or t == "-1":
        return []
    parts = re.split(r"[_;|]", t)
    skills = []
    for p in parts:
        p = p.strip()
        if not p or p == "-1":
            continue
        try:
            skills.append(int(p))
        except Exception:
            continue
    if not skills:
        return []
    return sorted(set(skills))


def _load_question_raw_mapping(dataset_dir: str):
    path = os.path.join(dataset_dir, "keyid2idx.json")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], dict):
        data = data["questions"]

    if not isinstance(data, dict):
        return {}

    idx_to_raw = {}
    for raw_qid, q_idx in data.items():
        try:
            idx_to_raw[int(q_idx)] = int(raw_qid)
        except Exception:
            continue
    return idx_to_raw


def _load_uid_raw_mapping(dataset_dir: str):
    path = os.path.join(dataset_dir, "keyid2idx.json")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    uid_map = data.get("uid", {}) if isinstance(data, dict) else {}
    if not isinstance(uid_map, dict):
        return {}

    idx_to_raw = {}
    for raw_uid, uid_idx in uid_map.items():
        try:
            idx_to_raw[int(uid_idx)] = str(raw_uid)
        except Exception:
            continue
    return idx_to_raw


def _get_allowed_raw_users_from_folds(sequence_path: str, dataset_dir: str, exclude_folds):
    if exclude_folds is None or len(exclude_folds) == 0:
        return None

    df = pd.read_csv(sequence_path, dtype=str, keep_default_na=False)
    required_cols = {"uid", "fold"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Fold-aware original preprocessing requires uid/fold columns in sequence file. "
            f"Missing: {missing}. File: {sequence_path}"
        )

    drop_folds = {int(f) for f in exclude_folds}
    df["fold"] = df["fold"].astype(int)
    kept = df[~df["fold"].isin(drop_folds)]

    kept_uid_idx = set()
    for v in kept["uid"].tolist():
        try:
            kept_uid_idx.add(int(v))
        except Exception:
            continue

    idx_to_raw_uid = _load_uid_raw_mapping(dataset_dir)
    allowed_raw_users = {idx_to_raw_uid[i] for i in kept_uid_idx if i in idx_to_raw_uid}
    if not allowed_raw_users:
        raise ValueError(
            "No train users resolved for fold-aware original preprocessing. "
            "Please verify keyid2idx.json contains uid mapping and sequence fold split is valid."
        )
    return allowed_raw_users


def _extract_original_assist09_assets(raw_csv_path: str, min_inter_num: int = 3, allowed_user_ids=None):
    """Reproduce original PEBG assist09 preprocessing in data_assist09.py.

    The generated assets follow the original pipeline assumptions:
    - filter invalid skills and scaffolding rows
    - remove users with too few interactions
    - build problem-skill bipartite graph from first skill_id string per problem
    - build pro_feat as [ms_first_response, answer_type one-hot, mean correctness]
    - keep joint skills in skill_id_dict
    """
    df = pd.read_csv(raw_csv_path, low_memory=False, encoding="ISO-8859-1")

    if "skill_id" not in df.columns:
        raise ValueError(f"Original assist09 csv must contain skill_id column: {raw_csv_path}")

    df = df.dropna(subset=["skill_id"])
    df = df[~df["skill_id"].isin(["noskill"])]
    if "original" in df.columns:
        df = df[df["original"].isin([1])]

    if allowed_user_ids is not None:
        allowed_user_ids = {str(u) for u in allowed_user_ids}
        df = df[df["user_id"].astype(str).isin(allowed_user_ids)]

    if "user_id" not in df.columns:
        raise ValueError(f"Original assist09 csv must contain user_id column: {raw_csv_path}")

    user_sizes = df.groupby("user_id").size()
    keep_users = user_sizes[user_sizes >= int(min_inter_num)].index
    df = df[df["user_id"].isin(keep_users)]

    required_cols = {"problem_id", "correct", "answer_type", "ms_first_response"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for original assist09 preprocessing: {missing}")

    problems = df["problem_id"].dropna().unique().tolist()
    pro_id_dict = {int(pid): int(i) for i, pid in enumerate(problems)}

    pro_types = df["answer_type"].fillna("NA").astype(str).unique().tolist()
    pro_type_dict = {t: i for i, t in enumerate(pro_types)}

    pro_feat_rows = []
    pro_skill_edges = []
    skill_id_dict = {}
    skill_cnt = 0

    for pid in problems:
        tmp_df = df[df["problem_id"] == pid]
        if len(tmp_df) == 0:
            continue

        first_row = tmp_df.iloc[0]
        ms = float(tmp_df["ms_first_response"].abs().mean())
        p = float(tmp_df["correct"].mean())
        pro_type = str(first_row["answer_type"])
        pro_type_id = int(pro_type_dict[pro_type])

        # Same feature layout as original script: [ms, one-hot(answer_type), mean_correct].
        feat = [0.0] * (len(pro_type_dict) + 2)
        feat[0] = ms
        feat[pro_type_id + 1] = 1.0
        feat[-1] = p
        pro_feat_rows.append(feat)

        raw_skill = str(first_row["skill_id"]).strip()
        for s in raw_skill.split("_"):
            s = s.strip()
            if not s:
                continue
            if s not in skill_id_dict:
                skill_id_dict[s] = int(skill_cnt)
                skill_cnt += 1
            pro_skill_edges.append((int(pro_id_dict[int(pid)]), int(skill_id_dict[s])))

    if not pro_skill_edges:
        raise ValueError("No problem-skill edges extracted from original assist09 csv.")

    pro_feat = np.asarray(pro_feat_rows, dtype=np.float32)
    if pro_feat.shape[0] > 0:
        pro_feat[:, 0] = _minmax(pro_feat[:, 0])

    edge_rows = np.asarray([e[0] for e in pro_skill_edges], dtype=np.int64)
    edge_cols = np.asarray([e[1] for e in pro_skill_edges], dtype=np.int64)
    edge_data = np.ones(edge_rows.shape[0], dtype=np.float32)

    pro_num = int(len(pro_id_dict))
    skill_num = int(np.max(edge_cols) + 1)
    pro_skill = sparse.coo_matrix((edge_data, (edge_rows, edge_cols)), shape=(pro_num, skill_num)).tocsr()

    # Keep joint skills as new skill ids, same as original implementation.
    for s in df["skill_id"].dropna().astype(str).unique().tolist():
        if "_" in s and s not in skill_id_dict:
            skill_id_dict[s] = int(skill_cnt)
            skill_cnt += 1

    pro_pro = (pro_skill @ pro_skill.T).astype(np.float32).tocsr()
    if pro_pro.nnz > 0:
        pro_pro.data[:] = 1.0
    pro_pro = pro_pro.tocoo()

    skill_skill = (pro_skill.T @ pro_skill).astype(np.float32).tocsr()
    if skill_skill.nnz > 0:
        skill_skill.data[:] = 1.0
    skill_skill = skill_skill.tocoo()

    # skill_id_dict keys are strings in original code.
    skill_id_dict = {str(k): int(v) for k, v in skill_id_dict.items()}
    pro_id_dict = {int(k): int(v) for k, v in pro_id_dict.items()}

    return {
        "pro_skill_sparse": pro_skill.tocoo(),
        "pro_pro_sparse": pro_pro,
        "skill_skill_sparse": skill_skill,
        "pro_feat": pro_feat,
        "pro_id_dict": pro_id_dict,
        "skill_id_dict": skill_id_dict,
    }


def _extract_graph_and_features(sequence_path: str, exclude_folds=None):
    df = pd.read_csv(sequence_path, dtype=str, keep_default_na=False)

    if exclude_folds is not None and len(exclude_folds) > 0:
        if "fold" not in df.columns:
            raise ValueError(
                "Fold-aware pretraining requires a 'fold' column in sequence data. "
                f"File: {sequence_path}"
            )
        drop_folds = {int(f) for f in exclude_folds}
        df["fold"] = df["fold"].astype(int)
        df = df[~df["fold"].isin(drop_folds)]

    required_cols = {"questions", "concepts", "responses"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {sequence_path}: {missing}")

    has_sm = "selectmasks" in df.columns
    has_ut = "usetimes" in df.columns

    q_seen = set()
    skill_seen = set()
    q_skill_sets = defaultdict(set)
    q_correct_sum = defaultdict(float)
    q_count = defaultdict(int)
    q_usetime_sum = defaultdict(float)
    q_usetime_count = defaultdict(int)

    for _, row in df.iterrows():
        qs = _parse_int_list(row.get("questions", ""))
        cs = _split_csv_list(row.get("concepts", ""))
        rs = _parse_float_list(row.get("responses", ""))
        sm = _parse_int_list(row.get("selectmasks", "")) if has_sm else []
        ut = _parse_float_list(row.get("usetimes", "")) if has_ut else []

        if not sm:
            sm = [1] * len(qs)

        max_len = min(len(qs), len(cs), len(rs), len(sm))
        if has_ut and len(ut) > 0:
            max_len = min(max_len, len(ut))

        for i in range(max_len):
            if sm[i] == -1:
                continue

            q = qs[i]
            r = rs[i]
            if q < 0 or r < 0:
                continue

            q_seen.add(q)
            q_count[q] += 1
            q_correct_sum[q] += float(r)

            if has_ut and len(ut) > i and ut[i] >= 0:
                q_usetime_sum[q] += float(ut[i])
                q_usetime_count[q] += 1

            skills = _parse_concept_token(cs[i])
            for s in skills:
                skill_seen.add(s)
                q_skill_sets[q].add(s)

    if not q_seen:
        raise ValueError("No valid question interactions found for PEBG pretraining.")
    if not skill_seen:
        raise ValueError("No valid concept interactions found for PEBG pretraining.")

    q_ids = sorted(q_seen)
    skill_ids = sorted(skill_seen)

    qid_to_row = {q: i for i, q in enumerate(q_ids)}
    skill_to_row = {s: i for i, s in enumerate(skill_ids)}

    rows, cols = [], []
    for q in q_ids:
        p = qid_to_row[q]
        for s in sorted(q_skill_sets[q]):
            if s in skill_to_row:
                rows.append(p)
                cols.append(skill_to_row[s])

    if not rows:
        raise ValueError("No question-skill edges extracted; cannot train PEBG.")

    data = np.ones(len(rows), dtype=np.float32)
    pro_skill = sparse.coo_matrix((data, (np.array(rows), np.array(cols))), shape=(len(q_ids), len(skill_ids))).tocsr()

    # Similarity graph by shared neighbors, binarized.
    pro_pro = (pro_skill @ pro_skill.T).astype(np.float32).tocsr()
    if pro_pro.nnz > 0:
        pro_pro.data[:] = 1.0
    pro_pro = pro_pro.tocoo()

    skill_skill = (pro_skill.T @ pro_skill).astype(np.float32).tocsr()
    if skill_skill.nnz > 0:
        skill_skill.data[:] = 1.0
    skill_skill = skill_skill.tocoo()

    count_arr = np.array([q_count[q] for q in q_ids], dtype=np.float32)
    acc_arr = np.array([
        q_correct_sum[q] / float(max(q_count[q], 1))
        for q in q_ids
    ], dtype=np.float32)

    if any(q_usetime_count[q] > 0 for q in q_ids):
        ut_arr = np.array([
            q_usetime_sum[q] / float(max(q_usetime_count[q], 1))
            for q in q_ids
        ], dtype=np.float32)
    else:
        ut_arr = np.zeros_like(count_arr, dtype=np.float32)

    count_norm = _minmax(count_arr)
    ut_norm = _minmax(ut_arr)
    pro_feat = np.stack([count_norm, ut_norm, acc_arr], axis=1).astype(np.float32)

    return {
        "q_ids": q_ids,
        "skill_ids": skill_ids,
        "qid_to_row": qid_to_row,
        "skill_to_row": skill_to_row,
        "pro_skill_sparse": pro_skill.tocoo(),
        "pro_pro_sparse": pro_pro,
        "skill_skill_sparse": skill_skill,
        "pro_feat": pro_feat,
    }


def _save_assets(assets: dict, dataset_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    sparse.save_npz(os.path.join(output_dir, "pro_skill_sparse.npz"), assets["pro_skill_sparse"])
    sparse.save_npz(os.path.join(output_dir, "pro_pro_sparse.npz"), assets["pro_pro_sparse"])
    sparse.save_npz(os.path.join(output_dir, "skill_skill_sparse.npz"), assets["skill_skill_sparse"])

    np.savez(os.path.join(output_dir, "pro_feat.npz"), pro_feat=assets["pro_feat"])

    if "skill_id_dict" in assets:
        skill_id_dict = assets["skill_id_dict"]
    else:
        # sequence mode: base skills only
        skill_id_dict = {int(s): int(assets["skill_to_row"][s]) for s in assets["skill_ids"]}
    with open(os.path.join(output_dir, "skill_id_dict.txt"), "w", encoding="utf-8") as f:
        f.write(str(skill_id_dict))

    if "pro_id_dict" in assets:
        pro_id_dict = assets["pro_id_dict"]
    else:
        idx_to_raw = _load_question_raw_mapping(dataset_dir)
        pro_id_dict = {}
        for q_idx in assets["q_ids"]:
            raw_qid = idx_to_raw.get(int(q_idx), int(q_idx))
            pro_id_dict[int(raw_qid)] = int(assets["qid_to_row"][q_idx])
    with open(os.path.join(output_dir, "pro_id_dict.txt"), "w", encoding="utf-8") as f:
        f.write(str(pro_id_dict))

    # Save debugging metadata for traceability.
    meta = {
        "num_questions": int(assets["pro_skill_sparse"].shape[0]),
        "num_skills": int(assets["pro_skill_sparse"].shape[1]),
        "num_pro_skill_edges": int(assets["pro_skill_sparse"].nnz),
        "num_pro_pro_edges": int(assets["pro_pro_sparse"].nnz),
        "num_skill_skill_edges": int(assets["skill_skill_sparse"].nnz),
    }
    with open(os.path.join(output_dir, "pebg_assets_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)


def _run_pebg_training(args, output_dir: str):
    pebg_script = os.path.join(str(ROOT), "modules", "pebg_torch.py")
    if not os.path.exists(pebg_script):
        raise FileNotFoundError(f"PEBG training script not found: {pebg_script}")

    output_name = args.output_name.strip() if args.output_name else f"embedding_{args.epochs}_pt.npz"
    cmd = [
        sys.executable,
        pebg_script,
        "--data_dir", output_dir,
        "--con_sym", args.con_sym,
        "--embed_dim", str(args.embed_dim),
        "--hidden_dim", str(args.hidden_dim),
        "--keep_prob", str(args.keep_prob),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output_name", output_name,
    ]

    if args.save_every:
        cmd.extend(["--save_every", args.save_every])

    print("[pretrain_pebg] Launch training:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    return os.path.join(output_dir, output_name)


def main():
    parser = argparse.ArgumentParser(description="KT-Toolkit native PEBG pretraining pipeline")
    parser.add_argument("--dataset_name", required=True, help="Dataset name in data_config.json")
    parser.add_argument("--data_config", default="configs/data_config.json", help="Path to data_config.json")
    parser.add_argument(
        "--preprocess_mode",
        choices=["original_assist09", "original_assist09_fold", "sequence"],
        default="original_assist09",
        help="Preprocess mode. Use original_assist09 to reproduce original PEBG assist09 preprocessing.",
    )
    parser.add_argument(
        "--raw_file",
        default="",
        help="Raw assist09 csv path or filename (used by original_assist09 mode).",
    )
    parser.add_argument("--min_inter_num", type=int, default=3, help="Minimum interactions per user in original_assist09 mode")
    parser.add_argument("--train_file", default="", help="Optional explicit sequence CSV path")
    parser.add_argument("--output_dir", default="", help="Asset/output dir. Default: <dataset_dpath>/pebg")
    parser.add_argument("--fold", type=int, default=None, help="Target CV fold id. When set, sequence mode excludes this fold and writes to pebg/fold{fold} by default")

    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--keep_prob", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save_every", default="50,100,200")
    parser.add_argument("--output_name", default="")
    parser.add_argument("--con_sym", default="_", help="Composite skill separator used in skill_id_dict merge stage")

    args = parser.parse_args()

    data_cfg = load_cfg(args.data_config)
    if args.dataset_name not in data_cfg:
        raise KeyError(f"dataset_name '{args.dataset_name}' not found in {args.data_config}")

    dataset_cfg = data_cfg[args.dataset_name]
    dpath = dataset_cfg.get("dpath", "")
    if not dpath:
        raise ValueError(f"Dataset {args.dataset_name} has empty dpath in data config.")

    if not os.path.isabs(dpath):
        dataset_dir = os.path.normpath(os.path.join(str(ROOT), dpath))
    else:
        dataset_dir = os.path.normpath(dpath)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if args.train_file:
        sequence_path = os.path.normpath(args.train_file)
        if not os.path.isabs(sequence_path):
            sequence_path = os.path.normpath(os.path.join(str(ROOT), sequence_path))
    else:
        train_file = dataset_cfg.get("train_valid_file", "train_valid_sequences.csv")
        sequence_path = os.path.join(dataset_dir, train_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

    output_dir = args.output_dir.strip()
    if not output_dir:
        output_dir = os.path.join(dataset_dir, "pebg")
        if args.fold is not None:
            output_dir = os.path.join(output_dir, f"fold{int(args.fold)}")
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(str(ROOT), output_dir))

    print(f"[pretrain_pebg] dataset={args.dataset_name}")
    print(f"[pretrain_pebg] dataset_dir={dataset_dir}")
    print(f"[pretrain_pebg] sequence_path={sequence_path}")
    print(f"[pretrain_pebg] output_dir={output_dir}")
    print(f"[pretrain_pebg] preprocess_mode={args.preprocess_mode}")
    if args.fold is not None:
        print(f"[pretrain_pebg] fold={int(args.fold)} (excluded from pretrain assets)")

    if args.preprocess_mode == "original_assist09":
        if args.dataset_name != "assist2009":
            raise ValueError("original_assist09 mode only supports dataset_name=assist2009")
        if args.fold is not None:
            raise ValueError(
                "Fold-aware no-leak pretraining is not supported in original_assist09 mode. "
                "Use --preprocess_mode original_assist09_fold --fold <id>."
            )

        raw_file = args.raw_file.strip() if args.raw_file else "skill_builder_data_corrected_collapsed.csv"
        raw_path = os.path.normpath(raw_file)
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(dataset_dir, raw_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw assist09 csv not found: {raw_path}")

        print(f"[pretrain_pebg] raw_path={raw_path}")
        assets = _extract_original_assist09_assets(raw_path, min_inter_num=args.min_inter_num)
    elif args.preprocess_mode == "original_assist09_fold":
        if args.dataset_name != "assist2009":
            raise ValueError("original_assist09_fold mode only supports dataset_name=assist2009")
        if args.fold is None:
            raise ValueError("original_assist09_fold mode requires --fold <id>.")

        raw_file = args.raw_file.strip() if args.raw_file else "skill_builder_data_corrected_collapsed.csv"
        raw_path = os.path.normpath(raw_file)
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(dataset_dir, raw_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw assist09 csv not found: {raw_path}")

        excluded_folds = [int(args.fold)]
        allowed_users = _get_allowed_raw_users_from_folds(
            sequence_path=sequence_path,
            dataset_dir=dataset_dir,
            exclude_folds=excluded_folds,
        )
        print(f"[pretrain_pebg] raw_path={raw_path}")
        print(f"[pretrain_pebg] allowed_users_from_train_folds={len(allowed_users)}")
        assets = _extract_original_assist09_assets(
            raw_path,
            min_inter_num=args.min_inter_num,
            allowed_user_ids=allowed_users,
        )
    else:
        excluded_folds = [int(args.fold)] if args.fold is not None else None
        assets = _extract_graph_and_features(sequence_path, exclude_folds=excluded_folds)

    _save_assets(assets, dataset_dir=dataset_dir, output_dir=output_dir)

    output_emb = _run_pebg_training(args, output_dir=output_dir)

    print("[pretrain_pebg] Done.")
    print(f"[pretrain_pebg] embedding_file={output_emb}")
    print("[pretrain_pebg] To use with DKT strategy mode, set dkt_pebg.booster_strategy='pebg_auto'.")


if __name__ == "__main__":
    main()
