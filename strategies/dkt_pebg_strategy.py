import os
import re
from typing import Dict, Tuple


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return os.path.normpath(p)
    return ""


def _find_best_pebg_embedding(pebg_dir: str) -> str:
    if not pebg_dir or not os.path.isdir(pebg_dir):
        return ""

    # Prefer PyTorch-trained artifacts with epoch suffix, e.g., embedding_50_pt.npz
    pt_candidates = []
    pattern = re.compile(r"^embedding_(\d+)_pt.*\.npz$")
    for name in os.listdir(pebg_dir):
        m = pattern.match(name)
        if m:
            pt_candidates.append((int(m.group(1)), os.path.join(pebg_dir, name)))

    if pt_candidates:
        pt_candidates.sort(key=lambda x: x[0], reverse=True)
        return os.path.normpath(pt_candidates[0][1])

    # Fallback to classic TensorFlow artifact if present.
    return _first_existing([
        os.path.join(pebg_dir, "embedding_200.npz"),
        os.path.join(pebg_dir, "embedding.npz"),
    ])


def _strategy_none(model_cfg: Dict, **kwargs) -> Tuple[Dict, Dict]:
    cfg = dict(model_cfg)
    cfg["use_pebg_booster"] = False
    cfg["use_original_pebg_dkt"] = False
    return cfg, {
        "strategy": "none",
        "enabled": False,
        "emb_path": "",
        "keyid2idx_path": "",
        "pro_id_dict_path": "",
    }


def _strategy_custom(model_cfg: Dict, **kwargs) -> Tuple[Dict, Dict]:
    cfg = dict(model_cfg)
    emb_path = cfg.get("emb_path", "")
    cfg["use_pebg_booster"] = bool(cfg.get("use_pebg_booster", False) or emb_path)
    cfg["require_fold_embedding"] = bool(cfg.get("require_fold_embedding", True))
    if cfg["use_pebg_booster"]:
        cfg["use_original_pebg_dkt"] = bool(cfg.get("use_original_pebg_dkt", True))
        cfg["pebg_hidden_size"] = int(cfg.get("pebg_hidden_size", 128))
    else:
        cfg["use_original_pebg_dkt"] = False
    return cfg, {
        "strategy": "custom",
        "enabled": bool(cfg.get("use_pebg_booster", False)),
        "emb_path": cfg.get("emb_path", ""),
        "keyid2idx_path": cfg.get("keyid2idx_path", ""),
        "pro_id_dict_path": cfg.get("pro_id_dict_path", ""),
        "use_original_pebg_dkt": bool(cfg.get("use_original_pebg_dkt", False)),
        "pebg_hidden_size": int(cfg.get("pebg_hidden_size", 128)),
        "require_fold_embedding": bool(cfg.get("require_fold_embedding", True)),
    }


def _strategy_pebg_auto(model_cfg: Dict, dataset_name: str, dataset_cfg: Dict, root_dir: str, fold_id=None) -> Tuple[Dict, Dict]:
    cfg = dict(model_cfg)

    pebg_dir_native = os.path.join(dataset_cfg.get("dpath", ""), "pebg") if dataset_cfg.get("dpath", "") else ""
    fold_id = int(fold_id) if fold_id is not None else None
    require_fold_embedding = bool(cfg.get("require_fold_embedding", True))

    selected_dir = pebg_dir_native

    emb_path = cfg.get("emb_path", "")
    if not emb_path:
        if fold_id is not None:
            fold_dir = os.path.join(pebg_dir_native, f"fold{fold_id}") if pebg_dir_native else ""
            selected_dir = fold_dir
            emb_path = _find_best_pebg_embedding(selected_dir)

            if not emb_path and not require_fold_embedding:
                selected_dir = pebg_dir_native
                emb_path = _find_best_pebg_embedding(selected_dir)

            if require_fold_embedding and not emb_path:
                raise FileNotFoundError(
                    "Fold-aware PEBG embedding not found. "
                    f"Expected under: {os.path.normpath(fold_dir)}. "
                    "Run scripts/pretrain_pebg.py with --preprocess_mode sequence --fold <id>."
                )
        else:
            emb_path = _find_best_pebg_embedding(selected_dir)

    keyid2idx_path = cfg.get("keyid2idx_path", "")
    if not keyid2idx_path:
        keyid2idx_path = _first_existing([
            os.path.join(dataset_cfg.get("dpath", ""), "keyid2idx.json"),
            os.path.join(root_dir, "data", dataset_name, "keyid2idx.json"),
        ])

    pro_id_dict_path = cfg.get("pro_id_dict_path", "")
    if not pro_id_dict_path:
        pro_id_dict_path = _first_existing([
            os.path.join(selected_dir, "pro_id_dict.txt"),
            os.path.join(dataset_cfg.get("dpath", ""), "pro_id_dict.txt"),
        ])

    cfg["emb_path"] = emb_path
    cfg["keyid2idx_path"] = keyid2idx_path
    cfg["pro_id_dict_path"] = pro_id_dict_path
    cfg["use_pebg_booster"] = bool(emb_path)
    cfg["require_fold_embedding"] = require_fold_embedding
    if cfg["use_pebg_booster"]:
        # Match original pebg_dkt behavior by default.
        cfg["use_original_pebg_dkt"] = True
        cfg["freeze_pretrained"] = True
        cfg["pebg_hidden_size"] = int(cfg.get("pebg_hidden_size", 128))
    else:
        cfg["use_original_pebg_dkt"] = False

    return cfg, {
        "strategy": "pebg_auto",
        "enabled": bool(cfg["use_pebg_booster"]),
        "emb_path": emb_path,
        "keyid2idx_path": keyid2idx_path,
        "pro_id_dict_path": pro_id_dict_path,
        "use_original_pebg_dkt": bool(cfg.get("use_original_pebg_dkt", False)),
        "freeze_pretrained": bool(cfg.get("freeze_pretrained", False)),
        "pebg_hidden_size": int(cfg.get("pebg_hidden_size", 128)),
        "require_fold_embedding": bool(cfg.get("require_fold_embedding", True)),
        "fold_id": fold_id,
        "pebg_dir": os.path.normpath(selected_dir),
        "pebg_dir_native": os.path.normpath(pebg_dir_native) if pebg_dir_native else "",
    }


def apply_dkt_pebg_strategy(model_cfg: Dict, dataset_name: str, dataset_cfg: Dict, root_dir: str, fold_id=None) -> Tuple[Dict, Dict]:
    strategy = str(model_cfg.get("booster_strategy", "none")).strip().lower()

    if strategy in ("", "none", "off"):
        return _strategy_none(model_cfg)
    if strategy in ("custom", "manual"):
        return _strategy_custom(model_cfg)
    if strategy in ("pebg_auto", "auto", "pebg_torch_auto"):
        return _strategy_pebg_auto(
            model_cfg,
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            root_dir=root_dir,
            fold_id=fold_id,
        )

    raise ValueError(
        f"Unknown DKT-PEBG booster_strategy: {strategy}. "
        "Supported: none, custom, pebg_auto"
    )