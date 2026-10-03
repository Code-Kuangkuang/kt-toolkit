"""Content-based experiment identity shared by training, reuse and summaries."""
import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from core.dataset_names import normalize_dataset_name
from core.model_inputs import spec_for
from core.registry import MODEL_REGISTRY
from core.run_support import apply_overrides


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1024)
def _file_digest(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_revision(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def source_revision(root):
    root = Path(root)
    resources = {str(p.relative_to(root)).replace("\\", "/"): file_revision(p)
                 for p in sorted((root / "utils").rglob("*"))
                 if p.is_file() and p.suffix in {".json", ".npy", ".npz"}}
    return fingerprint({"resources": resources, "source": {str(p.relative_to(root)).replace("\\", "/"): file_revision(p)
                        for folder in ("core", "models", "datasets", "modules", "plugins", "strategies", "scripts")
                        for p in sorted((root / folder).rglob("*.py"))}})


def build_experiment_identity(*, root_dir, dataset_name, model_name, kt_cfg_raw,
                              data_config_raw, seed, fold_id, emb_type=None,
                              overrides=None, train_label_flip_ratio=0.0,
                              train_label_flip_seed=None):
    # Resolve exactly the request, before InputSpec adds fitted constructor values.
    from core.train_runner import MODEL_NAME_ALIASES, resolve_dataset_mode
    dataset_name = normalize_dataset_name(dataset_name)
    model_name = MODEL_NAME_ALIASES.get(model_name.lower(), model_name.lower())
    train = copy.deepcopy(kt_cfg_raw["train_config"])
    model = copy.deepcopy(kt_cfg_raw[model_name])
    apply_overrides(train, model, overrides)
    model["emb_type"] = emb_type or model.get("emb_type", "qid")
    train["dataset_mode"] = resolve_dataset_mode(model_name, train, model, overrides,
                                                spec=spec_for(MODEL_REGISTRY.get(model_name)))
    model["dataset_mode"] = train["dataset_mode"]
    if model.get("eval_window") is not None:
        train["eval_window"] = bool(model["eval_window"])
    if train.get("patience") == -1:
        train["patience"] = None
    # Epoch recovery is an execution option, not a new scientific experiment.
    train.pop("resume_checkpoint", None)
    dataset = copy.deepcopy(data_config_raw[dataset_name])
    data_root = Path(dataset.get("dpath", ""))
    if not data_root.is_absolute():
        data_root = Path(root_dir) / data_root
    try:
        dataset["dpath"] = data_root.resolve().relative_to(Path(root_dir).resolve()).as_posix()
    except ValueError:
        dataset["dpath"] = str(data_root.resolve())
    sources = {key: file_revision(data_root / value)
               for key, value in dataset.items()
               if isinstance(value, str) and ("file" in key or key.endswith("path")) and key != "dpath"}
    sources["keyid2idx.json"] = file_revision(data_root / "keyid2idx.json")
    sources["dataset_manifest.json"] = file_revision(data_root / "dataset_manifest.json")
    payload = dict(dataset_name=dataset_name, model_name=model_name, seed=int(seed),
                   train_config=train, model_config=model, dataset_config=dataset,
                   data_revision=sources, source_revision=source_revision(root_dir),
                   score_repeated_kc=os.environ.get("KT_SCORE_REPEATED_KC", "0") == "1",
                   train_label_flip_ratio=float(train_label_flip_ratio),
                   train_label_flip_seed=int(seed if train_label_flip_seed is None else train_label_flip_seed))
    comparison_key = fingerprint(payload)
    return {"version": 1, "comparison_key": comparison_key,
            "key": fingerprint({"comparison_key": comparison_key, "fold": int(fold_id)}),
            "request": payload}


def best_checkpoint(run_dir, config):
    """Select only the best-validation weights, never last/recovery/graph tensors."""
    run_dir = Path(run_dir)
    for name in (f"{config['model_name']}_{config.get('emb_type', 'qid')}_model.pt",
                 f"{config['model_name']}_model.pt"):
        path = run_dir / name
        if path.is_file():
            return str(path)
    return None


def validate_fold_results(results):
    if not results:
        raise ValueError("No completed folds to aggregate.")
    seen = set()
    reference = results[0]
    for result in results:
        for key in ("dataset_name", "model_name", "seed", "emb_type", "comparison_key"):
            if result.get(key) is None:
                raise ValueError(f"Missing experiment metadata: {key}")
        fold = result.get("fold")
        if fold is None or fold in seen:
            raise ValueError(f"Missing or duplicate fold: {fold!r}")
        seen.add(fold)
        if not result.get("protocol"):
            raise ValueError(f"fold {fold}: missing protocol; cannot establish comparability.")
        for key in ("dataset_name", "model_name", "seed", "emb_type", "protocol", "comparison_key"):
            if result.get(key) != reference.get(key):
                raise ValueError(f"fold {fold}: {key} differs from fold {reference.get('fold')}; cannot average.")
