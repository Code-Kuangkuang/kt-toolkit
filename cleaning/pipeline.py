# Cleaning pipeline dispatcher.
from cleaning.adapters import ADAPTERS

def run_cleaning(cfg):
    name = cfg["dataset_name"]
    if name not in ADAPTERS:
        raise ValueError(f"Unknown dataset name: {name}")
    from pathlib import Path
    from core.experiment_identity import file_revision
    from core.run_support import save_run_config
    for key in ("raw_path", "configf", "min_seq_len", "maxlen", "kfold"):
        if key not in cfg:
            raise ValueError(f"Cleaning {name}: missing {key}.")
    source = Path(cfg["raw_path"])
    if not cfg["raw_path"] or not source.exists():
        raise ValueError(f"Cleaning {name}: raw_path does not exist: {source}")
    if not 1 <= int(cfg["min_seq_len"]) <= int(cfg["maxlen"]) or int(cfg["kfold"]) < 2:
        raise ValueError("Require 1 <= min_seq_len <= maxlen and kfold >= 2.")
    result = ADAPTERS[name](cfg)
    dpath = Path(cfg.get("dpath", source.parent))
    files = {p.name: file_revision(p) for p in sorted(dpath.glob("*.csv"))}
    save_run_config(str(dpath / "dataset_manifest.json"), {
        "version": 1, "dataset_name": name, "split_seed": 1024,
        "cleaning_config": dict(cfg), "source_revision": file_revision(source),
        "output_revision": files})
    return result
