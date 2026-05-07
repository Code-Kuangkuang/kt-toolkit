import os

from torch.utils.data import DataLoader

from core.registry import DATASET_REGISTRY
from .kt_dataset import KTDataset, KTQueDataset


def _resolve_sequence_path(cfg, primary_key, fallback_key):
    primary_name = cfg.get(primary_key, cfg[fallback_key])
    primary_path = os.path.join(cfg["dpath"], primary_name)
    if os.path.exists(primary_path):
        return primary_path
    return os.path.join(cfg["dpath"], cfg[fallback_key])


@DATASET_REGISTRY.register("kt_default")
def build_dataloaders(dataset_name, data_config, fold, batch_size, model_name=None, dataset_mode=None, num_workers=0, use_timestamps=False, **kwargs):
    """Build dataloaders with optional mode override.

    Args:
        dataset_mode: Explicitly specify dataset mode:
            - "one_by_one": Use KTDataset (1D concept sequences)
            - "all_in_one": Use KTQueDataset (2D concept sequences)
            - None: Auto-detect (for backward compatibility, use KTDataset)
        use_timestamps: Whether to load timestamps (for HawkesKT, DKT-forget, etc.)
    """
    if dataset_name in data_config:
        cfg = data_config[dataset_name]
    else:
        cfg = data_config

    # LPKT requires question-level data (quelevel) - always use all_in_one
    if model_name == "lpkt":
        dataset_mode = "all_in_one"

    # Hawkes uses default dataset (one_by_one), not quelevel - same as pykt
    # Only override if not explicitly set by user
    if model_name == "hawkes" and dataset_mode is None:
        dataset_mode = "one_by_one"

    if dataset_mode == "one_by_one":
        # One-by-One mode: use KTDataset with 1D concept sequences
        train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])
        all_folds = set(cfg["folds"])
        train_ds = KTDataset(train_valid_path, cfg["input_type"], all_folds - {fold}, use_timestamps=use_timestamps)
        valid_ds = KTDataset(train_valid_path, cfg["input_type"], {fold}, use_timestamps=use_timestamps)
    elif dataset_mode == "all_in_one":
        # ALL-in-One mode: use KTQueDataset with 2D concept sequences
        train_valid_path = _resolve_sequence_path(cfg, "train_valid_file_quelevel", "train_valid_file")
        max_concepts = cfg.get("max_concepts", 4)
        all_folds = set(cfg["folds"])

        concept_mode = "multi" if (model_name or "").lower() in {"qikt", "gbkt", "gbktv2"} else "first"
        train_ds = KTQueDataset(
            train_valid_path, cfg["input_type"], all_folds - {fold},
            concept_num=cfg.get("num_c", 0), max_concepts=max_concepts, concept_mode=concept_mode,
            use_timestamps=use_timestamps
        )
        valid_ds = KTQueDataset(
            train_valid_path, cfg["input_type"], {fold},
            concept_num=cfg.get("num_c", 0), max_concepts=max_concepts, concept_mode=concept_mode,
            use_timestamps=use_timestamps
        )
    else:
        # Default: use KTDataset (One-by-One mode) for backward compatibility
        train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])
        all_folds = set(cfg["folds"])
        train_ds = KTDataset(train_valid_path, cfg["input_type"], all_folds - {fold}, use_timestamps=use_timestamps)
        valid_ds = KTDataset(train_valid_path, cfg["input_type"], {fold}, use_timestamps=use_timestamps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


@DATASET_REGISTRY.register("kt_quelevel")
def build_quelevel_dataloaders(dataset_name, data_config, fold, batch_size, num_workers=0, use_timestamps=False, **kwargs):
    """Explicitly use que-level dataset (ALL-in-One mode)."""
    if dataset_name in data_config:
        cfg = data_config[dataset_name]
    else:
        cfg = data_config

    train_valid_path = _resolve_sequence_path(cfg, "train_valid_file_quelevel", "train_valid_file")
    max_concepts = cfg.get("max_concepts", 4)
    all_folds = set(cfg["folds"])

    train_ds = KTQueDataset(
        train_valid_path, cfg["input_type"], all_folds - {fold},
        concept_num=cfg.get("num_c", 0), max_concepts=max_concepts,
        use_timestamps=use_timestamps
    )
    valid_ds = KTQueDataset(
        train_valid_path, cfg["input_type"], {fold},
        concept_num=cfg.get("num_c", 0), max_concepts=max_concepts,
        use_timestamps=use_timestamps
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


@DATASET_REGISTRY.register("kt_test")
def build_test_dataloaders(dataset_name, data_config, batch_size, model_name=None, dataset_mode=None, num_workers=0, use_timestamps=False, **kwargs):
    """Build test dataloaders for evaluation.

    Args:
        dataset_mode: "one_by_one" or "all_in_one"
        use_timestamps: Whether to load timestamps
    """
    if dataset_name in data_config:
        cfg = data_config[dataset_name]
    else:
        cfg = data_config

    if dataset_mode == "all_in_one":
        test_path = _resolve_sequence_path(cfg, "test_file_quelevel", "test_file")
        max_concepts = cfg.get("max_concepts", 4)
        concept_mode = "multi" if (model_name or "").lower() in {"qikt", "gbkt", "gbktv2"} else "first"
        test_ds = KTQueDataset(
            test_path, cfg["input_type"], {-1},
            concept_num=cfg.get("num_c", 0), max_concepts=max_concepts, concept_mode=concept_mode,
            use_timestamps=use_timestamps
        )
    else:
        test_path = os.path.join(cfg["dpath"], cfg["test_file"])
        test_ds = KTDataset(test_path, cfg["input_type"], {-1}, use_timestamps=use_timestamps)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
