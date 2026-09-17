import os

from torch.utils.data import DataLoader

from core.registry import DATASET_REGISTRY
from .kt_dataset import KTDataset, KTQueDataset
from .label_noise import apply_train_label_flip


ALL_IN_ONE_DATASET_MODELS = {
    "lpkt",
    "hdkt",
    "hd_dkt",
    "hd_akt",
    "hd_simplekt",
    "atdkt",
    "dimkt",
    "stablekt",
    "sparsekt",
    "robustkt",
    "dtransformer",
    "dkt_forget",
    "dkt-forget",
    "skvmn",
    "rekt",
    "lefokt_akt",
    "lefokt",
    "hqaf",
    "hqaf_kt",
    "keenkt",
    "dgekt",
}
ONE_BY_ONE_DATASET_MODELS = {"hawkes"}
MULTI_CONCEPT_MODELS = {
    "dkt",
    "sakt",
    "akt",
    "simplekt",
    "dkvmn",
    "dkt+",
    "deep_irt",
    "stablekt",
    "sparsekt",
    "lefokt_akt",
    "skvmn",
    "atkt",
    "robustkt",
    "dimkt",
    "saint",
    "saint_plus",
    "iekt",
    "lpkt",
    "atdkt",
    "dtransformer",
    "dkt_forget",
    "dkt_pebg",
    "hqaf",
    "keenkt",
    "ukt",
    "kqn",
    "hd_dkt",
    "hd_akt",
    "hd_simplekt",
    "hdkt",
    "qikt",
}


def resolve_concept_mode(model_name, override=None):
    """Whether a model is fed every KC of a question or only the first.

    The default comes from MULTI_CONCEPT_MODELS -- membership means the model's
    embedding code pools a [B,T,K] concept tensor.  `override` (a `concept_mode`
    key in the model's config block) forces it either way, which is what makes
    "how much does truncation cost?" a one-variable experiment rather than a
    code edit.
    """
    if override in ("multi", "first"):
        return override
    if override is not None:
        raise ValueError(f"concept_mode must be 'multi' or 'first', got {override!r}.")
    return "multi" if str(model_name).lower() in MULTI_CONCEPT_MODELS else "first"


FIT_SCOPES = ("none", "train_folds", "train_valid_test")


def protocol_stamp(model_name, dataset_mode, max_concepts, concept_mode_override=None,
                   eval_window=True, feature_fit_scope="none", graph_scope="none"):
    """The evaluation protocol a run actually used, for run_config.json.

    Two runs are only comparable when these values match.  Recording them per
    run is what lets a finished comparison table be checked after the fact,
    instead of trusting that every row was produced the same way -- which is
    how a table came to mix `multi` and `first` concept handling.

    `feature_fit_scope` and `graph_scope` record which splits the run's derived
    inputs were fitted from -- difficulty maps, gap dimensions, time buckets,
    concept graphs.  None of these read responses, so none of them is label
    leakage; what they decide is whether the run is transductive, that is
    whether the model's structure was built already knowing what the test set
    contains.  The scopes are genuinely mixed across models, so they are
    recorded rather than assumed:

      dimkt, hqaf, lpkt/hdkt   train_folds       already fold-clean
      dkt_forget               train_valid_test  gap dimensions max over all splits
      gkt                      train_valid_test  transition counts include test
      dgekt                    train_valid_test  test question metadata by default
      everything else          none              fits nothing from the data

    These are descriptive.  Changing the behaviour would be a deliberate
    deviation from pyKT, which does the same in all three cases; recording it
    costs nothing and makes the question answerable from an artifact.
    """
    if feature_fit_scope not in FIT_SCOPES or graph_scope not in FIT_SCOPES:
        raise ValueError(
            f"fit scopes must be one of {FIT_SCOPES}, got "
            f"feature_fit_scope={feature_fit_scope!r}, graph_scope={graph_scope!r}."
        )
    from datasets.kt_dataset import SCORE_REPEATED_KC

    width = int(max_concepts or 1)
    if dataset_mode == "one_by_one":
        # Every KC is its own row, so the model sees all of them regardless of
        # the MULTI_CONCEPT_MODELS list -- that list only drives all_in_one.
        concept_mode, visible = "expanded", "all"
    else:
        concept_mode = resolve_concept_mode(model_name, concept_mode_override)
        visible = "all" if concept_mode == "multi" or width <= 1 else f"first_of_{width}"
    return {
        "dataset_mode": dataset_mode,
        "concept_mode": concept_mode,
        "max_concepts": width,
        "concepts_visible": visible,
        "score_repeated_kc": bool(SCORE_REPEATED_KC),
        # False means this run has no pykt-comparable windowed number.
        "eval_window": bool(eval_window),
        # Which splits the run's derived inputs were fitted from. See above:
        # descriptive, not a claim that any of it is correct.
        "feature_fit_scope": feature_fit_scope,
        "graph_scope": graph_scope,
    }


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

    time_idx_maps = kwargs.get("time_idx_maps")
    include_dkt_forget = kwargs.get("include_dkt_forget", False)
    difficulty_maps = kwargs.get("difficulty_maps")
    include_history = kwargs.get("include_history", False)
    include_hqaf_attrs = kwargs.get("include_hqaf_attrs", False)
    hqaf_feature_maps = kwargs.get("hqaf_feature_maps")
    dkt_forget_caps = kwargs.get("dkt_forget_caps")
    train_label_flip_ratio = float(kwargs.get("train_label_flip_ratio", 0.0))
    train_label_flip_seed = int(kwargs.get("train_label_flip_seed", 3407))
    model_name_lower = (model_name or "").lower()

    if dataset_mode is None:
        if model_name_lower in ALL_IN_ONE_DATASET_MODELS:
            dataset_mode = "all_in_one"
        elif model_name_lower in ONE_BY_ONE_DATASET_MODELS:
            dataset_mode = "one_by_one"

    if dataset_mode == "one_by_one":
        # One-by-One mode: use KTDataset with 1D concept sequences
        train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])
        all_folds = set(cfg["folds"])
        train_ds = KTDataset(
            train_valid_path, cfg["input_type"], all_folds - {fold},
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )
        valid_ds = KTDataset(
            train_valid_path, cfg["input_type"], {fold},
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )
    elif dataset_mode == "all_in_one":
        # ALL-in-One mode: use KTQueDataset with 2D concept sequences
        train_valid_path = _resolve_sequence_path(cfg, "train_valid_file_quelevel", "train_valid_file")
        max_concepts = cfg.get("max_concepts", 4)
        all_folds = set(cfg["folds"])

        concept_mode = resolve_concept_mode(model_name_lower, kwargs.get("concept_mode"))
        train_ds = KTQueDataset(
            train_valid_path, cfg["input_type"], all_folds - {fold},
            concept_num=cfg.get("num_c", 0), max_concepts=max_concepts, concept_mode=concept_mode,
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )
        valid_ds = KTQueDataset(
            train_valid_path, cfg["input_type"], {fold},
            concept_num=cfg.get("num_c", 0), max_concepts=max_concepts, concept_mode=concept_mode,
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )
    else:
        # Default: use KTDataset (One-by-One mode) for backward compatibility
        train_valid_path = os.path.join(cfg["dpath"], cfg["train_valid_file"])
        all_folds = set(cfg["folds"])
        train_ds = KTDataset(
            train_valid_path, cfg["input_type"], all_folds - {fold},
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )
        valid_ds = KTDataset(
            train_valid_path, cfg["input_type"], {fold},
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )

    train_ds = apply_train_label_flip(
        train_ds,
        ratio=train_label_flip_ratio,
        seed=train_label_flip_seed,
    )
    flip_info = train_ds.label_flip_info
    print(
        "Train label flip: "
        f"requested={flip_info['requested_ratio']:.4f}, "
        f"actual={flip_info['actual_ratio']:.4f}, "
        f"flipped={flip_info['flipped_count']}/{flip_info['eligible_count']}, "
        f"seed={flip_info['seed']}, "
        f"mask_sha256={flip_info['mask_sha256'][:12]}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


@DATASET_REGISTRY.register("kt_test")
def build_test_dataloaders(dataset_name, data_config, batch_size, model_name=None, dataset_mode=None, num_workers=0, use_timestamps=False, window=False, **kwargs):
    """Build test dataloaders for evaluation.

    Args:
        dataset_mode: "one_by_one" or "all_in_one"
        use_timestamps: Whether to load timestamps
        window: read the windowed test file instead of the plain one.  The
            windowed file gives every scored position a full-length history
            (one row per position) and is the file pykt reports on; it is ~20x
            larger, so it is only worth scoring once at the end of a run.
    """
    if dataset_name in data_config:
        cfg = data_config[dataset_name]
    else:
        cfg = data_config

    time_idx_maps = kwargs.get("time_idx_maps")
    include_dkt_forget = kwargs.get("include_dkt_forget", False)
    difficulty_maps = kwargs.get("difficulty_maps")
    include_history = kwargs.get("include_history", False)
    include_hqaf_attrs = kwargs.get("include_hqaf_attrs", False)
    hqaf_feature_maps = kwargs.get("hqaf_feature_maps")
    dkt_forget_caps = kwargs.get("dkt_forget_caps")
    model_name_lower = (model_name or "").lower()
    if dataset_mode is None:
        if model_name_lower in ALL_IN_ONE_DATASET_MODELS:
            dataset_mode = "all_in_one"
        elif model_name_lower in ONE_BY_ONE_DATASET_MODELS:
            dataset_mode = "one_by_one"

    if dataset_mode == "all_in_one":
        test_path = _resolve_sequence_path(
            cfg,
            "test_window_file_quelevel" if window else "test_file_quelevel",
            "test_window_file" if window else "test_file",
        )
        max_concepts = cfg.get("max_concepts", 4)
        concept_mode = resolve_concept_mode(model_name_lower, kwargs.get("concept_mode"))
        test_ds = KTQueDataset(
            test_path, cfg["input_type"], {-1},
            concept_num=cfg.get("num_c", 0), max_concepts=max_concepts, concept_mode=concept_mode,
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )
    else:
        test_key = "test_window_file" if window else "test_file"
        test_path = os.path.join(cfg["dpath"], cfg[test_key])
        test_ds = KTDataset(
            test_path, cfg["input_type"], {-1},
            use_timestamps=use_timestamps, time_idx_maps=time_idx_maps,
            include_dkt_forget=include_dkt_forget, difficulty_maps=difficulty_maps,
            include_history=include_history,
            include_hqaf_attrs=include_hqaf_attrs, hqaf_feature_maps=hqaf_feature_maps,
            dkt_forget_caps=dkt_forget_caps,
        )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
