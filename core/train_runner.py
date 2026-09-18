import copy
import datetime
import os
import uuid

import torch
from rich import print

from core.factory import build_dataset, build_model, build_trainer
from core.dataset_names import is_hidden_label_dataset, normalize_dataset_name
from core.model_info import collect_model_info, save_model_info_once
from core.model_inputs import RunContext, spec_for
from core.registry import MODEL_REGISTRY
from core.run_support import (
    aggregate_fold_metrics,
    apply_overrides,
    build_hooks,
    build_optimizer,
    print_cv_summary,
    print_run_overview,
    save_cv_summary,
    save_run_config,
    set_seed,
)
from datasets.init_dataset import protocol_stamp


MODEL_NAME_ALIASES = {
    "dkt-forget": "dkt_forget",
    "hd-kt": "hdkt",
    "hd_lpkt": "hdkt",
    "hd-lpkt": "hdkt",
    "hd-dkt": "hd_dkt",
    "hd-akt": "hd_akt",
    "hd-simplekt": "hd_simplekt",

    "lefokt": "lefokt_akt",
    "hqaf-kt": "hqaf",
    "hqaf_kt": "hqaf",
}


# Keys that live in a model's config block but are consumed by something other
# than its constructor, so passing them on would either be a TypeError or -- for
# a model taking **kwargs -- silently absorbed.
#
# Grouped by who actually reads each one, because an ungrouped list is how this
# accumulated nine dead entries from the deleted a removed model models before anyone
# noticed. tests/test_config_keys_are_consumed.py checks that every key in every
# config block reaches one of these consumers, so a typo cannot hide here.
NON_MODEL_CONFIG_KEYS = {
    # Read by this runner.
    "emb_type", "learning_rate", "use_timestamps", "dpath", "dataset_mode",
    "num_at", "num_it",
    # Read by the dataset builder.
    "concept_mode", "eval_window",
    # Read by strategies/dkt_pebg_strategy.py.
    "booster_strategy", "require_fold_embedding",
    # Read by the trainer through `other_config`, not by the model.
    "loss_c_all_lambda", "loss_q_all_lambda", "loss_c_next_lambda",
    "loss_q_next_lambda", "output_mode", "output_c_all_lambda",
    "output_c_next_lambda", "output_q_all_lambda", "output_q_next_lambda",
}


def resolve_dataset_mode(model_name, train_cfg, model_cfg, overrides=None, spec=None):
    """Resolve the data mode.

    Priority, highest first:

      1. CLI override
      2. the model's config block
      3. the model's `Inputs.dataset_mode` declaration
      4. the global training config

    This replaced two hand-maintained sets, ALL_IN_ONE_MODELS and
    ONE_BY_ONE_MODELS. A model now states its own mode next to its code, so
    adding one cannot silently miss a list, and `spec=None` simply falls through
    to the global default.

    Public because the contract tests have to resolve the mode exactly as the
    runner does; picking a mode independently means testing a model in a
    configuration that never occurs.
    """
    overrides = overrides or {}
    if overrides.get("dataset_mode") is not None:
        mode = overrides["dataset_mode"]
    elif model_cfg.get("dataset_mode") is not None:
        mode = model_cfg["dataset_mode"]
    elif spec is not None and getattr(spec, "dataset_mode", None) is not None:
        mode = spec.dataset_mode
    else:
        mode = train_cfg.get("dataset_mode", "one_by_one")
    if mode not in {"one_by_one", "all_in_one"}:
        raise ValueError(
            f"Unsupported dataset_mode={mode!r} for {model_name}; "
            "expected 'one_by_one' or 'all_in_one'."
        )
    return mode


# Kept so existing callers and tests that used the private name keep working.
_resolve_dataset_mode = resolve_dataset_mode


def _fmt(value):
    """Format a metric that may legitimately be absent (single-class split)."""
    return "N/A" if value is None else f"{value:.4f}"


def _resolve_existing_sequence_filename(dataset_cfg, primary_key, fallback_key):
    """Return the filename that the dataset builder will actually load."""
    primary_name = dataset_cfg.get(primary_key) or dataset_cfg.get(fallback_key)
    fallback_name = dataset_cfg.get(fallback_key)
    for filename in (primary_name, fallback_name):
        if not filename:
            continue
        if os.path.exists(os.path.join(dataset_cfg["dpath"], filename)):
            return filename
    return primary_name or fallback_name


def train_one_fold(
    *,
    dataset_name,
    model_name,
    emb_type,
    fold_id,
    root_dir,
    kt_cfg_raw,
    data_config_raw,
    seed,
    save_root,
    add_uuid,
    wandb_cfg,
    kt_config_path,
    data_config_path,
    wandb_config_path,
    cv_run_name,
    overrides,
    gpu_id=0,
    train_label_flip_ratio=0.0,
    train_label_flip_seed=None,
):
    dataset_name = normalize_dataset_name(dataset_name)
    model_name = MODEL_NAME_ALIASES.get(model_name.lower(), model_name.lower())
    kt_cfg = copy.deepcopy(kt_cfg_raw)
    train_cfg_local = kt_cfg["train_config"]
    if model_name not in kt_cfg:
        raise KeyError(
            f"No hyperparameter block for model {model_name!r} in the kt config. "
            f"Add a top-level {model_name!r} object to configs/kt_config.json; "
            "see docs/architecture.md, 'Adding A Model'."
        )
    model_cfg_local = kt_cfg[model_name]

    resolved_emb_type = emb_type or model_cfg_local.get("emb_type", "qid")
    model_cfg_local["emb_type"] = resolved_emb_type

    apply_overrides(train_cfg_local, model_cfg_local, overrides)

    train_label_flip_ratio = float(train_label_flip_ratio)
    if not 0.0 <= train_label_flip_ratio <= 1.0:
        raise ValueError(
            "train_label_flip_ratio must be in [0, 1], "
            f"got {train_label_flip_ratio}."
        )
    if train_label_flip_seed is None:
        train_label_flip_seed = seed
    train_label_flip_seed = int(train_label_flip_seed)
    train_cfg_local["train_label_flip_ratio"] = train_label_flip_ratio
    train_cfg_local["train_label_flip_seed"] = train_label_flip_seed
    # The seed arrives as a parameter, not through the config, but everything
    # downstream that wants to stamp a row with the run's identity reads
    # train_cfg. Without this, metrics.jsonl records seed=null.
    train_cfg_local["seed"] = seed

    if train_cfg_local.get("patience") == -1:
        train_cfg_local["patience"] = None

    data_config = copy.deepcopy(data_config_raw)
    dataset_cfg_local = data_config[dataset_name]
    if "dpath" in dataset_cfg_local:
        dpath = dataset_cfg_local["dpath"]
        if not os.path.isabs(dpath):
            dpath = os.path.join(root_dir, dpath)
        dataset_cfg_local["dpath"] = os.path.normpath(dpath)

    # Migration in progress: models that declare an `Inputs` spec get their
    # extra inputs from it, and the `model_name` chain below is skipped for
    # them. See "Model Input Specs" in docs/architecture.md; the chain shrinks by
    # one model at a time, and each move is checked for bit-identical metrics by
    # research/check_input_refactor.py.
    #
    # Looked up before the mode is resolved, because the spec is allowed to
    # declare that mode.
    if model_name not in MODEL_REGISTRY.get_all():
        raise KeyError(
            f"Model {model_name!r} is not registered. Registered models: "
            f"{', '.join(sorted(MODEL_REGISTRY.get_all()))}. A model class only "
            "registers once its module is imported -- check models/__init__.py."
        )
    spec = spec_for(MODEL_REGISTRY.get(model_name))

    resolved_dataset_mode = resolve_dataset_mode(
        model_name, train_cfg_local, model_cfg_local, overrides, spec=spec
    )
    train_cfg_local["dataset_mode"] = resolved_dataset_mode
    # A per-model opt-out for the windowed test set, declared in the model's
    # config block but consumed as a training-run setting.
    if model_cfg_local.get("eval_window") is not None:
        train_cfg_local["eval_window"] = bool(model_cfg_local["eval_window"])
    model_cfg_local["dataset_mode"] = resolved_dataset_mode

    spec_ctx = RunContext(
        model_name=model_name,
        dataset_name=dataset_name,
        fold_id=fold_id,
        dataset_mode=resolved_dataset_mode,
        model_cfg=model_cfg_local,
        dataset_cfg=dataset_cfg_local,
        train_cfg=train_cfg_local,
        root_dir=root_dir,
        resolve_file=lambda primary, fallback: _resolve_existing_sequence_filename(
            dataset_cfg_local, primary, fallback
        ),
    )
    spec.validate(spec_ctx)
    spec_inputs = spec.prepare(spec_ctx)
    model_cfg_local.update(spec_inputs.model_cfg_updates)
    dataset_cfg_local.update(spec_inputs.dataset_cfg_updates)

    # Which splits this run's derived inputs were fitted from, recorded in the
    # protocol block. Set at the site that does the fitting rather than from a
    # lookup table, because a table drifts away from the code -- which is the
    # failure mode of the membership sets above.
    feature_fit_scope = "none"
    graph_scope = "none"

    # AGENTS.md: "数据派生特征只能用当前 fold 的训练部分拟合，再应用到 valid/test."
    # Three models did not follow it -- dkt_forget sized its gap tables from
    # every split, gkt counted its transition graph from train and test, dgekt
    # pulled in test question metadata. None reads a response, so none is label
    # leakage, but all three make the run transductive: the model's structure is
    # built already knowing what the test set holds, which is not something
    # deployment gives you.
    #
    # The default now follows the rule. `pykt_transductive: true` restores the
    # old behaviour for tables that have to line up with pyKT's published
    # numbers, exactly as `keep_scaffolding` does for the ASSISTments filter.
    # The two are recorded separately in the protocol block and cannot share a
    # table.
    pykt_transductive = bool(train_cfg_local.get("pykt_transductive", False))

    # Migrated: every model that derives inputs now declares them in its own
    # `Inputs` spec, applied above. What remains below is the generic path.

    model_kwargs = {
        k: v for k, v in model_cfg_local.items() if k not in NON_MODEL_CONFIG_KEYS
    }
    # Applied last so a spec wins over the legacy chain during the migration.
    model_kwargs.update(spec_inputs.model_kwargs)
    # A spec reports its own fit scope; `None` means it has nothing to declare.
    if spec_inputs.feature_fit_scope is not None:
        feature_fit_scope = spec_inputs.feature_fit_scope
    if spec_inputs.graph_scope is not None:
        graph_scope = spec_inputs.graph_scope

    # Every spec has run by now, so the RNG stream from here on is identical to
    # what it was before the migration. Nothing below may consume randomness
    # ahead of this call.
    set_seed(seed)

    # Set device with specified GPU ID
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"

    model = build_model(
        model_name,
        num_c=dataset_cfg_local["num_c"],
        num_q=dataset_cfg_local["num_q"],
        emb_type=resolved_emb_type,
        seq_len=train_cfg_local.get("seq_len"),
        device=device,
        dpath=dataset_cfg_local.get("dpath", ""),
        num_at=model_cfg_local.get("num_at"),
        num_it=model_cfg_local.get("num_it"),
        **model_kwargs,
    ).to(device)

    # Hawkes applies its own init and runs in double precision; that is the only
    # post-construction work any model needs, and it lives in its spec.
    model = spec.post_build(model, spec_ctx)

    # What was built, as opposed to what was asked for. Written to save_root
    # rather than the fold's directory so a five-fold run produces one file --
    # see save_model_info_once for the models where it legitimately produces
    # more.
    model_info = collect_model_info(model, device=device)
    save_model_info_once(save_root, model_info, fold_id)

    # Resolve timestamp loading before any run overview/logging.
    model_use_timestamps = model_cfg_local.get("use_timestamps", False)
    use_timestamps = bool(train_cfg_local.get("use_timestamps", False) or model_use_timestamps)
    train_cfg_local["use_timestamps"] = use_timestamps
    print_run_overview(
        device, model, model_cfg_local, dataset_cfg_local, train_cfg_local,
        model_info=model_info,
    )

    # Get dataset_mode from overrides (if specified)
    dataset_mode = train_cfg_local.get("dataset_mode")
    dataset_feature_kwargs = {
        # Optional `concept_mode` in the model's config block forces multi/first
        # instead of taking it from MULTI_CONCEPT_MODELS, so the cost of
        # truncation can be measured without editing code.
        "concept_mode": model_cfg_local.get("concept_mode"),
    }
    dataset_feature_kwargs.update(spec_inputs.dataset_kwargs)

    train_loader, valid_loader = build_dataset(
        "kt_default",
        dataset_name=dataset_name,
        data_config=data_config,
        fold=fold_id,
        batch_size=train_cfg_local["batch_size"],
        model_name=model_name,
        dataset_mode=dataset_mode,
        use_timestamps=use_timestamps,
        train_label_flip_ratio=train_label_flip_ratio,
        train_label_flip_seed=train_label_flip_seed,
        **dataset_feature_kwargs,
    )
    train_label_flip_info = train_loader.dataset.label_flip_info

    # A loader that fails to build used to print a warning and continue, which
    # meant the run finished with no test metric while looking successful. That
    # fold then dropped out of the cross-validation mean, which is exactly the
    # partial-average problem `aggregate_fold_metrics` now reports -- but a
    # visible shortfall is second best to not having one. Failing here costs a
    # rerun; not failing costs a table that is quietly four folds wide.
    #
    # `allow_missing_test_loader: true` in the training config restores the old
    # behaviour for the case where it is genuinely expected.
    allow_missing_test = bool(train_cfg_local.get("allow_missing_test_loader", False))

    def _loader_failed(kind, exc):
        if allow_missing_test:
            print(f"Warning: could not build the {kind} loader, continuing without it: {exc}")
            return None
        raise RuntimeError(
            f"Could not build the {kind} loader for {dataset_name} "
            f"{model_name} fold {fold_id}: {exc}\n"
            "This run would finish with no test metric and silently narrow any "
            "cross-validation mean it feeds. Set "
            "`allow_missing_test_loader: true` in the training config if that is "
            "genuinely expected here."
        ) from exc

    # Build test dataloader if data exists. AAAI2023's official test file has
    # hidden targets marked as -1, so it is for prediction/submission only.
    test_loader = None
    has_hidden_test_labels = is_hidden_label_dataset(dataset_name)
    test_file_key = (
        "test_file_quelevel" if dataset_mode == "all_in_one" else "test_file"
    )
    test_filename = _resolve_existing_sequence_filename(
        dataset_cfg_local,
        test_file_key,
        "test_file",
    )
    test_path = os.path.join(dataset_cfg_local["dpath"], test_filename or "")
    if has_hidden_test_labels:
        print(
            "AAAI2023 test evaluation disabled: pykt_test.csv contains hidden "
            "targets marked as -1. Use scripts/predict_aaai2023.py to generate prediction.csv."
        )
    elif not test_filename or not os.path.exists(test_path):
        # A missing file used to fall through this branch and skip the loader
        # without a word, which is the same silent outcome as a build failure.
        test_loader = _loader_failed(
            "test", FileNotFoundError(f"no test sequence file at {test_path}")
        )
    else:
        try:
            test_loader = build_dataset(
                "kt_test",
                dataset_name=dataset_name,
                data_config=data_config,
                batch_size=train_cfg_local["batch_size"],
                model_name=model_name,
                dataset_mode=dataset_mode,
                use_timestamps=use_timestamps,
                **dataset_feature_kwargs,
            )
            print(f"Test loader built from: {test_path}")
        except Exception as e:
            test_loader = _loader_failed("test", e)

    # The windowed test file is what pykt reports on: one row per position, each
    # with a full-length history, instead of non-overlapping chunks that leave
    # boundary positions with almost none.  It is ~20x larger, so it is scored
    # once at the end rather than during training.
    window_test_loader = None
    # Models whose forward runs a per-timestep Python loop (SKVMN's hop-LSTM is
    # the worst) take hours on the windowed file, which has ~20x the rows.  They
    # can opt out with `eval_window: false` in their config block; the choice is
    # recorded in the protocol stamp so a table cannot silently mix rows that
    # have a windowed number with rows that do not.
    eval_window = bool(train_cfg_local.get("eval_window", True))
    window_file_key = (
        "test_window_file_quelevel" if dataset_mode == "all_in_one" else "test_window_file"
    )
    window_filename = _resolve_existing_sequence_filename(
        dataset_cfg_local,
        window_file_key,
        "test_window_file",
    )
    window_path = os.path.join(dataset_cfg_local["dpath"], window_filename or "")
    if not eval_window:
        print("Windowed test evaluation disabled for this model (eval_window=false).")
    elif test_loader is None:
        # Only reachable with allow_missing_test_loader, or on a hidden-label
        # dataset, both of which already explained themselves.
        pass
    elif not window_filename:
        # The dataset names no windowed file at all, which is how it says it has
        # no windowed split -- junyi_sub5k is built that way. That is a property
        # of the dataset rather than a broken run, so it continues; the claim is
        # withdrawn below, where eval_window is reconciled with what happened.
        print(
            f"Dataset {dataset_name} declares no windowed test file, so this run "
            "records eval_window=false and has no pyKT-comparable number."
        )
    elif not os.path.exists(window_path):
        # Declared and absent is a different thing: the dataset is incomplete.
        # `eval_window` goes into the protocol block, so continuing quietly would
        # produce a run claiming a pyKT-comparable windowed metric it does not
        # have. Either the file exists or the claim is withdrawn deliberately.
        window_test_loader = _loader_failed(
            "windowed test",
            FileNotFoundError(
                f"{dataset_name} names {window_filename!r} but it is not at "
                f"{window_path}. Regenerate it, or set `eval_window: false` so "
                "the protocol block stops claiming a windowed metric."
            ),
        )
    else:
        try:
            window_test_loader = build_dataset(
                "kt_test",
                dataset_name=dataset_name,
                data_config=data_config,
                batch_size=train_cfg_local["batch_size"],
                model_name=model_name,
                dataset_mode=dataset_mode,
                use_timestamps=use_timestamps,
                window=True,
                **dataset_feature_kwargs,
            )
            print(f"Windowed test loader built from: {window_path}")
        except Exception as e:
            window_test_loader = _loader_failed("windowed test", e)

    # The protocol block must describe what happened, not what was asked for.
    # With allow_missing_test_loader set, a run can reach here having wanted a
    # windowed metric and not got one; recording eval_window=true then would put
    # it in the same table group as runs that have the number.
    if eval_window and window_test_loader is None:
        print(
            "No windowed test loader was built, so this run records "
            "eval_window=false and has no pyKT-comparable number."
        )
        eval_window = False
        train_cfg_local["eval_window"] = False

    opt = build_optimizer(train_cfg_local, model_cfg_local, model)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dataset_name}-{model_name}-fold{fold_id}-{ts}"
    if train_label_flip_ratio > 0:
        ratio_tag = f"{train_label_flip_ratio:g}".replace(".", "p")
        run_name = f"{run_name}-flip{ratio_tag}"
    if add_uuid == 1:
        run_name = f"{run_name}-{uuid.uuid4()}"
    ckpt_dir = os.path.join(save_root, run_name)

    run_config = {
        "run_name": run_name,
        "timestamp": ts,
        "save_dir": save_root,
        "ckpt_dir": ckpt_dir,
        "device": device,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "emb_type": resolved_emb_type,
        "fold": fold_id,
        "seed": seed,
        "train_label_flip": train_label_flip_info,
        # Two runs are comparable only when this block matches.
        "protocol": protocol_stamp(
            model_name,
            train_cfg_local.get("dataset_mode"),
            dataset_cfg_local.get("max_concepts"),
            model_cfg_local.get("concept_mode"),
            eval_window=bool(train_cfg_local.get("eval_window", True)),
            feature_fit_scope=feature_fit_scope,
            graph_scope=graph_scope,
        ),
        "use_wandb": bool(wandb_cfg),
        "add_uuid": bool(add_uuid),
        "cv": {
            "enabled": bool(cv_run_name),
            "cv_run_name": cv_run_name,
        },
        "config_paths": {
            "kt_config": kt_config_path,
            "data_config": data_config_path,
            "wandb_config": wandb_config_path if wandb_cfg else None,
        },
        "train_config": train_cfg_local,
        "model_config": model_cfg_local,
        "dataset_config": dataset_cfg_local,
        # dgekt_graph and booster_info arrive here from their specs.
        **spec_inputs.run_config_extras,
        "wandb": {
            "enabled": bool(wandb_cfg),
            "project": wandb_cfg.get("project") if wandb_cfg else None,
            "entity": wandb_cfg.get("entity") if wandb_cfg else None,
            "mode": wandb_cfg.get("mode") if wandb_cfg else None,
        },
    }
    save_run_config(os.path.join(ckpt_dir, "run_config.json"), run_config)

    hooks = build_hooks(
        wandb_cfg,
        run_name,
        dataset_name,
        model_name,
        fold_id,
        train_cfg_local,
        model_cfg_local,
        cv_run_name,
        ckpt_dir,
        resolved_emb_type,
    )

    trainer_kwargs = {
        "model": model,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "optimizer": opt,
        "num_epochs": train_cfg_local["num_epochs"],
        "device": device,
        "hooks": hooks,
        "other_config": model_cfg_local,
        "test_loader": test_loader,
    }
    if "patience" in train_cfg_local:
        trainer_kwargs["patience"] = train_cfg_local["patience"]

    trainer = build_trainer(model_name, **trainer_kwargs)
    # Attached rather than passed through __init__: every trainer subclass
    # declares its own constructor, so a new keyword would break all of them.
    trainer.window_test_loader = window_test_loader
    trainer.run()

    best_path = getattr(trainer, "best_path", None)



    last_epoch_path = os.path.join(ckpt_dir, "last_epoch_model.pt")
    torch.save(trainer.model.state_dict(), last_epoch_path)


    best_test_metrics = None
    best_window_metrics = None
    if best_path and os.path.exists(best_path):
        trainer.model.load_state_dict(torch.load(best_path, weights_only=True))
        if test_loader is not None:
            print(f"Loaded best model from epoch {trainer.best_metrics.get('epoch', '?')} for test evaluation")
            best_test_metrics = trainer.evaluate_test()
            if best_test_metrics:
                print(
                    f"[Best-Valid Epoch] Test AUC={_fmt(best_test_metrics.get('test_auc'))}, "
                    f"ACC={_fmt(best_test_metrics.get('test_acc'))}"
                )
            if window_test_loader is not None:
                best_window_metrics = trainer.evaluate_window_test()
                if best_window_metrics:
                    print(
                        f"[Best-Valid Epoch] Window Test AUC="
                        f"{_fmt(best_window_metrics.get('window_test_auc'))}, "
                        f"ACC={_fmt(best_window_metrics.get('window_test_acc'))}"
                        "   (pykt-comparable protocol)"
                    )
        else:
            print(f"Loaded best model from epoch {trainer.best_metrics.get('epoch', '?')}")


    # The last-epoch checkpoint is a diagnostic -- how far the model drifted
    # after its best validation epoch -- not a reportable result. It is off by
    # default so the test set is touched exactly once per run, on the
    # best-validation checkpoint; `eval_last_epoch: true` in the training config
    # turns it back on.
    last_test_metrics = None
    if test_loader is not None and bool(train_cfg_local.get("eval_last_epoch", False)):
        trainer.model.load_state_dict(torch.load(last_epoch_path, weights_only=True))
        print(f"Loaded last epoch model for test evaluation")
        last_test_metrics = trainer.evaluate_test()
        if last_test_metrics:
            print(
                f"[Last Epoch]        Test AUC={_fmt(last_test_metrics.get('test_auc'))}, "
                f"ACC={_fmt(last_test_metrics.get('test_acc'))}"
            )
        # Leave the model holding the weights this run actually selected. The
        # last-epoch load above is only for the secondary metric, and nothing
        # should inherit it by accident.
        if best_path and os.path.exists(best_path):
            trainer.model.load_state_dict(torch.load(best_path, weights_only=True))

    best_metrics = getattr(trainer, "best_metrics", None)
    if best_metrics is not None:
        # Rename keys so best_metric dict carries unambiguous names. None is
        # carried through rather than defaulted to -1: aggregate_fold_metrics
        # skips None and counts the fold as missing, whereas -1 would be
        # averaged in as a real score.
        if best_test_metrics:
            best_metrics["best_test_auc"] = best_test_metrics.get("test_auc")
            best_metrics["best_test_acc"] = best_test_metrics.get("test_acc")
        if best_window_metrics:
            best_metrics["best_window_test_auc"] = best_window_metrics.get("window_test_auc")
            best_metrics["best_window_test_acc"] = best_window_metrics.get("window_test_acc")
        if last_test_metrics:
            best_metrics["last_test_auc"] = last_test_metrics.get("test_auc")
            best_metrics["last_test_acc"] = last_test_metrics.get("test_acc")
        save_run_config(os.path.join(ckpt_dir, "best_metrics.json"), best_metrics)

    return {
        "fold": fold_id,
        "run_name": run_name,
        "ckpt_dir": ckpt_dir,
        "emb_type": resolved_emb_type,
        "train_label_flip": train_label_flip_info,
        "best_metrics": best_metrics,
        "best_path": best_path,
        # Carried so the cross-validation loop can check that every fold it is
        # about to average was produced under the same protocol.
        "protocol": run_config["protocol"],
        "seed": seed,
    }
