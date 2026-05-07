import multiprocessing as mp
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import optuna


# =========================
# User Config: edit here
# =========================
# Hyperparameter configuration table
HYPERPARAMS = [
    {"name": "C1", "exp_key": "C1_exp", "range": (-6, 10), "base": 2.0},
    {"name": "C2", "exp_key": "C2_exp", "range": (-6, 10), "base": 2.0},
    {"name": "C4", "exp_key": "C4_exp", "range": (-6, 10), "base": 2.0},
    {"name": "epsilon", "exp_key": "epsilon_exp", "range": (-3, 0), "base": 10.0},
]

# Optuna settings
TRIALS_PER_WORKER = 5
SEED = 3407
STUDY_NAME = "gbsvkt_c_search"
STORAGE = "sqlite:///optuna_gbsvkt.db"
OBJECTIVE_METRIC = "valid_auc"
DIRECTION = "maximize"  # "maximize" or "minimize"

# Training settings
DATASET_NAME = "assist2009"
MODEL_NAME = "gbsvkt"
FOLDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 200
USE_WANDB = 0
SAVE_DIR = "saved_model"

# Fold-level parallel settings
FOLD_PARALLEL = False
FOLD_PARALLEL_WORKERS = 1

# Worker settings: all workers share one DB/study and sample from the same full range.
WORKER_PROCESSES = 2
GPU_IDS = [0, 1]

_LAUNCH_TRAIN = None


def _runtime_log(event: str, **kwargs) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    print(f"[{ts}] [{event}] {payload}", flush=True)


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handler)


def _ensure_objective_metric() -> None:
    if OBJECTIVE_METRIC in {"test_auc", "best_test_auc", "test_acc", "best_test_acc"}:
        raise ValueError(
            "OBJECTIVE_METRIC must not use the test set for hyperparameter tuning. "
            f"Got: {OBJECTIVE_METRIC}. Use 'valid_auc' instead."
        )


def _get_launch_train():
    global _LAUNCH_TRAIN
    if _LAUNCH_TRAIN is None:
        from scripts.train import launch_train

        _LAUNCH_TRAIN = launch_train
    return _LAUNCH_TRAIN


def _run_single_fold(job: dict) -> dict:
    """Run a single fold training job in one process."""
    # CUDA_DEVICE_ORDER already set by parent process, just ensure CUDA_VISIBLE_DEVICES
    if "cuda_visible_devices" in job:
        os.environ["CUDA_VISIBLE_DEVICES"] = job["cuda_visible_devices"]

    worker_id = job.get("worker_id", "unknown")
    trial_number = job.get("trial_number", "unknown")
    bound_gpu = job.get("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    _runtime_log(
        "fold-start",
        worker_id=worker_id,
        pid=os.getpid(),
        gpu=bound_gpu,
        trial=trial_number,
        fold=job["fold"],
    )

    launch_train = _get_launch_train()

    result = launch_train(
        dataset_name=job["dataset_name"],
        model_name=job["model_name"],
        fold=job["fold"],
        num_epochs=job["num_epochs"],
        use_wandb=job["use_wandb"],
        save_dir=job["save_dir"],
        seed=job["seed"],
        C1=job["C1"],
        C2=job["C2"],
        C4=job["C4"],
        epsilon=job["epsilon"],
        add_uuid=1,
    )

    best_metrics = (result or {}).get("best_metrics") or {}
    metric_value = best_metrics.get(job["metric_key"])
    if metric_value is None:
        raise RuntimeError(
            f"Metric '{job['metric_key']}' not found in best_metrics for fold={job['fold']}. "
            f"Available keys: {list(best_metrics.keys())}"
        )

    _runtime_log(
        "fold-end",
        worker_id=worker_id,
        pid=os.getpid(),
        gpu=bound_gpu,
        trial=trial_number,
        fold=job["fold"],
        metric_key=job["metric_key"],
        metric=f"{float(metric_value):.6f}",
    )

    return {
        "fold": job["fold"],
        "metric": float(metric_value),
        "valid_auc": best_metrics.get("valid_auc"),
        "run_name": result.get("run_name"),
        "ckpt_dir": result.get("ckpt_dir"),
    }


def objective(trial: optuna.Trial) -> float:
    # Sample hyperparameters using configuration table
    param_exps = {hp["exp_key"]: trial.suggest_int(hp["exp_key"], *hp["range"]) 
                  for hp in HYPERPARAMS}
    
    # Convert exponents to actual values
    param_values = {hp["name"]: hp["base"] ** param_exps[hp["exp_key"]] 
                    for hp in HYPERPARAMS}
    
    worker_id = os.environ.get("OPTUNA_WORKER_ID", "unknown")
    worker_pid = os.getpid()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    _runtime_log(
        "trial-start",
        worker_id=worker_id,
        pid=worker_pid,
        gpu=cuda_visible_devices,
        trial=trial.number,
        **{hp["name"]: f"{param_values[hp['name']]:.6g}" for hp in HYPERPARAMS}
    )

    save_root = os.path.join(SAVE_DIR, f"{MODEL_NAME}_optuna_shared")
    os.makedirs(save_root, exist_ok=True)

    # Create trial-specific directory based on hyperparameter values
    param_folder = (
        f"C1-{param_values['C1']:.4g}_C2-{param_values['C2']:.4g}_C4-{param_values['C4']:.4g}_eps-{param_values['epsilon']:.4g}"
    )
    trial_save_dir = os.path.join(save_root, param_folder)
    os.makedirs(trial_save_dir, exist_ok=True)

    fold_jobs = []
    for fold in FOLDS:
        fold_jobs.append(
            {
                "dataset_name": DATASET_NAME,
                "model_name": MODEL_NAME,
                "fold": fold,
                "num_epochs": NUM_EPOCHS,
                "use_wandb": USE_WANDB,
                "save_dir": trial_save_dir,
                "seed": SEED + trial.number * 100 + fold,
                "metric_key": OBJECTIVE_METRIC,
                "cuda_visible_devices": cuda_visible_devices,
                "worker_id": worker_id,
                "trial_number": trial.number,
                **param_values,  # Unpack hyperparameter values
            }
        )

    if FOLD_PARALLEL:
        ctx = mp.get_context("spawn")
        fold_results = []
        max_workers = min(FOLD_PARALLEL_WORKERS, len(FOLDS))
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_run_single_fold, job) for job in fold_jobs]
            for future in as_completed(futures):
                fold_results.append(future.result())
    else:
        fold_results = [_run_single_fold(job) for job in fold_jobs]

    fold_results.sort(key=lambda x: x["fold"])
    fold_metrics = [r["metric"] for r in fold_results]
    fold_run_names = [r["run_name"] for r in fold_results]
    fold_valid_auc = [r["valid_auc"] for r in fold_results if r["valid_auc"] is not None]

    # Use metric_value (from OBJECTIVE_METRIC) for Optuna optimization
    metric_value = sum(fold_metrics) / len(fold_metrics) if fold_metrics else 0.0
    valid_auc_mean = sum(fold_valid_auc) / len(fold_valid_auc) if fold_valid_auc else None
    
    # Set trial attributes using configuration table
    trial.set_user_attr("worker_pid", os.getpid())
    trial.set_user_attr("gpu_visible", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    trial.set_user_attr("run_names", fold_run_names)
    trial.set_user_attr("fold_metrics", fold_metrics)
    trial.set_user_attr("valid_auc_fold_metrics", fold_valid_auc)
    trial.set_user_attr("valid_auc_mean", valid_auc_mean)
    trial.set_user_attr("metric_key", OBJECTIVE_METRIC)
    
    # Set hyperparameter attributes
    for hp in HYPERPARAMS:
        trial.set_user_attr(f"{hp['name']}_exp", param_exps[hp["exp_key"]])
        trial.set_user_attr(f"{hp['name']}_value", param_values[hp["name"]])

    _runtime_log(
        "trial-end",
        worker_id=worker_id,
        pid=worker_pid,
        gpu=cuda_visible_devices,
        trial=trial.number,
        objective=f"{metric_value:.6f}",
        valid_auc_mean="N/A" if valid_auc_mean is None else f"{valid_auc_mean:.6f}",
    )

    return float(metric_value)


def _validate_existing_study_metric() -> None:
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except Exception:
        return

    mismatched = []
    for t in study.trials:
        metric_key = t.user_attrs.get("metric_key")
        if metric_key is not None and metric_key != OBJECTIVE_METRIC:
            mismatched.append((t.number, metric_key))

    if mismatched:
        sample = ", ".join([f"trial#{n}:{m}" for n, m in mismatched[:5]])
        raise RuntimeError(
            "Existing study contains trials optimized with a different metric. "
            f"Expected '{OBJECTIVE_METRIC}', found mismatches: {sample}. "
            "Please use a new STUDY_NAME or STORAGE for valid-only optimization."
        )


def _worker_main(worker_id: int, gpu_id: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OPTUNA_WORKER_ID"] = str(worker_id)
    _install_signal_handlers()

    _runtime_log("worker-start", worker_id=worker_id, pid=os.getpid(), gpu=gpu_id)

    # Import after CUDA env setup so this worker only sees one assigned GPU.
    _get_launch_train()

    sampler = optuna.samplers.TPESampler(seed=SEED + worker_id * 1000)
    study = optuna.create_study(
        direction=DIRECTION,
        sampler=sampler,
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
    )
    try:
        study.optimize(
            objective,
            n_trials=TRIALS_PER_WORKER,
            n_jobs=1,
            catch=(Exception,),
        )
    except BaseException as e:
        _runtime_log(
            "worker-error",
            worker_id=worker_id,
            pid=os.getpid(),
            gpu=gpu_id,
            error_type=type(e).__name__,
            error=str(e),
        )
        raise

    _runtime_log("worker-end", worker_id=worker_id, pid=os.getpid(), gpu=gpu_id)


def main() -> None:
    _install_signal_handlers()
    _ensure_objective_metric()

    if not GPU_IDS:
        raise ValueError("GPU_IDS cannot be empty.")
    if WORKER_PROCESSES <= 0:
        raise ValueError("WORKER_PROCESSES must be > 0.")
    if TRIALS_PER_WORKER <= 0:
        raise ValueError("TRIALS_PER_WORKER must be > 0.")

    _validate_existing_study_metric()

    total_trials = WORKER_PROCESSES * TRIALS_PER_WORKER

    print("=" * 80)
    print("Shared-study parallel plan:")
    print(f"- one shared study: {STUDY_NAME}")
    print(f"- one shared storage: {STORAGE}")
    print(f"- objective metric: {OBJECTIVE_METRIC}")
    
    # Print hyperparameter ranges using configuration table
    for hp in HYPERPARAMS:
        base_name = "2" if hp["base"] == 2.0 else "10"
        print(f"- {hp['name']} range: {base_name}^{hp['range'][0]} .. {base_name}^{hp['range'][1]}")
    
    print(f"- workers: {WORKER_PROCESSES} (configurable), trials per worker: {TRIALS_PER_WORKER}")
    print(f"- expected total trials: {total_trials}")
    print(f"- gpu_ids: {GPU_IDS}")
    print(f"- fold parallel: {FOLD_PARALLEL}, fold workers per trial: {FOLD_PARALLEL_WORKERS}")
    print("=" * 80)

    ctx = mp.get_context("spawn")
    workers = []
    for worker_id in range(WORKER_PROCESSES):
        gpu_id = GPU_IDS[worker_id % len(GPU_IDS)]
        proc = ctx.Process(target=_worker_main, args=(worker_id, gpu_id), daemon=False)
        proc.start()
        workers.append(proc)

    failed = []
    try:
        for proc in workers:
            proc.join()
            if proc.exitcode != 0:
                failed.append((proc.pid, proc.exitcode))
    except KeyboardInterrupt as e:
        _runtime_log("main-interrupt", pid=os.getpid(), reason=str(e))
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join()
        raise

    if failed:
        print("ERROR: some workers exited abnormally:")
        for pid, code in failed:
            print(f"- pid={pid}, exit_code={code}")
        raise RuntimeError(
            f"Failed workers: {len(failed)}. Search incomplete. "
            "Check logs and resolve issues before retrying."
        )

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    best_trial = study.best_trial
    
    if best_trial is None:
        print("ERROR: No completed trials found in study.")
        raise RuntimeError("No completed trials. Optimization failed.")
    
    # Extract best parameters with None safety using configuration table
    best_param_exps = {}
    for hp in HYPERPARAMS:
        exp_value = best_trial.params.get(hp["exp_key"])
        if exp_value is None:
            raise RuntimeError(
                f"Missing parameter '{hp['exp_key']}' in best trial #{best_trial.number}. "
                f"Available params: {list(best_trial.params.keys())}"
            )
        best_param_exps[hp["exp_key"]] = exp_value
    
    # Convert exponents to values
    best_param_values = {hp["name"]: hp["base"] ** best_param_exps[hp["exp_key"]] 
                        for hp in HYPERPARAMS}
    
    best_valid_auc = best_trial.user_attrs.get("valid_auc_mean")

    print("=" * 80)
    if best_valid_auc is None:
        print("Best valid_auc: N/A")
    else:
        print(f"Best valid_auc: {float(best_valid_auc):.6f}")
    
    # Print best hyperparameters using configuration table
    for hp in HYPERPARAMS:
        exp_val = best_param_exps[hp["exp_key"]]
        actual_val = best_param_values[hp["name"]]
        print(f"Best {hp['name']} exponent: {exp_val}, value: {actual_val:.6g}")
    
    print(f"Best trial number: {best_trial.number}")
    print(f"Stored study: {STUDY_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    main()
