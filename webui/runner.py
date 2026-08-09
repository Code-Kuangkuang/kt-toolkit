import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from core.config import load_cfg
from webui.store import JobStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / ".webui" / "jobs.sqlite3"
DEFAULT_SAVE_BASE = ROOT / ".webui" / "runs"


class JobRunner:
    def __init__(self, root=ROOT, db_path=None):
        self.root = Path(root)
        self.store = JobStore(db_path or os.environ.get("KT_WEBUI_DB") or DEFAULT_DB)
        self._processes = {}
        self._lock = threading.Lock()

    @staticmethod
    def now():
        return datetime.now().isoformat(timespec="seconds")

    def load_runtime_config(self):
        kt_cfg = load_cfg(str(self.root / "configs" / "kt_config.json"))
        data_cfg = load_cfg(str(self.root / "configs" / "data_config.json"))
        models = {k: v for k, v in kt_cfg.items() if k != "train_config"}
        return {
            "train_config": kt_cfg.get("train_config", {}),
            "models": models,
            "datasets": data_cfg,
            "model_names": sorted(models.keys()),
            "dataset_names": sorted(data_cfg.keys()),
            "override_fields": [
                "batch_size",
                "num_epochs",
                "learning_rate",
                "emb_size",
                "dropout",
                "emb_type",
                "fold",
                "cv",
                "folds",
                "gpu",
                "seed",
                "use_wandb",
            ],
        }

    def create_job(self, request):
        cfg = self.load_runtime_config()
        dataset_name = request["dataset_name"]
        model_name = request["model_name"]
        if dataset_name not in cfg["datasets"]:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if model_name not in cfg["models"]:
            raise ValueError(f"Unknown model: {model_name}")

        job_id = uuid.uuid4().hex[:12]
        save_base = Path(request.get("save_dir") or DEFAULT_SAVE_BASE)
        if not save_base.is_absolute():
            save_base = self.root / save_base
        run_dir = save_base / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "webui.log"

        kt_config_path = self._write_job_kt_config(request, run_dir)
        command = self._build_command(request, run_dir)
        command.extend(["--kt-config", str(kt_config_path)])
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": self.now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "return_code": None,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "fold": int(request.get("fold", 0)),
            "cv": 1 if request.get("cv") else 0,
            "folds": request.get("folds") or "0-4",
            "gpu": int(request.get("gpu", 0)),
            "save_dir": str(run_dir),
            "log_path": str(log_path),
            "command": command,
            "request": request,
            "error_message": None,
        }
        self.store.create_job(job)
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return self.store.get_job(job_id)

    def _write_job_kt_config(self, request, run_dir):
        kt_cfg = load_cfg(str(self.root / "configs" / "kt_config.json"))
        model_name = request["model_name"]
        model_cfg = kt_cfg.setdefault(model_name, {})
        train_cfg = kt_cfg.setdefault("train_config", {})

        for key in ("batch_size", "num_epochs"):
            value = request.get(key)
            if value is not None and value != "":
                train_cfg[key] = value

        model_overrides = request.get("model_config") or {}
        for key, value in model_overrides.items():
            if value is not None and value != "":
                model_cfg[key] = value

        config_path = Path(run_dir) / "kt_config.webui.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(kt_cfg, f, indent=2, ensure_ascii=True)
        return config_path

    def _build_command(self, request, run_dir):
        command = [
            sys.executable,
            "-u",
            str(self.root / "scripts" / "train.py"),
            "--dataset-name",
            request["dataset_name"],
            "--model-name",
            request["model_name"],
            "--fold",
            str(int(request.get("fold", 0))),
            "--gpu",
            str(int(request.get("gpu", 0))),
            "--seed",
            str(int(request.get("seed", 3407))),
            "--use-wandb",
            str(int(request.get("use_wandb", 0))),
            "--save-dir",
            str(run_dir),
        ]
        if request.get("cv"):
            command.extend(["--cv", "1", "--folds", str(request.get("folds") or "0-4")])
            if request.get("cv_run_dir"):
                command.extend(["--cv-run-dir", str(request["cv_run_dir"])])
            if request.get("skip_completed"):
                command.extend(["--skip-completed", "1"])
        return command

    def _run_job(self, job_id):
        job = self.store.get_job(job_id)
        if job is None:
            return
        if job["status"] != "queued":
            return
        log_path = Path(job["log_path"])
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"[webui] starting job {job_id}\n")
                log_file.write("[webui] command: " + " ".join(job["command"]) + "\n")
                log_file.flush()
                creationflags = 0
                if os.name == "nt":
                    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                proc = subprocess.Popen(
                    job["command"],
                    cwd=str(self.root),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=creationflags,
                )
                with self._lock:
                    self._processes[job_id] = proc
                self.store.update_job(
                    job_id,
                    status="running",
                    started_at=self.now(),
                    pid=proc.pid,
                )
                return_code = proc.wait()
                status = "finished" if return_code == 0 else "failed"
                self.store.update_job(
                    job_id,
                    status=status,
                    finished_at=self.now(),
                    return_code=return_code,
                )
        except Exception as exc:
            self.store.update_job(
                job_id,
                status="failed",
                finished_at=self.now(),
                error_message=str(exc),
            )
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def stop_job(self, job_id):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if job["status"] not in {"queued", "running"}:
            return job
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        elif job.get("pid"):
            self._terminate_pid(job["pid"])
        return self.store.update_job(
            job_id,
            status="stopped",
            finished_at=self.now(),
            error_message="Stopped by user",
        )

    def delete_job(self, job_id):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        if job["status"] in {"queued", "running"}:
            job = self.stop_job(job_id) or job

        artifact_deleted = self._delete_job_artifacts(job)
        self.store.delete_job(job_id)
        job["deleted"] = True
        job["artifact_deleted"] = artifact_deleted
        return job

    def _delete_job_artifacts(self, job):
        path = self._job_artifact_root(job)
        if not path.exists():
            return False

        root = self.root.resolve()
        target = path.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Refusing to delete outside project root: {target}") from exc

        if target == root or not (target.name == job["id"] or target.name.startswith("cv-")):
            raise ValueError(f"Refusing to delete non-job directory: {target}")
        if not target.is_dir():
            raise ValueError(f"Refusing to delete non-directory artifact path: {target}")

        shutil.rmtree(target)
        return True

    @staticmethod
    def _terminate_pid(pid):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.kill(int(pid), 15)
        except Exception:
            pass

    def list_jobs(self, limit=100):
        return self.store.list_jobs(limit=limit)

    def get_job(self, job_id):
        return self.store.get_job(job_id)

    def _job_artifact_root(self, job):
        request = job.get("request") or {}
        cv_run_dir = request.get("cv_run_dir")
        if cv_run_dir:
            path = Path(cv_run_dir)
            if not path.is_absolute():
                path = self.root / path
            return path
        return Path(job["save_dir"])

    def tail_log(self, job_id, lines=300):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        path = Path(job["log_path"])
        if not path.exists():
            return ""
        items = deque(maxlen=max(1, min(int(lines), 5000)))
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                items.append(line)
        return self._collapse_carriage_returns("".join(items))

    @staticmethod
    def _collapse_carriage_returns(text):
        lines = []
        current = []
        for ch in text:
            if ch == "\r":
                current = []
            elif ch == "\n":
                lines.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            lines.append("".join(current))
        return "\n".join(lines)

    def collect_metrics(self, job_id):
        job = self.store.get_job(job_id)
        if job is None:
            return None
        save_dir = self._job_artifact_root(job)
        runs = []
        for metrics_path in sorted(save_dir.rglob("metrics.jsonl")):
            metrics = []
            with metrics_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            run_dir = metrics_path.parent
            best_metrics = None
            best_path = run_dir / "best_metrics.json"
            if best_path.exists():
                with best_path.open("r", encoding="utf-8", errors="replace") as f:
                    best_metrics = json.load(f)
            runs.append(
                {
                    "name": str(run_dir.relative_to(save_dir)),
                    "path": str(run_dir),
                    "metrics": metrics,
                    "best_metrics": best_metrics,
                }
            )
        cv_summary = save_dir / "cv_summary.json"
        if not cv_summary.exists():
            matches = sorted(save_dir.rglob("cv_summary.json"))
            cv_summary = matches[0] if matches else cv_summary
        summary = None
        if cv_summary.exists():
            with cv_summary.open("r", encoding="utf-8", errors="replace") as f:
                summary = json.load(f)
        return {"job_id": job_id, "runs": runs, "cv_summary": summary}
