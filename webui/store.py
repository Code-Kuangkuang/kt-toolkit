import json
import sqlite3
import threading
from pathlib import Path


class JobStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    pid INTEGER,
                    return_code INTEGER,
                    dataset_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    fold INTEGER,
                    cv INTEGER NOT NULL,
                    folds TEXT,
                    gpu INTEGER NOT NULL,
                    save_dir TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        data = dict(row)
        for key in ("command_json", "request_json"):
            try:
                data[key[:-5]] = json.loads(data[key])
            except Exception:
                data[key[:-5]] = None
        return data

    def create_job(self, job):
        payload = dict(job)
        payload["command_json"] = json.dumps(payload.pop("command"), ensure_ascii=True)
        payload["request_json"] = json.dumps(payload.pop("request"), ensure_ascii=True)
        columns = ", ".join(payload.keys())
        placeholders = ", ".join(["?"] * len(payload))
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
                list(payload.values()),
            )
        return self.get_job(job["id"])

    def update_job(self, job_id, **updates):
        if not updates:
            return self.get_job(job_id)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [job_id]
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
        return self.get_job(job_id)

    def get_job(self, job_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._decode(row)

    def list_jobs(self, limit=100):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._decode(row) for row in rows]

