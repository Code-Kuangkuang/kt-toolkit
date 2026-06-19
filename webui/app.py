from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webui.model_structure import build_model_structure
from webui.runner import JobRunner, ROOT


STATIC_DIR = Path(__file__).resolve().parent / "static"
runner = JobRunner()
app = FastAPI(title="KT-Toolkit WebUI", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CreateJobRequest(BaseModel):
    dataset_name: str
    model_name: str
    emb_type: Optional[str] = None
    fold: int = 0
    cv: bool = False
    folds: str = "0-4"
    cv_run_dir: Optional[str] = None
    skip_completed: int = 0
    batch_size: Optional[int] = Field(default=None, gt=0)
    num_epochs: Optional[int] = Field(default=None, gt=0)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    emb_size: Optional[int] = Field(default=None, gt=0)
    dropout: Optional[float] = Field(default=None, ge=0)
    gpu: int = 0
    seed: int = 3407
    use_wandb: int = 0
    save_dir: Optional[str] = None
    model_params: Dict[str, Any] = Field(default_factory=dict, alias="model_config")


class ModelStructureRequest(BaseModel):
    dataset_name: str
    model_name: str
    emb_type: Optional[str] = None
    model_params: Dict[str, Any] = Field(default_factory=dict, alias="model_config")


def _payload(model):
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=True)
    return model.dict(by_alias=True)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/configs")
def get_configs():
    return runner.load_runtime_config()


@app.get("/api/jobs")
def list_jobs(limit: int = Query(default=100, ge=1, le=500)):
    return {"jobs": runner.list_jobs(limit=limit)}


@app.post("/api/jobs")
def create_job(request: CreateJobRequest):
    try:
        job = runner.create_job(_payload(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = runner.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = runner.stop_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        job = runner.delete_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str, lines: int = Query(default=300, ge=1, le=5000)):
    text = runner.tail_log(job_id, lines=lines)
    if text is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return text


@app.get("/api/jobs/{job_id}/metrics")
def get_metrics(job_id: str):
    metrics = runner.collect_metrics(job_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return metrics


@app.post("/api/model-structure")
def model_structure(request: ModelStructureRequest):
    try:
        return build_model_structure(ROOT, _payload(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not inspect model: {exc}") from exc


@app.get("/api/health")
def health():
    return {"ok": True, "root": str(ROOT)}
