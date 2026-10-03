"""Behavioral regressions for experiment identity, data protocols and scheduling."""
import json
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

import models  # noqa: F401
from core.experiment_identity import build_experiment_identity, validate_fold_results
from datasets.kt_dataset import KTDataset, KTQueDataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    monkeypatch.setenv("KT_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    rows = [dict(fold=f, uid=f * 10 + i, questions="0,1,2,3,0,1",
                 concepts="0_1,1_2,2,0,0_1,1", responses="0,1,0,1,1,0",
                 selectmasks="1,1,1,1,1,1") for f in range(5) for i in range(2)]
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    test = pd.DataFrame(rows[:2]).assign(fold=-1)
    test.to_csv(tmp_path / "test.csv", index=False)
    config = dict(dpath=str(tmp_path), num_c=3, num_q=4, max_concepts=2, folds=list(range(5)),
                  input_type=["questions", "concepts"], train_valid_file="train.csv",
                  train_valid_file_quelevel="train.csv", test_file="test.csv",
                  test_file_quelevel="test.csv")
    kt = dict(train_config=dict(batch_size=2, num_epochs=2, seq_len=6,
                               dataset_mode="all_in_one", eval_window=False),
              dkt=dict(learning_rate=.001, emb_size=8, dropout=0),
              akt=dict(learning_rate=.001, d_model=8, d_ff=16, n_blocks=1,
                       num_attn_heads=2, dropout=0))
    return tmp_path, kt, {"tiny": config}


def identity(data, **kwargs):
    directory, kt, datasets = data
    return build_experiment_identity(root_dir=ROOT, dataset_name="tiny", model_name="dkt",
                                     kt_cfg_raw=kt, data_config_raw=datasets, seed=3407,
                                     fold_id=kwargs.pop("fold_id", 0), **kwargs)


def test_identity_distinguishes_configuration_data_and_folds(tiny_data):
    original = identity(tiny_data)
    other_fold = identity(tiny_data, fold_id=1)
    assert original["comparison_key"] == other_fold["comparison_key"]
    assert original["key"] != other_fold["key"]
    assert original["key"] != identity(tiny_data, overrides={"learning_rate": .02})["key"]
    assert original["key"] != identity(tiny_data, emb_type="qid_norasch")["key"]
    with (tiny_data[0] / "train.csv").open("a") as stream:
        stream.write("\n")
    assert original["key"] != identity(tiny_data)["key"]


@pytest.mark.parametrize("field,value", [("protocol", {"concept_mode": "first"}),
                                          ("seed", 1), ("comparison_key", "different"), ("fold", 0)])
def test_summary_rejects_mixed_or_duplicate_folds(field, value):
    ref = dict(fold=0, dataset_name="tiny", model_name="dkt", seed=3407,
               emb_type="qid", comparison_key="same", protocol={"concept_mode": "multi"})
    other = dict(ref, fold=1)
    other[field] = value
    with pytest.raises(ValueError):
        validate_fold_results([ref, other])


@pytest.mark.parametrize("dataset_cls", [KTDataset, KTQueDataset])
def test_cache_respects_input_fields(tiny_data, dataset_cls):
    # KTDataset uses single-KC CSVs; both loaders must separate input schemas.
    path = tiny_data[0] / "train.csv"
    if dataset_cls is KTDataset:
        frame = pd.read_csv(path)
        frame["concepts"] = "0,1,2,0,0,1"
        frame.to_csv(path, index=False)
    kwargs = dict(concept_num=3, max_concepts=2) if dataset_cls is KTQueDataset else {}
    first = dataset_cls(path, ["concepts"], {0}, **kwargs)
    second = dataset_cls(path, ["questions", "concepts"], {0}, **kwargs)
    assert "qseqs" not in first[0]
    assert second[0]["qseqs"].numel() > 0


def test_prediction_keeps_multi_concept_input(monkeypatch):
    import scripts.predict_aaai2023 as prediction
    captured = []
    monkeypatch.setattr(prediction, "_akt_predict_row",
                        lambda model, q, c, r, use_pred, device: captured.append(c) or [.5])
    frame = pd.DataFrame([dict(questions="1,2", concepts="0_1,2", responses="1,-1")])
    prediction._predict_all_rows(None, frame, "akt", False, torch.device("cpu"), "test", "multi", 2)
    assert captured == [[[0, 1], [2, -1]]]


def test_independent_prediction_excludes_unobserved_history():
    from scripts.predict_aaai2023 import _akt_predict_row
    calls = []
    class CaptureModel:
        n_pid = 4
        def __call__(self, query, target, pid):
            calls.append((query.tolist(), target.tolist()))
            return torch.full(target.shape, .5), torch.tensor(0.)
    values = _akt_predict_row(CaptureModel(), [0, 1, 2], [[0, -1], [1, -1], [2, -1]],
                              [1, -1, -1], False, torch.device("cpu"))
    assert values == [.5, .5]
    assert calls[1] == ([[[0, -1], [2, -1]]], [[1, 0]])


def test_webui_applies_top_level_parameters(tmp_path):
    from webui.runner import JobRunner
    runner = JobRunner(db_path=tmp_path / "jobs.db")
    request = dict(model_name="dkt", learning_rate=.0123, emb_size=32, dropout=.234, emb_type="qid")
    config = json.loads(runner._write_job_kt_config(request, tmp_path).read_text())
    assert all(config["dkt"][k] == v for k, v in request.items() if k != "model_name")


def test_webui_cancel_during_spawn_is_preserved(tmp_path, monkeypatch):
    from webui.runner import JobRunner
    runner = JobRunner(db_path=tmp_path / "jobs.db")
    job = dict(id="fixture", status="queued", created_at=runner.now(), dataset_name="tiny",
               model_name="dkt", cv=0, gpu=0, save_dir=str(tmp_path), log_path=str(tmp_path / "run.log"),
               command=["mock"], request={})
    runner.store.create_job(job)
    process = Mock(pid=123)
    process.wait.return_value = 0
    def spawn(*args, **kwargs):
        runner.stop_job("fixture")
        return process
    monkeypatch.setattr("webui.runner.subprocess.Popen", spawn)
    runner._run_job("fixture")
    assert runner.get_job("fixture")["status"] == "stopped"
    process.terminate.assert_called_once()


def test_same_gpu_jobs_do_not_overlap(tmp_path, monkeypatch):
    from webui.runner import JobRunner
    runner = JobRunner(db_path=tmp_path / "jobs.db")
    for job_id in ("a", "b"):
        runner.store.create_job(dict(id=job_id, status="queued", created_at=runner.now(),
                                    dataset_name="tiny", model_name="dkt", cv=0, gpu=0,
                                    save_dir=str(tmp_path), log_path=str(tmp_path / "run.log"),
                                    command=["mock"], request={}))
    started = threading.Event()
    release = threading.Event()
    active = []
    def execute(job_id):
        active.append(job_id)
        started.set()
        assert release.wait(3)
    monkeypatch.setattr(runner, "_execute_job", execute)
    first = threading.Thread(target=runner._run_job, args=("a",))
    second = threading.Thread(target=runner._run_job, args=("b",))
    first.start()
    assert started.wait(3)
    second.start()
    runner.stop_job("b")
    release.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert active == ["a"]


def test_webui_rejects_cv_path_before_launch(tmp_path, monkeypatch):
    from webui.runner import JobRunner
    runner = JobRunner(db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(runner, "load_runtime_config", lambda: {"datasets": {"tiny": {}}, "models": {"dkt": {}}})
    with pytest.raises(ValueError, match="outside"):
        runner.create_job(dict(dataset_name="tiny", model_name="dkt", cv=True, cv_run_dir=str(tmp_path)))
    assert runner.list_jobs() == []


def test_epoch_recovery_matches_uninterrupted_training(tmp_path):
    from core.checkpoint import load_recovery
    from core.trainer import BaseTrainer
    from core.run_support import set_seed
    class ToyTrainer(BaseTrainer):
        def __init__(self):
            super().__init__(num_epochs=2)
            self.model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(.2))
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, .9)
            self.device = "cpu"
            self.train_loader = [(torch.ones(3, 2), torch.zeros(3))]
            self.valid_loader = self.train_loader
            self.experiment_key = "same"
        def _forward_batch(self, batch):
            pred = self.model(batch[0]).sigmoid().flatten()
            return pred, batch[1], torch.nn.functional.binary_cross_entropy(pred, batch[1])
        def _train_epoch(self, epoch):
            result = super()._train_epoch(epoch)
            self.scheduler.step()
            return result
    set_seed(3407)
    reference = ToyTrainer()
    reference.run()
    set_seed(3407)
    interrupted = ToyTrainer()
    interrupted.num_epochs = 1
    interrupted.recovery_path = str(tmp_path / "checkpoint.pt")
    interrupted.run()
    resumed = ToyTrainer()
    resumed.recovery_path = str(tmp_path / "resumed.pt")
    load_recovery(resumed, tmp_path / "checkpoint.pt", "same")
    resumed.run()
    for left, right in zip(reference.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert reference.scheduler.state_dict() == resumed.scheduler.state_dict()


@pytest.mark.parametrize("model_name", ["dkt", "akt"])
def test_train_runner_produces_complete_artifacts(tiny_data, model_name):
    import core.trainers  # noqa: F401
    import datasets.init_dataset  # noqa: F401
    from core.train_runner import train_one_fold
    directory, kt, datasets = tiny_data
    result = train_one_fold(dataset_name="tiny", model_name=model_name, emb_type="qid", fold_id=0,
                            root_dir=str(ROOT), kt_cfg_raw=kt, data_config_raw=datasets,
                            seed=3407, save_root=str(directory / "runs"), add_uuid=1, wandb_cfg=None,
                            kt_config_path="fixture", data_config_path="fixture", wandb_config_path="fixture",
                            cv_run_name=None, overrides={"num_epochs": 1}, gpu_id=-1)
    run = Path(result["ckpt_dir"])
    for name in ("run_config.json", "metrics.jsonl", "best_metrics.json", "last_epoch_model.pt", "training_checkpoint.pt"):
        assert (run / name).is_file()
    assert Path(result["best_path"]).is_file()
    assert result["best_metrics"]["best_test_auc"] is not None


def test_blocked_pebg_loss_keeps_dense_gradients():
    from scipy import sparse
    from modules.pebg_torch import backward_pair_loss
    torch.manual_seed(1)
    dense = torch.nn.Embedding(7, 4)
    blocked = torch.nn.Embedding(7, 4)
    blocked.load_state_dict(dense.state_dict())
    indices = torch.tensor([0, 2, 4])
    targets = torch.randint(0, 2, (7, 7)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        dense(indices) @ dense.weight.T, targets[indices])
    loss.backward()
    value = backward_pair_loss(blocked, indices, sparse.csr_matrix(targets.numpy()), 2)
    assert value == pytest.approx(loss.item(), rel=1e-6)
    torch.testing.assert_close(dense.weight.grad, blocked.weight.grad)


def test_streamed_preprocessing_keeps_window_protocol(tmp_path):
    from preprocess.split_datasets import generate_window_sequences, iter_window_sequence_rows, read_data
    frame = pd.DataFrame([dict(uid=1, fold=-1, concepts="0,1,0,2,1", responses="0,1,1,0,1")])
    keys = ["uid", "fold", "concepts", "responses"]
    old = generate_window_sequences(frame, keys, maxlen=3)
    streamed = pd.DataFrame(iter_window_sequence_rows(frame, keys, maxlen=3))
    pd.testing.assert_frame_equal(old.sort_index(axis=1), streamed.sort_index(axis=1), check_dtype=False)
    path = tmp_path / "raw.txt"
    path.write_text("1,3\n0,1,2\n0,1,2\n0,1,0\nNA\nNA\n2,2\n0,1\n0,1\n1,0\nNA\nNA\n", encoding="utf-8")
    result, effective = read_data(path, min_seq_len=3)
    assert result["uid"].tolist() == ["1"]
    assert effective == {"uid", "questions", "concepts", "responses"}


def test_preview_uses_input_spec_and_preserves_rng(tiny_data, monkeypatch):
    from core.model_inputs import ModelInputs
    from core.registry import MODEL_REGISTRY
    from webui.model_structure import build_model_structure
    from core.checkpoint import rng_state
    root = tiny_data[0]
    config_dir = root / "configs"
    config_dir.mkdir()
    _, kt, datasets = tiny_data
    (config_dir / "kt_config.json").write_text(json.dumps(kt))
    (config_dir / "data_config.json").write_text(json.dumps(datasets))
    spec = MODEL_REGISTRY.get("dkt").Inputs
    prepared = []
    def prepare(cls, ctx):
        prepared.append(ctx.fold_id)
        torch.rand(5)
        np.random.rand(5)
        return ModelInputs()
    monkeypatch.setattr(spec, "prepare", classmethod(prepare))
    before = rng_state()
    available = torch.cuda.is_available
    result = build_model_structure(root, dict(dataset_name="tiny", model_name="dkt", fold=2))
    assert result["total_params"] > 0
    assert prepared == [2]
    torch.testing.assert_close(before["torch"], torch.get_rng_state())
    assert before["numpy"][1] == np.random.get_state()[1].tolist()
    assert torch.cuda.is_available is available


def test_webui_metrics_handles_partial_append(tmp_path):
    from webui.runner import JobRunner
    runner = JobRunner(db_path=tmp_path / "jobs.db")
    runner.store.create_job(dict(id="metrics", status="running", created_at=runner.now(),
                                dataset_name="tiny", model_name="dkt", cv=0, gpu=0,
                                save_dir=str(tmp_path), log_path=str(tmp_path / "run.log"),
                                command=["mock"], request={}))
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n{"epoch":', encoding="utf-8")
    assert runner.collect_metrics("metrics")["runs"][0]["metrics"] == [{"epoch": 1}]
    with path.open("a") as stream:
        stream.write(' 2}\n')
    assert runner.collect_metrics("metrics")["runs"][0]["metrics"] == [{"epoch": 1}, {"epoch": 2}]


def test_cleaning_configs_have_registered_adapters():
    import yaml
    from cleaning.adapters import ADAPTERS
    for path in (ROOT / "configs" / "dataset").glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["dataset_name"] in ADAPTERS, path


def test_cli_cv_resume_reuses_only_the_same_experiment(tiny_data, monkeypatch):
    from typer.testing import CliRunner
    from scripts.train import app
    directory, kt, datasets = tiny_data
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kt_path, data_path = directory / "kt.json", directory / "data.json"
    kt_path.write_text(json.dumps(kt))
    data_path.write_text(json.dumps(datasets))
    args = ["--dataset-name", "tiny", "--model-name", "dkt", "--cv", "1", "--folds", "0,1",
            "--num-epochs", "1", "--gpu", "-1", "--use-wandb", "0", "--kt-config", str(kt_path),
            "--data-config", str(data_path), "--save-dir", str(directory / "cv")]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output + str(result.exception)
    cv_dir = next((directory / "cv").iterdir())
    summary = json.loads((cv_dir / "cv_summary.json").read_text())
    assert summary["aggregate"]["valid_auc"]["n"] == 2
    resumed_args = args + ["--cv-run-dir", str(cv_dir), "--skip-completed", "1"]
    result = runner.invoke(app, resumed_args)
    assert result.exit_code == 0, result.output + str(result.exception)
    assert result.output.count("completed run found") == 2
    result = runner.invoke(app, resumed_args + ["--learning-rate", ".02"])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert "completed run found" not in result.output
    assert len(list(cv_dir.glob("*/run_config.json"))) == 4
