"""Core model, dataset preparation and baseline failure regressions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.model_inputs import RunContext
from datasets.init_dataset import build_dataloaders, protocol_stamp
from models import cgmkt_graphs
from models.cgmkt import CGMKTModel
from models.fokt import FoKT
from models.akt import AKT


def test_empty_qmatrix_and_configured_source_names(tmp_path):
    from datasets.feature_utils import _question_groups_from_qmatrix
    from scripts.build_qmatrix import build

    np.savez(tmp_path / "qmatrix.npz", matrix=np.zeros((4, 3)))
    np.testing.assert_array_equal(_question_groups_from_qmatrix(str(tmp_path), 3), [-1] * 4)
    pd.DataFrame([{"questions": "2", "concepts": "0_1"}]).to_csv(
        tmp_path / "pykt_test_quelevel.csv", index=False)
    matrix, used = build(str(tmp_path), 3, 2, sources=[None, "pykt_test_quelevel.csv"])
    assert used == ["pykt_test_quelevel.csv"]
    np.testing.assert_array_equal(matrix[2], [1, 1, 0])


@pytest.mark.parametrize("train_status, summary_status, expected", [(7, 0, 1), (0, 0, 0), (0, 1, 1)])
def test_baseline_sweep_propagates_training_and_summary_failures(monkeypatch, train_status, summary_status, expected):
    import scripts.run_baseline_table as baseline

    monkeypatch.setattr("sys.argv", ["run_baseline_table.py", "--datasets", "fixture",
                                     "--models", "dkt", "--folds", "0"])
    monkeypatch.setattr(baseline, "find_result", lambda *args: None)
    monkeypatch.setattr(baseline, "train_one", lambda *args: (train_status, .1, "mock.log"))
    monkeypatch.setattr(baseline, "record_run", lambda *args: None)
    monkeypatch.setattr(baseline, "summarize", lambda *args, **kwargs: summary_status)
    assert baseline.main() == expected


def test_cgmkt_fit_ignores_validation_and_tracks_training_changes(tmp_path):
    path = tmp_path / "train_valid.csv"

    def graph(train, valid):
        pd.DataFrame([{"fold": 1, "concepts": train},
                      {"fold": 0, "concepts": valid}]).to_csv(path, index=False)
        return cgmkt_graphs.build_kc_transition(3, str(tmp_path), path.name, folds=[1])

    expected = graph("0,1", "0,2")
    np.testing.assert_array_equal(expected, graph("0,1", "0,0"))
    assert not np.array_equal(expected, graph("0,2", "0,0"))
    with pytest.raises(ValueError, match="training folds"):
        cgmkt_graphs.build_kc_transition(3, str(tmp_path), path.name, folds=[])


def test_cgmkt_prepare_passes_resolved_source_and_training_folds(tmp_path, monkeypatch):
    from models import kc_graph_utils

    (tmp_path / "keyid2idx.json").write_text('{"max_concepts": 1}')
    captured = {}

    def transition(num_c, dpath, source, *, folds):
        captured.update(source=source, folds=folds)
        return np.eye(num_c, dtype=np.float32)

    monkeypatch.setattr(cgmkt_graphs, "build_kc_transition", transition)
    monkeypatch.setattr(cgmkt_graphs, "incidence_question_graph", lambda *args: torch.eye(4))
    monkeypatch.setattr(kc_graph_utils, "load_kc_text_embeddings", lambda *args: np.ones((3, 4)))
    monkeypatch.setattr(kc_graph_utils, "build_question_concept_map", lambda *args: np.array([[0], [1], [2], [0]]))
    ctx = RunContext("cgmkt", "fixture", 0, "all_in_one",
                     {"kc_graph_source": "sbm_fit", "group_source": "sbm_fit", "num_clusters": 2},
                     {"dpath": str(tmp_path), "num_q": 4, "num_c": 3, "folds": [0, 1, 2]},
                     {}, str(tmp_path), lambda *args: "resolved.csv")
    inputs = CGMKTModel.Inputs.prepare(ctx)
    assert captured == {"source": "resolved.csv", "folds": [1, 2]}
    assert inputs.run_config_extras["graph_fit_folds"] == [1, 2]


def test_fokt_loader_keeps_all_concepts_and_qid_matches_akt(tmp_path):
    path = tmp_path / "sequences.csv"
    pd.DataFrame([
        {"fold": fold, "uid": fold, "questions": "0,1,2,-1", "concepts": "0_1,1_2,2_0,-1",
         "responses": "0,1,0,-1", "selectmasks": "1,1,1,-1"}
        for fold in [0, 1]
    ]).to_csv(path, index=False)
    cfg = {"dpath": str(tmp_path), "folds": [0, 1], "input_type": ["questions", "concepts"],
           "train_valid_file": path.name, "train_valid_file_quelevel": path.name,
           "num_q": 3, "num_c": 3, "max_concepts": 2}
    train, _ = build_dataloaders("fixture", {"fixture": cfg}, 0, 1, model_name="fokt")
    batch = next(iter(train))
    assert batch["cseqs"].ndim == 3 and batch["cseqs"].shape[-1] == 2
    assert batch["cseqs"][0, 0].tolist() == [0, 1]
    assert protocol_stamp("fokt", "all_in_one", 2) == protocol_stamp("akt", "all_in_one", 2)
    kwargs = dict(num_c=3, num_q=3, d_model=8, d_ff=16, n_blocks=1,
                  num_attn_heads=2, final_fc_dim=16, dropout=0, emb_type="qid")
    torch.manual_seed(3407)
    baseline = AKT(**kwargs).eval()
    torch.manual_seed(3407)
    model = FoKT(**kwargs).eval()
    full = {name: torch.cat((batch[name][:, :1], batch["shft_" + name]), dim=1).long()
            for name in ["qseqs", "cseqs", "rseqs"]}
    with torch.no_grad():
        expected = baseline(full["cseqs"], full["rseqs"], full["qseqs"])[0]
        actual = model(full["cseqs"], full["rseqs"], full["qseqs"])[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
