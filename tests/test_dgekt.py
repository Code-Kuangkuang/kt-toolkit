import pandas as pd
import torch

from core.trainers.dgekt_trainer import DGEKTTrainer
from models.dgekt import DGEKT
from models.dgekt_utils import build_dgekt_graphs


def test_dgekt_forward_backward_and_fold_safe_transition_graph(tmp_path):
    data = pd.DataFrame(
        [
            {
                "fold": 0,
                "questions": "0,1,2",
                "concepts": "0,0,1",
                "responses": "1,0,1",
            },
            {
                "fold": 1,
                "questions": "2,3,0",
                "concepts": "1,1,0",
                "responses": "0,1,1",
            },
        ]
    )
    path = tmp_path / "sequences.csv"
    data.to_csv(path, index=False)
    hypergraph, transition_out, transition_in, stats = build_dgekt_graphs(
        tmp_path, path.name, num_q=4, num_c=2, train_folds=[0]
    )
    assert stats["train_folds"] == [0]
    assert stats["transition_count"] == 2
    assert stats["covered_questions"] == 4
    assert hypergraph.shape == (8, 4)

    model = DGEKT(
        num_q=4,
        num_c=2,
        emb_size=8,
        hidden_dim=6,
        hypergraph=hypergraph,
        transition_out=transition_out,
        transition_in=transition_in,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DGEKTTrainer(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=optimizer,
        num_epochs=1,
        device="cpu",
        other_config={"kd_lambda": 5e-6, "kd_temperature": 0.5},
    )
    batch = {
        "qseqs": torch.tensor([[0, 1], [2, 3]]),
        "shft_qseqs": torch.tensor([[1, 2], [3, 0]]),
        "rseqs": torch.tensor([[1, 0], [0, 1]]),
        "shft_rseqs": torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
        "smasks": torch.tensor([[True, True], [True, False]]),
    }
    pred, target, loss = trainer._forward_batch(batch)
    assert pred.shape == target.shape == (3,)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.interaction_emb.weight.grad is not None
    assert torch.isfinite(model.interaction_emb.weight.grad).all()
