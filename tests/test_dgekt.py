import pandas as pd
import torch

from core.trainers.dgekt_trainer import DGEKTTrainer
from models.dgekt import DGEKT
from models.dgekt_utils import build_dgekt_graphs


def _build(tmp_path, kd_lambda=5e-6):
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
    model = DGEKT(
        num_q=4,
        num_c=2,
        emb_size=8,
        hidden_dim=6,
        hypergraph=hypergraph,
        transition_out=transition_out,
        transition_in=transition_in,
    )
    trainer = DGEKTTrainer(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=1,
        device="cpu",
        other_config={"kd_lambda": kd_lambda, "kd_temperature": 0.5},
    )
    return model, trainer, stats, hypergraph


def test_dgekt_forward_backward_and_fold_safe_transition_graph(tmp_path):
    model, trainer, stats, hypergraph = _build(tmp_path)
    assert stats["train_folds"] == [0]
    assert stats["transition_count"] == 2
    assert stats["covered_questions"] == 4
    assert hypergraph.shape == (8, 4)

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


def test_kd_term_does_not_grow_with_batch_size(tmp_path):
    """Repeating a student must not change the loss.

    The supervised term is a mean over valid positions, so it is already
    batch-invariant. The distillation term is a `.sum()` over every logit, so
    without the matching `/ num_students` it grows linearly with the batch while
    the supervised term does not -- which is how it came to be 93% of the
    gradient on assist2009 at batch 64 and pinned training at chance. A large
    kd_lambda is used here so the imbalance, not floating-point noise, is what
    the comparison sees.
    """
    _, trainer, _, _ = _build(tmp_path, kd_lambda=1.0)
    trainer.model.eval()

    one = {
        "qseqs": torch.tensor([[0, 1]]),
        "shft_qseqs": torch.tensor([[1, 2]]),
        "rseqs": torch.tensor([[1, 0]]),
        "shft_rseqs": torch.tensor([[0.0, 1.0]]),
        "smasks": torch.tensor([[True, True]]),
    }
    four = {key: value.repeat(4, 1) for key, value in one.items()}

    _, _, loss_one = trainer._forward_batch(one)
    _, _, loss_four = trainer._forward_batch(four)

    assert torch.allclose(loss_one, loss_four, atol=1e-6), (
        f"loss changed with batch size ({loss_one.item():.6f} -> "
        f"{loss_four.item():.6f}); the distillation term is scaling with the batch"
    )
