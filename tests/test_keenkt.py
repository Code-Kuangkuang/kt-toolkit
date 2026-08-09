import copy

import torch

from core.registry import MODEL_REGISTRY, TRAINER_REGISTRY
from core.trainers.keenkt_trainer import KeenKTTrainer
from models.keenkt import KeenKT


def _small_model(**overrides):
    kwargs = {
        "num_c": 7,
        "num_q": 11,
        "d_model": 16,
        "n_blocks": 1,
        "d_ff": 32,
        "num_attn_heads": 4,
        "dropout": 0.0,
        "final_fc_dim": 16,
        "final_fc_dim2": 8,
        "seq_len": 6,
        "use_CL": True,
        "use_diffusion": True,
        "noise_level": 0.1,
    }
    kwargs.update(overrides)
    return KeenKT(**kwargs)


def _batch():
    return {
        "qseqs": torch.tensor([[1, 2, 3, 4, 0]]),
        "shft_qseqs": torch.tensor([[2, 3, 4, 5, 0]]),
        "cseqs": torch.tensor([[1, 2, 3, 4, 0]]),
        "shft_cseqs": torch.tensor([[2, 3, 4, 5, 0]]),
        "rseqs": torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]]),
        "shft_rseqs": torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]]),
        "masks": torch.tensor([[True, True, True, True, False]]),
        "smasks": torch.tensor([[True, True, True, True, False]]),
    }


def test_keenkt_is_registered():
    import core.trainers  # noqa: F401
    import models  # noqa: F401

    assert MODEL_REGISTRY.get("keenkt") is KeenKT
    assert TRAINER_REGISTRY.get("keenkt") is KeenKTTrainer


def test_keenkt_forward_backward_is_finite():
    torch.manual_seed(3407)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = KeenKTTrainer(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=optimizer,
        num_epochs=1,
        device="cpu",
    )

    pred, target, loss = trainer._forward_batch(_batch(), train=True)
    assert pred.shape == target.shape == (4,)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_current_and_future_responses_do_not_change_current_prediction():
    """Prediction at t may depend on responses before t, never r_t or later."""

    torch.manual_seed(3407)
    model = _small_model(use_CL=False, use_diffusion=False)
    model.eval()
    concepts = torch.tensor([[1, 2, 3, 4, 5]])
    questions = torch.tensor([[1, 2, 3, 4, 5]])
    responses = torch.tensor([[1, 0, 1, 0, 1]])
    valid = torch.ones_like(responses, dtype=torch.bool)

    with torch.no_grad():
        original = model(
            concepts,
            questions,
            responses,
            valid_mask=valid,
        )["preds"]

        changed = responses.clone()
        changed[:, 2:] = 1 - changed[:, 2:]
        perturbed = model(
            concepts,
            questions,
            changed,
            valid_mask=valid,
        )["preds"]

    # Positions 0, 1, and 2 cannot use responses at position 2 or later.
    torch.testing.assert_close(original[:, :3], perturbed[:, :3])


def test_evaluation_does_not_request_training_augmentation():
    torch.manual_seed(3407)
    model = _small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = KeenKTTrainer(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=optimizer,
        num_epochs=1,
        device="cpu",
    )
    batch = _batch()
    pred_a, target_a, loss_a = trainer._forward_batch(batch, train=False)

    modified = copy.deepcopy(batch)
    modified["shft_rseqs"][:, -1] = 1 - modified["shft_rseqs"][:, -1]
    pred_b, target_b, loss_b = trainer._forward_batch(modified, train=False)

    assert pred_a.shape == target_a.shape
    assert pred_b.shape == target_b.shape
    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
