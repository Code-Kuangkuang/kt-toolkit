import torch
from torch.nn.functional import binary_cross_entropy

from core.trainers.hdkt_trainer import HDKTTrainer
from models.hdkt import HDKT


def _model(use_time=True):
    torch.manual_seed(7)
    return HDKT(
        num_q=12,
        num_c=6,
        num_at=8,
        num_it=8,
        d_a=8,
        d_e=8,
        d_k=8,
        detector_hidden=8,
        latent_dim=4,
        dropout=0.0,
        hard_detection=False,
        use_time=use_time,
    )


def _full_inputs():
    questions = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 0, 0]], dtype=torch.long
    )
    concepts = torch.tensor(
        [
            [[0, 1], [1, -1], [2, 3], [3, -1], [4, 5], [5, -1]],
            [[1, -1], [2, 3], [3, -1], [4, -1], [0, 0], [0, 0]],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [[0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.float
    )
    valid_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
    )
    interval_time = torch.tensor(
        [[0, 1, 2, 1, 3, 1], [0, 2, 2, 1, 0, 0]], dtype=torch.long
    )
    answer_time = torch.tensor(
        [[1, 2, 2, 3, 1, 4], [2, 1, 3, 2, 0, 0]], dtype=torch.long
    )
    return questions, concepts, responses, valid_mask, interval_time, answer_time


def test_hdkt_forward_backward_is_finite():
    model = _model()
    model.train()
    questions, concepts, responses, valid_mask, interval_time, answer_time = (
        _full_inputs()
    )
    output = model(
        questions,
        concepts,
        responses,
        it_data=interval_time,
        at_data=answer_time,
        valid_mask=valid_mask,
        return_details=True,
    )

    assert output["predictions"].shape == responses.shape
    assert output["denoise_gate"].shape == responses.shape
    assert torch.isfinite(output["predictions"]).all()
    assert torch.all((output["predictions"] >= 0) & (output["predictions"] <= 1))

    score_mask = valid_mask[:, 1:]
    prediction_loss = binary_cross_entropy(
        output["predictions"][:, 1:][score_mask], responses[:, 1:][score_mask]
    )
    loss = prediction_loss + model.reconstruction_weight * output[
        "reconstruction_loss"
    ]
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hdkt_predictions_are_causal_and_eval_is_deterministic():
    model = _model()
    model.eval()
    questions, concepts, responses, valid_mask, interval_time, answer_time = (
        _full_inputs()
    )
    changed_responses = responses.clone()
    changed_responses[:, 4:] = 1.0 - changed_responses[:, 4:]

    with torch.no_grad():
        original = model(
            questions,
            concepts,
            responses,
            it_data=interval_time,
            at_data=answer_time,
            valid_mask=valid_mask,
            return_details=True,
        )
        repeated = model(
            questions,
            concepts,
            responses,
            it_data=interval_time,
            at_data=answer_time,
            valid_mask=valid_mask,
            return_details=True,
        )
        changed = model(
            questions,
            concepts,
            changed_responses,
            it_data=interval_time,
            at_data=answer_time,
            valid_mask=valid_mask,
            return_details=True,
        )

    assert torch.equal(original["predictions"], repeated["predictions"])
    # A response at t=4 may affect prediction t=5, but never predictions <= 4.
    assert torch.allclose(
        original["predictions"][:, :5], changed["predictions"][:, :5]
    )


def test_hdkt_trainer_aligns_shifted_prediction_and_target():
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = HDKTTrainer(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=optimizer,
        num_epochs=1,
        device="cpu",
    )
    questions, concepts, responses, _, interval_time, answer_time = _full_inputs()
    batch = {
        "qseqs": questions[:, :-1],
        "shft_qseqs": questions[:, 1:],
        "cseqs": concepts[:, :-1],
        "shft_cseqs": concepts[:, 1:],
        "rseqs": responses[:, :-1],
        "shft_rseqs": responses[:, 1:],
        "itseqs": interval_time[:, :-1],
        "shft_itseqs": interval_time[:, 1:],
        "utseqs": answer_time[:, :-1],
        "shft_utseqs": answer_time[:, 1:],
        "masks": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
        ),
        "smasks": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool
        ),
    }

    pred, target, loss = trainer._forward_batch(batch)
    assert pred.shape == target.shape == (8,)
    assert torch.isfinite(loss)
    loss.backward()

