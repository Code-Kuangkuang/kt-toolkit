import torch

from core.trainers.hd_akt_trainer import HDAKTTrainer
from core.trainers.hd_dkt_trainer import HDDKTTrainer
from core.trainers.hd_simplekt_trainer import HDSimpleKTTrainer
from models.hd_akt import HDAKT
from models.hd_dkt import HDDKT
from models.hd_simplekt import HDSimpleKT


def _batch():
    questions = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=torch.long
    )
    concepts = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
    )
    responses = torch.tensor(
        [[0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 1]], dtype=torch.float
    )
    mask = torch.ones(2, 5, dtype=torch.bool)
    return {
        "qseqs": questions[:, :-1],
        "shft_qseqs": questions[:, 1:],
        "cseqs": concepts[:, :-1],
        "shft_cseqs": concepts[:, 1:],
        "rseqs": responses[:, :-1],
        "shft_rseqs": responses[:, 1:],
        "masks": mask,
        "smasks": mask,
    }


def _assert_trainer_backward(model, trainer_class):
    trainer = trainer_class(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=1,
        device="cpu",
    )
    result = trainer._forward_batch(_batch())
    pred, target, loss = result[0], result[1], result[-1]
    assert pred.shape == target.shape == (10,)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)


def test_all_hdkt_backbones_forward_backward():
    torch.manual_seed(11)
    _assert_trainer_backward(
        HDDKT(
            num_c=6,
            num_q=12,
            emb_size=8,
            detector_hidden=8,
            latent_dim=4,
            dropout=0.0,
            hard_detection=False,
        ),
        HDDKTTrainer,
    )
    _assert_trainer_backward(
        HDAKT(
            num_c=6,
            num_q=12,
            d_model=8,
            d_ff=16,
            final_fc_dim=16,
            num_attn_heads=2,
            detector_hidden=8,
            latent_dim=4,
            dropout=0.0,
            hard_detection=False,
        ),
        HDAKTTrainer,
    )
    _assert_trainer_backward(
        HDSimpleKT(
            num_c=6,
            num_q=12,
            emb_size=8,
            num_blocks=1,
            num_attn_heads=2,
            d_ff=16,
            final_fc_dim=16,
            final_fc_dim2=8,
            detector_hidden=8,
            latent_dim=4,
            dropout=0.0,
            hard_detection=False,
            seq_len=6,
        ),
        HDSimpleKTTrainer,
    )


def test_hd_backbones_do_not_use_future_responses():
    batch = _batch()
    questions = torch.cat(
        (batch["qseqs"][:, :1], batch["shft_qseqs"]), dim=1
    )
    concepts = torch.cat(
        (batch["cseqs"][:, :1], batch["shft_cseqs"]), dim=1
    )
    responses = torch.cat(
        (batch["rseqs"][:, :1], batch["shft_rseqs"]), dim=1
    ).long()
    changed = responses.clone()
    changed[:, 3:] = 1 - changed[:, 3:]
    full_mask = torch.ones_like(responses, dtype=torch.bool)

    models = [
        HDDKT(
            num_c=6,
            num_q=12,
            emb_size=8,
            detector_hidden=8,
            latent_dim=4,
            dropout=0.0,
            hard_detection=False,
        ),
        HDAKT(
            num_c=6,
            num_q=12,
            d_model=8,
            d_ff=16,
            final_fc_dim=16,
            num_attn_heads=2,
            detector_hidden=8,
            latent_dim=4,
            dropout=0.0,
            hard_detection=False,
        ),
    ]
    for model in models:
        model.eval()
    simple = HDSimpleKT(
        num_c=6,
        num_q=12,
        emb_size=8,
        num_blocks=1,
        num_attn_heads=2,
        d_ff=16,
        final_fc_dim=16,
        final_fc_dim2=8,
        detector_hidden=8,
        latent_dim=4,
        dropout=0.0,
        hard_detection=False,
        seq_len=6,
    ).eval()

    with torch.no_grad():
        dkt_before = models[0](
            concepts, responses, item_data=questions, valid_mask=full_mask
        )
        dkt_after = models[0](
            concepts, changed, item_data=questions, valid_mask=full_mask
        )
        akt_before = models[1](
            concepts, responses, pid_data=questions, valid_mask=full_mask
        )[0]
        akt_after = models[1](
            concepts, changed, pid_data=questions, valid_mask=full_mask
        )[0]
        simple_before = simple(
            qseqs=batch["qseqs"],
            rseqs=batch["rseqs"].long(),
            cseqs=batch["cseqs"],
            qshft=batch["shft_qseqs"],
            cshft=batch["shft_cseqs"],
            rshft=batch["shft_rseqs"].long(),
            valid_mask=full_mask,
        )
        changed_simple = simple(
            qseqs=batch["qseqs"],
            rseqs=changed[:, :-1],
            cseqs=batch["cseqs"],
            qshft=batch["shft_qseqs"],
            cshft=batch["shft_cseqs"],
            rshft=changed[:, 1:],
            valid_mask=full_mask,
        )

    # DKT prediction at t consumes response t; transformer predictions at t do not.
    assert torch.allclose(dkt_before[:, :3], dkt_after[:, :3])
    assert torch.allclose(akt_before[:, :4], akt_after[:, :4])
    assert torch.allclose(simple_before[:, :4], changed_simple[:, :4])

