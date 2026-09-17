import torch

from core.train_runner import _resolve_dataset_mode
from core.trainers.akt_trainer import AKTTrainer
from core.trainers.dkt_trainer import DKTTrainer
from core.trainers.lpkt_trainer import LPKTTrainer
from core.trainers.sakt_trainer import SAKTTrainer
from core.trainers.simplekt_trainer import SimpleKTTrainer
from models.akt import AKT
from models.dkt import DKT
from models.lpkt import LPKT
from models.sakt import SAKT
from models.simplekt import SimpleKT


def _question_batch():
    concepts = torch.tensor(
        [
            [[1, 2, -1], [2, -1, -1], [3, 4, 5], [1, 5, -1], [6, 7, -1]],
            [[2, 3, -1], [1, 4, -1], [5, -1, -1], [6, 0, -1], [7, -1, -1]],
        ],
        dtype=torch.long,
    )
    questions = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long
    )
    responses = torch.tensor(
        [[1, 0, 1, 0, 1], [0, 1, 1, 0, 0]], dtype=torch.long
    )
    interval = torch.ones_like(responses)
    answer_time = torch.tensor(
        [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=torch.long
    )
    mask = torch.ones(2, 4, dtype=torch.bool)
    return {
        "cseqs": concepts[:, :-1],
        "shft_cseqs": concepts[:, 1:],
        "qseqs": questions[:, :-1],
        "shft_qseqs": questions[:, 1:],
        "rseqs": responses[:, :-1],
        "shft_rseqs": responses[:, 1:].float(),
        "itseqs": interval[:, :-1],
        "shft_itseqs": interval[:, 1:],
        "utseqs": answer_time[:, :-1],
        "shft_utseqs": answer_time[:, 1:],
        "masks": mask,
        "smasks": mask,
    }


def _models_and_trainers():
    return [
        (DKT(num_c=8, emb_size=16, dropout=0.0), DKTTrainer),
        (
            SAKT(
                num_c=8,
                seq_len=5,
                emb_size=16,
                num_attn_heads=4,
                dropout=0.0,
                num_en=1,
            ),
            SAKTTrainer,
        ),
        (
            AKT(
                num_c=8,
                num_q=20,
                n_pid=20,
                d_model=16,
                n_blocks=1,
                dropout=0.0,
                d_ff=32,
                num_attn_heads=4,
            ),
            AKTTrainer,
        ),
        (
            SimpleKT(
                num_c=8,
                num_q=20,
                num_pid=20,
                emb_size=16,
                num_blocks=1,
                dropout=0.0,
                d_ff=32,
                num_attn_heads=4,
                seq_len=5,
                final_fc_dim=32,
                final_fc_dim2=16,
            ),
            SimpleKTTrainer,
        ),
    ]


def _trainer(model, trainer_class):
    return trainer_class(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=1,
        device="cpu",
    )


def test_model_config_controls_dataset_mode():
    train_cfg = {"dataset_mode": "one_by_one"}
    assert _resolve_dataset_mode(
        "dkt", train_cfg, {"dataset_mode": "all_in_one"}
    ) == "all_in_one"
    assert _resolve_dataset_mode(
        "dkt", {"dataset_mode": "all_in_one"}, {"dataset_mode": "one_by_one"}
    ) == "one_by_one"
    assert _resolve_dataset_mode(
        "dkt",
        train_cfg,
        {"dataset_mode": "one_by_one"},
        {"dataset_mode": "all_in_one"},
    ) == "all_in_one"


def test_all_in_one_baselines_have_one_prediction_per_question():
    torch.manual_seed(11)
    batch = _question_batch()
    for model, trainer_class in _models_and_trainers():
        trainer = _trainer(model, trainer_class)
        result = trainer._forward_batch(batch)
        pred, target, loss = result[0], result[1], result[-1]
        assert pred.shape == target.shape == (8,)
        assert torch.isfinite(loss)
        loss.backward()


def test_one_by_one_embedding_path_is_unchanged():
    concepts = torch.tensor([[1, 2, 3, 4], [2, 1, 5, 6]], dtype=torch.long)
    responses = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.long)

    dkt = DKT(num_c=8, emb_size=16, dropout=0.0).eval()
    with torch.no_grad():
        expected_embedding = dkt.interaction_emb(concepts + 8 * responses)
        hidden, _ = dkt.lstm_layer(expected_embedding)
        expected = torch.sigmoid(dkt.out_layer(dkt.dropout_layer(hidden)))
        actual = dkt(concepts, responses)
    torch.testing.assert_close(actual, expected)

    sakt = SAKT(
        num_c=8,
        seq_len=4,
        emb_size=16,
        num_attn_heads=4,
        dropout=0.0,
        num_en=1,
    )
    query, interaction = sakt.base_emb(concepts, responses, concepts)
    expected_query = sakt.exercise_emb(concepts)
    positions = torch.arange(4).unsqueeze(0)
    expected_interaction = (
        sakt.interaction_emb(concepts + 8 * responses)
        + sakt.position_emb(positions)
    )
    torch.testing.assert_close(query, expected_query)
    torch.testing.assert_close(interaction, expected_interaction)

    for model in (
        AKT(
            num_c=8,
            num_q=20,
            n_pid=20,
            d_model=16,
            d_ff=32,
            num_attn_heads=4,
        ),
        SimpleKT(
            num_c=8,
            num_q=20,
            num_pid=20,
            emb_size=16,
            num_blocks=1,
            d_ff=32,
            num_attn_heads=4,
            seq_len=4,
            final_fc_dim=32,
            final_fc_dim2=16,
        ),
    ):
        question_embedding, interaction_embedding = model.base_emb(
            concepts, responses
        )
        torch.testing.assert_close(question_embedding, model.q_embed(concepts))
        torch.testing.assert_close(
            interaction_embedding,
            model.qa_embed(responses) + model.q_embed(concepts),
        )


def test_multi_concept_order_is_permutation_invariant():
    torch.manual_seed(13)
    batch = _question_batch()
    permutation = torch.tensor([1, 0, 2])
    permuted = dict(batch)
    permuted["cseqs"] = batch["cseqs"][..., permutation]
    permuted["shft_cseqs"] = batch["shft_cseqs"][..., permutation]

    for model, trainer_class in _models_and_trainers():
        model.eval()
        trainer = _trainer(model, trainer_class)
        with torch.no_grad():
            original = trainer._forward_batch(batch)[0]
            reordered = trainer._forward_batch(permuted)[0]
        torch.testing.assert_close(original, reordered)


def test_all_in_one_predictions_do_not_use_current_target_response():
    torch.manual_seed(17)
    batch = _question_batch()
    changed = dict(batch)
    changed["shft_rseqs"] = batch["shft_rseqs"].clone()
    changed["shft_rseqs"][:, -1] = 1.0 - changed["shft_rseqs"][:, -1]

    for model, trainer_class in _models_and_trainers():
        model.eval()
        trainer = _trainer(model, trainer_class)
        with torch.no_grad():
            original = trainer._forward_batch(batch)[0].view(2, 4)
            counterfactual = trainer._forward_batch(changed)[0].view(2, 4)
        torch.testing.assert_close(original, counterfactual)


def test_lpkt_all_in_one_is_deterministic_and_uses_runtime_concepts():
    torch.manual_seed(19)
    batch = _question_batch()
    model = LPKT(
        num_at=10,
        num_it=10,
        num_q=20,
        num_c=8,
        d_a=8,
        d_e=8,
        d_k=8,
        dropout=0.0,
        use_runtime_concepts=True,
        dpath="does-not-need-a-qmatrix",
    )
    model.eval()
    trainer = _trainer(model, LPKTTrainer)
    with torch.no_grad():
        first = trainer._forward_batch(batch)[0]
        second = trainer._forward_batch(batch)[0]
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    model.train()
    pred, target, loss = trainer._forward_batch(batch)
    assert pred.shape == target.shape == (8,)
    loss.backward()
    assert model.at_embed.weight.grad is not None
    assert model.initial_knowledge.grad is not None
