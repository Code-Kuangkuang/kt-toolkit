"""Mechanism, causality, isolation and mask tests beyond registry contracts."""

import copy

import pytest
import torch

from core.run_support import apply_overrides
from core.trainers.simplekt_delta_trainer import SimpleKTDeltaTrainer
from models.residual_memory import ResidualMemory, delta_write
from models.simplekt_delta import SimpleKTDelta


def small_model(rule="delta", **kwargs):
    torch.manual_seed(71)
    return SimpleKTDelta(num_c=3, num_q=5, emb_type="qid_gru", emb_size=8,
                         final_fc_dim=12, final_fc_dim2=8, dropout=0.0,
                         memory_rule=rule, memory_dim=4, **kwargs)


def sequences():
    return {
        "questions": torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, -1, -1]]),
        "concepts": torch.tensor([[[0, -1], [0, 1], [1, -1], [0, 2], [2, -1]],
                                  [[1, -1], [0, 1], [1, 2], [-1, -1], [-1, -1]]]),
        "responses": torch.tensor([[1., 0., 0., 1., 0.], [0., 1., 0., 0., 0.]]),
        "valid_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
    }


def loader_batch(data):
    batch = {}
    for name, field in (("questions", "qseqs"), ("concepts", "cseqs"), ("responses", "rseqs")):
        batch[field] = data[name][:, :-1]
        batch["shft_" + field] = data[name][:, 1:]
    batch["masks"] = data["valid_mask"][:, 1:]
    batch["smasks"] = batch["masks"].clone()
    return batch


@pytest.mark.parametrize("rate", [0., 0.1, 1.])
def test_delta_preserves_orthogonal_component_and_moves_read_towards_value(rate):
    torch.manual_seed(3)
    m = torch.randn(4, 7, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn_like(m), dim=-1)
    value = torch.randn(4, dtype=torch.float64)
    after = delta_write(m, k, value, rate)
    old_read, new_read = (m * k).sum(-1), (after * k).sum(-1)
    torch.testing.assert_close(new_read, (1 - rate) * old_read + rate * value)
    torch.testing.assert_close(after - new_read[:, None] * k, m - old_read[:, None] * k)


def test_one_hot_delta_is_single_kc_ema():
    features = torch.eye(3)[torch.tensor([[0, 1, 1, 2, 1, 0]])]
    residual = torch.tensor([[.8, -.6, .4, .5, -.1, .3]])
    valid = torch.ones_like(residual, dtype=torch.bool)
    dense = ResidualMemory(3, 3, 3, .1, "delta")
    with torch.no_grad():
        dense.key_projection.weight.copy_(torch.eye(3))
    sparse = ResidualMemory(3, 3, 3, .1, "kc_ema")
    d_read, d_last = dense(features, residual, valid)
    k_read, k_last = sparse(features, residual, valid, features.argmax(-1))
    torch.testing.assert_close(d_read, k_read)
    torch.testing.assert_close(d_last, k_last)


def test_multi_kc_duplicates_are_updated_once_and_padding_never_writes():
    memory = ResidualMemory(2, 3, memory_rule="kc_ema", memory_rate=.5)
    features = torch.zeros(1, 4, 2)
    valid = torch.tensor([[True, True, True, False]])
    concepts = torch.tensor([[[0, -1, -1], [0, 0, 1], [0, 1, -1], [2, 2, 2]]])
    reads, last = memory(features, torch.tensor([[1., .8, .2, 99.]]), valid, concepts)
    torch.testing.assert_close(reads, torch.tensor([[0., 0., .4, 0.]]))
    torch.testing.assert_close(last, torch.tensor([[.3, .3, 0.]]))


@pytest.mark.parametrize("rule", ResidualMemory.RULES)
def test_current_and_future_labels_cannot_change_current_or_past_predictions(rule):
    model, data = small_model(rule).eval(), sequences()
    before = model(**data)
    changed = copy.deepcopy(data)
    changed["responses"][:, 2:] = 1 - changed["responses"][:, 2:]
    after = model(**changed)
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)


@pytest.mark.parametrize("rule", ["delta", "ema", "kc_ema"])
def test_state_resets_and_students_do_not_share_state(rule):
    model, data = small_model(rule).eval(), sequences()
    together = model(**data)
    for row in range(2):
        alone = model(**{name: value[row:row+1] for name, value in data.items()})
        torch.testing.assert_close(together[row:row+1], alone)
    torch.testing.assert_close(together, model(**data), rtol=0, atol=0)


def test_none_replays_original_simplekt_on_scored_positions():
    model, data = small_model("none").eval(), sequences()
    batch = loader_batch(data)
    # KTQueDataset multiplies padded question IDs by masks before delivery.
    batch["qseqs"] = batch["qseqs"].clamp(min=0)
    batch["shft_qseqs"] = batch["shft_qseqs"].clamp(min=0)
    original = model.backbone(qseqs=batch["qseqs"], rseqs=batch["rseqs"],
        cseqs=batch["cseqs"], qshft=batch["shft_qseqs"], cshft=batch["shft_cseqs"],
        rshft=batch["shft_rseqs"])
    torch.testing.assert_close(model(**data)[data["valid_mask"]], original[data["valid_mask"]])


def test_zero_rate_reduces_to_calibrated_backbone():
    model, data = small_model(memory_rate=0).eval(), sequences()
    details = model(**data, return_details=True)
    assert torch.count_nonzero(details["memory_read"]) == 0
    torch.testing.assert_close(details["logits"], details["base_logits"])


def test_frozen_backbone_stays_in_eval_and_only_memory_parameters_learn():
    model, data = small_model(), sequences()
    original = copy.deepcopy(model.backbone.state_dict())
    model.load_backbone(original, freeze=True)
    model.train()
    assert not model.backbone.training
    output = model(**data, return_details=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["logits"][data["valid_mask"]], data["responses"][data["valid_mask"]])
    loss.backward()
    grad = model.memory.key_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    torch.optim.Adam(model.parameters(), lr=.01).step()
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_padding_content_cannot_change_predictions_or_final_memory():
    model, data = small_model().eval(), sequences()
    expected = model(**data, return_details=True)
    changed = copy.deepcopy(data)
    changed["questions"][~data["valid_mask"]] = 999
    changed["concepts"][~data["valid_mask"]] = 999
    changed["responses"][~data["valid_mask"]] = float("nan")
    actual = model(**changed, return_details=True)
    torch.testing.assert_close(actual["final_state"], expected["final_state"])
    torch.testing.assert_close(actual["logits"], expected["logits"])


def test_trainer_uses_score_mask_for_loss_but_valid_context_for_writing():
    model, data = small_model().eval(), sequences()
    batch = loader_batch(data)
    batch["smasks"][:, 0] = False
    trainer = SimpleKTDeltaTrainer(model, None, None, None, 1, "cpu")
    pred, target, loss = trainer._forward_batch(batch)
    assert len(pred) == len(target) == int(batch["smasks"].sum())
    assert torch.isfinite(loss)
    full, score = trainer.full_batch(batch, "cpu")
    details = model(**full, return_details=True)
    assert details["memory_read"][:, 2].abs().sum() > 0
    torch.testing.assert_close(pred, details["logits"][:, 1:][score].sigmoid())
    empty = copy.deepcopy(batch)
    empty["smasks"].fill_(False)
    pred, target, loss = trainer._forward_batch(empty)
    assert pred.numel() == target.numel() == 0 and loss.item() == 0


def test_invalid_valid_data_fails_and_memory_overrides_are_consumed():
    model, data = small_model(), sequences()
    data["questions"][0, 1] = 5
    with pytest.raises(ValueError, match="Question ID"):
        model(**data)
    cfg = {}
    expected = dict(memory_rule="ema", memory_dim=8, memory_rate=.2, residual_detach=False)
    apply_overrides({}, cfg, expected)
    assert cfg == expected


@pytest.mark.parametrize("rate", [-.1, 1.1, float("nan")])
def test_invalid_rate_fails(rate):
    with pytest.raises(ValueError, match="memory_rate"):
        small_model(memory_rate=rate)
