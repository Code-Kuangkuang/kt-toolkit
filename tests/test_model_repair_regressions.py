"""Regression coverage for the October 2026 model correctness audit."""

import copy
import random

import pytest
import torch

from core.trainers.atdkt_trainer import ATDKTTrainer
from core.trainers.dtransformer_trainer import DTransformerTrainer
from core.trainers.fakt_trainer import FAKTTrainer
from core.trainers.keenkt_trainer import KeenKTTrainer
from models.atdkt import ATDKT
from models.dtransformer import DTransformerModel
from models.fakt import FAKT
from models.keenkt import KeenKT
from models.stablekt import Architecture


def batch(length=8, lengths=(7, 6)):
    generator = torch.Generator().manual_seed(7)
    full = {
        "qseqs": torch.randint(0, 11, (2, length), generator=generator),
        "cseqs": torch.randint(0, 7, (2, length, 2), generator=generator),
        "rseqs": torch.randint(0, 2, (2, length), generator=generator).float(),
        "historycorrs": torch.rand(2, length, generator=generator),
    }
    for key in ("rgaps", "sgaps", "pcounts"):
        full[key] = torch.randint(0, 5, (2, length), generator=generator)
    mask = torch.arange(length - 1)[None, :] < torch.tensor(lengths)[:, None] - 1
    result = {}
    for key, values in full.items():
        result[key] = values[:, :-1].clone()
        result["shft_" + key] = values[:, 1:].clone()
    result.update(masks=mask, smasks=mask.clone())
    return result


def trainer(cls, model):
    return cls(model=model, train_loader=[], valid_loader=[],
               optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
               num_epochs=1, device="cpu")


def small_kwargs():
    return dict(num_c=7, num_q=11, d_model=16, n_blocks=1, d_ff=32,
                num_attn_heads=4, dropout=0.0, final_fc_dim=16, final_fc_dim2=8)


@pytest.mark.parametrize("arm", ["qidbandnomoe", "qidbandonlylstm", "qidbandonlycnn"])
def test_fakt_prefix_predictions_ignore_future_answers_and_other_students(arm):
    torch.manual_seed(7)
    model = FAKT(**small_kwargs(), num_rgap=5, num_sgap=5, num_pcount=5,
                 seq_len=8, emb_type=arm).eval()
    adapter = trainer(FAKTTrainer, model)
    original = batch(lengths=(8, 8))
    changed = copy.deepcopy(original)
    full = torch.cat((original["rseqs"][:, :1], original["shft_rseqs"]), dim=1)
    full[:, 3:] = 1 - full[:, 3:]
    changed["rseqs"], changed["shft_rseqs"] = full[:, :-1], full[:, 1:]
    with torch.no_grad():
        before = adapter._forward_batch(original)[0].view(2, -1)
        after = adapter._forward_batch(changed)[0].view(2, -1)
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=0)
        changed = copy.deepcopy(original)
        for key in ("rseqs", "shft_rseqs"):
            changed[key][1] = 1 - changed[key][1]
        other = adapter._forward_batch(changed)[0].view(2, -1)
        torch.testing.assert_close(before[0], other[0], atol=1e-6, rtol=0)
    model.train()
    _, _, loss = adapter._forward_batch(original, train=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("arm", ["qiddelxembhistranscembpredcurc", "qidcembpredhis"])
@pytest.mark.parametrize("start", [2, 50])
def test_atdkt_short_or_unscored_history_has_finite_training_loss(arm, start):
    model = ATDKT(num_q=11, num_c=7, seq_len=200, emb_size=16,
                  num_attn_heads=4, dropout=0.0, emb_type=arm, start=start)
    adapter = trainer(ATDKTTrainer, model)
    data = batch(length=200, lengths=(4, 4))
    pred, target, loss = adapter._forward_batch(data, train=True)
    assert pred.shape == target.shape == (6,)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    data["smasks"].zero_()
    pred, _, loss = adapter._forward_batch(data, train=True)
    assert pred.numel() == 0
    assert loss.item() == 0


def test_keenkt_uses_all_concepts_and_is_invariant_to_slot_order(monkeypatch):
    torch.manual_seed(7)
    model = KeenKT(**small_kwargs(), seq_len=8).eval()
    adapter = trainer(KeenKTTrainer, model)
    data = batch()
    data["cseqs"][:, :, 1] = -1
    data["shft_cseqs"][:, :, 1] = -1
    calls = []
    original = model.forward
    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, "forward", record)
    with torch.no_grad():
        before = adapter._forward_batch(data)[0]
        changed = copy.deepcopy(data)
        for key in ("cseqs", "shft_cseqs"):
            changed[key][:, :, 1] = (changed[key][:, :, 0] + 1) % 7
        with_second = adapter._forward_batch(changed)[0]
        assert not torch.allclose(before, with_second)
        for key in ("cseqs", "shft_cseqs"):
            changed[key] = changed[key].flip(-1)
        permuted = adapter._forward_batch(changed)[0]
        torch.testing.assert_close(with_second, permuted)
    assert all(call["augmented_responses"] is None and not call["compute_auxiliary"]
               for call in calls)


def dtransformer(window=1):
    torch.manual_seed(7)
    model = DTransformerModel(num_c=7, num_q=11, emb_size=16, n_blocks=1,
                              d_ff=32, num_attn_heads=4, n_know=2,
                              dropout=0.0, emb_type="qid_cl", window=window).eval()
    return model, trainer(DTransformerTrainer, model)


@pytest.mark.parametrize("lengths", [(4, 4), (7, 6)])
def test_dtransformer_cl_excludes_padding_and_preserves_input(lengths, monkeypatch):
    model, adapter = dtransformer(window=3)
    data = batch(lengths=lengths)
    saved = copy.deepcopy(data)
    seen_lengths = []
    original = model.sim
    def record(z1, z2):
        seen_lengths.append(z1.size(1))
        return original(z1, z2)
    monkeypatch.setattr(model, "sim", record)
    random.seed(7)
    before = adapter._forward_batch(data, train=True)[-1]
    changed = copy.deepcopy(data)
    for key in ("qseqs", "shft_qseqs", "cseqs", "shft_cseqs", "rseqs", "shft_rseqs"):
        mask = changed["masks"]
        if changed[key].dim() == 3:
            mask = mask.unsqueeze(-1).expand_as(changed[key])
        changed[key][~mask] = 999
    random.seed(7)
    after = adapter._forward_batch(changed, train=True)[-1]
    torch.testing.assert_close(before, after)
    before.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert seen_lengths == ([] if min(lengths) < 5 else [min(lengths)] * 2)
    for key in data:
        assert torch.equal(data[key], saved[key])


def test_dtransformer_augmentation_swaps_instead_of_duplicating(monkeypatch):
    model, _ = dtransformer()
    captured = []
    original = model.predict
    def record(q, s, pid=None, **kwargs):
        captured.append((q.clone(), pid.clone()))
        return original(q, s, pid, **kwargs)
    monkeypatch.setattr(model, "predict", record)
    monkeypatch.setattr(random, "sample", lambda population, count: [0])
    concepts = torch.arange(6).view(1, 6)
    answers = torch.tensor([[0, 1, 0, 1, 0, 1]])
    questions = concepts.clone()
    model.get_cl_loss(concepts, answers, questions)
    expected = torch.tensor([[1, 0, 2, 3, 4, 5]])
    assert torch.equal(captured[1][0], expected)
    assert torch.equal(captured[1][1], expected)


def test_dtransformer_window_loss_reaches_training_objective():
    model, adapter = dtransformer()
    data = batch()
    random.seed(7)
    base = adapter._forward_batch(data, train=True)[-1]
    model.window = 3
    random.seed(7)
    extended = adapter._forward_batch(data, train=True)[-1]
    assert extended > base
    extended.backward()
    assert torch.isfinite(model.out[-1].weight.grad).all()


@pytest.mark.parametrize("arm,expected", [("qid", 0), ("qid_wha", 2), ("qid_sin", 2)])
def test_stablekt_absolute_position_encoding_obeys_the_arm(arm, expected):
    model = Architecture(n_question=7, n_blocks=1, d_model=16, d_feature=4,
                         d_ff=32, n_heads=4, dropout=0.0, kq_same=True,
                         model_type="stablekt", seq_len=8, r=1, gamma=1,
                         emb_type=arm, num_buckets=16, max_distance=50)
    calls = []
    model.position_emb.register_forward_hook(lambda *args: calls.append(1))
    output = model(torch.randn(2, 8, 16), torch.randn(2, 8, 16))
    assert output.shape == (2, 8, 16)
    assert len(calls) == expected
