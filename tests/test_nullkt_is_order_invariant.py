"""`nullkt` is only a null if it provably cannot use order.

The claim the model makes is structural: every feature is a count or a mean over
the prefix, so permuting a student's history before position `t` leaves the
prediction at `t` untouched.  A claim like that is worth nothing asserted in a
docstring -- one `cumsum` written inclusively instead of exclusively, or one
feature that indexes the previous position, and the "null" quietly becomes a
lag-1 model that beats the thing it was supposed to be a floor for.

So it is pinned here, on real shapes, for all three rungs.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models  # noqa: F401,E402  (registers nullkt)
from core.registry import MODEL_REGISTRY  # noqa: E402
from models.nullkt import LADDER, exclusive_cumsum  # noqa: E402

NUM_C, NUM_Q, B, T = 7, 23, 4, 11


def build(emb_type, multi_concept=False):
    torch.manual_seed(0)
    model = MODEL_REGISTRY.get("nullkt")(
        num_c=NUM_C, num_q=NUM_Q, emb_type=emb_type, base_rate=0.6
    )
    # A zero head would make every prediction identical and the test vacuous.
    with torch.no_grad():
        model.head.weight.normal_(0.0, 1.0)
    model.eval()
    return model


def sequences(multi_concept=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    if multi_concept:
        concepts = torch.randint(0, NUM_C, (B, T, 3), generator=g)
        concepts[:, :, 2] = -1  # a ragged row, as real multi-concept data has
    else:
        concepts = torch.randint(0, NUM_C, (B, T), generator=g)
    questions = torch.randint(0, NUM_Q, (B, T), generator=g)
    responses = torch.randint(0, 2, (B, T), generator=g)
    return questions, concepts, responses


def predict(model, questions, concepts, responses):
    """Feed full sequences through the (current, shifted) signature."""
    return model(
        qseqs=questions[:, 0:1],
        rseqs=responses[:, 0:1],
        cseqs=concepts[:, 0:1],
        qshft=questions[:, 1:],
        cshft=concepts[:, 1:],
        rshft=responses[:, 1:],
    )


@pytest.mark.parametrize("emb_type", sorted(LADDER))
@pytest.mark.parametrize("multi_concept", [False, True])
def test_prefix_order_does_not_change_the_last_prediction(emb_type, multi_concept):
    model = build(emb_type)
    questions, concepts, responses = sequences(multi_concept)

    with torch.no_grad():
        before = predict(model, questions, concepts, responses)[:, -1]

    # Shuffle every position except the one being predicted. Questions, concepts
    # and responses move together, so this is a permutation of the student's
    # history, not a relabelling of it.
    perm = torch.randperm(T - 1, generator=torch.Generator().manual_seed(7))
    idx = torch.cat([perm, torch.tensor([T - 1])])
    with torch.no_grad():
        after = predict(
            model, questions[:, idx], concepts[:, idx], responses[:, idx]
        )[:, -1]

    assert torch.allclose(before, after, atol=1e-6), (
        f"{emb_type}: permuting the prefix moved the prediction by "
        f"{(before - after).abs().max().item():.2e}; nullkt is using order."
    )


@pytest.mark.parametrize("emb_type", sorted(LADDER))
def test_a_future_response_cannot_move_an_earlier_prediction(emb_type):
    model = build(emb_type)
    questions, concepts, responses = sequences()

    with torch.no_grad():
        before = predict(model, questions, concepts, responses)

    flipped = responses.clone()
    flipped[:, -1] = 1 - flipped[:, -1]
    with torch.no_grad():
        after = predict(model, questions, concepts, flipped)

    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6), (
        f"{emb_type}: flipping the last response changed an earlier prediction."
    )


def test_the_item_rung_ignores_responses_entirely():
    """The bottom rung is pure item bias: it has no student term at all."""
    model = build("item")
    questions, concepts, responses = sequences()
    with torch.no_grad():
        before = predict(model, questions, concepts, responses)
        after = predict(model, questions, concepts, 1 - responses)
    assert torch.allclose(before, after, atol=1e-7)


def test_exclusive_cumsum_excludes_the_current_position():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(
        exclusive_cumsum(x), torch.tensor([[0.0, 1.0, 3.0, 6.0]])
    )


def test_first_position_makes_no_claim_about_the_student():
    """With nothing observed, the prefix features must be exactly 0."""
    model = build("count")
    questions, concepts, responses = sequences()
    feats = model.features(questions, concepts, responses)
    for name in ("student_acc", "student_n", "skill_acc", "skill_n"):
        col = model.feature_names.index(name)
        assert torch.allclose(
            feats[:, 0, col], torch.zeros(B), atol=1e-6
        ), f"{name} is nonzero at position 0"
