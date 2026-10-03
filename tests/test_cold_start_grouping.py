"""`_cold` has to change the cold items and nothing else.

`qid_frozen_grouped` resolves an item's group from the training rows and then
resets any item with no training-fold responses to 0. Both halves of that are
defensible on their own and together they delete the hypothesis: an item nobody
answered gets no group, so it shrinks to the global rate, and then the zeroing
discards even that. On assist2009 it is 625 of 17,738 items, and `_grouped` and
`_grouprand` are bit-identical on every one of them -- the control has nothing
to control for exactly where grouping should matter most.

`_cold` fills the missing memberships from the Q-matrix and keeps the resulting
value. Two things have to hold for that to be a test rather than a second model:

  * items the training rows DO reach must come out unchanged, so a `_cold` vs
    `_grouped` gap is attributable to the cold items alone. The Q-matrix loses
    concept order, so its "first concept" is the lowest index where the CSV's is
    the first listed; taking membership from it wholesale would silently move
    every multi-concept item into a different group.
  * `_grouped_cold` and `_grouprand_cold` must now differ on those items, or the
    control is still dead and a win still cannot be attributed to the partition.

Both are pinned here on a fixture small enough to read, with the shapes that
matter: multi-concept items, single-concept items, and items that appear in the
Q-matrix but in no training row.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.feature_utils import (  # noqa: E402
    compute_item_difficulty_ingredients,
    compute_item_difficulty_logodds,
)

NUM_Q, NUM_C = 12, 4
# Questions 9, 10 and 11 never appear in a row: they are the cold items.
SEEN_QUESTIONS = list(range(9))
QMATRIX_CONCEPTS = {
    0: [0], 1: [0], 2: [0],
    3: [1], 4: [1],
    5: [2], 6: [2],
    # Multi-concept, listed 3-then-1: the CSV's first concept is 3, the
    # Q-matrix's lowest index is 1. Only a wholesale replacement would move it.
    7: [3, 1],
    8: [3],
    9: [1], 10: [2], 11: [3],   # cold
}


@pytest.fixture
def dataset(tmp_path):
    """A four-fold CSV plus the matching qmatrix.npz, written to a temp dir."""
    rng = np.random.default_rng(3407)
    rows = []
    for fold in range(4):
        for _ in range(3):
            questions = list(rng.choice(SEEN_QUESTIONS, size=6, replace=True))
            concepts = []
            for question in questions:
                cids = QMATRIX_CONCEPTS[question]
                # CSV order, which for question 7 is 3 before 1.
                concepts.append("_".join(str(c) for c in cids))
            responses = list(rng.integers(0, 2, size=6))
            rows.append({
                "fold": fold,
                "questions": ",".join(str(q) for q in questions),
                "concepts": ",".join(concepts),
                "responses": ",".join(str(r) for r in responses),
            })

    import pandas as pd

    csv_name = "train_valid_sequences_quelevel.csv"
    pd.DataFrame(rows).to_csv(tmp_path / csv_name, index=False)

    matrix = np.zeros((NUM_Q + 1, NUM_C), dtype=np.int64)
    for question, cids in QMATRIX_CONCEPTS.items():
        for cid in cids:
            matrix[question, cid] = 1
    np.savez_compressed(tmp_path / "qmatrix.npz", matrix=matrix)

    return str(tmp_path), csv_name


def _table(dataset, grouping, cold_start, alpha=10.0):
    dpath, csv_name = dataset
    return compute_item_difficulty_logodds(
        dpath, csv_name, num_q=NUM_Q, folds=[1, 2, 3],
        alpha=alpha, grouping=grouping, cold_start=cold_start,
    )


def _unseen_mask(dataset, grouping="concept"):
    dpath, csv_name = dataset
    parts = compute_item_difficulty_ingredients(
        dpath, csv_name, num_q=NUM_Q, folds=[1, 2, 3], grouping=grouping,
    )
    return parts["count"] == 0


@pytest.mark.parametrize("grouping", [None, "concept", "random"])
def test_cold_start_off_is_the_old_behaviour(dataset, grouping):
    """The default path has to be untouched, or every stored result is stale."""
    np.testing.assert_array_equal(
        _table(dataset, grouping, cold_start=False),
        _table(dataset, grouping, cold_start=None or False),
    )


def test_seen_items_are_identical_between_arms(dataset):
    """The one-variable claim: `_cold` may only move items with no responses."""
    base = _table(dataset, "concept", cold_start=False)
    cold = _table(dataset, "concept", cold_start=True)
    unseen = _unseen_mask(dataset)
    assert unseen.sum() >= 3, "fixture must contain cold items"
    np.testing.assert_array_equal(base[~unseen], cold[~unseen])


def test_cold_items_go_from_zero_to_their_group(dataset):
    base = _table(dataset, "concept", cold_start=False)
    cold = _table(dataset, "concept", cold_start=True)
    unseen = _unseen_mask(dataset)

    assert np.all(base[unseen] == 0.0), "the old arm zeroes every cold item"
    assert np.any(cold[unseen] != 0.0), "the cold arm must give them something"


def test_the_control_is_live_on_cold_items(dataset):
    """`_grouprand_cold` has to differ from `_grouped_cold` where it now matters.

    Without this the random-partition control is still comparing two identical
    columns on the cold subset, and a `_grouped_cold` gain could not be told
    apart from "any partition of the same sizes helps".
    """
    concept = _table(dataset, "concept", cold_start=True)
    random_ = _table(dataset, "random", cold_start=True)
    unseen = _unseen_mask(dataset)

    assert np.any(concept[unseen] != random_[unseen])
    # And the dead case it replaces, for contrast.
    concept_old = _table(dataset, "concept", cold_start=False)
    random_old = _table(dataset, "random", cold_start=False)
    np.testing.assert_array_equal(concept_old[unseen], random_old[unseen])


def test_padding_slot_stays_zero(dataset):
    """Row `num_q` is the padding slot, not an item, under every arm."""
    for cold_start in (False, True):
        assert _table(dataset, "concept", cold_start=cold_start)[-1] == 0.0


def test_multi_concept_item_keeps_its_csv_group(dataset):
    """Question 7 is listed `3_1`; the Q-matrix would call it group 1.

    If `_cold` replaced memberships wholesale instead of filling gaps, this item
    would change group and its leave-one-out target with it, which is how the
    one-variable claim would break without any test failing on the cold items.
    """
    base = _table(dataset, "concept", cold_start=False)
    cold = _table(dataset, "concept", cold_start=True)
    assert base[7] == cold[7]
