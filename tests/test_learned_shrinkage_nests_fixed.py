"""The learned-weight difficulty arm has to contain the fixed-alpha arm exactly.

`qid_frozen*` shrinks an item's training-fold rate towards a target with weight
`alpha / (n_i + alpha)` at a fixed alpha. `qid_frozen*_learnw` fits

    w_i = sigmoid(level - slope * log n_i)

instead, initialised at `slope=1, level=log alpha`. Those two are the same
function, because

    alpha / (n + alpha)  ==  sigmoid(log alpha - log n),

so at step 0 the learned arm must reproduce the fixed arm element for element.

That identity is the whole experimental design: it makes the fixed arm the null
hypothesis *inside* the alternative, so a difference between the two arms can
only come from the fitting. If the initialisation drifts -- a `ddof` that does
not match numpy's, a `log1p` where the algebra wants `log`, a clamp applied on
the wrong side -- the two arms start from different models and the comparison
stops answering the question, silently and without any run failing.

So it is pinned here, against the numpy path itself rather than against a
remembered constant, for every grouping the frozen arm supports.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.feature_utils import _standardised_logodds  # noqa: E402
from models.simplekt import LearnedShrinkageDifficulty  # noqa: E402

NUM_Q = 40
BASE_RATE = 0.63


def _synthetic_counts(seed=3407):
    """Counts spanning the range that decides the shrinkage, plus unseen items.

    The weight only differs between arms where `n` is small, so a fixture drawn
    from a narrow band would pass while the formula was wrong. This covers
    n=0 (unseen), n=1 (assist2009 has 2,023 of them in a fold), and on up past
    the point where shrinkage stops mattering.
    """
    rng = np.random.default_rng(seed)
    count = np.concatenate([
        np.zeros(5),                      # unseen
        np.ones(8),                       # seen exactly once
        rng.integers(2, 6, 10),
        rng.integers(6, 21, 9),
        rng.integers(21, 400, 8),
    ]).astype(np.float64)
    assert len(count) == NUM_Q
    correct = np.floor(count * rng.uniform(0.1, 0.95, NUM_Q))
    rate = np.divide(correct, count, out=np.zeros_like(count), where=count > 0)
    return correct, count, rate


@pytest.mark.parametrize("alpha", [1.0, 10.0, 50.0])
@pytest.mark.parametrize("grouped_target", [False, True])
def test_learned_arm_reproduces_fixed_arm_at_initialisation(alpha, grouped_target):
    correct, count, rate = _synthetic_counts()
    if grouped_target:
        # Stands in for the leave-one-out group rate: a per-item target that is
        # not the global rate, which is the case where centring and shrinking
        # use two different numbers and are easiest to conflate.
        rng = np.random.default_rng(11)
        target = rng.uniform(0.2, 0.9, NUM_Q)
    else:
        target = np.full(NUM_Q, BASE_RATE)

    fixed = _standardised_logodds(correct, count, BASE_RATE, alpha, target=target)

    module = LearnedShrinkageDifficulty(
        rate=rate, target=target, count=count,
        base_rate=BASE_RATE, init_alpha=alpha,
    )
    learned = module.table().detach().numpy().ravel()

    np.testing.assert_allclose(learned, fixed, atol=1e-5, rtol=1e-4)


def test_initial_weights_are_the_closed_form():
    """`w_i` itself, not just the table it feeds, must equal alpha/(n+alpha)."""
    _, count, rate = _synthetic_counts()
    alpha = 10.0
    module = LearnedShrinkageDifficulty(
        rate=rate, target=np.full(NUM_Q, BASE_RATE), count=count,
        base_rate=BASE_RATE, init_alpha=alpha,
    )
    weights = module.weights().detach().numpy()

    seen = count > 0
    expected = alpha / (count + alpha)
    np.testing.assert_allclose(weights[seen], expected[seen], atol=1e-6)
    # An unseen item has no rate of its own, so it must sit entirely on the
    # target; anything less would mix in the zero that stands for "no data".
    assert np.all(weights[~seen] == 1.0)


def test_both_parameters_receive_gradient():
    """A weight that cannot move is the fixed arm wearing a different name."""
    _, count, rate = _synthetic_counts()
    module = LearnedShrinkageDifficulty(
        rate=rate, target=np.full(NUM_Q, BASE_RATE), count=count,
        base_rate=BASE_RATE, init_alpha=10.0,
    )
    pid = torch.arange(NUM_Q)
    module(pid).sum().backward()

    for name, parameter in (("slope", module.slope), ("level", module.level)):
        assert parameter.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} gradient not finite"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_frozen_inputs_stay_frozen():
    """Only the mixing weight is fitted; the rates and targets are not.

    The premise of the whole frozen line is that the item term is not learning
    per-item numbers. Two shared scalars cannot memorise 40 items, but only as
    long as the per-item tensors are buffers rather than parameters.
    """
    _, count, rate = _synthetic_counts()
    module = LearnedShrinkageDifficulty(
        rate=rate, target=np.full(NUM_Q, BASE_RATE), count=count,
        base_rate=BASE_RATE, init_alpha=10.0,
    )
    names = {name for name, _ in module.named_parameters()}
    assert names == {"slope", "level"}
    assert sum(p.numel() for p in module.parameters()) == 2
