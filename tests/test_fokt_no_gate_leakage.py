"""The forget gate must not see the response it is helping to predict.

This exists because the first version of `models/fokt.py` leaked, and the leak
was invisible to every check already in the repository:

* `tests/test_model_contracts.py` check 7 asks whether flipping a *future*
  response moves an *earlier* prediction. The gate leaked `r_t` into the
  prediction at `t` itself, which that check does not look at.
* The obvious algebra says it cannot happen: in `D_ij = c_i - c_j` the
  query-side `c_i` is constant across `j` and cancels in the softmax. What it
  misses is the key-side term at `j = i`, which contributes `-c_i`; the weight
  the diagonal gets relative to every other key therefore encodes `c_i`, and
  `c_i` contains `log f_i`, which contained `r_i`.

What it cost, before the fix: two epochs on assist2017 reached 0.958 window
AUC, against 0.696 for the same file's `qid` control arm, which is AKT.

So the invariant is pinned directly rather than argued: flip the response at
position `p` and the prediction at position `p` must not move by a single bit,
for every `p` and every arm.
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401,E402 -- registers the model
from core.registry import MODEL_REGISTRY  # noqa: E402

ARMS = ("qid", "qid_rkt", "qid_fox", "qid_fox_notime", "qid_fox_noresp",
        "qid_fox_akt", "qid_foxlm")
KWARGS = dict(
    n_question=20, n_pid=50, d_model=32, n_blocks=2,
    dropout=0.0, d_ff=32, num_attn_heads=4,
)


def _batch(batch=2, seq=10, max_c=2, seed=3):
    torch.manual_seed(seed)
    concepts = torch.randint(0, 20, (batch, seq, max_c))
    responses = torch.randint(0, 2, (batch, seq))
    questions = torch.randint(0, 50, (batch, seq))
    gaps = torch.randint(5_000, 86_400_000, (batch, seq))
    timestamps = 1_700_000_000_000 + torch.cumsum(gaps, dim=1)
    return concepts, responses, questions, timestamps


def _build(arm):
    torch.manual_seed(0)
    return MODEL_REGISTRY.get("fokt")(**KWARGS, emb_type=arm).eval()


class FoKTGateLeakageTest(unittest.TestCase):
    def test_flipping_a_response_cannot_move_its_own_prediction(self):
        c, r, q, t = _batch()
        for arm in ARMS:
            model = _build(arm)
            kwargs = {"t_data": t} if model.use_time else {}
            with torch.no_grad():
                base, _ = model(c, r, q, **kwargs)
            for pos in range(r.size(1)):
                with self.subTest(arm=arm, position=pos):
                    flipped = r.clone()
                    flipped[:, pos] = 1 - flipped[:, pos]
                    with torch.no_grad():
                        alt, _ = model(c, flipped, q, **kwargs)
                    moved = float((alt[:, pos] - base[:, pos]).abs().max())
                    self.assertEqual(
                        moved, 0.0,
                        f"{arm}: flipping r[{pos}] moved pred[{pos}] by {moved:.3e}; "
                        f"the gate is reading the label it predicts",
                    )

    def test_the_control_arm_is_bit_identical_to_akt(self):
        """If `qid` drifts from AKT the whole comparison is against a stranger."""
        c, r, q, t = _batch()
        torch.manual_seed(0)
        akt = MODEL_REGISTRY.get("akt")(**KWARGS, emb_type="qid").eval()
        fokt = _build("qid")
        with torch.no_grad():
            expected, _ = akt(c, r, q)
            actual, _ = fokt(c, r, q, t_data=t)
        self.assertTrue(torch.equal(expected, actual))

    def test_elapsed_time_reaches_the_arms_that_claim_to_use_it(self):
        """A silently-ignored `dt` would make the time arms a reparametrised
        control, and the resulting null would be read as "time does not help"."""
        c, r, q, _ = _batch()
        batch, seq = r.shape
        base = 1_700_000_000_000
        tight = base + torch.cumsum(torch.full((batch, seq), 10_000), dim=1)
        loose = base + torch.cumsum(torch.full((batch, seq), 10 * 86_400_000), dim=1)
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = _build(arm)
                if not model.use_time:
                    continue
                with torch.no_grad():
                    near, _ = model(c, r, q, t_data=tight)
                    far, _ = model(c, r, q, t_data=loose)
                self.assertGreater(float((near - far).abs().max()), 1e-6)

    def test_a_time_arm_refuses_to_run_without_timestamps(self):
        """Falling back to dt=0 would turn the time arm into the notime arm
        under a name that says otherwise."""
        c, r, q, _ = _batch()
        model = _build("qid_fox")
        with self.assertRaises(ValueError):
            model(c, r, q)


if __name__ == "__main__":
    unittest.main()
