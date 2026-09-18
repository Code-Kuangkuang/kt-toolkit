"""`hdkt` must be `lpkt` plus the denoiser, and nothing else.

That is what makes them an ablation pair. If anything else differs, the row
"HD-LPKT vs LPKT" in a results table measures denoising plus that difference,
and no amount of care in running the sweep recovers the separation.

Something else did differ, and it took a targeted probe to find: LPKT floors
every concept weight at `gamma` (the Q-matrix smoothing from its paper) and
spreads its knowledge state over `num_c + 1` slots, while hdkt.py -- forked from
lpkt.py -- had a hard multi-hot over `num_c`. On a synthetic batch the gap was
1.3e-3 in predicted probability; on a results table it is indistinguishable from
a denoising effect.

So this asserts the relationship rather than any number: with the gate forced to
1 and every shared weight copied across, the two models must agree bit for bit.
Written this way it also fails if someone changes only one of them later, which
a value-based test on either model alone would not.

The sibling check for the three plugin-composed backbones is
tests/test_plugin_composition.py; there the composition makes drift impossible
by construction, so it asserts a weaker property. `hdkt` keeps its own
implementation, so it needs this.
"""

import unittest

import torch

from models.hdkt import HDKT
from models.lpkt import LPKT

B, T, L = 2, 10, 6
NUM_Q, NUM_C, NUM_AT, NUM_IT = 20, 8, 30, 30
DIMS = dict(d_a=8, d_e=8, d_k=8)


def _inputs():
    g = torch.Generator().manual_seed(7)
    e = torch.randint(1, NUM_Q, (B, T), generator=g)
    a = torch.randint(0, 2, (B, T), generator=g)
    it = torch.randint(0, 20, (B, T), generator=g)
    at = torch.randint(0, 20, (B, T), generator=g)
    # Distinct KCs per position. LPKT takes amax over the K slots; a duplicated
    # KC would be a separate question from the one being asked here.
    c0 = torch.randint(0, NUM_C - 1, (B, T), generator=g)
    c = torch.stack((c0, c0 + 1), dim=-1)
    valid = torch.ones(B, T, dtype=torch.bool)
    valid[0, L:] = False
    e[0, L:] = 0
    a[0, L:] = 0
    return e, a, it, at, c, valid


def _scored_mask():
    """Positions whose target is real. Predictions are read from `[:, 1:]`."""
    scored = torch.zeros(B, T - 1, dtype=torch.bool)
    scored[0, : L - 1] = True
    scored[1, :] = True
    return scored


def _pair():
    torch.manual_seed(0)
    lpkt = LPKT(
        num_q=NUM_Q, num_c=NUM_C, num_at=NUM_AT, num_it=NUM_IT,
        dropout=0.0, use_runtime_concepts=True, **DIMS
    ).eval()
    hdkt = HDKT(
        num_q=NUM_Q, num_c=NUM_C, num_at=NUM_AT, num_it=NUM_IT,
        dropout=0.0, **DIMS
    ).eval()
    return lpkt, hdkt


def _copy_shared_weights(lpkt, hdkt):
    """Returns (copied, mismatched). Weight init must not explain any gap."""
    src = dict(lpkt.named_parameters())
    copied, mismatched = [], []
    with torch.no_grad():
        for name, p in hdkt.named_parameters():
            if name not in src:
                continue  # the detector stack, which LPKT does not have
            if src[name].shape != p.shape:
                mismatched.append(
                    (name, tuple(src[name].shape), tuple(p.shape))
                )
            else:
                p.copy_(src[name])
                copied.append(name)
    return copied, mismatched


def _disable_denoising(hdkt):
    def gate_of_one(exercise_data, concept_data, responses, valid_mask):
        ones = torch.ones_like(exercise_data, dtype=torch.float)
        zeros = torch.zeros_like(ones)
        return ones, zeros, zeros, torch.zeros((), dtype=torch.float)

    hdkt._detect_anomalies = gate_of_one


class HDKTIsLPKTPlusDenoisingTest(unittest.TestCase):
    def test_every_shared_parameter_has_the_same_shape(self):
        """A shape gap means the two hold different amounts of state.

        `initial_knowledge` was (num_c, d_k) against LPKT's (num_c + 1, d_k),
        which is how the missing knowledge slot showed up.
        """
        lpkt, hdkt = _pair()
        _, mismatched = _copy_shared_weights(lpkt, hdkt)
        self.assertEqual(mismatched, [])

    def test_shared_parameters_actually_overlap(self):
        """Guards the test above: it passes trivially if nothing is shared."""
        lpkt, hdkt = _pair()
        copied, _ = _copy_shared_weights(lpkt, hdkt)
        self.assertGreaterEqual(len(copied), 20)
        for expected in ("e_embed.weight", "linear_5.weight", "initial_knowledge"):
            self.assertIn(expected, copied)

    def test_with_the_gate_off_hdkt_is_lpkt(self):
        lpkt, hdkt = _pair()
        _copy_shared_weights(lpkt, hdkt)
        _disable_denoising(hdkt)

        e, a, it, at, c, valid = _inputs()
        with torch.no_grad():
            lpkt_out = lpkt(
                e, a.float(), it_data=it, at_data=at,
                concept_data=c, valid_mask=valid,
            )
            hdkt_out = hdkt(
                e, c, a, it_data=it, at_data=at, valid_mask=valid
            )["predictions"]

        scored = _scored_mask()
        lp, hp = lpkt_out[:, 1:][scored], hdkt_out[:, 1:][scored]
        self.assertTrue(
            torch.equal(lp, hp),
            f"hdkt is not lpkt with denoising off: max |diff| = "
            f"{(lp - hp).abs().max().item():.3e}. Something other than the "
            "denoiser differs, so the two cannot share an ablation table.",
        )

    def test_gamma_is_what_the_previous_test_depends_on(self):
        """The check has to be able to fail, or it is decorative.

        `gamma=0` is the hard multi-hot hdkt.py used to have. If that still
        agreed with LPKT, the test above would be proving nothing.
        """
        torch.manual_seed(0)
        lpkt = LPKT(
            num_q=NUM_Q, num_c=NUM_C, num_at=NUM_AT, num_it=NUM_IT,
            dropout=0.0, use_runtime_concepts=True, **DIMS
        ).eval()
        hdkt = HDKT(
            num_q=NUM_Q, num_c=NUM_C, num_at=NUM_AT, num_it=NUM_IT,
            dropout=0.0, gamma=0.0, **DIMS
        ).eval()
        _copy_shared_weights(lpkt, hdkt)
        _disable_denoising(hdkt)

        e, a, it, at, c, valid = _inputs()
        with torch.no_grad():
            lpkt_out = lpkt(
                e, a.float(), it_data=it, at_data=at,
                concept_data=c, valid_mask=valid,
            )
            hdkt_out = hdkt(
                e, c, a, it_data=it, at_data=at, valid_mask=valid
            )["predictions"]

        scored = _scored_mask()
        lp, hp = lpkt_out[:, 1:][scored], hdkt_out[:, 1:][scored]
        self.assertFalse(torch.equal(lp, hp))

    def test_the_gate_still_changes_the_output(self):
        """With denoising on, the two must differ -- otherwise the denoiser is
        wired up but inert, and the ablation would measure nothing."""
        lpkt, hdkt = _pair()
        _copy_shared_weights(lpkt, hdkt)

        e, a, it, at, c, valid = _inputs()
        with torch.no_grad():
            lpkt_out = lpkt(
                e, a.float(), it_data=it, at_data=at,
                concept_data=c, valid_mask=valid,
            )
            hdkt_out = hdkt(
                e, c, a, it_data=it, at_data=at, valid_mask=valid
            )["predictions"]

        scored = _scored_mask()
        lp, hp = lpkt_out[:, 1:][scored], hdkt_out[:, 1:][scored]
        self.assertFalse(torch.equal(lp, hp))


if __name__ == "__main__":
    unittest.main()
