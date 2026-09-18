"""What has to stay true now that plugged models are composed rather than forked.

The three HD models used to be hand-written copies of their backbones. The tests
that guarded them checked the copies ran and were causal. Those still matter and
are kept below, but the failure mode that actually bit -- a copy quietly drifting
from its original -- needs tests the old file could not express, because there
was no shared structure to compare against. Those are the first three here.
"""

import unittest

import torch

import core.trainers  # noqa: F401  (import-time registration)
import models  # noqa: F401
from core.factory import build_model
from core.registry import TRAINER_REGISTRY
from models.akt import AKT
from models.backbone import infer_valid_mask
from models.dkt import DKT
from models.plugin import PluggedKT
from models.simplekt import SimpleKT

NUM_C, NUM_Q = 6, 12
HD_KW = dict(detector_hidden=8, latent_dim=4, dropout=0.0, hard_detection=False)

PLUGGED = {
    "hd_dkt": dict(num_c=NUM_C, num_q=NUM_Q, emb_size=8, **HD_KW),
    "hd_akt": dict(
        num_c=NUM_C, num_q=NUM_Q, d_model=8, d_ff=16, final_fc_dim=16,
        num_attn_heads=2, **HD_KW
    ),
    "hd_simplekt": dict(
        num_c=NUM_C, num_q=NUM_Q, emb_size=8, num_blocks=1, num_attn_heads=2,
        d_ff=16, final_fc_dim=16, final_fc_dim2=8, seq_len=6, **HD_KW
    ),
}

BACKBONES = {
    "hd_dkt": lambda: DKT(num_c=NUM_C, emb_size=8, dropout=0.0),
    "hd_akt": lambda: AKT(
        num_c=NUM_C, num_q=NUM_Q, d_model=8, d_ff=16, final_fc_dim=16,
        num_attn_heads=2, dropout=0.0,
    ),
    "hd_simplekt": lambda: SimpleKT(
        num_c=NUM_C, num_q=NUM_Q, emb_size=8, num_blocks=1, num_attn_heads=2,
        d_ff=16, final_fc_dim=16, final_fc_dim2=8, dropout=0.0, seq_len=6,
    ),
}

BASE_TRAINER = {"hd_dkt": "dkt", "hd_akt": "akt", "hd_simplekt": "simplekt"}


def _batch(pad_tail=0):
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
    if pad_tail:
        questions[:, -pad_tail:] = -1
        concepts[:, -pad_tail:] = -1
        responses[:, -pad_tail:] = -1
        mask = (concepts[:, :-1] != -1) & (concepts[:, 1:] != -1)
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


def _build(name, seed=11):
    torch.manual_seed(seed)
    return build_model(name, **PLUGGED[name])


def _trainer(model, key):
    return TRAINER_REGISTRY.get(key)(
        model=model,
        train_loader=[],
        valid_loader=[],
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=1,
        device="cpu",
    )


class CompositionTest(unittest.TestCase):
    def test_backbone_inside_a_plugged_model_is_the_real_backbone(self):
        """Not a copy of it. This is the whole point of the refactor.

        A fork passes every behavioural test its author wrote and still diverges
        from the original the first time the original changes. Identity cannot
        drift.
        """
        for name, backbone_factory in BACKBONES.items():
            with self.subTest(name):
                model = _build(name)
                self.assertIsInstance(model, PluggedKT)
                self.assertIsInstance(
                    model.backbone, type(backbone_factory())
                )

    def test_plugged_weights_match_the_bare_backbone(self):
        """Same seed, same construction order, same initial weights.

        The backbone is built before the plugin so that it consumes the same
        prefix of the RNG stream as it would alone. Without that, a plugged run
        and its baseline start from different weights and every comparison
        between them carries an extra, undeclared difference.
        """
        for name, backbone_factory in BACKBONES.items():
            with self.subTest(name):
                torch.manual_seed(11)
                bare = backbone_factory()
                plugged = _build(name)
                for (bn, bp), (pn, pp) in zip(
                    bare.named_parameters(),
                    plugged.backbone.named_parameters(),
                ):
                    self.assertEqual(bn, pn)
                    self.assertTrue(torch.equal(bp, pp), f"{name}.{bn}")

    def test_plugged_loss_is_the_backbone_loss_plus_the_plugin_term(self):
        """The mixin adds; it does not reimplement.

        Each hand-written HD trainer re-derived its base trainer's loss and got
        it slightly wrong -- float32 BCE against the baseline's float64, a
        dropped item L2 penalty, a regularisation term added outside `cal_loss`.
        Every one of those was invisible without a side-by-side diff, so this
        asserts the relationship instead of the value.
        """
        for name in PLUGGED:
            with self.subTest(name):
                model = _build(name).eval()
                with torch.no_grad():
                    plugged_loss = _trainer(model, name)._forward_batch(
                        _batch()
                    )[-1]

                    model_again = _build(name).eval()
                    base = _trainer(model_again, BASE_TRAINER[name])
                    backbone_loss = base._forward_batch(_batch())[-1]
                    plugin_term = model_again.plugin.extra_loss(
                        model_again.take_side()
                    )

                self.assertTrue(
                    torch.allclose(
                        plugged_loss.double(),
                        (backbone_loss + plugin_term).double(),
                        atol=0,
                        rtol=0,
                    ),
                    f"{name}: {plugged_loss!r} != {backbone_loss!r} + {plugin_term!r}",
                )

    def test_side_channel_cannot_be_read_twice(self):
        """A stale plugin output must not silently enter the next batch's loss."""
        model = _build("hd_dkt").eval()
        with torch.no_grad():
            model(_batch()["cseqs"], _batch()["rseqs"])
        model.take_side()
        with self.assertRaises(RuntimeError):
            model.take_side()

    def test_inferred_valid_mask_matches_the_loaders_masks(self):
        """`infer_valid_mask` replaces `cat((masks[:, :1], masks), dim=1)`.

        The old expression reported `valid(t-1) & valid(t)` at position t. With
        suffix padding the two agree, which is what makes the replacement safe;
        this pins that rather than leaving it as an argument in a docstring.
        """
        for pad_tail in (0, 1, 2):
            with self.subTest(pad_tail=pad_tail):
                batch = _batch(pad_tail)
                concepts = torch.cat(
                    (batch["cseqs"][:, :1], batch["shft_cseqs"]), dim=1
                )
                masks = batch["masks"].bool()
                old = torch.cat((masks[:, 0:1], masks), dim=1)
                self.assertTrue(
                    torch.equal(infer_valid_mask(concepts), old),
                    f"pad_tail={pad_tail}",
                )


class BehaviourTest(unittest.TestCase):
    """Carried over from tests/test_hdkt_backbones.py."""

    def test_all_hdkt_backbones_forward_backward(self):
        for name in PLUGGED:
            with self.subTest(name):
                model = _build(name)
                result = _trainer(model, name)._forward_batch(_batch())
                pred, target, loss = result[0], result[1], result[-1]
                self.assertEqual(pred.shape, target.shape)
                self.assertEqual(pred.shape, (10,))
                self.assertTrue(torch.isfinite(pred).all())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(
                    all(torch.isfinite(g).all() for g in grads)
                )

    def test_plugin_gates_history_without_leaking_future_responses(self):
        """Flipping responses from t=3 on must not move predictions before it.

        The gate is what makes this non-obvious: it is computed from the whole
        response sequence, so a non-causal denoiser would leak here even though
        the backbone alone is causal.
        """
        batch = _batch()
        responses = torch.cat(
            (batch["rseqs"][:, :1], batch["shft_rseqs"]), dim=1
        ).long()
        changed = responses.clone()
        changed[:, 3:] = 1 - changed[:, 3:]

        for name, prefix in (("hd_dkt", 3), ("hd_akt", 4), ("hd_simplekt", 4)):
            with self.subTest(name):
                model = _build(name).eval()
                kw = (
                    dict(
                        qseqs=batch["qseqs"], cseqs=batch["cseqs"],
                        qshft=batch["shft_qseqs"], cshft=batch["shft_cseqs"],
                    )
                    if name == "hd_simplekt"
                    else {}
                )
                with torch.no_grad():
                    if name == "hd_simplekt":
                        before = model(
                            rseqs=responses[:, :-1], rshft=responses[:, 1:], **kw
                        )
                        after = model(
                            rseqs=changed[:, :-1], rshft=changed[:, 1:], **kw
                        )
                    else:
                        concepts = torch.cat(
                            (batch["cseqs"][:, :1], batch["shft_cseqs"]), dim=1
                        )
                        questions = torch.cat(
                            (batch["qseqs"][:, :1], batch["shft_qseqs"]), dim=1
                        )
                        key = "item_data" if name == "hd_dkt" else "pid_data"
                        before = model(concepts, responses, **{key: questions})
                        after = model(concepts, changed, **{key: questions})
                        if name == "hd_akt":
                            before, after = before[0], after[0]
                self.assertTrue(
                    torch.allclose(before[:, :prefix], after[:, :prefix]),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
