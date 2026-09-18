"""HD-KT as a plugin: one class, three registered backbones, no forked forwards.

Replaces `models/hd_dkt.py`, `models/hd_akt.py` and `models/hd_simplekt.py`,
which between them carried 341 lines to wrap 196 lines of denoiser -- most of it
a copy of the backbone's own forward pass, kept only so that a single
multiplication could be inserted in the middle of it.

Adding a fourth backbone is now one `register_plugged` line, provided that
backbone has the four stages described in core/backbone.py.
"""

from core.model_inputs import InputSpec
from core.plugin import BackbonePlugin, register_plugged
from modules.hd_denoiser import HybridInteractionDenoiser


class HDInputs(InputSpec):
    """Declares what the HD backbones need; they derive nothing from the data."""

    dataset_mode = "all_in_one"


class HDPlugin(BackbonePlugin):
    """Causal hybrid denoising: gate the response-carrying history.

    The gate at position t is computed from positions strictly before t, so
    nothing here lets a prediction see its own answer. Only `history` is gated;
    the query stays intact at its prediction position, which is what the three
    wrappers each did and what the causality test in
    tests/test_plugin_composition.py pins.
    """

    def __init__(
        self,
        num_c,
        embedding_dim,
        num_q=0,
        detector_hidden=None,
        latent_dim=None,
        dropout=0.1,
        gumbel_tau=1.0,
        hard_detection=True,
        reconstruction_weight=0.01,
        kl_weight=0.001,
        **kwargs,
    ):
        super().__init__()
        self.reconstruction_weight = float(reconstruction_weight)
        self.denoiser = HybridInteractionDenoiser(
            num_items=max(int(num_q), int(num_c)),
            num_c=num_c,
            embedding_dim=embedding_dim,
            detector_hidden=detector_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
            gumbel_tau=gumbel_tau,
            hard_detection=hard_detection,
            kl_weight=kl_weight,
        )

    def transform(self, emb, batch):
        details = self.denoiser(
            batch.items(),
            batch.concepts,
            batch.responses.long().clamp(min=0, max=1),
            valid_mask=batch.valid_mask,
        )
        emb.history = emb.history * details["gate"].unsqueeze(-1)
        return emb, details

    def extra_loss(self, side):
        return self.reconstruction_weight * side["reconstruction_loss"]


for _name, _backbone, _width_key, _width_default in (
    ("hd_dkt", "dkt", "emb_size", 200),
    ("hd_akt", "akt", "d_model", 256),
    ("hd_simplekt", "simplekt", "emb_size", 256),
):
    register_plugged(
        _name,
        backbone=_backbone,
        plugin=HDPlugin,
        width_key=_width_key,
        width_default=_width_default,
        spec=HDInputs,
    )
