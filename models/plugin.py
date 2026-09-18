"""Compose a backbone with a plugin, once, instead of forking it per pairing.

A plugin is a module that wants to change one intermediate tensor inside a
backbone and add a term to its loss. Before this existed, each (plugin,
backbone) pairing was a hand-written model file plus a hand-written trainer, so
the cost of a plugin was O(backbones) and every copy could drift from its
original. `models/backbone.py` names the seam; this file does the wiring.

    register_plugged("hd_akt", backbone="akt", plugin=HDPlugin, width_key="d_model")

produces a registered model that behaves like `akt` with the plugin's transform
applied between `embed` and `encode`, and takes the same constructor arguments
as `akt` plus the plugin's own.

Two things are deliberately explicit rather than clever:

* No `register_forward_hook`. A hook would need no backbone changes at all, but
  it scales a tensor with nothing in the source saying so. This repository is
  built around being able to audit what a run did; an invisible side effect is
  the wrong trade even when it is shorter.

* `PluggedKT.side` is the one piece of state passed out of band, because the
  backbone's `forward` return type is fixed by its 34 existing callers and the
  plugin's auxiliary loss has nowhere else to go. It is cleared on read, so a
  stale value from the previous batch cannot quietly enter a loss.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch.nn as nn

from core.factory import _filter_to_signature
from core.registry import MODEL_REGISTRY

from .backbone import Embeddings, SeqBatch


class BackbonePlugin(nn.Module):
    """Override `transform`, and `extra_loss` if the plugin has one."""

    def transform(
        self, emb: Embeddings, batch: SeqBatch
    ) -> Tuple[Embeddings, Dict[str, Any]]:
        """Return the embeddings to encode, plus anything the loss needs.

        Only `emb.history` should be written. `emb.query` and `emb.extras` are
        the backbone's, and a plugin that needs to change them is describing a
        different model, not a plugin.
        """
        return emb, {}

    def extra_loss(self, side: Dict[str, Any]):
        """Added to whatever loss the backbone's own trainer computed.

        `None` means the plugin contributes nothing.
        """
        return None


class PluggedKT(nn.Module):
    """`backbone` with `plugin` spliced in at the embed/encode seam.

    Constructor arguments are split by signature, the same way
    `core/factory.py` narrows a config block for any model, so a single config
    block can carry both the backbone's hyperparameters and the plugin's.

    The backbone is built first and the plugin second, matching the order the
    hand-written wrappers used. That ordering is not cosmetic: it fixes how much
    of the RNG stream each consumes, so a composed model initialises to exactly
    the weights its wrapper did.
    """

    def __init__(self, model_name, backbone_cls, plugin_cls, kwargs, width):
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone_cls(**_filter_to_signature(backbone_cls, kwargs))
        self.plugin = plugin_cls(
            **_filter_to_signature(plugin_cls, dict(kwargs, embedding_dim=width))
        )
        self._side: Optional[Dict[str, Any]] = None

    def __getattr__(self, name):
        """Fall through to the backbone, so trainers keep working unchanged.

        `AKTTrainer` reads `self.model.n_pid`, `SimpleKTTrainer` reads
        `self.model.num_c`. Those trainers now hold a `PluggedKT`, and requiring
        each of them to know about the wrapper would put the coupling back.

        The cost is that a genuine typo on a composed model reports the
        backbone's AttributeError rather than this class's.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("backbone", "plugin"):
                raise
            return getattr(super().__getattr__("backbone"), name)

    def take_side(self) -> Dict[str, Any]:
        """The plugin's output for the batch just run. Cleared by reading it."""
        if self._side is None:
            raise RuntimeError(
                f"{self.model_name}: no plugin output to read. Either forward() "
                "has not run for this batch, or something already consumed it."
            )
        side, self._side = self._side, None
        return side

    def forward(self, *args, **kwargs):
        batch = self.backbone.make_batch(*args, **kwargs)
        emb = self.backbone.embed(batch)
        emb, side = self.plugin.transform(emb, batch)
        hidden = self.backbone.encode(emb)
        preds = self.backbone.readout(hidden, emb)
        self._side = side
        return self.backbone.pack_output(preds, emb)


def register_plugged(
    name, backbone, plugin, width_key, width_default, spec=None
):
    """Register `name` as `backbone` + `plugin`.

    `width_key` names the constructor argument that sets the backbone's hidden
    width, because backbones disagree on what to call it (`emb_size`,
    `d_model`) and the plugin's own modules have to match it.
    """
    backbone_cls = MODEL_REGISTRY.get(backbone)

    class _Plugged(PluggedKT):
        #: The classes this one is made of. `__init__` takes `**kwargs` and
        #: splits them at runtime, so a reader of the signature alone -- which
        #: is what tests/test_config_keys_are_consumed.py is -- cannot otherwise
        #: tell which hyperparameters this model accepts.
        composed_of = (backbone_cls, plugin)

        def __init__(self, **kwargs):
            super().__init__(
                name,
                backbone_cls,
                plugin,
                kwargs,
                width=int(kwargs.get(width_key, width_default)),
            )

    _Plugged.__name__ = f"{plugin.__name__}_{backbone_cls.__name__}"
    _Plugged.__qualname__ = _Plugged.__name__
    _Plugged.__doc__ = (
        f"{backbone_cls.__name__} with {plugin.__name__} at the embed/encode "
        f"seam. Generated by models/plugin.py:register_plugged."
    )
    if spec is not None:
        _Plugged.Inputs = spec
    MODEL_REGISTRY.register(name)(_Plugged)
    return _Plugged
