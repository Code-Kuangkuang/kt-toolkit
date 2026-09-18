"""One mixin instead of one trainer per (plugin, backbone) pairing.

The three hand-written HD trainers each re-implemented their base trainer's
`_forward_batch` in order to add one term to the loss, and each drifted from it
in the process:

* all three computed BCE in float32 while `dkt`, `akt` and `simplekt` compute it
  in float64, so every HD-vs-baseline comparison differed in loss precision for
  reasons unrelated to denoising;
* `HDSimpleKTTrainer` dropped SimpleKT's item L2 penalty entirely;
* `HDAKTTrainer` added AKT's Rasch regularisation outside `cal_loss` rather than
  through its `preloss` argument.

None of that was intended, and none of it was visible without diffing the files
side by side. The mixin calls the base trainer instead, so the only difference
between a plugged run and its baseline is the plugin.

This does change the numbers the HD models produce relative to what they
produced before -- it removes a confound rather than preserving one.
"""

import torch

from core.registry import TRAINER_REGISTRY
from core.trainers.akt_trainer import AKTTrainer
from core.trainers.dkt_trainer import DKTTrainer
from core.trainers.simplekt_trainer import SimpleKTTrainer


class PluginTrainer:
    """Mix in front of a backbone's trainer: `class X(PluginTrainer, YTrainer)`."""

    def _forward_batch(self, batch):
        result = super()._forward_batch(batch)
        extra = self.model.plugin.extra_loss(self.model.take_side())
        if extra is None:
            return result
        loss = result[-1] + extra
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"{self.model.model_name} produced a non-finite loss."
            )
        return (*result[:-1], loss)


@TRAINER_REGISTRY.register("hd_dkt")
class HDDKTTrainer(PluginTrainer, DKTTrainer):
    pass


@TRAINER_REGISTRY.register("hd_akt")
class HDAKTTrainer(PluginTrainer, AKTTrainer):
    pass


@TRAINER_REGISTRY.register("hd_simplekt")
class HDSimpleKTTrainer(PluginTrainer, SimpleKTTrainer):
    pass
