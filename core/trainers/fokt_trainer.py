"""AKT's trainer plus the one extra tensor FoKT needs: raw timestamps.

Everything about the loss, the alignment and the scored positions is inherited
unchanged, which is the point -- the comparison between `fokt` and `akt` has to
differ in the decay term and nothing else. The HD trainers are the cautionary
tale here: they recomputed BCE in float32 while their own baselines used
float64, and that confound survived unnoticed through a whole results table
(see core/backbone.py).

`tseqs` rather than `itseqs`: the loader floors `itseqs` to whole minutes
(`datasets/kt_dataset.py::_build_itseqs`), which would erase every sub-minute
gap -- and on algebra2005 the median gap between consecutive interactions is 16
seconds. `tseqs` is int64 milliseconds and keeps full precision.
"""

import torch

from core.registry import TRAINER_REGISTRY
from .akt_trainer import AKTTrainer, cal_loss
from models.multi_concept import concept_validity


@TRAINER_REGISTRY.register("fokt")
class FoKTTrainer(AKTTrainer):
    def _forward_batch(self, batch):
        qseqs = self._optional(batch, "qseqs")
        cseqs = self._optional(batch, "cseqs")
        rseqs = batch["rseqs"].to(self.device)
        qshft = self._optional(batch, "shft_qseqs")
        cshft = self._optional(batch, "shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)

        if c_full is not None:
            q_data, pid_data = c_full, q_full
        else:
            q_data, pid_data = q_full, None
        if q_data is None:
            raise ValueError("FoKTTrainer requires concept or question sequences.")

        target_concepts = cshft if cshft is not None else qshft
        _, target_has_concept = concept_validity(target_concepts, self.model.n_question)
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError("FoKT found a scored question without a valid concept id.")
        if pid_data is None and getattr(self.model, "n_pid", 0) > 0:
            raise ValueError("FoKTTrainer requires question ids when n_pid > 0.")

        t_full = None
        if getattr(self.model, "use_time", False):
            tseqs = self._optional(batch, "tseqs")
            tshft = self._optional(batch, "shft_tseqs")
            t_full = self._concat_full(tseqs, tshft)
            if t_full is None:
                raise ValueError(
                    "FoKT's time-based arms need a `timestamps` column; this "
                    "dataset has none. assist2009 is the known case -- run a "
                    "`_notime` arm there instead of silently decaying nothing."
                )
            # A whole batch of zeros means the column exists but is unusable,
            # which would make `dt` identically zero and turn the time arm into
            # the notime arm without saying so.
            if not bool((t_full > 0).any()):
                raise ValueError(
                    "FoKT got an all-zero `timestamps` batch; the column is "
                    "present but carries no usable epochs."
                )

        preds, reg_loss = self.model(
            q_data.long(),
            r_full.long(),
            None if pid_data is None else pid_data.long(),
            t_data=t_full,
        )

        preds = preds[:, 1:]
        loss = cal_loss(
            self.model, [preds], rseqs, rshft, sm,
            preloss=[reg_loss] if reg_loss is not None else [],
        )
        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, reg_loss, loss
