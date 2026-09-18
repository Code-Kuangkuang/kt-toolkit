"""The one place a plugin is allowed to reach into a backbone.

`models/hd_akt.py` used to carry a line-by-line copy of `AKT.forward` so that it
could insert a single multiplication into the middle of it, and `hd_simplekt.py`
did the same to SimpleKT. A copy is worse than coupling: when the original is
fixed, the copy keeps the bug and nothing says so. That already happened --
`HDSimpleKTTrainer` silently dropped SimpleKT's item L2 penalty, and all three
HD trainers computed BCE in float32 while their own baselines used float64, so
every HD-vs-baseline comparison carried a confound that had nothing to do with
denoising.

So the copies are gone and each backbone names its own stages instead:

    batch  = model.make_batch(**whatever its forward takes)
    emb    = model.embed(batch)        # ids -> query / history representations
    hidden = model.encode(emb)         # the sequence model
    preds  = model.readout(hidden, emb)

`forward` still exists and still takes exactly the arguments it always took, so
the 34 existing trainers are untouched; it just calls the four stages in order.
The split is pure code motion -- no arithmetic moved, no argument changed --
which is what makes it checkable: the same input must produce bit-identical
output before and after.

The seam that matters is between `embed` and `encode`. HD-KT gates the
response-carrying history there; DKT, AKT and SimpleKT all happen to have that
exact boundary already (`lstm_layer(interaction_embedding)`,
`model(q_embed, qa_embed, ...)`), because that boundary is what the method is
defined on. Naming it is the whole refactor.

A backbone that no plugin targets does not need any of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class SeqBatch:
    """Full-length sequences, in the one layout every backbone agrees on.

    "Full-length" means position `t` covers t=0..T-1, not the shifted
    `(seqs, shft_seqs)` pair the loaders hand out. AKT and SimpleKT already
    rebuild this internally; DKT is already in it. A plugin reads ids from here
    rather than guessing which of a backbone's arguments held them.
    """

    #: [B,T] or [B,T,K] concept ids, -1 padded.
    concepts: torch.Tensor
    #: [B,T] responses. May contain -1 for an unknown future response.
    responses: torch.Tensor
    #: [B,T] question ids, or None when the dataset has none.
    questions: Optional[torch.Tensor] = None
    #: [B,T] bool, True where the position is real rather than padding.
    valid_mask: Optional[torch.Tensor] = None

    def items(self) -> torch.Tensor:
        """Question ids when the dataset has them, concept ids otherwise.

        What the HD wrappers each spelled out as
        `concepts if pid_data is None else pid_data`.
        """
        if self.questions is not None:
            return self.questions.long()
        concepts = self.concepts
        if concepts.dim() == 3:
            # [B,T,K] has no single item id; the first listed concept stands in,
            # matching what the wrappers passed when questions were absent.
            concepts = concepts[..., 0]
        return concepts.long()


@dataclass
class Embeddings:
    """What `embed` produced, and what `encode`/`readout` will consume.

    `history` is the tensor a plugin may replace. `query` and `extras` are the
    backbone's own business; a plugin that writes to them is reaching past the
    seam and should be a separate backbone instead.
    """

    #: What the model is asked about at each position. None for DKT, which
    #: scores every concept from the hidden state rather than a query.
    query: Optional[torch.Tensor]
    #: The response-carrying representation fed to the sequence encoder.
    history: torch.Tensor
    #: Backbone-private values that `encode` or the loss needs, e.g. AKT's
    #: `pid_embed` and its Rasch regularisation term.
    extras: Dict[str, Any] = field(default_factory=dict)


def infer_valid_mask(concepts: torch.Tensor, pad_val: int = -1) -> torch.Tensor:
    """[B,T] bool: positions that are real rather than padding.

    The HD trainers previously built this as
    `cat((masks[:, :1], masks), dim=1)` from the loader's `masks`, which is
    `valid(t) & valid(t+1)` shifted by one -- so position `t` reported
    `valid(t-1) & valid(t)`. Padding in this codebase is a suffix
    (`datasets/kt_dataset.py` pads at the end), so the two agree everywhere
    except on length-1 sequences, which `min_seq_len >= 3` already excludes.
    Deriving it from the ids directly removes the off-by-one and the need for
    every caller to remember the concatenation.

    tests/test_plugin_composition.py pins the agreement on padded input.
    """
    if concepts.dim() == 3:
        return (concepts != pad_val).any(dim=-1)
    return concepts != pad_val
