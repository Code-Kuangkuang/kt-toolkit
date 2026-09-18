import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

from core.registry import MODEL_REGISTRY
from core.backbone import Embeddings, SeqBatch, infer_valid_mask
from .multi_concept import pool_interaction_embeddings


@MODEL_REGISTRY.register("dkt")
class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type="qid"):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = int(num_c)
        self.emb_type = emb_type
        self.emb_size = int(emb_size)
        self.hidden_size = int(emb_size)

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    # -- Stages. See core/backbone.py for why these exist. --

    def make_batch(self, q, r, item_data=None, valid_mask=None):
        """`item_data` is the question id when the caller has one to offer.

        DKT itself never uses it -- it embeds concepts -- but a plugin gated on
        item identity does, so it is carried rather than dropped.
        """
        if valid_mask is None:
            valid_mask = infer_valid_mask(q)
        return SeqBatch(
            concepts=q,
            responses=r,
            questions=item_data,
            valid_mask=valid_mask,
        )

    def embed(self, batch):
        # Test sequences can contain -1 (unknown future response); map to valid index range.
        r = batch.responses.long().clamp(min=0, max=1)
        xemb = pool_interaction_embeddings(
            self.interaction_emb, batch.concepts, r, self.num_c
        )
        return Embeddings(query=None, history=xemb)

    def encode(self, emb):
        h, _ = self.lstm_layer(emb.history)
        return self.dropout_layer(h)

    def readout(self, hidden, emb):
        return torch.sigmoid(self.out_layer(hidden))

    def pack_output(self, preds, emb):
        """What `forward` hands back, so a plugged variant returns the same
        shape to the same trainer."""
        return preds

    def forward(self, q, r, item_data=None, return_features=False):
        emb = self.embed(self.make_batch(q, r, item_data))
        h = self.encode(emb)
        y = self.readout(h, emb)

        if return_features:
            return {"preds": y, "hidden": h}
        return y
