import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

from core.registry import MODEL_REGISTRY


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

    def forward(self, q, r):
        q = q.long().clamp(min=0, max=self.num_c - 1)
        # Test sequences can contain -1 (unknown future response); map to valid index range.
        r = r.long().clamp(min=0, max=1)
        x = q + self.num_c * r
        xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y