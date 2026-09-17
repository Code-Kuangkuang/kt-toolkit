import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MODEL_REGISTRY.register("kqn")
class KQN(nn.Module):
    """Knowledge Quantization Network

    Args:
        num_c: number of skills/concepts
        n_hidden: dimensionality of skill and knowledge state vectors
        n_rnn_hidden: number of hidden units in rnn knowledge encoder
        n_mlp_hidden: number of hidden units in mlp skill encoder
        n_rnn_layers: number of layers in rnn knowledge encoder
        rnn_type: type of rnn cell, 'gru' or 'lstm'
        dropout: dropout probability
        emb_type: embedding type
    """

    def __init__(
        self,
        num_c,
        n_hidden=128,
        n_rnn_hidden=128,
        n_mlp_hidden=128,
        dropout=0.4,
        n_rnn_layers=1,
        rnn_type="lstm",
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        **kwargs,
    ):
        super(KQN, self).__init__()
        self.model_name = "kqn"
        self.emb_type = emb_type
        self.num_c = num_c
        self.n_hidden = n_hidden
        self.n_rnn_hidden = n_rnn_hidden
        self.n_mlp_hidden = n_mlp_hidden
        self.n_rnn_layers = n_rnn_layers
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout

        if emb_type.startswith("qid"):
            if self.rnn_type == "lstm":
                self.rnn = nn.LSTM(
                    input_size=2 * num_c,
                    hidden_size=n_rnn_hidden,
                    num_layers=n_rnn_layers,
                    batch_first=True,
                )
            elif self.rnn_type == "gru":
                self.rnn = nn.GRU(
                    input_size=2 * num_c,
                    hidden_size=n_rnn_hidden,
                    num_layers=n_rnn_layers,
                    batch_first=True,
                )

        self.linear = nn.Linear(n_rnn_hidden, n_hidden)

        self.skill_encoder = nn.Sequential(
            nn.Linear(num_c, n_mlp_hidden),
            nn.ReLU(),
            nn.Linear(n_mlp_hidden, n_hidden),
            nn.ReLU(),
        )
        self.drop_layer = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        # Register buffers for one-hot encoding
        self.register_buffer("two_eye", torch.eye(2 * num_c))
        self.register_buffer("eye", torch.eye(num_c))

    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        if self.rnn_type == "lstm":
            return (
                Variable(
                    weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()
                ),
                Variable(
                    weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()
                ),
            )
        else:
            return Variable(
                weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()
            )

    def forward(self, q, r, qshft=None, qtest=False, **kwargs):
        """Forward pass

        Args:
            q: question indices [batch_size, seq_len]
            r: response values [batch_size, seq_len]
            qshft: shifted question indices [batch_size, seq_len]
            qtest: if True, return additional tensors

        Returns:
            logits: predicted probabilities [batch_size, seq_len]
        """
        if qshft is None:
            raise ValueError("KQN.forward requires qshft.")

        q = q.long()
        qshft = qshft.long()
        r = r.long()

        if q.numel() > 0:
            # -1 is the dataset's padding marker and is masked out by the
            # pooling below, so only ids at or above num_c are out of range.
            q_max = int(q.max().item())
            qs_max = int(qshft.max().item())
            bad_low = int(q.min().item()) < -1 or int(qshft.min().item()) < -1
            if bad_low or q_max >= self.num_c or qs_max >= self.num_c:
                raise ValueError(
                    f"KQN ids out of range: q max {q_max}, qshft max {qs_max}, "
                    f"expected -1 (padding) or [0, {self.num_c - 1}]."
                )

        if r.numel() > 0:
            r_min = int(r.min().item())
            r_max = int(r.max().item())
            if r_min < 0 or r_max > 1:
                raise ValueError(f"KQN responses must be 0/1, got range [{r_min}, {r_max}].")

        # One-hot encoding r * num_c + q.  The lookup tables are identity
        # buffers rather than nn.Embedding, so they are wrapped to reuse the
        # same pooling the other models use: identity on [B,T], and on [B,T,K]
        # the mean of the question's K one-hot rows with -1 padding masked.
        # That mean is a soft multi-hot giving each KC weight 1/K, which is the
        # natural reading of "this question exercises K concepts" -- but note it
        # changes the input scale from 1 to 1/K, unlike a learned embedding
        # where pooling keeps the magnitude roughly constant.
        in_data = pool_interaction_embeddings(
            lambda ids: self.two_eye[ids], q, r, self.num_c
        )
        next_skills = pool_concept_embeddings(
            lambda ids: self.eye[ids], qshft, self.num_c
        )

        # Encode knowledge state using RNN
        encoded_knowledge = self.encode_knowledge(in_data)
        # Encode skills using MLP
        encoded_skills = self.encode_skills(next_skills)

        encoded_knowledge = self.drop_layer(encoded_knowledge)

        # Dot product for prediction
        logits = torch.sum(encoded_knowledge * encoded_skills, dim=2)
        logits = self.sigmoid(logits)

        if not qtest:
            return logits
        else:
            return logits, encoded_knowledge, encoded_skills

    def encode_knowledge(self, in_data):
        batch_size = in_data.size(0)
        self.hidden = self.init_hidden(batch_size)

        rnn_output, _ = self.rnn(in_data, self.hidden)
        encoded_knowledge = self.linear(rnn_output)
        return encoded_knowledge

    def encode_skills(self, next_skills):
        encoded_skills = self.skill_encoder(next_skills)
        encoded_skills = F.normalize(encoded_skills, p=2, dim=2)
        return encoded_skills
