# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODEL_REGISTRY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MODEL_REGISTRY.register("atkt")
class ATKT(nn.Module):
    """Attention-based Knowledge Tracing

    Args:
        num_c: number of skills/concepts
        skill_dim: embedding dimension for skills
        answer_dim: embedding dimension for answers
        hidden_dim: LSTM hidden dimension
        attention_dim: attention dimension
        epsilon: epsilon for adversarial perturbation
        beta: beta parameter
        dropout: dropout probability
        emb_type: embedding type
        fix: whether to use fixed attention (atktfix mode)
    """

    def __init__(
        self,
        num_c,
        skill_dim=256,
        answer_dim=96,
        hidden_dim=80,
        attention_dim=80,
        epsilon=10,
        beta=0.2,
        dropout=0.2,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        fix=True,
        **kwargs,
    ):
        super(ATKT, self).__init__()
        self.model_name = "atktfix" if fix else "atkt"
        self.emb_type = emb_type
        self.skill_dim = skill_dim
        self.answer_dim = answer_dim
        self.hidden_dim = hidden_dim
        self.num_c = num_c
        self.epsilon = epsilon
        self.beta = beta
        self.fix = fix

        # LSTM for knowledge state encoding
        self.rnn = nn.LSTM(
            self.skill_dim + self.answer_dim,
            self.hidden_dim,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_c)
        self.sig = nn.Sigmoid()

        # Skill and answer embeddings
        self.skill_emb = nn.Embedding(self.num_c + 1, self.skill_dim, padding_idx=self.num_c)
        self.answer_emb = nn.Embedding(2 + 1, self.answer_dim, padding_idx=2)

        # Attention module
        self.attention_dim = attention_dim
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

    def attention_module(self, lstm_output):
        """Attention module to aggregate historical knowledge states."""
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)

        if self.fix:
            # Fixed attention with causal mask
            attn_mask = ut_mask(lstm_output.shape[1])
            att_w = att_w.transpose(1, 2).expand(
                lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]
            ).clone()
            att_w = att_w.masked_fill(attn_mask, float("-inf"))
            alphas = F.softmax(att_w, dim=-1)
            attn_output = torch.bmm(alphas, lstm_output)
        else:
            # Original attention (non-causal)
            alphas = F.softmax(att_w, dim=1)
            attn_output = alphas * lstm_output

        # Cumulative attention
        attn_output_cum = torch.cumsum(attn_output, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_output

        # Concatenate cumulative attention and current output
        final_output = torch.cat((attn_output_cum_1, lstm_output), dim=2)
        return final_output

    def forward(self, skill, answer, perturbation=None, **kwargs):
        """Forward pass

        Args:
            skill: skill/question indices [batch_size, seq_len]
            answer: answer values [batch_size, seq_len]
            perturbation: adversarial perturbation (optional)

        Returns:
            res: predicted probabilities [batch_size, seq_len, num_c]
            skill_answer_embedding: embeddings for loss calculation
        """
        r = answer.long()

        # Get embeddings
        skill_embedding = self.skill_emb(skill)
        answer_embedding = self.answer_emb(answer)

        # Concatenate skill and answer embeddings in two ways
        skill_answer = torch.cat((skill_embedding, answer_embedding), dim=2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), dim=2)

        # Replace based on answer value
        answer = answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)

        # Add perturbation if provided
        if perturbation is not None:
            skill_answer_embedding += perturbation

        # LSTM forward
        out, _ = self.rnn(skill_answer_embedding)

        # Attention module
        out = self.attention_module(out)

        # Prediction
        res = self.sig(self.fc(self.dropout_layer(out)))

        return res, skill_answer_embedding


def ut_mask(seq_len):
    """Upper triangular mask for causal attention."""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(device)
