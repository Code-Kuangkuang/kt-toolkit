# coding: utf-8
import numpy as np
import torch
import torch.nn as nn

from core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("hawkes")
class HawkesKT(nn.Module):
    """Hawkes Process based Knowledge Tracing.

    Args:
        n_skills: number of skills/concepts
        n_problems: number of questions/problems
        emb_size: embedding dimension
        time_log: time log base for delta_t calculation
        emb_type: embedding type
    """

    def __init__(
        self,
        num_c,
        num_q,
        emb_size=64,
        time_log=5,
        emb_type="qid",
        **kwargs,
    ):
        super().__init__()
        self.model_name = "hawkes"
        self.emb_type = emb_type
        self.problem_num = num_q
        self.skill_num = num_c
        self.emb_size = emb_size
        self.time_log = time_log
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Bias embeddings
        self.problem_base = nn.Embedding(self.problem_num, 1)
        self.skill_base = nn.Embedding(self.skill_num, 1)

        # Hawkes process parameters
        self.alpha_inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = nn.Embedding(self.skill_num, self.emb_size)

        # Cast to double for numerical precision with timestamps
        self.double()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, skills, problems, times, labels, qtest=False):
        """Forward pass.

        Args:
            skills: skill/concept sequences [batch_size, seq_len]
            problems: question sequences [batch_size, seq_len]
            times: timestamp sequences [batch_size, seq_len]
            labels: answer sequences [batch_size, seq_len]
            qtest: whether to return hidden states

        Returns:
            predictions: predicted probabilities
            (optional) h: hidden states
        """
        # Embedding indices must be integer type.
        skills = skills.long()
        problems = problems.long()
        mask_labels = labels.long().clamp(min=0, max=1)

        # Compute interaction indices: skill + label * skill_num
        inters = skills + mask_labels * self.skill_num

        # Alpha: interaction strength
        alpha_src_emb = self.alpha_inter_embeddings(inters)
        alpha_target_emb = self.alpha_skill_embeddings(skills)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))

        # Beta: decay rate
        beta_src_emb = self.beta_inter_embeddings(inters)
        beta_target_emb = self.beta_skill_embeddings(skills)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))
        betas = torch.clamp(betas + 1, min=0, max=10)

        # Time delta calculation
        if times is not None and times.numel() > 0:
            times = times.double() / 1000
            delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
        else:
            delta_t = torch.ones(
                skills.shape[0],
                skills.shape[1],
                skills.shape[1],
                dtype=torch.double,
                device=skills.device,
            )

        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        # Hawkes cross effects
        cross_effects = alphas * torch.exp(-betas * delta_t)

        # Upper triangular mask (causal - only past affects present)
        seq_len = skills.shape[1]
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0)
        mask = mask.to(skills.device)
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        # Bias terms
        problem_bias = self.problem_base(problems).squeeze(dim=-1)
        skill_bias = self.skill_base(skills).squeeze(dim=-1)

        # Final prediction
        prediction = (problem_bias + skill_bias + sum_t).sigmoid()
        h = problem_bias + skill_bias + sum_t

        if not qtest:
            return prediction
        else:
            return prediction, h
