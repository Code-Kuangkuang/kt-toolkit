"""Shared causal hybrid-denoising modules for HD-KT backbones."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalStateAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        hidden = F.relu(self.encoder(x))
        mu = self.mu(hidden)
        log_var = self.log_var(hidden).clamp(min=-10.0, max=10.0)
        if self.training:
            standard_deviation = torch.exp(0.5 * log_var)
            latent = mu + torch.randn_like(standard_deviation) * standard_deviation
        else:
            latent = mu
        reconstruction = torch.sigmoid(
            self.decoder(F.relu(self.decoder_hidden(latent)))
        )
        return reconstruction, mu, log_var


class HybridInteractionDenoiser(nn.Module):
    """Knowledge-state and student-profile anomaly detectors.

    The recurrent encoder is unidirectional and the profile at position ``t``
    is computed from positions strictly before ``t``.  Consequently the gate
    at ``t`` cannot depend on future interactions.
    """

    def __init__(
        self,
        num_items,
        num_c,
        embedding_dim,
        detector_hidden=None,
        latent_dim=None,
        dropout=0.1,
        gumbel_tau=1.0,
        hard_detection=True,
        kl_weight=0.001,
    ):
        super().__init__()
        if num_items <= 0 or num_c <= 0:
            raise ValueError("HD-KT denoiser requires positive item/concept counts.")
        self.num_items = int(num_items)
        self.num_c = int(num_c)
        self.embedding_dim = int(embedding_dim)
        self.gumbel_tau = float(gumbel_tau)
        self.hard_detection = bool(hard_detection)
        self.kl_weight = float(kl_weight)
        detector_hidden = int(detector_hidden or embedding_dim)
        latent_dim = int(latent_dim or max(8, embedding_dim // 2))

        self.item_embed = nn.Embedding(self.num_items + 1, embedding_dim)
        self.interaction_embed = nn.Embedding(2 * self.num_c, embedding_dim)
        self.state_encoder = nn.GRU(
            2 * embedding_dim, embedding_dim, batch_first=True
        )
        self.state_vae = VariationalStateAutoEncoder(
            embedding_dim, detector_hidden, latent_dim
        )
        self.state_detector = nn.Sequential(
            nn.Linear(3 * embedding_dim, detector_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(detector_hidden, 2),
        )
        self.profile_detector = nn.Sequential(
            nn.Linear(3 * embedding_dim, detector_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(detector_hidden, 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _interaction_embedding(self, concepts, responses):
        if concepts.dim() == 2:
            concepts = concepts.unsqueeze(-1)
        concept_mask = (concepts >= 0) & (concepts < self.num_c)
        safe_concepts = concepts.clamp(min=0, max=self.num_c - 1)
        interaction_ids = (
            safe_concepts
            + responses.long().clamp(min=0, max=1).unsqueeze(-1) * self.num_c
        )
        embeddings = self.interaction_embed(interaction_ids)
        embeddings = embeddings * concept_mask.unsqueeze(-1)
        denominator = concept_mask.sum(dim=2, keepdim=True).clamp(min=1)
        return embeddings.sum(dim=2) / denominator

    def _anomaly_probability(self, logits):
        if self.training and self.hard_detection:
            return F.gumbel_softmax(
                logits, tau=self.gumbel_tau, hard=True, dim=-1
            )[..., 1]
        probabilities = torch.softmax(logits, dim=-1)
        if self.hard_detection:
            return (probabilities.argmax(dim=-1) == 1).to(probabilities.dtype)
        return probabilities[..., 1]

    def forward(self, items, concepts, responses, valid_mask=None):
        if items.dim() != 2 or responses.dim() != 2:
            raise ValueError("HD-KT items and responses must have shape [B,T].")
        if concepts.dim() not in (2, 3):
            raise ValueError("HD-KT concepts must have shape [B,T] or [B,T,K].")
        if items.shape != responses.shape or concepts.shape[:2] != items.shape:
            raise ValueError("HD-KT denoiser input sequences are not aligned.")
        if valid_mask is None:
            valid_mask = torch.ones_like(items, dtype=torch.bool)
        valid_mask = valid_mask.bool()
        if valid_mask.shape != items.shape:
            raise ValueError("HD-KT valid mask is not aligned with inputs.")

        valid_items = items[valid_mask]
        if valid_items.numel() and (
            int(valid_items.min().item()) < 0
            or int(valid_items.max().item()) > self.num_items
        ):
            raise ValueError("HD-KT denoiser encountered an out-of-range item id.")
        concept_time_mask = valid_mask
        if concepts.dim() == 3:
            concept_time_mask = valid_mask.unsqueeze(-1).expand_as(concepts)
        valid_concepts = concepts[concept_time_mask & (concepts >= 0)]
        if valid_concepts.numel() and int(valid_concepts.max().item()) >= self.num_c:
            raise ValueError("HD-KT denoiser encountered an out-of-range concept id.")
        valid_responses = responses[valid_mask]
        if valid_responses.numel() and not torch.all(
            (valid_responses == 0) | (valid_responses == 1)
        ).item():
            raise ValueError("HD-KT responses must be binary on valid positions.")

        safe_items = torch.where(valid_mask, items, torch.zeros_like(items))
        item_features = self.item_embed(safe_items)
        interaction_features = self._interaction_embedding(concepts, responses)
        sequence_features = torch.cat(
            (item_features, interaction_features), dim=-1
        ) * valid_mask.unsqueeze(-1)

        encoded_state, _ = self.state_encoder(sequence_features)
        reconstruction, mu, log_var = self.state_vae(encoded_state)
        state_logits = self.state_detector(
            torch.cat(
                (
                    encoded_state,
                    reconstruction,
                    (encoded_state - reconstruction).abs(),
                ),
                dim=-1,
            )
        )

        weighted_state = encoded_state * valid_mask.unsqueeze(-1)
        inclusive_sum = torch.cumsum(weighted_state, dim=1)
        inclusive_count = torch.cumsum(valid_mask.to(encoded_state.dtype), dim=1)
        previous_sum = inclusive_sum - weighted_state
        previous_count = inclusive_count - valid_mask.to(encoded_state.dtype)
        profile = previous_sum / previous_count.clamp(min=1.0).unsqueeze(-1)
        profile = profile * (previous_count > 0).unsqueeze(-1)
        profile_logits = self.profile_detector(
            torch.cat((profile, encoded_state, item_features), dim=-1)
        )

        state_anomaly = self._anomaly_probability(state_logits)
        profile_anomaly = self._anomaly_probability(profile_logits)
        gate = (1.0 - state_anomaly * profile_anomaly) * valid_mask

        reconstruction_error = (reconstruction - encoded_state).pow(2).mean(dim=-1)
        kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1)
        valid_count = valid_mask.sum().clamp(min=1).to(encoded_state.dtype)
        reconstruction_loss = (
            (reconstruction_error + self.kl_weight * kl)
            * valid_mask.to(encoded_state.dtype)
        ).sum() / valid_count
        return {
            "gate": gate,
            "state_anomaly": state_anomaly * valid_mask,
            "profile_anomaly": profile_anomaly * valid_mask,
            "reconstruction_loss": reconstruction_loss,
        }

