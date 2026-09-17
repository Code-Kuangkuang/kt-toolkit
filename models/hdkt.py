# coding: utf-8
"""Causal HD-KT with an LPKT prediction backbone.

The implementation follows the two-detector design of HD-KT (WWW 2024): a
knowledge-state detector based on a sequential variational autoencoder and a
student-profile detector.  Unlike the released reference code, both detectors
are causal and the next-response label is never used to gate the corresponding
prediction.

Reference implementation:
https://github.com/BIMK/Intelligent-Education/tree/main/HD-KT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY


class _VariationalAutoEncoder(nn.Module):
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
            std = torch.exp(0.5 * log_var)
            latent = mu + torch.randn_like(std) * std
        else:
            # Deterministic validation/test metrics.
            latent = mu
        reconstruction = torch.sigmoid(
            self.decoder(F.relu(self.decoder_hidden(latent)))
        )
        return reconstruction, mu, log_var


@MODEL_REGISTRY.register("hdkt")
class HDKT(nn.Module):
    class Inputs(InputSpec):
        """Builds the answer-time and interval-time vocabularies.

        LPKT bins a raw duration into an index and embeds it, so `num_at` and
        `num_it` are table heights. `always_train_folds` is the difference
        between the two models: HDKT always fits on the current fold's training
        rows, while LPKT does so only under all_in_one -- `folds=None` in
        one_by_one reads every split, which is the historical pyKT behaviour and
        is recorded as such rather than silently changed.
        """

        dataset_mode = "all_in_one"
        requires_question_ids = True
        always_train_folds = True

        @classmethod
        def prepare(cls, ctx):
            from datasets.lpkt_utils import generate_time2idx

            fold_scoped = cls.always_train_folds or ctx.dataset_mode == "all_in_one"
            folds = ctx.train_folds() if fold_scoped else None
            at2idx, it2idx = generate_time2idx(ctx.dataset_cfg, folds=folds)

            sizes = {"num_at": len(at2idx) + 1, "num_it": len(it2idx) + 1}
            dataset_updates = dict(sizes)
            if fold_scoped:
                dataset_updates["time_index_scope"] = "train_folds_only"
                dataset_updates["time_index_folds"] = folds

            inputs = ModelInputs(
                model_cfg_updates=dict(sizes),
                dataset_cfg_updates=dataset_updates,
                dataset_kwargs={"time_idx_maps": {"at2idx": at2idx, "it2idx": it2idx}},
                feature_fit_scope="train_folds" if fold_scoped else "train_valid_test",
            )
            cls.extend(inputs, ctx)
            return inputs

        @classmethod
        def extend(cls, inputs, ctx):
            """Hook for the subclass-specific part. Nothing shared to add."""

    """Hybrid interaction denoising for knowledge tracing.

    This repository implementation uses LPKT as the KT backbone, matching the
    public HD-LPKT code.  The student profile is computed from the causal
    history instead of a dataset-specific student-id embedding, so validation
    and test students do not require entries learned from another split.

    Tensor flow:
        questions       [B, T]
        concepts        [B, T] or [B, T, K]
        responses       [B, T]
        predictions     [B, T] (position t predicts the response at t)
        anomaly gates   [B, T] (gate t is only used to update history at t)
    """

    def __init__(
        self,
        num_at=100,
        num_it=100,
        num_q=100,
        num_c=10,
        d_a=64,
        d_e=64,
        d_k=64,
        dropout=0.2,
        detector_hidden=None,
        latent_dim=None,
        gumbel_tau=1.0,
        hard_detection=True,
        reconstruction_weight=0.01,
        kl_weight=0.001,
        use_time=True,
        emb_type="qid",
        **kwargs,
    ):
        super().__init__()
        if num_q <= 0:
            raise ValueError("HDKT requires question ids (num_q must be positive).")
        if num_c <= 0:
            raise ValueError("HDKT requires concept ids (num_c must be positive).")
        if min(d_a, d_e, d_k) <= 0:
            raise ValueError("HDKT embedding dimensions must be positive.")

        self.model_name = "hdkt"
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.d_a = int(d_a)
        self.d_e = int(d_e)
        self.d_k = int(d_k)
        self.use_time = bool(use_time)
        self.emb_type = emb_type
        self.gumbel_tau = float(gumbel_tau)
        self.hard_detection = bool(hard_detection)
        self.reconstruction_weight = float(reconstruction_weight)
        self.kl_weight = float(kl_weight)

        detector_hidden = int(detector_hidden or d_k)
        latent_dim = int(latent_dim or max(8, d_k // 2))

        # LPKT embeddings.
        self.at_embed = nn.Embedding(int(num_at) + 10, d_k)
        self.it_embed = nn.Embedding(int(num_it) + 10, d_k)
        self.e_embed = nn.Embedding(int(num_q) + 10, d_e)

        # HD-KT interaction representation: concept-response + exercise.
        self.interaction_embed = nn.Embedding(2 * int(num_c), d_k)
        self.exercise_projection = nn.Linear(d_e, d_k)
        self.state_encoder = nn.GRU(2 * d_k, d_k, batch_first=True)
        self.state_vae = _VariationalAutoEncoder(
            d_k, detector_hidden, latent_dim
        )
        self.state_detector = nn.Sequential(
            nn.Linear(3 * d_k, detector_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(detector_hidden, 2),
        )
        self.profile_detector = nn.Sequential(
            nn.Linear(3 * d_k, detector_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(detector_hidden, 2),
        )

        # LPKT learning, forgetting and prediction modules.
        self.linear_0 = nn.Linear(d_a + d_e, d_k)
        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        self.linear_6 = nn.Linear(3 * d_k, d_k)
        self.linear_7 = nn.Linear(3 * d_k, d_k)
        self.linear_8 = nn.Linear(2 * d_k, d_k)
        self.dropout = nn.Dropout(dropout)
        self.initial_knowledge = nn.Parameter(torch.empty(num_c, d_k))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.initial_knowledge)

    def _concept_weights(self, concept_data, valid_mask):
        """Build a safe multi-hot concept tensor [B, T, num_c]."""
        if concept_data.dim() == 2:
            concept_data = concept_data.unsqueeze(-1)
        if concept_data.dim() != 3:
            raise ValueError(
                "HDKT concepts must have shape [B,T] or [B,T,K], got "
                f"{tuple(concept_data.shape)}."
            )
        concept_valid = (concept_data >= 0) & (concept_data < self.num_c)
        safe_concepts = concept_data.clamp(min=0, max=self.num_c - 1)
        one_hot = F.one_hot(safe_concepts, num_classes=self.num_c)
        weights = (one_hot * concept_valid.unsqueeze(-1)).sum(dim=2)
        weights = weights.clamp(max=1).to(self.initial_knowledge.dtype)
        return weights * valid_mask.unsqueeze(-1).to(weights.dtype)

    def _interaction_features(self, exercise_data, concept_data, responses):
        if concept_data.dim() == 2:
            concept_data = concept_data.unsqueeze(-1)
        concept_valid = (concept_data >= 0) & (concept_data < self.num_c)
        safe_concepts = concept_data.clamp(min=0, max=self.num_c - 1)
        safe_responses = responses.long().clamp(min=0, max=1)
        interaction_ids = safe_concepts + safe_responses.unsqueeze(-1) * self.num_c
        concept_response = self.interaction_embed(interaction_ids)
        concept_response = concept_response * concept_valid.unsqueeze(-1)
        concept_count = concept_valid.sum(dim=2, keepdim=True).clamp(min=1)
        concept_response = concept_response.sum(dim=2) / concept_count
        exercise = self.exercise_projection(self.e_embed(exercise_data))
        return torch.cat((exercise, concept_response), dim=-1)

    def _anomaly_probability(self, logits):
        if self.training and self.hard_detection:
            return F.gumbel_softmax(
                logits, tau=self.gumbel_tau, hard=True, dim=-1
            )[..., 1]
        probabilities = torch.softmax(logits, dim=-1)
        if self.hard_detection:
            hard = F.one_hot(probabilities.argmax(dim=-1), num_classes=2)
            return hard.to(probabilities.dtype)[..., 1]
        return probabilities[..., 1]

    def _detect_anomalies(
        self, exercise_data, concept_data, responses, valid_mask
    ):
        item_features = self._interaction_features(
            exercise_data, concept_data, responses
        )
        item_features = item_features * valid_mask.unsqueeze(-1)

        # The unidirectional GRU makes the knowledge-state detector causal.
        encoded_state, _ = self.state_encoder(item_features)
        reconstruction, mu, log_var = self.state_vae(encoded_state)
        state_input = torch.cat(
            (encoded_state, reconstruction, (encoded_state - reconstruction).abs()),
            dim=-1,
        )
        state_logits = self.state_detector(state_input)

        # Long-term profile before t; it contains no current or future response.
        weighted_state = encoded_state * valid_mask.unsqueeze(-1)
        inclusive_sum = torch.cumsum(weighted_state, dim=1)
        inclusive_count = torch.cumsum(valid_mask.to(encoded_state.dtype), dim=1)
        previous_sum = inclusive_sum - weighted_state
        previous_count = inclusive_count - valid_mask.to(encoded_state.dtype)
        profile = previous_sum / previous_count.clamp(min=1.0).unsqueeze(-1)
        profile = profile * (previous_count > 0).unsqueeze(-1)
        profile_logits = self.profile_detector(
            torch.cat((profile, encoded_state, item_features[..., : self.d_k]), dim=-1)
        )

        state_anomaly = self._anomaly_probability(state_logits)
        profile_anomaly = self._anomaly_probability(profile_logits)
        gate = 1.0 - state_anomaly * profile_anomaly
        gate = gate * valid_mask.to(gate.dtype)

        reconstruction_error = (reconstruction - encoded_state).pow(2).mean(dim=-1)
        kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean(dim=-1)
        valid_count = valid_mask.sum().clamp(min=1).to(encoded_state.dtype)
        reconstruction_loss = (
            (reconstruction_error + self.kl_weight * kl)
            * valid_mask.to(encoded_state.dtype)
        ).sum() / valid_count

        return gate, state_anomaly, profile_anomaly, reconstruction_loss

    def _read_knowledge(self, concept_weights, knowledge_state):
        denominator = concept_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return torch.bmm(
            concept_weights.unsqueeze(1), knowledge_state
        ).squeeze(1) / denominator

    def forward(
        self,
        exercise_data,
        concept_data,
        responses,
        it_data=None,
        at_data=None,
        valid_mask=None,
        return_details=False,
    ):
        if exercise_data.dim() != 2 or responses.dim() != 2:
            raise ValueError("HDKT exercise_data and responses must be [B,T].")
        if concept_data.dim() not in (2, 3):
            raise ValueError("HDKT concept_data must be [B,T] or [B,T,K].")
        if exercise_data.shape != responses.shape:
            raise ValueError("HDKT exercise and response sequence shapes must match.")
        if concept_data.shape[:2] != exercise_data.shape:
            raise ValueError("HDKT concept and exercise sequence shapes must align.")
        if valid_mask is None:
            valid_mask = torch.ones_like(exercise_data, dtype=torch.bool)
        valid_mask = valid_mask.bool()
        if valid_mask.shape != exercise_data.shape:
            raise ValueError("HDKT valid_mask must align with exercise_data.")

        valid_questions = exercise_data[valid_mask]
        if valid_questions.numel() and (
            int(valid_questions.min().item()) < 0
            or int(valid_questions.max().item()) >= self.e_embed.num_embeddings
        ):
            raise ValueError("HDKT encountered an out-of-range question id.")
        expanded_valid_mask = valid_mask
        if concept_data.dim() == 3:
            expanded_valid_mask = valid_mask.unsqueeze(-1).expand_as(concept_data)
        valid_concepts = concept_data[expanded_valid_mask & (concept_data >= 0)]
        if valid_concepts.numel() and int(valid_concepts.max().item()) >= self.num_c:
            raise ValueError("HDKT encountered an out-of-range concept id.")
        valid_responses = responses[valid_mask]
        if valid_responses.numel() and not torch.all(
            (valid_responses == 0) | (valid_responses == 1)
        ).item():
            raise ValueError("HDKT responses must be binary on valid positions.")

        if self.use_time and it_data is None:
            raise ValueError("HDKT requires interval-time indices when use_time=True.")

        # Dataset padding is represented by -1 before collation in some paths.
        # Map only invalid positions to safe embedding indices.
        exercise_data = torch.where(
            valid_mask, exercise_data, torch.zeros_like(exercise_data)
        )
        if it_data is not None:
            it_data = torch.where(valid_mask, it_data, torch.zeros_like(it_data))
        if at_data is not None:
            at_data = torch.where(valid_mask, at_data, torch.zeros_like(at_data))

        batch_size, seq_len = exercise_data.shape
        exercise_embeddings = self.e_embed(exercise_data)
        concept_weights = self._concept_weights(concept_data, valid_mask)
        gate, state_anomaly, profile_anomaly, reconstruction_loss = (
            self._detect_anomalies(
                exercise_data, concept_data, responses, valid_mask
            )
        )

        denoised_exercise = exercise_embeddings * gate.unsqueeze(-1)
        response_features = responses.to(exercise_embeddings.dtype).unsqueeze(-1)
        response_features = response_features.expand(-1, -1, self.d_a)

        if self.use_time:
            interval_embeddings = self.it_embed(it_data) * gate.unsqueeze(-1)
            if at_data is None:
                answer_time_embeddings = torch.zeros(
                    batch_size,
                    seq_len,
                    self.d_k,
                    device=exercise_data.device,
                    dtype=exercise_embeddings.dtype,
                )
            else:
                answer_time_embeddings = self.at_embed(at_data) * gate.unsqueeze(-1)
            learning_sequence = self.linear_1(
                torch.cat(
                    (denoised_exercise, answer_time_embeddings, response_features),
                    dim=-1,
                )
            )
        else:
            interval_embeddings = None
            learning_sequence = self.linear_0(
                torch.cat((denoised_exercise, response_features), dim=-1)
            )

        knowledge_state = self.initial_knowledge.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        learning_previous = torch.zeros(
            batch_size, self.d_k, device=exercise_data.device
        )
        previous_readout = self._read_knowledge(
            concept_weights[:, 0], knowledge_state
        )
        predictions = [torch.zeros(batch_size, device=exercise_data.device)]
        readouts = [previous_readout]

        for t in range(seq_len - 1):
            current_concepts = concept_weights[:, t]
            current_learning = learning_sequence[:, t]
            current_active = valid_mask[:, t].unsqueeze(-1)

            if self.use_time:
                interval = interval_embeddings[:, t]
                learning_context = torch.cat(
                    (
                        learning_previous,
                        interval,
                        current_learning,
                        previous_readout,
                    ),
                    dim=-1,
                )
                learning_gain = torch.tanh(self.linear_2(learning_context))
                learning_gate = torch.sigmoid(self.linear_3(learning_context))
            else:
                learning_context = torch.cat(
                    (learning_previous, current_learning, previous_readout), dim=-1
                )
                learning_gain = torch.tanh(self.linear_6(learning_context))
                learning_gate = torch.sigmoid(self.linear_7(learning_context))

            learning_gain = learning_gate * ((learning_gain + 1.0) / 2.0)
            distributed_gain = self.dropout(
                current_concepts.unsqueeze(-1) * learning_gain.unsqueeze(1)
            )
            repeated_gain = learning_gain.unsqueeze(1).expand(
                -1, self.num_c, -1
            )
            if self.use_time:
                repeated_interval = interval.unsqueeze(1).expand(
                    -1, self.num_c, -1
                )
                forgetting_gate = torch.sigmoid(
                    self.linear_4(
                        torch.cat(
                            (knowledge_state, repeated_gain, repeated_interval),
                            dim=-1,
                        )
                    )
                )
            else:
                forgetting_gate = torch.sigmoid(
                    self.linear_8(
                        torch.cat((knowledge_state, repeated_gain), dim=-1)
                    )
                )
            candidate_state = distributed_gain + forgetting_gate * knowledge_state
            knowledge_state = torch.where(
                current_active.unsqueeze(-1), candidate_state, knowledge_state
            )

            next_readout = self._read_knowledge(
                concept_weights[:, t + 1], knowledge_state
            )
            # Raw next-exercise embedding is used here.  Its denoising gate would
            # depend on response[t+1] and would therefore leak the target label.
            next_prediction = torch.sigmoid(
                self.linear_5(
                    torch.cat((exercise_embeddings[:, t + 1], next_readout), dim=-1)
                )
            ).mean(dim=-1)
            predictions.append(next_prediction)
            readouts.append(next_readout)

            learning_previous = torch.where(
                current_active, current_learning, learning_previous
            )
            previous_readout = torch.where(
                current_active, next_readout, previous_readout
            )

        output = {
            "predictions": torch.stack(predictions, dim=1),
            "reconstruction_loss": reconstruction_loss,
            "denoise_gate": gate,
            "state_anomaly": state_anomaly * valid_mask,
            "profile_anomaly": profile_anomaly * valid_mask,
            "knowledge_readout": torch.stack(readouts, dim=1),
        }
        if return_details:
            return output
        return output
