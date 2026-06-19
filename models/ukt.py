import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from core.registry import MODEL_REGISTRY


class Dim:
    batch = 0
    seq = 1
    feature = 2


def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)
    return ret


def d2s_1overx(distance):
    return 1 / (1 + distance)


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = -2 * torch.matmul(
        torch.sqrt(torch.clamp(cov1, min=1e-24)),
        torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2),
    ) + cov1_2 + cov2_2.transpose(-1, -2)
    return ret + cov_ret


class WassersteinNCELoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.activation = nn.ELU()

    def forward(
        self, batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov
    ):
        batch_sample_one_cov = self.activation(batch_sample_one_cov) + 1
        batch_sample_two_cov = self.activation(batch_sample_two_cov) + 1

        sim11 = (
            d2s_1overx(
                wasserstein_distance_matmul(
                    batch_sample_one_mean,
                    batch_sample_one_cov,
                    batch_sample_one_mean,
                    batch_sample_one_cov,
                )
            )
            / self.temperature
        )
        sim22 = (
            d2s_1overx(
                wasserstein_distance_matmul(
                    batch_sample_two_mean,
                    batch_sample_two_cov,
                    batch_sample_two_mean,
                    batch_sample_two_cov,
                )
            )
            / self.temperature
        )
        sim12 = (
            -d2s_1overx(
                wasserstein_distance_matmul(
                    batch_sample_one_mean,
                    batch_sample_one_cov,
                    batch_sample_two_mean,
                    batch_sample_two_cov,
                )
            )
            / self.temperature
        )
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


@MODEL_REGISTRY.register("ukt")
class UKT(nn.Module):
    """
    Uncertainty-aware Knowledge Tracing (UKT) model.
    Uses stochastic embeddings (mean and covariance) to represent uncertainty in learning.
    """

    def __init__(
        self,
        num_c,
        num_q,
        num_pid=None,
        emb_size=None,
        num_blocks=None,
        dropout=0.1,
        d_ff=256,
        num_layers=2,
        seq_len=200,
        num_attn_heads=8,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
        use_CL=True,
        use_mean_cov_diff=False,
        cl_weight=0.02,
        use_uncertainty_aug=True,
        atten_type="w2",
        emb_type="stoc_qid",
        **kwargs
    ):
        super().__init__()
        if emb_size is None:
            emb_size = kwargs.pop("d_model", 256)
        if num_blocks is None:
            num_blocks = kwargs.pop("n_blocks", 2)
        if num_pid is None:
            num_pid = num_q

        self.model_name = "ukt"
        self.num_c = num_c
        self.num_q = num_q
        self.num_pid = num_pid
        self.n_question = num_c
        self.n_pid = num_pid
        self.model_type = self.model_name
        self.emb_size = emb_size
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.use_CL = use_CL
        self.use_uncertainty_aug = use_uncertainty_aug
        self.atten_type = atten_type
        self.cl_weight = cl_weight

        embed_l = emb_size

        if use_CL:
            self.wloss = WassersteinNCELoss(1)

        self.embed_l = emb_size

        # Problem difficulty embedding
        if self.num_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.num_pid + 1, 1)
            else:
                self.difficult_param = nn.Embedding(self.num_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_c + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_c + 1, embed_l)

        # Stochastic embeddings for questions
        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            self.mean_q_embed = nn.Embedding(self.num_c, embed_l)
            self.cov_q_embed = nn.Embedding(self.num_c, embed_l)
            if self.separate_qa:
                self.mean_qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
                self.cov_qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
            else:
                self.mean_qa_embed = nn.Embedding(2, embed_l)
                self.cov_qa_embed = nn.Embedding(2, embed_l)

        # Architecture
        self.model = UKTArchitecture(
            num_c=num_c,
            num_blocks=num_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=emb_size,
            d_feature=emb_size // num_attn_heads,
            d_ff=d_ff,
            kq_same=kq_same,
            seq_len=seq_len,
        )

        self.out = nn.Sequential(
            nn.Linear(embed_l + embed_l + embed_l + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_pid + 1 and self.num_pid > 0:
                torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_mean_embed_data = self.mean_q_embed(q_data)
        q_cov_embed_data = self.cov_q_embed(q_data)

        if self.separate_qa:
            qa_data = q_data + self.num_c * target
            qa_mean_embed_data = self.mean_qa_embed(qa_data)
            qa_cov_embed_data = self.cov_qa_embed(qa_data)
        else:
            qa_mean_embed_data = self.mean_qa_embed(target) + q_mean_embed_data
            qa_cov_embed_data = self.cov_qa_embed(target) + q_cov_embed_data

        return q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data

    def forward(
        self,
        qseqs,
        rseqs,
        cseqs,
        qshft,
        cshft,
        rshft,
        pidseqs=None,
        pidshft=None,
        masks=None,
        train=False,
        shft_r_aug=None,
        r_aug=None,
        qtest=False,
        **kwargs
    ):
        q = qseqs.long() if qseqs is not None else None
        c = cseqs.long() if cseqs is not None else q
        if c is None:
            raise ValueError("UKT requires concept sequences or question sequences.")
        r = rseqs.long()
        qshft = qshft.long() if qshft is not None else None
        cshft = cshft.long() if cshft is not None else qshft
        if cshft is None:
            raise ValueError("UKT requires shifted concept or question sequences.")
        rshft = rshft.long()

        # Match pykt: concepts drive base embeddings, questions drive problem difficulty.
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        # Handle augmented responses for CL
        if train and self.use_CL:
            if self.use_uncertainty_aug and shft_r_aug is not None and r_aug is not None:
                target_aug = torch.cat((r_aug[:, 0:1], shft_r_aug), dim=1)
            else:
                target_aug = target

        emb_type = self.emb_type

        # Generate stochastic embeddings
        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data = self.base_emb(
                q_data, target
            )

            if train and self.use_CL:
                (
                    mean_q_aug_embed_data,
                    cov_q_aug_embed_data,
                    mean_qa_aug_embed_data,
                    cov_qa_aug_embed_data,
                ) = self.base_emb(q_data, target_aug)

        # Add problem difficulty
        if self.num_pid > 0 and emb_type.find("norasch") == -1:
            if pidseqs is not None:
                pid = pidseqs.long()
                next_pid = pidshft.long() if pidshft is not None else qshft
            else:
                pid = q
                next_pid = qshft
            if pid is None or next_pid is None:
                raise ValueError(
                    "UKT Rasch difficulty requires qseqs/shft_qseqs "
                    "or pidseqs/shft_pidseqs. Set num_pid=0 for concept-only data."
                )
            pid_data = torch.cat((pid[:, 0:1], next_pid), dim=1)
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)
                q_mean_embed_data = q_mean_embed_data + pid_embed_data * q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data * q_embed_diff_data
                if train and self.use_CL:
                    mean_q_aug_embed_data = mean_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    cov_q_aug_embed_data = cov_q_aug_embed_data + pid_embed_data * q_embed_diff_data
            else:
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)
                q_mean_embed_data = q_mean_embed_data + pid_embed_data * q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data * q_embed_diff_data

                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_mean_embed_data = qa_mean_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )
                qa_cov_embed_data = qa_cov_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )
                if train and self.use_CL:
                    qa_aug_embed_diff_data = self.qa_embed_diff(target_aug)
                    mean_q_aug_embed_data = mean_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    cov_q_aug_embed_data = cov_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    mean_qa_aug_embed_data = mean_qa_aug_embed_data + pid_embed_data * (
                        qa_aug_embed_diff_data + q_embed_diff_data
                    )
                    cov_qa_aug_embed_data = cov_qa_aug_embed_data + pid_embed_data * (
                        qa_aug_embed_diff_data + q_embed_diff_data
                    )

        # Pass through transformer
        mean_d_output, cov_d_output = self.model(
            q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data, self.atten_type
        )

        # Calculate CL loss if training with CL
        cl_loss = None
        if train and self.use_CL:
            mean_d2_output, cov_d2_output = self.model(
                mean_q_aug_embed_data,
                cov_q_aug_embed_data,
                mean_qa_aug_embed_data,
                cov_qa_aug_embed_data,
                self.atten_type,
            )
            mas = masks if masks is not None else torch.ones_like(rseqs)
            true_tensor = torch.ones(mas.size(0), 1, dtype=torch.bool, device=mas.device)
            mas = torch.cat((true_tensor, mas), dim=1).unsqueeze(-1)

            pooled_mean_d_output = torch.mean(mean_d_output * mas, dim=1)
            pooled_cov_d_output = torch.mean(cov_d_output * mas, dim=1)
            pooled_mean_d2_output = torch.mean(mean_d2_output * mas, dim=1)
            pooled_cov_d2_output = torch.mean(cov_d2_output * mas, dim=1)

            if emb_type == "stoc_qid":
                cl_loss = self.wloss(
                    pooled_mean_d_output,
                    pooled_cov_d_output,
                    pooled_mean_d2_output,
                    pooled_cov_d2_output,
                )
            else:
                cl_loss = self.wloss(
                    pooled_mean_d_output,
                    pooled_mean_d_output,
                    pooled_mean_d2_output,
                    pooled_mean_d2_output,
                )

        # Calculate uncertainty measure
        activation = nn.ELU()
        temp = torch.mean(torch.mean(activation(cov_d_output) + 1, dim=-1), -1)

        # Final prediction
        if emb_type == "stoc_qid":
            concat_q = torch.cat(
                [mean_d_output, cov_d_output, q_mean_embed_data, q_cov_embed_data], dim=-1
            )
        else:
            concat_q = torch.cat(
                [mean_d_output, mean_d_output, q_cov_embed_data, q_cov_embed_data], dim=-1
            )
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)

        if train:
            if self.use_CL:
                return preds, cl_loss, temp
            else:
                return preds
        else:
            if qtest:
                return preds, concat_q
            return preds


class UKTArchitecture(nn.Module):
    def __init__(
        self,
        num_c,
        num_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
    ):
        super().__init__()
        self.d_model = d_model

        self.position_mean_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        self.position_cov_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

        self.blocks_2 = nn.ModuleList(
            [
                UKTTransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data, atten_type="w2"):
        elu_act = torch.nn.ELU()

        mean_q_posemb = self.position_mean_embeddings(q_mean_embed_data)
        cov_q_posemb = self.position_cov_embeddings(q_cov_embed_data)

        q_mean_embed_data = q_mean_embed_data + mean_q_posemb
        q_cov_embed_data = q_cov_embed_data + cov_q_posemb

        qa_mean_posemb = self.position_mean_embeddings(qa_mean_embed_data)
        qa_cov_posemb = self.position_cov_embeddings(qa_cov_embed_data)

        qa_mean_embed_data = qa_mean_embed_data + qa_mean_posemb
        qa_cov_embed_data = qa_cov_embed_data + qa_cov_posemb

        # ELU activation + 1 for covariance
        q_cov_embed_data = elu_act(q_cov_embed_data) + 1
        qa_cov_embed_data = elu_act(qa_cov_embed_data) + 1

        y_mean = qa_mean_embed_data
        y_cov = qa_cov_embed_data
        x_mean = q_mean_embed_data
        x_cov = q_cov_embed_data

        for block in self.blocks_2:
            x_mean, x_cov = block(
                mask=0,
                query_mean=x_mean,
                query_cov=x_cov,
                key_mean=x_mean,
                key_cov=x_cov,
                values_mean=y_mean,
                values_cov=y_cov,
                atten_type=atten_type,
                apply_pos=True,
            )

        return x_mean, x_cov


class UKTTransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1

        self.masked_attn_head = UKTMultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()

    def forward(
        self, mask, query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, atten_type="w2", apply_pos=True
    ):
        seqlen = query_mean.size(1)

        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query_mean.device)

        if mask == 0:
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean,
                query_cov,
                key_mean,
                key_cov,
                values_mean,
                values_cov,
                mask=src_mask,
                atten_type=atten_type,
                zero_pad=True,
            )
        else:
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean,
                query_cov,
                key_mean,
                key_cov,
                values_mean,
                values_cov,
                mask=src_mask,
                atten_type=atten_type,
                zero_pad=False,
            )

        query_mean = query_mean + self.dropout1((query2_mean))
        query_cov = query_cov + self.dropout1((query2_cov))

        query_mean = self.layer_norm1(query_mean)
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)

        if apply_pos:
            query2_mean = self.mean_linear2(self.dropout(self.activation(self.mean_linear1(query_mean))))
            query2_cov = self.cov_linear2(self.dropout(self.activation(self.cov_linear1(query_cov))))

            query_mean = query_mean + self.dropout2((query2_mean))
            query_cov = query_cov + self.dropout2((query2_cov))
            query_mean = self.layer_norm2(query2_mean)
            query_cov = self.layer_norm2(self.activation2(query2_cov) + 1)

        return query_mean, query_cov


class UKTMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.activation = nn.ELU()

        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        if kq_same is False:
            self.q_mean_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias

        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_mean_linear.weight)
        nn.init.xavier_uniform_(self.k_cov_linear.weight)
        nn.init.xavier_uniform_(self.v_mean_linear.weight)
        nn.init.xavier_uniform_(self.v_cov_linear.weight)

        if self.kq_same is False:
            nn.init.xavier_uniform_(self.q_mean_linear.weight)
            nn.init.xavier_uniform_(self.q_cov_linear.weight)

        if self.proj_bias:
            nn.init.constant_(self.k_mean_linear.bias, 0.0)
            nn.init.constant_(self.k_cov_linear.bias, 0.0)
            nn.init.constant_(self.v_mean_linear.bias, 0.0)
            nn.init.constant_(self.v_cov_linear.bias, 0.0)

            if self.kq_same is False:
                nn.init.constant_(self.q_mean_linear.bias, 0.0)
                nn.init.constant_(self.q_cov_linear.bias, 0.0)

            nn.init.constant_(self.out_mean_proj.bias, 0.0)
            nn.init.constant_(self.out_cov_proj.bias, 0.0)

    def forward(self, q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, mask, atten_type, zero_pad):
        bs = q_mean.size(0)

        k_mean = self.k_mean_linear(k_mean).view(bs, -1, self.h, self.d_k)
        k_cov = self.k_cov_linear(k_cov).view(bs, -1, self.h, self.d_k)

        if self.kq_same is False:
            q_mean = self.q_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.q_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)
        else:
            q_mean = self.k_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.k_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)

        v_mean = self.v_mean_linear(v_mean).view(bs, -1, self.h, self.d_k)
        v_cov = self.v_cov_linear(v_cov).view(bs, -1, self.h, self.d_k)

        k_mean = k_mean.transpose(1, 2)
        q_mean = q_mean.transpose(1, 2)
        v_mean = v_mean.transpose(1, 2)
        k_cov = k_cov.transpose(1, 2)
        q_cov = q_cov.transpose(1, 2)
        v_cov = v_cov.transpose(1, 2)

        gammas = self.gammas
        if atten_type == "w2":
            scores_mean, scores_cov = uattention(
                q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, self.d_k, mask, self.dropout, zero_pad, gammas
            )
        elif atten_type == "dp":
            scores_mean, scores_cov = attention_dp(
                q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, self.d_k, mask, self.dropout, zero_pad, gammas
            )

        concat_mean = scores_mean.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        concat_cov = scores_cov.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output_mean = self.out_mean_proj(concat_mean)
        output_cov = self.out_cov_proj(concat_cov)

        return output_mean, output_cov


def attention_dp(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    scores_mean = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / math.sqrt(d_k)
    scores_cov = torch.matmul(q_cov, k_cov.transpose(-2, -1)) / math.sqrt(d_k)

    bs, head, seqlen = scores_mean.size(0), scores_mean.size(1), scores_mean.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(q_mean.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_mean_ = scores_mean.masked_fill(mask == 0, -1e32)
        scores_cov_ = scores_cov.masked_fill(mask == 0, -1e32)

        scores_mean_ = F.softmax(scores_mean_, dim=-1)
        scores_cov_ = F.softmax(scores_cov_, dim=-1)

        scores_mean_ = scores_mean_ * mask.float().to(q_mean.device)
        scores_cov_ = scores_cov_ * mask.float().to(q_mean.device)

        distcum_scores_mean = torch.cumsum(scores_mean_, dim=-1)
        distcum_scores_cov = torch.cumsum(scores_cov_, dim=-1)

        disttotal_scores_mean = torch.sum(scores_mean_, dim=-1, keepdim=True)
        disttotal_scores_cov = torch.sum(scores_cov_, dim=-1, keepdim=True)

        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(q_mean.device)

        dist_scores_mean = torch.clamp((disttotal_scores_mean - distcum_scores_mean) * position_effect, min=0.0)
        dist_scores_cov = torch.clamp((disttotal_scores_cov - distcum_scores_cov) * position_effect, min=0.0)

        dist_scores_mean = dist_scores_mean.sqrt().detach()
        dist_scores_cov = dist_scores_cov.sqrt().detach()

    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect_mean = torch.clamp(torch.clamp((dist_scores_mean * gamma).exp(), min=1e-5), max=1e5)
    total_effect_cov = torch.clamp(torch.clamp((dist_scores_cov * gamma).exp(), min=1e-5), max=1e5)

    scores_mean = scores_mean * total_effect_mean
    scores_cov = scores_cov * total_effect_cov

    scores_mean.masked_fill_(mask == 0, -1e32)
    scores_cov.masked_fill_(mask == 0, -1e32)

    scores_mean = F.softmax(scores_mean, dim=-1)
    scores_cov = F.softmax(scores_cov, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(q_mean.device)
        scores_mean = torch.cat([pad_zero, scores_mean[:, :, 1:, :]], dim=2)
        scores_cov = torch.cat([pad_zero, scores_cov[:, :, 1:, :]], dim=2)

    scores_mean = dropout(scores_mean)
    scores_cov = dropout(scores_cov)

    output_mean = torch.matmul(scores_mean, v_mean)
    output_cov = torch.matmul(scores_cov, v_cov)

    return output_mean, output_cov


def uattention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    scores = -wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(q_mean.device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(q_mean.device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(q_mean.device)
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.0)
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)

    scores = scores * total_effect
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(q_mean.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)

    output_mean = torch.matmul(scores, v_mean)
    output_cov = torch.matmul(scores**2, v_cov)

    return output_mean, output_cov


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]
