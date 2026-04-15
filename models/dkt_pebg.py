import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

from core.registry import MODEL_REGISTRY
from .utils import build_question_embedding, load_pretrained_question_matrix


@MODEL_REGISTRY.register("dkt_pebg")
class DKTPEBG(Module):
    def __init__(
        self,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_q=None,
        dpath="",
        keyid2idx_path="",
        pro_id_dict_path="",
        freeze_pretrained=False,
        use_pebg_booster=False,
        use_original_pebg_dkt=True,
        pebg_hidden_size=128,
    ):
        super().__init__()
        self.model_name = "dkt_pebg"
        self.num_c = int(num_c)
        self.num_q = int(num_q) if num_q is not None else 0
        self.emb_size = int(emb_size)
        self.hidden_size = int(emb_size)
        self.emb_type = emb_type
        self.pretrain_dim = int(pretrain_dim)

        booster_requested = bool(use_pebg_booster or emb_path)
        self.use_question_inputs = bool(booster_requested and emb_type.startswith("qid") and self.num_q > 0)
        self.use_original_pebg_dkt = bool(self.use_question_inputs and use_original_pebg_dkt)
        self.uses_binary_target = False
        self.output_mode = "concept"
        self.pretrained_meta = None

        if bool(use_pebg_booster) and not emb_path:
            raise ValueError("use_pebg_booster=True requires non-empty emb_path.")

        if self.use_question_inputs and self.use_original_pebg_dkt:
            matrix, self.pretrained_meta = load_pretrained_question_matrix(
                num_q=self.num_q,
                emb_path=emb_path,
                dpath=dpath,
                keyid2idx_path=keyid2idx_path,
                pro_id_dict_path=pro_id_dict_path,
            )
            if emb_path and self.pretrained_meta and self.pretrained_meta.get("status") == "missing":
                raise FileNotFoundError(f"PEBG embedding file not found: {emb_path}")

            raw_dim = int(matrix.shape[1]) if matrix is not None else int(self.emb_size)
            self.question_emb = Embedding(self.num_q + 1, raw_dim)
            with torch.no_grad():
                self.question_emb.weight.zero_()
                if matrix is not None:
                    self.question_emb.weight[1:].copy_(torch.from_numpy(matrix))
            self.question_emb.weight.requires_grad = not bool(freeze_pretrained)

            self.q_embed_dim = raw_dim
            self.hidden_size = int(pebg_hidden_size)
            self.lstm_layer = LSTM(self.q_embed_dim * 2, self.hidden_size, batch_first=True)
            self.dropout_layer = Dropout(dropout)
            self.out_layer = Linear(self.hidden_size + self.q_embed_dim, 1)
            self.output_mode = "binary_seq"
            self.uses_binary_target = True

            if self.pretrained_meta and self.pretrained_meta.get("status") != "missing":
                print(
                    f"[DKT-PEBG] Original PEBG-DKT mode enabled "
                    f"(status={self.pretrained_meta.get('status')}, matched={self.pretrained_meta.get('aligned_count', 0)})"
                )
            else:
                print("[DKT-PEBG] Original PEBG-DKT mode enabled, but no pretrained matrix was loaded.")
        elif self.use_question_inputs:
            self.question_emb, self.question_proj, self.pretrained_meta = build_question_embedding(
                num_q=self.num_q,
                emb_size=self.emb_size,
                emb_path=emb_path,
                dpath=dpath,
                keyid2idx_path=keyid2idx_path,
                pro_id_dict_path=pro_id_dict_path,
                freeze_pretrained=freeze_pretrained,
            )
            if emb_path and self.pretrained_meta and self.pretrained_meta.get("status") == "missing":
                raise FileNotFoundError(f"PEBG embedding file not found: {emb_path}")
            self.interaction_proj = Linear(self.emb_size * 2, self.emb_size)
            self.output_mode = "concept"
            if self.pretrained_meta and self.pretrained_meta.get("status") != "missing":
                print(
                    f"[DKT-PEBG] Loaded pretrained question embeddings "
                    f"(status={self.pretrained_meta.get('status')}, matched={self.pretrained_meta.get('aligned_count', 0)})"
                )
            else:
                print("[DKT-PEBG] PEBG booster enabled, but no pretrained matrix was loaded.")
        else:
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
            self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.dropout_layer = Dropout(dropout)
            self.out_layer = Linear(self.hidden_size, self.num_c)

        if self.use_question_inputs and not self.use_original_pebg_dkt:
            self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.dropout_layer = Dropout(dropout)
            self.out_layer = Linear(self.hidden_size, self.num_c)

    def _build_interaction_from_question(self, q, r):
        qidx = q.long().clamp(min=0, max=self.num_q - 1)
        qemb = self.question_proj(self.question_emb(qidx))
        r = r.float()

        wrong_embed = qemb * (1.0 - r.unsqueeze(-1))
        correct_embed = qemb * r.unsqueeze(-1)
        return self.interaction_proj(torch.cat([wrong_embed, correct_embed], dim=-1))

    def _map_qids_with_padding(self, q):
        q = q.long()
        q = torch.where(q < 0, torch.zeros_like(q), q + 1)
        return q.clamp(min=0, max=self.num_q)

    def _build_original_rnn_inputs(self, q, r):
        qidx = self._map_qids_with_padding(q)
        qemb = self.question_emb(qidx)
        r = r.float()
        pos_mask = (r > 0).float().unsqueeze(-1)
        rnn_inputs = torch.cat([qemb * pos_mask, qemb * (1.0 - pos_mask)], dim=-1)
        return qemb, rnn_inputs

    def forward(self, q, r, q_next=None):
        if self.use_question_inputs and self.use_original_pebg_dkt:
            _, rnn_inputs = self._build_original_rnn_inputs(q, r)
            if q_next is None:
                q_next = q
            q_next_idx = self._map_qids_with_padding(q_next)
            next_qemb = self.question_emb(q_next_idx)

            h, _ = self.lstm_layer(rnn_inputs)
            h = self.dropout_layer(h)
            logits = self.out_layer(torch.cat([h, next_qemb], dim=-1)).squeeze(-1)
            return torch.sigmoid(logits)

        if self.use_question_inputs:
            xemb = self._build_interaction_from_question(q, r)
        else:
            q = q.long().clamp(min=0, max=self.num_c - 1)
            r = r.long()
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y