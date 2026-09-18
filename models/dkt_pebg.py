import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings
from .utils import build_question_embedding, load_pretrained_question_matrix


@MODEL_REGISTRY.register("dkt_pebg")
class DKTPEBG(Module):
    class Inputs(InputSpec):
        """Resolves which pretrained PEBG embedding, if any, to warm-start from.

        The embedding is a derived feature like any other: scripts/pretrain_pebg.py
        fits it from the sequence data, and with `--fold N` it excludes fold N and
        writes to `pebg/fold{N}`. Which one a run picked up therefore belongs in
        the protocol block, and the scope is read back from the directory the
        strategy actually selected rather than from what was requested.
        """

        @classmethod
        def _booster_scope(cls, booster, ctx):
            """Where the embedding this run loaded was fitted from.

            `train_folds` only when the selected directory is this fold's, since
            that is the one pretrain_pebg builds with the fold held out. An
            explicit `emb_path` bypasses the fold lookup entirely, and a
            fold-less `pebg/` directory was pretrained on everything; neither can
            be shown to exclude the test split, so both are recorded as
            transductive rather than assumed clean.
            """
            import os

            if not booster.get("enabled"):
                return "none"
            selected = os.path.normpath(booster.get("pebg_dir", "") or "")
            expected = os.path.normpath(
                os.path.join(
                    booster.get("pebg_dir_native", "") or "", f"fold{ctx.fold_id}"
                )
            )
            return "train_folds" if selected and selected == expected else "train_valid_test"

        @classmethod
        def prepare(cls, ctx):
            from strategies import apply_dkt_pebg_strategy

            model_cfg, booster = apply_dkt_pebg_strategy(
                model_cfg=dict(ctx.model_cfg),
                dataset_name=ctx.dataset_name,
                dataset_cfg=ctx.dataset_cfg,
                root_dir=ctx.root_dir,
                fold_id=ctx.fold_id,
            )
            scope = cls._booster_scope(booster, ctx)
            booster["fit_scope"] = scope
            print(
                "DKT-PEBG booster strategy resolved: "
                f"strategy={booster.get('strategy')} "
                f"enabled={booster.get('enabled')} "
                f"fit_scope={scope} "
                f"emb_path={booster.get('emb_path', '')}"
            )
            if scope == "train_valid_test":
                print(
                    "  Warning: this embedding was not shown to exclude the "
                    "current fold. Pretrain per fold with "
                    "`python scripts/pretrain_pebg.py --preprocess_mode sequence "
                    f"--fold {ctx.fold_id}`, or accept a transductive run."
                )
            return ModelInputs(
                model_cfg_updates=model_cfg,
                run_config_extras={"booster_info": booster},
                feature_fit_scope=scope,
            )

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
            # Identity on [B,T]; on [B,T,K] mean-pools the question's KCs
            # with -1 padding masked, as DKT and pykt's
            # QueEmb.get_avg_skill_emb do.
            xemb = pool_interaction_embeddings(
                self.interaction_emb, q, r, self.num_c
            )

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y