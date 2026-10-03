"""ATDKT adapted from pykt-team/pykt-toolkit (audited 2026-10-03).

Local adaptations pool multi-concept interactions and score concept sets with
multi-label BCE. Empty concept/history supervision returns differentiable zero,
so short or unscored sequences do not produce NaN auxiliary losses.
"""

import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss
from .utils import ut_mask

from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings

device = "cpu" if not torch.cuda.is_available() else "cuda"

@MODEL_REGISTRY.register("atdkt")
class ATDKT(Module):
    class Inputs(InputSpec):
        """Optionally asks the loader for a running correctness history.

        Only the `his` embedding variants consume it, and computing it for the
        others would cost a pass over the data for a tensor nobody reads.
        """
        supports_multi_concept = True

        dataset_mode = "all_in_one"
        requires_question_ids = True

        @classmethod
        def prepare(cls, ctx):
            inputs = super().prepare(ctx)
            emb_type = str(ctx.model_cfg.get("emb_type", ""))
            inputs.dataset_kwargs["include_history"] = "his" in emb_type
            inputs.run_config_extras["model_correctness_revision"] = "2026-10-03"
            return inputs

    def __init__(self, num_q, num_c, seq_len, emb_size, dropout=0.1, emb_type='qid', 
            num_layers=1, num_attn_heads=5, l1=0.5, l2=0.5, l3=0.5, start=50, emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "atdkt"
        print(f"qnum: {num_q}, cnum: {num_c}")
        print(f"emb_type: {emb_type}")
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                Linear(self.hidden_size//2, self.num_c))

        if self.emb_type.endswith("predhis"):
            self.l1 = l1
            self.l2 = l2
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb
            if self.emb_type.find("qemb") != -1:
                self.question_emb = Embedding(self.num_q, self.emb_size)
            
            self.start = start
            self.hisclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(self.hidden_size//2, 1))
            self.hisloss = nn.MSELoss()

        if self.emb_type.endswith("predcurc"): # predict cur question' cur concept
            self.l1 = l1
            self.l2 = l2
            self.l3 = l3
            if self.num_q > 0:
                self.question_emb = Embedding(self.num_q, self.emb_size) # 1.2
            if self.emb_type.find("trans") != -1:
                self.nhead = num_attn_heads
                d_model = self.hidden_size# * 2
                encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
                encoder_norm = LayerNorm(d_model)
                self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
            else:    
                self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
            self.qclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                Linear(self.hidden_size//2, self.num_c))
            if self.emb_type.find("cemb") != -1:
                self.concept_emb = Embedding(self.num_c, self.emb_size) # add concept emb

            self.closs = CrossEntropyLoss()
            if self.emb_type.find("his") != -1:
                self.start = start
                self.hisclasifier = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(self.hidden_size//2, 1))
                self.hisloss = nn.MSELoss()

    def _concept_loss(self, logits, target):
        """ATDKT's `predcurc` auxiliary target, for one or many concepts per item.

        Upstream assumes exactly one concept per position, because pykt runs this
        model over KC-expanded sequences. Under `all_in_one` a position carries
        up to `max_concepts` of them padded with -1, so the target arrives as
        `[N, K]` and `CrossEntropyLoss` rejects it -- which is why enabling
        `predcurc` surfaced a failure that the configured `emb_type="qid"` had
        been hiding.

        `K == 1` keeps upstream's cross-entropy exactly, so assist2017,
        assist2012 and statics2011 are bit-identical to pykt's formulation.

        `K > 1` scores every concept of the item instead of picking one. This
        follows the choice already made for the embedding path a few lines
        below: `pool_concept_embeddings` averages over all of an item's
        concepts, so a target that kept only the first would be asking the
        classifier to predict something the encoder was never shown alone.
        Truncating to the first concept is the other defensible option -- it is
        what gkt and rekt do -- but it costs a `concepts_visible: first_of_K`
        stamp, which would move this model into its own protocol group and out
        of the comparison table.
        """
        if logits.size(0) == 0:
            return logits.sum() * 0.0
        if target.dim() == 1:
            return self.closs(logits, target)
        if target.size(-1) == 1:
            return self.closs(logits, target.squeeze(-1))
        valid = target >= 0
        multi_hot = torch.zeros_like(logits)
        rows = torch.arange(target.size(0), device=target.device)
        rows = rows.unsqueeze(-1).expand_as(target)[valid]
        multi_hot[rows, target[valid]] = 1.0
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, multi_hot
        )

    def predcurc(self, dcur, q, c, r, xemb, train):
        emb_type = self.emb_type
        y2, y3 = 0, 0
        if emb_type.find("delxemb") != -1:
            qemb = self.question_emb(q)
            cemb = pool_concept_embeddings(self.concept_emb, c, self.num_c)
            catemb = qemb + cemb
        else:
            catemb = xemb
            if self.num_q > 0:
                qemb = self.question_emb(q)
                catemb = qemb + xemb
                
            if emb_type.find("cemb") != -1:
                cemb = pool_concept_embeddings(self.concept_emb, c, self.num_c)
                catemb += cemb

        # cemb = self.concept_emb(c)
        # catemb = cemb
        if emb_type.find("trans") != -1:
            mask = ut_mask(seq_len=catemb.shape[1], target_device=catemb.device)
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            sm = dcur["smasks"].long()
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self._concept_loss(cpreds[flag], c[:,start:][flag])

        # predict response
        xemb = xemb + qh + cemb
        if emb_type.find("qemb") != -1:
            xemb = xemb+qemb
        h, _ = self.lstm_layer(xemb)

        # predict history correctness rates
        rpreds = None
        if train and emb_type.find("his") != -1:
            sm = dcur["smasks"].long()
            start = self.start
            rpreds = torch.sigmoid(self.hisclasifier(h)).squeeze(-1)
            rsm = sm[:,start:]
            rflag = rsm==1
            rtrues = dcur["historycorrs"][:,start:]
            y3 = (self.hisloss(rpreds[:,start:][rflag], rtrues[rflag])
                  if rflag.any() else rpreds.sum() * 0.0)

        # predict response
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y, y2, y3

    def forward(self, dcur, train=False): ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        
        y2, y3 = 0, 0

        emb_type = self.emb_type
        if emb_type.startswith("qid"):
            # Identity on [B,T]; on [B,T,K] mean-pools the question's KCs
            # with -1 padding masked, as DKT and pykt's
            # QueEmb.get_avg_skill_emb do.
            xemb = pool_interaction_embeddings(
                self.interaction_emb, c, r, self.num_c
            )
        rpreds, qh = None, None
        if emb_type == "qid":
            h, _ = self.lstm_layer(xemb)
            h = self.dropout_layer(h)
            y = torch.sigmoid(self.out_layer(h))
        elif emb_type.endswith("predhis"): # only predict history correct ratios
            # predict response
            if self.emb_type.find("cemb") != -1:
                cemb = pool_concept_embeddings(self.concept_emb, c, self.num_c)
                xemb = xemb + cemb
            if emb_type.find("qemb") != -1:
                qemb = self.question_emb(q)
                xemb = xemb+qemb
            h, _ = self.lstm_layer(xemb)
            # predict history correctness rates
            if train:
                sm = dcur["smasks"].long()
                start = self.start
                rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
                rsm = sm[:,start:]
                rflag = rsm==1
                rtrues = dcur["historycorrs"][:,start:]
                y2 = (self.hisloss(rpreds[rflag], rtrues[rflag])
                      if rflag.any() else rpreds.sum() * 0.0)

            h = self.dropout_layer(h)
            y = self.out_layer(h)
            y = torch.sigmoid(y)
        elif emb_type.endswith("predcurc"): # predict current question' current concept
            y, y2, y3 = self.predcurc(dcur, q, c, r, xemb, train)

        if train:
            return y, y2, y3
        else:
            return y
  
