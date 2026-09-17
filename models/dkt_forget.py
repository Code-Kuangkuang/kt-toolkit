import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings

device = "cpu" if not torch.cuda.is_available() else "cuda"

@MODEL_REGISTRY.register("dkt_forget")
@MODEL_REGISTRY.register("dkt-forget")
class DKTForget(Module):
    class Inputs(InputSpec):
        """Sizes the three gap embedding tables this model indexes into.

        The counts are table heights, not statistics: a gap is log2-bucketed, so
        `num_rgap` is "how many buckets do we need". That is why the original
        code read the test file -- it reads `timestamps` only, never a response.
        AGENTS.md requires fitting on the training folds, which leaves a table
        that can be too short, so one row is reserved as out-of-vocabulary and
        the dataset clamps onto it. `pykt_transductive` restores pyKT's
        every-split maximum.
        """

        dataset_mode = "all_in_one"

        @classmethod
        def prepare(cls, ctx):
            from datasets.feature_utils import compute_dkt_forget_stats

            transductive = bool(ctx.train_cfg.get("pykt_transductive", False))
            gap_files = [
                ctx.resolve_file(ctx.quelevel_key("train_valid_file"), "train_valid_file"),
                ctx.resolve_file(ctx.quelevel_key("test_file"), "test_file"),
            ]
            stats = compute_dkt_forget_stats(
                ctx.dataset_cfg["dpath"],
                gap_files,
                ctx.dataset_cfg["input_type"],
                folds=None if transductive else ctx.train_folds(),
            )
            return ModelInputs(
                # The gap counts land in both configs; from model_cfg they flow
                # into model_kwargs, which is how the constructor receives them.
                model_cfg_updates={**stats, "use_timestamps": True},
                dataset_cfg_updates=dict(stats),
                dataset_kwargs={
                    "include_dkt_forget": True,
                    "dkt_forget_caps": None if transductive else dict(stats),
                },
                feature_fit_scope="train_valid_test" if transductive else "train_folds",
            )

    def __init__(self, num_c, num_rgap, num_sgap, num_pcount, emb_size, dropout=0.1, emb_type='qid', emb_path=""):
        super().__init__()
        self.model_name = "dkt_forget"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, emb_size)
        ntotal = num_rgap + num_sgap + num_pcount
    
        self.lstm_layer = LSTM(self.emb_size + ntotal, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size + ntotal, self.num_c)
        

    def forward(self, q, r, dgaps):
        emb_type = self.emb_type
        if emb_type == "qid":
            # Identity on [B,T]; on [B,T,K] mean-pools the question's KCs
            # with -1 padding masked, as DKT and pykt's
            # QueEmb.get_avg_skill_emb do.
            xemb = pool_interaction_embeddings(
                self.interaction_emb, q, r, self.num_c
            )
            theta_in = self.c_integration(xemb, dgaps["rgaps"].long(), dgaps["sgaps"].long(), dgaps["pcounts"].long())

        h, _ = self.lstm_layer(theta_in)
        theta_out = self.c_integration(h, dgaps["shft_rgaps"].long(), dgaps["shft_sgaps"].long(), dgaps["shft_pcounts"].long())
        theta_out = self.dropout_layer(theta_out)
        y = self.out_layer(theta_out)
        y = torch.sigmoid(y)

        return y


class CIntegration(Module):
    def __init__(self, num_rgap, num_sgap, num_pcount, emb_dim) -> None:
        super().__init__()
        self.rgap_eye = torch.eye(num_rgap)
        self.sgap_eye = torch.eye(num_sgap)
        self.pcount_eye = torch.eye(num_pcount)

        ntotal = num_rgap + num_sgap + num_pcount
        self.cemb = Linear(ntotal, emb_dim, bias=False)
        print(f"num_sgap: {num_sgap}, num_rgap: {num_rgap}, num_pcount: {num_pcount}, ntotal: {ntotal}")
        # print(f"total: {ntotal}, self.cemb.weight: {self.cemb.weight.shape}")

    def forward(self, vt, rgap, sgap, pcount):
        dev = vt.device
        rgap = self.rgap_eye.to(dev)[rgap]
        sgap = self.sgap_eye.to(dev)[sgap]
        pcount = self.pcount_eye.to(dev)[pcount]
        # print(f"vt: {vt.shape}, rgap: {rgap.shape}, sgap: {sgap.shape}, pcount: {pcount.shape}")
        ct = torch.cat((rgap, sgap, pcount), -1) # bz * seq_len * num_fea
        # print(f"ct: {ct.shape}, self.cemb.weight: {self.cemb.weight.shape}")
        # element-wise mul
        Cct = self.cemb(ct) # bz * seq_len * emb
        # print(f"ct: {ct.shape}, Cct: {Cct.shape}")
        theta = torch.mul(vt, Cct)
        theta = torch.cat((theta, ct), -1)
        return theta
