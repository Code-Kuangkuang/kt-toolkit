import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gkt")
class GKTTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        num_epochs,
        device,
        hooks=None,
        metric_key="valid_auc",
        patience=10,
        test_loader=None,
    ):
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience


    def _forward_batch(self, batch):
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        # IMPORTANT: GKT expects concept/skill ids in range [0, num_c-1].
        # Using question ids (qseqs) will create out-of-range interaction indices (q*2+r) and crash on CUDA.
        if cseqs is None or cseqs.numel() == 0:
            raise ValueError("GKTTrainer requires concept sequences (cseqs).")
        if cshft is None or cshft.numel() == 0:
            raise ValueError("GKTTrainer requires shifted concept sequences (shft_cseqs).")

        cseqs = self._first_concept(cseqs).to(self.device).long()
        cshft = self._first_concept(cshft).to(self.device).long()
        cc = torch.cat((cseqs[:, 0:1], cshft), dim=1)
        cr = torch.cat((rseqs[:, 0:1], rshft.long()), dim=1)

        preds = self.model(cc, cr)
        loss = cal_loss(self.model, [preds], rseqs, rshft, sm)

        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss

    @staticmethod
    def _first_concept(seqs):
        if seqs.dim() == 3:
            return seqs[..., 0]
        return seqs


def cal_loss(model, ys, r, rshft, sm, preloss=None):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double())
    return loss
