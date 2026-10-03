import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("dimkt")
class DIMKTTrainer(BaseTrainer):
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

    training_forward_kwargs = {'train': True}

    def _forward_batch(self, batch, train=False):
        required = (
            "qseqs",
            "cseqs",
            "sdseqs",
            "qdseqs",
            "rseqs",
            "shft_qseqs",
            "shft_cseqs",
            "shft_sdseqs",
            "shft_qdseqs",
        )
        missing = [key for key in required if key not in batch or batch[key].numel() == 0]
        if missing:
            raise ValueError(f"DIMKT requires missing batch fields: {missing}.")

        q = batch["qseqs"].to(self.device).long()
        c = batch["cseqs"].to(self.device).long()
        sd = batch["sdseqs"].to(self.device).long()
        qd = batch["qdseqs"].to(self.device).long()
        r = batch["rseqs"].to(self.device).long()
        qshft = batch["shft_qseqs"].to(self.device).long()
        cshft = batch["shft_cseqs"].to(self.device).long()
        sdshft = batch["shft_sdseqs"].to(self.device).long()
        qdshft = batch["shft_qdseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        y = self.model(q, c, sd, qd, r, qshft, cshft, sdshft, qdshft)
        loss = _masked_bce(y, rshft, sm)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
