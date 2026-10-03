import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("saint_plus")
@TRAINER_REGISTRY.register("saint")
class SAINTTrainer(BaseTrainer):
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
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        sm = batch["smasks"]
        masks = batch.get("masks")

        if qseqs is not None:
            qseqs = qseqs.to(self.device)
        if cseqs is not None:
            cseqs = cseqs.to(self.device)
        if rseqs is not None:
            rseqs = rseqs.to(self.device)
        if qshft is not None:
            qshft = qshft.to(self.device)
        if cshft is not None:
            cshft = cshft.to(self.device)
        if rshft is not None:
            rshft = rshft.to(self.device)
        if sm is not None:
            sm = sm.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)

        data = {
            "qseqs": qseqs,
            "cseqs": cseqs,
            "rseqs": rseqs,
            "shft_qseqs": qshft,
            "shft_cseqs": cshft,
            "shft_rseqs": rshft,
            "masks": masks if masks is not None else torch.zeros_like(sm),
            "smasks": sm,
        }

        pred = self.model(data, return_details=False)

        target = rshft

        pred = torch.masked_select(pred, sm)
        target = torch.masked_select(target, sm)

        loss = binary_cross_entropy(pred, target)

        return pred, target, loss


SAINTpTrainer = SAINTTrainer
