import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("deep_irt")
class DeepIRTTrainer(BaseTrainer):
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

        base = cseqs if cseqs is not None and cseqs.numel() > 0 else qseqs
        base_shft = cshft if cshft is not None and cshft.numel() > 0 else qshft
        if base is None or base_shft is None:
            raise ValueError("DeepIRT requires concept or question sequences.")

        full_q = torch.cat((base[:, 0:1], base_shft), dim=1).long()
        full_r = torch.cat((rseqs[:, 0:1], rshft), dim=1).long()
        pred = self.model(full_q, full_r)[:, 1:]

        target = rshft

        # Mask padding
        pred = torch.masked_select(pred, sm)
        target = torch.masked_select(target, sm)

        loss = binary_cross_entropy(pred, target)

        return pred, target, loss
