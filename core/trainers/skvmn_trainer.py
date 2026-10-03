import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("skvmn")
class SKVMNTrainer(BaseTrainer):
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
        q_full = _full_sequence(batch, "cseqs", "shft_cseqs", self.device)
        if q_full is None:
            q_full = _full_sequence(batch, "qseqs", "shft_qseqs", self.device)
        if q_full is None:
            raise ValueError("SKVMN requires concept or question sequences.")
        r_full = _full_sequence(batch, "rseqs", "shft_rseqs", self.device)
        if r_full is None:
            raise ValueError("SKVMN requires response sequences.")

        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
        y = self.model(q_full.long(), r_full.long())[:, 1:]
        loss = _masked_bce(y, rshft, sm)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _full_sequence(batch, seq_key, shft_key, device, dtype=torch.long):
    seqs = batch.get(seq_key)
    shft = batch.get(shft_key)
    if seqs is None or seqs.numel() == 0 or shft is None or shft.numel() == 0:
        return None
    seqs = seqs.to(device).to(dtype)
    shft = shft.to(device).to(dtype)
    return torch.cat((seqs[:, 0:1], shft), dim=1)


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
