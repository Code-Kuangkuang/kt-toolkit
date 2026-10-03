import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


# csKT has the same dcur-in / tuple-out shape as stableKT.
@TRAINER_REGISTRY.register("cskt")
@TRAINER_REGISTRY.register("sparsekt")
@TRAINER_REGISTRY.register("stablekt")
class StableKTTrainer(BaseTrainer):
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
        dcur = _to_device_dict(batch, self.device)
        rshft = dcur["shft_rseqs"].float()
        sm = dcur["smasks"]

        result = self.model(dcur, train=train)
        preds = result[0] if isinstance(result, tuple) else result
        preds_for_loss = _align_shifted_preds(preds, rshft)
        loss = _masked_bce(preds_for_loss, rshft, sm)
        pred = torch.masked_select(preds_for_loss, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _to_device_dict(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _align_shifted_preds(preds, target):
    preds_for_loss = preds[:, 1:] if preds.size(1) == target.size(1) + 1 else preds
    if preds_for_loss.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(preds.shape)} does not align with shifted targets {tuple(target.shape)}."
        )
    return preds_for_loss


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)


SparseKTTrainer = StableKTTrainer
