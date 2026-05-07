import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("qikt")
class QIKTTrainer(BaseTrainer):
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
        other_config=None,
        test_loader=None,
    ):
        if other_config is None:
            other_config = {}
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience
        self.other_config = other_config

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        total_batches = len(self.train_loader)

        print(f"\n== Epoch {epoch}/{self.num_epochs} ==")
        print("=" * 50)

        for batch_idx, batch in enumerate(self.train_loader):
            pred, target, loss = self._forward_batch(batch)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # Progress bar
            self._print_progress(batch_idx, total_batches, loss.item())


        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        # Prepare data - move ALL tensors to device
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        sm = batch["smasks"]
        masks = batch.get("masks")

        # Move ALL to device
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

        # Build full sequences
        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)

        if q_full is None or c_full is None:
            raise ValueError("QIKTTrainer requires both question and concept sequences.")

        # Prepare data dict for model - ensure all tensors on device
        data = {
            "cq": q_full.long(),
            "cc": c_full.long(),
            "cr": r_full.long(),
            "q": qseqs,
            "c": cseqs,
            "r": rseqs,
            "qshft": qshft,
            "cshft": cshft,
            "rshft": rshft,
            "m": masks if masks is not None else torch.zeros_like(sm),
            "sm": sm,
        }

        # Forward through model
        outputs = self.model(data["cq"], data["cc"], data["cr"], data=data)

        # Compute losses with configurable weights
        loss_q_all = self._get_loss(outputs['y_question_all'], rshft, sm)
        loss_c_all = self._get_loss(outputs['y_concept_all'], rshft, sm)
        loss_q_next = self._get_loss(outputs['y_question_next'], rshft, sm)
        loss_c_next = self._get_loss(outputs['y_concept_next'], rshft, sm)

        # Get loss weights from config (pykt uses loss_x * output_x)
        loss_c_all_lambda = self.other_config.get('loss_c_all_lambda', 0) * self.other_config.get('output_c_all_lambda', 1)
        loss_c_next_lambda = self.other_config.get('loss_c_next_lambda', 0) * self.other_config.get('output_c_next_lambda', 1)
        loss_q_all_lambda = self.other_config.get('loss_q_all_lambda', 0) * self.other_config.get('output_q_all_lambda', 1)
        loss_q_next_lambda = self.other_config.get('loss_q_next_lambda', 0) * self.other_config.get('output_q_next_lambda', 0)

        # Get output weights from config (for fusion)
        output_c_all_lambda = self.other_config.get('output_c_all_lambda', 1)
        output_c_next_lambda = self.other_config.get('output_c_next_lambda', 1)
        output_q_all_lambda = self.other_config.get('output_q_all_lambda', 1)
        output_q_next_lambda = self.other_config.get('output_q_next_lambda', 0)

        if self.model.output_mode == "an_irt":
            # IRT mode: use sigmoid inverse for fusion
            def sigmoid_inverse(x, epsilon=1e-8):
                return torch.log(x / (1 - x + epsilon) + epsilon)
            y = sigmoid_inverse(outputs['y_question_all']) * output_q_all_lambda + \
                sigmoid_inverse(outputs['y_concept_all']) * output_c_all_lambda + \
                sigmoid_inverse(outputs['y_concept_next']) * output_c_next_lambda
            y = torch.sigmoid(y)
            outputs['y'] = y
        else:
            # Normal mode: weighted average
            y = outputs['y_question_all'] * output_q_all_lambda + \
                outputs['y_concept_all'] * output_c_all_lambda + \
                outputs['y_concept_next'] * output_c_next_lambda
            y = y / (output_q_all_lambda + output_c_all_lambda + output_c_next_lambda)
            outputs['y'] = y

        loss_kt = self._get_loss(outputs['y'], rshft, sm)

        if self.model.output_mode == "an_irt":
            loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next
        else:
            loss = loss_kt + loss_q_all_lambda * loss_q_all + loss_c_all_lambda * loss_c_all + loss_c_next_lambda * loss_c_next + loss_q_next_lambda * loss_q_next

        pred = torch.masked_select(outputs['y'], sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss

    @staticmethod
    def _concat_full(seqs, shft):
        if seqs is None or seqs.numel() == 0:
            return None
        return torch.cat((seqs[:, :1], shft), dim=1)

    def _get_loss(self, ys, rshft, sm):
        y_pred = torch.masked_select(ys, sm)
        y_true = torch.masked_select(rshft, sm)
        return binary_cross_entropy(y_pred.double(), y_true.double())
