import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("lpkt")
class LPKTTrainer(BaseTrainer):
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
        # Add scheduler for LPKT (same as pykt)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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


        # Step scheduler after each epoch (same as pykt)
        # Use step() without epoch parameter (PyTorch 2.0+ recommendation)
        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        return float(np.mean(losses)) if losses else 0.0

    def _bucketize_time(self, time_tensor, max_index):
        # Convert raw timestamps/durations into compact, bounded indices.
        # If input is already a valid categorical index, keep it unchanged.
        safe = torch.clamp(time_tensor.long(), min=0)
        if safe.numel() == 0:
            return safe
        if int(torch.max(safe).item()) <= int(max_index):
            return safe

        # Make time relative within each sequence to avoid epoch-scale values.
        if safe.dim() >= 2:
            relative = safe - safe[:, :1]
        else:
            relative = safe - safe[0]
        relative = torch.clamp(relative, min=0)

        rel_max = int(torch.max(relative).item())
        if rel_max > 10_000_000:
            # Likely milliseconds.
            base = torch.div(relative, 60000, rounding_mode="floor")
        elif rel_max > 100_000:
            # Likely seconds.
            base = torch.div(relative, 60, rounding_mode="floor")
        else:
            base = relative

        buckets = torch.log2(base.to(torch.float32) + 1.0).to(torch.long)
        return torch.clamp(buckets, min=0, max=max_index)

    def _forward_batch(self, batch):
        # LPKT: e_data (exercises), a_data (answers), it_data (interaction times)
        # Note: pykt passes itseqs directly, not calculating from tseqs
        qseqs = batch.get("qseqs")  # exercises
        qshft = batch.get("shft_qseqs")
        cseqs = batch.get("cseqs")
        cshft = batch.get("shft_cseqs")
        rseqs = batch["rseqs"].to(self.device)  # answers
        itseqs = batch.get("itseqs")  # interaction time intervals (pre-computed)
        itshft = batch.get("shft_itseqs")
        rshft = batch["shft_rseqs"].to(self.device)
        sm = batch["smasks"].to(self.device)

        # Debug: print keys in batch
        # print(f"Batch keys: {batch.keys()}")
        # print(f"itseqs: {itseqs is not None}, itshft: {itshft is not None}")

        # Build full sequences (same as pykt)
        e_data = torch.cat((qseqs[:, 0:1], qshft), dim=1) if qseqs is not None else None
        a_data = torch.cat((rseqs[:, 0:1], rshft), dim=1)
        concept_data = (
            torch.cat((cseqs[:, 0:1], cshft), dim=1)
            if cseqs is not None and cshft is not None
            else None
        )
        transition_mask = batch["masks"].to(self.device).bool()
        valid_mask = torch.cat(
            (transition_mask[:, 0:1], transition_mask), dim=1
        )

        # Build it_data (interaction time) - same as pykt
        # pykt: cit = torch.cat((dcur["itseqs"][:,0:1], dcur["shft_itseqs"]), dim=1)
        it_data = None
        if itseqs is not None and itshft is not None:
            it_data = torch.cat((itseqs[:, 0:1], itshft), dim=1).long()
            it_max_idx = self.model.it_embed.num_embeddings - 1
            it_data = self._bucketize_time(it_data, max_index=it_max_idx)
        else:
            # LPKT requires timestamps - warn if not available
            print(f"Warning: itseqs={itseqs is not None}, itshft={itshft is not None}, use_time={self.model.use_time}")

        at_data = None
        if self.model.use_runtime_concepts:
            atseqs = batch.get("utseqs")
            atshft = batch.get("shft_utseqs")
            if atseqs is not None and atshft is not None:
                at_data = torch.cat((atseqs[:, 0:1], atshft), dim=1).long()
                at_data = self._bucketize_time(
                    at_data, self.model.at_embed.num_embeddings - 1
                )

        # Validate index ranges explicitly to avoid opaque CUDA device-side asserts.
        if e_data is not None:
            e_min = int(torch.min(e_data).item())
            e_max = int(torch.max(e_data).item())
            e_max_idx = self.model.e_embed.num_embeddings - 1
            if e_min < 0 or e_max > e_max_idx:
                raise ValueError(
                    f"LPKT e_data index out of range: min={e_min}, max={e_max}, allowed=[0, {e_max_idx}]"
                )

        if it_data is not None:
            it_min = int(torch.min(it_data).item())
            it_max = int(torch.max(it_data).item())
            it_max_idx = self.model.it_embed.num_embeddings - 1
            if it_min < 0 or it_max > it_max_idx:
                raise ValueError(
                    f"LPKT it_data index out of range after bucketization: min={it_min}, max={it_max}, allowed=[0, {it_max_idx}]"
                )

        # Move final tensors to device
        if e_data is not None:
            e_data = e_data.to(self.device)
        a_data = a_data.to(self.device)
        if it_data is not None:
            it_data = it_data.to(self.device)
        if concept_data is not None:
            concept_data = concept_data.to(self.device).long()
        if at_data is not None:
            at_data = at_data.to(self.device)

        predictions = self.model(
            e_data,
            a_data,
            it_data=it_data,
            at_data=at_data,
            concept_data=concept_data,
            valid_mask=valid_mask,
        )

        # Use predictions from position 1 onwards (same as pykt: y[:, 1:])
        predictions = predictions[:, 1:]

        # Compute loss
        pred = torch.masked_select(predictions, sm)
        target = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(pred.double(), target.double())

        return pred, target, loss
