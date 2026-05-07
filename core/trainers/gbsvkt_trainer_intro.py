"""
GBSVKTTrainer: Granular Ball Support Vector Knowledge Tracing Trainer.

本 trainer 实现四路并行损失:
  1. Loss_shortcut  (SVM hinge, 有偏置)   → 驱动 W_shortcut 学习最终预测
  2. Loss_theta     (SVM hinge, 无偏置)   → 驱动 W_theta 学习 IRT 能力参数
  3. Loss_radius    (Log-barrier)         → 防止知识状态半径 r_h/r_d 塌缩
  4. Loss_conf      (ε-insensitive SVM)  → 驱动 W_conf 学习预测置信度

与 gbsvkt_intro.py 模型配合使用, 训练时直接引用模型的:
  W_shortcut [4*d_p,1], b_shortcut [1]
  W_theta    [d_p,1],   b_theta    [1]
  W_conf     [d_p,1],   b_conf     [1]
"""

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gbsvkt")

class GBSVKTTrainer(BaseTrainer):
    # ============================================================










    # ============================================================
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



        model_ref = self.model.module if hasattr(self.model, "module") else self.model




        if not hasattr(model_ref, "W_shortcut") or not hasattr(model_ref, "b_shortcut"):
            raise AttributeError("GBSVKT model must define parameters 'W_shortcut' and 'b_shortcut'.")
        if not isinstance(model_ref.W_shortcut, torch.nn.Parameter) or \
           not isinstance(model_ref.b_shortcut, torch.nn.Parameter):
            raise TypeError("GBSVKT model attributes 'W_shortcut' and 'b_shortcut' must be torch.nn.Parameter.")








        self.W_shortcut = model_ref.W_shortcut  # [4*d_p, 1]
        self.b_shortcut = model_ref.b_shortcut   # [1]
        self.W_theta    = model_ref.W_theta     # [d_p, 1]
        self.W_conf     = model_ref.W_conf       # [d_p, 1]


        opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if id(self.W_shortcut) not in opt_param_ids or \
           id(self.b_shortcut) not in opt_param_ids:
            raise ValueError("Optimizer must include model parameters 'W_shortcut' and 'b_shortcut' for GBSVKT.")

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer    = optimizer
        self.device       = device
        self.metric_key   = metric_key
        self.patience     = patience
        self.other_config = other_config


        self.C       = float(other_config.get("C", 1.0))
        self.epsilon = float(other_config.get("epsilon", 0.01))






        warmup_epochs = int(other_config.get("warmup_epochs", 3))
        use_scheduler = bool(other_config.get("use_scheduler", True))
        self.scheduler = None
        if use_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

    # ============================================================





    #



    #   y  : [B, T, 4*d_p]  (feature1 = shortcut_input)


    #



    #


    #   y.permute(0,2,1)  : [B, 4*d_p, T]
    #   matmul + b        : [B, T, 1]
    #   squeeze(-1)       : [B, T]

    #   sigmoid           : [?]
    # ============================================================
    def cal_loss1(self, w, b, y, t, sm):

        t = t * 2 - 1  # [B*T]

        norm_w = torch.norm(w)



        wxeb = (torch.matmul(y.permute(0, 2, 1), w) + b).squeeze(-1)  # [B, T]


        wxeb = torch.masked_select(wxeb, sm)  # [?]


        y = torch.sigmoid(wxeb / norm_w)




        loss = 0.5 * w.pow(2).sum() + self.C * torch.relu(1 - t * wxeb).pow(2).mean()

        return loss, y

    # ============================================================



    #

    #   w  : [d_p, 1]  (W_theta)
    #   y  : [B, T, d_p]  (feature2 = effective_diff)


    # ============================================================
    def cal_loss2(self, w, y, t, sm):
        t = t * 2 - 1  # [B*T]

        norm_w = torch.norm(w)


        wxeb = torch.matmul(y.permute(0, 2, 1), w).squeeze(-1)  # [B, T]
        wxeb = torch.masked_select(wxeb, sm)  # [?]
        y = torch.sigmoid(wxeb / norm_w)       # [?]

        loss = 0.5 * w.pow(2).sum() + self.C * torch.relu(1 - t * wxeb).pow(2).mean()

        return loss, y

    # ============================================================






    #

    #   w  : [d_p, 1]  (W_conf)
    #   y  : [B, T, d_p]  (feature4 = -radius_sum)

    #   sm : [B, T]



    # ============================================================
    def cal_loss4(self, w, y, t, sm):

        y = torch.matmul(y.permute(0, 2, 1), w).squeeze(-1)  # [B, T]
        y = torch.masked_select(y, sm)  # [?]


        loss = 0.5 * w.pow(2).sum() + self.C * torch.relu(
            torch.abs(t - y) - self.epsilon).pow(2).mean()

        return loss, y

    # ============================================================



    # ============================================================
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())
            self._print_progress(batch_idx, total_batches, loss.item())


        # Use step() without epoch parameter (PyTorch 2.0+ recommendation)
        if self.scheduler is not None:
            self.scheduler.step()

        return float(np.mean(losses)) if losses else 0.0

    # ============================================================



    # ============================================================

    # ============================================================
    # _should_stop: Early Stopping


    # ============================================================

    # ============================================================








    #



    # ============================================================
    @staticmethod
    def _concat_full(seqs, shft):
        if seqs is None or seqs.numel() == 0:
            return None


        return torch.cat((seqs[:, :1], shft), dim=1)

    # ============================================================



    #   batch = {



    #   }




    #

    #   q_full/c_full/r_full    : [B, T+1]

    #   feature1/2/4            : [B, d_p, T]

    #   feature*.slice          : [B, d_p, L]

    # ============================================================
    def _forward_batch(self, batch):



        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        sm    = batch["smasks"]




        if qseqs is not None: qseqs = qseqs.to(self.device)
        if cseqs is not None: cseqs = cseqs.to(self.device)
        if rseqs is not None: rseqs = rseqs.to(self.device)
        if qshft is not None: qshft = qshft.to(self.device)
        if cshft is not None: cshft = cshft.to(self.device)
        if rshft is not None: rshft = rshft.to(self.device)
        if sm    is not None: sm    = sm.to(self.device)




        #   c_full: [B, T+1]
        #   r_full: [B, T+1]

        q_full = self._concat_full(qseqs, qshft)  # [B, T+1]
        c_full = self._concat_full(cseqs, cshft)   # [B, T+1]
        r_full = self._concat_full(rseqs, rshft)   # [B, T+1]

        if q_full is None or c_full is None:
            raise ValueError("GBSVKTTrainer requires both question and concept sequences.")





        #




        #   conf      : [B, T]
        #   r_h_mean  : [B, T]
        #   r_d_mean  : [B, T]

        outputs = self.model(q_full.long(), c_full.long(), r_full.float())

        y        = outputs["y"]
        conf     = outputs["confidence"]  # [B, T]
        r_h_mean = outputs["r_h_mean"]    # [B, T]
        r_d_mean = outputs["r_d_mean"]    # [B, T]
        feature1 = outputs["feature1"]  # [B, 4*d_p, T]
        feature2 = outputs["feature2"]  # [B, d_p, T]
        feature4 = outputs["feature4"]  # [B, d_p, T]



        #   y.size(1) = T, rshft.size(1) = T, sm.size(1) = T
        #   common_len = min(T, T, T) = T

        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.tensor(0.0, device=self.device)


        feature1 = feature1[:, :, :common_len]  # [B, 4*d_p, L]
        feature2 = feature2[:, :, :common_len]  # [B, d_p, L]
        feature4 = feature4[:, :, :common_len]  # [B, d_p, L]
        conf     = conf[:, :common_len]           # [B, L]
        r_h_mean = r_h_mean[:, :common_len]       # [B, L]
        r_d_mean = r_d_mean[:, :common_len]       # [B, L]
        rshft    = rshft[:, :common_len]          # [B, L]
        sm       = sm[:, :common_len]             # [B, L]





        target = torch.masked_select(rshft, sm)  # [?]
        target = target.float()                  # [?] float



        #
        # feature1: [B, 4*d_p, L]






        loss_shortcut, pred_shortcut = self.cal_loss1(
            self.W_shortcut, self.b_shortcut, feature1, target, sm)

        if pred_shortcut.numel() == 0:
            return pred_shortcut, target, torch.tensor(0.0, device=self.device)



        #
        # feature2: [B, d_p, L]







        loss_theta, pred_theta = self.cal_loss2(
            self.W_theta, torch.sigmoid(feature2), target, sm)
        if pred_theta.numel() <= 0:
            loss_theta = torch.tensor(0.0, device=self.device)



        #





        rh = torch.masked_select(r_h_mean, sm)  # [?]
        rd = torch.masked_select(r_d_mean, sm)  # [?]
        loss_radius = torch.tensor(0.0, device=self.device)
        if rh.numel() > 0 and rd.numel() > 0:

            loss_radius = -(torch.log(rh + 1e-6).mean() +
                            torch.log(rd + 1e-6).mean())



        #
        # conf_target = 1 - |pred_shortcut - target|


        #




        conf_sel = torch.masked_select(conf, sm)  # [?]
        loss_conf = torch.tensor(0.0, device=self.device)
        if conf_sel.numel() > 0:

            pred_error = torch.abs(pred_shortcut.detach() - target)

            conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
            loss_conf, pred_conf = self.cal_loss4(
                self.W_conf, feature4, conf_target, sm)



        #   loss = loss_shortcut



        #





        lambda_theta  = float(self.other_config.get("lambda_theta", 0.3))
        lambda_radius = float(self.other_config.get("lambda_radius", 0.001))
        lambda_conf   = float(self.other_config.get("lambda_conf", 0.05))

        loss = (loss_shortcut
                + lambda_theta * loss_theta
                + lambda_radius * loss_radius
                + lambda_conf * loss_conf)


        return pred_shortcut, target, loss

    # ============================================================



    # ============================================================
