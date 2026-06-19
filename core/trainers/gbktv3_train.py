import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainers.gbktv2_train import GBKTV2Trainer


@TRAINER_REGISTRY.register("gbktv3")
class GBKTV3Trainer(GBKTV2Trainer):
    """Trainer for GBKTV3 with time features and optional question diagnostics."""

    use_radius_loss = True
    use_conf_loss = True

    def _uses_question_aux(self):
        question_fusion = float(getattr(self.model, "max_question_fusion_weight", 0.0)) > 0.0
        return question_fusion

    def _log_epoch(self, metrics_dict):
        super()._log_epoch(metrics_dict)
        if not bool(self.other_config.get("log_aux_metrics", True)):
            return

        aux_keys = [
            "valid_ball_auc",
            "valid_concept_next_auc",
            "valid_fusion_concept_weight_mean",
        ]
        if self._uses_question_aux():
            aux_keys.extend(["valid_question_next_auc", "valid_fusion_question_weight_mean"])
        available = [(key, metrics_dict[key]) for key in aux_keys if key in metrics_dict]
        if not available:
            return

        print("  GBKTV3 diagnostics:")
        for key, value in available:
            label = key.replace("valid_", "").replace("_", " ")
            print(f"    {label}: {value:.4f}" if value >= 0 else f"    {label}: N/A")
        print("")

    @staticmethod
    def _branch_metrics(target, score, prefix):
        if target.size == 0 or score.size == 0:
            return {f"{prefix}_auc": -1, f"{prefix}_acc": -1}
        try:
            auc = metrics.roc_auc_score(y_true=target, y_score=score)
        except Exception:
            auc = -1
        pred_label = (score >= 0.5).astype(np.int64)
        acc = metrics.accuracy_score(target, pred_label)
        return {f"{prefix}_auc": auc, f"{prefix}_acc": acc}

    def _score_loader(self, loader, prefix):
        if loader is None:
            return {f"{prefix}_auc": -1, f"{prefix}_acc": -1}

        self.model.eval()
        y_true = []
        y_score = []
        aux_scores = {
            "ball": [],
            "concept_next": [],
        }
        aux_weights = {
            "fusion_concept_weight": [],
        }
        if self._uses_question_aux():
            aux_scores["question_next"] = []
            aux_weights["fusion_question_weight"] = []

        with torch.no_grad():
            for batch in loader:
                pred, target, _, details = self._forward_batch(batch, return_details=True)
                if pred.numel() == 0:
                    continue
                y_score.append(pred.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy())
                for key in aux_scores:
                    value = details.get(key)
                    if value is not None and value.numel() > 0:
                        aux_scores[key].append(value.detach().cpu().numpy())
                for key in aux_weights:
                    value = details.get(key)
                    if value is not None and value.numel() > 0:
                        aux_weights[key].append(value.detach().cpu().numpy())

        if not y_true:
            return {f"{prefix}_auc": -1, f"{prefix}_acc": -1}

        ts = np.concatenate(y_true, axis=0)
        ps = np.concatenate(y_score, axis=0)
        result = self._branch_metrics(ts, ps, prefix)
        for key, values in aux_scores.items():
            if values:
                score = np.concatenate(values, axis=0)
                result.update(self._branch_metrics(ts, score, f"{prefix}_{key}"))
        for key, values in aux_weights.items():
            if values:
                result[f"{prefix}_{key}_mean"] = float(np.concatenate(values, axis=0).mean())
        return result

    def _forward_batch(self, batch, return_details=False):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        itseqs = batch.get("itseqs")
        itshft = batch.get("shft_itseqs")
        utseqs = batch.get("utseqs")
        utshft = batch.get("shft_utseqs")
        sm = batch["smasks"]

        if qseqs is not None:
            qseqs = qseqs.to(self.device)
        if cseqs is not None:
            cseqs = cseqs.to(self.device)
        rseqs = rseqs.to(self.device)
        if qshft is not None:
            qshft = qshft.to(self.device)
        if cshft is not None:
            cshft = cshft.to(self.device)
        rshft = rshft.to(self.device)
        if itseqs is not None:
            itseqs = itseqs.to(self.device)
        if itshft is not None:
            itshft = itshft.to(self.device)
        if utseqs is not None:
            utseqs = utseqs.to(self.device)
        if utshft is not None:
            utshft = utshft.to(self.device)
        sm = sm.to(self.device)

        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)
        it_full = self._concat_full(itseqs, itshft)
        ut_full = self._concat_full(utseqs, utshft)

        if q_full is None or c_full is None:
            raise ValueError("GBKTV3Trainer requires both question and concept sequences.")

        outputs = self.model(
            q_full.long(),
            c_full.long(),
            r_full.float(),
            it=it_full.long() if it_full is not None else None,
            ut=ut_full.long() if ut_full is not None else None,
        )
        y = outputs["y"]
        y_ball = outputs.get("y_ball")
        y_concept_next = outputs.get("y_concept_next")
        y_question_next = outputs.get("y_question_next")
        fusion_concept_weight = outputs.get("fusion_concept_weight")
        fusion_question_weight = outputs.get("fusion_question_weight")
        theta = outputs["theta"]
        conf = outputs["confidence"]
        r_h_mean = outputs.get("r_h_mean")
        r_d_mean = outputs.get("r_d_mean")
        item_difficulty = outputs.get("item_difficulty")

        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            loss = torch.tensor(0.0, device=self.device)
            if return_details:
                return empty, empty, loss, {}
            return empty, empty, loss

        y = y[:, :common_len]
        if y_ball is not None:
            y_ball = y_ball[:, :common_len]
        if y_concept_next is not None:
            y_concept_next = y_concept_next[:, :common_len]
        if y_question_next is not None:
            y_question_next = y_question_next[:, :common_len]
        if fusion_concept_weight is not None:
            fusion_concept_weight = fusion_concept_weight[:, :common_len]
        if fusion_question_weight is not None:
            fusion_question_weight = fusion_question_weight[:, :common_len]
        if item_difficulty is not None:
            item_difficulty = item_difficulty[:, :common_len]
        theta = theta[:, :common_len]
        conf = conf[:, :common_len]
        rshft = rshft[:, :common_len]
        sm = sm[:, :common_len]
        if r_h_mean is not None:
            r_h_mean = r_h_mean[:, :common_len]
        else:
            r_h_mean = torch.ones_like(y)
        if r_d_mean is not None:
            r_d_mean = r_d_mean[:, :common_len]
        else:
            r_d_mean = torch.ones_like(y)

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm).float()
        if pred.numel() == 0:
            loss = torch.tensor(0.0, device=self.device)
            if return_details:
                return pred, target, loss, {}
            return pred, target, loss

        loss_pred = binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), target)

        loss_ball = torch.tensor(0.0, device=self.device)
        ball_pred = None
        if y_ball is not None:
            ball_pred = torch.masked_select(y_ball, sm)
            loss_ball = binary_cross_entropy(ball_pred.clamp(1e-5, 1.0 - 1e-5), target)

        loss_concept_next = torch.tensor(0.0, device=self.device)
        concept_pred = None
        if y_concept_next is not None:
            concept_pred = torch.masked_select(y_concept_next, sm)
            loss_concept_next = binary_cross_entropy(concept_pred.clamp(1e-5, 1.0 - 1e-5), target)

        question_pred = None
        if y_question_next is not None:
            question_pred = torch.masked_select(y_question_next, sm)
        fusion_concept_sel = None
        fusion_question_sel = None
        if fusion_concept_weight is not None:
            fusion_concept_sel = torch.masked_select(fusion_concept_weight, sm)
        if fusion_question_weight is not None:
            fusion_question_sel = torch.masked_select(fusion_question_weight, sm)

        theta_prob = torch.sigmoid(theta)
        theta_pred = torch.masked_select(theta_prob, sm)
        loss_theta = (
            binary_cross_entropy(theta_pred.clamp(1e-5, 1.0 - 1e-5), target)
            if theta_pred.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )

        lambda_theta = float(self.other_config.get("lambda_theta", 0.05))
        lambda_radius = (
            float(self.other_config.get("lambda_radius", 0.001))
            if self.use_radius_loss
            else 0.0
        )
        lambda_conf = (
            float(self.other_config.get("lambda_conf", 0.02))
            if self.use_conf_loss
            else 0.0
        )
        lambda_ball = float(self.other_config.get("lambda_ball", 0.2))
        lambda_concept_next = float(self.other_config.get("lambda_concept_next", 0.2))
        lambda_item_difficulty = float(self.other_config.get("lambda_item_difficulty", 0.0))

        loss_radius = torch.tensor(0.0, device=self.device)
        if lambda_radius > 0.0:
            rh = torch.masked_select(r_h_mean, sm)
            rd = torch.masked_select(r_d_mean, sm)
            loss_radius = -(torch.log(rh + 1e-6).mean() + torch.log(rd + 1e-6).mean())

        loss_conf = torch.tensor(0.0, device=self.device)
        if lambda_conf > 0.0:
            conf_sel = torch.masked_select(conf, sm)
            pred_error = torch.abs(pred.detach() - target)
            conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
            loss_conf = mse_loss(conf_sel.float(), conf_target.float())

        loss_item_difficulty = torch.tensor(0.0, device=self.device)
        if lambda_item_difficulty > 0.0 and item_difficulty is not None:
            item_difficulty_sel = torch.masked_select(item_difficulty, sm)
            if item_difficulty_sel.numel() > 0:
                loss_item_difficulty = item_difficulty_sel.pow(2).mean()

        loss = (
            loss_pred
            + lambda_ball * loss_ball
            + lambda_concept_next * loss_concept_next
            + lambda_theta * loss_theta
            + lambda_radius * loss_radius
            + lambda_conf * loss_conf
            + lambda_item_difficulty * loss_item_difficulty
        )
        if return_details:
            details = {
                "ball": ball_pred,
                "concept_next": concept_pred,
                "question_next": question_pred,
                "fusion_concept_weight": fusion_concept_sel,
                "fusion_question_weight": fusion_question_sel,
            }
            return pred, target, loss, details
        return pred, target, loss
