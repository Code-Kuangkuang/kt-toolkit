# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.registry import MODEL_REGISTRY


class mygru(nn.Module):
    """GRU-based knowledge state update."""

    def __init__(self, n_layer, input_dim, hidden_dim):
        super().__init__()
        self.g_ir = funcsgru(n_layer, input_dim, hidden_dim, 0)
        self.g_iz = funcsgru(n_layer, input_dim, hidden_dim, 0)
        self.g_in = funcsgru(n_layer, input_dim, hidden_dim, 0)
        self.g_hr = funcsgru(n_layer, hidden_dim, hidden_dim, 0)
        self.g_hz = funcsgru(n_layer, hidden_dim, hidden_dim, 0)
        self.g_hn = funcsgru(n_layer, hidden_dim, hidden_dim, 0)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, h):
        r_t = self.sigmoid(self.g_ir(x) + self.g_hr(h))
        z_t = self.sigmoid(self.g_iz(x) + self.g_hz(h))
        n_t = self.tanh(self.g_in(x) + self.g_hn(h).mul(r_t))
        h_t = (1 - z_t) * n_t + z_t * h
        return h_t


class funcsgru(nn.Module):
    """MLP for GRU gates."""

    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()
        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class funcs(nn.Module):
    """MLP for predictions."""

    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()
        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class QueEmb(nn.Module):
    """Question and concept embedding module."""

    def __init__(self, num_q, num_c, emb_size, model_name, device='cpu', emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.device = device
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.model_name = model_name

        emb_type_map = {
            "iekt-qc_merge": "qc_merge",
            "iekt-iekt": "iekt",
        }
        tmp_emb_type = f"{model_name}-{emb_type}"
        emb_type = emb_type_map.get(tmp_emb_type, emb_type)
        self.emb_type = emb_type

        self.emb_path = emb_path
        self.pretrain_dim = pretrain_dim

        if emb_type in ["qc_merge", "qaid_qc"]:
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.que_c_linear = nn.Linear(2 * self.emb_size, self.emb_size)

        if emb_type == "iekt":
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size).to(device), requires_grad=True)
            self.que_c_linear = nn.Linear(2 * self.emb_size, self.emb_size)
            self.output_emb_dim = emb_size

        if emb_type.startswith("qid"):
            self.que_emb = nn.Embedding(self.num_q, self.emb_size)
            self.output_emb_dim = emb_size

    def get_avg_skill_emb(self, c):
        """Get average skill embedding."""
        concept_emb_cat = torch.cat(
            [self.concept_emb.new_zeros((1, self.emb_size)), self.concept_emb], dim=0
        )
        related_concepts = (c + 1).long()
        concept_emb = concept_emb_cat[related_concepts, :]

        # IEKT always calls this one timestep at a time, so the input is [B] for
        # a single concept and [B, K] when the question carries several.  The
        # guard used to be `dim() <= 2`, which silently took the single-concept
        # branch for [B, K] and returned [B, K, D] where the caller expects
        # [B, D] -- a shape error further down rather than a wrong number.
        if related_concepts.dim() <= 1:
            return concept_emb

        concept_emb_sum = concept_emb.sum(dim=-2)
        concept_num = (related_concepts != 0).sum(dim=-1, keepdim=True).clamp(min=1)
        concept_avg = concept_emb_sum / concept_num.to(concept_emb_sum.dtype)
        return concept_avg

    def forward(self, q, c, r=None):
        """Forward pass for embeddings."""
        emb_type = self.emb_type

        if emb_type == "iekt":
            concept_avg = self.get_avg_skill_emb(c)
            que_emb = self.que_emb(q)
            emb_qc = torch.cat([que_emb, concept_avg], dim=-1)
            # Keep a 2*emb_size representation for IEKT policy/state modules.
            xemb = torch.cat([self.que_c_linear(emb_qc), que_emb], dim=-1)
        elif "qc_merge" in emb_type:
            concept_avg = self.get_avg_skill_emb(c)
            que_emb = self.que_emb(q)
            xemb = torch.cat([concept_avg, que_emb], dim=-1)
        elif emb_type == "qid":
            que_emb = self.que_emb(q)
            # IEKT downstream expects 2*emb_size regardless of embedding mode.
            xemb = torch.cat([que_emb, que_emb], dim=-1)
        else:
            concept_avg = self.get_avg_skill_emb(c)
            que_emb = self.que_emb(q)
            xemb = torch.cat([concept_avg, que_emb], dim=-1)

        return xemb


class IEKTNet(nn.Module):
    """Item Embedding Knowledge Tracing Network.

    Args:
        num_q: number of questions
        num_c: number of concepts
        emb_size: embedding dimension
        max_concepts: maximum concepts per question
        lamb: hyperparameter for loss
        n_layer: number of MLP hidden layers
        cog_levels: action space for cognition estimation
        acq_levels: action space for sensitivity estimation
        dropout: dropout probability
        gamma: discount factor for RL
        emb_type: embedding type
        emb_path: path to pretrained embeddings
        pretrain_dim: dimension of pretrained embeddings
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        max_concepts,
        lamb=40,
        n_layer=1,
        cog_levels=10,
        acq_levels=10,
        dropout=0,
        gamma=0.93,
        emb_type='qc_merge',
        emb_path="",
        pretrain_dim=768,
        device='cpu',
        **kwargs,
    ):
        super().__init__()
        self.model_name = "iekt"
        self.concept_num = num_c
        self.max_concept = max_concepts
        self.device = device
        self.emb_type = emb_type
        self.gamma = gamma
        self.lamb = lamb

        # Question embedding module (create first to get output dimension)
        self.que_emb = QueEmb(
            num_q=num_q,
            num_c=num_c,
            emb_size=emb_size,
            emb_type=self.emb_type,
            model_name=self.model_name,
            device=device,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim
        )
        # Use output_emb_dim from QueEmb for MLP input dimensions
        emb_out_dim = self.que_emb.output_emb_dim
        self.emb_size = emb_out_dim

        # Predictor
        self.predictor = funcs(n_layer, emb_out_dim * 5, 1, dropout)

        # RL parameters for cognition estimation
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_out_dim * 2).to(self.device), requires_grad=True)
        self.select_preemb = funcs(n_layer, emb_out_dim * 3, cog_levels, dropout)

        # RL parameters for sensitivity estimation
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_out_dim * 2).to(self.device), requires_grad=True)
        self.checker_emb = funcs(n_layer, emb_out_dim * 12, acq_levels, dropout)

        # Question embedding
        self.prob_emb = nn.Parameter(torch.randn(num_q, emb_out_dim).to(self.device), requires_grad=True)

        # GRU for knowledge state update
        self.gru_h = mygru(0, emb_out_dim * 4, emb_out_dim)

        # Concept embedding
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num, emb_out_dim).to(self.device), requires_grad=True)

        self.sigmoid = torch.nn.Sigmoid()

    def get_ques_representation(self, q, c):
        """Get question representation."""
        return self.que_emb(q, c)

    def pi_cog_func(self, x, softmax_dim=1):
        """Policy for cognition estimation."""
        return F.softmax(self.select_preemb(x), dim=softmax_dim)

    def pi_sens_func(self, x, softmax_dim=1):
        """Policy for sensitivity estimation."""
        return F.softmax(self.checker_emb(x), dim=softmax_dim)

    def obtain_v(self, q, c, h, x, emb):
        """Obtain question representation and prediction."""
        v = self.get_ques_representation(q, c)
        predict_x = torch.cat([h, v], dim=1)
        prob = self.predictor(torch.cat([predict_x, emb], dim=1))
        return torch.cat([h, v], dim=1), v, prob, x

    def update_state(self, h, v, emb, operate):
        """Update knowledge state."""
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2))
        ], dim=1)
        e_cat = torch.cat([
            emb.mul((1 - operate).repeat(1, self.emb_size * 2)),
            emb.mul(operate.repeat(1, self.emb_size * 2))
        ], dim=1)
        inputs = v_cat + e_cat
        h_t_next = self.gru_h(inputs, h)
        return h_t_next


@MODEL_REGISTRY.register("iekt")
class IEKT(nn.Module):
    """IEKT wrapper with training logic.

    Args:
        num_q: number of questions
        num_c: number of concepts
        emb_size: embedding dimension
        max_concepts: maximum concepts per question
        lamb: hyperparameter for loss
        n_layer: number of MLP hidden layers
        cog_levels: action space for cognition estimation
        acq_levels: action space for sensitivity estimation
        dropout: dropout probability
        gamma: discount factor for RL
        emb_type: embedding type
        emb_path: path to pretrained embeddings
        pretrain_dim: dimension of pretrained embeddings
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size=64,
        max_concepts=1,
        lamb=40,
        n_layer=1,
        cog_levels=10,
        acq_levels=10,
        dropout=0,
        gamma=0.93,
        emb_type='qc_merge',
        emb_path="",
        pretrain_dim=768,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "iekt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.max_concepts = max_concepts
        self.lamb = lamb
        self.n_layer = n_layer
        self.cog_levels = cog_levels
        self.acq_levels = acq_levels
        self.dropout = dropout
        self.gamma = gamma
        self.emb_type = emb_type

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = IEKTNet(
            num_q=num_q,
            num_c=num_c,
            emb_size=emb_size,
            max_concepts=max_concepts,
            lamb=lamb,
            n_layer=n_layer,
            cog_levels=cog_levels,
            acq_levels=acq_levels,
            dropout=dropout,
            gamma=gamma,
            emb_type=emb_type,
            emb_path=emb_path,
            pretrain_dim=pretrain_dim,
            device=device
        )

        self.sigmoid = torch.nn.Sigmoid()

    def _policy_action(self, probs):
        """Draw a policy action: sampled while training, greedy at evaluation.

        IEKT is trained with REINFORCE, so the action has to be sampled during
        training -- the gradient is defined against that distribution. At
        evaluation the sample only injects noise: the same checkpoint scoring
        the same batch twice differed by 0.287 in prediction space, which makes
        a reported metric one draw from a distribution rather than a value.

        argmax is the standard deterministic reduction of a categorical policy.
        It changes IEKT's evaluation numbers, so results produced before this
        are not comparable with results produced after.
        """
        if self.training:
            return Categorical(probs).sample()
        return probs.argmax(dim=-1)

    def forward(self, data, return_details=False, process=True):
        """Forward pass for training/prediction.

        Args:
            data: dict with keys 'qseqs', 'cseqs', 'rseqs', 'shft_qseqs', 'shft_cseqs', 'shft_rseqs', 'smasks'
            return_details: whether to return detailed RL information
            process: whether to process data

        Returns:
            predictions or (predictions, details)
        """
        data_new = self._batch_to_device(data, process)

        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.model.emb_size).to(self.device)

        batch_probs = []
        uni_prob_list = []
        emb_action_list = []
        p_action_list = []
        states_list = []
        pre_state_list = []
        reward_list = []
        predict_list = []
        ground_truth_list = []

        rt_x = torch.zeros(data_len, 1, self.model.emb_size * 2).to(self.device)

        for seqi in range(seq_len):
            # Get question representation
            ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:, seqi], c=data_new['cc'][:, seqi]),
                h
            ], dim=1)

            # Sample concept embedding (policy gradient)
            flip_prob_emb = self.model.pi_cog_func(ques_h)
            emb_ap = self._policy_action(flip_prob_emb)
            emb_p = self.model.cog_matrix[emb_ap, :]

            # Get prediction
            h_v, v, logits, rt_x = self.model.obtain_v(
                q=data_new['cq'][:, seqi],
                c=data_new['cc'][:, seqi],
                h=h,
                x=rt_x,
                emb=emb_p
            )
            prob = self.sigmoid(logits)

            # Get ground truth
            out_operate_groundtruth = data_new['cr'][:, seqi].unsqueeze(-1)
            ground_truth = data_new['cr'][:, seqi]

            # Sample sensitivity embedding (policy gradient)
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())
            ], dim=1)

            out_operate_logits = torch.where(
                prob > 0.5,
                torch.tensor(1).to(self.device),
                torch.tensor(0).to(self.device)
            )
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1 - out_operate_logits).repeat(1, h_v.size()[-1]).float())
            ], dim=1)
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim=1)

            flip_prob_sens = self.model.pi_sens_func(out_x)
            emb_a = self._policy_action(flip_prob_sens)
            emb = self.model.acq_matrix[emb_a, :]

            # Update knowledge state
            h = self.model.update_state(h, v, emb, ground_truth.unsqueeze(1))

            uni_prob_list.append(prob.detach())
            emb_action_list.append(emb_a)
            p_action_list.append(emb_ap)
            states_list.append(out_x)
            pre_state_list.append(ques_h)
            ground_truth_list.append(ground_truth)
            predict_list.append(logits.squeeze(1))

            this_reward = torch.where(
                out_operate_logits.squeeze(1).float() == ground_truth,
                torch.tensor(1).to(self.device),
                torch.tensor(0).to(self.device)
            )
            reward_list.append(this_reward)

        prob_tensor = torch.cat(uni_prob_list, dim=1)

        if return_details:
            return prob_tensor[:, 1:], data_new, emb_action_list, p_action_list, states_list, pre_state_list, reward_list, predict_list, ground_truth_list
        else:
            return prob_tensor[:, 1:]

    def _batch_to_device(self, data, process=True):
        """Move data to device."""
        if not process:
            return data

        dcur = data
        data_new = {}
        data_new['cq'] = torch.cat((dcur["qseqs"][:, 0:1], dcur["shft_qseqs"]), dim=1)
        data_new['cc'] = torch.cat((dcur["cseqs"][:, 0:1], dcur["shft_cseqs"]), dim=1)
        data_new['cr'] = torch.cat((dcur["rseqs"][:, 0:1], dcur["shft_rseqs"]), dim=1)
        data_new['q'] = dcur["qseqs"]
        data_new['c'] = dcur["cseqs"]
        data_new['r'] = dcur["rseqs"]
        data_new['qshft'] = dcur["shft_qseqs"]
        data_new['cshft'] = dcur["shft_cseqs"]
        data_new['rshft'] = dcur["shft_rseqs"]
        data_new['m'] = dcur["masks"]
        data_new['sm'] = dcur["smasks"]

        for k, v in list(data_new.items()):
            if torch.is_tensor(v):
                data_new[k] = v.to(self.device)
        return data_new

    def train_one_step(self, data, process=True):
        """Train one step with RL loss."""
        BCELoss = torch.nn.BCEWithLogitsLoss()

        prob_tensor, data_new, emb_action_list, p_action_list, states_list, pre_state_list, reward_list, predict_list, ground_truth_list = self.forward(data, return_details=True, process=process)

        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]

        # Number of valid rollout steps per row. The rollout runs over
        # data_new['cc'], which prepends one column to the shifted sequence, so a
        # row with `masks.sum()` valid interactions has one more step than that.
        #
        # This was `(qseqs != 0).sum(-1) + 1`, which treats question id 0 as
        # padding. Zero is a legitimate question id here -- padding is trailing
        # zeros, not the value zero -- so the count came out wrong by a different
        # amount on every row, depending on how many times that learner happened
        # to answer question 0.
        valid_mask = data.get('masks')
        if valid_mask is None:
            valid_mask = data['smasks']
        seq_num = valid_mask.bool().long().sum(dim=-1) + 1

        emb_action_tensor = torch.stack(emb_action_list, dim=1)
        p_action_tensor = torch.stack(p_action_list, dim=1)
        state_tensor = torch.stack(states_list, dim=1)
        pre_state_tensor = torch.stack(pre_state_list, dim=1)
        reward_tensor = torch.stack(reward_list, dim=1).float() / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float()
        logits_tensor = torch.stack(predict_list, dim=1)
        ground_truth_tensor = torch.stack(ground_truth_list, dim=1)

        loss_list = []

        for i in range(data_len):
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]

            this_cog_state = torch.cat([
                pre_state_tensor[i][0: this_seq_len],
                torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
            ], dim=0)
            this_sens_state = torch.cat([
                state_tensor[i][0: this_seq_len],
                torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
            ], dim=0)

            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens.detach().cpu().numpy()

            # Compute advantage for cognition estimation
            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)

            pi_cog = self.model.pi_cog_func(this_cog_state[:-1])
            pi_a_cog = pi_cog.gather(1, p_action_tensor[i][0: this_seq_len].unsqueeze(1))
            loss_cog = -torch.log(pi_a_cog) * advantage_cog
            loss_list.append(torch.sum(loss_cog))

            # Compute advantage for sensitivity estimation
            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)

            pi_sens = self.model.pi_sens_func(this_sens_state[:-1])
            pi_a_sens = pi_sens.gather(1, emb_action_tensor[i][0: this_seq_len].unsqueeze(1))
            loss_sens = -torch.log(pi_a_sens) * advantage_sens
            loss_list.append(torch.sum(loss_sens))

        # Scored positions come from smasks, not from a per-row length.
        #
        # data_new['cc'] is cseqs[:, :1] concatenated with shft_cseqs, so column
        # j + 1 of the rollout predicts shft_rseqs[:, j] -- the target that
        # smasks[:, j] marks. Column 0 predicts the learner's very first
        # response, which no protocol scores.
        #
        # The previous code took a prefix `[0:seq_num[i]]` of the full rollout
        # instead. That both included column 0 and ignored smasks, so it scored
        # padding and the repeated-KC rows that `score_repeated_kc=False` drops.
        # Measured on assist2009 fold 0, first batch of 64: 4556 positions where
        # smasks selects 3886, 17% more. IEKT's AUC therefore covered a
        # different set of positions than every other model in the same table.
        scored = data['smasks'].bool()
        y = logits_tensor[:, 1:][scored]
        y_true = ground_truth_tensor[:, 1:][scored]

        bce = BCELoss(y, y_true)
        label_len = max(y_true.size(0), 1)
        loss_l = sum(loss_list)
        loss = self.lamb * (loss_l / label_len) + bce

        # Logits for the loss above (BCEWithLogitsLoss), probabilities for the
        # caller. BaseTrainer._score_loader thresholds accuracy at `p >= 0.5`,
        # which is only meaningful on a probability; returning raw linear output
        # made every reported IEKT accuracy wrong.
        return torch.sigmoid(y), y_true, loss
