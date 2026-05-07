import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.registry import MODEL_REGISTRY


# ============================================================
# HistoricalBallAttention: 历史知识状态中心的 Attention
# ─────────────────────────────────────────────────────────────
# 作用: 对当前时刻的知识状态中心 mu_h_current,
#       做一次 Multi-Head Attention, 聚合历史所有时刻的
#       知识状态中心 (mu_h_history), 输出增强后的中心向量。
#
# 维度变化:
#   mu_h_current     : [B, d_h]
#   mu_h_history     : [B, T_hist, d_h]
#   query            : [B, 1, d_h]  (unsqueeze后)
#   attn_out         : [B, 1, d_h]
#   enhanced         : [B, 1, d_h]
#   return           : [B, d_h]     (squeeze后)
# ============================================================
class HistoricalBallAttention(nn.Module):
    """Attention over historical knowledge-state centers."""

    def __init__(self, d_h, n_heads=4, dropout=0.1):
        super().__init__()
        # Multi-Head Attention: d_h维输入, n_heads个头, batch_first模式
        self.attn = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 残差连接后的 LayerNorm
        self.norm = nn.LayerNorm(d_h)

    def forward(self, mu_h_current, mu_h_history):
        # mu_h_current : [B, d_h]        当前时刻知识状态中心
        # mu_h_history : [B, T_hist, d_h] 历史知识状态中心列表
        # ──────────────────────────────────────────────────────
        # 无历史记录时直接透传 (第一个时间点)
        if mu_h_history.size(1) == 0:
            return mu_h_current  # [B, d_h]

        # Query: 当前时刻中心 (扩展一个序列维度作为"查询")
        query = mu_h_current.unsqueeze(1)  # [B, 1, d_h]

        # Multi-Head Attention: query关注key=value=历史中心
        # attn_out: [B, 1, d_h]
        attn_out, _ = self.attn(query=query, key=mu_h_history, value=mu_h_history)

        # 残差 + LayerNorm
        enhanced = self.norm(query + attn_out)  # [B, 1, d_h]
        return enhanced.squeeze(1)  # [B, d_h]


# ============================================================
# GBSVKT: Granular Ball Support Vector Knowledge Tracing
# ─────────────────────────────────────────────────────────────
# 模型结构:
#   QDB (Question Difficulty Ball)   : 题目难度球 — 建模题目难度
#   KSB (Knowledge State Ball)      : 知识状态球 — 建模学生掌握程度
#   BBP (Ball-to-Ball Prediction)    : 球间预测 — 用两球关系做IRT预测
#   KAB (Knowledge Acquisition Ball) : 知识获取球 — 建模答题后的状态更新
#
# 核心思想: 将学生知识状态和题目难度建模为"球"
#   - 球心(mu): 代表中心知识/难度水平
#   - 半径(r):  代表不确定性/掌握/难度范围
#   - 两球关系: 用归一化距离 (mu_h-mu_d)/(r_h+r_d) 衡量学生是否会做该题
# ============================================================
@MODEL_REGISTRY.register("gbsvkt")
class GBSVKT(nn.Module):
    """Granular Ball Support Vector Knowledge Tracing.

    This implementation follows the GBSV-KT design:
    - Question Difficulty Ball (QDB)
    - Knowledge State Ball (KSB)
    - Ball-to-Ball Prediction (BBP)
    - Knowledge Acquisition Ball (KAB)
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size=64,
        d_h=128,
        d_g=64,
        d_p=64,
        dropout=0.2,
        init_radius=0.5,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()

        # ─────────────────────────────────────────────────────
        # 超参保存
        # ─────────────────────────────────────────────────────
        self.num_q   = int(num_q)      # 题目数量 (含padding idx = num_q)
        self.num_c   = int(num_c)      # 概念数量 (含padding idx = num_c)
        self.emb_size = int(emb_size)   # embedding 维度
        self.d_h      = int(d_h)       # 知识状态隐维度 (GRU hidden)
        self.d_g      = int(d_g)       # QDB 投影维度
        self.d_p      = int(d_p)       # BBP 投影维度 (统一空间)
        self.eps      = float(eps)     # 数值稳定项 (防止除零)

        # ─────────────────────────────────────────────────────
        # 基础 Embedding 查找表
        # question_emb: [num_q+1, emb_size], padding_idx=num_q (题目padding ID映射到零向量)
        self.question_emb = nn.Embedding(self.num_q + 1, self.emb_size, padding_idx=self.num_q)
        # concept_emb:  [num_c+1, emb_size], padding_idx=num_c (概念padding ID映射到零向量)
        self.concept_emb  = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=self.num_c)
        # response_emb: [2, 2*emb_size] (r=0/correct 和 r=1/incorrect 各一个向量)
        self.response_emb = nn.Embedding(2, 2 * self.emb_size)

        # ─────────────────────────────────────────────────────
        # QDB: Question Difficulty Ball (题目难度球)
        # 将题目表示 q_repr ([B,T,2*emb]) 投影为:
        #   mu_d: 难度中心   [B,T,d_g]
        #   r_d : 难度半径   [B,T,d_g] (softplus保证正数)
        # ─────────────────────────────────────────────────────
        self.W_mu_d = nn.Linear(2 * self.emb_size, self.d_g)  # 难度中心投影
        self.W_r_d = nn.Linear(2 * self.emb_size, self.d_g)  # 难度半径投影 (→ softplus → 正)
        # 两层MLP做题目表示的非线性变换 (含残差连接在get_question_repr中)
        self.q_repr_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_size, 2 * self.emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.emb_size, 2 * self.emb_size),
        )

        # ─────────────────────────────────────────────────────
        # KSB: Knowledge State Ball (知识状态球)
        # 用GRU建模学生知识状态随时间步的演化
        # ─────────────────────────────────────────────────────
        # GRU: [2*emb + d_h] → d_h  (输入=题目响应交互+状态增量)
        self.gru_mu = nn.GRUCell(2 * self.emb_size + self.d_h, self.d_h)
        # 半径更新门: [2*d_h + 2*emb] → d_h  (决定半径如何根据答题结果调整)
        self.W_gamma = nn.Linear(2 * self.d_h + 2 * self.emb_size, self.d_h)
        # 概念感知的key: [emb] → d_h (概念对状态更新的贡献程度)
        self.concept_key   = nn.Linear(self.emb_size, self.d_h)
        # 概念感知的gate: [2*d_h] → d_h (控制历史状态的保留比例)
        self.concept_gate   = nn.Linear(2 * self.d_h, self.d_h)
        # 可学习的初始知识状态中心 (batch展开后作为t=0的隐状态)
        self.mu_h0 = nn.Parameter(torch.zeros(self.d_h))
        # 可学习的初始半径 (log空间初始化, softplus后保证正数)
        init_radius = max(float(init_radius), 1e-6)
        init_val = math.log(math.exp(init_radius) - 1.0)  # softplus的逆
        self.r_h0 = nn.Parameter(torch.full((self.d_h,), init_val))
        self.state_dropout = nn.Dropout(dropout)

        # 历史中心Attention: 聚合历史所有时间步的知识状态中心
        self.ball_attn = HistoricalBallAttention(self.d_h, n_heads=4, dropout=dropout)

        # ─────────────────────────────────────────────────────
        # BBP: Ball-to-Ball Prediction (球到球预测)
        # 将四个球 (mu_h,r_h,mu_d,r_d) 投影到统一 d_p 空间后计算关系
        # ─────────────────────────────────────────────────────
        self.W_proj_h  = nn.Linear(self.d_h, self.d_p)   # mu_h  → d_p
        self.W_proj_rh = nn.Linear(self.d_h, self.d_p)   # r_h   → d_p (→ softplus → 正)
        self.W_proj_d  = nn.Linear(self.d_g, self.d_p)   # mu_d  → d_p
        self.W_proj_rd = nn.Linear(self.d_g, self.d_p)   # r_d   → d_p (→ softplus → 正)

        # BBP 输出头
        self.W_b = nn.Linear(self.d_p, 1)  # 题目难度偏置 b
        self.W_a = nn.Linear(self.d_p, 1)  # 能力尺度参数 a (→ softplus → 正)

        # ─────────────────────────────────────────────────────
        # 三个SVM预测器的可学习参数
        # Trainer (gbsvkt_trainer_intro.py) 直接引用这些参数计算损失
        # ─────────────────────────────────────────────────────
        # W_shortcut: [4*d_p, 1]  (feature1=shortcut_input=[mu_h,mu_d,cd,mul] 维度4*d_p)
        self.W_shortcut = torch.nn.Parameter(torch.randn(4 * self.d_p, 1) * 0.01)
        self.b_shortcut = torch.nn.Parameter(torch.zeros(1))

        # W_theta: [d_p, 1]  (feature2=effective_diff 维度 d_p)
        self.W_theta = torch.nn.Parameter(torch.randn(self.d_p, 1) * 0.01)
        self.b_theta = torch.nn.Parameter(torch.zeros(1))

        # W_conf: [d_p, 1]  (feature4=-radius_sum 维度 d_p)
        self.W_conf = torch.nn.Parameter(torch.randn(self.d_p, 1) * 0.01)
        self.b_conf = torch.nn.Parameter(torch.zeros(1))

        # ─────────────────────────────────────────────────────
        # KAB: Knowledge Acquisition Ball (知识获取球)
        # 根据当前题目难度球和响应, 计算知识状态的增量 (mu_delta, r_delta)
        # ─────────────────────────────────────────────────────
        # Plausibility MLP: [3*d_p+1] → d_p → d_p (预测答题后的状态变更幅度)
        self.W_pl = nn.Sequential(
            nn.Linear(3 * self.d_p + 1, self.d_p),
            nn.ReLU(),
            nn.Linear(self.d_p, self.d_p),
        )
        # 状态中心增量: [4*emb + d_h] → d_h
        self.W_mu_delta = nn.Linear(4 * self.emb_size + self.d_h, self.d_h)
        # Plausibility → 状态更新门: d_p → d_h
        self.W_pl2h = nn.Linear(self.d_p, self.d_h)
        # 状态半径增量: d_p → d_h (→ softplus → 正)
        self.W_r_delta = nn.Linear(self.d_p, self.d_h)

    # ============================================================
    # _sanitize_question_ids: 题目ID边界检查 + padding替换
    # ─────────────────────────────────────────────────────────────
    # 输入: q [B, T]  原始题目ID (可能含-1等无效值)
    # 输出: q [B, T]  有效ID保留, 无效ID替换为num_q (padding_idx)
    # ============================================================
    def _sanitize_question_ids(self, q):
        q = q.long()
        # 创建全padding的tensor (每个元素都是num_q)
        pad_q = torch.full_like(q, self.num_q)
        # 有效ID: 0 <= q < num_q; 无效: 替换为num_q (查表得零向量)
        return torch.where((q >= 0) & (q < self.num_q), q, pad_q)

    # ============================================================
    # _sanitize_concept_ids: 概念ID边界检查 + padding替换
    # ─────────────────────────────────────────────────────────────
    # 同上, 针对概念ID
    # ============================================================
    def _sanitize_concept_ids(self, c):
        c = c.long()
        pad_c = torch.full_like(c, self.num_c)
        return torch.where((c >= 0) & (c < self.num_c), c, pad_c)

    # ============================================================
    # _response_to_index: 响应值 → 离散index (0/1)
    # ─────────────────────────────────────────────────────────────
    # r: float tensor (0.0 或 1.0)
    # → r_idx: long tensor (0 或 1)
    # ============================================================
    @staticmethod
    def _response_to_index(r):
        return (r > 0.5).long().clamp(min=0, max=1)

    # ============================================================
    # get_avg_concept_emb: 概念序列 → embedding (支持多种格式)
    # ─────────────────────────────────────────────────────────────
    # 输入:
    #   c: [B, T]       每时刻单个概念ID
    #   或 [B, T, K]    每时刻多个概念ID (K个概念同时学习)
    #
    # 输出:
    #   cemb: [B, T, emb_size]  (每时刻一个embedding向量)
    #
    # 维度变化:
    #   cidx      : [B, T] 或 [B, T, K]
    #   cemb      : [B, T, emb] 或 [B, T, K, emb]
    #   cmask     : [B, T, K, 1]  (padding位置为0)
    #   csum      : [B, T, emb]   (沿K求和)
    #   cnum      : [B, T, 1]     (沿K计数)
    #   return    : [B, T, emb]  (平均后)
    # ============================================================
    def get_avg_concept_emb(self, c):
        cidx = self._sanitize_concept_ids(c)
        cemb = self.concept_emb(cidx)  # [B,T,emb] 或 [B,T,K,emb]

        # 情况A: [B, T] 或 [B] → 直接返回 (已经是2D)
        if cidx.dim() <= 2:
            return cemb  # [B, T, emb_size]

        # 情况B: [B, T, K] → 多概念取平均
        #   cmask: 有效概念位置=1, padding位置=0
        cmask = (cidx != self.num_c).float().unsqueeze(-1)  # [B, T, K, 1]
        csum  = (cemb * cmask).sum(dim=-2)                  # [B, T, emb] 有效emb之和
        cnum  = cmask.sum(dim=-2).clamp(min=1.0)             # [B, T, 1] 有效概念数量
        return csum / cnum  # [B, T, emb] 每时刻的平均概念embedding

    # ============================================================
    # get_question_repr: 题目表示构建 (shallow + MLP残差)
    # ─────────────────────────────────────────────────────────────
    # 输入:
    #   q: [B, T]  题目ID
    #   c: [B, T]  概念ID
    # 输出:
    #   q_repr: [B, T, 2*emb_size]
    #
    # 维度变化:
    #   qidx       : [B, T]
    #   qemb       : [B, T, emb]
    #   cemb       : [B, T, emb]
    #   shallow    : concat([B,T,emb]+[B,T,emb], [B,T,emb]*[B,T,emb]) → [B, T, 2*emb]
    #   q_repr     : shallow + mlp(shallow) → [B, T, 2*emb]
    # ============================================================
    def get_question_repr(self, q, c):
        qidx = self._sanitize_question_ids(q)
        qemb = self.question_emb(qidx)         # [B, T, emb]
        cemb = self.get_avg_concept_emb(c)      # [B, T, emb]

        # shallow: 加性交互 + 乘性交互 (element-wise product)
        #   qemb + cemb : [B, T, emb]
        #   qemb * cemb : [B, T, emb]
        #   concat      : [B, T, 2*emb]
        shallow = torch.cat([qemb + cemb, qemb * cemb], dim=-1)

        # 残差连接: 原始交互 + MLP非线性变换
        return shallow + self.q_repr_mlp(shallow)  # [B, T, 2*emb]

    # ============================================================
    # question_difficulty_ball: QDB前向
    # ─────────────────────────────────────────────────────────────
    # 输入: q_repr [B, T, 2*emb]
    # 输出:
    #   mu_d: [B, T, d_g]  难度中心
    #   r_d : [B, T, d_g]  难度半径 (softplus → 严格正)
    # ============================================================
    def question_difficulty_ball(self, q_repr):
        mu_d = self.W_mu_d(q_repr)              # [B, T, d_g]
        r_d  = F.softplus(self.W_r_d(q_repr))  # [B, T, d_g] (正值)
        return mu_d, r_d

    # ============================================================
    # _project_balls: 球投影到统一 d_p 空间
    # ─────────────────────────────────────────────────────────────
    # 输入:
    #   mu_h: [B, d_h]  知识状态中心
    #   r_h : [B, d_h]  知识状态半径
    #   mu_d: [B, d_g]  题目难度中心
    #   r_d : [B, d_g]  题目难度半径
    # 输出:
    #   mu_h_p: [B, d_p]  投影后知识状态中心
    #   r_h_p : [B, d_p]  投影后知识状态半径 (softplus → 正)
    #   mu_d_p: [B, d_p]  投影后难度中心
    #   r_d_p : [B, d_p]  投影后难度半径 (softplus → 正)
    # ============================================================
    def _project_balls(self, mu_h, r_h, mu_d, r_d):
        mu_h_p = self.W_proj_h(mu_h)                 # [B, d_p]
        r_h_p  = F.softplus(self.W_proj_rh(r_h))     # [B, d_p] (正)
        mu_d_p = self.W_proj_d(mu_d)                 # [B, d_p]
        r_d_p  = F.softplus(self.W_proj_rd(r_d))     # [B, d_p] (正)
        return mu_h_p, r_h_p, mu_d_p, r_d_p

    # ============================================================
    # pred1: 最终预测概率 (SVM-style, 有偏置)
    # ─────────────────────────────────────────────────────────────
    # 用于计算最终预测 p_hat = pred1(W_shortcut, b_shortcut, logit_irt, shortcut_input)
    #
    # 输入:
    #   w        : [4*d_p, 1]  (W_shortcut)
    #   b        : [1]          (b_shortcut)
    #   logit_irt: [B]          IRT logit = a * (theta - b)
    #   y        : [B, 4*d_p]   (shortcut_input = [mu_h_p, mu_d_p, cd, mu_h_p*mu_d_p])
    # 输出:
    #   y        : [B]  (归一化预测概率)
    #
    # 维度变化:
    #   torch.matmul(y, w)      : [B, 4*d_p] × [4*d_p,1] → [B, 1]
    #   +b / norm_w             : [B, 1]
    #   squeeze(-1)             : [B]
    #   logit_irt + 0.1*y       : [B] + [B] → [B]
    #   sigmoid                 : [B] → [B] (概率)
    # ============================================================
    def pred1(self, w, b, logit_irt, y):
        norm_w = torch.norm(w)  # 标量, w的L2范数
        y = (torch.matmul(y, w) + b) / norm_w  # [B, 1] 归一化投影
        y = y.squeeze(-1)  # [B]
        y = logit_irt + 0.1 * y  # [B] IRT logits + 0.1*shortcut
        y = torch.sigmoid(y)  # [B] 归一化为概率
        return y

    # ============================================================
    # pred2: 能力参数 theta (SVM-style, 无偏置)
    # ─────────────────────────────────────────────────────────────
    # 用于计算 theta = pred2(W_theta, b_theta, effective_diff)
    #
    # 输入:
    #   w: [d_p, 1]  (W_theta)
    #   b: [1]       (b_theta)
    #   y: [B, d_p]  (effective_diff = (mu_h_p - mu_d_p)/(r_h_p+r_d_p))
    # 输出:
    #   y: [B]  (归一化能力参数)
    # ============================================================
    def pred2(self, w, b, y):
        norm_w = torch.norm(w)
        y = (torch.matmul(y, w) + b) / norm_w  # [B, d_p] × [d_p,1] → [B, 1]
        y = torch.sigmoid(y)  # [B, 1] → [B]
        return y

    # ============================================================
    # pred4: 置信度logit (无激活, 线性输出)
    # ─────────────────────────────────────────────────────────────
    # 用于计算 confidence = sigmoid(pred4(W_conf, b_conf, -radius_sum))
    #
    # 输入:
    #   w: [d_p, 1]  (W_conf)
    #   b: [1]       (b_conf)
    #   y: [B, d_p]  (-radius_sum = -(r_h_p + r_d_p))
    # 输出:
    #   y: [B, 1]
    # ============================================================
    def pred4(self, w, b, y):
        y = (torch.matmul(y, w) + b)  # [B, d_p] × [d_p,1] → [B, 1]
        return y

    # ============================================================
    # ball_to_ball_predict: BBP核心预测
    # ─────────────────────────────────────────────────────────────
    # 用当前知识状态球 (mu_h, r_h) 预测下一题难度球 (mu_d, r_d)
    #
    # 输入:
    #   mu_h, r_h: [B, d_h]  当前知识状态球
    #   mu_d, r_d: [B, d_g]  目标题目难度球
    # 输出: dict
    #   p_hat        : [B]     最终预测概率 (0~1)
    #   theta        : [B]     IRT能力参数
    #   b            : [B]     题目难度偏置
    #   a            : [B]     能力尺度 (>0)
    #   confidence   : [B]     预测置信度 (0~1)
    #   feature1     : [B, 4*d_p]  → cal_loss1 (shortcut)
    #   feature2     : [B, d_p]    → cal_loss2 (theta)
    #   feature4     : [B, d_p]    → cal_loss4 (confidence)
    # ============================================================
    def ball_to_ball_predict(self, mu_h, r_h, mu_d, r_d):
        # Step 1: 投影到统一 d_p 空间
        # mu_h_p, mu_d_p: [B, d_p]
        # r_h_p, r_d_p  : [B, d_p] (正值)
        mu_h_p, r_h_p, mu_d_p, r_d_p = self._project_balls(mu_h, r_h, mu_d, r_d)

        # Step 2: 球间归一化距离
        # center_diff  = mu_h_p - mu_d_p          : [B, d_p]
        # radius_sum   = r_h_p + r_d_p            : [B, d_p] (正值)
        # effective_diff = center_diff / (radius_sum + eps) : [B, d_p]
        center_diff   = mu_h_p - mu_d_p
        radius_sum    = r_h_p + r_d_p
        effective_diff = center_diff / (radius_sum + self.eps)

        # Step 3: theta (IRT能力参数) — 归一化距离的SVM投影
        #   pred2: [B,d_p] → [B,1] → sigmoid → [B]
        theta = self.pred2(self.W_theta, self.b_theta, effective_diff).squeeze(-1)

        # Step 4: b (题目难度偏置) = W_b(center_diff)
        #   W_b: [d_p, 1] → [B, d_p] × [d_p, 1] = [B, 1] → [B]
        b = self.W_b(center_diff).squeeze(-1)

        # Step 5: a (能力尺度) = softplus(W_a(-radius_sum)) > 0
        #   球半径和越大 → a越小 → 预测越不确定
        a = F.softplus(self.W_a(-radius_sum)).squeeze(-1)

        # Step 6: confidence = sigmoid(pred4)
        #   球半径和越大 → -radius_sum越小 → pred4越小 → confidence越低
        confidence = torch.sigmoid(
            self.pred4(self.W_conf, self.b_conf, -radius_sum).squeeze(-1))

        # Step 7: IRT logit = a * (theta - b)
        logit_irt = a * (theta - b)  # [B]

        # Step 8: shortcut_input (交互特征拼接)
        #   [mu_h_p, mu_d_p, center_diff, mu_h_p * mu_d_p]
        #   4个[B,d_p] → concat → [B, 4*d_p]
        shortcut_input = torch.cat([
            mu_h_p, mu_d_p, center_diff, mu_h_p * mu_d_p
        ], dim=-1)

        # Step 9: 最终预测概率
        #   pred1(logit_irt, shortcut_input) → [B]
        p_hat = self.pred1(self.W_shortcut, self.b_shortcut, logit_irt, shortcut_input)

        return {
            "p_hat": p_hat,
            "theta": theta,
            "b": b,
            "a": a,
            "confidence": confidence,
            "mu_h_p": mu_h_p,
            "mu_d_p": mu_d_p,
            "r_h_p": r_h_p,
            "r_d_p": r_d_p,
            "radius_sum": radius_sum,
            "feature1": shortcut_input,   # [B, 4*d_p] → cal_loss1 (shortcut)
            "feature2": effective_diff,    # [B, d_p]   → cal_loss2 (theta)
            "feature4": -radius_sum,      # [B, d_p]   → cal_loss4 (confidence)
        }

    # ============================================================
    # knowledge_acquisition_ball: KAB (知识获取球)
    # ─────────────────────────────────────────────────────────────
    # 根据当前知识状态、题目难度和响应, 计算状态增量
    #
    # 输入:
    #   mu_h, r_h: [B, d_h]  当前知识状态球
    #   mu_d, r_d: [B, d_g]  当前题目难度球
    #   q_repr   : [B, 2*emb] 当前题目表示
    #   r_t      : [B]        当前响应 (0或1)
    # 输出:
    #   mu_delta : [B, d_h]  状态中心增量
    #   r_delta  : [B, d_h]  状态半径增量
    #   plausibility: [B, d_p]  答题"合理性" (用于控制更新幅度)
    #
    # 维度变化:
    #   pl_input   : [B, 3*d_p+1]
    #   plausibility: [B, d_p]
    #   e_r        : [B, 2*emb]
    #   acq_input  : [B, 4*emb+d_h]
    #   mu_delta_raw: [B, d_h]
    #   pl_gate    : [B, d_h]
    #   mu_delta   : [B, d_h]
    #   r_delta    : [B, d_h] (正值)
    # ============================================================
    def knowledge_acquisition_ball(self, mu_h, r_h, mu_d, r_d, q_repr, r_t):
        # 投影到 d_p 空间
        mu_h_p, r_h_p, mu_d_p, r_d_p = self._project_balls(mu_h, r_h, mu_d, r_d)

        center_diff    = mu_h_p - mu_d_p           # [B, d_p]
        radius_sum     = r_h_p + r_d_p             # [B, d_p]
        effective_diff = center_diff / (radius_sum + self.eps)  # [B, d_p]

        # 响应符号: r=0 → -1, r=1 → +1
        r_idx = self._response_to_index(r_t)       # [B] (0或1)
        sign  = r_idx.float().mul(2.0).sub(1.0).unsqueeze(-1)  # [B, 1]: ±1

        # Plausibility输入: [sign*cd, eff_diff, radius_sum, sign]
        #   concat后: [B, 3*d_p + 1]
        pl_input = torch.cat([
            sign * center_diff, effective_diff, radius_sum, sign,
        ], dim=-1)
        plausibility = torch.sigmoid(self.W_pl(pl_input))  # [B, d_p] (0~1)

        # 状态中心增量
        e_r = self.response_emb(r_idx)                      # [B, 2*emb]
        # acq_input: [q_repr, e_r, mu_h] = [2*emb] + [2*emb] + [d_h]
        acq_input = torch.cat([q_repr, e_r, mu_h], dim=-1)  # [B, 4*emb + d_h]
        mu_delta_raw = self.W_mu_delta(acq_input)          # [B, d_h]
        pl_gate = torch.sigmoid(self.W_pl2h(plausibility))  # [B, d_h] (0~1)
        mu_delta = pl_gate * mu_delta_raw                   # [B, d_h] (门控后的增量)

        # 状态半径增量 (与plausibility负相关: 预测"不应该对则"→ 大幅度更新)
        r_delta = F.softplus(self.W_r_delta(1.0 - plausibility))  # [B, d_h] (正值)

        return mu_delta, r_delta, plausibility

    # ============================================================
    # update_state: KSB状态更新
    # ─────────────────────────────────────────────────────────────
    # 输入:
    #   mu_h, r_h   : [B, d_h]  当前知识状态
    #   q_repr      : [B, 2*emb] 当前题目表示
    #   r_t         : [B]        当前响应
    #   mu_delta    : [B, d_h]  KAB产生的状态增量中心
    #   r_delta     : [B, d_h]  KAB产生的状态增量半径
    #   c_emb_t     : [B, emb]  当前概念embedding
    # 输出:
    #   mu_h_next   : [B, d_h]  更新后的知识状态中心
    #   r_h_next    : [B, d_h]  更新后的知识状态半径
    #
    # 维度变化:
    #   e_r         : [B, 2*emb]
    #   x_t         : [B, 2*emb+d_h]
    #   mu_h_gru    : [B, d_h]
    #   c_key       : [B, d_h]
    #   gate_input  : [B, 2*d_h]
    #   c_gate      : [B, d_h]
    #   update_mask : [B, d_h]
    #   gamma_input : [B, 2*d_h+2*emb]
    #   gamma       : [B, d_h]
    # ============================================================
    def update_state(self, mu_h, r_h, q_repr, r_t, mu_delta, r_delta, c_emb_t):
        r_idx = self._response_to_index(r_t)    # [B] (0或1)
        e_r = self.response_emb(r_idx)          # [B, 2*emb]

        # GRU输入: [q_repr + e_r, mu_delta] = [2*emb] + [d_h] = [2*emb+d_h]
        x_t = torch.cat([q_repr + e_r, mu_delta], dim=-1)  # [B, 2*emb + d_h]
        mu_h_gru = self.gru_mu(x_t, mu_h)    # [B, d_h] (GRU更新中心)
        mu_h_gru = self.state_dropout(mu_h_gru)

        # 概念感知的遗忘门 (控制历史状态保留程度)
        c_key = torch.sigmoid(self.concept_key(c_emb_t))  # [B, d_h] (概念相关性)
        gate_input = torch.cat([mu_h_gru - mu_h, c_key * mu_h], dim=-1)  # [B, 2*d_h]
        c_gate = torch.sigmoid(self.concept_gate(gate_input))  # [B, d_h]
        update_mask = c_key * c_gate  # [B, d_h] (综合概念相关性和状态门)

        # 选择性更新: update_mask大 → 更多采纳新状态; 小 → 保留历史
        mu_h_next = (1.0 - update_mask) * mu_h + update_mask * mu_h_gru  # [B, d_h]

        # 半径更新 (根据答题结果和概念相关性调整)
        gamma_input = torch.cat([mu_h, r_h, q_repr + e_r], dim=-1)  # [B, 2*d_h+2*emb]
        gamma = torch.sigmoid(self.W_gamma(gamma_input))  # [B, d_h] (更新强度)
        r_h_next = (1.0 - gamma * c_key) * r_h + (gamma * c_key) * r_delta  # [B, d_h]

        return mu_h_next, r_h_next

    # ============================================================
    # forward: 完整序列前向 (逐时刻循环, T-1步预测)
    # ─────────────────────────────────────────────────────────────
    # 输入:
    #   q: [B, T]  题目ID序列 (必须提供, T>=2)
    #   c: [B, T]  概念ID序列 (必须提供)
    #   r: [B, T]  响应序列 (0.0/1.0)
    # 输出:
    #   y         : [B, T-1]  预测概率
    #   theta     : [B, T-1]  能力参数
    #   confidence: [B, T-1]  置信度
    #   r_h_mean  : [B, T-1]  知识状态半径均值
    #   r_d_mean  : [B, T-1]  题目难度半径均值
    #   feature1  : [B, 4*d_p, T-1]  (送入cal_loss1)
    #   feature2  : [B, d_p, T-1]    (送入cal_loss2)
    #   feature4  : [B, d_p, T-1]    (送入cal_loss4)
    # ============================================================
    def forward(self, q, c, r):
        if q is None or c is None:
            raise ValueError("GBKT requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()

        if q.dim() != 2:
            raise ValueError(f"GBKT expects q shape [B, T], but got {tuple(q.shape)}")

        batch_size, seq_len = q.shape  # T = 序列长度
        device = q.device

        # 序列长度<2时无有效预测目标, 返回空tensor
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty, "theta": empty, "confidence": empty,
                "r_h_mean": empty, "r_d_mean": empty,
            }

        # ─────────────────────────────────────────────────────
        # Step 1: 预处理整个序列 (一次性计算, 节省重复计算)
        # ─────────────────────────────────────────────────────
        # q_repr_all: [B, T, 2*emb]  所有时间步的题目表示
        q_repr_all = self.get_question_repr(q, c)
        # mu_d_all/r_d_all: [B, T, d_g]  所有时间步的题目难度球
        mu_d_all, r_d_all = self.question_difficulty_ball(q_repr_all)

        # 初始化知识状态 (从可学习参数展开)
        # mu_h0: [d_h] → unsqueeze(0) → [1,d_h] → expand → [B, d_h]
        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        # r_h0: [d_h] → softplus → unsqueeze → expand → [B, d_h]
        r_h = F.softplus(self.r_h0).unsqueeze(0).expand(batch_size, -1)

        # 输出列表 (沿T-1个时间步收集)
        p_list, theta_list, conf_list = [], [], []
        r_h_list, r_d_list = [], []
        feature1_list, feature2_list, feature4_list = [], [], []
        mu_h_history_list = []

        # ──────────────────────────────────────────────────────
        # 逐时刻循环 (T-1步, 每步用t时刻状态预测t+1时刻)
        # ──────────────────────────────────────────────────────
        for t in range(seq_len - 1):
            # ── t时刻的输入 ──
            q_t   = q[:, t]                      # [B]
            r_t   = r[:, t]                      # [B]
            q_repr_t = q_repr_all[:, t, :]        # [B, 2*emb]
            mu_d_t   = mu_d_all[:, t, :]          # [B, d_g]
            r_d_t    = r_d_all[:, t, :]           # [B, d_g]

            # 概念embedding (支持2D/3D两种concept输入格式)
            if c.dim() == 2:
                c_emb_t = self.get_avg_concept_emb(c[:, t])     # [B, emb]
            else:
                # c: [B, T, K] → 取第t时刻 → [B, K] → 平均 → [B, emb]
                c_emb_t = self.get_avg_concept_emb(
                    c[:, t:t + 1, :]).squeeze(1)

            # ── KAB: 计算知识获取增量 ──
            mu_delta, r_delta, _ = self.knowledge_acquisition_ball(
                mu_h, r_h, mu_d_t, r_d_t, q_repr_t, r_t)
            # mu_delta: [B, d_h]   状态中心增量
            # r_delta : [B, d_h]   状态半径增量

            # ── KSB: 更新知识状态 ──
            mu_h_next, r_h_next = self.update_state(
                mu_h, r_h, q_repr_t, r_t, mu_delta, r_delta, c_emb_t)
            # mu_h_next, r_h_next: [B, d_h]

            # ── Historical Ball Attention ──
            # 用历史所有时刻的mu_h增强当前中心 (避免长期遗忘)
            if mu_h_history_list:
                mu_h_history = torch.stack(mu_h_history_list, dim=1)  # [B, t, d_h]
                mu_h_next = self.ball_attn(mu_h_next, mu_h_history)  # [B, d_h]

            # ── 掩码: 题目ID无效时保持上一时刻状态 ──
            valid_cur = (q_t >= 0).float().unsqueeze(-1)  # [B, 1]: 有效=1, padding=0
            mu_h = valid_cur * mu_h_next + (1.0 - valid_cur) * mu_h  # [B, d_h]
            r_h = valid_cur * r_h_next + (1.0 - valid_cur) * r_h      # [B, d_h]

            # detach: 避免构建O(T^2)级别的计算图 (梯度仅传一步)
            mu_h_history_list.append(mu_h.detach())

            # ── BBP: 用当前状态预测 t+1 时刻的题目 ──
            mu_d_next = mu_d_all[:, t + 1, :]   # [B, d_g]
            r_d_next  = r_d_all[:, t + 1, :]    # [B, d_g]
            pred = self.ball_to_ball_predict(mu_h, r_h, mu_d_next, r_d_next)

            # ── 收集输出 ──
            p_list.append(pred["p_hat"])         # [B] × (T-1)
            theta_list.append(pred["theta"])      # [B] × (T-1)
            conf_list.append(pred["confidence"])  # [B] × (T-1)
            r_h_list.append(r_h.mean(dim=-1))     # [B] (沿d_h维度平均)
            r_d_list.append(r_d_next.mean(dim=-1)) # [B]
            feature1_list.append(pred["feature1"]) # [B, 4*d_p] × (T-1)
            feature2_list.append(pred["feature2"]) # [B, d_p]   × (T-1)
            feature4_list.append(pred["feature4"]) # [B, d_p]   × (T-1)

        # ── 沿时间维度stack ──
        return {
            "y":         torch.stack(p_list,      dim=1),  # [B, T-1]
            "theta":     torch.stack(theta_list,   dim=1),  # [B, T-1]
            "confidence":torch.stack(conf_list,    dim=1),  # [B, T-1]
            "r_h_mean":  torch.stack(r_h_list,     dim=1),  # [B, T-1]
            "r_d_mean":  torch.stack(r_d_list,     dim=1),  # [B, T-1]
            "feature1":  torch.stack(feature1_list, dim=2), # [B, 4*d_p, T-1]
            "feature2":  torch.stack(feature2_list, dim=2),# [B, d_p, T-1]
            "feature4":  torch.stack(feature4_list, dim=2),# [B, d_p, T-1]
        }
