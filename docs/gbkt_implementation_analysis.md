# GBKT 当前实现逐步解析（含维度流）

本文针对当前仓库中的 GBKT 实现进行逐步拆解，覆盖：
- 模块构成
- 单步前向计算路径
- 时间循环中的状态更新
- 输出张量与 trainer 损失对接
- 在当前配置下的数值化维度示例

源码入口：
- 模型主体：[models/gbkt.py](../models/gbkt.py#L33)
- Trainer： [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L11)
- 配置： [configs/kt_config.json](../configs/kt_config.json#L140)

---

## 1. 记号与当前超参数

记号：
- B: batch size
- T: 输入序列长度（完整序列）
- e: emb_size
- d_h: 知识状态球维度
- d_g: 题目难度球维度
- d_p: 预测空间维度

当前 gbkt 配置（来自配置文件）：
- emb_size = 64，见 [configs/kt_config.json](../configs/kt_config.json#L142)
- d_h = 128，见 [configs/kt_config.json](../configs/kt_config.json#L143)
- d_g = 64，见 [configs/kt_config.json](../configs/kt_config.json#L144)
- d_p = 64，见 [configs/kt_config.json](../configs/kt_config.json#L145)

---

## 2. 从 Trainer 到模型输入

Trainer 在一个 batch 内将原序列与 shifted 序列拼接成完整序列：
- 拼接函数：[_concat_full](../core/trainers/gbkt_trainer.py#L129)
- 生成 q_full： [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L158)
- 生成 c_full： [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L159)
- 生成 r_full： [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L160)
- 模型调用： [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L165)

输入给模型 forward 的主要张量：
- q: [B, T]
- c: [B, T] 或 [B, T, K]
- r: [B, T]

模型入口： [forward](../models/gbkt.py#L240)

---

## 3. 模块结构与参数维度

### 3.1 Embedding 层

定义位置：
- question_emb: [models/gbkt.py](../models/gbkt.py#L67)
- concept_emb: [models/gbkt.py](../models/gbkt.py#L68)
- response_emb: [models/gbkt.py](../models/gbkt.py#L69)

输出维度：
- question_emb(q): [..., e]
- concept_emb(c): [..., e]
- response_emb(r_idx): [..., 2e]

### 3.2 QDB（Question Difficulty Ball）

定义位置：
- question_difficulty_ball: [models/gbkt.py](../models/gbkt.py#L152)

映射：
- q_repr -> mu_d: Linear(2e -> d_g)
- q_repr -> r_d: Linear(2e -> d_g) 再 softplus 保正

输出：
- mu_d: [B, T, d_g]
- r_d: [B, T, d_g]

### 3.3 KSB（Knowledge State Ball）

关键层定义：
- GRUCell: [models/gbkt.py](../models/gbkt.py#L82)
- 半径更新门 W_gamma: [models/gbkt.py](../models/gbkt.py#L83)
- concept_key: [models/gbkt.py](../models/gbkt.py#L84)
- concept_gate: [models/gbkt.py](../models/gbkt.py#L85)

状态变量：
- mu_h: [B, d_h]
- r_h: [B, d_h]

注意：当前 gbkt 是全局单状态向量，不是 per-concept 独立状态矩阵。

### 3.4 BBP（Ball-to-Ball Prediction）

定义位置：
- ball_to_ball_predict: [models/gbkt.py](../models/gbkt.py#L164)
- shortcut 线性层: [models/gbkt.py](../models/gbkt.py#L105)

流程：
1) 将知识球和难度球都投影到 d_p 空间
2) 计算 center_diff 与 radius_sum
3) 通过 IRT 风格参数 theta, b, a 构建 logit_irt
4) 加上 shortcut 分支
5) sigmoid 得到 p_hat

### 3.5 KAB（Knowledge Acquisition Ball）

定义位置：
- knowledge_acquisition_ball: [models/gbkt.py](../models/gbkt.py#L195)
- W_mu_delta: [models/gbkt.py](../models/gbkt.py#L113)
- W_pl2h: [models/gbkt.py](../models/gbkt.py#L114)
- W_r_delta: [models/gbkt.py](../models/gbkt.py#L115)

输出：
- mu_delta: [B, d_h]
- r_delta: [B, d_h]
- plausibility: [B, d_p]

---

## 4. 前处理函数与输入规范化

### 4.1 id 清洗

- 题目 id 清洗： [_sanitize_question_ids](../models/gbkt.py#L117)
- 概念 id 清洗： [_sanitize_concept_ids](../models/gbkt.py#L122)
- 响应转索引： [_response_to_index](../models/gbkt.py#L128)

作用：
- 无效 id 统一映射到 padding 索引，避免 embedding 越界
- 响应值通过阈值转 0/1 索引

### 4.2 多概念聚合

函数： [get_avg_concept_emb](../models/gbkt.py#L131)

输入分支：
- 若 c 是 [B, T] 或 [B]，直接 embedding
- 若 c 是 [B, T, K]，对 K 维做 mask 平均，得到 [B, T, e]

---

## 5. get_question_repr 与 QDB 输出

函数： [get_question_repr](../models/gbkt.py#L145)

步骤：
1) qemb = question_emb(qidx) -> [B, T, e]
2) cemb = get_avg_concept_emb(c) -> [B, T, e]
3) shallow = concat(qemb + cemb, qemb * cemb) -> [B, T, 2e]
4) 经过 q_repr_mlp 后残差相加，输出 q_repr_all -> [B, T, 2e]

然后进入 QDB：
- mu_d_all, r_d_all = question_difficulty_ball(q_repr_all)
- mu_d_all, r_d_all 维度均为 [B, T, d_g]

对应代码：
- q_repr_all 生成： [models/gbkt.py](../models/gbkt.py#L263)
- mu_d_all/r_d_all 生成： [models/gbkt.py](../models/gbkt.py#L264)

---

## 6. forward 主循环逐步拆解（按 t 时刻）

循环位置： [models/gbkt.py](../models/gbkt.py#L273)

### 6.1 状态初始化

- mu_h 初始化： [models/gbkt.py](../models/gbkt.py#L266)
- r_h 初始化： [models/gbkt.py](../models/gbkt.py#L267)

维度：
- mu_h: [B, d_h]
- r_h: [B, d_h]

### 6.2 取当前时刻输入

在每个 t：
- q_t, r_t: [B]
- q_repr_t: [B, 2e]
- mu_d_t, r_d_t: [B, d_g]

### 6.3 当前时刻概念嵌入 c_emb_t

分支代码：
- 单概念路径： [models/gbkt.py](../models/gbkt.py#L280)
- 多概念路径： [models/gbkt.py](../models/gbkt.py#L282)

结果：
- c_emb_t: [B, e]

### 6.4 KAB 计算 mu_delta 与 r_delta

调用位置： [models/gbkt.py](../models/gbkt.py#L286)

在 KAB 内部：
1) 投影到 d_p 空间：mu_h_p, r_h_p, mu_d_p, r_d_p -> [B, d_p]
2) center_diff = mu_h_p - mu_d_p -> [B, d_p]
3) radius_sum = r_h_p + r_d_p -> [B, d_p]
4) effective_diff = center_diff / (radius_sum + eps) -> [B, d_p]
5) sign 由 r_t 生成，sign -> [B, 1]
6) pl_input = concat(sign*center_diff, effective_diff, radius_sum, sign)
   - 维度 [B, 3d_p+1]
7) plausibility = sigmoid(W_pl(pl_input)) -> [B, d_p]
8) e_r = response_emb(r_idx) -> [B, 2e]
9) acq_input = concat(q_repr_t, e_r, mu_h) -> [B, 4e + d_h]
10) mu_delta_raw = W_mu_delta(acq_input) -> [B, d_h]
11) pl_gate = sigmoid(W_pl2h(plausibility)) -> [B, d_h]
12) mu_delta = pl_gate * mu_delta_raw -> [B, d_h]
13) r_delta = softplus(W_r_delta(1 - plausibility)) -> [B, d_h]

### 6.5 KSB 更新全局状态

调用位置： [models/gbkt.py](../models/gbkt.py#L289)

在 update_state 内部（函数在 [models/gbkt.py](../models/gbkt.py#L221)）：
1) e_r = response_emb(r_idx) -> [B, 2e]
2) x_t = concat(q_repr_t + e_r, mu_delta) -> [B, 2e + d_h]
3) mu_h_gru = GRUCell(x_t, mu_h) -> [B, d_h]
4) c_key = sigmoid(concept_key(c_emb_t)) -> [B, d_h]
5) gate_input = concat(mu_h_gru - mu_h, c_key * mu_h) -> [B, 2d_h]
6) c_gate = sigmoid(concept_gate(gate_input)) -> [B, d_h]
7) update_mask = c_key * c_gate -> [B, d_h]
8) mu_h_next = (1 - update_mask) * mu_h + update_mask * mu_h_gru -> [B, d_h]
9) gamma_input = concat(mu_h, r_h, q_repr_t + e_r) -> [B, 2d_h + 2e]
10) gamma = sigmoid(W_gamma(gamma_input)) -> [B, d_h]
11) r_h_next = (1 - gamma*c_key) * r_h + (gamma*c_key) * r_delta -> [B, d_h]

### 6.6 历史注意力增强

代码位置： [models/gbkt.py](../models/gbkt.py#L293)

- 若已有历史，mu_h_history = stack(history, dim=1)
- 若历史长度为 H：mu_h_history 维度 [B, H, d_h]
- 当前 query 是 mu_h_next.unsqueeze(1) -> [B, 1, d_h]
- 注意力输出后回到 [B, d_h]

历史注意力模块定义： [models/gbkt.py](../models/gbkt.py#L9)

### 6.7 padding 位置屏蔽更新

代码位置： [models/gbkt.py](../models/gbkt.py#L300)

- valid_cur = (q_t >= 0).float().unsqueeze(-1) -> [B, 1]
- 通过广播将无效位置保留旧状态：
  - mu_h = valid_cur * mu_h_next + (1 - valid_cur) * mu_h
  - r_h 同理

### 6.8 下一题预测

代码位置：
- 取下一步难度球： [models/gbkt.py](../models/gbkt.py#L304)
- 调用 BBP： [models/gbkt.py](../models/gbkt.py#L305)

每步输出：
- p_hat, theta, confidence: [B]
- r_h_mean_step, r_d_mean_step: [B]

### 6.9 时间维堆叠返回

返回位置： [models/gbkt.py](../models/gbkt.py#L314)

最终输出张量：
- y: [B, T-1]
- theta: [B, T-1]
- confidence: [B, T-1]
- r_h_mean: [B, T-1]
- r_d_mean: [B, T-1]

---

## 7. BBP 的几何与 IRT 映射公式

定义位置： [ball_to_ball_predict](../models/gbkt.py#L164)

关键量：
- center_diff = mu_h_p - mu_d_p
- radius_sum = r_h_p + r_d_p
- effective_diff = center_diff / (radius_sum + eps)

IRT 相关：
- theta = W_theta(effective_diff)
- b = W_b(center_diff)
- a = softplus(W_a(-radius_sum))
- confidence = sigmoid(W_conf(-radius_sum))

主 logit：
- logit_irt = a * (theta - b)

shortcut：
- shortcut_input = concat(mu_h_p, mu_d_p, center_diff, mu_h_p * mu_d_p)
- shortcut = W_shortcut(shortcut_input)
- logit = logit_irt + 0.1 * shortcut
- p_hat = sigmoid(logit)

---

## 8. Trainer 如何消费模型输出

入口： [_forward_batch](../core/trainers/gbkt_trainer.py#L134)

步骤：
1) 构造 q_full/c_full/r_full 并调用模型
2) 取 y/theta/confidence/r_h_mean/r_d_mean
3) 按 common_len 截断对齐，见 [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L172)
4) 用 sm 掩码拉平：
   - pred = masked_select(y, sm)
   - target = masked_select(rshft, sm)

损失项：
- loss_pred: BCE(pred, target)，见 [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L191)
- loss_theta: BCE(sigmoid(theta), target)，见 [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L195)
- loss_radius: -log(r_h_mean) - log(r_d_mean)（log-barrier），见 [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L206)
- loss_conf: confidence 校准 MSE，见 [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L209)

总损失：
- loss = loss_pred + lambda_theta * loss_theta + lambda_radius * loss_radius + lambda_conf * loss_conf
- 权重读取： [core/trainers/gbkt_trainer.py](../core/trainers/gbkt_trainer.py#L215)

---

## 9. 当前配置下的数值化维度速查

在 e=64, d_h=128, d_g=64, d_p=64 时：

- q_repr_t: [B, 128]
- KAB 中 pl_input: [B, 193]（3*64 + 1）
- KAB 中 acq_input: [B, 384]（4*64 + 128）
- update_state 中 x_t: [B, 256]（128 + 128）
- update_state 中 gamma_input: [B, 384]（2*128 + 128）
- BBP 中 shortcut_input: [B, 256]（4*64）

---

## 10. 实现形态总结

你当前 gbkt 的本质是：
- 全局单知识球状态（每个学生每步一个 [d_h]）
- 概念通过 concept_key/concept_gate 做软选择更新
- 通过球几何关系 + IRT + shortcut 进行下一题预测
- 输出多辅助量，配合多项损失联合训练

这也是它与旧版 per-concept 状态矩阵方案最核心的结构差异。