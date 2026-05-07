# GBKT 在 ASSIST2017 表现最好的原因与改进方向

> 生成时间：2026-04-27  
> 依据文件：`experiment/last_test_summary_wide.csv`、`data/doc/KT_DATASET_ANALYSIS_LOCAL.md`、`models/gbkt.py`、`core/trainers/gbkt_trainer.py`

## 1. 先看实验事实

从 `last_test_summary_wide.csv` 看，GBKT 并不是所有数据集都差，而是“在适配的数据集上很强，在部分数据形态上吃亏”。

| 数据集 | GBKT best AUC | 最优模型 | 最优 AUC | GBKT 排名 | 与最优差距 |
|---|---:|---|---:|---:|---:|
| algebra2005 | 0.8239 | AKT | 0.8371 | 3 | -0.0132 |
| assist2009 | 0.7787 | QIKT | 0.7832 | 2 | -0.0045 |
| assist2017 | 0.7935 | GBKT | 0.7935 | 1 | 0 |
| nips_task34 | 0.8013 | QIKT | 0.8015 | 2 | -0.0002 |
| peiyou | - | QIKT | 0.8483 | - | 未见 GBKT 结果 |

所以更准确的结论是：

- GBKT 在 ASSIST2017 明显最强；
- 在 NIPS34 几乎与 QIKT 打平；
- 在 ASSIST2009 接近最优；
- 在 Algebra2005 明显落后于 AKT；
- Peiyou 目前没有 GBKT 汇总结果，不能判断。

## 2. GBKT 当前实现的核心假设

当前 `models/gbkt.py` 的 GBKT 本质是：

1. 每个学生维护一个全局知识状态球 `mu_h/r_h`，不是每个概念一个独立状态。
2. 题目难度球由 `question_emb + concept_emb` 共同生成。
3. 概念只通过 `concept_key` 和 `concept_gate` 影响全局状态更新。
4. 预测端用球间几何关系 + IRT 风格参数 + shortcut。
5. 多概念题目只是对多个 concept embedding 做平均，而不是显式建多个概念状态。

这套结构天然适合：

- 题目数量适中；
- 单概念为主；
- 学生序列相对长；
- 正负样本不极端偏高；
- 难度/能力区分明显的数据。

ASSIST2017 正好比较接近这个形态。

## 3. 为什么 ASSIST2017 最适合 GBKT

ASSIST2017 的本地统计特征：

| 指标 | 数值 |
|---|---:|
| 学生数 | 1,708 |
| 题目数 | 3,162 |
| 概念数 | 102 |
| 交互数 | 942,785 |
| 最大概念数 | 1 |
| 正确率 | 37.27% |
| 平均序列长度 | 170.27 |
| 中位序列长度 | 200 |

这些特征与 GBKT 的结构高度匹配。

第一，ASSIST2017 是单概念数据。GBKT 当前不是 per-concept 状态矩阵，多概念只做平均聚合。因此在 ASSIST2017 上不会因为多概念混合而损失概念粒度。

第二，题目数只有 3,162，题目嵌入足够可学习。GBKT 的 QDB 依赖 question embedding 生成难度球，ASSIST2017 的题目空间不大，每个题目获得的训练信号相对充分。

第三，正确率只有 37.27%，难度区分强。GBKT 的球几何和 IRT 分支本质上在建模“能力球能否覆盖/接近难度球”。当数据中题目难度、学生能力差异明显时，这个归纳偏置有优势。

第四，序列较长，平均 170，中位数 200。GBKT 的历史状态更新和 historical ball attention 能从长序列中得到足够历史信号。

第五，ASSIST2017 的原始数据行为特征丰富，即使当前序列只用核心字段，它的数据生成过程也可能让学生能力轨迹更清晰，适合状态空间模型。

## 4. 为什么其他数据集不如 ASSIST2017

### 4.1 Algebra2005：题目极多，多概念明显，GBKT 的全局状态吃亏

Algebra2005 有 173,113 个题目、112 个概念、最大 7 个概念。它的问题是题目粒度非常细，题目由 problem + step 构成。

GBKT 在这里落后 AKT 约 0.0132 AUC，主要原因可能是：

- question embedding 极稀疏，17 万题目对 88 万交互，很多题目训练样本很少；
- 多概念最多 7 个，但当前 GBKT 只是平均 concept embedding，概念贡献被混合；
- 全局单知识球难以同时表达 112 个概念的细粒度状态；
- AKT 的 attention 对长历史和题目上下文匹配更强，更适合细粒度步骤数据。

### 4.2 ASSIST2009：序列短且预处理无时间，GBKT 只接近最优

ASSIST2009 有 17,737 个题目、123 个概念、最大 4 个概念，平均序列长度 72.39，中位数只有 35。

GBKT 在这里排名第 2，落后 QIKT 0.0045。主要原因：

- 序列比 ASSIST2017 短，历史球 attention 的优势不够充分；
- 多概念存在，但 GBKT 没有显式多概念状态；
- 当前序列无 timestamp，GBKT 也没有使用响应耗时或时间间隔；
- QIKT 的 question/concept 双通道输出更直接，适合 ASSIST2009 这种经典 q-c 混合数据。

### 4.3 NIPS34：GBKT 已经很接近最优，但缺少元数据利用

NIPS34 只有 948 个题目、57 个概念、最大 2 个概念，GBKT 只比 QIKT 低 0.0002 AUC，几乎打平。

这里的瓶颈不是 GBKT 结构完全不适配，而是：

- NIPS34 有 question metadata 和 subject tree；
- 当前 GBKT 没有利用概念层级、题目元数据、subject 父子关系；
- QIKT 对 question/concept 两个预测目标更直接，因此略占优势。

### 4.4 Peiyou：目前没有 GBKT 汇总结果，建议补跑

Peiyou 有 7,633 个题目、865 个概念、最大 6 个概念，交互 575 万。它对 GBKT 是一个高风险但值得试的数据集。

风险在于：

- 概念数 865，远高于 ASSIST2017；
- 多概念最多 6 个；
- 当前全局单状态可能不足；
- 时间范围有异常早期 timestamp，需要先清洗。

如果直接用当前 GBKT，可能会被 QIKT、SAINT+ 这类更强的 question-aware 或 attention 模型压制。

## 5. 当前 GBKT 的主要短板

### 5.1 缺少 per-concept 知识状态

当前每个学生只有一个 `mu_h/r_h`。概念通过 gate 影响更新，但状态本身不是 `[num_c, hidden]`。

这会导致：

- 单概念数据上还好；
- 多概念数据上不同概念的掌握状态容易互相污染；
- 概念数很大时，全局状态容量不足。

这解释了为什么 ASSIST2017 强，而 Algebra2005、Peiyou 这类多概念/多题目数据更难。

### 5.2 多概念处理过于简单

`get_avg_concept_emb` 对多概念做平均。平均会丢掉：

- 主概念/辅概念差异；
- 概念顺序或权重；
- 多概念之间的组合难度；
- 哪些概念应该被更新。

对 Algebra2005、Bridge、Peiyou，这会明显限制上限。

### 5.3 题目难度球强依赖 question embedding

在 Algebra2005 这种 17 万题目的数据上，question embedding 很容易稀疏。GBKT 的 QDB 直接从题目表示生成难度球，如果题目 embedding 学不好，难度球也会不稳。

ASSIST2017 只有 3,162 个题目，所以这个问题不明显。

### 5.4 没有使用时间与耗时

数据分析文档显示，ASSIST2017、Algebra2005、Bridge、Junyi、NIPS、Peiyou 都有 timestamp，ASSIST2012 还有响应耗时。但 GBKT 当前 forward 只吃 `q/c/r`，没有时间间隔、遗忘和耗时。

对长跨度数据，时间缺失会让状态更新过于“等间隔”。

### 5.5 训练上存在过拟合迹象

GBKT 的 best 与 last 差距较明显：

- Algebra2005: best AUC 0.8239，last AUC 0.8000；
- ASSIST2009: best AUC 0.7787，last AUC 0.7615；
- ASSIST2017: best AUC 0.7935，last AUC 0.7887；
- NIPS34: best AUC 0.8013，last AUC 0.7986。

这说明 GBKT 后期继续训练会退化，尤其在 Algebra2005 和 ASSIST2009 上更明显。当前配置对不同数据集共用 `lr=1e-3, dropout=0.2, lambda_theta=0.1, lambda_radius=0.001, lambda_conf=0.05`，没有按数据集调参。

## 6. 优先级最高的改进路线

### 6.1 做 per-concept GBKT，这是最关键的结构升级

把状态从：

```text
mu_h: [B, d_h]
r_h:  [B, d_h]
```

升级为：

```text
mu_h: [B, num_c, d_h]
r_h:  [B, num_c, d_h]
```

每次只更新当前题目相关概念。预测下一题时，对下一题相关概念做 attention/readout，得到局部知识球，再和题目难度球比较。

预期收益：

- ASSIST2017 不一定大幅提升，但应保持；
- Algebra2005、Bridge、Peiyou 这类多概念数据会更受益；
- 模型解释性更强，可以输出每个概念的掌握球。

### 6.2 多概念不要平均，改成 attention / gated pooling

把当前简单平均：

```text
c_emb = mean(concept_embs)
```

改为：

```text
alpha_k = softmax(f(q_emb, c_emb_k, state_c_k))
c_emb = sum(alpha_k * c_emb_k)
state_readout = sum(alpha_k * state_c_k)
```

这样模型能学习一个题目中哪个概念更关键。对 Algebra2005、Bridge、Peiyou 最有价值。

### 6.3 给 QDB 加题目难度先验，缓解稀疏题目

对题目很多的数据，单纯 question embedding 不稳。可以加入：

- 题目历史正确率；
- 题目交互次数；
- 概念平均正确率；
- question bias；
- difficulty scalar embedding。

QDB 输入从：

```text
[q_emb + c_emb, q_emb * c_emb]
```

扩展为：

```text
[q_emb, c_emb, q_emb*c_emb, item_correct_rate, item_count_log, concept_correct_rate]
```

这对 Algebra2005 最关键。

### 6.4 加时间衰减/遗忘门

在状态更新前加入时间间隔 `delta_t`：

```text
forget_gate = exp(-softplus(w_c) * log1p(delta_t))
mu_h = forget_gate * mu_h
r_h = r_h + uncertainty_growth(delta_t)
```

直觉是：时间越久，知识中心衰减，不确定半径变大。

优先在有 timestamp 的 ASSIST2017、NIPS34、Algebra2005、Bridge、Peiyou 上试。ASSIST2009 当前序列无 timestamp，需要先改预处理。

### 6.5 按数据集调训练策略

建议至少分三组配置：

| 数据类型 | 数据集 | 建议 |
|---|---|---|
| 单概念、题目适中 | ASSIST2017、NIPS34 | 保持当前结构，调小辅助损失 |
| 题目极多、稀疏 | Algebra2005、Bridge | 降学习率到 5e-4 或 3e-4，加 weight decay，加 item difficulty prior |
| 多概念、多概念数大 | Peiyou、Bridge | per-concept 状态 + concept attention |

训练上建议：

- `learning_rate`: 先试 `5e-4`、`3e-4`；
- `dropout`: Algebra/ASSIST2009 试 `0.3`；
- `lambda_theta`: 试 `0.05`、`0.0`，避免 theta 分支过强牵制主 BCE；
- `lambda_radius`: 试 `0.0001`、`0.0`，观察半径是否被 log-barrier 放大；
- `patience`: 保持早停，以 best checkpoint 为准，不要汇报 last。

## 7. 推荐实验顺序

第一阶段：不改结构，只做诊断和轻量调参。

1. 补跑 Peiyou 的 GBKT。
2. 对 Algebra2005、ASSIST2009 做 `lr/dropout/lambda_theta/lambda_radius` 网格。
3. 记录 `r_h_mean/r_d_mean/confidence/theta` 的分布，看是否半径过大或 theta 分支失效。
4. 加一个 “no auxiliary loss” 版本，只保留 BCE，验证辅助损失是否拖累泛化。

第二阶段：改多概念聚合。

1. 把平均 concept embedding 改成 attention pooling。
2. 对 Algebra2005、Bridge、Peiyou 优先测试。
3. 保留 ASSIST2017 作为 sanity check，确保不损害单概念优势。

第三阶段：升级 per-concept GBKT。

1. 实现 `[B, num_c, d_h]` 状态。
2. 当前题目相关概念局部更新。
3. 下一题相关概念 attention readout。
4. 在 Algebra2005、Bridge、Peiyou 上验证是否缩小与 AKT/QIKT/SAINT+ 的差距。

第四阶段：加入时间机制。

1. 从现有 timestamp 构造 `delta_t`。
2. 在 update_state 前加遗忘门和半径增长。
3. 优先测试 ASSIST2017、NIPS34、Peiyou。

## 8. 论文/汇报中的解释口径

可以这样表述：

GBKT 在 ASSIST2017 上取得最优，主要是因为 ASSIST2017 具有单概念、题目规模适中、长序列、正确率较低且难度区分明显等特点，与 GBKT 的全局知识球、题目难度球和球间 IRT 判别机制高度匹配。相比之下，Algebra2005 的题目空间极大且多概念严重，ASSIST2009 序列较短且当前预处理缺少时间信息，Peiyou 概念数和多概念复杂度更高，这些数据形态都暴露了当前 GBKT 的全局单状态、多概念平均聚合和题目嵌入稀疏问题。因此，下一步应围绕 per-concept 知识球、多概念注意力聚合、题目难度先验和时间遗忘机制进行升级。

## 9. 最重要的结论

当前 GBKT 的优势不是“通用 attention 建模”，而是“能力球-难度球的几何判别”。ASSIST2017 的数据形态刚好让这个归纳偏置发挥出来；其他数据集不够好，主要不是想法错了，而是当前实现还没有处理好多概念、大题目空间、时间间隔和数据集自适应训练这四个问题。
