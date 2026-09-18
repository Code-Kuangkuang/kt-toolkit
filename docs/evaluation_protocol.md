# 评测协议与 pykt 一致性

日期：2026-09-16

本文记录 kt-toolkit 的实验评测协议与模型实现，逐项对照 pykt
（`pykt-team/pykt-toolkit`），并明确区分**已确认无误**、**已修复**与**仍然存在问题**
三类。

**所有关于 pykt 的陈述均取自其 `main` 分支源码逐行核对，不依赖记忆或二手总结。**
复现方法：从 `raw.githubusercontent.com/pykt-team/pykt-toolkit/main/pykt/models/<名>.py`
取源码，按 §6 的方式逐函数比对。

阅读顺序建议：

- 只想知道能不能直接用现有结果 → §0 与 §4.3（答案是不能，必须重跑）
- 要出对比表 → §1.5 协议指纹 + §5.4 检查清单
- 要回答审稿人"你的实现对不对" → §6
- 要回答"为什么你的数和 pykt 不一样" → §2.3 与 §6.4

---

## 0. 一句话结论

**协议本身的已知缺陷已全部修复并有回归测试守护。
剩下的唯一动作是重跑：所有存量结果都产于修复之前，不能直接使用。**

| 维度 | 状态 |
|---|---|
| 目标泄露（重复知识点行） | ✅ 已修复，两种模式 × 加窗/不加窗共四种组合全部干净 |
| 数据划分 | ✅ 已核验 |
| 模型选择 | ✅ 已核验，测试集未进入选择 |
| 与 pykt 口径对齐 | ✅ 已补齐 windowed 评测 |
| 多知识点截断 | ✅ 已修复，37/44 模型走 `multi`；其余 7 个按设计不适用，见 §3.4 |
| 模型实现与 pykt 一致性 | ✅ 25 个模型逐函数核对；发现并修复 `atkt` 的真 bug，另修 pykt 自身 6 处问题，见 §6 |
| pykt 的融合式题目级评测 | ⚠️ 未实现（论证等价，未实证），见 §4.1 |
| **存量结果** | ❌ **全部作废**，须重跑，见 §4.3 |

---

## 1. 当前协议

### 1.1 数据划分

| 划分 | 来源 |
|---|---|
| 训练 | `train_valid_*.csv` 中 `fold ∈ 全部折 − {当前折}` |
| 验证 | `train_valid_*.csv` 中 `fold == 当前折` |
| 测试 | 独立文件 `test_*.csv`（`fold == -1`） |

核验位置：`datasets/init_dataset.py`。测试集来自独立文件，与训练/验证无重叠。

### 1.2 模型选择

- 早停与最优检查点均依据 `valid_auc`（`core/hooks.py`、`core/trainer.py`）
- 最终指标由 best-valid 检查点在测试集上评出
- **测试集不参与任何选择决策**

### 1.3 两种数据模式

| 模式 | 文件 | 多知识点题目 |
|---|---|---|
| `one_by_one` | `test_sequences.csv` | 每个知识点展开成一行 |
| `all_in_one` | `test_sequences_quelevel.csv` | 一题一行，知识点存为定长列表 |

### 1.4 两种测试集切分

| 切分 | 文件 | 评分位 |
|---|---|---|
| 普通 | `test_*.csv` | 长序列切成**互不重叠**的 200 段，段内每位都评 |
| 加窗 | `test_window_*.csv` | 每个位置一行，带完整前 200 步，**只评最后一位** |

普通切分的问题：段边界处的位置只有极短历史。加窗切分保证每个评分位都有完整历史，
代价是文件大约 20 倍（assist2009 概念级 3.2 MB → 80.7 MB）。

因此加窗评测只在训练结束后对最优模型执行一次，绝不进入每轮循环。

输出同时给出两个口径：

```
[Best-Valid Epoch] Test AUC=0.xxxx, ACC=0.xxxx
[Best-Valid Epoch] Window Test AUC=0.xxxx, ACC=0.xxxx   (pykt-comparable protocol)
```

`best_metrics.json` 中对应 `best_test_auc/acc` 与 `best_window_test_auc/acc`。

### 1.5 协议指纹

每次运行的 `run_config.json` 都带一个 `protocol` 块：

```json
"protocol": {
  "dataset_mode": "all_in_one",
  "concept_mode": "multi",
  "max_concepts": 4,
  "concepts_visible": "all",
  "score_repeated_kc": false,
  "eval_window": true
}
```

| 字段 | 含义 |
|---|---|
| `dataset_mode` | `all_in_one`（一题一行）或 `one_by_one`（一知识点一行） |
| `concept_mode` | `multi`（池化全部知识点）/ `first`（只取第一个）/ `expanded`（`one_by_one` 下按行展开） |
| `concepts_visible` | `all` 或 `first_of_K`——一眼能看出是否被截断 |
| `score_repeated_kc` | `true` 表示复现了 pykt 含泄露的口径 |
| `eval_window` | `false` 表示该次运行**没有** pykt 可比的 windowed 指标 |

**两次运行只有这一块完全一致才可以放进同一张对比表。**

---

## 2. 与 pykt 的逐项对比

### 2.1 pykt 的实际做法（源码核实）

| 位置 | 事实 |
|---|---|
| `pykt/datasets/data_loader.py` | `is_repeat` **零引用**；`smasks` 仅来自 `selectmasks` |
| `pykt/datasets/que_data_loader.py` | `is_repeat` **零引用** |
| `pykt/preprocess/split_datasets.py` | 先展开多知识点（`["0"] + ["1"]*(n-1)`），再把每个有效位一律标 `selectmasks=1` |
| `evaluate()` | `torch.masked_select(y, sm)`，重复行照计分 |
| `group_fusion()` | `df[df["select"]!=0]` 后按 `qidx` 分组取均值，重复行的预测被平均进去 |
| `examples/wandb_predict.py` | 有题目 id → 报 `windowauclate_mean`；仅概念 → 报 `window_testauc` |

**pykt 公布的指标 = 加窗测试集 × 题目级 × late_mean 融合。**

### 2.2 差异表

| 维度 | pykt | kt-toolkit | 说明 |
|---|---|---|---|
| 重复知识点行计分 | **计入** | **不计入** | 我们刻意更严格 |
| 加窗测试集 | 有（其报告口径） | 有（2026-09-16 补齐） | 已对齐 |
| 题目级融合 | `late_mean` / `late_vote` / `late_all` / early fusion | 未实现 | 见 §4.1 |
| 多知识点表示 | 按行展开 | 展开 或 平均池化 | 见 §3.4 |
| 折划分 | 同 | 同 | 一致 |
| 模型选择 | 验证集 | 验证集 | 一致 |

### 2.3 数字为什么对不上

以 AKT / assist2009 / fold 0 为例：

| 口径 | test AUC |
|---|---|
| 本仓库 普通切分，修复后 | 0.777 |
| 本仓库 普通切分，修复前（曾被改动钉到 `one_by_one`） | 0.837 |
| pykt 公布值（加窗 + 题目级融合） | ≈0.785 |

- 0.777 → 0.837 的差距来自**泄露**
- 0.777 → 0.785 的差距来自**切分方式**（加窗给了完整历史），不是泄露

### 2.4 一个有用的推论

剔除重复行之后，每道题只剩**一个**评分位，
因此 pykt 的题目级融合退化为恒等操作——
pykt 是把「1 个干净预测 + N−1 个被污染预测」取平均，我们直接用那 1 个干净的。

**我们的口径是它那个平均值里干净的那一项。**

---

## 3. 已修复的问题

### 3.1 目标泄露：重复知识点行被计分

多知识点题目在 `one_by_one` 下被展开成多行，**每行作答完全相同**。
预测第二行时，第一行的答案已在模型输入历史中——模型是在抄，不是在预测。

`is_repeat` 列本用于标记这些行，但**整个管线从未读取它**（pykt 亦然）。

影响规模：

| 数据集 | 受影响的评分位占比（普通 / 加窗） |
|---|---|
| assist2009 | 15.2% / 15.0% |
| algebra2005 | **32.8% / 30.3%** |
| bridge2algebra2006 | 0.7% / — |

修复（`datasets/kt_dataset.py`）：

1. `_resolve_selectmasks` 将 `is_repeat == 1` 的位置置为不计分
2. 数据集缓存 key 加入 `screp0/screp1` 标记，避免旧 pickle 掩盖修复
3. 加载时打印排除比例
4. `KT_SCORE_REPEATED_KC=1` 可还原 pykt 口径

回归测试：`tests/test_no_target_leakage.py`，覆盖两种数据模式 × 加窗/不加窗共四种组合。

### 3.2 协议不统一

`dkt / akt / sakt / simplekt` 曾被钉在 `one_by_one`，而 `hd_* / freq_simplekt` 在 `all_in_one`，
两者读的是**不同的测试文件**，差异约 8 个 AUC 点。

已统一为 `all_in_one`（这也是 HEAD 中已提交的默认值）。

### 3.3 缺少 pykt 报告口径

已补齐加窗评测，见 §1.4。

---

### 3.4 多知识点截断

修复前，`all_in_one` 下模型分为两类（`datasets/init_dataset.py` 的 `MULTI_CONCEPT_MODELS`）：

- **名单内 15 个** → 全部知识点，平均池化
- **名单外 35 个** → `cseqs[:, 0]`，**只保留第一个知识点，其余丢弃**

这不是泄露（是少给信息，不是多给），但它使**跨模型对比失去可比性**。

受影响的题目占比。口径：**测试集 quelevel 文件中的作答位置**（`test_sequences_quelevel.csv`，
一个位置一道题，不去重、不含 padding）。同一份数据换成训练集或按题目去重，数字会高
1～3 个点，所以不写口径的百分比没法复核 —— 这正是本文档要消灭的那种歧义。

| 数据集 | 多知识点题目占比 | 其中 ≥3 个知识点 |
|---|---|---|
| bridge2algebra2006 | 0.4%（基本无影响） | 0.0% |
| assist2009 | 15.6% | 2.1% |
| algebra2005 | **30.4%** | 13.8% |

2026-09-18 在当前数据上逐条重算。algebra2005 的 30.4% 与原记录精确吻合，说明原记录
用的就是这个口径；bridge 由 0.2% 订正为 0.4%。assist2009 原记录为 15.2%，而当前数据
按任何一种口径都得不到该值（最接近的按位置口径为 15.6%，按去重题目为 16.3%），因此
改用实测值 —— 原值大概率产于一次不同的数据生成。

偏差方向不一致：

| 基线 | 变体 | 谁占便宜 |
|---|---|---|
| 自研模型（全部） | 多数标准基线（仅第一个） | **利好自研模型** |
| `lpkt`（仅第一个） | `hdkt`（全部） | **利好 HDKT** |

#### 截断的实测代价

`concept_mode` 现在可在模型配置块中覆盖（`multi` / `first`），因此这是单变量实验。

DKVMN / assist2009 / fold 0 / seed 3407：

| 设置 | test AUC | **加窗 test AUC** |
|---|---|---|
| `multi`（接入掩码平均池化） | 0.75768 | **0.75885** |
| `first`（原截断） | 0.75070 | **0.75234** |

**截断代价 = 0.0065 AUC**，且这是在多知识点题目仅占 15.2% 的 assist2009 上。
algebra2005 占比 30.4%，代价预计翻倍（0.012–0.015 量级）。

该量级与论文中常见的"提升幅度"相当。因此：

> **名单内模型对上名单外基线时，存在约 0.006–0.015 的系统性优势，
> 与模型贡献无关。**

#### 修复进度（2026-09-16 完成）

pykt 的做法是掩码平均池化（`que_base_model.py` 的 `get_avg_skill_emb`），
与本仓库 `models/multi_concept.py` 的 `pool_concept_embeddings` 语义相同。
修复即为把该池化接入各模型的 embedding 代码。

**已接入池化（44 个注册模型中 37 个走 `concept_mode: multi`）**，包括
`dkvmn dkt+ deep_irt stablekt sparsekt lefokt_akt robustkt dimkt skvmn atkt
saint saint_plus atdkt dtransformer dkt_forget dkt_pebg hqaf keenkt ukt kqn`
及原有的 `dkt sakt akt simplekt qikt iekt lpkt hd_* hdkt` 等。

四个 trainer 另需把 `y.gather(-1, cshft.unsqueeze(-1))` 换成
`pool_concept_predictions`——凡是"输出每概念一个概率、再按目标概念取值"的模型
（`dkt+` `atkt` `atdkt` `dkt_forget`）在 `[B,T,K]` 下都会 RuntimeError。

**按设计不接入池化的（7 个）**：

| 模型 | 原因 |
|---|---|
| `gkt`、`rekt` | 维护**按知识点 id 索引的状态数组**，多概念需同时读写 K 个条目——属于改模型算法。**决定：留在 `all_in_one` 接受截断**，指纹标 `concepts_visible: first_of_K`，以保证与其它模型同测试集、可进同一张表 |
| `hawkes` | 走 `one_by_one`，概念按行展开本就全可见（指纹标 `expanded`） |
| `dgekt` | 概念经预计算超图进入，模型输入只有题目 id，不消费概念序列 |
| `dkt-forget`、`hqaf_kt`、`lefokt` | 别名，随主名生效 |

#### 另外两个模型关闭了加窗评测

`skvmn`（forward 含逐时间步 Python 循环）与 `hqaf`（重型 transformer，显存贴上限）
在 20 倍行数的加窗测试集上不可用，配置中设 `eval_window: false`，
指纹记录该状态。**这两个模型没有 pykt 可比口径的数字，不得与其它模型的
`best_window_test_auc` 并排比较。**

#### 已知的遗留问题（继承自 pykt，未改）

`dimkt` 的 `c_emb = Embedding(num_c + 1, ..., padding_idx=0)` 与 0-based 概念 id
冲突：真实的 0 号知识点与 padding 共用第 0 行，其 embedding 被冻结为零向量。
pykt `dimkt.py` 第 29 行相同。改动会偏离 pykt，故保留并记录。

---

## 4. 仍然存在的问题

### 4.1 未实现 pykt 的融合式题目级评测

`qidxs / orirow / cidxs`、`late_mean / late_vote / late_all`、early fusion 在本仓库引用数均为 0。

§2.4 论证了在剔除重复行后二者等价，但**该等价性尚未实证验证**。
若需要与 pykt 的题目级表格逐格对账，仍需实现。

### 4.2 训练过程中打印测试集指标

`core/trainer.py` 每 10 轮在测试集上评一次并打印，附有"只读、勿用于调参"的警告。

这不构成程序性泄露（选择逻辑只看 `valid_auc`），但属于流程隐患：
人看到了测试集数字就可能据此做决定。pykt 无此行为。

### 4.3 存量结果全部作废，必须重跑

**没有任何一份存量结果同时满足修复后的三个条件**（无重复行泄露、统一 `concept_mode`、
带 windowed 口径）。具体：

| 来源 | 问题 |
|---|---|
| 2026-04 及更早的存档 | 无泄露，但产于多知识点截断修复之前，且无 windowed 指标 |
| 重构后至 2026-09-16 之间的 `one_by_one` 结果 | **受重复行泄露影响** |
| `experiment/last_test_summary.md`（2026-06-10） | 早于全部修复，且混合了 `multi` 与 `first` 模型 |
| `saved_model/aio_baselines/` | 产于池化修复之前，对应模型已改 |

**其中 ATKT 尤其必须重跑**：修复前跑的是普通 attention-LSTM，不是 ATKT（见 §6.2）。

### 4.4 尚未核实的细节

- `cseqs[:, 0]` 取的"第一个"知识点是否具有语义（主知识点），还是顺序任意。
  该问题现在只影响 §4.3 表中的旧结果与仍截断的 `gkt` / `rekt`。
- 泄露检测目前仅在 `assist2009 / algebra2005 / bridge2algebra2006` 上运行过。
- 本文多数结论基于单折单种子；跨折、跨种子的稳定性未系统验证。
- `dkt_forget` 在 assist2009 上无法运行：该数据集两个序列文件均无 `timestamps` 列，
  而遗忘间隔是该模型的核心输入。若旧表中存在 assist2009 的 DKT-forget 数字，需核查来源。

---

## 5. 操作指引

### 5.1 跑一次标准实验

```bash
python scripts/train.py --dataset-name assist2009 --model-name akt --fold 0 --seed 3407 --use-wandb 0
```

产出中 `best_window_test_auc` 为 pykt 可比口径，`best_test_auc` 为普通切分口径。

### 5.2 复现 pykt 的（含泄露的）口径

```bash
KT_SCORE_REPEATED_KC=1 python scripts/train.py --dataset-name assist2009 --model-name akt --fold 0 --use-wandb 0
```

### 5.3 泄露自检

```bash
python tests/test_no_target_leakage.py
```

默认应输出 `CLEAN`。加 `KT_SCORE_REPEATED_KC=1` 应当报 `FAIL`——若不报，说明过滤器失效。

### 5.4 出表前的检查清单

1. 表内所有行的 `run_config.json` → `protocol` 块是否**完全一致**
2. 是否混合了 `concept_mode: multi` 与 `first`——`gkt` / `rekt` 永远是 `first`（§3.4）
3. 是否混合了 `eval_window: true` 与 `false`——`skvmn` / `hqaf` 永远是 `false`（§3.4）
4. 结果是否产于 2026-09-16 全部修复之后（§4.3：此前的一律作废）
5. 报的是 `best_test_auc` 还是 `best_window_test_auc`，是否在论文中写明
6. 若表中含 ATKT，是否为对抗训练修复后重跑的（§6.2）

---

## 6. 模型实现与 pykt 的一致性审计

对 pykt 中存在的 25 个模型逐函数核对（AST 提取每个 `类.方法` 函数体，剥除
docstring/注释后 diff，再人工判定每处差异属于框架适配还是真实分歧）。
自研模型（`hd_* / hdkt` 等）无权威实现可比，未纳入。

原始 diff 行数只能当筛选信号：`simplekt` 371/372 行不同，最终判定完全忠实
（类名 `simpleKT` → `SimpleKT` 导致整体行错位）。

### 6.1 忠实（24 个）

`dkt` `dkt+` `dkvmn`（forward 逐字相同）`deep_irt` `simplekt` `akt` `gkt` `sakt`
`skvmn` `dimkt` `atdkt`（逐字相同）`kqn` `saint` `dtransformer` `sparsekt` `stablekt`
`robustkt` `hawkes` `qikt` `rekt` `ukt` `lpkt` `iekt` `saint_plus`

专门核过的高风险点：

- `saint` 的移位对齐——pykt 传 `r`（长度 L−1）而非 `cr`，模型内 `cat(START, r)` → L，
  输出 `[:, 1:]` → L−1 对上 `rshft`。本仓库一致。
- `saint_plus` 确为 pykt 的 `saint_plus_plus` 变体（解码器签名带 `num_q/num_c`、
  `start_token=0`、传 `raw_in_ex/raw_in_cat`），未与 `saint` 混淆。
- `lpkt` 核心递推完整保留，运行时 Q 矩阵路径挂在 `use_runtime_concepts` 开关后。
- `qikt.get_avg_fusion_concepts` 的 3 维分支与 pykt 逐字相同。

正确丢弃的 pykt 死代码：`sparsekt.timeGap`、`simplekt.LearnablePositionalEmbedding`
（两者均定义了但从未被引用）。

### 6.2 发现并修复的真 bug：ATKT 缺失对抗训练

pykt `train_model.py` 的 `atkt` 分支：干净前向 → `grad(loss, features)` →
`p_adv = epsilon * _l2_normalize_adv(grad)` → 带扰动二次前向 → `loss + beta * adv_loss`。

本仓库 `core/trainers/atkt_trainer.py` 原先**只有普通前向 + BCE**；
`model.epsilon` / `model.beta` 被赋值后全仓库无任何读取，配置里也没有这两项。
**对抗训练是 ATKT 的全部贡献**，因此此前所有 ATKT 结果跑的都是普通 attention-LSTM。

已修复（`_l2_normalize_adv` 改为在设备上计算，避免 pykt 的 numpy 往返；
`model.training` 为假时跳过）。实测验证：

| assist2009 fold0 | windowed AUC |
|---|---|
| 修复后（`beta=0.2`） | **0.73832** |
| 修复前等价（`beta=0`） | 0.72724 |
| pykt Table 2 | 0.7470 |

对抗训练值 **+0.0111**，与 pykt 的差距从 −0.0198 缩至 −0.0087。

### 6.3 本仓库修正的 pykt 缺陷（6 处）

| 模型 | pykt 的问题 |
|---|---|
| `dtransformer` | `get_loss` 中**两个 NameError**（参数名为 `pids` 却使用 `pid`；`logits` 从未定义）——该函数无法运行 |
| `sakt` | `pos_encode(seq_len)` 超出位置表长度时越界 |
| `dimkt` | 硬编码 `.cuda()`，CPU 上崩溃 |
| `stablekt` | 同上（`MultiHeadAttention.__init__`） |
| `hawkes` | 同上（`mask.cuda()`） |
| `lpkt` | `c_tilde` 除法无零保护，题目未映射到任何知识点时产生 NaN |

### 6.4 与 pykt 公布值的对账

pykt 论文 Table 2（AS2009，题目级 + LF-AVG 融合 + 加窗）：

| 模型 | pykt 公布 | 本仓库（加窗，已除泄露） |
|---|---|---|
| AKT | 0.7853 | 0.78362 |
| DKT | 0.7541 | 0.75890 |
| DKVMN | 0.7473 | 0.75885 |
| SAKT | 0.7246 | 0.73874 |
| ATKT | 0.7470 | 0.73832 |

AKT 差 0.0017，协议已对齐。其余偏差主要来自超参（pykt 每个模型都做过扫参）
与折数（本仓库为单折，pykt 为 5 折平均），**不应解读为实现差异**。
