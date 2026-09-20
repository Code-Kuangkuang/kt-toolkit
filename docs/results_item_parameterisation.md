# 题目参数化与 null 基线，assist2009 / algebra2005 / assist2017

日期：2026-09-19。记录于此的原因与 `docs/results_hdkt_ablation.md` 相同：
`saved_model/` 与 `experiment/` 都在 gitignore 里，产物过不了一次 clean，而这批数
是 38 个 run 的 GPU 时间。

---

## 0. 一句话结论

**在 assist2009 上，一个 7 个可学参数、且在数学上不可能使用做题顺序的计数模型，
达到 0.76370 窗口 AUC，与 DKT（0.76069）打平，并显著优于去掉题目参数的
SimpleKT（0.75377）。从随机（0.5）到本仓库最好的深度模型（0.78584）的全部增益中，
计数占 92%。**

这不是"深度模型无用"。这是**基线表一直缺一个分母**：`experiment/baseline_table.md`
以 `dkt` 为地板，而 DKT 是模型不是 null，所以此前每一个 Δ 都没有参照系。

---

## 1. 实验设计

两组消融，共用一套协议。

### 1.1 题目参数宽度（`simplekt`，四档）

`models/simplekt.py` 的嵌入式是

```
x_t = c_{c_t} + mu_{q_t} (*) d_{c_t}
```

`c`（`q_embed`）与 `d`（`q_embed_diff`）按**知识点**索引，各 256 维；
`mu`（`difficult_param`）按**题目**索引。四档只改 `mu`：

| `emb_type` | mu 每题 | 可学 | 出处 |
|---|---|---|---|
| `qid_norasch` | 整项跳过 | — | 上游已有 |
| `qid_scalar` | 1 个数 | 是 | 上游已有，等价于 AKT 的 Rasch 标量 |
| `qid_frozen` | 1 个数 | **否** | **本仓库新增**，直接填训练折数出的 log-odds |
| `qid`（默认） | 256 个数 | 是 | 上游默认 |

`qid_frozen` 见 `models/simplekt.py` 的 `Inputs.prepare` 与
`datasets/feature_utils.py::compute_item_difficulty_logodds`。

### 1.2 null 阶梯（`nullkt`，三档）

`models/nullkt.py`。**全部特征都是前缀计数或前缀均值**，因此打乱学生历史顺序不改变
任何预测——这一条由 `tests/test_nullkt_is_order_invariant.py`（12 个用例）钉死，
而不是靠文档声称。可学部分只有一个 `Linear(F, 1)`。

| `emb_type` | 特征 | 可学参数 |
|---|---|---|
| `item` | 题目难度、知识点难度（均为训练折统计，冻结） | 3 |
| `student` | + 学生总体正确率、已答题数 | 5 |
| `count` | + 该题所属知识点上的历史正确率与次数 | **7** |

`core/trainers/nullkt_trainer.py` 与 `SimpleKTTrainer` 同形：同样的 batch 键、
同样的 `preds[:, 1:]` 对齐、同样的 float64 BCE、同样的 `smasks` 评分位。
唯一的差异必须是模型本身——float32/float64 曾是 HD 系列的真实混淆项，见
`core/backbone.py`。

---

## 2. 主结果：assist2009，五折，seed 3407

窗口测试 AUC（`best_window_test_auc`）。

| 行 | 序列模型 | 有效可学参数 | AUC | sd |
|---|---|---|---|---|
| 随机 | — | — | 0.50000 | — |
| `nullkt` `item` | **无** | **3** | 0.68687 | 0.00193 |
| `nullkt` `student` | **无** | **5** | 0.75330 | 0.00230 |
| **`nullkt` `count`** | **无** | **7** | **0.76370** | 0.00201 |
| `simplekt` `qid_norasch` | transformer | 1,086,209 | 0.75377 | 0.00096 |
| `dkt` | LSTM | 395,523 | 0.76069 | 0.00141 |
| `simplekt` `qid_frozen` | transformer | 1,117,953（题目参数 **0**） | 0.78003 | 0.00204 |
| `simplekt` `qid_scalar` | transformer | 1,135,691 | 0.78499 | 0.00282 |
| `akt` | transformer | 1,528,931 | 0.78100 | 0.00264 |
| `simplekt` `qid`（默认） | transformer | 5,658,881 | 0.78584 | 0.00319 |

"有效"= 收到梯度的参数，即已扣除该档下分配但不参与的表
（`qid_norasch` 下的 `difficult_param` / `q_embed_diff` / `qa_embed_diff` 共 4,635,904；
其余各档下的 `qa_embed_diff` 共 63,232，默认分支从不引用它）。
不可训练的余弦位置编码（51,200）在所有 SimpleKT 档中均已排除。
`dkt` / `akt` 取自 `core/model_info.py`，与 `docs/results_hdkt_ablation.md` 同源。

### 2.1 配对检验（df = 4，|t| 需 ≥ 2.78 才到 p < 0.05）

| 对比 | Δ | t | 赢的折 |
|---|---|---|---|
| null `item` → null `student` | +0.06643 | +228.69 | 5/5 |
| null `student` → null `count` | +0.01041 | +27.09 | 5/5 |
| `simplekt` k=0 → null `count` | **+0.00993** | **+12.24** | **5/5** |
| null `count` → `dkt` | **−0.00301** | **−2.03** | **2/5** |
| null `count` → `akt` | +0.01730 | +15.76 | 5/5 |
| null `count` → `simplekt` k=1 | +0.02129 | +23.42 | 5/5 |
| k=0 → k=1（学） | **+0.03122** | **+24.46** | **5/5** |
| k=0 → k=1（数） | +0.02626 | +27.44 | 5/5 |
| k=1 数 → k=1 学 | +0.00496 | +7.91 | 5/5 |
| **k=1 → k=256** | **+0.00084** | **+0.69** | **3/5** |

### 2.2 三点读法

1. **计数与 DKT 打平。** −0.00301 的点估计为负，|t| = 2.03 未到显著线，DKT 只赢 2/5 折。
   准确的说法是"打平"，不是"计数更好"。
2. **去掉题目参数的 transformer 显著不如计数**（−0.00993，5/5 折）。
3. **一个标量顶一个十年。** k=0 → k=1 是 +0.0312；`dkt` → `akt` 的十年跨度是 +0.0203。

### 2.3 增益构成

从 0.5 到 0.78584 共 0.28584。

| 来源 | Δ | 性质 |
|---|---|---|
| 题目难度 | +0.187 | 与学生无关 |
| 学生总体能力 | +0.066 | 与题目无关 |
| 知识点级对错计数 | +0.010 | **唯一沾得上"知识追踪"的一项** |
| 以上为计数小计 | **+0.264（92.3%）** | 零时序建模 |
| 深度序列模型的边际贡献 | +0.022（7.7%） | 十年 |

### 2.4 逐折

```
                fold0     fold1     fold2     fold3     fold4
nullkt item     0.68886   0.68753   0.68523   0.68444   0.68829
nullkt student  0.75495   0.75398   0.75194   0.75002   0.75559
nullkt count    0.76589   0.76462   0.76172   0.76140   0.76488
qid_norasch     0.75507   0.75383   0.75427   0.75273   0.75297
qid_frozen      0.78111   0.78015   0.77850   0.77764   0.78275
qid_scalar      0.78727   0.78281   0.78344   0.78269   0.78876
qid             0.78781   0.78803   0.78463   0.78074   0.78798
dkt             0.75890   0.76135   0.76197   0.76178   0.75946
akt             0.78292   0.77825   0.78174   0.77821   0.78388
```

`nullkt count` 与 `dkt` 的逐折比较值得单看：计数在 fold 0 / 1 / 4 上更高，
DKT 在 fold 2 / 3 上更高。配对差的标准差（0.00331）**大于**差的均值（0.00301），
这正是 §2.2 说"打平"而非"计数更好"的理由。

---

## 3. 学到的标量到底是什么

fold 0，单知识点题目，取样本量足够的 50 个知识点（5,315 道题，每题 ≥10 次作答）。
与**训练折实际正确率**的 Spearman 相关。

**必须取绝对值**：那一项是 `mu_q * d_c`，方向由 `d_c` 承担，同一个正号在 A 知识点上
表示"更难"、在 B 上表示"更易"，跨知识点平均会互相抵消（未取绝对值时均值仅 −0.044，
而中位数是 −0.785）。

| | mean\|r\| | 加权 | 中位数 |
|---|---|---|---|
| **k=1 学到的标量** | **0.911** | 0.919 | 0.918 |
| k=256 的等效标量 | 0.681 | 0.697 | 0.750 |
| 随机标量（对照） | 0.099 | — | 0.061 |

**梯度学出来的那个数，就是这道题的经验正确率。** 而 256 维反而把这个信号稀释了。

## 4. 256 维不是"悄悄被正则化了"

一个自然的猜测是：多出来的维度稀疏、没被训练到，所以不伤害。**实测否定了它。**

fold 0，`difficult_param` 17,738 × 256：

- 训练折中出现过的题 17,113（96.5%），**其中每一行都非零**
- 未出现的 625 行**全部恰好为 0**（`SimpleKT.reset()` 零初始化 + 无梯度）
- top-1 奇异方向仅占 7.7% 能量，top-25 占 57.7%
- **有效秩（参与率）= 198.11 / 256**

即：这 454 万个参数是实打实拟合出来的、接近满秩的东西，**而它们的预测价值为零**
（+0.00084，不显著）。

复现：`python research/item_param_capacity.py --ckpt-root saved_model/itemparam_full`

---

## 5. 跨数据集：最优每题自由度随每题观测数上升

**这是本文唯一一条事先声明、可被证伪、而未被证伪的预测。**

fold 0（未扩五折，故仅为方向）。

| 数据集 | 每题观测 | k=0 | k=1 数 | k=1 学 | k=256 | **k=256 − k=1** |
|---|---|---|---|---|---|---|
| algebra2005 | 3.3 | 0.81589 | 0.80803 | 0.83671 | 0.83171 | **−0.00500** |
| assist2009 | 10.8 | 0.75507 | 0.78111 | 0.78727 | 0.78781 | +0.00054 |
| assist2017 | 196.3 | 0.70007 | 0.73757 | 0.74808 | 0.76267 | **+0.01458** |

最后一列单调。跨 60 倍观测数，方向一次没错。

**计数在哪里失效**：algebra2005 上 `qid_frozen` 比 `qid_norasch` **低** 0.00787——
数出来的难度比不给题目参数还糟。见 §7.2，这一条有已知缺陷未排除。

### 5.1 难度本身是否可估计（折半信度）

把 fold 0 的训练折按学生随机劈两半，各自统计题目正确率，再求 Spearman。
与模型无关，训练前即可算。

| 数据集 | 每题观测 | 折半 r | n≥10 的题的 r | n≥10 的题占比 |
|---|---|---|---|---|
| algebra2005 | 3.3 | 0.518 | 0.832 | **4.1%** |
| bridge2algebra2006 | 10.1 | 0.353 | 0.677 | 20.5% |
| assist2009 | 10.8 | 0.417 | 0.630 | 37.3% |
| assist2012 | 33.4 | 0.496 | 0.757 | 60.6% |
| ednet | 32.7 | 0.482 | 0.711 | 78.9% |
| assist2017 | 196.3 | 0.770 | 0.900 | 75.6% |
| nips_task34 | 936.4 | 0.862 | 0.938 | 95.7% |

难度是真实存在的（n 足够时 r 达 0.83–0.94），但 algebra2005 上 96% 的题目
根本数不出难度。

---

## 6. 协议

八字段戳在所有行上完全一致，**除 `feature_fit_scope` 外**：

```json
{"dataset_mode": "all_in_one", "concept_mode": "multi", "max_concepts": 4,
 "concepts_visible": "all", "score_repeated_kc": false, "eval_window": true,
 "graph_scope": "none"}
```

| 行 | `feature_fit_scope` |
|---|---|
| `dkt` / `akt` / `qid_norasch` / `qid_scalar` / `qid` | `none` |
| **`qid_frozen` / `nullkt` 三档** | **`train_folds`** |

**差异的性质**：前者不派生任何特征，后者只从**当前折的训练折**派生。
**两者都没有读过 valid/test，不是泄露差异。** `nullkt` 的
`run_config.json` 另记 `difficulty_folds`，fold 0 为 `[1,2,3,4]`。

但 `scripts/run_baseline_table.py::protocol_key` 按全部八个字段分组，
**因此它会拒绝把这些行合并成一张表**。本文是它们唯一能并排的地方，前提是读者
接受上面这段关于差异性质的说明。

---

## 7. 已知缺陷与未做的事

### 7.1 `qid_frozen` 的 shrinkage 被归一化抵消（**影响 §5 的一个结论**）

`compute_item_difficulty_logodds` 先按 `alpha=10` 向全局正确率收缩，
**随后又除以自身标准差归一化到单位方差**——后一步把前一步压下去的方差放大了回来。

在 assist2009 上无碍（计数本身有信号）。在 algebra2005 上，计数几乎全是噪声，
这等于向模型注入单位方差的噪声。**因此"计数有害"（−0.00787）这一条里有多少来自
这个缺陷、多少来自计数本身，目前分不开。**

干净的对照：给冻结表一个**可学的全局缩放**（1 个参数），让模型能把噪声压到 0。
约 5 行，未做。

### 7.2 其余

- **单种子**（3407），单 backbone（SimpleKT）。§5 的两个新数据集**只跑了一折，无误差棒**。
- `nullkt` 的超参一个都没调（`difficulty_alpha=10`、`prior_strength=5`、`lr=0.05`）。
  调参只会抬高 null，**故当前对比对 null 是保守的**。
- **只测了 k ∈ {0, 1, 256}**。中间点（4/16/64）未测，因此"196 次观测时最优是 256"
  这句话说不出来，只能说"256 优于 1"。
- **未做跨 backbone**：给 DKT / SAKT 加题目参数需要各改一次模型。
  `core/backbone.py` 的接缝开在 `embed` 与 `encode` **之间**（为 HD-KT 定制），
  而题目参数化需要的接缝在 `embed` **内部**。
- **AUC 本身是混淆项**：它把所有 (学生, 题目) 对混在一起排序，因此"区分难易题"与
  "区分强弱生"直接计入分数。**改为在单个学生序列内部计算 AUC，题目难度那 +0.187
  应大部分消失**——那才是教学系统实际面对的问题。未做，是最该做的下一个诊断。

---

## 8. 复现

```bash
python scripts/run_baseline_table.py --datasets assist2009 --models simplekt \
  --folds 0 1 2 3 4 --seed 3407 --save-root saved_model/itemparam_scalar \
  --out experiment/itemparam_scalar.md --train-args "--emb_type qid_scalar"
```

```bash
python scripts/run_baseline_table.py --datasets assist2009 --models nullkt \
  --folds 0 1 2 3 4 --seed 3407 --save-root saved_model/null_count \
  --out experiment/null_count.md --train-args "--emb_type count"
```

`emb_type` 换成 `qid_norasch` / `qid_frozen` / `qid` 与 `item` / `student` / `count`
即得其余各档。**每档必须用独立的 `--save-root`**：`run_dir_name` 不含 `emb_type`，
同一根目录下第二档会被当作已完成而跳过——与 `results_hdkt_ablation.md` 记的
`train_label_flip_ratio` 是同一个坑。

分析脚本：

```bash
python research/itemparam_table.py                                    # 四档配对表
python research/item_param_capacity.py --ckpt-root saved_model/itemparam_full   # 谱与有效秩
```

## 9. 代价

| 组 | run 数 | 合计 |
|---|---|---|
| `simplekt` 四档 × 五折（assist2009） | 20 | 约 20 分钟 |
| `nullkt` 三档 × 五折（assist2009） | 15 | 约 5 分钟 |
| 四档 × 一折（algebra2005 + assist2017） | 8 | 约 20 分钟 |

`nullkt` 每折 0.2–0.3 分钟。**整条 null 阶梯比一次 AKT 单折（11.6 分钟）还便宜。**
这是它此前不存在的唯一理由不成立的证据——它不贵，只是没人跑。
