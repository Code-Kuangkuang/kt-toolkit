# 2026-09-17 / 09-18 工作汇总

37 个提交（09-17 十一个、09-18 二十六个），外加一批尚未提交的模型移植。
主题只有一个：**让"两次运行的数字可比"这件事从约定变成机制**，然后在这个地基上扩模型。

统计口径：`git log --since=2026-09-13`；测试数与注册表数取自当前工作树
（`pytest --collect-only` 190 个用例 / 22 个测试文件，`MODEL_REGISTRY` 46 个模型）。

---

## 1. 实验协议护栏

这是这两天的主线，也是后面一切结论能成立的前提。详见 `docs/architecture.md` 的
"Model Input Specs" 与 "Model Contract Tests" 两节。

| 做了什么 | 结果 |
|---|---|
| **InputSpec 迁移**：每个模型用嵌套 `Inputs(InputSpec)` 声明自己要什么 | `core/train_runner.py` 的 `model_name` 分发 **20 → 0**；三份手工名单全删 |
| **契约测试** `tests/test_model_contracts.py` 遍历注册表 | 每个模型查 9 项：注册配对、config 块、构造、前后向、pred/target/smasks 对齐、梯度有限、eval 确定性、未来 response 不影响过去预测、CPU 构造不跑到 GPU |
| **协议戳** `protocol_stamp` 扩到 8 个字段 | `run_baseline_table.py::protocol_key` 改为按全部 8 个字段分组（之前只读 5 个，等于记了不用） |
| **fit scope**：新增 `feature_fit_scope` / `graph_scope` | 派生特征默认只用当前 fold 的训练折 |
| **fail-fast** | loader 建不起来、test 文件缺失、监督 mask 错位一律停；AUC 算不出返回 `None` 而非 `-1`（`-1` 会被 CV 均值当有效值平均） |
| **CV resume** | 不再可能把不同协议的折平均到一起 |

**一处有代价的行为变更**（`66e3cd7`）：gkt / dkt_forget / dgekt 默认从
transductive 改为 `train_folds`。实测 GKT assist2009 fold0 test AUC
**0.7278 → 0.7248**。原因量化得出来：assist2009 的 transductive 图有 3953 条边，
只用训练折是 3511 条——**11% 的边只因为看过 valid/test 才存在**。
`pykt_transductive: true` 可精确复现旧行为。

---

## 2. 因为上面这些护栏而暴露的真 bug

这一节是护栏的收益证明。

**IEKT 三个独立缺陷**（`6a4aea7`）
- 记分位置用 `(qseqs!=0).sum()+1` 推导，实测 4556 vs `smasks` 3886，**多记 17%**
- 预测头是裸 `nn.Linear` 没有 sigmoid，而评分用 `p>=0.5` 算 ACC
- `Categorical(...).sample()` 没有 `self.training` 门控，**eval 同 batch 两次差 0.287**
- 外加：`device` 参数被构造函数内部覆盖，`--gpu_id` 对它一直无效

**LPKT**：`models/lpkt.py:252` 在 forward 里调 `nn.init.xavier_uniform_`，
`one_by_one` 模式下每次前向重抽初始知识状态。

**测试基建本身是坏的**：5 个 pytest 风格文件、16 个用例**从来没跑过**
（unittest 收集 0 个却报成功）；`requirements.txt` 是 UTF-16，pip 读不了。

**更早但同属这条线**（09-16）：`is_repeat` 记分泄漏
（AKT assist2009 读数 0.837 → 修复后 0.777）；baseline 与 HD 变体跑在不同
dataset_mode，差约 8 个 AUC 点；**ATKT 根本没有实现对抗训练**，所有存量 ATKT 数字
都是普通注意力 LSTM。

---

## 3. 架构重构

- **HD-KT 从"三个 backbone 的分叉"改成插件**（`7d77c2a`）。原来 `hd_dkt` / `hd_akt` /
  `hd_simplekt` 是三份复制的模型文件，改后是同一套插件机制挂三个 backbone。
- **插件机制移出 `models/`**（`a812923`），进 `core/` 和 `plugins/`，`models/` 只放模型。
- **HD-KT 对齐 LPKT 的概念加权**（`d2dc92a`），这样这一对才真的是消融关系而不是两个不同模型。
- **统一 config-key strip 列表**（`2647691`），外加一个"打错 key 藏不住"的测试。

---

## 4. 数据集治理

- `statics2011` 从 OLI Fall 2011 导出重建（`c52c340`），并加了下载文件的预检（`8926461`）
- `ednet` 切片改为从 KT1 可复现地生成（`d915e13`、`2a3c554`），并记录 KT1 的来源（`09e7147`）
- 数据集清单审计，**退役两份已经烂掉的文档**（`7b15a38`）
- 明确标注哪些数据集是样本、样本自什么（`dd6daf8`）
- WebUI 提供的目录被限制在项目内（`2b69220`）

---

## 5. 模型扩充（09-18，尚未提交）

registry **38 → 46**。八个模型全部来自 pykt：

| 模型 | 论文 | 备注 |
|---|---|---|
| `mockt` | MoC-KT, TOIS 2026 | 按序列长度分三桶的混合卷积 |
| `fluckt` | FlucKT, AAAI 2025 | 单核版本的同一滤波器 |
| `denoisekt` | DenoiseKT | 邻接矩阵从 qmatrix 重建 |
| `hcgkt` | HCGKT | 需 FLAG 对抗训练循环 + 下载的 KC 文本向量 |
| `extrakt` | 长度外推 | 复用 RobustKT trainer |
| `folibikt` | FoLiBiKT, CIKM 2023 | 同上 |
| `cskt` | csKT, ESWA 2025 | 复用 StableKT trainer |
| `mtkt` | MTKT, Neurocomputing 2025 | 需时间戳；只能在 assist2017 跑 |

**移植模式**：pykt 类原样保留，外面套一个 adapter 子类承载注册、`Inputs` 规格和参数改名
（沿用 `models/robustkt.py` 已有的写法），这样以后 diff 上游仍然可读。

**产物重建** `models/kc_graph_utils.py`：上游只通过 Google Drive 分发的
题目–题目邻接与题目–概念映射，改为从 `qmatrix.npz` 重建。
⚠️ **对称归一化 `D^-1/2(A+I)D^-1/2` 是推断，上游没有明说**，对齐发表数字时不能假设逐边相同。
HCGKT 另需下载的 KC 文本 BGE 向量（`utils/kc_embedding/`），其索引对齐已核验
（assist2009 上 113/123 按 index 精确同名，其余 10 个是源数据里本就无名的技能），
并加了行数守卫防止在未核验的数据集上误用。

**发现的一族关系**：MoC-KT / FlucKT / RobustKT 是同一机制
（低通因果卷积 + 高通残差，`low + sqrt_beta²·high`），只是核的数量不同。
这正是 2026-09-17 被删掉的 frequency 线。三个现在同协议在册，可以一次比清楚。

---

## 6. 移植后的保真度审计

用的是当初审计 25 个原生 pykt 模型的同一套方法：`ast` 抽取每个 `Class.method`，
`ast.unparse` 归一化（注释与格式消失），剥 docstring，逐方法比对。

- **模型本体**：158 个方法相同，58 个不同，**58 个全部能归因到有意改动**
  （设备修复、多知识点池化、buffer 注册、adapter、产物参数化），反向过滤后
  **零条未解释的语义差异**
- **trainer**：八个全部对上 pykt `train_model.py` 里各自的分支

**抓到一个真 bug——MoC-KT 的分桶键。** pykt 的
`mockt_data_loader.py` 用 `(rseqs != -1).sum(dim=1, keepdim=True).float()`，
但**本仓库的 `rseqs` padding 是 0 不是 -1**，照搬会让每一行都返回 200，
三个长度桶塌成一个，**"mixture of convolutions" 退化成单核，论文贡献静默失效**。
正确写法是 `masks.sum(1) + 1`，且必须保持 `[B,1]` float 形状——
`FrequencyLayer` 在 `s.shape[0]==1` 时调 `.squeeze(dim=0)`，`[B]` 会变成 0 维掩码，
batch size 为 1 时崩。已修并在真实数据上验证（fold0 一个 batch 分桶 5/3/0）。

**外加四处超参错误。** 凭经验配的超参没有一个可信，对着
`examples/seedwandb/<model>.yaml` 核完才改对：mockt 的 `emb_type` 应为
`qid_conv_ker_noexp`（走的是不同的注意力分支）、`kernel_size` 应在 [4,8,16,32]；
denoisekt 的 `bf` 应在 0.01–0.99（大于 1 会让 `bf**distance` 从衰减变增长）；
hcgkt 的 `step_size` 应为 1e-2~1e-1、`grad_clip` 应为 5~20。

**上游缺陷（四族都有）**：模块级 `device = cuda if available`；
extrakt/folibikt/mtkt 里硬编码 `.cuda()` 导致 CPU 完全跑不了；
ALiBi 与邻接表是普通属性，`model.to()` 搬不走；
`mtkt` 的 `timeGap` 里 `num_pcount=15` 被硬编码（带 `#test` 注释）直接丢弃调用方的值。
另注：pykt 里有两份 `CausalConv1d`，mtkt 那份有 `padding != 0` 守卫，mockt 那份没有。

---

## 7. 研究方向

**HD-KT 干净标签基线**（`002e237`，详见 `docs/results_hdkt_ablation.md`）：
assist2009 上三对全无效果（`hd_dkt` −0.00006、`hd_akt` −0.00004、
`hd_simplekt` +0.00120，|t| ≤ 0.85 at df=4）。
**这是 ratio=0 的锚点，不是否定结论**——HD-KT 是鲁棒性方法，它的主张是噪声上升时的
*斜率*，不是干净数据上的截距。噪声斜率实验已设计未跑。

**两个候选方向被否**：

- **诊断—学习解离（DLD）**：撞 CE-KT（arXiv 2608.22267）的 RQ1，它已经做完同一个实验；
  更早还有 Baker 2011 的 moment-by-moment learning
- **无关历史污染**：撞 Yeung & Yeung 2018 的 wavy transition problem，
  而那篇的实现就是仓库里的 `models/dkt_plus.py`

两次都**不是败在新颖性上，而是没有 well-posed 的 null**。由此给选题补了第五道门，
排在原来四道查新颖性之前：

> 这个现象有没有一个 well-posed 的 null？"正常"该长什么样，能不能独立于我的模型算出来？

**文献与代码审计**：
OpenAlex 口径坑（会议计数不可用、2026 年三成是仓储灌水）；三年主题形态
（注意力退潮、图/超图稳居第一、不确定性是唯一从 0 长出来的主题）；
**顶会顶刊 65 篇里只有 22% 有公开代码**（AAAI 56% vs 期刊 11–13%）；
**pykt-toolkit 是近年 KT 论文事实上的代码仓库**——按论文名或缩写搜 GitHub 一个都搜不到。

---

## 8. 当前缺口（都是已知的，不是遗漏）

1. **八个新模型一次都没训练过**，没有任何 AUC。审计能证明"实现了论文说的东西"，
   证明不了"数字复现得出来"。
2. **HD-KT 噪声斜率没跑**——这是唯一能决定这条线死活的实验，基建已全在
   （`apply_train_label_flip` + `train_label_flip_ratio` 贯穿 runner 和 `run_config.json`）。
3. **所有存量基线因协议变更失效**，任何表都要重生。
4. **`mtkt` 与 `dkt_forget` 需要带 `timestamps` 列的数据集**。判据不是 `input_type`
   （pykt 自己的 config 也没有任何数据集列它），而是生成出来的序列 CSV。
   assist2009 没有（预处理写死 `seq_start_time = ['NA']`，这是 pykt 自己的行为），
   **assist2017 有**。
5. **RKT 没有移植**：它的 `phi_array_<folds>.pkl` 关系矩阵在 pykt 和原始仓库
   `shalini1194/RKT` 里都没有发布生成代码，且定义无法从消费方式反推。
6. **一个未解疑点**：pykt 自己的 fluckt sweep 用 `emb_type: ["qid"]`，
   按代码里的门控这会把"认知波动"滤波器整个关掉。
