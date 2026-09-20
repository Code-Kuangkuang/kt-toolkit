# KT-Toolkit Agent Guide

本文件定义 AI 编码代理在本仓库中的默认工作方式。目标不是单纯“让模型跑起来”，而是产出可复现、无数据泄漏、可公平比较、可解释的知识追踪（Knowledge Tracing, KT）研究结果。

## 1. 项目目标

KT-Toolkit 是一个基于 PyTorch 的知识追踪研究工具箱，覆盖数据清洗、序列预处理、模型训练、交叉验证、实验记录和 WebUI 调度。

代理处理任务时应优先保证：

1. **研究正确性**：时序因果关系、标签对齐、数据划分和评价流程正确。
2. **公平可比性**：新模型与基线使用相同数据、fold、随机种子和指标口径。
3. **最小改动**：复用现有注册表、Trainer、数据集和 artifact 机制，不复制训练框架。
4. **可复现性**：记录配置、随机种子、fold、模型选择依据和实验产物路径。
5. **可维护性**：模型结构、训练逻辑、数据特征和研究脚本边界清晰。

## 2. 开始任务前

先阅读与任务直接相关的文件，不要只凭文件名推断行为。推荐顺序：

1. `README.md` 和 `docs/architecture.md`（尤其 "Model Input Specs" 与 "Model Contract Tests" 两节）
2. `scripts/train.py`
3. `core/train_runner.py` 和 `core/trainer.py`
4. `core/model_inputs.py`：`InputSpec` / `ModelInputs` / `RunContext` 声明协议
5. `datasets/init_dataset.py` 和 `datasets/kt_dataset.py`
6. 目标模型及其 Trainer
7. `configs/kt_config.json` 和 `configs/data_config.json`
8. `tests/test_model_contracts.py`：它遍历整个注册表，新增模型会自动被它检查

开始修改前执行：

```bash
git status --short
```

仓库可能包含用户正在进行的实验改动。不得重置、覆盖或清理与当前任务无关的修改。不要使用 `git reset --hard`、`git checkout -- <file>` 或批量删除命令。

## 3. 核心架构

标准训练链路：

```text
scripts/train.py
  -> core/train_runner.py
  -> core/factory.py + core/registry.py
  -> datasets/ + models/ + core/trainers/
  -> core/hooks.py
  -> saved_model/<run_name>/
```

主要目录职责：

- `models/`：模型网络和可复用神经模块。
- `core/trainers/`：batch 适配、目标对齐、loss 和训练过程。
- `datasets/`：运行时 Dataset、DataLoader 和特征构造。
- `cleaning/`：原始数据清洗适配器。
- `preprocess/`：序列生成、划分等预处理工具。
- `configs/`：训练、模型、数据集配置。
- `scripts/`：训练、评估、消融、解释性分析等入口。
- `core/`：注册表、工厂、训练编排、hooks 和 artifact 管理。
- `webui/`：现有训练入口的调度层，不应复制训练逻辑。
- `saved_model/`：运行产物，不提交到 Git。

注册采用 import-time decorator。新增对象后必须同时更新包入口，否则装饰器不会执行：

- 模型：`@MODEL_REGISTRY.register("<model_name>")` + `models/__init__.py`
- Trainer：`@TRAINER_REGISTRY.register("<model_name>")` + `core/trainers/__init__.py`
- Dataset builder：`@DATASET_REGISTRY.register("<dataset_name>")`

**模型自述其输入需求，runner 不按模型名分支。** `core/train_runner.py` 中不存在
`if model_name == ...`；每个模型用嵌套类声明自己要什么：

```python
class Inputs(InputSpec):
    dataset_mode = "all_in_one"        # 或 "one_by_one"
    requires_question_ids = True
    needs_num_pid = True

    @classmethod
    def prepare(cls, ctx):             # 只在需要从数据派生产物时覆盖
        ...                            # 图、难度表、时间桶等
        return inputs
```

新增模型时不要回到 runner 里加分支；缺什么就在 `Inputs` 里声明。`prepare` 派生的
特征必须通过 `feature_fit_scope` / `graph_scope` 声明拟合范围
（`"none"` / `"train_folds"` / `"train_valid_test"`），这三个值会进 `protocol_stamp`，
`scripts/run_baseline_table.py` 按它分组——协议不同的 fold 不会被平均到一起。

## 4. KT 数据契约

常见 batch 字段：

- `qseqs`：题目 ID 序列。
- `cseqs`：知识点 ID 序列。
- `rseqs`：当前交互作答序列。
- `shft_qseqs` / `shft_cseqs` / `shft_rseqs`：下一时刻目标。
- `masks`：有效输入位置。
- `smasks`：参与 loss 和 metric 的 shifted 有效位置。
- `tseqs` / `itseqs`：可选时间特征。
- `rgaps` / `sgaps` / `pcounts` 及其 `shft_` 版本：log2 分桶的时间间隔特征，
  仅当 `Inputs.prepare` 设置 `dataset_kwargs={"include_dkt_forget": True}` 时才存在
  （`dkt_forget`、`mtkt`、`fakt` 用它）。表高 `num_rgap` / `num_sgap` / `num_pcount`
  由 `prepare` 计算后作为构造参数传入。

**有效位置只能从 `masks` / `smasks` 读，不能靠哨兵值反推。** `rseqs` 到达 Trainer 时
是 float 的 0/1，padding 位是 `0.0`——`0` 同时是合法作答，所以任何
`(rseqs != -1)`、`(rseqs >= 0)` 之类的判据都会把整条序列判成有效。上游 pykt 的
CSV 用 `-1` 填充作答，照搬它的长度公式在本仓库会让每行都返回满长度。
实测 assist2009 fold 0：`(rseqs != -1).sum(1)` 每行都是 199，而
`masks.sum(1) + 1` 给出真实长度 58 / 200 / 200 / 8 / 200。
`masks` 覆盖 shifted 序列，所以完整序列长度是 `masks.sum(1) + 1`。
同理，`qseqs` / `cseqs` 的 padding 是 `-1`，`0` 是合法 ID。

数据模式：

- `KTDataset`：单知识点序列，通常为 `[B, T]`。
- `KTQueDataset`：题目级多知识点序列，`cseqs` 原始形状通常为 `[B, T, K]`，填充值为 `-1`。
- 模型拿到 `[B, T]` 还是 `[B, T, K]`，由 `datasets/init_dataset.py` 的
  `MULTI_CONCEPT_MODELS` 决定：在集合内的模型收到 `[B, T, K]`，需要用
  `models/multi_concept.py` 的 `pool_concept_embeddings` /
  `pool_interaction_embeddings` 做掩码均值池化；不在集合内的走 `concept_mode="first"`，
  只读第一个知识点。
- **两者的结果不可直接比较。** 新增模型若不加入该集合，会静默变成"只读首知识点"，
  与同批对比模型跑在不同的概念协议上。`dataset_mode` 同理由
  `ALL_IN_ONE_DATASET_MODELS` / `ONE_BY_ONE_DATASET_MODELS` 决定。

修改模型或 Trainer 前必须写清楚张量流，例如：

```text
qseqs          [B, T]
cseqs          [B, T] 或 [B, T, K]
rseqs          [B, T]
model output   [B, T] 或 [B, T, num_c]
shift target   [B, T-1]
masked pred    [N_valid]
masked target  [N_valid]
```

强制要求：

- 预测第 `t+1` 次作答时，只允许使用不晚于 `t` 的交互信息。
- prediction、`shft_rseqs` 与 `smasks` 必须逐位置对齐。
- padding 不得进入 embedding、loss 或 metric；多知识点的 `-1` 必须先 mask 或安全映射。
- question ID、concept ID 和 response ID 的范围必须与 embedding/output 维度匹配。
- 不得把当前目标答案、未来时间、未来统计量或测试集统计量输入模型。
- 数据派生特征只能用当前 fold 的训练部分拟合，再应用到 valid/test。
- 数据集缺少题目、时间或知识点信息时，应显式报错或说明降级方案，不得静默伪造有效特征。

## 5. 新增或迁移模型

新增模型默认执行以下步骤：

1. 在 `models/<model_name>.py` 实现并注册模型，**包含嵌套的 `class Inputs(InputSpec)`**。
2. 在 `models/__init__.py` 导入模型。
3. 在 `core/trainers/<model_name>_trainer.py` 实现并注册 Trainer。
4. 在 `core/trainers/__init__.py` 导入 Trainer。
5. 在 `configs/kt_config.json` 增加默认超参数。
6. 确认 `configs/data_config.json` 中目标数据集包含所需字段。
7. 在 `datasets/init_dataset.py` 把模型名加入 `ALL_IN_ONE_DATASET_MODELS`
   （或 `ONE_BY_ONE_DATASET_MODELS`），多知识点模型还要加入 `MULTI_CONCEPT_MODELS`。
8. 若新增了需要被扫参的超参数名，加入 `core/run_support.py::apply_overrides` 的白名单；
   不在名单里的 key 会抛 `KeyError`。
9. 若 `Inputs.prepare` 需要读真实数据（图、难度表、时间桶），在
   `tests/test_model_contracts.py` 的 `FITS_FROM_REAL_DATA` 登记，并在
   `COMPUTED_CONSTRUCTOR_ARGS` 给出等价的合成参数。
10. 跑 `python -m pytest`，再做单 epoch 训练检查。

移植上游模型（pykt 等）时的附加约定：

- **上游类原样保留**，注册、`Inputs`、参数改名放在文件末尾的 adapter 子类里
  （见 `models/mtkt.py`、`models/fakt.py`），这样以后 diff 上游仍然可读。
- 文件 docstring 写明来源与抓取日期，并逐条记录你相对上游做了哪些改动、为什么。
- 超参数对着上游自己的 sweep 配置核（`examples/seedwandb/<model>.yaml`），不要凭经验填。
  凭经验配错过四次，每次都要跑完才发现。
- 常见上游缺陷：模块级 `device = cuda if available`、硬编码 `.cuda()`、
  普通属性存张量导致 `model.to()` 搬不走、`emb_type` 门控把论文的核心模块整个关掉。
  改设备相关代码时按作用域核实，不要只做文本替换——函数内的 `device = query.device`
  是正确的遮蔽，不该动。

模型实现规则：

- 模型文件负责网络结构；Trainer 负责 batch 解包、shift、mask、loss 和 metric 输入。
- 优先继承 `torch.nn.Module`，不要引入外部工具箱的重量级基类。
- 保持构造参数与现有工厂兼容，如 `num_c`、`num_q`、`emb_size`、`dropout`、`emb_type`、`device`。
- 创新模块应封装为独立 `nn.Module` 或独立方法，便于消融和复用。
- 不要在 `forward()` 中读取文件、修改全局配置或创建依赖数据规模的大型持久张量。
- 新建 tensor 时继承输入的 device 和 dtype，避免硬编码 `.cuda()`。
- 多任务返回值优先使用语义明确的字典。
- 对 attention、gather、multi-concept fusion 和 shifted prediction 写关键形状注释。

Trainer 实现规则：

- 优先继承 `BaseTrainer`，仅覆盖必要方法。
- `_forward_batch()` 应返回 `(pred, target, loss)`，其中 `pred` 和 `target` 已通过同一 `smasks` 展平。
- 二分类 KT 默认使用 BCE；若模型输出 logits，使用 `binary_cross_entropy_with_logits`，不要重复 sigmoid。
- 空 mask、越界 ID、NaN/Inf 和 shape mismatch 应尽早报出可定位的错误。
- test metric 不得参与 early stopping、超参数选择或模型结构选择。

## 6. 研究实验协议

### 6.1 建立基线

实现创新前，先选择与研究问题匹配的基线并验证其可运行。至少记录：

- dataset、fold、seed
- model 和 `emb_type`
- 数据模式及输入特征
- batch size、learning rate、epoch、patience
- 参数量和运行成本
- best validation epoch
- validation/test AUC、ACC
- checkpoint 与完整配置路径

不要只比较单次训练的最佳数字。

### 6.2 模型选择与测试集纪律

- 默认模型选择指标为 `valid_auc`。
- early stopping 和最佳 checkpoint 只能依据 validation 指标。
- 调参阶段不得依据 test AUC 选择超参数、epoch 或模型版本。
- 仓库可能打印周期性 test 指标；它们只能视为只读诊断，不得用于决策。
- 最终 test 结果使用 best-valid checkpoint。
- `last_epoch` 结果仅用于诊断，不得替代 best-valid 结果。
- AAAI2023（兼容别名 `peiyou`）等隐藏标签测试集只生成预测文件，不计算伪 test metric。

### 6.3 公平比较

比较模型或组件时固定：

- 相同数据预处理、fold 和 seed 列表。
- 相同最大序列长度和评价 mask。
- 相同训练/验证/测试划分。
- 相同调参预算。
- 尽可能相同的训练轮数、early stopping 规则和指标实现。

如果参数量、训练时长或额外特征明显不同，应在报告中披露，不要把资源差异包装成纯结构收益。

### 6.4 交叉验证

正式结果默认使用 5-fold CV，而不是单 fold：

```bash
python scripts/train.py \
  --dataset-name assist2009 \
  --model-name dkt \
  --cv 1 \
  --folds 0-4 \
  --seed 3407 \
  --use-wandb 0
```

报告均值和标准差，并保留每个 fold 的结果。续跑时使用 `--cv-run-dir` 和 `--skip-completed 1`，不要覆盖已完成 fold。

### 6.5 消融和稳健性

一个新方法至少应考虑：

- 完整模型。
- 移除每个核心创新组件。
- 与最接近基线的直接比较。
- 多 seed 或多 fold 方差。
- 参数量、显存或运行时间开销。
- 对序列长度、稀疏知识点、冷启动或多知识点题目的分组表现。

超参数搜索空间应在运行前定义。不要在看到 test 结果后反向修改搜索范围。

## 7. 常用命令

单 fold：

```bash
python scripts/train.py --dataset-name assist2009 --model-name dkt --fold 0 --seed 3407 --use-wandb 0
```

快速 smoke run：

```bash
python scripts/train.py --dataset-name assist2009 --model-name dkt --fold 0 --num-epochs 1 --use-wandb 0 --save-dir saved_model/smoke
```

测试套件（`unittest discover` 会收集到 0 个用例却报成功，必须用 pytest）：

```bash
python -m pytest -q
```

只跑遍历全注册表的契约测试：

```bash
python -m pytest -q tests/test_model_contracts.py
```

注册检查：

```bash
python -c "import models, core.trainers; from core.registry import MODEL_REGISTRY, TRAINER_REGISTRY; print(MODEL_REGISTRY.get_all()); print(TRAINER_REGISTRY.get_all())"
```

语法检查：

```bash
python -m py_compile models/<model_name>.py core/trainers/<model_name>_trainer.py
```

## 8. 验证要求

根据改动范围选择最低充分验证。

### 仅文档或配置

- 检查 JSON/YAML 可解析。
- 检查命令、模型名和路径与代码一致。

### 模型或 Trainer

- 执行 `py_compile` 和注册检查。
- **跑 `python -m pytest`。** `tests/test_model_contracts.py` 遍历整个注册表，
  对每个模型查九项：注册配对、config 块、构造、前后向、pred/target/smasks 对齐、
  梯度有限、eval 确定性、未来 response 不影响过去预测、CPU 构造不跑到 GPU。
  新增模型没接好会直接让这一套变红。
- 构造小型 batch 执行 forward/backward。
- 检查 prediction/target shape 相同。
- 检查 loss、gradient 和输出均为 finite。
- 至少完成 1 epoch smoke run。

### 数据或预处理

- 检查样本数、用户数、序列长度和 ID 范围。
- 检查 train/valid/test 用户或交互是否按预期隔离。
- 检查 padding、shift 和 mask。
- 检查缓存 key 是否覆盖影响数据内容的参数。
- 不覆盖原始数据；新产物写入明确的新路径。

### 训练框架或公共组件

- 至少运行两个结构不同的模型，例如一个 RNN 模型和一个 attention/multi-concept 模型。
- 验证单 fold 和 CV 路径。
- 验证 best checkpoint、`run_config.json`、`metrics.jsonl`、`best_metrics.json` 和 CV summary。

若因缺少 GPU、数据或依赖无法完成验证，必须明确写出未验证项和原因，不得声称任务已完全通过。

## 9. 实验产物与报告

标准运行目录：

```text
saved_model/<dataset>-<model>-fold<fold>-<timestamp>/
```

优先复用 `run_config.json`、`metrics.jsonl`、`best_metrics.json`、best checkpoint、`last_epoch_model.pt`、`cv_summary.json` 和 `cv_summary.csv`。

研究报告至少包含：研究问题与假设、数据集与预处理、基线与公平性控制、模型结构与张量流、loss 与训练配置、主结果及方差、消融与效率、失败实验、局限性和下一步。

不要手工复制终端中某个瞬时最好值作为最终结果；应从保存的配置和 summary 文件读取。

## 10. 代码与仓库卫生

- 遵循现有 Python 风格，使用清晰的 snake_case 命名。
- 优先小而集中的修改，避免为单个模型重构整个框架。
- 不提交数据集、checkpoint、W&B 文件、缓存、PDF 提取物或大体积生成文件。
- 不把 API key、W&B 凭据或本机绝对路径写入代码和文档。
- 不修改 `configs/wandb.json`，除非用户明确要求。
- 不通过原地修改共享配置影响后续 fold；每次运行使用独立副本。
- 不吞掉异常后继续产生看似有效的指标。
- 不为了得到更高结果删除困难样本、改变 mask 或改变数据划分而不披露。
- 不改动无关文件，不格式化整个仓库。

## 11. 任务完成标准

代理在宣布完成前应确认：

- 改动与研究问题直接相关。
- 新模型、Trainer 和配置已完整注册。
- 数据字段、shape、shift 和 mask 已核对。
- 不存在明显的标签或测试集泄漏。
- 至少执行了与风险相称的验证。
- 实验命令可复现。
- 未验证内容、已知限制和用户需要执行的长时实验已明确列出。
- `git diff` 中没有无关修改。

最终回复应简洁说明：修改内容、验证结果、关键文件路径，以及尚未执行的长时训练或待决定事项。
