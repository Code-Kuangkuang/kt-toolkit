# KT-Toolkit 架构冗余与优化建议

日期：2026-06-15

## 总体判断

项目主架构是清楚的：CLI/WebUI 入口、`core` 训练编排、registry/factory、`datasets`、`models`、`trainers` 分层基本合理，不需要推倒重来。

当前真正需要优化的是“模型特例”和“训练模板代码”已经开始扩散。继续新增模型时，如果仍然依靠多处 `if model_name == ...` 分支和复制 trainer 模板，维护成本会持续升高。

## 主要冗余点

### 1. 模型能力分散在字符串分支里

`core/train_runner.py` 中维护了 `QUESTION_REQUIRED_MODELS`、LPKT/Hawkes/DIMKT/GKT/DKT-forget 等大量模型特例；`datasets/init_dataset.py` 又重复按 `model_name` 决定 `dataset_mode`、`concept_mode`。

建议引入 `ModelSpec` 或 `ModelRuntimeSpec`。每个模型注册自己的运行能力：

- `requires_questions`
- `dataset_mode`
- `concept_mode`
- `use_timestamps`
- `extra_features`
- `prepare_context()`

这样新增模型时，不需要同时修改 `train_runner`、`datasets`、WebUI 和多个脚本。

### 2. Trainer 样板代码重复很高

多个 trainer 中重复了以下逻辑：

- `__init__`
- `_train_epoch`
- `optimizer.zero_grad/backward/step`
- 进度条打印
- `masked_bce`
- shifted prediction 对齐
- tensor 移动到 device

典型重复文件包括：

- `core/trainers/dkt_trainer.py`
- `core/trainers/simplekt_trainer.py`
- `core/trainers/stablekt_trainer.py`
- `core/trainers/sparsekt_trainer.py`

建议在 `core/trainer.py` 上再拆一个 `StandardKTTrainer`，统一构造函数、训练 epoch、`to_device_dict`、`align_shifted_preds`、`masked_bce`。各模型 trainer 只保留 `_forward_batch` 或 `compute_loss`。

### 3. 模型构建参数过滤重复

`OTHER_CONFIG_KEYS` / `MODEL_CONFIG_EXCLUDE` 这类过滤逻辑至少出现在：

- `core/train_runner.py`
- `webui/model_structure.py`
- `scripts/reproduce_best_valid_test.py`
- `scripts/predict_peiyou.py`

建议集中成 `core/model_builder.py`，提供：

- `resolve_model_kwargs()`
- `build_model_from_configs()`
- `build_model_from_run_config()`

WebUI、训练入口、复现实验脚本、预测脚本都复用同一套模型构建逻辑。

### 4. 数据加载 builder 重复

`datasets/init_dataset.py` 的 train/valid builder 和 test builder 重复了 mode 决策、路径选择、Dataset 构造。

建议抽一个统一函数：

```python
build_split_dataset(split="train" | "valid" | "test")
```

再由 `build_dataloaders` 组合 train/valid，由 `build_test_dataloaders` 复用同一逻辑。

### 5. 清洗/预处理层有历史包袱

`cleaning/adapters/*.py` 基本都是同一模板：

```text
process_raw_data -> split_concept -> split_question
```

同时 `configs/dataset/*.yaml` 和 `configs/data_config.json` 维护了大量同类字段。

建议：

- 把普通 adapter 合并成一个通用 adapter。
- 特殊数据集再单独实现专用 adapter。
- 长期让 `configs/dataset/*.yaml` 成为清洗输入源。
- 让 `configs/data_config.json` 作为生成后的训练元数据，而不是人工长期维护两份相近配置。

### 6. 存在潜在冲突/死代码

`core/trainers/my_gbkt_trainer.py` 也注册了 `"gbkt"`，和 `core/trainers/gbkt_trainer.py` 冲突。

当前 `my_gbkt_trainer.py` 没有在 `core/trainers/__init__.py` 中导入，所以暂时不会触发注册冲突。但未来如果被导入，会直接注册失败。

建议删除、归档或改名，避免后续误导维护者。

### 7. 模型内部 Transformer/Attention 组件重复

多个模型内部都有相似的 Transformer/Attention/Positional Embedding 组件，例如：

- `models/akt.py`
- `models/simplekt.py`
- `models/stablekt.py`
- `models/sparsekt.py`
- `models/robustkt.py`
- `models/ukt.py`

这部分重复确实存在，但牵涉论文实现细节和指标复现，不建议第一阶段大规模抽象。更稳妥的做法是先补训练回归测试，再逐步提取稳定的底层层组件，例如 `models/layers/attention.py`、`models/layers/position.py`。

## 优先级建议

### 第一阶段：低风险高收益

1. 删除或归档 `core/trainers/my_gbkt_trainer.py` 这类未接入且会冲突的文件。
2. 把模型参数过滤和模型构建集中到 `core/model_builder.py`。
3. 提取 `StandardKTTrainer`，先覆盖 `stablekt/sparsekt/simplekt/dkt` 这类重复明显的 trainer。

### 第二阶段：收敛模型运行能力

引入 `ModelSpec`，把 `train_runner` 和 `datasets.init_dataset` 里的模型名分支迁进去。

示例结构：

```python
@MODEL_SPEC_REGISTRY.register("lpkt")
ModelSpec(
    dataset_mode="all_in_one",
    use_timestamps=True,
    requires_questions=True,
    prepare_context=prepare_lpkt_context,
)
```

这样 `train_one_fold` 只负责读取 spec 并执行通用流程。

### 第三阶段：收敛数据与清洗配置

1. 合并普通 cleaning adapter。
2. 明确 `configs/dataset/*.yaml` 和 `configs/data_config.json` 的职责边界。
3. 将重复字段从人工维护改为生成或继承。

### 第四阶段：谨慎抽取模型层组件

在关键模型有回归测试后，再提取 Transformer/Attention/Positional Embedding 等公共组件。不要一开始就大改模型内部实现，否则容易影响论文复现实验。

## 建议补充的测试

在做架构改造前，建议先补最小测试集：

- `ModelSpec` 解析测试。
- `build_model_from_run_config` 测试。
- `build_split_dataset` 的 train/valid/test 路径选择测试。
- `StandardKTTrainer` 的 mask、loss、shape 对齐测试。
- CV 聚合结果测试。

这些测试不需要完整训练模型，可以用小 batch 和假模型覆盖核心行为。

## 结论

当前项目的方向是合理的，问题不是分层错误，而是研究型项目常见的“新增模型靠复制模板 + 多处字符串分支”的增长方式已经出现。

最推荐的优化路线是先收敛 trainer 样板和模型构建，再用 `ModelSpec` 消化模型特例，最后再碰模型内部公共层。这样改动收益明显，风险也可控。
