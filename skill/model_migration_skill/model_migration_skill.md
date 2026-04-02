---
name: model-migration
description: 将KT模型从pykt-toolkit迁移到kt-toolkit。当用户请求将某个模型（如qikt、dkt、akt等）从pykt-toolkit项目迁移到kt-toolkit项目时触发此skill。
---

# 模型迁移 Skill

# Context
当用户请求将模型从 pykt-toolkit 迁移到 kt-toolkit 时触发。

# Task
将指定模型从 pykt-toolkit 迁移到 kt-toolkit，并创建对应的 trainer。

## 迁移步骤

### 0. 分析源模型
- 读取 `pykt-toolkit/pykt/models/{model_name}.py` 源文件
- 理解模型的网络结构、参数、forward逻辑
- 识别模型依赖的基类（如 QueBaseModel）
- 查看是否有对应的配置文件（如 yaml）

### 1. 创建 Model 文件
在 `kt-toolkit/models/{model_name}.py` 中：

1. **注册模型**：使用 `@MODEL_REGISTRY.register("{model_name}")`
2. **保留核心结构**：
   - Embedding层
   - 核心网络层（LSTM/Transformer/Attention等）
   - 输出层
3. **简化继承**：直接继承 `nn.Module`，不使用 pykt 的 QueBaseModel
4. **适配kt-toolkit风格**：
   - 使用统一的参数命名（num_c, num_q, emb_size, dropout等）
   - 保持与现有模型（如AKT、SAKT）一致的接口

### 2. 创建 Trainer 文件
在 `kt-toolkit/core/trainers/{model_name}_trainer.py` 中：

1. **注册Trainer**：使用 `@TRAINER_REGISTRY.register("{model_name}")`
2. **继承BaseTrainer**：参考现有trainer（AKTTrainer、SAKTTrainer）
3. **实现核心方法**：
   - `_train_epoch`: 训练一个epoch
   - `_eval_epoch`: 评估并返回metrics
   - `_forward_batch`: 处理单个batch
   - `_should_stop`: 早停逻辑
4. **计算Loss**：使用 `binary_cross_entropy`
5. **测试集评估**：实现 `evaluate_test()` 方法（在最佳epoch后在测试集上评估）

### 3. 更新 __init__.py
- 在 `models/__init__.py` 添加导入
- 在 `core/trainers/__init__.py` 添加导入

### 4. 注册到工厂函数
确保 `core/factory.py` 的 build_model 和 build_trainer 能正确创建模型。

### 5. 数据集配置
- **默认ALL-in-One模式**：kt-toolkit 默认使用 `all_in_one` 模式（避免数据泄露）
- 如需 One-by-One 模式：显式指定 `--dataset_mode one_by_one`
- 如需要2D概念序列（ALL-in-One），确保 `data_config.json` 中有 `train_valid_file_quelevel` 和 `max_concepts` 配置

## 注意事项
- 处理数据对齐：pykt和kt-toolkit的数据格式可能略有不同
- 保持emb_type兼容性
- 如有特殊loss权重配置，通过other_config传递
- 考虑是否需要QueEmb（概念嵌入层）
- **训练后自动测试集评估**：trainer需要实现 `evaluate_test()` 方法，训练Runner会在最佳epoch后自动在测试集上评估

## ONE-BY-ONE vs ALL-IN-ONE 模式

### One-by-One 模式
- 使用 `KTDataset`，概念序列为 1D `[B, T]`
- 每个位置只有一个概念ID
- 适用模型：DKT, SAKT, AKT, DKVMN等

### ALL-in-One 模式
- 使用 `KTQueDataset`，概念序列为 2D `[B, T, K]`
- 每个位置最多K个概念，用`_`分隔，`-1`填充
- 适用模型：QIKT, IEKT, QDKT等
- 数据文件：`*_quelevel.csv`
- **kt-toolkit默认模式**

## kt-toolkit 当前默认配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| dataset_mode | "all_in_one" | 默认使用2D概念序列 |
| add_uuid | 0 | 默认不添加UUID后缀 |
| 5折CV | 启用 | 每折训练后会在测试集上评估 |

## 使用示例

**用户输入**：
```
将qikt模型迁移到kt-toolkit
```

**执行流程**：
1. 读取 `pykt-toolkit/pykt/models/qikt.py`
2. 分析QIKTNet网络结构和QueEmb
3. 在 `kt-toolkit/models/qikt.py` 创建模型
4. 在 `kt-toolkit/core/trainers/qikt_trainer.py` 创建trainer（含evaluate_test方法）
5. 更新 `models/__init__.py` 和 `core/trainers/__init__.py`
6. 输出完成提示

**用户输入**：
```
迁移dkt模型
```

**执行流程**：
1. 读取 `pykt-toolkit/pykt/models/dkt.py`
2. 分析DKT网络结构
3. 在 `kt-toolkit/models/dkt.py` 创建模型
4. 在 `kt-toolkit/core/trainers/dkt_trainer.py` 创建trainer（含evaluate_test方法）
5. 更新相关 `__init__.py`
6. 输出完成提示
