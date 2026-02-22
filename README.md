## KT-Toolkit

面向知识追踪（KT）的研究工具包，涵盖数据预处理、模型训练与多数据集评估流程。

## 架构图

![架构图](docs/arch.png)

## 快速开始

1. 使用 [requirements.txt](requirements.txt) 安装依赖。
2. 在 [configs/](configs/) 配置数据集与训练参数。
3. 通过 [scripts/train.py](scripts/train.py) 启动训练。

## 环境准备

- 推荐使用 Python 3.8+。
- 依赖安装示例：

```bash
pip install -r requirements.txt
```

## 数据准备

- 原始数据位于 [data/](data/)。
- 预处理脚本位于 [preprocess/](preprocess/)；可根据数据集选择对应脚本运行。
- 处理后的数据与数据集初始化逻辑位于 [datasets/](datasets/)。

## 训练与评估

- 训练入口： [scripts/train.py](scripts/train.py)
- 清洗入口： [scripts/run_clean.py](scripts/run_clean.py)
- 训练配置： [configs/](configs/) 下的 `kt_config.json` 与 `trainer/` 子目录配置。

## 目录概览

- [models/](models/)：KT 模型实现（如 DKT、SAKT、AKT 等）。
- [datasets/](datasets/)：数据集初始化与加载逻辑。
- [preprocess/](preprocess/)：数据预处理脚本。
- [core/](core/)：训练框架与通用组件。
- [cleaning/](cleaning/)：数据清洗与适配逻辑。
- [configs/](configs/)：配置文件。
- [docs/](docs/)：文档与架构图。
