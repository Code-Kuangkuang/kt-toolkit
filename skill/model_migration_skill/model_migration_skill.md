---
name: model-migration
description: 将知识追踪模型从 pykt-toolkit 迁移到 KT-Toolkit，并核对输入协议、配置与测试。
---

# 模型迁移

以仓库 AGENTS.md 为准。开始前检查 git status，保留用户改动；阅读目标模型、上游 Trainer 和上游 sweep 配置。

1. 保留上游网络类，注册、参数改名及框架适配写在文件末尾的 adapter 子类。文件说明来源、日期、修复点与原因。
2. adapter 声明嵌套 `Inputs(InputSpec)`，写明 `dataset_mode`、`supports_multi_concept`、题目需求和特殊构造输入。
   Dataset 的模式集合由声明生成，不再维护另一份模型名单。
3. 派生图、难度、时间桶和统计量通过 `Inputs.prepare(ctx)` 返回 `ModelInputs`；使用 `ctx.train_folds()` 拟合，记录
   `feature_fit_scope` / `graph_scope`。不要在 forward 中读文件或在 runner 中增加模型名分支。
4. Trainer 继承 BaseTrainer，通常只实现 `_forward_batch`。使用公共训练、评估和早停；特殊训练参数声明
   `training_forward_kwargs`，只有对抗训练等确有不同的更新过程才覆盖循环。
5. 写出张量流：`cseqs [B,T] 或 [B,T,K] -> full [B,T+1,...] -> shifted prediction/target [B,T] -> smasks [N]`。
   padding 概念为 -1，response padding 为 0；有效位置只读 masks/smasks。预测不得使用目标答案或未来统计。
6. 多知识点使用已有掩码池化模块；不能静默取首知识点。预测与评估须遵循 checkpoint 保存的协议。
7. 更新 models / core.trainers 包导入及 kt_config；新增可覆盖参数登记到 core.run_support 的共享参数集合。
   检查构造器、CLI、WebUI、配置和 optimizer 实际消费了该参数。
8. 若 prepare 读取真实数据，给模型契约测试登记对应的合成构造参数。跑完整 pytest、注册和语法检查，
   再做单 epoch 训练；公共组件改动还要检查 RNN、attention、多 fold 和续跑产物。

默认 best checkpoint 与 early stopping 只看 validation。正式实验固定划分、fold、seed、预算和评价 mask；
不同概念或特征拟合协议不得合并。训练恢复 checkpoint 与 best-validation 权重分开保存。

不要为普通模型复制 `_train_epoch`、`_eval_epoch`、`_should_stop` 或 `evaluate_test`；不要复制 WebUI 训练框架。
不要凭猜测配置上游超参数、覆盖原始数据，或提交模型权重、缓存和研究临时文件。
