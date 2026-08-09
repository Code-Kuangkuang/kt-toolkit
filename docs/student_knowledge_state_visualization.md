# a removed model 学生知识状态可视化

入口脚本：`scripts/plot_student_knowledge_state.py`。

脚本从保存的 `run_config.json` 和 best-validation checkpoint 恢复模型，按
`uid` 读取 question-level 序列，并绘制：

1. 学生实际遇到知识点的状态热力图；
2. 指定知识点的状态热力图；
3. 单个知识点的状态变化折线图。

## 分数定义

图中的数值为 a removed model 的几何覆盖代理值：

```text
distance[t, c] = ||student_point[t] - concept_point[c]||_2 / 2
margin[t, c]   = student_radius[t] - distance[t, c]
mastery[t, c]  = sigmoid(beta * margin[t, c])
```

它位于 `[0, 1]`，但不是经过监督标定的“知识点答对概率”。论文图、图例和
表格中应使用 `Coverage-based mastery proxy`，不能直接称为 calibrated
mastery probability。

第 `t` 列表示观察第 `t` 次题目、知识点和作答结果之后的状态，不使用未来
交互。

## Seaborn 色图

连续型色图固定使用：

```python
sns.heatmap(data, cmap="viridis", vmin=0.0, vmax=1.0)
```

可选 `viridis`、`mako`、`rocket`，默认 `viridis`。

发散型色图固定使用：

```python
sns.heatmap(
    data,
    cmap="vlag",
    vmin=0.0,
    vmax=1.0,
    center=0.5,
)
```

可选 `vlag`、`coolwarm`。无论单个学生的实际取值范围如何，0、0.5、1 的
颜色含义始终保持一致。

## 示例

```powershell
python scripts/plot_student_knowledge_state.py `
  --run-dir saved_model/removed_model_experiment/cv-assist2009-removed_model-20260719-144319/assist2009-removed_model-fold0-20260719-144324 `
  --split test `
  --uid 520 `
  --span 30 `
  --selected-concepts 31,46,47,20 `
  --target-concept 31 `
  --cmap viridis `
  --output-dir output/student_knowledge_state/assist2009_uid520
```

输出包括 PNG、PDF、SVG、完整知识状态矩阵 `mastery_proxy.csv`、交互序列
`interaction_sequence.csv` 和记录 checkpoint、fold、配色与分数定义的
`manifest.json`。

## 搜索几何分支有解释力的案例

不要按图片观感手工挑选案例。可以在 validation split 上按固定窗口协议搜索：

```powershell
python scripts/find_removed_model_geometry_case.py `
  --run-dir saved_model/removed_model_experiment/cv-assist2009-removed_model-20260719-144319/assist2009-removed_model-fold0-20260719-144324 `
  --split valid `
  --window-size 30 `
  --window-stride 10 `
  --device cuda:0 `
  --output-dir output/removed_model_geometry_case_search/assist2009_fold0_valid
```

筛选同时考虑几何分支相对 base 分支的 log-loss 改善、目标方向一致率，以及
知识状态矩阵的动态范围。完整候选和排名会写入 CSV，避免只报告最终选中的
窗口。案例选择属于定性解释，不能代替整体 AUC、交叉验证或消融结果。

绘制后段窗口时使用 `--start-step`。脚本仍会输入此前全部历史，只裁剪展示
区域，因此不会把窗口起点错误地当作冷启动状态：

```powershell
python scripts/plot_student_knowledge_state.py `
  --run-dir <fold-run-dir> `
  --split valid `
  --uid 592 `
  --start-step 10 `
  --span 30 `
  --cmap vlag
```

额外输出的 `geometry_contribution_heatmap` 将知识状态与目标对齐的预测修正放
在同一张图中。修正值大于零表示完整 a removed model 相比 base 分支把预测推向了
实际作答结果，小于零表示该步的几何分支产生了反作用。
