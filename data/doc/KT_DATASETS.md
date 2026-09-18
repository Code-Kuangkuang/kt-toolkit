# KT 数据集完整文档

> 最后更新：2026-09-18（第 5 节按磁盘实况重写；其余为 2026-03-18 原文）
> 数据来源：pykt-toolkit + 本地预处理文件

---

## 1. 数据集概览

| 数据集 | 交互数 | 学生数 | 题目数 | KC数 | 来源 |
|--------|--------|--------|--------|------|------|
| ASSISTments2009 | 346,860 | 4,217 | 26,688 | 123 | ASSISTments |
| ASSISTments2012 | 2,541,201 | 27,066 | 45,716 | - | ASSISTments |
| ASSISTments2015 | 708,631 | 19,917 | - | 100 | ASSISTments |
| ASSISTments2017 | 942,816 | 686 | 102 | - | ASSISTments |
| Algebra2005 | 809,694 | 574 | 210,710 | 112 | KDD Cup 2010 |
| Bridge2006 | 3,679,199 | 1,146 | 207,856 | 493 | KDD Cup 2010 |
| EdNet | 131,317,236 | 784,309 | - | - | Riiid |
| NIPS34 (Eedi) | 1,382,727 | - | 948 | 57 | NeurIPS 2020 |
| POJ | 996,240 | 22,916 | 2,750 | - | 北大多校 |

---

## 2. 数据列与特征支持

### 2.1 核心特征列

| 特征列 | 说明 | 必需 |
|--------|------|------|
| `user_id` | 学生ID | ✅ |
| `problem_id` | 题目ID | ✅ |
| `skill_id` / `kc` | 知识点ID | ✅ |
| `correct` | 回答是否正确 (0/1) | ✅ |
| `timestamp` | 回答时间戳 | ⚠️ 部分需要 |
| `response_time` | 答题耗时 (ms) | ⚠️ 部分需要 |

### 2.2 各数据集特征矩阵

| 数据集 | Question ID | Skill ID | Answer Result | Response Duration | Answer Submit Time |
|--------|:------------:|:--------:|:-------------:|:-----------------:|:------------------:|
| **Statics2011** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **ASSISTments2009** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **ASSISTments2012** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ASSISTments2015** | ❌ | ✅ | ✅ | ❌ | ❌ |
| **ASSISTments2017** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Algebra2005** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Bridge2006** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **EdNet** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NIPS34** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **POJ** | ❌ | ✅ | ✅ | ❌ | ❌ |

---

## 3. 各数据集详细说明

### 3.1 ASSISTments2009

| 项目 | 内容 |
|------|------|
| **来源** | https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010 |
| **本地文件** | `data/assist2009/skill_builder_data_corrected_collapsed.csv` |
| **预处理输出** | `data/assist2009/data.txt` |
| **核心字段** | `user_id`, `problem_id`, `skill_id`, `correct` |
| **可选字段** | - |
| **特点** | 经典基线数据集，无时间信息 |

**模型兼容性**：
- ✅ DKT, SAKT, SAINT, AKT, SimpleKT
- ✅ UKT (需要时间特征 → **不支持**)

---

### 3.2 ASSISTments2012

| 项目 | 内容 |
|------|------|
| **来源** | https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect |
| **本地文件** | `data/assist2012/2012-2013-data-with-predictions-4-final.csv` |
| **核心字段** | `user_id`, `problem_id`, `skill_id`, `correct`, `start_time`, `ms_first_response` |
| **可选字段** | `ms_first_response` (答题耗时) |
| **特点** | 包含 affect (情感) 数据，但预处理未提取 |

**模型兼容性**：
- ✅ 所有基于 attention 的模型
- ✅ UKT (有 timestamp)

---

### 3.3 ASSISTments2015

| 项目 | 内容 |
|------|------|
| **来源** | https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data |
| **本地文件** | `data/assist2015/2015_100_skill_builders_main_problems.csv` |
| **核心字段** | `user_id`, `sequence_id` (作为 skill), `correct` |
| **可选字段** | - |
| **特点** | 无 question_id，以 sequence_id 替代；学生数最多 |

**模型兼容性**：
- ✅ DKT, SAKT, AKT, SimpleKT
- ❌ 需要 question_id 的模型 (如 SAKT)
- ❌ UKT

---

### 3.4 ASSISTments2017

| 项目 | 内容 |
|------|------|
| **来源** | https://sites.google.com/view/assistmentsdatamining/dataset |
| **本地文件** | `data/assist2017/anonymized_full_release_competition_dataset.csv` |
| **预处理输出** | `data/assist2017/data.txt` |
| **核心字段** | `studentId`, `problemId`, `skill`, `correct`, `startTime`, `timeTaken` |
| **可选字段** | `timeTaken` (ms) |
| **特点** | 数据量适中，包含答题时间 |

**模型兼容性**：
- ✅ 所有模型
- ✅ UKT (完整时间特征)

---

### 3.5 Algebra2005

| 项目 | 内容 |
|------|------|
| **来源** | https://pslcdatashop.web.cmu.edu/KDDCup/ |
| **本地文件** | `data/algebra2005/algebra_2005_2006_master.txt` |
| **预处理输出** | `data/algebra2005/data.txt` |
| **核心字段** | `Anon Student Id`, `Questions` (=Problem Name + Step Name), `KC(Default)`, `Correct First Attempt`, `First Transaction Time` |
| **可选字段** | `First Transaction Time` |
| **特点** | 题目数远大于 KC 数 (Q:KC ≈ 1500:1)，需要 concat problem+step 作为唯一题目 |

**模型兼容性**：
- ✅ 所有模型
- ✅ UKT (有 timestamp)
- ⚠️ 题目太细，需聚合使用

---

### 3.6 Bridge2006

| 项目 | 内容 |
|------|------|
| **来源** | https://pslcdatashop.web.cmu.edu/KDDCup/ |
| **本地文件** | `data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt` |
| **核心字段** | `Anon Student Id`, `Questions`, `KC(SubSkills)`, `Correct First Attempt`, `First Transaction Time` |
| **可选字段** | `First Transaction Time` |
| **特点** | 与 Algebra2005 格式相同，KC 数更多 (493) |

**模型兼容性**：
- ✅ 所有模型
- ✅ UKT

---

### 3.7 EdNet

| 项目 | 内容 |
|------|------|
| **来源** | https://github.com/riiid/ednet |
| **本地文件** | `data/ednet/`（已建好，见第 5 节） |
| **核心字段** | `user_id`, `question_id`, `tags` (作为 skill), `correct`, `timestamp`, `elapsed_time` |
| **可选字段** | `elapsed_time` |
| **特点** | 规模最大 (1.3亿交互)，需采样使用；包含 tags (多知识点) |

**模型兼容性**：
- ✅ 所有模型
- ✅ UKT (完整时间特征)

---

### 3.8 NIPS34 (Eedi)

| 项目 | 内容 |
|------|------|
| **来源** | https://eedi.com/projects/neurips-education-challenge |
| **预处理脚本** | `data/preprocess/nips_task34_preprocess.py` |
| **核心字段** | `UserId`, `QuestionId`, `SubjectId_level3` (作为 skill), `IsCorrect`, `answer_timestamp` |
| **可选字段** | `answer_timestamp` |
| **特点** | 从 subject tree 取 leaf node 作为 KC；包含 metadata 文件 |

**模型兼容性**：
- ✅ 所有模型
- ✅ UKT

---

### 3.9 POJ (Peking University OJ)

| 项目 | 内容 |
|------|------|
| **来源** | Google Drive (见 pykt) |
| **预处理脚本** | `data/preprocess/poj_preprocess.py` |
| **核心字段** | `User`, `Problem` (作为 skill), `Result` |
| **可选字段** | - |
| **特点** | 编程题数据集，Problem 充当 skill；无 question_id |

**模型兼容性**：
- ✅ DKT, AKT, SimpleKT
- ❌ 需要 question_id 的模型
- ❌ UKT (无时间特征)

---

## 4. 模型特征需求对照表

| 模型 | 必须特征 | 可选特征 | 不支持的数据集 |
|------|----------|----------|----------------|
| **DKT** | user_id, skill_id, correct | - | - |
| **SAKT** | user_id, problem_id, skill_id, correct | - | ASSIST2015, POJ |
| **SAINT** | user_id, problem_id, skill_id, correct | - | ASSIST2015, POJ |
| **AKT** | user_id, problem_id, skill_id, correct | difficulty | ASSIST2015, POJ |
| **SimpleKT** | user_id, problem_id, skill_id, correct | - | ASSIST2015, POJ |
| **UKT** | user_id, skill_id, correct | timestamp | ASSIST2009, ASSIST2015, POJ |
| **DKVMN** | user_id, skill_id, correct | - | - |

---

## 5. 本地数据集文件结构

上一版这里是一棵手写的目录树，到 2026-09 已经与磁盘不符（写着 junyi2015「待下载」、
ednet 只有 `KT1/ contents/` 结构，而两者都已建好）。手写树留不住，换成从磁盘可核对的
状态表。

`data/<数据集>/` 下的产物一律被 `.gitignore` 排除，仓库里只有本目录的文档。

| 数据集 | data.txt | keyid2idx | 切分文件 | qmatrix |
|---|:--:|:--:|:--:|:--:|
| aaai2023 | – | ✅ | ✅ | – |
| algebra2005 | ✅ | ✅ | ✅ | – |
| assist2009 | ✅ | ✅ | ✅ | ✅ |
| assist2012 | ✅ | ✅ | ✅ | – |
| assist2015 | – | – | – | – |
| assist2017 | ✅ | ✅ | ✅ | ✅ |
| bridge2algebra2006 | ✅ | ✅ | ✅ | – |
| ednet | ✅ | ✅ | ✅ | – |
| junyi2015 | ✅ | ✅ | ✅ | – |
| junyi_sub5k | – | ✅ | ✅ | – |
| nips_task34 | ✅ | ✅ | ✅ | – |
| slepemapy | ✅ | ✅ | ✅ | – |
| statics2011 | ✅ | ✅ | ✅ | – |

十三个里十二个已完整预处理，只有 assist2015 是空的。`qmatrix.npz` 只有两个数据集有，
因为它由题目级切分产出，其余数据集没跑过那一步。

**别在这里查数据集的来源和采样。** 每个数据集的 `source` / `sampling` /
`sampling_note` 字段在 `configs/data_config.json` 里，`tests/test_dataset_inventory.py`
会强制它们存在且自洽；三个「名字是全量、实际是样本」的数据集在
`docs/architecture.md` 的 "Which datasets are samples" 一节有说明。那里是单一事实来源，
本文件只描述格式与字段语义。


---

## 6. 缺失特征与解决方案

### 6.1 缺失 timestamp

**影响数据集**：ASSISTments2009, ASSISTments2015, POJ

**解决方案**：
1. 放弃时间相关特征 (如 UKT)
2. 使用序列位置作为隐式时间信号
3. 自行补充时间戳 (若原始数据有)

### 6.2 缺失 question_id

**影响数据集**：ASSISTments2015, POJ

**解决方案**：
1. 使用 skill_id 替代 question_id
2. 自行构造虚拟 question_id (= skill_id)

### 6.3 缺失 response_time

**影响数据集**：ASSISTments2009, Algebra2005, Bridge2006, NIPS34, POJ

**解决方案**：
1. 忽略需要答题时间的模型
2. 自行计算 (若有 start_time 和 end_time)

---

## 7. 预处理命令参考

```bash
# 预处理单个数据集
cd data/preprocess
python -c "from assist2009_preprocess import read_data_from_csv; read_data_from_csv('../assist2009/skill_builder_data_corrected_collapsed.csv', '../assist2009/data.txt')"

# 批量预处理 (参考 split_datasets.py)
python split_datasets.py
```

---

## 8. 建议

### 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| **基线对比** | SimpleKT | 简单高效 |
| **长序列** | UKT + Mamba | 不确定性建模 |
| **有时间特征** | AKT | 遗忘机制 |
| **多知识点** | GIKT | 图建模 |

### 数据集选择建议

| 目标 | 推荐数据集 |
|------|------------|
| **快速验证** | ASSISTments2009 |
| **规模测试** | EdNet (采样) |
| **真实场景** | ASSISTments2017 |
| **知识点关联** | Bridge2006 / Algebra2005 |

---

*文档自动生成于 2026-03-18*
