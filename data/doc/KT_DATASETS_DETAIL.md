# KT 数据集完整文档 - 详细列信息

> 最后更新：2026-03-18
> 数据来源：pykt-toolkit + 本地原始文件分析

---

## 目录

1. [数据列汇总表](#1-数据列汇总表)
2. [ASSISTments 数据集详细列](#2-assistments-数据集详细列)
3. [KDD Cup 数据集 (Algebra/Bridge)](#3-kdd-cup-数据集-algebrabridge)
4. [其他数据集](#4-其他数据集)
5. [特征可用性矩阵](#5-特征可用性矩阵)
6. [模型适配建议](#6-模型适配建议)

---

## 1. 数据列汇总表

### 1.1 核心必需列 (KT 模型必须)

| 列名 | 说明 | 数据类型 |
|------|------|----------|
| `user_id` / `studentId` / `Anon Student Id` | 学生唯一标识 | string/int |
| `skill_id` / `kc` / `skill` | 知识点/技能ID | string/int |
| `correct` | 回答是否正确 | 0/1 |

### 1.2 核心可选列 (增强模型)

| 列名 | 说明 | 数据类型 |
|------|------|----------|
| `problem_id` / `question_id` | 题目唯一标识 | string/int |
| `timestamp` / `start_time` / `First Transaction Time` | 答题时间戳 | int (Unix) |
| `response_time` / `ms_first_response` / `timeTaken` | 答题耗时 | int (ms) |

---

## 2. ASSISTments 数据集详细列

### 2.1 ASSISTments2009

**文件**: `assist2009/skill_builder_data_corrected_collapsed.csv`  
**官网**: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `order_id` | 交互顺序号 | ❌ | int |
| `assignment_id` | 作业ID | ❌ | int |
| `user_id` | **学生ID** | ✅ | int |
| `assistment_id` | ASSISTment系统ID | ❌ | int |
| `problem_id` | **题目ID** | ✅ | int |
| `original` | 原始题目ID | ❌ | int |
| `correct` | **回答是否正确 (0/1)** | ✅ | int |
| `attempt_count` | 尝试次数 | ❌ | int |
| `ms_first_response` | 首次响应时间(ms) | ❌ | int |
| `tutor_mode` | 辅导模式 | ❌ | string |
| `answer_type` | 答案类型 | ❌ | string |
| `sequence_id` | 序列ID | ❌ | int |
| `student_class_id` | 班级ID | ❌ | int |
| `position` | 位置 | ❌ | int |
| `type` | 类型 | ❌ | string |
| `base_sequence_id` | 基础序列ID | ❌ | int |
| `skill_id` | **知识点ID** | ✅ | int |
| `skill_name` | 知识点名称 | ❌ | string |
| `teacher_id` | 教师ID | ❌ | int |
| `school_id` | 学校ID | ❌ | int |
| `hint_count` | 使用提示数 | ❌ | int |
| `hint_total` | 提示总数 | ❌ | int |
| `overlap_time` | **实际答题时间** (毫秒，学生在该题实际花费的总时长) | ⚠️ | int |
| `template_id` | 模板ID | ❌ | int |
| `answer_id` | 答案ID | ❌ | int |
| `answer_text` | 答案文本 | ❌ | string |
| `first_action` | 首次动作 | ❌ | string |
| `bottom_hint` | 底部提示 | ❌ | int |
| `opportunity` | 机会数 | ❌ | int |
| `opportunity_original` | 原始机会数 | ❌ | int |

> **⚠️ 时间字段说明** (ASSISTments2009):
> - `ms_first_response`: 学生首次响应题目所用时间 (毫秒)
>   - 数据分布：平均 44s，52% 在 5-30s 区间
> - `overlap_time`: 学生实际花费在该题目的总时间 (毫秒)
>   - 当 `ms_first_response == overlap_time`：学生连续做题，无重叠
>   - 当 `ms_first_response < overlap_time`：可能同时处理多题或有延迟操作
> - **注意**: 当前预处理脚本 `assist2009_preprocess.py` 未提取这两个字段，如需使用需手动修改预处理代码

**KT模型关键字段映射**:
```
user_id = user_id
problem_id = problem_id
skill_id = skill_id
correct = correct
```

---

### 2.2 ASSISTments2012

**文件**: `assist2012/2012-2013-data-with-predictions-4-final.csv`  
**官网**: https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `problem_log_id` | 问题日志ID | ❌ | int |
| `skill` | 技能名称 | ⚠️ | string |
| `problem_id` | **题目ID** | ✅ | int |
| `user_id` | **学生ID** | ✅ | int |
| `assignment_id` | 作业ID | ❌ | int |
| `assistment_id` | ASSISTment ID | ❌ | int |
| `start_time` | **开始时间** | ✅ | timestamp |
| `end_time` | 结束时间 | ❌ | timestamp |
| `problem_type` | 题目类型 | ❌ | string |
| `original` | 原始题目ID | ❌ | int |
| `correct` | **回答是否正确** | ✅ | int |
| `bottom_hint` | 底部提示 | ❌ | int |
| `hint_count` | 提示数 | ❌ | int |
| `actions` | 动作数 | ❌ | int |
| `attempt_count` | 尝试次数 | ❌ | int |
| `ms_first_response` | **首次响应时间** | ✅ | int |
| `tutor_mode` | 辅导模式 | ❌ | string |
| `sequence_id` | 序列ID | ❌ | int |
| `student_class_id` | 班级ID | ❌ | int |
| `position` | 位置 | ❌ | int |
| `type` | 类型 | ❌ | string |
| `base_sequence_id` | 基础序列ID | ❌ | int |
| `skill_id` | **知识点ID** | ✅ | int |
| `teacher_id` | 教师ID | ❌ | int |
| `school_id` | 学校ID | ❌ | int |
| `overlap_time` | 重叠时间 | ❌ | int |
| `template_id` | 模板ID | ❌ | int |
| `answer_id` | 答案ID | ❌ | int |
| `answer_text` | 答案文本 | ❌ | string |
| `first_action` | 首次动作 | ❌ | string |
| `Average_confidence(FRUSTRATED)` | 平均沮丧置信度 | ❌ | float |
| `Average_confidence(CONFUSED)` | 平均困惑置信度 | ❌ | float |
| `Average_confidence(CONCENTRATING)` | 平均专注置信度 | ❌ | float |
| `Average_confidence(BORED)` | 平均无聊置信度 | ❌ | float |

**KT模型关键字段映射**:
```
user_id = user_id
problem_id = problem_id
skill_id = skill_id (或 skill)
correct = correct
timestamp = start_time
response_time = ms_first_response
```

---

### 2.3 ASSISTments2015

**文件**: `assist2015/2015_100_skill_builders_main_problems.csv`  
**官网**: https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `user_id` | **学生ID** | ✅ | int |
| `log_id` | 日志ID | ❌ | int |
| `sequence_id` | **序列/技能ID** | ⚠️ | int |
| `correct` | **回答是否正确** | ✅ | int |

**特点**: 列极少，无 question_id，无 timestamp

**KT模型关键字段映射**:
```
user_id = user_id
# 无 problem_id，使用 sequence_id 代替
skill_id = sequence_id
correct = correct
```

---

### 2.4 ASSISTments2017 (最丰富)

**文件**: `assist2017/anonymized_full_release_competition_dataset.csv`  
**官网**: https://sites.google.com/view/assistmentsdatamining/dataset

#### 2.4.1 学生层面特征 (静态)

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `studentId` | **学生ID** | ✅ | int |
| `MiddleSchoolId` | 中学ID | ❌ | int |
| `InferredGender` | 推断性别 | ❌ | string |
| `SY ASSISTments Usage` | 学年使用情况 | ❌ | string |
| `AveKnow` | 平均知识水平 | ❌ | float |
| `AveCarelessness` | 平均粗心程度 | ❌ | float |
| `AveCorrect` | 平均正确率 | ❌ | float |
| `NumActions` | 动作数量 | ❌ | int |

#### 2.4.2 情感/行为特征 (学生-题目粒度)

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `AveResBored` | 无聊反应平均 | ❌ | float |
| `AveResEngcon` | 参与反应平均 | ❌ | float |
| `AveResConf` | 自信反应平均 | ❌ | float |
| `AveResFrust` | 沮丧反应平均 | ❌ | float |
| `AveResOfftask` | 分心反应平均 | ❌ | float |
| `AveResGaming` | 作弊反应平均 | ❌ | float |

#### 2.4.3 交互层面特征 (每条记录)

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `action_num` | 动作编号 | ❌ | int |
| `skill` | **技能名称** | ✅ | string |
| `problemId` | **题目ID** | ✅ | int |
| `problemType` | 题目类型 | ❌ | string |
| `assignmentId` | 作业ID | ❌ | int |
| `assistmentId` | ASSISTment ID | ❌ | int |
| `startTime` | **开始时间 (Unix timestamp × 1000)** | ✅ | int |
| `endTime` | 结束时间 | ❌ | int |
| `timeTaken` | **答题耗时 (ms)** | ✅ | int |
| `correct` | **回答是否正确** | ✅ | int |
| `original` | 原始题目ID | ❌ | int |
| `hint` | 是否使用提示 | ❌ | int |
| `hintCount` | 提示数量 | ❌ | int |
| `hintTotal` | 提示总数 | ❌ | int |
| `scaffold` | 脚手架使用 | ❌ | int |
| `bottomHint` | 底部提示 | ❌ | int |
| `attemptCount` | 尝试次数 | ❌ | int |

#### 2.4.4 学习行为特征

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `frIsHelpRequest` | 是否请求帮助 | ❌ | int |
| `frPast5HelpRequest` | 过去5次帮助请求 | ❌ | int |
| `frPast8HelpRequest` | 过去8次帮助请求 | ❌ | int |
| `stlHintUsed` | STL提示使用 | ❌ | int |
| `past8BottomOut` | 过去8次底部突破 | ❌ | int |
| `totalFrPercentPastWrong` | 过去错误比例 | ❌ | float |
| `totalFrPastWrongCount` | 过去错误次数 | ❌ | int |
| `consecutiveErrorsInRow` | 连续错误数 | ❌ | int |

#### 2.4.5 置信度特征

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `confidence(BORED)` | 无聊置信度 | ❌ | float |
| `confidence(CONCENTRATING)` | 专注置信度 | ❌ | float |
| `confidence(CONFUSED)` | 困惑置信度 | ❌ | float |
| `confidence(FRUSTRATED)` | 沮丧置信度 | ❌ | float |
| `confidence(OFF TASK)` | 分心置信度 | ❌ | float |
| `confidence(GAMING)` | 作弊置信度 | ❌ | float |

#### 2.4.6 其他特征

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `Ln-1` | 前测成绩 | ❌ | int |
| `Ln` | 后测成绩 | ❌ | int |
| `MCAS` | MCAS成绩 | ❌ | int |
| `Enrolled` | 入学状态 | ❌ | int |
| `Selective` | 选择性 | ❌ | int |
| `isSTEM` | 是否STEM | ❌ | int |

**KT模型关键字段映射**:
```
user_id = studentId
problem_id = problemId
skill_id = skill
correct = correct
timestamp = startTime (÷1000)
response_time = timeTaken
```

---

## 3. KDD Cup 数据集 (Algebra/Bridge)

### 3.1 Algebra2005

**文件**: `algebra2005/algebra_2005_2006_master.txt`  
**官网**: https://pslcdatashop.web.cmu.edu/KDDCup/

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `Row` | 行号 | ❌ | int |
| `Anon Student Id` | **学生ID** | ✅ | string |
| `Problem Hierarchy` | 问题层级 | ❌ | string |
| `Problem Name` | **问题名称** | ⚠️ | string |
| `Problem View` | 问题视图 | ❌ | int |
| `Step Name` | **步骤名称** | ⚠️ | string |
| `Step Start Time` | 步骤开始时间 | ❌ | timestamp |
| `First Transaction Time` | **首次交易时间** | ✅ | timestamp |
| `Correct Transaction Time` | 正确交易时间 | ❌ | timestamp |
| `Step End Time` | 步骤结束时间 | ❌ | timestamp |
| `Step Duration (sec)` | 步骤时长(秒) | ❌ | float |
| `Correct Step Duration (sec)` | 正确步骤时长 | ❌ | float |
| `Error Step Duration (sec)` | 错误步骤时长 | ❌ | float |
| `Correct First Attempt` | **首次尝试是否正确** | ✅ | int |
| `Incorrects` | 错误次数 | ❌ | int |
| `Hints` | 提示次数 | ❌ | int |
| `Corrects` | 正确次数 | ❌ | int |
| `KC(Default)` | **知识点(默认)** | ✅ | string |
| `Opportunity(Default)` | 机会数(默认) | ❌ | int |

**预处理方式** (pykt):
- `question_id` = `Problem Name` + `----` + `Step Name`
- `timestamp` = `First Transaction Time`

---

### 3.2 Bridge2006

**文件**: `bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt`  
**官网**: https://pslcdatashop.web.cmu.edu/KDDCup/

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `Row` | 行号 | ❌ | int |
| `Anon Student Id` | **学生ID** | ✅ | string |
| `Problem Hierarchy` | 问题层级 | ❌ | string |
| `Problem Name` | **问题名称** | ⚠️ | string |
| `Problem View` | 问题视图 | ❌ | int |
| `Step Name` | **步骤名称** | ⚠️ | string |
| `Step Start Time` | 步骤开始时间 | ❌ | timestamp |
| `First Transaction Time` | **首次交易时间** | ✅ | timestamp |
| `Correct Transaction Time` | 正确交易时间 | ❌ | timestamp |
| `Step End Time` | 步骤结束时间 | ❌ | timestamp |
| `Step Duration (sec)` | 步骤时长(秒) | ❌ | float |
| `Correct Step Duration (sec)` | 正确步骤时长 | ❌ | float |
| `Error Step Duration (sec)` | 错误步骤时长 | ❌ | float |
| `Correct First Attempt` | **首次尝试是否正确** | ✅ | int |
| `Incorrects` | 错误次数 | ❌ | int |
| `Hints` | 提示次数 | ❌ | int |
| `Corrects` | 正确次数 | ❌ | int |
| `KC(SubSkills)` | **知识点(子技能)** | ✅ | string |
| `Opportunity(SubSkills)` | 机会数(子技能) | ❌ | int |

**与 Algebra2005 区别**: 使用 `KC(SubSkills)` 而非 `KC(Default)`

---

## 4. 其他数据集

### 4.1 EdNet

**来源**: https://github.com/riiid/ednet  
**结构**: `dataset/ednet/`

**KT1 文件 (用户答题记录)**:
| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `user_id` | **学生ID** | ✅ | int |
| `question_id` | **题目ID** | ✅ | string |
| `correct_answer` | 正确答案 | ❌ | string |
| `user_answer` | 用户答案 | ❌ | string |
| `elapsed_time` | **答题耗时** | ✅ | int |
| `timestamp` | **时间戳** | ✅ | int |

**contents/questions.csv (题目信息)**:
| 列名 | 说明 | 类型 |
|------|------|------|
| `question_id` | 题目ID | string |
| `bundle_id` | Bundle ID | string |
| `exam_id` | 考试ID | string |
| `type` | 类型 | string |
| `content_id` | 内容ID | string |
| `correct_answer` | 正确答案 | string |
| `tags` | **知识点标签** | string |

---

### 4.2 POJ

**来源**: 北大编程练习平台  
**预处理脚本**: `preprocess/poj_preprocess.py`

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `User` | **学生ID** | ✅ | int |
| `Problem` | **题目ID (兼作skill)** | ✅ | string |
| `Result` | **提交结果** | ✅ | string |
| `Submit Time` | 提交时间 | ❌ | timestamp |

**Result 映射**:
- `Accepted` → 1
- 其他 (Wrong Answer, Compile Error 等) → 0

---

### 4.3 NIPS34 (Eedi)

**来源**: NeurIPS 2020 Education Challenge  
**预处理脚本**: `preprocess/nips_task34_preprocess.py`

| 列名 | 说明 | KT可用 | 类型 |
|------|------|:------:|------|
| `UserId` | **学生ID** | ✅ | int |
| `QuestionId` | **题目ID** | ✅ | int |
| `IsCorrect` | **是否正确** | ✅ | int |
| `SubjectId_level3` | **知识点(level 3)** | ✅ | string |
| `answer_timestamp` | 回答时间戳 | ✅ | int |

---

## 5. 特征可用性矩阵

| 数据集 | user_id | problem_id | skill_id | correct | timestamp | response_time |
|--------|:-------:|:----------:|:--------:|:-------:|:---------:|:------------:|
| **ASSISTments2009** | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ (有ms_first_response和overlap_time，可提取) |
| ASSISTments2012 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ASSISTments2015 | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| ASSISTments2017 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Algebra2005 | ✅ | ✅* | ✅ | ✅ | ✅ | ❌ |
| Bridge2006 | ✅ | ✅* | ✅ | ✅ | ✅ | ❌ |
| EdNet | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NIPS34 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| POJ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |

*需拼接 Problem Name + Step Name

---

## 6. 模型适配建议

### 6.1 需要 timestamp 的模型

| 模型 | 必需特征 | 可用数据集 |
|------|----------|------------|
| UKT | timestamp | ASSIST2012, ASSIST2017, EdNet, NIPS34 |
| AKT (遗忘) | timestamp | 同上 |
| ATKT | timestamp | 同上 |

### 6.2 需要 response_time 的模型

| 模型 | 必需特征 | 可用数据集 |
|------|----------|------------|
| 时间感知模型 | response_time | ASSIST2012, ASSIST2017, EdNet |

### 6.3 完整特征可用性

**完全支持所有模型的数据集**:
- ✅ ASSISTments2017 (最丰富)
- ✅ EdNet

**基本支持 (无时间特征)**:
- ⚠️ ASSISTments2009
- ⚠️ Algebra2005
- ⚠️ Bridge2006

**受限支持**:
- ❌ ASSISTments2015 (无 problem_id, 无时间)
- ❌ POJ (无 problem_id, 无时间)

---

*文档自动生成于 2026-03-18*
