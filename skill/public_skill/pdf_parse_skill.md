# Role: 通用论文 PDF 解析助手
# Task: 在论文解析前，将 PDF 稳定转换为可读文本并输出元信息。

请在每次论文内容分析前，先执行本 skill。

## 1. 输入
* 必填：`pdf_path`（目标论文 PDF 路径）
* 可选：`out_path`（输出 txt 路径）

## 2. 执行规范
* 优先调用公共脚本：`./scripts/extract_pdf_text.py`
* 推荐命令：
  * `python scripts/extract_pdf_text.py --pdf "<pdf_path>"`
  * 如需指定输出：`python scripts/extract_pdf_text.py --pdf "<pdf_path>" --out "<out_path>"`
* 如果环境缺少依赖，先安装：`pip install pypdf`

## 3. 输出要求
* 产出 UTF-8 编码文本，默认与 PDF 同目录同名 `.txt`。
* 文本中保留分页标记，格式：`===== PAGE N =====`。
* 返回以下结果供后续 skill 使用：
  * `out_path`
  * `pages`
  * `chars`

## 4. 约束
* 本 skill 只负责“解析 PDF -> 文本”，不做论文总结。
* 若解析后字符数为 0 或异常中断，必须显式报错并停止后续分析。
