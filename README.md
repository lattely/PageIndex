# 基于PageIndex的医学文献结构化与元数据抽取

<p align="center"><b>面向医学共识 / 指南&nbsp; ◦ &nbsp;文档结构树 + 医学元数据&nbsp; ◦ &nbsp;服务下游 RAG 与知识库构建</b></p>
  

## 📑 本项目做什么？

在官方 PageIndex 的基础上，本项目针对 **中文医学文献（专家共识、诊疗指南等）** 做了定制化扩展，用于：

- 从 PDF / Markdown 中构建 **章节树结构（PageIndex tree）**；
- 在此基础上，自动抽取 **医学文献元数据和关键信息**，生成结构化 JSON，方便后续：
  - 构建医学知识图谱 / 元数据库；
  - 为 RAG、检索问答、标注系统提供高质量结构化输入；
  - 做批量统计（如疾病分布、推荐等级、数值阈值等）。

核心输出为每篇文献一个独立目录：

- `results/<文档名>/structure.json`
- `results/<文档名>/assets/figures.json`（图元数据）
- `results/<文档名>/assets/tables.json`（表元数据）
- `results/<文档名>/assets/figures/*`（图片）
- `results/<文档名>/assets/tables/*`（表格 `csv/md/png`）

---

## 🎯 生成的 JSON 结构（当前版本）

每篇文献会在 `results/<文档名>/structure.json` 生成结构树与路径索引（简化示例）：

```jsonc
{
  "doc_name": "中主动脉综合征致高血压诊断与治疗多学科专家共识(2026).pdf",
  "abstract": "...摘要全文或开篇概括...",
  "keywords": ["高血压", "中主动脉综合征", "..."],
  "year": "2026",
  "month": "4",
  "author": "XXX, YYY, ...",
  "article_type": "专家共识",        // 也可能是：诊疗指南、临床路径、综述等
  "disease": ["高血压", "中主动脉综合征"],
  "raw_pdf_path": "documents/pdf/中主动脉综合征致高血压诊断与治疗多学科专家共识(2026).pdf",
  "raw_md_path": "documents/markdown/中主动脉综合征致高血压诊断与治疗多学科专家共识(2026).md",
  "figures_info_path": "results/中主动脉综合征致高血压诊断与治疗多学科专家共识(2026)/assets/figures.json",
  "tables_info_path": "results/中主动脉综合征致高血压诊断与治疗多学科专家共识(2026)/assets/tables.json",
  "structure": [
    {
      "title": "前言",
      "node_id": "0000",
      "start_index": 1,
      "end_index": 2,
      "summary": "该节主要介绍...",
      "special_number": {
        "高血压患病率": "约 27.9%",
        "控制率": "约 16.8%"
      },
      "nodes": [ ... 子章节 ... ]
    }
  ]
}
```

`figures.json`（简化）示例：

```jsonc
{
  "figures_info": [
    {
      "id": "图 1-1",
      "title": "图 1 中主动脉综合征解剖学分型 (1) 肾动脉上",
      "main_figure_no": 1,
      "sub_figure_no": 1,
      "sub_figure_title": "肾动脉上",
      "location": "第177行",
      "line_no": 177,
      "asset_status": "success",
      "image_path": "results/.../assets/figures/figure_1_中主动脉综合征解剖学分型_1_肾动脉上.png",
      "main_figure_detected_count": 3,
      "medical_meaning": "...",
      "keywords": ["..."],
      "possible_entities": ["..."]
    }
  ]
}
```

`tables.json`（简化）示例：

```jsonc
{
  "tables_info": [
    {
      "id": "表1",
      "title": "本共识推荐强度的分级",
      "location": "第220行",
      "asset_status": "success",
      "csv_path": "results/.../assets/tables/table_1_本共识推荐强度的分级.csv",
      "md_path": "results/.../assets/tables/table_1_本共识推荐强度的分级.md",
      "medical_meaning": "...",
      "keywords": ["..."],
      "possible_entities": ["..."]
    }
  ]
}
```

其中：

- **文档级字段**（在 `doc_name` 和 `structure` 之间）由 LLM 结合文档内容抽取；
- **轻量化原则**：默认不输出节点 `text`，仅输出 `summary`；
- **`special_number`**：仅在节点出现明确数值/阈值/剂量等时出现；
- **图表信息**单独存储到 `figures.json` / `tables.json`，`structure.json` 仅保留路径引用；
- **图表提取**基于 MinerU 产出的 Markdown（含 `![]()`、`<img>`、`<table>`），表格会导出 `csv + md`。

更完整的字段示例可参考仓库内的 `documents/template.json`。

---

## ⚙️ 使用方法

### 1. 安装依赖

```bash
pip3 install --upgrade -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`，例如：

```bash
DEEPSEEK_API_KEY=your_deepseek_key_here
MINERU_API_KEY=your_mineru_key_here
```

可选环境变量：

```bash
MINERU_API_BASE=https://mineru.net/api/v4
MINERU_MODEL_VERSION=vlm
USE_LLM_FIGURE_VALIDATOR=0   # 1/true/yes 开启图号歧义时的LLM校验
```

默认 LLM 为 `deepseek/deepseek-chat`（见 `pageindex/config.yaml`）。图表提取以 MinerU Markdown 资产导出为主，不再走旧的 LLM 裁剪流程。

### 3. 对单篇 PDF 生成结构化 JSON（含医学元数据）

```bash
python3 run_pageindex.py --pdf_path "./documents/pdf/中主动脉综合征致高血压诊断与治疗多学科专家共识(2026).pdf"
```

运行完成后目录示例：

```text
results/
  中主动脉综合征致高血压诊断与治疗多学科专家共识(2026)/
    structure.json
    assets/
      figures.json
      tables.json
      figures/
        figure_1_中主动脉综合征解剖学分型_1_肾动脉上.png
      tables/
        table_1_本共识推荐强度的分级.csv
        table_1_本共识推荐强度的分级.md
```

**可选参数（与医学抽取相关）：**

```bash
--model                    使用的 LLM 模型（默认见 pageindex/config.yaml）
--toc-check-pages          检测目录的最大页数（默认 20）
--max-pages-per-node       单节点最大页数（默认 10）
--max-tokens-per-node      单节点最大 token 数（默认 20000）
--if-add-node-id           是否给节点添加 node_id（yes/no，默认 yes）
--if-add-node-summary      是否生成节点摘要（yes/no，默认 yes）
--if-add-doc-description   是否生成整篇文档描述（yes/no，默认 no）
--if-add-node-text         是否在节点中保留原文 text（yes/no，默认 no）
--if-add-medical-metadata  是否启用医学元数据抽取（yes/no，默认 yes）
```

### 4. Markdown 文档支持

如果你已经有结构良好的 Markdown（如某些期刊提供的 HTML/Markdown 版本），可以直接对 `.md` 跑同样的流程：

```bash
python3 run_pageindex.py --md_path "./documents/markdown/中主动脉综合征致高血压诊断与治疗多学科专家共识(2026).md"
```

- 标题层级通过 `# / ## / ###` 判定；
- `structure` 中用 `line_num` 表示位置；
- 如果是 `--md_path` 直接处理，仅输出结构树（不会自动跑 MinerU 资产导出）；
- 当 `--pdf_path` 且 `documents/markdown/<文档名>.md` 已存在时，会优先复用现有 markdown，缺失资源时才补拉 MinerU zip。

---

## 🧩 图表命名与提取规则（当前实现）

- 图文件：`figure_{主图号}_{主标题}_{子图号}_{子标题}.png`
- 表文件：`table_{表号}_{中文标题}.csv/.md`
- 多子图按 `图 N-1 / 图 N-2 ...` 组织；若标题中存在 `A/B/C/D` 或 `(1)/(2)` 会优先使用
- Markdown 表格与 HTML `<table>...</table>` 都会解析并导出为结构化表格文件
- 每条图/表记录会附带医学语义增强字段（如 `medical_meaning`、`keywords`、`possible_entities`）

<!--
# ☁️ Improved Tree Generation with PageIndex OCR

This repo is designed for generating PageIndex tree structure for simple PDFs, but many real-world use cases involve complex PDFs that are hard to parse by classic Python tools. However, extracting high-quality text from PDF documents remains a non-trivial challenge. Most OCR tools only extract page-level content, losing the broader document context and hierarchy.

To address this, we introduced PageIndex OCR — the first long-context OCR model designed to preserve the global structure of documents. PageIndex OCR significantly outperforms other leading OCR tools, such as those from Mistral and Contextual AI, in recognizing true hierarchy and semantic relationships across document pages.

- Experience next-level OCR quality with PageIndex OCR at our [Dashboard](https://dash.pageindex.ai/).
- Integrate PageIndex OCR seamlessly into your stack via our [API](https://docs.pageindex.ai/quickstart).

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb35d8ae-865c-4e60-a33b-ebbd00c41732" width="80%">
</p>
-->

---

---

## 🧭 典型应用场景

- 批量解析 **高血压、心血管、内分泌等领域的专家共识 / 指南**，统一输出结构化 JSON；
- 为自建 **医学知识库 / RAG 系统** 提供“文献结构 + 元数据 + 数值阈值”输入；
- 结合下游工具，将 `special_number` 与 EHR、指标库对齐，做自动质控或提醒。

本仓库主要关注 **本地/自托管场景**，不依赖官方云端 API，可完全在内网环境中运行（前提是能访问所配置的 LLM 服务）。
