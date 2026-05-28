"""
Medical article metadata extraction for PageIndex structure JSON output.

Enriches document-level fields (author, abstract, keywords, figures/tables, etc.)
and node-level special_number (clinical thresholds, doses, ranges).
"""
import asyncio
import os
import re
from typing import Any

try:
    from .utils import (
        count_tokens,
        extract_json,
        llm_acompletion,
        llm_completion,
        structure_to_list,
        summary_language_instruction,
    )
except ImportError:
    from utils import (
        count_tokens,
        extract_json,
        llm_acompletion,
        llm_completion,
        structure_to_list,
        summary_language_instruction,
    )

# Heuristic: node text likely contains extractable clinical numbers
_MEDICAL_NUMBER_HINT = re.compile(
    r"(\d+[\d.\-/]*\s*(?:"
    r"mmHg|kPa|mg|g|ml|mL|L|mmol|μmol|mol|IU|U|%|次|分|分钟|小时|天|周|月|年|岁|cm|mm|m|kg|μg|ng|pg"
    r")|≥|≤|>|<|～|~|\d+\s*[~～\-–—]\s*\d+)",
    re.IGNORECASE,
)
_NON_CLINICAL_NUMBER_LABEL = re.compile(
    r"(参考文献|页码|页|卷|期|No\.?|Vol\.?|doi|ISSN|PMID|收稿|修回|出版)",
    re.IGNORECASE,
)
_NON_CLINICAL_NUMBER_VALUE = re.compile(
    r"(^\d+\s*[-–—~～]\s*\d+$|^\d+\s*页$|Vol\.?\s*\d+|No\.?\s*\d+|^\d{1,4}:\d{1,4}$)",
    re.IGNORECASE,
)

_FIGURE_REF = re.compile(
    r"(?:图|Figure|Fig\.?)\s*(\d+)",
    re.IGNORECASE,
)
_TABLE_REF = re.compile(
    r"(?:表|Table|Tab\.?)\s*(\d+)",
    re.IGNORECASE,
)
# Line-start caption (real figure/table title), not inline body refs like "见图1"
_FIGURE_CAPTION_LINE = re.compile(
    r"(?:^|\n)\s*(?:图|Figure|Fig\.?)\s*(\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_CAPTION_LINE = re.compile(
    r"(?:^|\n)\s*(?:表|Table|Tab\.?)\s*(\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)

DOC_METADATA_FIELD_ORDER = [
    "doc_name",
    "abstract",
    "keywords",
    "year",
    "month",
    "author",
    "article_type",
    "disease",
    "raw_pdf_path",
    "figures_info",
    "tables_info",
    "structure",
]

NODE_FIELD_ORDER = [
    "title",
    "node_id",
    "start_index",
    "end_index",
    "line_num",
    "special_number",
    "summary",
    "nodes",
]


def compute_raw_source_path(source_path: str, project_root: str | None = None) -> str:
    """Return a stable relative path like documents/pdf/xxx.pdf."""
    if not source_path or not isinstance(source_path, str):
        return ""
    normalized = os.path.abspath(os.path.expanduser(source_path)).replace("\\", "/")
    if project_root:
        root = os.path.abspath(os.path.expanduser(project_root)).replace("\\", "/")
        try:
            rel = os.path.relpath(normalized, root).replace("\\", "/")
            if not rel.startswith(".."):
                return rel
        except ValueError:
            pass
    for marker in ("documents/pdf/", "documents/markdown/"):
        idx = normalized.find(marker)
        if idx >= 0:
            return normalized[idx:]
    return normalized


def _truncate_text(text: str, max_tokens: int, model: str | None) -> str:
    if not text:
        return ""
    if count_tokens(text, model=model) <= max_tokens:
        return text
    # Rough char-based truncation when over token budget
    ratio = max_tokens / max(count_tokens(text, model=model), 1)
    return text[: max(500, int(len(text) * ratio * 0.9))]


def get_header_text_from_pages(page_list: list, max_pages: int = 3) -> str:
    parts = []
    for i, (page_text, _) in enumerate(page_list[:max_pages]):
        parts.append(f"<page_{i + 1}>\n{page_text}\n</page_{i + 1}>")
    return "\n".join(parts)


def get_full_text_from_pages(page_list: list) -> str:
    parts = []
    for i, (page_text, _) in enumerate(page_list):
        parts.append(f"<page_{i + 1}>\n{page_text}\n</page_{i + 1}>")
    return "\n".join(parts)


def _extract_context_snippet(text: str, match_start: int, window: int = 280) -> str:
    start = max(0, match_start - window // 2)
    end = min(len(text), match_start + window // 2)
    snippet = text[start:end].strip()
    return re.sub(r"\s+", " ", snippet)


def _scan_figure_table_refs(page_list: list) -> tuple[list[dict], list[dict]]:
    """Regex scan for figure/table refs; prefer caption lines over inline citations."""
    figures: dict[str, dict] = {}
    tables: dict[str, dict] = {}
    figure_inline: dict[str, dict] = {}
    table_inline: dict[str, dict] = {}

    for page_idx, (page_text, _) in enumerate(page_list):
        page_num = page_idx + 1
        if not page_text:
            continue

        for m in _FIGURE_CAPTION_LINE.finditer(page_text):
            key = f"图{m.group(1)}"
            figures[key] = {
                "id": key,
                "page": page_num,
                "ref_type": "caption",
                "context": _extract_context_snippet(page_text, m.start()),
            }
        for m in _TABLE_CAPTION_LINE.finditer(page_text):
            key = f"表{m.group(1)}"
            tables[key] = {
                "id": key,
                "page": page_num,
                "ref_type": "caption",
                "context": _extract_context_snippet(page_text, m.start()),
            }

        for m in _FIGURE_REF.finditer(page_text):
            key = f"图{m.group(1)}"
            if key not in figures and key not in figure_inline:
                figure_inline[key] = {
                    "id": key,
                    "page": page_num,
                    "ref_type": "inline",
                    "context": _extract_context_snippet(page_text, m.start()),
                }
        for m in _TABLE_REF.finditer(page_text):
            key = f"表{m.group(1)}"
            if key not in tables and key not in table_inline:
                table_inline[key] = {
                    "id": key,
                    "page": page_num,
                    "ref_type": "inline",
                    "context": _extract_context_snippet(page_text, m.start()),
                }

    for key, val in figure_inline.items():
        if key not in figures:
            figures[key] = val
    for key, val in table_inline.items():
        if key not in tables:
            tables[key] = val

    return list(figures.values()), list(tables.values())


def _has_medical_numbers(text: str) -> bool:
    return bool(text and _MEDICAL_NUMBER_HINT.search(text))


def _is_special_number_relevant(label: str, value: str, title: str = "") -> bool:
    """Filter out publication metadata numbers not clinically meaningful."""
    l = (label or "").strip()
    v = (value or "").strip()
    t = (title or "").strip()
    combined = f"{l} {t}"
    if not l or not v:
        return False
    if _NON_CLINICAL_NUMBER_LABEL.search(combined):
        return False
    if _NON_CLINICAL_NUMBER_VALUE.search(v):
        return False
    # Plain bibliographic ranges without unit/context
    if re.match(r"^\d+\s*[-–—~～]\s*\d+$", v) and not re.search(
        r"(mmHg|kPa|mg|g|ml|mL|L|mmol|μmol|mol|IU|U|%|次|分|分钟|小时|天|周|月|年|岁|cm|mm|m|kg|μg|ng|pg)",
        v,
        re.IGNORECASE,
    ):
        return False
    return True


def extract_document_metadata(
    header_text: str,
    doc_name: str,
    model: str | None = None,
    summary_language: str = "auto",
) -> dict[str, Any]:
    """Extract document-level metadata from header/front matter via LLM."""
    lang_instruction = summary_language_instruction(summary_language)
    prompt = f"""You are extracting structured metadata from a Chinese medical article (PDF text).

Document file name: {doc_name}

Extract the following fields from the header/front matter text. Use null for unknown scalar fields and [] for unknown lists.

Return JSON only:
{{
  "author": "作者姓名，多个作者用逗号分隔",
  "year": "发表年份，如 2026",
  "month": "发表月份 1-12，未知为 null",
  "article_type": "文章类型，如：专家共识、诊疗指南、临床路径、专家意见、综述、标准/规范、其他",
  "abstract": "摘要全文；若文中无独立摘要则尽量从开篇概括",
  "keywords": ["关键词1", "关键词2"],
  "disease": ["涉及的主要疾病或综合征"]
}}

{lang_instruction}
Keep extracted text in the same language as the source (usually Chinese).

Header text:
{header_text}
"""
    response = llm_completion(model=model, prompt=prompt)
    data = extract_json(response) or {}
    return {
        "author": data.get("author") or "",
        "year": str(data.get("year") or ""),
        "month": str(data.get("month") or "") if data.get("month") not in (None, "null") else "",
        "article_type": data.get("article_type") or "",
        "abstract": data.get("abstract") or "",
        "keywords": data.get("keywords") if isinstance(data.get("keywords"), list) else [],
        "disease": data.get("disease") if isinstance(data.get("disease"), list) else [],
    }


async def _enrich_figures_tables_with_llm(
    figures: list[dict],
    tables: list[dict],
    model: str | None = None,
    summary_language: str = "auto",
) -> tuple[list[dict], list[dict]]:
    if not figures and not tables:
        return [], []

    lang_instruction = summary_language_instruction(summary_language)
    prompt = f"""Based on the figure/table references and surrounding text context from a medical article,
produce structured summaries. Text extraction only — images/tables themselves were NOT parsed.

For each item fill:
- id: keep original id (e.g. 图1, 表2)
- title: short title if inferable from context, else ""
- description: main clinical/content summary (1-3 sentences)
- page: page number (integer)
- location: human-readable location e.g. "第3页"

Return JSON:
{{
  "figures_info": [{{"id": "...", "title": "...", "description": "...", "page": 1, "location": "第1页"}}],
  "tables_info": [{{"id": "...", "title": "...", "description": "...", "page": 1, "location": "第1页"}}]
}}

{lang_instruction}

Figure references:
{figures}

Table references:
{tables}
"""
    response = await llm_acompletion(model=model, prompt=prompt)
    data = extract_json(response) or {}
    figs = data.get("figures_info") if isinstance(data.get("figures_info"), list) else figures
    tabs = data.get("tables_info") if isinstance(data.get("tables_info"), list) else tables

    # Ensure minimal schema on fallback
    def _normalize(items: list, kind: str) -> list[dict]:
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            page = item.get("page")
            loc = item.get("location") or (f"第{page}页" if page else "")
            out.append({
                "id": item.get("id") or "",
                "title": item.get("title") or "",
                "description": item.get("description") or item.get("context") or "",
                "page": page,
                "location": loc,
            })
        return out

    return _normalize(figs, "figure"), _normalize(tabs, "table")


async def extract_node_special_number(
    node: dict,
    model: str | None = None,
    summary_language: str = "auto",
) -> dict[str, str] | None:
    text = (node.get("text") or node.get("summary") or "").strip()
    if not text or not _has_medical_numbers(text):
        return None

    title = node.get("title") or ""
    lang_instruction = summary_language_instruction(summary_language)
    prompt = f"""从以下医学文本节选中，提取所有具体的临床数值、阈值、范围、剂量、比例、时间等。
以 JSON 对象返回：键为简短中文描述（可含章节语境），值为具体数值或范围字符串。
若无明确数值则返回 {{}}。

示例:
{{"成人人高血压诊断阈值": "≥140/90 mmHg", "推荐降压目标": "<130/80 mmHg", "随访间隔": "3~6个月"}}

{lang_instruction}

章节标题: {title}

正文:
{text}
"""
    response = await llm_acompletion(model=model, prompt=prompt)
    data = extract_json(response)
    if not isinstance(data, dict) or not data:
        return None
    cleaned = {str(k): str(v) for k, v in data.items() if k and v}
    return cleaned or None


async def extract_document_numbers(
    full_text: str,
    model: str | None = None,
    summary_language: str = "auto",
    max_tokens: int = 12000,
) -> list[dict]:
    """Document-level key numeric facts as a list of {label, value, context}."""
    text = _truncate_text(full_text, max_tokens, model)
    lang_instruction = summary_language_instruction(summary_language)
    prompt = f"""从以下医学文献全文中，提取文档级别最重要的具体数值/阈值/范围/剂量（约 5-20 条）。
返回 JSON 数组，每项: {{"label": "描述", "value": "数值或范围", "context": "一句上下文"}}

{lang_instruction}

Document text:
{text}
"""
    response = await llm_acompletion(model=model, prompt=prompt)
    data = extract_json(response)
    if isinstance(data, dict) and "numbers" in data:
        data = data["numbers"]
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("label") and item.get("value"):
            out.append({
                "label": str(item["label"]),
                "value": str(item["value"]),
                "context": str(item.get("context") or ""),
            })
    return out


async def enrich_nodes_with_special_numbers(
    structure: list | dict,
    model: str | None = None,
    summary_language: str = "auto",
) -> None:
    nodes = structure_to_list(structure)
    candidates = [n for n in nodes if _has_medical_numbers(n.get("text") or n.get("summary") or "")]
    if not candidates:
        return

    print(f"Extracting special_number for {len(candidates)} nodes...")
    tasks = [
        extract_node_special_number(node, model=model, summary_language=summary_language)
        for node in candidates
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for node, result in zip(candidates, results):
        if isinstance(result, Exception) or not result:
            continue
        filtered = {
            str(k): str(v)
            for k, v in result.items()
            if _is_special_number_relevant(str(k), str(v), node.get("title", ""))
        }
        if filtered:
            node["special_number"] = filtered


def _merge_doc_numbers(
    doc_numbers: list[dict],
    structure: list | dict,
) -> list[dict]:
    """Merge document-level numbers with node special_number entries (dedupe by label)."""
    seen = set()
    merged: list[dict] = []
    for item in doc_numbers:
        label = item.get("label", "")
        if label and label not in seen:
            seen.add(label)
            merged.append(item)
    for node in structure_to_list(structure):
        sn = node.get("special_number")
        if not isinstance(sn, dict):
            continue
        for label, value in sn.items():
            if label not in seen:
                seen.add(label)
                merged.append({"label": label, "value": value, "context": node.get("title") or ""})
    return merged


def format_medical_result(result: dict) -> dict:
    """Reorder top-level and node fields to match medical JSON schema."""
    structure = result.get("structure", [])
    node_order = NODE_FIELD_ORDER if _uses_line_num(structure) else [
        k for k in NODE_FIELD_ORDER if k != "line_num"
    ]

    ordered = {}
    for key in DOC_METADATA_FIELD_ORDER:
        if key in result:
            ordered[key] = result[key]
    for key, val in result.items():
        if key not in ordered:
            ordered[key] = val

    if "structure" in ordered:
        ordered["structure"] = _format_nodes(ordered["structure"], node_order)
    return ordered


def _uses_line_num(structure) -> bool:
    for node in structure_to_list(structure):
        if node.get("line_num") is not None:
            return True
    return False


def _format_nodes(structure, order: list[str]):
    if isinstance(structure, dict):
        if structure.get("nodes"):
            structure["nodes"] = _format_nodes(structure["nodes"], order)
        else:
            structure.pop("nodes", None)
        if not structure.get("special_number"):
            structure.pop("special_number", None)
        return {k: structure[k] for k in order if k in structure}
    if isinstance(structure, list):
        return [_format_nodes(item, order) for item in structure]
    return structure


def _scan_figure_table_refs_from_text(text: str) -> tuple[list[dict], list[dict]]:
    """Scan plain/markdown text for figure/table refs; use line number as location."""
    figures: dict[str, dict] = {}
    tables: dict[str, dict] = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        for m in _FIGURE_REF.finditer(line):
            fid = m.group(1)
            key = f"图{fid}"
            if key not in figures:
                figures[key] = {
                    "id": key,
                    "page": line_no,
                    "context": _extract_context_snippet(line, m.start()),
                }
        for m in _TABLE_REF.finditer(line):
            tid = m.group(1)
            key = f"表{tid}"
            if key not in tables:
                tables[key] = {
                    "id": key,
                    "page": line_no,
                    "context": _extract_context_snippet(line, m.start()),
                }
    return list(figures.values()), list(tables.values())


def get_header_text_from_markdown(markdown_content: str, max_lines: int = 80) -> str:
    lines = markdown_content.splitlines()[:max_lines]
    return "\n".join(lines)


async def enrich_with_medical_metadata(
    result: dict,
    *,
    page_list: list | None = None,
    full_text: str | None = None,
    source_path: str | None = None,
    project_root: str | None = None,
    model: str | None = None,
    summary_language: str = "auto",
    metadata_max_pages: int = 3,
) -> dict:
    """
    Enrich a PageIndex result dict with medical article metadata.

    Requires node text to be present on structure nodes when extracting special_number.
    """
    doc_name = result.get("doc_name", "")
    structure = result.get("structure", [])

    if page_list:
        header_text = get_header_text_from_pages(page_list, max_pages=metadata_max_pages)
        if not full_text:
            full_text = get_full_text_from_pages(page_list)
        figure_refs, table_refs = _scan_figure_table_refs(page_list)
    else:
        header_text = _truncate_text(full_text or "", 8000, model)
        figure_refs, table_refs = _scan_figure_table_refs_from_text(full_text or "")

    print("Extracting medical document metadata...")
    meta = extract_document_metadata(
        header_text, doc_name, model=model, summary_language=summary_language
    )

    raw_path = compute_raw_source_path(source_path or "", project_root=project_root)
    print(f"Found {len(figure_refs)} figure refs, {len(table_refs)} table refs")
    figures_info, tables_info = await _enrich_figures_tables_with_llm(
        figure_refs, table_refs, model=model, summary_language=summary_language
    )
    # Markdown uses line-based location labels
    if not page_list:
        for item in figures_info + tables_info:
            ln = item.get("page")
            if ln and not item.get("location"):
                item["location"] = f"第{ln}行"

    await enrich_nodes_with_special_numbers(
        structure, model=model, summary_language=summary_language
    )

    enriched = {
        **result,
        **meta,
        "raw_pdf_path": raw_path,
        "figures_info": figures_info,
        "tables_info": tables_info,
        "structure": structure,
    }
    return format_medical_result(enriched)
