"""
Optional LLM fallback for ambiguous figure-to-main-number assignment.
Enabled via USE_LLM_FIGURE_VALIDATOR=1 (default off).
"""

import os
import re
from typing import Any

from .utils import extract_json, llm_completion


def is_llm_figure_validator_enabled() -> bool:
    return os.getenv("USE_LLM_FIGURE_VALIDATOR", "0").strip().lower() in ("1", "true", "yes")


def _infer_main_num_for_line(
    line_no: int,
    md_lines: list[str],
    parsed_titles: list[dict],
    search_radius: int = 40,
) -> tuple[int | None, str, int, float]:
    """
    Find nearest 图N caption before/after line_no.
    Returns (main_num, title_tail, caption_line, distance).
    """
    best_num = None
    best_title = ""
    best_line = 0
    best_dist = 10**9
    pat = re.compile(r"^\s*图\s*([0-9]+)\s*(.*)$")
    for i, line in enumerate(md_lines, start=1):
        if abs(i - line_no) > search_radius:
            continue
        m = pat.match((line or "").strip())
        if not m:
            continue
        num = int(m.group(1))
        raw = (m.group(2) or "").strip(" ：:.-")
        title_tail = raw
        if len(title_tail) > 120:
            title_tail = title_tail[:120]
        dist = abs(i - line_no)
        if dist < best_dist:
            best_dist = dist
            best_num = num
            best_title = title_tail
            best_line = i
    if best_num is not None:
        return best_num, best_title, best_line, best_dist
    return None, "", 0, 10**9


def _llm_resolve_figure_assignment(
    line_no: int,
    md_lines: list[str],
    candidates: list[dict],
    model: str | None,
) -> dict[str, Any]:
    """
    Ask LLM to assign main figure number for ambiguous image blocks.
    Returns dict with keys: main_num, sub_idx, title, confidence, reason.
    """
    start = max(1, line_no - 15)
    end = min(len(md_lines), line_no + 15)
    context_lines = []
    for i in range(start, end + 1):
        context_lines.append(f"{i}: {md_lines[i-1][:500]}")
    candidates_text = "\n".join(
        [
            f"- line {c.get('line_no')}, src={c.get('src')}, hint={c.get('sub_idx_hint')}"
            for c in candidates
        ]
    )
    prompt = f"""你是医学文献结构化助手。请根据 markdown 上下文，为某张图片确定它属于哪个主图编号。

上下文行（节选）:
{chr(10).join(context_lines)}

候选图片:
{candidates_text}

请返回 JSON（只返回 JSON）:
{{
  "main_num": 1,
  "sub_idx": 1,
  "title": "主图标题短描述",
  "confidence": "high|medium|low",
  "reason": "简短原因"
}}

规则:
- 优先依据最近的 `图 N ...` 标题行判断
- 若一行有多个候选图，结合 caption 与上下文语义选择
- sub_idx 可为 null（单图）或 1/2/3...
"""
    res = llm_completion(model=model, prompt=prompt)
    data = extract_json(res) or {}
    if not isinstance(data, dict):
        return {}
    return data


def resolve_figure_assignment(
    line_no: int,
    md_lines: list[str],
    parsed_titles: list[dict],
    candidates: list[dict],
    model: str | None,
) -> tuple[int | None, str, int | None, bool, str]:
    """
    Resolve main figure number with rules first, optional LLM on ambiguity.
    Returns: main_num, title, sub_idx, needs_llm, reason
    """
    # Rule: nearest caption line
    main_num, title, cap_line, dist = _infer_main_num_for_line(line_no, md_lines, parsed_titles)
    needs_llm = False
    reason = "nearest_caption"

    if main_num is None:
        # Try forward search for any 图 line in radius
        for i in range(line_no, min(len(md_lines), line_no + 25)):
            m = re.match(r"^\s*图\s*([0-9]+)\s*", (md_lines[i - 1] if i > 0 else ""))
            if m:
                n = int(m.group(1))
                t = (m.group(2) or "").strip(" ：:.-")[:80]
                main_num, title, cap_line, dist = n, t, i
                reason = "forward_caption_search"
                break

    # Ambiguity: multiple distinct main numbers near this line
    nearby_nums = set()
    for c in candidates:
        for i in range(max(1, line_no - 20), min(len(md_lines), line_no + 20)):
            m = re.match(r"^\s*图\s*([0-9]+)", (md_lines[i - 1] if i > 0 else ""))
            if m:
                nearby_nums.add(int(m.group(1)))

    if main_num is None and len(nearby_nums) > 1:
        needs_llm = True
        reason = f"multiple_main_nums_{sorted(nearby_nums)}"
    elif main_num is None:
        needs_llm = True
        reason = "no_caption_found"

    sub_idx = None
    if needs_llm and is_llm_figure_validator_enabled() and model:
        llm_data = _llm_resolve_figure_assignment(line_no, md_lines, candidates, model)
        try:
            main_num = int(llm_data.get("main_num") or main_num or 0)
            sub_idx = llm_data.get("sub_idx")
            if sub_idx is not None:
                try:
                    sub_idx = int(sub_idx)
                except (TypeError, ValueError):
                    sub_idx = None
            title = str(llm_data.get("title") or title)
            reason = f"llm:{llm_data.get('reason','')}"
        except (TypeError, ValueError, KeyError):
            reason = "llm_parse_failed"

    if not title and main_num:
        title = f"图{main_num}"

    return main_num, title, sub_idx, needs_llm, reason
