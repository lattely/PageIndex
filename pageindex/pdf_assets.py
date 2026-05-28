"""
Export figure/table assets from PDF with LLM-guided full-region cropping.

Primary strategy:
1. Refine page number via text match
2. Ask LLM for a FULL bounding box (not caption-only): diagram/table body + labels + caption + notes
3. Heuristic expansion around caption block as safety net
4. PyMuPDF clip + optional table structured export
"""
import csv
import os
import re

import pymupdf

try:
    from .utils import extract_json, llm_completion
except ImportError:
    from utils import extract_json, llm_completion

# DPI for final tight crop (higher than coarse pass for clarity)
REFINE_CROP_DPI = 250
# Minimum LLM quality score (0~1) to mark asset as success
CROP_ACCURACY_THRESHOLD = 0.75
# Table crops tolerate small adjacent text blocks more often
TABLE_CROP_ACCURACY_THRESHOLD = 0.5
# Padding around figure diagram / caption after anchor merge
FIGURE_EDGE_PAD = 18


# ── paths & ids ─────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rel(path: str, project_root: str) -> str:
    return os.path.relpath(path, project_root).replace("\\", "/")


def _safe_id(prefix: str, raw: str, idx: int) -> str:
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return f"{prefix}_{digits if digits else str(idx + 1)}"


def _extract_numeric_id(raw_id: str) -> str:
    return "".join(ch for ch in str(raw_id or "") if ch.isdigit())


# ── geometry ──────────────────────────────────────────────────────────────────

def _export_clip(page, rect, out_path: str, dpi: int = 200) -> None:
    pix = page.get_pixmap(clip=rect, dpi=dpi)
    pix.save(out_path)


def _rect_from_bbox(bbox) -> pymupdf.Rect | None:
    if not bbox or len(bbox) < 4:
        return None
    try:
        return pymupdf.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except Exception:
        return None


def _rect_contains(outer: pymupdf.Rect, inner: pymupdf.Rect) -> pymupdf.Rect:
    """Clamp inner to lie inside outer."""
    return pymupdf.Rect(
        max(outer.x0, inner.x0),
        max(outer.y0, inner.y0),
        min(outer.x1, inner.x1),
        min(outer.y1, inner.y1),
    )


def _clamp_rect(rect: pymupdf.Rect, page_rect: pymupdf.Rect, pad: float = 4) -> pymupdf.Rect:
    return pymupdf.Rect(
        max(page_rect.x0, rect.x0 - pad),
        max(page_rect.y0, rect.y0 - pad),
        min(page_rect.x1, rect.x1 + pad),
        min(page_rect.y1, rect.y1 + pad),
    )


def _union_rects(rects: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    valid = [r for r in rects if r is not None]
    if not valid:
        return None
    x0 = min(r.x0 for r in valid)
    y0 = min(r.y0 for r in valid)
    x1 = max(r.x1 for r in valid)
    y1 = max(r.y1 for r in valid)
    return pymupdf.Rect(x0, y0, x1, y1)


def _all_image_rects(page) -> list[pymupdf.Rect]:
    d = page.get_text("dict")
    rects = [
        pymupdf.Rect(b["bbox"])
        for b in d.get("blocks", [])
        if b.get("type") == 1 and b.get("bbox")
    ]
    rects.sort(key=lambda r: (r.y0, r.x0))
    return rects


def _all_tables(page) -> list:
    try:
        finder = page.find_tables()
        return list(getattr(finder, "tables", None) or [])
    except Exception:
        return []


# ── page layout for LLM ─────────────────────────────────────────────────────────

def _page_blocks(page) -> list[dict]:
    blocks = page.get_text("blocks") or []
    out = []
    for idx, b in enumerate(blocks):
        if len(b) < 5:
            continue
        txt = (b[4] or "").strip()
        if not txt:
            continue
        out.append(
            {
                "idx": idx,
                "bbox": [round(float(b[0]), 1), round(float(b[1]), 1),
                         round(float(b[2]), 1), round(float(b[3]), 1)],
                "text": txt[:300],
            }
        )
    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out


def _build_page_text_index(doc) -> list[str]:
    return [page.get_text() or "" for page in doc]


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _compact_caption_regex(token_num: str, kind: str) -> re.Pattern:
    """Match 图1 / 表1 even when PDF inserts line breaks between 图 and number."""
    if kind == "figure":
        return re.compile(rf"(?:图|Figure|Fig\.?)\s*{re.escape(token_num)}\b", re.IGNORECASE)
    return re.compile(rf"(?:表|Table|Tab\.?)\s*{re.escape(token_num)}\b", re.IGNORECASE)


def _caption_line_regex(token_num: str, kind: str) -> re.Pattern:
    if kind == "figure":
        return re.compile(
            rf"(?:^|\n)\s*(?:图|Figure|Fig\.?)\s*{re.escape(token_num)}\b",
            re.IGNORECASE | re.MULTILINE,
        )
    return re.compile(
        rf"(?:^|\n)\s*(?:表|Table|Tab\.?)\s*{re.escape(token_num)}\b",
        re.IGNORECASE | re.MULTILINE,
    )


def _mention_regex(token_num: str, kind: str) -> re.Pattern:
    if kind == "figure":
        return re.compile(
            rf"(?:图|Figure|Fig\.?)\s*{re.escape(token_num)}\b",
            re.IGNORECASE,
        )
    return re.compile(
        rf"(?:表|Table|Tab\.?)\s*{re.escape(token_num)}\b",
        re.IGNORECASE,
    )


def _page_has_kind_caption(page, token_num: str, kind: str) -> bool:
    page_text = _normalize_ws(page.get_text() or "")
    if _compact_caption_regex(token_num, kind).search(page_text):
        return True
    if _caption_line_regex(token_num, kind).search(page.get_text() or ""):
        return True
    return bool(_find_caption_blocks(_page_blocks(page), token_num, kind))


def _find_best_page_for_item(item: dict, page_texts: list[str], kind: str = "figure") -> int | None:
    """Score pages: caption line >> inline mention. Figure/table patterns are never mixed."""
    raw_page = int(item.get("page") or 0)
    token_num = _extract_numeric_id(item.get("id", ""))
    title = str(item.get("title") or "").strip()
    if not token_num:
        return raw_page if raw_page > 0 else None

    mention_pat = _mention_regex(token_num, kind)
    caption_pat = _caption_line_regex(token_num, kind)

    candidate_pages = list(range(1, len(page_texts) + 1))
    if raw_page > 0:
        window = [p for p in range(raw_page - 2, raw_page + 3) if 1 <= p <= len(page_texts)]
        candidate_pages = window + [p for p in candidate_pages if p not in window]

    scored = []
    for p in candidate_pages:
        txt = page_texts[p - 1]
        if not mention_pat.search(txt):
            continue
        score = 1
        if title and title in txt:
            score += 2
        if caption_pat.search(txt):
            score += 10
        scored.append((score, p))

    if not scored:
        for p, txt in enumerate(page_texts, start=1):
            if mention_pat.search(txt):
                score = 10 if caption_pat.search(txt) else 1
                scored.append((score, p))

    if not scored:
        return raw_page if raw_page > 0 else None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def _llm_find_figure_page(
    item: dict,
    doc,
    page_texts: list[str],
    model=None,
) -> tuple[int | None, dict]:
    """
    LLM picks the page where the target figure's caption + diagram actually appear.
    """
    token_num = _extract_numeric_id(item.get("id", ""))
    if not token_num:
        return None, {}

    id_text = str(item.get("id") or "")
    title = str(item.get("title") or "")
    desc = str(item.get("description") or "")

    pages_evidence = []
    for pno, txt in enumerate(page_texts, start=1):
        if not _mention_regex(token_num, "figure").search(txt):
            continue
        blocks = _page_blocks(doc[pno - 1])
        cap_blocks = _find_caption_blocks(blocks, token_num, "figure")
        cap_lines = [_normalize_ws(b.get("text", ""))[:120] for b in cap_blocks[:3]]
        table_cap = _find_caption_blocks(blocks, token_num, "table")
        pages_evidence.append({
            "page": pno,
            "has_figure_caption": _page_has_kind_caption(doc[pno - 1], token_num, "figure"),
            "has_table_caption_same_num": _page_has_kind_caption(doc[pno - 1], token_num, "table"),
            "figure_caption_samples": cap_lines,
            "has_embedded_images": len(_all_image_rects(doc[pno - 1])) > 0,
            "text_snippet": txt[:800],
        })

    if not pages_evidence:
        return None, {"method": "llm_find_figure_page", "reason": "no_pages_with_figure_mention"}

    prompt = f"""你是医学 PDF 图表定位专家。请找出目标**图片/示意图**（不是表格）真正所在的页码。

目标图片：
- id: {id_text}
- title: {title}
- description: {desc}

候选页面证据（JSON 列表）:
{pages_evidence}

判断规则：
1. 必须选择存在**图题行**的页面（figure_caption_samples 含「图 {token_num}」或 Figure {token_num}，作为独立图题而非正文括号引用）
2. **禁止**选择仅有正文「（图{token_num}）」引用、但图题在别页的页面
3. 若某页 has_table_caption_same_num=true 且 has_figure_caption=false，那是表{token_num}而非图{token_num}，不可选
4. 优先有示意图/流程图内容（has_embedded_images 或图题下方/上方有流程图文字）的页面

返回 JSON：
{{
  "page": <页码整数>,
  "confidence": "high|medium|low",
  "reason": "<为何选该页>"
}}
只返回 JSON。
"""
    try:
        res = llm_completion(model=model, prompt=prompt)
        data = extract_json(res) or {}
        page = int(data.get("page") or 0)
        if 1 <= page <= len(page_texts):
            return page, {
                "method": "llm_find_figure_page",
                "confidence": data.get("confidence") or "medium",
                "reason": data.get("reason") or "",
            }
    except Exception as e:
        return None, {"method": "llm_find_figure_page", "reason": f"error:{e}"}

    return None, {"method": "llm_find_figure_page", "reason": "llm_no_page"}


def _resolve_figure_page(
    item: dict,
    doc,
    page_texts: list[str],
    model=None,
) -> tuple[int | None, dict]:
    """Heuristic caption page first; LLM disambiguation when needed."""
    token_num = _extract_numeric_id(item.get("id", ""))
    meta: dict = {}

    best = _find_best_page_for_item(item, page_texts, kind="figure")
    if best and _page_has_kind_caption(doc[best - 1], token_num, "figure"):
        meta["page_source"] = "caption_heuristic"
        return best, meta

    caption_pages = [
        p for p in range(1, len(page_texts) + 1)
        if _caption_line_regex(token_num, "figure").search(page_texts[p - 1])
    ]
    if len(caption_pages) == 1:
        meta["page_source"] = "caption_unique"
        return caption_pages[0], meta

    llm_page, llm_meta = _llm_find_figure_page(item, doc, page_texts, model=model)
    meta.update(llm_meta)
    if llm_page:
        meta["page_source"] = "llm_find_figure_page"
        return llm_page, meta

    if caption_pages:
        meta["page_source"] = "caption_first_of_many"
        return caption_pages[0], meta

    if best:
        meta["page_source"] = "mention_heuristic_fallback"
        return best, meta

    return None, meta


def _caption_pattern(token_num: str, kind: str) -> re.Pattern:
    if kind == "figure":
        return re.compile(
            rf"^\s*(图\s*{re.escape(token_num)}|Figure\s*{re.escape(token_num)}|Fig\.?\s*{re.escape(token_num)})\b",
            re.IGNORECASE,
        )
    return re.compile(
        rf"^\s*(表\s*{re.escape(token_num)}|Table\s*{re.escape(token_num)}|Tab\.?\s*{re.escape(token_num)})\b",
        re.IGNORECASE,
    )


def _blocks_are_vertically_close(b1: dict, b2: dict, max_gap: float = 45) -> bool:
    return abs(b2["bbox"][1] - b1["bbox"][3]) < max_gap or abs(b1["bbox"][1] - b2["bbox"][3]) < max_gap


def _find_caption_blocks(blocks: list[dict], token_num: str, kind: str) -> list[dict]:
    """Find caption text blocks; handles PDFs that split 「图」 and 「1 …」 into separate blocks."""
    pat = _caption_pattern(token_num, kind)
    hits = [b for b in blocks if pat.search(_normalize_ws(b.get("text", "")))]
    if hits:
        return hits

    compact = _compact_caption_regex(token_num, kind)
    hits = [b for b in blocks if compact.search(_normalize_ws(b.get("text", "")))]
    if hits:
        return hits

    label = "图" if kind == "figure" else "表"
    label_only = re.compile(rf"^(?:{label}|Figure|Fig\.?|Table|Tab\.?)\.?$", re.IGNORECASE)
    num_lead = re.compile(rf"^{re.escape(token_num)}(?:\s|[^\d]|$)", re.IGNORECASE)

    def _pair_caption(b_label: dict, b_num: dict) -> list[dict]:
        return sorted([b_label, b_num], key=lambda x: (x["bbox"][1], x["bbox"][0]))

    for b in blocks:
        t = _normalize_ws(b.get("text", ""))
        if label_only.match(t):
            for ob in blocks:
                if ob is b:
                    continue
                ot = _normalize_ws(ob.get("text", ""))
                if num_lead.match(ot) and (
                    _blocks_are_vertically_close(b, ob)
                    or abs(b["bbox"][0] - ob["bbox"][2]) < 120
                ):
                    return _pair_caption(b, ob)

    for b in blocks:
        ot = _normalize_ws(b.get("text", ""))
        if num_lead.match(ot):
            for ob in blocks:
                if ob is b:
                    continue
                lt = _normalize_ws(ob.get("text", ""))
                if label_only.match(lt) and (
                    _blocks_are_vertically_close(b, ob)
                    or abs(ob["bbox"][0] - b["bbox"][2]) < 120
                ):
                    return _pair_caption(ob, b)

    for i in range(len(blocks)):
        for j in range(i, min(i + 6, len(blocks))):
            group = blocks[i : j + 1]
            combined = _normalize_ws(" ".join(x.get("text", "") for x in group))
            if not compact.search(combined):
                continue
            y0 = min(x["bbox"][1] for x in group)
            y1 = max(x["bbox"][3] for x in group)
            if y1 - y0 < 80:
                return group

    loose = compact
    return [b for b in blocks if loose.search(_normalize_ws(b.get("text", "")))]


def _union_caption_block_rect(blocks: list[dict]) -> pymupdf.Rect | None:
    rects = [_rect_from_bbox(b["bbox"]) for b in blocks]
    rects = [r for r in rects if r is not None]
    return _union_rects(rects)


def _heuristic_full_region(page, blocks: list[dict], token_num: str, kind: str) -> pymupdf.Rect | None:
    """
    Expand from caption block to full figure/table region without LLM.
    - Figure: caption usually at bottom; expand upward to include diagram + sub-labels
    - Table: caption usually at top; expand downward through table body
    """
    pr = page.rect
    captions = _find_caption_blocks(blocks, token_num, kind)
    if not captions:
        return None

    cap_rect = _union_caption_block_rect(captions)
    if cap_rect is None:
        return None

    cap_y0, cap_y1 = cap_rect.y0, cap_rect.y1
    margin_x = pr.width * 0.06

    if kind == "figure":
        image_rects = _all_image_rects(page)
        above_images = [r for r in image_rects if r.y1 <= cap_y1 + 80]

        if above_images:
            parts = list(above_images) + [cap_rect]
            for b in blocks:
                txt = (b.get("text") or "").strip()
                y0, y1 = b["bbox"][1], b["bbox"][3]
                if y0 < cap_y0 - 5 and y1 > cap_y1 + 15:
                    continue
                if above_images[0].y1 < y0 < cap_y1 + 15 and (
                    txt.startswith(("注：", "注:")) or "注:" in txt[:4]
                ):
                    rect = _rect_from_bbox(b["bbox"])
                    if rect:
                        parts.append(rect)
            u = _union_rects(parts)
            if u:
                return pymupdf.Rect(
                    max(pr.x0 + margin_x * 0.5, u.x0 - 12),
                    max(pr.y0, u.y0 - 12),
                    min(pr.x1 - margin_x * 0.5, u.x1 + 12),
                    min(pr.y1, u.y1 + 15),
                )

        diagram_rects: list[pymupdf.Rect] = []
        for b in blocks:
            y0, y1 = b["bbox"][1], b["bbox"][3]
            if y1 > cap_y0 + 25:
                continue
            txt = (b.get("text") or "").strip()
            if _caption_pattern(token_num, "figure").search(txt):
                continue
            if _is_section_body_text(txt) or len(txt) > 95:
                continue
            if not above_images and y0 < cap_y0 - 280 and (len(txt) > 50 or "。" in txt):
                continue
            rect = _rect_from_bbox(b["bbox"])
            if rect:
                diagram_rects.append(rect)

        y_top = cap_y0 - pr.height * 0.55
        if above_images:
            y_top = min(y_top, min(r.y0 for r in above_images) - 15)
        if diagram_rects:
            y_top = min(y_top, min(r.y0 for r in diagram_rects) - 12)

        y_bottom = cap_y1 + 25
        fig_pat = re.compile(rf"Figure\s*{re.escape(token_num)}\b", re.IGNORECASE)
        for b in blocks:
            if fig_pat.search(b.get("text", "")):
                y_bottom = max(y_bottom, b["bbox"][3] + 12)
        next_cap_pat = re.compile(r"^\s*(图\s*\d+|表\s*\d+|Figure\s*\d+|Table\s*\d+)", re.IGNORECASE)
        for b in blocks:
            if b["bbox"][1] <= cap_y1 + 5:
                continue
            txt = b.get("text", "")
            if next_cap_pat.match(txt) and _extract_numeric_id(txt) != token_num:
                y_bottom = min(y_bottom, b["bbox"][1] - 8)
                break
        for b in blocks:
            if cap_y1 < b["bbox"][1] < y_bottom + 5:
                txt = (b.get("text") or "").strip()
                if txt and not next_cap_pat.match(txt):
                    y_bottom = max(y_bottom, b["bbox"][3] + 8)

        region = pymupdf.Rect(
            pr.x0 + margin_x,
            max(pr.y0, y_top),
            pr.x1 - margin_x,
            min(pr.y1, y_bottom),
        )
        if diagram_rects or above_images:
            parts = diagram_rects + above_images + [cap_rect]
            u = _union_rects(parts)
            if u:
                region = pymupdf.Rect(
                    max(pr.x0 + margin_x * 0.5, u.x0 - 12),
                    max(pr.y0, min(y_top, u.y0 - 12)),
                    min(pr.x1 - margin_x * 0.5, u.x1 + 12),
                    min(pr.y1, max(y_bottom, u.y1 + 12)),
                )
        return region

    # table: caption on top, extend down until next major block (图/表/section) or page margin
    next_cap_pat = re.compile(r"^\s*(图\s*\d+|表\s*\d+|Figure\s*\d+|Table\s*\d+)", re.IGNORECASE)
    y_bottom = cap_y1 + pr.height * 0.45
    for b in blocks:
        if b["bbox"][1] <= cap_y1 + 5:
            continue
        txt = b.get("text", "")
        # 截断到下一张图/表或下一节标题，避免两个表拼在一起
        if next_cap_pat.match(txt) and _extract_numeric_id(txt) != token_num:
            y_bottom = min(y_bottom, b["bbox"][1] - 8)
            break
        y_bottom = max(y_bottom, b["bbox"][3] + 8)

    # 只选择离当前表题最近的那一张表作为主表，防止一页上多个表被一起 union
    table_rects = [pymupdf.Rect(t.bbox) for t in _all_tables(page)]
    primary_table: pymupdf.Rect | None = None
    if table_rects:
        # 优先选 y0 紧邻 caption 之下的表；若都在 caption 上方，则选与 caption 垂直距离最近者
        def _table_score(r: pymupdf.Rect) -> tuple[float, float]:
            if r.y0 >= cap_y1 - 6:
                return (r.y0 - cap_y1, r.y0)
            # 表整体略高于 caption（少见），按距 caption 底边的距离排序
            return (cap_y1 - r.y1 + pr.height, r.y0)

        primary_table = sorted(table_rects, key=_table_score)[0]

    relevant_tables = [primary_table] if primary_table is not None else []

    # 文本块仅取从 caption 到 y_bottom 之间的部分，避免把下一张表的正文一起并入
    parts = [
        _rect_from_bbox(b["bbox"])
        for b in blocks
        if cap_y0 - 5 <= b["bbox"][1] <= y_bottom + 5
    ]
    parts = [r for r in parts if r is not None] + relevant_tables
    parts.append(cap_rect)
    if parts:
        u = _union_rects(parts)
        if u:
            return pymupdf.Rect(
                max(pr.x0 + margin_x * 0.5, u.x0 - 10),
                max(pr.y0, u.y0 - 12),
                min(pr.x1 - margin_x * 0.5, u.x1 + 10),
                min(pr.y1, u.y1 + 15),
            )
    return pymupdf.Rect(pr.x0 + margin_x, cap_y0 - 8, pr.x1 - margin_x, min(pr.y1, y_bottom))


def _llm_locate_full_region(
    item: dict,
    page,
    blocks: list[dict],
    model=None,
    kind: str = "figure",
) -> dict:
    """
    Ask LLM for complete crop bbox in PDF points — must include entire visual, not caption only.
    """
    pr = page.rect
    page_w, page_h = pr.width, pr.height
    id_text = str(item.get("id") or "")
    title = str(item.get("title") or "")
    desc = str(item.get("description") or "")

    if kind == "figure":
        kind_zh = "图片/示意图/流程图（非表格）"
        layout_hint = """
【目标类型：图片/流程图 — 不是表格】
- 必须框选：流程图/示意图主体（含所有框、箭头、分支文字）、图下「图 N …」图题（「图」与数字可能在相邻两个文本块中）。
- 严禁框选：表 N 的网格表格、表头+数据行、正文段落；即使与图题在同一页，也不能把表当图。
- 图题常在流程图**下方**（如「图 1 PPH 诊断流程图」）；主体在图题**上方**。
- 可含图下「注：」；坐标为 PDF points，原点在页面左上角。
"""
        type_check = """
5. 返回的 bbox 必须包含图题「图 N」所在区域，且 y 方向覆盖图主体；不得主要是表格式行列文本
6. 若本页仅有表 N 而无独立图 N 图题+流程图，返回 confidence=low 且 bbox 尽量为空
"""
    else:
        kind_zh = "表格（非图片）"
        layout_hint = """
【目标类型：表格 — 不是流程图/示意图】
- 必须框选：表题「表 N …」+ 表头 + 全部数据行 + 表注（如有）。
- 严禁框选：图 N 流程图、示意图、正文段落。
- 表题常在表格**上方**；坐标为 PDF points。
"""
        type_check = """
5. 返回的 bbox 必须覆盖表格网格区域，不得主要是流程图框线
6. 不得将图 N 流程图误识别为本表
"""

    prompt = f"""你是医学 PDF 版面分析专家。请为目标{kind_zh}给出**完整截图区域**的边界框。

{layout_hint}

目标对象：
- id: {id_text}
- title: {title}
- description: {desc}
- 页面尺寸: width={page_w:.1f}, height={page_h:.1f} (PDF points)

当前页文本块（按从上到下排序；注意「图」「1」可能被拆成两个块，需合并理解）:
{blocks}

请根据文本块位置关系，推断该{kind_zh}的**完整可视区域**（含主体+标注+表题/图题），返回 JSON：
{{
  "bbox": [x0, y0, x1, y1],
  "confidence": "high|medium|low",
  "includes": ["diagram_or_table_body", "caption", "sub_labels_or_notes"],
  "reason": "<为何这样框选；明确区分图/表，说明是否包含主体而非仅标题>"
}}

要求：
1. bbox 必须落在页面内：0<=x<=width, 0<=y<=height
2. 宽度应覆盖该{kind_zh}主要内容（通常 x 跨度 > 页面宽度的 40%）
3. 高度必须足够包含完整图或表（禁止只框选一行图题/表题）
4. 只返回 JSON，不要其他文字
{type_check}
"""
    try:
        res = llm_completion(model=model, prompt=prompt)
        data = extract_json(res) or {}
        bbox = data.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            rect = pymupdf.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            rect = _clamp_rect(rect, pr)
            return {
                "rect": rect,
                "confidence": data.get("confidence") or "medium",
                "includes": data.get("includes") or [],
                "reason": data.get("reason") or "",
                "method": "llm_full_region",
            }
    except Exception as e:
        return {"rect": None, "reason": f"llm_error:{e}", "method": "llm_full_region"}

    return {"rect": None, "reason": "llm_no_bbox", "method": "llm_full_region"}


def _blocks_in_rect(blocks: list[dict], rect: pymupdf.Rect) -> list[dict]:
    out = []
    for b in blocks:
        r = _rect_from_bbox(b["bbox"])
        if r and rect.intersects(r):
            out.append(b)
    return out


def _is_section_body_text(text: str) -> bool:
    """Continuous narrative paragraph / section heading — not figure/table content."""
    t = (text or "").strip()
    if not t:
        return False
    if re.match(r"^\d+(\.\d+)+\s", t) and len(t) > 25:
        return True
    if re.match(r"^\d+(\.\d+)+\s*[\u4e00-\u9fff]", t):
        return True
    return len(t) > 120 and not re.match(r"^(注[：:]|图\s*\d|Figure\s*\d|表\s*\d|Table\s*\d)", t, re.I)


def _figure_caption_y_range(blocks: list[dict], token_num: str) -> tuple[float, float] | None:
    """Bottom of last caption line (Chinese + English Figure N)."""
    caps = _find_caption_blocks(blocks, token_num, "figure")
    if not caps:
        return None
    y0 = min(c["bbox"][1] for c in caps)
    y1 = max(c["bbox"][3] for c in caps)
    fig_pat = re.compile(rf"Figure\s*{re.escape(token_num)}\b", re.IGNORECASE)
    for b in blocks:
        if fig_pat.search(b.get("text", "")):
            y0 = min(y0, b["bbox"][1])
            y1 = max(y1, b["bbox"][3])
    return y0, y1


def _figure_content_anchor(
    page,
    blocks: list[dict],
    token_num: str,
    coarse_rect: pymupdf.Rect,
) -> pymupdf.Rect | None:
    """
    Geometry anchor from PDF objects + layout heuristics (not LLM).
    Ensures embedded images, flowchart labels, 注, and full captions are kept.
    """
    cap_range = _figure_caption_y_range(blocks, token_num)
    if not cap_range:
        return None
    cap_y0, cap_y1 = cap_range
    cap_blocks = _find_caption_blocks(blocks, token_num, "figure")
    fig_pat = re.compile(rf"Figure\s*{re.escape(token_num)}\b", re.IGNORECASE)

    parts: list[pymupdf.Rect] = []
    for c in cap_blocks:
        rect = _rect_from_bbox(c["bbox"])
        if rect:
            parts.append(rect)
    for b in blocks:
        txt = b.get("text", "")
        if fig_pat.search(txt):
            rect = _rect_from_bbox(b["bbox"])
            if rect:
                parts.append(rect)
        if txt.strip().startswith(("注：", "注:")) and cap_y0 - 300 <= b["bbox"][1] <= cap_y1 + 8:
            rect = _rect_from_bbox(b["bbox"])
            if rect:
                parts.append(rect)

    images = [r for r in _all_image_rects(page) if r.y1 <= cap_y1 + 45]
    parts.extend(images)

    col_x0, col_x1 = coarse_rect.x0, coarse_rect.x1
    if parts:
        col_x0 = min(r.x0 for r in parts) - 25
        col_x1 = max(r.x1 for r in parts) + 25
    col_x0 = max(coarse_rect.x0, col_x0)
    col_x1 = min(coarse_rect.x1, col_x1)

    y_top_limit = cap_y0 - 20
    if images:
        y_top_limit = min(y_top_limit, min(r.y0 for r in images) - FIGURE_EDGE_PAD)
    else:
        y_top_limit = cap_y0 - 420

    for b in blocks:
        txt = b.get("text", "").strip()
        if not txt or _is_section_body_text(txt):
            continue
        y0, y1, x0, x1 = b["bbox"][1], b["bbox"][3], b["bbox"][0], b["bbox"][2]
        if y1 > cap_y0 + 15 or y0 < y_top_limit:
            continue
        if x1 < col_x0 or x0 > col_x1:
            continue
        if fig_pat.search(txt) or _caption_pattern(token_num, "figure").search(txt):
            continue
        if txt.startswith(("注：", "注:")):
            continue
        if len(txt) > 95 or (txt.count("。") >= 2 and len(txt) > 40):
            continue
        if not images and y0 < cap_y0 - 280 and (len(txt) > 50 or "。" in txt):
            continue
        rect = _rect_from_bbox(b["bbox"])
        if rect:
            parts.append(rect)

    anchor = _union_rects(parts)
    if anchor is None:
        return None

    y_bottom = cap_y1 + FIGURE_EDGE_PAD
    for b in blocks:
        if b["bbox"][1] <= cap_y1 + 2:
            continue
        if _is_section_body_text(b.get("text", "")):
            y_bottom = min(y_bottom, b["bbox"][1] - 6)
            break

    if images:
        y_top = min(r.y0 for r in images) - FIGURE_EDGE_PAD
    else:
        diagram_rects = [r for r in parts if r.y0 > 60 and r.y1 < cap_y0 + 8]
        y_top = (min(r.y0 for r in diagram_rects) if diagram_rects else anchor.y0) - FIGURE_EDGE_PAD

    anchor = pymupdf.Rect(
        max(col_x0, anchor.x0) - FIGURE_EDGE_PAD,
        max(coarse_rect.y0, y_top),
        min(col_x1, anchor.x1) + FIGURE_EDGE_PAD,
        min(anchor.y1, y_bottom),
    )
    anchor = _rect_contains(coarse_rect, anchor)
    return _clamp_rect(anchor, page.rect, pad=FIGURE_EDGE_PAD)


def _finalize_figure_rect(
    coarse_rect: pymupdf.Rect,
    refined_rect: pymupdf.Rect | None,
    anchor_rect: pymupdf.Rect | None,
    page_rect: pymupdf.Rect,
) -> pymupdf.Rect:
    """Merge anchor + refined inside coarse bounds; prefer anchor if refine over-crops."""
    parts: list[pymupdf.Rect] = []
    if anchor_rect:
        parts.append(anchor_rect)
    if refined_rect:
        parts.append(refined_rect)
    if not parts:
        final = coarse_rect
    else:
        final = _union_rects(parts) or coarse_rect
        if refined_rect and anchor_rect and anchor_rect.height > 20:
            if refined_rect.height < anchor_rect.height * 0.88:
                final = pymupdf.Rect(
                    min(refined_rect.x0, anchor_rect.x0),
                    anchor_rect.y0,
                    max(refined_rect.x1, anchor_rect.x1),
                    max(refined_rect.y1, anchor_rect.y1, refined_rect.y1),
                )
        final = _rect_contains(coarse_rect, final)
    return _clamp_rect(final, page_rect, pad=FIGURE_EDGE_PAD)


def _llm_refine_boundaries(
    item: dict,
    page,
    blocks: list[dict],
    coarse_rect: pymupdf.Rect,
    model=None,
    kind: str = "figure",
) -> dict:
    """
    Second pass: within coarse crop, LLM trims top/bottom/left/right to exclude 正文段落.
    Returns refined bbox in page coordinates (must stay inside coarse_rect).
    """
    pr = page.rect
    inner = _blocks_in_rect(blocks, coarse_rect)
    kind_zh = "图片/流程图" if kind == "figure" else "表格"
    id_text = str(item.get("id") or "")
    title = str(item.get("title") or "")
    desc = str(item.get("description") or "")

    coarse = [round(coarse_rect.x0, 1), round(coarse_rect.y0, 1),
              round(coarse_rect.x1, 1), round(coarse_rect.y1, 1)]

    figure_rules = ""
    if kind == "figure":
        figure_rules = """
【图片专用规则 — 极其重要】
- 示意图/流程图主体可能是**矢量图或嵌入图片**，页面上**没有对应文本块**；不得因“上方无文本”而上移裁切掉图主体。
- 必须保留：图主体、图中短标注（如“(1)肾动脉上”）、流程图各框与箭头、图下“注：”行、中文“图 N …”与英文 “Figure N …” **完整两行图题**。
- 仅裁掉：带小节号的连续论述段落（如“1.2.2 病理分型…”）、上一节/下一节正文；**不要把图题、图注、流程图文字当作正文裁掉**。
- bottom 必须 ≥ 英文 Figure 行下边界；top 必须 ≤ 图主体/流程图最顶元素（含少量留白）。
"""

    prompt = f"""你是医学 PDF 版面裁剪专家。已有一个**偏大的初步裁剪框**（coarse_bbox），其中包含了目标{kind_zh}，但也混入了部分正文段落。

你的任务：在 coarse_bbox 内部，给出**紧致边界框**（refined_bbox），要求：
1. **必须完整保留**目标{kind_zh}的全部可视内容（表头+所有数据行 / 嵌入图+流程图+子标注+注+中英文图题）。
2. **必须裁掉**与目标无关的正文段落（连续论述性文字、上一节/下一节正文、页眉页脚）。
3. refined_bbox 必须完全落在 coarse_bbox 内部（不能超出）。
4. 坐标为 PDF points，页面尺寸 width={pr.width:.1f}, height={pr.height:.1f}。
{figure_rules}

目标：
- id: {id_text}
- title: {title}
- description: {desc}

coarse_bbox: {coarse}

coarse 区域内的文本块（bbox=[x0,y0,x1,y1], text=...）:
{inner}

请判断各边界应落在哪里以排除正文：
- top: 从何处开始才是{kind_zh}区域（裁掉上方正文；**勿裁图主体**）
- bottom: 到何处结束（裁掉下方正文；**须包含完整图题/表题**）
- left/right: 栏宽内紧致边界

返回 JSON：
{{
  "refined_bbox": [x0, y0, x1, y1],
  "excluded_regions": ["above_body_text", "below_body_text"],
  "reason": "<如何区分图表区域与正文>"
}}
只返回 JSON。
"""
    try:
        res = llm_completion(model=model, prompt=prompt)
        data = extract_json(res) or {}
        bbox = data.get("refined_bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            refined = pymupdf.Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            refined = _rect_contains(coarse_rect, refined)
            refined = _clamp_rect(refined, pr, pad=2)
            if refined.width > 10 and refined.height > 10:
                return {
                    "rect": refined,
                    "reason": data.get("reason") or "",
                    "excluded_regions": data.get("excluded_regions") or [],
                    "method": "llm_boundary_refine",
                }
    except Exception as e:
        return {"rect": None, "reason": f"refine_error:{e}", "method": "llm_boundary_refine"}

    return {"rect": None, "reason": "refine_no_bbox", "method": "llm_boundary_refine"}


def _crop_contains_figure_caption(page, rect: pymupdf.Rect, token_num: str) -> bool:
    if not token_num:
        return False
    crop_text = page.get_textbox(rect) or ""
    if _caption_line_regex(token_num, "figure").search(crop_text):
        return True
    inner = _blocks_in_rect(_page_blocks(page), rect)
    return bool(_find_caption_blocks(inner, token_num, "figure"))


def _crop_is_table_not_figure(page, rect: pymupdf.Rect, token_num: str) -> bool:
    """True when crop has 表N caption but no 图N caption (common mis-crop)."""
    if not token_num:
        return False
    crop_text = page.get_textbox(rect) or ""
    has_table = bool(_caption_line_regex(token_num, "table").search(crop_text))
    has_figure = _crop_contains_figure_caption(page, rect, token_num)
    return has_table and not has_figure


def _llm_verify_crop_quality(
    item: dict,
    page,
    rect: pymupdf.Rect,
    model=None,
    kind: str = "figure",
) -> dict:
    """
    LLM quality gate (similar spirit to verify_toc accuracy check).
    Uses text extracted from the final crop region.
    """
    crop_text = (page.get_textbox(rect) or "").strip()
    if len(crop_text) > 6000:
        crop_text = crop_text[:6000] + "\n...(truncated)"

    token_num = _extract_numeric_id(item.get("id", ""))
    if kind == "figure" and token_num:
        if _crop_is_table_not_figure(page, rect, token_num):
            return {
                "accuracy": 0.0,
                "passed": False,
                "contains_target": False,
                "body_text_pollution": True,
                "thinking": f"裁切区域为表{token_num}而非图{token_num}（含表题、无图题）",
            }
        if not _crop_contains_figure_caption(page, rect, token_num):
            return {
                "accuracy": 0.0,
                "passed": False,
                "contains_target": False,
                "body_text_pollution": True,
                "thinking": f"裁切区域内未找到图{token_num}图题行，非有效图片截图",
            }

    threshold = CROP_ACCURACY_THRESHOLD if kind == "figure" else TABLE_CROP_ACCURACY_THRESHOLD
    kind_zh = "图片" if kind == "figure" else "表格"
    id_text = str(item.get("id") or "")
    title = str(item.get("title") or "")
    desc = str(item.get("description") or "")

    figure_extra = ""
    if kind == "figure" and token_num:
        figure_extra = f"""
【图片硬性要求】裁切区域必须包含独立图题行「图 {token_num}」或 Figure {token_num}，且主体为示意图/流程图（可含矢量框线文字），不能主要是表{token_num}或正文段落。
"""

    prompt = f"""你是医学文档截图质检员。请判断：从 PDF 裁出的区域文本，是否**主要是**目标{kind_zh}，且**没有明显混入大段无关正文**。
{figure_extra}

目标{kind_zh}:
- id: {id_text}
- title: {title}
- description: {desc}

裁切区域内的提取文本:
---
{crop_text}
---

评判标准：
- accuracy=1.0：几乎全是该{kind_zh}（含表题/图题、表体/图体、图注），无大段正文污染
- accuracy=0.5：约一半正文一半图表
- accuracy=0.0：主要是正文，或几乎不包含目标{kind_zh}
- passed=yes 仅当 accuracy >= {threshold} 且能识别出目标{kind_zh}的核心内容

返回 JSON：
{{
  "accuracy": <0到1的小数>,
  "passed": "yes或no",
  "contains_target": "yes或no",
  "body_text_pollution": "yes或no",
  "thinking": "<简短说明>"
}}
只返回 JSON。
"""
    try:
        res = llm_completion(model=model, prompt=prompt)
        data = extract_json(res) or {}
        acc = float(data.get("accuracy", 0))
        passed_raw = str(data.get("passed", "")).lower()
        passed = passed_raw == "yes" or acc >= threshold
        return {
            "accuracy": round(acc, 4),
            "passed": passed,
            "contains_target": str(data.get("contains_target", "")).lower() == "yes",
            "body_text_pollution": str(data.get("body_text_pollution", "")).lower() == "yes",
            "thinking": data.get("thinking") or "",
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "passed": False,
            "contains_target": False,
            "body_text_pollution": True,
            "thinking": f"verify_error:{e}",
        }


def _resolve_crop_rect(item: dict, page, blocks: list[dict], model=None, kind: str = "figure") -> tuple[pymupdf.Rect | None, dict]:
    """
    Combine LLM full-region bbox + heuristic expansion; prefer union for safety.
    """
    token_num = _extract_numeric_id(item.get("id", ""))
    meta = {"fallback_used": True, "fallback_method": "llm_full_region_locator"}

    if kind == "figure" and token_num and not _page_has_kind_caption(page, token_num, "figure"):
        return None, {
            **meta,
            "fallback_status": "failed",
            "fallback_message": f"page_has_no_figure_caption_for_图{token_num} (block or page text)",
        }

    llm_result = _llm_locate_full_region(item, page, blocks, model=model, kind=kind)
    heuristic_rect = _heuristic_full_region(page, blocks, token_num, kind) if token_num else None
    llm_rect = llm_result.get("rect")

    parts = []
    if llm_rect:
        parts.append(llm_rect)
    if heuristic_rect:
        parts.append(heuristic_rect)

    # Also union image blocks / detected tables on page for same figure/table
    if kind == "figure":
        cap_blocks = _find_caption_blocks(blocks, token_num, kind)
        if cap_blocks:
            cap_y = cap_blocks[0]["bbox"][3]
            for r in _all_image_rects(page):
                if r.y1 <= cap_y + 30:
                    parts.append(r)
    else:
        for t in _all_tables(page):
            parts.append(pymupdf.Rect(t.bbox))

    final = _union_rects(parts) if parts else None
    if final:
        final = _clamp_rect(final, page.rect, pad=8)
        meta.update({
            "fallback_status": "success",
            "fallback_message": (
                f"llm_reason={llm_result.get('reason','')}; "
                f"heuristic={'yes' if heuristic_rect else 'no'}; "
                f"confidence={llm_result.get('confidence','')}"
            ),
            "position_source": "llm_full_region",
            "crop_bbox": [round(final.x0, 1), round(final.y0, 1), round(final.x1, 1), round(final.y1, 1)],
        })
        return final, meta

    meta.update({
        "fallback_status": "failed",
        "fallback_message": llm_result.get("reason") or "could_not_resolve_region",
    })
    return None, meta


def _write_table_rows(rows: list[list], csv_path: str, md_path: str) -> bool:
    if not rows or len(rows) < 1:
        return False
    header = [str(c or "").strip() for c in rows[0]]
    if not any(header):
        return False
    body = rows[1:] if len(rows) > 1 else []

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[str(c or "") for c in r] for r in rows])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join("---" for _ in header) + " |\n")
        for r in body:
            row = [str(c or "") for c in r]
            row = row + [""] * max(0, len(header) - len(row))
            f.write("| " + " | ".join(row[: len(header)]) + " |\n")
    return True


def _llm_extract_table_from_text(
    item: dict,
    crop_text: str,
    csv_path: str,
    md_path: str,
    model=None,
) -> bool:
    """Structured table via LLM when PyMuPDF find_tables() fails (common in medical PDFs)."""
    text = (crop_text or "").strip()
    if len(text) < 15:
        return False

    id_text = str(item.get("id") or "")
    title = str(item.get("title") or "")
    desc = str(item.get("description") or "")

    prompt = f"""你是医学文献表格结构化专家。根据从 PDF 裁切区域提取的文本，还原为**完整表格**（含表题可忽略，重点是表头+全部数据行）。

目标表格：
- id: {id_text}
- title: {title}
- description: {desc}

裁切区域文本：
---
{text[:8000]}
---

请输出 JSON（不要其他文字）：
{{
  "headers": ["列1", "列2", ...],
  "rows": [
    ["单元格", "单元格", ...],
    ...
  ],
  "confidence": "high|medium|low"
}}

要求：
1. headers 为表头列名；rows 为每一行数据（列数与 headers 一致）
2. 保留原文中的分级符号（如 Ⅰ类、Ⅱa类）、合并语义可用空字符串占位
3. 若文本中明显是表格内容，必须尽量填满所有行，不要只输出占位符
4. 若完全无法识别为表格，返回 {{"headers": [], "rows": [], "confidence": "low"}}
"""
    try:
        res = llm_completion(model=model, prompt=prompt)
        data = extract_json(res) or {}
        headers = data.get("headers") or []
        rows_data = data.get("rows") or []
        if not headers or not rows_data:
            return False
        if str(data.get("confidence", "")).lower() == "low":
            return False
        table_rows = [[str(h) for h in headers]]
        ncol = len(headers)
        for r in rows_data:
            if not isinstance(r, list):
                continue
            row = [str(c or "").strip() for c in r]
            row = row + [""] * max(0, ncol - len(row))
            if any(cell for cell in row):
                table_rows.append(row[:ncol])
        if len(table_rows) < 2:
            return False
        return _write_table_rows(table_rows, csv_path, md_path)
    except Exception:
        return False


def _export_table_structured(
    item: dict,
    page,
    final_rect: pymupdf.Rect,
    csv_path: str,
    md_path: str,
    model=None,
) -> tuple[bool, str]:
    """PyMuPDF table finder first, then LLM on cropped text."""
    for t in _all_tables(page):
        t_rect = pymupdf.Rect(t.bbox)
        if t_rect.intersects(final_rect):
            try:
                rows = t.extract() or []
            except Exception:
                rows = []
            if rows and _write_table_rows(rows, csv_path, md_path):
                return True, "pymupdf_find_tables"

    crop_text = (page.get_textbox(final_rect) or "").strip()
    if _llm_extract_table_from_text(item, crop_text, csv_path, md_path, model=model):
        return True, "llm_text_structured"

    return False, "structured_export_failed"


def _table_to_files(table_obj, csv_path: str, md_path: str) -> bool:
    rows = []
    try:
        rows = table_obj.extract() or []
    except Exception:
        rows = []
    if not rows or (len(rows) == 1 and not any(str(c).strip() for c in rows[0])):
        return False
    return _write_table_rows(rows, csv_path, md_path)


def _export_item_crop(
    item: dict,
    doc,
    page_no: int,
    asset_subdir: str,
    prefix: str,
    idx: int,
    project_root: str,
    model=None,
    kind: str = "figure",
) -> None:
    page = doc[page_no - 1]
    blocks = _page_blocks(page)
    coarse_rect, fb_meta = _resolve_crop_rect(item, page, blocks, model=model, kind=kind)

    item.update(fb_meta)
    item["page"] = page_no
    item["location"] = f"第{page_no}页"

    if coarse_rect is None:
        item["asset_status"] = "failed"
        item["asset_message"] = "region_locate_failed"
        item["asset_accuracy"] = 0.0
        item["asset_accuracy_passed"] = False
        return

    item["crop_bbox_coarse"] = [
        round(coarse_rect.x0, 1), round(coarse_rect.y0, 1),
        round(coarse_rect.x1, 1), round(coarse_rect.y1, 1),
    ]

    # Pass 2: LLM trims body text from coarse region; figures also merge geometry anchor
    refine = _llm_refine_boundaries(item, page, blocks, coarse_rect, model=model, kind=kind)
    refined_rect = refine.get("rect")
    if kind == "figure":
        token_num = _extract_numeric_id(item.get("id", ""))
        anchor_rect = (
            _figure_content_anchor(page, blocks, token_num, coarse_rect) if token_num else None
        )
        final_rect = _finalize_figure_rect(coarse_rect, refined_rect, anchor_rect, page.rect)
        item["crop_bbox_anchor"] = (
            [
                round(anchor_rect.x0, 1), round(anchor_rect.y0, 1),
                round(anchor_rect.x1, 1), round(anchor_rect.y1, 1),
            ]
            if anchor_rect
            else None
        )
    else:
        final_rect = refined_rect or coarse_rect

    item["crop_bbox_refined"] = [
        round((refined_rect or coarse_rect).x0, 1), round((refined_rect or coarse_rect).y0, 1),
        round((refined_rect or coarse_rect).x1, 1), round((refined_rect or coarse_rect).y1, 1),
    ]
    item["crop_bbox_final"] = [
        round(final_rect.x0, 1), round(final_rect.y0, 1),
        round(final_rect.x1, 1), round(final_rect.y1, 1),
    ]
    item["boundary_refine_reason"] = refine.get("reason") or ""

    fid = _safe_id(prefix, item.get("id", ""), idx)
    _ensure_dir(asset_subdir)
    img_path = os.path.join(asset_subdir, f"{fid}.png")

    # Pass 3: accuracy gate (before persisting figure assets)
    verify = _llm_verify_crop_quality(item, page, final_rect, model=model, kind=kind)
    item["asset_accuracy"] = verify.get("accuracy", 0.0)
    item["asset_accuracy_passed"] = bool(verify.get("passed"))
    item["asset_accuracy_detail"] = {
        "contains_target": verify.get("contains_target"),
        "body_text_pollution": verify.get("body_text_pollution"),
        "thinking": verify.get("thinking"),
        "threshold": (CROP_ACCURACY_THRESHOLD if kind == "figure" else TABLE_CROP_ACCURACY_THRESHOLD),
    }

    # For tables, keep screenshot even if quality fails (avoid missing table image);
    # for figures, keep strict gate and drop failed crops.
    should_write_image = bool(verify.get("passed")) or kind == "table"
    if should_write_image:
        _export_clip(page, final_rect, img_path, dpi=REFINE_CROP_DPI)
        item["image_path"] = _rel(img_path, project_root)
    else:
        if os.path.isfile(img_path):
            os.remove(img_path)
        item["image_path"] = ""

    if verify.get("passed"):
        item["asset_status"] = "success"
        item["asset_message"] = "cropped_refined_and_verified"
    else:
        item["asset_status"] = "failed"
        item["asset_message"] = "crop_quality_check_failed"

    if kind == "table":
        csv_path = os.path.join(asset_subdir, f"{fid}.csv")
        md_path = os.path.join(asset_subdir, f"{fid}.md")
        structured_ok = False
        export_method = ""
        if verify.get("passed"):
            structured_ok, export_method = _export_table_structured(
                item, page, final_rect, csv_path, md_path, model=model
            )
        if structured_ok:
            item["table_export_method"] = export_method
            if item["asset_status"] == "success":
                item["asset_message"] = f"cropped_refined_verified_with_{export_method}"
        elif verify.get("passed"):
            item["table_export_method"] = "failed"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["note"])
                writer.writerow(["structured table extraction failed; see image_path crop"])
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("| note |\n| --- |\n| structured table extraction failed; see image_path crop |\n")
            if item["asset_status"] == "success":
                item["asset_message"] = "cropped_refined_verified_table_image_only"
        item["csv_path"] = _rel(csv_path, project_root)
        item["md_path"] = _rel(md_path, project_root)


def _discover_figures_from_doc(doc) -> list[dict]:
    """Scan PDF for all 图 N mentions on each page (whitespace-normalized; catches split 图/number)."""
    found: dict[str, dict] = {}
    for pno, page in enumerate(doc, start=1):
        text = _normalize_ws(page.get_text() or "")
        for m in re.finditer(r"(?:图|Figure|Fig\.?)\s*(\d+)\b", text, re.IGNORECASE):
            key = f"图{m.group(1)}"
            if key not in found:
                found[key] = {
                    "id": key,
                    "page": pno,
                    "location": f"第{pno}页",
                    "ref_type": "pdf_caption_scan",
                }
    return list(found.values())


def _merge_figures_info(figures_info: list[dict], discovered: list[dict]) -> list[dict]:
    by_id = {str(f.get("id")): dict(f) for f in (figures_info or []) if f.get("id")}
    for d in discovered:
        key = str(d.get("id"))
        if key not in by_id:
            by_id[key] = d
        else:
            if d.get("ref_type") == "pdf_caption_scan" and d.get("page"):
                by_id[key].setdefault("page", d["page"])
                by_id[key].setdefault("location", d.get("location"))
    return list(by_id.values())


def export_pdf_assets(
    pdf_path: str,
    figures_info: list[dict],
    tables_info: list[dict],
    result_folder: str,
    project_root: str,
    model: str | None = None,
) -> tuple[list[dict], list[dict]]:
    _ensure_dir(result_folder)
    assets_dir = os.path.join(result_folder, "assets")
    figures_dir = os.path.join(assets_dir, "figures")
    tables_dir = os.path.join(assets_dir, "tables")
    _ensure_dir(figures_dir)
    _ensure_dir(tables_dir)

    doc = pymupdf.open(pdf_path)
    try:
        page_texts = _build_page_text_index(doc)
        figures_info = _merge_figures_info(
            figures_info, _discover_figures_from_doc(doc)
        )

        for idx, fig in enumerate(figures_info or []):
            page_no, page_meta = _resolve_figure_page(fig, doc, page_texts, model=model)
            fig.update(page_meta)
            if page_no:
                fig["page"] = page_no
                fig["location"] = f"第{page_no}页"
            page_no = int(fig.get("page") or 0)
            if page_no < 1 or page_no > len(doc):
                fig["asset_status"] = "failed"
                fig["asset_message"] = "figure_page_not_found"
                fig["fallback_used"] = False
                fig["image_path"] = ""
                continue
            _export_item_crop(
                fig, doc, page_no, figures_dir, "figure", idx, project_root, model=model, kind="figure"
            )

        for idx, tab in enumerate(tables_info or []):
            refined = _find_best_page_for_item(tab, page_texts, kind="table")
            if refined:
                tab["page"] = refined
            page_no = int(tab.get("page") or 0)
            if page_no < 1 or page_no > len(doc):
                tab["asset_status"] = "failed"
                tab["asset_message"] = "invalid_page"
                tab["fallback_used"] = False
                continue
            _export_item_crop(
                tab, doc, page_no, tables_dir, "table", idx, project_root, model=model, kind="table"
            )
    finally:
        doc.close()

    return figures_info, tables_info
