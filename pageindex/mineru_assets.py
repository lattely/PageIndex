import csv
import os
import re
import shutil
import tempfile
import time
import zipfile
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests
try:
    from .utils import llm_completion, extract_json
    from .llm_figure_validator import resolve_figure_assignment, is_llm_figure_validator_enabled
except ImportError:
    from utils import llm_completion, extract_json
    from llm_figure_validator import resolve_figure_assignment, is_llm_figure_validator_enabled


MINERU_API_BASE = os.getenv("MINERU_API_BASE", "https://mineru.net/api/v4")
MINERU_MODEL_VERSION = os.getenv("MINERU_MODEL_VERSION", "vlm")
MINERU_POLL_INTERVAL_SECONDS = 3
MINERU_TIMEOUT_SECONDS = 600


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rel(path: str, project_root: str) -> str:
    return os.path.relpath(path, project_root).replace("\\", "/")


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", name).strip("_")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _cleanup_unreferenced_files(base_dir: str, keep_abs_paths: set[str], suffixes: tuple[str, ...]) -> None:
    if not os.path.isdir(base_dir):
        return
    normalized_keep = {os.path.abspath(p) for p in keep_abs_paths}
    for name in os.listdir(base_dir):
        p = os.path.abspath(os.path.join(base_dir, name))
        if not os.path.isfile(p):
            continue
        if not name.lower().endswith(suffixes):
            continue
        if p not in normalized_keep:
            os.remove(p)


def _download_binary(url: str, out_path: str, headers: dict | None = None, timeout: int = 120) -> None:
    r = requests.get(url, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)


def _mineru_headers() -> dict:
    token = os.getenv("MINERU_API_KEY")
    if not token:
        raise ValueError("MINERU_API_KEY is not set in environment.")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _upload_local_pdf(pdf_path: str) -> str:
    """Apply upload URLs, PUT local file, return batch_id for polling."""
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_name = os.path.basename(pdf_path)
    endpoint = f"{MINERU_API_BASE.rstrip('/')}/file-urls/batch"
    payload = {
        "files": [
            {
                "name": file_name,
                "data_id": _safe_name(os.path.splitext(file_name)[0])[:128] or "pdf",
            }
        ],
        "model_version": MINERU_MODEL_VERSION,
    }
    r = requests.post(endpoint, headers=_mineru_headers(), json=payload, timeout=60)
    r.raise_for_status()
    result = r.json() or {}
    if int(result.get("code", 0)) != 0:
        raise RuntimeError(f"MinerU upload URL request failed: {result}")

    data = result.get("data") or {}
    batch_id = (data.get("batch_id") or "").strip()
    urls = data.get("file_urls") or []
    if not batch_id or not urls:
        raise RuntimeError(f"MinerU upload response missing batch_id/file_urls: {result}")

    with open(pdf_path, "rb") as f:
        upload_res = requests.put(urls[0], data=f, timeout=300)
    if upload_res.status_code != 200:
        raise RuntimeError(
            f"MinerU file upload failed: status={upload_res.status_code}, body={upload_res.text[:500]}"
        )
    return batch_id


def _poll_batch_result(batch_id: str, file_name: str) -> dict:
    endpoint = f"{MINERU_API_BASE.rstrip('/')}/extract-results/batch/{batch_id}"
    deadline = time.time() + MINERU_TIMEOUT_SECONDS
    last_item = {}
    while time.time() < deadline:
        r = requests.get(endpoint, headers=_mineru_headers(), timeout=60)
        r.raise_for_status()
        payload = r.json() or {}
        if int(payload.get("code", 0)) != 0:
            raise RuntimeError(f"MinerU batch poll failed: {payload}")

        data = payload.get("data") or {}
        results = data.get("extract_result") or []
        if isinstance(results, dict):
            results = [results]

        item = None
        for row in results:
            if str(row.get("file_name") or "") == file_name:
                item = row
                break
        if item is None and len(results) == 1:
            item = results[0]
        if item is None:
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

        last_item = item
        state = str(item.get("state") or "").lower()
        if state == "done":
            return item
        if state == "failed":
            raise RuntimeError(f"MinerU task failed: {item.get('err_msg') or payload}")
        time.sleep(MINERU_POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"MinerU batch timeout, last state: {last_item}")


def _extract_zip_from_url(full_zip_url: str, target_dir: str) -> str:
    _ensure_dir(target_dir)
    zip_path = os.path.join(target_dir, "mineru_result.zip")
    _download_binary(full_zip_url, zip_path, timeout=300)
    out_dir = os.path.join(target_dir, "mineru_result")
    _ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def _fetch_mineru_zip_for_pdf(pdf_path: str, file_name: str, tmpdir: str) -> tuple[str, str]:
    batch_id = _upload_local_pdf(pdf_path)
    task_data = _poll_batch_result(batch_id, file_name)
    full_zip_url = task_data.get("full_zip_url")
    if not full_zip_url:
        raise RuntimeError(f"MinerU result missing full_zip_url: {task_data}")
    extracted_root = _extract_zip_from_url(full_zip_url, tmpdir)
    zip_md_path = _find_first_file(extracted_root, ("full.md",))
    if not zip_md_path:
        raise RuntimeError("MinerU zip does not include full.md")
    return extracted_root, zip_md_path


def _find_first_file(root_dir: str, names: tuple[str, ...]) -> str | None:
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f in names:
                return os.path.join(root, f)
    return None


def _collect_image_refs(md_text: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for m in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", md_text):
        refs.append((m.group(1).strip(), m.group(2).strip()))
    for m in re.finditer(r'<img[^>]*src=["\']([^"\']+)["\'][^>]*>', md_text, re.IGNORECASE):
        refs.append(("", m.group(1).strip()))
    seen = set()
    uniq = []
    for alt, src in refs:
        key = (alt, src)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((alt, src))
    return uniq


def _is_table_like(alt: str, src: str) -> bool:
    low = f"{alt} {src}".lower()
    return ("table" in low) or ("tab" in low) or ("表" in low)


def _resolve_source_file(src: str, md_path: str, extracted_root: str) -> str | None:
    src = (src or "").strip()
    if not src:
        return None
    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        return src
    if os.path.isabs(src) and os.path.exists(src):
        return src
    md_dir = os.path.dirname(md_path)
    local_md_ref = os.path.abspath(os.path.join(md_dir, src))
    if os.path.exists(local_md_ref):
        return local_md_ref
    local_zip_ref = os.path.abspath(os.path.join(extracted_root, src))
    if os.path.exists(local_zip_ref):
        return local_zip_ref
    return None


def _copy_or_download(src: str, dest_path: str, md_path: str, extracted_root: str) -> bool:
    resolved = _resolve_source_file(src, md_path, extracted_root)
    if not resolved:
        return False
    _ensure_dir(os.path.dirname(dest_path))
    parsed = urlparse(resolved)
    if parsed.scheme in ("http", "https"):
        _download_binary(resolved, dest_path, timeout=180)
    else:
        shutil.copyfile(resolved, dest_path)
    return True


def _normalize_md_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def _is_table_separator_row(row: str) -> bool:
    cells = _normalize_md_row(row)
    if not cells:
        return False
    for c in cells:
        if not re.fullmatch(r":?-{3,}:?", c.replace(" ", "")):
            return False
    return True


def _extract_md_table_blocks(md_text: str) -> list[list[str]]:
    lines = md_text.splitlines()
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if "|" in line and "|" in nxt and _is_table_separator_row(nxt):
            block = [line, nxt]
            j = i + 2
            while j < len(lines):
                cur = lines[j]
                if "|" not in cur or not cur.strip():
                    break
                block.append(cur)
                j += 1
            if len(block) >= 3:
                blocks.append(block)
            i = j
            continue
        i += 1
    return blocks


def _write_table_block(table_lines: list[str], csv_path: str, md_path: str) -> None:
    _write_text(md_path, "\n".join(table_lines).strip() + "\n")
    rows = [_normalize_md_row(x) for x in table_lines if "|" in x]
    if len(rows) < 2:
        return
    header = rows[0]
    data_rows = rows[2:]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in data_rows:
            row = row + [""] * max(0, len(header) - len(row))
            w.writerow(row[: len(header)])


class _SimpleHTMLTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._in_td = False
        self._cell_text_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        t = (tag or "").lower()
        if t == "tr":
            self._current_row = []
        elif t in ("td", "th"):
            self._in_td = True
            self._cell_text_parts = []

    def handle_data(self, data):
        if self._in_td:
            self._cell_text_parts.append(data or "")

    def handle_endtag(self, tag):
        t = (tag or "").lower()
        if t in ("td", "th"):
            text = re.sub(r"\s+", " ", "".join(self._cell_text_parts)).strip()
            self._current_row.append(text)
            self._in_td = False
            self._cell_text_parts = []
        elif t == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = []


def _extract_html_table_blocks(md_text: str) -> list[str]:
    return re.findall(r"<table[^>]*>.*?</table>", md_text, flags=re.IGNORECASE | re.DOTALL)


def _html_table_to_rows(table_html: str) -> list[list[str]]:
    parser = _SimpleHTMLTableParser()
    parser.feed(table_html)
    rows = [r for r in parser.rows if any(c.strip() for c in r)]
    return rows


def _write_rows_to_files(rows: list[list[str]], csv_path: str, md_path: str) -> None:
    if not rows:
        return
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in body:
            rr = r + [""] * max(0, len(header) - len(r))
            w.writerow(rr[: len(header)])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join("---" for _ in header) + " |\n")
        for r in body:
            rr = r + [""] * max(0, len(header) - len(r))
            f.write("| " + " | ".join(rr[: len(header)]) + " |\n")


def _extract_titles_by_line(md_lines: list[str], kind: str) -> dict[int, str]:
    if kind == "figure":
        pat = re.compile(r"^\s*(图\s*\d+|Figure\s*\d+)\s*(.*)$", re.IGNORECASE)
    else:
        pat = re.compile(r"^\s*(表\s*\d+|Table\s*\d+)\s*(.*)$", re.IGNORECASE)
    out: dict[int, str] = {}
    for i, line in enumerate(md_lines, start=1):
        m = pat.match((line or "").strip())
        if m:
            out[i] = (m.group(0) or "").strip()
    return out


def _extract_cn_titles_by_line(md_lines: list[str], kind: str) -> dict[int, str]:
    if kind == "figure":
        pat = re.compile(r"^\s*图\s*([0-9]+)\s*(.*)$")
    else:
        pat = re.compile(r"^\s*表\s*([0-9]+)\s*(.*)$")
    out: dict[int, str] = {}
    for i, line in enumerate(md_lines, start=1):
        m = pat.match((line or "").strip())
        if not m:
            continue
        num = m.group(1).strip()
        title = (m.group(2) or "").strip(" ：:.-")
        out[i] = f"{num}|{title}"
    return out


def _strip_table_prefix(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"^\s*(Table|Tab\.?)\s*\d+\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*表\s*\d+\s*", "", t)
    return t.strip(" ：:.-")


def _strip_figure_prefix(title: str) -> str:
    t = (title or "").strip()
    t = re.sub(r"^\s*(Figure|Fig\.?)\s*\d+\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*图\s*\d+\s*", "", t)
    return t.strip(" ：:.-")


def _parse_main_figure_titles(md_lines: list[str]) -> list[dict]:
    out: list[dict] = []
    # Prefer Chinese captions to avoid long wrapped English lines polluting titles.
    pat = re.compile(r"^\s*图\s*([0-9]+)\s*(.*)$")
    for i, line in enumerate(md_lines, start=1):
        m = pat.match((line or "").strip())
        if not m:
            continue
        num = int(m.group(1))
        raw = (m.group(2) or "").strip(" ：:.-")
        title_tail = _strip_figure_prefix(raw)
        if len(title_tail) > 120:
            title_tail = title_tail[:120]
        out.append({"line": i, "num": num, "title": title_tail})
    return out


def _match_main_figure_for_image(
    line_no: int,
    figure_titles: list[dict],
    max_forward: int = 90,
    max_backward: int = 12,
) -> dict | None:
    # Prefer nearest caption line after the image in markdown flow.
    after = [t for t in figure_titles if 0 <= int(t["line"]) - line_no <= max_forward]
    if after:
        after.sort(key=lambda x: int(x["line"]))
        return after[0]
    before = [t for t in figure_titles if 0 <= line_no - int(t["line"]) <= max_backward]
    if before:
        before.sort(key=lambda x: line_no - int(x["line"]))
        return before[0]
    return None


def _extract_subfigure_hint(md_lines: list[str], line_no: int) -> tuple[int | None, str]:
    """
    Parse nearby lines like '(1) 肾动脉上' / '1) xxx' / '（2）xxx'.
    Returns (sub_index, sub_title).
    """
    window_lines = []
    for i in range(line_no, min(len(md_lines), line_no + 4) + 1):
        window_lines.append(md_lines[i - 1].strip())
    pat = re.compile(r"^[（(]?\s*([0-9]{1,2})\s*[）)]\s*(.+)$")
    for txt in window_lines:
        m = pat.match(txt)
        if m:
            return int(m.group(1)), m.group(2).strip(" ：:.-")
    return None, ""


def _build_figure_filename(main_num: int, main_title: str, sub_idx: int | None, sub_title: str) -> str:
    if sub_idx:
        name = f"figure_{main_num}_{main_title}_{sub_idx}_{sub_title}"
    else:
        name = f"figure_{main_num}_{main_title}"
    return _safe_name(name) + ".png"


def _extract_caption_sub_labels(main_title: str) -> list[str]:
    text = (main_title or "").strip()
    labels = re.findall(r"\b([A-Z])\s*[\.．、:：)]", text)
    if not labels:
        labels = re.findall(r"\b([A-Z])\b", text)
    uniq: list[str] = []
    for x in labels:
        if x not in uniq:
            uniq.append(x)
    return uniq


def _build_table_filename(table_num: int, table_title: str) -> str:
    name = f"table_{table_num}_{table_title}"
    return _safe_name(name)


def _nearest_title(line_no: int, title_lines: dict[int, str], max_dist: int = 12) -> str:
    best = ""
    best_dist = 10**9
    for ln, title in title_lines.items():
        d = abs(ln - line_no)
        if d < best_dist and d <= max_dist:
            best_dist = d
            best = title
    return best


def _nearby_context(md_lines: list[str], line_no: int, window: int = 6) -> str:
    start = max(1, line_no - window)
    end = min(len(md_lines), line_no + window)
    chunk = [f"{i}:{md_lines[i-1]}" for i in range(start, end + 1)]
    return "\n".join(chunk)


def _find_line_no_by_substring(md_lines: list[str], marker: str) -> int | None:
    if not marker:
        return None
    for i, line in enumerate(md_lines, start=1):
        if marker in line:
            return i
    return None


def _llm_enrich_asset(item: dict, context: str, kind: str, model: str | None = None) -> dict:
    kind_zh = "图片" if kind == "figure" else "表格"
    prompt = f"""你是医学文献结构化专家。请根据{kind_zh}的标题、上下文和位置信息，输出该{kind_zh}代表的医学含义与关键词。

输入:
{item}

上下文:
{context[:4000]}

返回 JSON:
{{
  "medical_meaning": "1-4句，说明该{kind_zh}在文中的医学意义与临床用途",
  "keywords": ["关键词1", "关键词2", "关键词3", "..."],
  "possible_entities": ["疾病/指标/治疗/分型等实体词，可为空列表"]
}}
仅返回 JSON。
"""
    try:
        res = llm_completion(model=model, prompt=prompt)
        data = extract_json(res) or {}
        return {
            "medical_meaning": data.get("medical_meaning") or "",
            "keywords": data.get("keywords") if isinstance(data.get("keywords"), list) else [],
            "possible_entities": data.get("possible_entities") if isinstance(data.get("possible_entities"), list) else [],
        }
    except Exception:
        return {"medical_meaning": "", "keywords": [], "possible_entities": []}


def export_assets_from_mineru(
    pdf_path: str,
    result_folder: str,
    project_root: str,
    markdown_output_dir: str,
    model: str | None = None,
) -> tuple[list[dict], list[dict], str]:
    pdf_path = os.path.abspath(pdf_path)
    file_name = os.path.basename(pdf_path)

    assets_dir = os.path.join(result_folder, "assets")
    figures_dir = os.path.join(assets_dir, "figures")
    tables_dir = os.path.join(assets_dir, "tables")
    _ensure_dir(figures_dir)
    _ensure_dir(tables_dir)
    _ensure_dir(markdown_output_dir)

    pdf_name = os.path.splitext(file_name)[0]
    md_out_path = os.path.join(markdown_output_dir, f"{pdf_name}.md")

    extracted_root = ""
    full_md_path = md_out_path
    temp_dir_obj = None
    if os.path.isfile(md_out_path):
        # Reuse existing markdown to avoid repeated MinerU API calls.
        md_text = _read_text(md_out_path)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="mineru_")
        extracted_root, zip_md_path = _fetch_mineru_zip_for_pdf(pdf_path, file_name, temp_dir_obj.name)
        md_text = _read_text(zip_md_path)
        _write_text(md_out_path, md_text)
        full_md_path = zip_md_path

    md_lines = md_text.splitlines()
    figure_titles = _extract_titles_by_line(md_lines, "figure")
    table_titles = _extract_titles_by_line(md_lines, "table")
    figure_titles_cn = _extract_cn_titles_by_line(md_lines, "figure")
    table_titles_cn = _extract_cn_titles_by_line(md_lines, "table")
    parsed_figure_titles = _parse_main_figure_titles(md_lines)

    figures_info: list[dict] = []
    tables_info: list[dict] = []

    image_refs = _collect_image_refs(md_text)
    # If markdown exists but local image refs are missing, pull MinerU zip once as asset source.
    if image_refs and not extracted_root:
        need_assets = False
        for _, src in image_refs[:8]:
            resolved = _resolve_source_file(src, full_md_path, extracted_root)
            if not resolved:
                need_assets = True
                break
        if need_assets:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="mineru_")
            extracted_root, zip_md_path = _fetch_mineru_zip_for_pdf(pdf_path, file_name, temp_dir_obj.name)
            full_md_path = zip_md_path
    fig_idx = 0
    tab_img_idx = 0
    pending_fig_records: list[dict] = []
    pending_figure_candidates: list[dict] = []
    for alt, src in image_refs:
        marker = f"]({src})" if src else ""
        line_no = _find_line_no_by_substring(md_lines, marker) or 0
        if _is_table_like(alt, src):
            tab_img_idx += 1
            nearest_cn = _nearest_title(line_no, table_titles_cn)
            if nearest_cn and "|" in nearest_cn:
                _, cn_title = nearest_cn.split("|", 1)
                table_title = cn_title.strip() or f"表{tab_img_idx}"
            else:
                table_title = _strip_table_prefix(alt or _nearest_title(line_no, table_titles) or f"表{tab_img_idx}")
                table_title = table_title or f"表{tab_img_idx}"
            table_file_base = _build_table_filename(tab_img_idx, table_title)
            img_name = f"{table_file_base}.png"
            out_path = os.path.join(tables_dir, img_name)
            ok = _copy_or_download(src, out_path, full_md_path, extracted_root)
            if not ok:
                legacy_path = os.path.join(tables_dir, f"table_{tab_img_idx}.png")
                if os.path.isfile(legacy_path) and legacy_path != out_path:
                    shutil.copyfile(legacy_path, out_path)
                    ok = True
            tables_info.append(
                {
                    "id": f"表{tab_img_idx}",
                    "title": table_title,
                    "location": f"第{line_no}行" if line_no else "markdown_image",
                    "line_no": line_no if line_no else None,
                    "asset_status": "success" if ok else "failed",
                    "asset_message": "downloaded_from_markdown" if ok else "image_download_failed",
                    "image_path": _rel(out_path, project_root) if ok else "",
                }
            )
        else:
            fig_idx += 1
            main_match = _match_main_figure_for_image(line_no, parsed_figure_titles)
            if main_match:
                main_num = int(main_match["num"])
                main_title = (main_match["title"] or f"图{main_num}").strip()
            else:
                main_num = fig_idx
                main_title = (_nearest_title(line_no, figure_titles) or f"图{main_num}").strip()
            sub_idx_hint, sub_title_hint = _extract_subfigure_hint(md_lines, line_no)
            pending_figure_candidates.append(
                {
                    "src": src,
                    "line_no": line_no,
                    "main_num": main_num,
                    "main_title": main_title,
                    "sub_idx_hint": sub_idx_hint,
                    "sub_title_hint": (sub_title_hint or (alt or "")).strip(),
                    "fig_idx": fig_idx,
                }
            )

    by_main_candidates: dict[int, list[dict]] = {}
    for c in pending_figure_candidates:
        by_main_candidates.setdefault(int(c["main_num"]), []).append(c)

    for main_num, group in by_main_candidates.items():
        group_sorted = sorted(group, key=lambda x: int(x.get("line_no") or 10**9))
        multiple = len(group_sorted) > 1
        explicit_exists = any(g.get("sub_idx_hint") is not None for g in group_sorted)
        caption_labels = _extract_caption_sub_labels(str(group_sorted[0].get("main_title") or ""))

        for idx_in_group, g in enumerate(group_sorted, start=1):
            sub_idx = g.get("sub_idx_hint")
            if sub_idx is None and multiple:
                sub_idx = idx_in_group
            if explicit_exists and sub_idx is None:
                sub_idx = idx_in_group

            sub_title = str(g.get("sub_title_hint") or "").strip()
            assign_reason = ""
            needs_llm = False
            if not sub_title and multiple and idx_in_group <= len(caption_labels):
                sub_title = caption_labels[idx_in_group - 1]
            if not sub_title and is_llm_figure_validator_enabled() and model and multiple:
                _, sub_title_llm, sub_idx_llm, needs_llm, assign_reason = resolve_figure_assignment(
                    int(g["line_no"]),
                    md_lines,
                    parsed_figure_titles,
                    [g],
                    model,
                )
                if sub_title_llm:
                    sub_title = sub_title_llm
                if sub_idx_llm:
                    sub_idx = sub_idx_llm

            img_name = _build_figure_filename(main_num, str(g["main_title"]), sub_idx, sub_title)
            out_path = os.path.join(figures_dir, img_name)
            ok = _copy_or_download(str(g["src"]), out_path, full_md_path, extracted_root)
            if not ok:
                legacy_path = os.path.join(figures_dir, f"figure_{g['fig_idx']}.png")
                if os.path.isfile(legacy_path) and legacy_path != out_path:
                    shutil.copyfile(legacy_path, out_path)
                    ok = True

            title = f"图 {main_num} {g['main_title']}".strip()
            if sub_title and sub_idx is not None:
                title = f"{title} ({sub_idx}) {sub_title}"
            figure_id = f"图 {main_num}-{sub_idx}" if sub_idx is not None else f"图 {main_num}"
            pending_fig_records.append(
                {
                    "id": figure_id,
                    "title": title,
                    "main_figure_no": main_num,
                    "sub_figure_no": sub_idx,
                    "sub_figure_title": sub_title,
                    "location": f"第{g['line_no']}行" if g["line_no"] else "markdown_image",
                    "line_no": g["line_no"] if g["line_no"] else None,
                    "asset_status": "success" if ok else "failed",
                    "asset_message": "downloaded_from_markdown" if ok else "image_download_failed",
                    "image_path": _rel(out_path, project_root) if ok else "",
                    "main_figure_detected_count": len(group_sorted),
                    "figure_total_detected": len(group_sorted),
                    "figure_success_downloaded": sum(1 for x in group_sorted if x.get("asset_status") == "success"),
                    "assign_reason": assign_reason,
                    "needs_llm": needs_llm,
                }
            )

    figures_info.extend(pending_fig_records)

    # Parse markdown pipe tables
    all_table_entries: list[tuple[int, str, list[list[str]] | None]] = []
    md_pipe_tables = _extract_md_table_blocks(md_text)
    for b in md_pipe_tables:
        first = b[0]
        line_no = _find_line_no_by_substring(md_lines, first) or 0
        rows = [_normalize_md_row(x) for x in b if "|" in x]
        if len(rows) >= 2:
            all_table_entries.append((line_no, "\n".join(b), rows))

    # Parse HTML tables (<table>...</table>) from MinerU markdown
    html_tables = _extract_html_table_blocks(md_text)
    for t in html_tables:
        line_no = _find_line_no_by_substring(md_lines, t) or 0
        rows = _html_table_to_rows(t)
        if rows:
            all_table_entries.append((line_no, t, rows))

    for idx, (line_no, raw_block, rows) in enumerate(all_table_entries, start=1):
        nearest_cn = _nearest_title(line_no, table_titles_cn)
        if nearest_cn and "|" in nearest_cn:
            _, cn_title = nearest_cn.split("|", 1)
            table_title = cn_title.strip() or f"表{idx}"
        else:
            table_title = _strip_table_prefix(_nearest_title(line_no, table_titles) or f"表{idx}")
            table_title = table_title or f"表{idx}"
        table_file_base = _build_table_filename(idx, table_title)
        csv_path = os.path.join(tables_dir, f"{table_file_base}.csv")
        table_md_path = os.path.join(tables_dir, f"{table_file_base}.md")
        if raw_block.lstrip().lower().startswith("<table"):
            _write_rows_to_files(rows, csv_path, table_md_path)
        else:
            _write_table_block(raw_block.splitlines(), csv_path, table_md_path)

        existing = next((t for t in tables_info if t.get("id") == f"表{idx}"), None)
        title = table_title
        if existing:
            existing["title"] = existing.get("title") or title
            existing["location"] = f"第{line_no}行" if line_no else existing.get("location", "markdown_table")
            existing["line_no"] = line_no if line_no else existing.get("line_no")
            existing["csv_path"] = _rel(csv_path, project_root)
            existing["md_path"] = _rel(table_md_path, project_root)
            existing["asset_message"] = "downloaded_from_markdown_and_table_parsed"
        else:
            tables_info.append(
                {
                    "id": f"表{idx}",
                    "title": title,
                    "location": f"第{line_no}行" if line_no else "markdown_table",
                    "line_no": line_no if line_no else None,
                    "asset_status": "success",
                    "asset_message": "table_parsed_from_markdown",
                    "image_path": "",
                    "csv_path": _rel(csv_path, project_root),
                    "md_path": _rel(table_md_path, project_root),
                }
            )

    # Enrich figure/table medical meaning and keywords with LLM + local context
    for item in figures_info:
        line_no = int(item.get("line_no") or 0)
        context = _nearby_context(md_lines, line_no or 1)
        enrich = _llm_enrich_asset(item, context, kind="figure", model=model)
        item.update(enrich)

    for item in tables_info:
        line_no = int(item.get("line_no") or 0)
        context = _nearby_context(md_lines, line_no or 1)
        enrich = _llm_enrich_asset(item, context, kind="table", model=model)
        item.update(enrich)

    # Keep explicit extraction statistics for integrity checks.
    figure_total = len(image_refs)
    figure_success = sum(1 for x in figures_info if x.get("asset_status") == "success")
    for item in figures_info:
        item["figure_total_detected"] = figure_total
        item["figure_success_downloaded"] = figure_success

    figure_keep: set[str] = set()
    for x in figures_info:
        p = x.get("image_path") or ""
        if p:
            figure_keep.add(os.path.join(project_root, p))
    table_keep: set[str] = set()
    for x in tables_info:
        for k in ("image_path", "csv_path", "md_path"):
            p = x.get(k) or ""
            if p:
                table_keep.add(os.path.join(project_root, p))
    _cleanup_unreferenced_files(figures_dir, figure_keep, (".png", ".jpg", ".jpeg", ".webp"))
    _cleanup_unreferenced_files(tables_dir, table_keep, (".png", ".csv", ".md"))

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return figures_info, tables_info, md_out_path
